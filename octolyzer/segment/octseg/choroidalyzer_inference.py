import os
import torch
import torch.nn as nn 
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms import v2 as T
import numpy as np
from PIL import Image
from tqdm.autonotebook import tqdm
from pathlib import PurePath
import sys

SCRIPT_PATH = os.path.realpath(os.path.dirname(__file__))
sys.path.append(SCRIPT_PATH)


class FixShape(T.Transform):
    def __init__(self):
        """Forces input to have dimensions divisble by 32"""
        super().__init__()

    def __call__(self, img):
        M, N = img.shape[-2:]
        pad_M = (128 - M%128) % 128
        pad_N = (128 - N%128) % 128
        return TF.pad(img, padding=(0, 0, pad_N, pad_M)), (M, N)

    def __repr__(self):
        return self.__class__.__name__


def _get_fov_filter(kernel_size=21):
    """Build 1x1 convolution filter for processing fovea prediction map"""
    assert kernel_size % 2 == 1
    fov_filter = nn.Conv1d(1, 1, kernel_size=kernel_size, bias=False, padding_mode='reflect', padding='same')
    fov_filter.requires_grad_(False)
    ascending_weights = torch.linspace(0.1, 1, kernel_size // 2)
    fov_filter_weights = torch.cat([ascending_weights, torch.tensor([1.]), ascending_weights.flip(0)])
    fov_filter_weights /= fov_filter_weights.sum()
    fov_filter.weight = torch.nn.Parameter(fov_filter_weights.view(1, 1, -1), requires_grad=False)
    return fov_filter

def _agg_fov_signal(tens, d=1):
    """Aggregate column probabilities but summing over axis d"""
    return tens.sum(dim=d)

def process_fovea_prediction(preds):    
    """Wrapper to build filter, aggregate fovea prediction map and extract 
    largest row and column"""
    fovea = []
    fov_score = []
    for d, k in zip([-2, -1], [21, 51]):
        fovea_signal_filter = _get_fov_filter(k).to(preds.device)
        fov_signal = _agg_fov_signal(preds, d)
        fov_signal = fovea_signal_filter(fov_signal).squeeze().cpu().numpy()
        fov_score.append(fov_signal.max(axis=-1))
        fovea.append(fov_signal.argmax(axis=-1))
    
    return np.asarray(fovea).T.reshape(-1,2), np.asarray(fov_score).T.reshape(-1,2)


def get_default_img_transforms():
    """Tensor, dimension and normalisation default augs"""
    return T.Compose([
        T.PILToTensor(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=(0.5,), std=(0.5,)),
        FixShape()
    ])
    

class ImgListDataset(Dataset):
    """Torch Dataset from img list"""
    def __init__(self, img_list):
        self.img_list = img_list
        if isinstance(img_list[0], (str, PurePath)):
            self.is_arr = False
        elif isinstance(img_list[0], np.ndarray):
            self.is_arr = True
        self.transform = get_default_img_transforms()

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if self.is_arr:
            img = Image.fromarray((255*self.img_list[idx]/self.img_list[idx].max()).astype(np.uint8))
        else:
            img = Image.open(self.img_list[idx])

        img, shape = self.transform(img)
        return {'img': img, "crop":shape} 


def get_img_list_dataloader(img_list, batch_size=16, num_workers=0, pin_memory=False):
    """Wrapper of Dataset into DataLoader"""
    dataset = ImgListDataset(img_list)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return loader


class Choroidalyzer:

    DEFAULT_MODEL_URL = 'https://github.com/jaburke166/OCTolyzer/releases/download/v1.0/choroidalyzer_weights.pth'
    DEFAULT_CHOR_THRESHOLD = 0.5
    DEFAULT_FOV_THRESHOLD = 0.1
    DEFAULT_MODEL_PATHS = None
    # DEFAULT_MODEL_PATHS = [os.path.join(SCRIPT_PATH, r"weights/choroidalyzer_weights.pth")]
    VERBOSITY = True
    BASELINE_MODEL = True
    
    def __init__(self, model_path=DEFAULT_MODEL_URL, threshold=DEFAULT_CHOR_THRESHOLD, 
                 fov_thresh=DEFAULT_FOV_THRESHOLD, local_model_path=DEFAULT_MODEL_PATHS, 
                 verbose=VERBOSITY, base=BASELINE_MODEL):
        """
        Core inference class for macular OCT B-scan choroid segmentation model
        """
        self.transform = get_default_img_transforms()
        self.threshold = threshold
        self.fov_thresh = fov_thresh
        self.verbose = ~verbose
        if torch.cude.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        if local_model_path is not None:
            if base:
                self.model = torch.load(local_model_path[0], map_location=self.device)
        else:
            self.model = torch.hub.load_state_dict_from_url(model_path, map_location=self.device)
        if self.device != "cpu":
            print("Macular choroid segmentation model has been loaded with GPU acceleration!")
        self.model.eval()

    @torch.inference_mode()
    def predict_img(self, img, soft_pred=False):
        """Inference on a single image"""
        if isinstance(img, (str, PurePath)):
            # assume it's a path to an image
            img = Image.open(img)
        elif isinstance(img, np.ndarray):
            # assume it's a numpy array
            # NOTE: we assume that the image has not been normalized yet
            img = Image.fromarray(img)

        with torch.no_grad():
            img, (M, N) = self.transform(img)
            img = img.unsqueeze(0).to(self.device)
            pred = self.model(img).squeeze(0).sigmoid()
            pred_map = pred[:2]
            fov_map = pred[-1].unsqueeze(0)
            fovea, fov_score = process_fovea_prediction(fov_map)
            if self.fov_thresh is not None:
                fovea = np.array([0,0]) if fov_map.max() < self.fov_thresh else fovea
            if not soft_pred:
                pred = (pred_map > self.threshold).int()
            return pred.cpu().numpy()[:, :M, :N],  fovea[0], fov_score[0]

    def predict_list(self, img_list, soft_pred=False):
        """Inference on a list of images without batching"""
        preds = []
        foveas = []
        fov_scores = []
        with torch.no_grad():
            for img in tqdm(img_list, desc='Predicting', leave=False, disable=self.verbose):
                pred, fovea, fov_score = self.predict_img(img, soft_pred=soft_pred)
                preds.append(pred)
                foveas.append(fovea)
                fov_scores.append(fov_score)
        return preds, foveas, np.concatenate(fov_scores)

    # TODO: Peripapillary scans will not work here
    def _predict_loader(self, loader, soft_pred=False):
        """Inference from a DataLoader"""
        preds = []
        foveas = []
        fov_scores = []
        with torch.no_grad():
            for batch in tqdm(loader, desc='Predicting', leave=False, disable=self.verbose):
                img = batch['img'].to(self.device)
                batch_M, batch_N = batch['crop']
                pred = self.model(img).sigmoid().squeeze()
                pred_map = pred.cpu().numpy()
                fov_map = pred[:,-1].unsqueeze(1)
                fovea, fov_score = process_fovea_prediction(fov_map)
                if self.fov_thresh is not None:
                    ppole = fov_map.squeeze().cpu().numpy().max(axis=-1).max(axis=-1) <= self.fov_thresh
                    fovea[ppole] = 0
                if not soft_pred:
                    pred_map = (pred_map > self.threshold).astype(np.int64)
                pred_map = [p[:, :M,:N] for (p, M, N) in zip(pred_map, batch_M, batch_N)]
                preds.append(pred_map)
                foveas.append(fovea)
                fov_scores.append(fov_score)

        # Collect outputs
        pred_output = np.concatenate(preds)
        fov_output = np.concatenate(foveas)
        fovscore_output = np.concatenate(fov_scores)
        
        return pred_output, fov_output, fovscore_output

    def predict_batch(self, img_list, soft_pred=False, batch_size=16, num_workers=0, pin_memory=False):
        """Wrapper for DataLoader inference"""
        loader = get_img_list_dataloader(img_list, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
        preds, foveas, fov_scores = self._predict_loader(loader, soft_pred=soft_pred)
        return preds, foveas, fov_scores

    def __call__(self, x):
        """Direct call for inference on single  image"""
        return self.predict_img(x)

    def __repr__(self):
        return f'{self.__class__.__name__}(threshold={self.threshold})'
