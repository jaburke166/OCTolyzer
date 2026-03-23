import torch
import os
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms import v2 as TT
from torchvision import transforms as T
import numpy as np
from PIL import Image
from tqdm.autonotebook import tqdm
from pathlib import PurePath
import sys

SCRIPT_PATH = os.path.realpath(os.path.dirname(__file__))
sys.path.append(SCRIPT_PATH)
    

class PadandFixImage(TT.Transform):
    """For peripapillary scans, we reflect-pad image to ensure whole peripapillary scan is
        segmented as DeepGPET can sometimes not segment the full width of the choroid."""
    
    def __init__(self, pad_x=256, factor=32):
        super().__init__()
        self.factor = factor
        self.pad_x = pad_x - pad_x%factor
    
    def __call__(self, img):
        M, N = img.shape[-2:]
        pad_M = (self.factor - M%self.factor) % self.factor
        pad_N = (self.factor - N%self.factor) % self.factor
        return TF.pad(img, padding=(self.pad_x, 0, pad_N+self.pad_x, pad_M), padding_mode="reflect"), (M, N, self.pad_x)
    
    def __repr__(self):
        repr = f"{self.__class__.__name__}(padding={self.pad_x}, resolution_factor={self.factor})"
        return repr



def get_default_img_transforms(padding=0):
    """Tensor, dimension and normalisation default augs"""
    return T.Compose([T.ToTensor(), 
                    T.Normalize((0.1,), (0.2,)), 
                    PadandFixImage(pad_x=padding, factor=32),
                    ])

class ImgListDataset(Dataset):
    """Torch Dataset from img list"""
    def __init__(self, img_list, padding=0):
        self.img_list = img_list
        if isinstance(img_list[0], (str, PurePath)):
            self.is_arr = False
        elif isinstance(img_list[0], np.ndarray):
            self.is_arr = True
        self.transform = get_default_img_transforms(padding)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if self.is_arr:
            img = (255*self.img_list[idx]/self.img_list[idx].max()).astype(np.uint8)
        else:
            img = Image.open(self.img_list[idx])
        img, shape = self.transform(img)
        return {'img': img, "crop":shape}


def get_img_list_dataloader(img_list, padding=0, batch_size=16, num_workers=0, pin_memory=False):
    """Wrapper of Dataset into DataLoader"""
    dataset = ImgListDataset(img_list, padding=padding)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=pin_memory)
    return loader


class DeepGPET:

    DEFAULT_MODEL_URL = 'https://github.com/jaburke166/OCTolyzer/releases/download/v1.0/deepgpet_weights.pth'
    DEFAULT_THRESHOLD = 0.5
    DEFAULT_PADDING = 256
    DEFAULT_MODEL_PATH = None
    # DEFAULT_MODEL_PATH = os.path.join(SCRIPT_PATH, r"weights/deepgpet_weights.pth")    
    VERBOSITY = True

    def __init__(self, model_path=DEFAULT_MODEL_URL, threshold=DEFAULT_THRESHOLD, 
                 padding=DEFAULT_PADDING, local_model_path=DEFAULT_MODEL_PATH, verbose=VERBOSITY):
        """
        Core inference class for Peripapillary OCT B-scan choroid region segmentation model
        """
        self.padding = padding if padding >=0 else 0
        self.transform = get_default_img_transforms(self.padding)
        self.threshold = threshold
        self.verbose = ~verbose
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        if local_model_path is not None:
            self.model = torch.load(local_model_path, map_location=self.device)
        else:
            self.model = torch.hub.load_state_dict_from_url(model_path, map_location=self.device)
        if self.device != "cpu":
            print("Peripapillary choroid segmentation model has been loaded with GPU acceleration!")
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
            img, (M, N, pad_x) = self.transform(img)
            img = img.unsqueeze(0).to(self.device)
            pred = self.model(img).squeeze(0).sigmoid()
            if not soft_pred:
                pred = (pred > self.threshold).int()
            return pred.cpu().numpy()[0, :M, pad_x:N+pad_x]

    def predict_list(self, img_list, soft_pred=False):
        """Inference on a list of images without batching"""
        preds = []
        with torch.no_grad():
            for img in tqdm(img_list, desc='Predicting', leave=False, disable=self.verbose):
                pred = self.predict_img(img, soft_pred=soft_pred)
                preds.append(pred)
        return preds

    def _predict_loader(self, loader, soft_pred=False):
        """Inference from a DataLoader"""
        preds = []
        with torch.no_grad():
            for batch in tqdm(loader, desc='Predicting', leave=False, disable=self.verbose):
                img = batch['img'].to(self.device)
                batch_M, batch_N, batch_padx = batch['crop']
                pred = self.model(img).sigmoid().squeeze().cpu().numpy()
                if not soft_pred:
                    pred = (pred > self.threshold).astype(np.int64)
                pred = [p[:M, x:N+x] for (p, M, N, x) in zip(pred, batch_M, batch_N, batch_padx)]
                preds.append(pred)
        return preds

    def batch_predict(self, img_list, padding=0, soft_pred=False, batch_size=16, num_workers=0, pin_memory=False):
        """Wrapper for DataLoader inference"""
        loader = get_img_list_dataloader(img_list, padding=padding, 
                                         batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
        preds = self._predict_loader(loader, soft_pred=soft_pred)
        return preds

    def __call__(self, x):
        """Direct call for inference on single  image"""
        return self.predict_img(x)

    def __repr__(self):
        return f'{self.__class__.__name__}(threshold={self.threshold}, padding={self.padding})'
