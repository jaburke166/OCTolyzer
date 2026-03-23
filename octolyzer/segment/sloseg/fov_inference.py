import numpy as np
import torch
import os
from skimage import morphology as morph
from skimage import measure
from tqdm.autonotebook import tqdm
from pathlib import PurePath, PosixPath
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T
from torchvision.transforms import functional as TF
from torchvision import tv_tensors
import torch.nn as nn

SCRIPT_PATH = os.path.realpath(os.path.dirname(__file__))

class FixShape(T.Transform):
    def __init__(self, factor=32):
        """Forces input to have dimensions divisble by 32"""
        super().__init__()
        self.factor = factor

    def __call__(self, img):
        M, N = img.shape[-2:]
        pad_M = (self.factor - M%self.factor) % self.factor
        pad_N = (self.factor - N%self.factor) % self.factor
        return TF.pad(img, padding=(0, 0, pad_N, pad_M)), (M, N)

    def __repr__(self):
        return self.__class__.__name__
    

def get_default_img_transforms():
    """Tensor, dimension and normalisation default augs"""
    return T.Compose([
        T.PILToTensor(),
        T.Resize(size=(768,768), antialias=True),
        T.ToDtype(torch.float32, scale=True),
        FixShape(factor=32)
    ])
 


class ImgListDataset(Dataset):
    """Torch Dataset from img list"""
    def __init__(self, img_list):
        self.img_list = img_list
        if isinstance(img_list[0], (str, PurePath, PosixPath)):
            self.is_arr = False
        elif isinstance(img_list[0], np.ndarray):
            self.is_arr = True
        self.transform = get_default_img_transforms()

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if self.is_arr:
            img_uint = (255*(self.img_list[idx]/self.img_list[idx].max())).astype(np.uint8)
            img = ImageOps.grayscale(Image.fromarray(img_uint))
        else:
            img = ImageOps.grayscale(Image.open(self.img_list[idx]))
            
        img, shape = self.transform(img)
        return {'img': img, "crop":shape} 


def get_img_list_dataloader(img_list, batch_size=16, num_workers=0, pin_memory=False):
    """Wrapper of Dataset into DataLoader"""
    dataset = ImgListDataset(img_list)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=pin_memory)
    return loader


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


def _process_fovea_prediction(preds):    
    """Wrapper to build filter, aggregate fovea prediction map and extract 
    largest row and column"""
    fovea = []
    for d, k in zip([-2, -1], [21, 21]):
        fovea_signal_filter = _get_fov_filter(k).to(preds.device)
        fov_signal = preds.sum(dim=d)
        fov_signal = fovea_signal_filter(fov_signal).squeeze()
        fovea.append(fov_signal.argmax(dim=-1).numpy())
    
    return np.asarray(fovea).T.reshape(-1,2)


def _get_fovea(pred, threshold=0.5):
    '''
    Extract fovea coordinate from thresholded mask
    '''
    # If fovea mask predicts lower than threshold, apply filter instead of centroid
    if isinstance(pred, np.ndarray):
        binmask = pred > threshold
    else:
        binmask = (pred > threshold).cpu().numpy()[0]
    if binmask.sum() == 0:
        if isinstance(pred, np.ndarray):
            torch_pred = torch.from_numpy(pred.astype(np.float32)).view(1,1,*pred.shape)
        else:
            torch_pred = pred.clone()
        fovea = _process_fovea_prediction(torch_pred)[0]
        return fovea

    # Look at which of the region has the largest area, and set all other regions to 0
    labels_mask = measure.label(binmask)                       
    regions = measure.regionprops(labels_mask)
    regions.sort(key=lambda x: x.area, reverse=True)
    if len(regions) > 1:
        for rg in regions[1:]:
            labels_mask[rg.coords[:,0], rg.coords[:,1]] = 0
    labels_mask[labels_mask!=0] = 1

    # Select centroid for largest mask
    label_mask = measure.label(labels_mask)                       
    region = measure.regionprops(label_mask)[0]
    fovea = np.array(region.centroid).astype(int)[[1,0]]
    
    return fovea
    

class FOVSegmenter:

    DEFAULT_MODEL_URL = 'https://github.com/jaburke166/OCTolyzer/releases/download/v1.0/fovsegmenter_weights.pth'
    DEFAULT_THRESHOLD = 0.5
    DEFAULT_MODEL_PATH = None
    #DEFAULT_MODEL_PATH = os.path.join(SCRIPT_PATH, r"weights/fovsegmenter_weights.pth")  
    
    def __init__(self, model_path=DEFAULT_MODEL_URL, threshold=DEFAULT_THRESHOLD, local_model_path=DEFAULT_MODEL_PATH):
        """
        Core inference class for Fovea SLO segmentation model
        """
        self.transform = get_default_img_transforms()
        self.threshold = threshold
        if torch.cude.is_available():
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
            print("Fovea detection has been loaded with GPU acceleration!")
        self.model.eval()
        
    @torch.inference_mode()
    def predict_img(self, img, soft_pred=False):
        """
        Inference on a single image
        """
        if isinstance(img, (str, PurePath, PosixPath)):
            img = ImageOps.grayscale(Image.open(img))
            img_shape = (img.height, img.width)
        elif isinstance(img, np.ndarray):
            img_shape = img.shape
            img = ImageOps.grayscale(Image.fromarray(img))
            
        RESIZE = T.Resize(size=img_shape, antialias=True)
        with torch.no_grad():
            img, (M, N) = self.transform(img)
            img = img.unsqueeze(0).to(self.device)
            pred = self.model(img).squeeze(0).sigmoid()[:M, :N].cpu()

            # Resize back to native resolution
            if RESIZE is not None:
                pred = RESIZE(pred)

            # Return if soft_pred, otherwise post-process
            pred = pred.numpy()[0]
            if soft_pred:
                return pred
            fovea = _get_fovea(pred, self.threshold)
    
            return pred, fovea

    def predict_list(self, img_list, soft_pred=False):
        """Inference on a list of images without batching"""
        preds = []
        foveas = []
        with torch.no_grad():
            for img in tqdm(img_list, desc='Predicting', leave=False):
                pred, fovea = self.predict_img(img, soft_pred=soft_pred)
                preds.append(pred)
                foveas.append(fovea)
        return preds, foveas

    # Mixed resolution will not work here.
    # Currently does not support resizing to common (768,768) resolution
    def _predict_loader(self, loader, soft_pred=False):
        """Inference from a DataLoader"""
        preds = []
        foveas = []
        with torch.no_grad():
            for batch in tqdm(loader, desc='Predicting', leave=False):
                img = batch['img'].to(self.device)
                batch_M, batch_N = batch['crop']
                pred = self.model(img).sigmoid()
                if not soft_pred:
                    fovea = [_get_fovea(p, self.threshold) for p in pred]
                pred = [p[:M,:N] for (p, M, N) in zip(pred, batch_M, batch_N)]
                preds.append(pred)
                foveas.append(fovea)
        return preds, foveas

    def predict_batch(self, img_list, soft_pred=False, batch_size=16, num_workers=0, pin_memory=False):
        """Wrapper for DataLoader inference"""
        loader = get_img_list_dataloader(img_list, 
                                         batch_size=batch_size, 
                                         num_workers=num_workers,
                                         pin_memory=pin_memory)
        preds, foveas = self._predict_loader(loader, soft_pred=soft_pred)
        return preds, foveas

    def __call__(self, x):
        """Direct call for inference on single image"""
        return self.predict_img(x)

    def __repr__(self):
        return f'{self.__class__.__name__}(threshold={self.threshold})'
