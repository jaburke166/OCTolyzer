import numpy as np
import torch
import os
import sys
from skimage import morphology as morph
from skimage import measure
from tqdm.autonotebook import tqdm
from pathlib import PurePath, PosixPath
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T
from torchvision.transforms import functional as TF
from torchvision import tv_tensors
from octolyzer.segment.sloseg import unet

SCRIPT_PATH = os.path.realpath(os.path.dirname(__file__))
sys.path.append(SCRIPT_PATH)

def process_slomap(im, strel = morph.disk(5), hole_size = 10, object_size = 150):
    """
    Post-process the resulting slo vessel map to promote connectivity.
    """
    imc = morph.binary_closing(im, strel)*1.
    im = im/np.max(im)
    im_out = np.copy(im)
    imdiff = np.logical_xor(im, imc)*1.
    labeldiff = measure.label(imdiff)
    rp = measure.regionprops(labeldiff)

    for region in rp:
        if region.area < 10:
            continue
        if region.eccentricity < 0.95:
            continue
        if region.axis_minor_length > 5:
            continue
        im_out[tuple(np.rot90(region.coords, -1))] = 1.
    
    im_out = morph.remove_small_holes(im_out > 0, hole_size)*1.
    im_out = morph.remove_small_objects(im_out > 0, object_size)*1.
    
    return im_out


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


class SLOSegmenter:

    DEFAULT_MODEL_URL = 'https://github.com/jaburke166/OCTolyzer/releases/download/v1.0/slosegmenter_weights.pth'
    DEFAULT_THRESHOLD = 0.5
    DEFAULT_MODEL_PATH = None
    # DEFAULT_MODEL_PATH = os.path.join(SCRIPT_PATH, r"weights/slosegmenter_weights.pth")  
    
    def __init__(self, model_path=DEFAULT_MODEL_URL, threshold=DEFAULT_THRESHOLD, local_model_path=DEFAULT_MODEL_PATH):
        """
        Core inference class for SLO binary vessel segmentation model
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
            print("Binary vessel detection has been loaded with GPU acceleration!")
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
            pred = self.model(img).squeeze(0).sigmoid()[1][:M, :N].cpu()

            # Resize back to native resolution
            if img_shape != (768,768):
                pred = RESIZE(pred.unsqueeze(0))[0]

            # Return if soft_pred, otherwise post-process
            if soft_pred:
                return pred.numpy()
            pred = (pred > self.threshold).int().numpy()
            pred = process_slomap(pred)

            return pred

    def predict_list(self, img_list, soft_pred=False):
        """Inference on a list of images without batching"""
        preds = []
        with torch.no_grad():
            for img in tqdm(img_list, desc='Predicting', leave=False):
                pred = self.predict_img(img, soft_pred=soft_pred)
                preds.append(pred)
        return preds

    # Mixed resolution will not work here.
    # Currently does not support resizing to common (768,768) resolution
    def _predict_loader(self, loader, soft_pred=False):
        """Inference from a DataLoader"""
        preds = []
        with torch.no_grad():
            for batch in tqdm(loader, desc='Predicting', leave=False):
                img = batch['img'].to(self.device)
                batch_M, batch_N = batch['crop']
                pred = self.model(img).sigmoid()[:,1]
                if not soft_pred:
                    pred = (pred > self.threshold).int()
                    pred = [process_slomap(p.cpu().numpy()) for p in pred]
                pred = [p[:M,:N] for (p, M, N) in zip(pred, batch_M, batch_N)]
                preds.append(pred)
        return preds

    def predict_batch(self, img_list, soft_pred=False, batch_size=16, num_workers=0, pin_memory=False):
        """Wrapper for DataLoader inference"""
        loader = get_img_list_dataloader(img_list, 
                                         batch_size=batch_size, 
                                         num_workers=num_workers,
                                         pin_memory=pin_memory)
        preds = self._predict_loader(loader, soft_pred=soft_pred)
        return preds

    def __call__(self, x):
        """Direct call for inference on single  image"""
        return self.predict_img(x)

    def __repr__(self):
        return f'{self.__class__.__name__}(threshold={self.threshold})'
