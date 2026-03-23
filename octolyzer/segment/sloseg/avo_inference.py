import torch
import os
import cv2
import torch.nn as nn 
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T
from torchvision import tv_tensors
import numpy as np
from PIL import Image
from tqdm.autonotebook import tqdm
from pathlib import PurePath, PosixPath
from PIL import Image, ImageOps

import sys
from skimage import measure
from skimage import morphology as morph

SCRIPT_PATH = os.path.realpath(os.path.dirname(__file__))

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

def get_default_img_transforms():
    """Tensor, dimension and normalisation default augs"""
    return T.Compose([
        T.PILToTensor(),
        T.Resize(size=(768,768), antialias=True),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=(0.5,), std=(0.5,)),
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
    loader = DataLoader(dataset, batch_size=batch_size,shuffle=False,num_workers=num_workers,pin_memory=pin_memory)
    return loader


def _get_od_centre(binmask):
    '''
    Extract optic disc centre from optic disc binary mask.
    '''
    # Look at which of the region has the largest area, and set all other regions to 0
    labels_mask = measure.label(binmask)                       
    regions = measure.regionprops(labels_mask)
    if len(regions) > 0:
        optic_disc_centre = np.array(regions[0].centroid).astype(int)[[1,0]]  
    else:
        return None
    
    return optic_disc_centre


def _select_largest_mask(binmask):
    '''
    Enforce connectivity of region segmentation
    '''
    # Look at which of the region has the largest area, and set all other regions to 0
    labels_mask = measure.label(binmask)                       
    regions = measure.regionprops(labels_mask)
    regions.sort(key=lambda x: x.area, reverse=True)
    if len(regions) > 1:
        for rg in regions[1:]:
            labels_mask[rg.coords[:,0], rg.coords[:,1]] = 0
    labels_mask[labels_mask!=0] = 1

    return (255*labels_mask).astype(np.uint8)


def _get_filled_mask(binmask):
    '''
    If the mask has holes, fill them
    '''

    label = measure.regionprops(measure.label(binmask))[0]
    bincoords = label.coords
    sty = bincoords[:,0].min()
    stx = bincoords[:,1].min()
    
    Ny = binmask.shape[0] - bincoords[:,0].max() -1
    Nx = binmask.shape[1] - bincoords[:,1].max() -1
    
    filled_patch = label.image_convex#image_filled
    filled_mask = np.pad(filled_patch, ((sty, Ny),(stx, Nx)))

    return (255*filled_mask).astype(np.uint8)
    

def _fit_ellipse(mask, get_contours=False):

    # fit minimum area ellipse around disc
    _, thresh = cv2.threshold(mask, 127, 255, 1)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    if get_contours:
        return cnt
    ellipse = cv2.fitEllipse(cnt)
    new_mask = cv2.ellipse(np.zeros_like(mask), ellipse, (255,255,255), -1)/255

    return new_mask


def _process_odmask(odmask, location=None):

    # select largest blob and ensure it's filled
    odmask = _select_largest_mask(odmask)
    odmask = _get_filled_mask(odmask)
    
    img_shape = odmask.shape
    od_centre = _get_od_centre(odmask)
    new_odmask = odmask.copy()/255.

    # Only fit ellipse is disc-centred SLO
    if location is None:
        # If location unspecified, check x-location of OD centre, if not toward edge, assume optic 
        # disc is fully visible to fit an ellipse.
        if (0.1*img_shape[1] < od_centre[0] < 0.9*img_shape[1]):
            new_odmask = _fit_ellipse(odmask)

    # Fit ellipse is location is known
    elif location == 'Optic disc':
        new_odmask = _fit_ellipse(odmask)

    
    return new_odmask


def combine_classes(pred, location=None, process_od=True):
    imAV,imA,imV,imOD,imVClass = pred
    img_shape = imAV.shape
    imout = np.zeros((*img_shape,4), dtype = np.int64)

    # artery ands vein
    imout[imA == 1]  = [1, 0, 0, 1]
    imout[imV == 1]  = [0, 0, 1, 1]
    # imout[imVClass > 0] = [1, 0, 0, 1] #if positive, red
    # imout[imVClass < 0] = [0, 0, 1, 1] #otherwise, blue

    # process optic disc and add to output
    if imOD.sum() != 0 and process_od:
        imOD = _process_odmask(imOD, location)
    imout[imOD == 1] = [0, 1, 0, 1]

    return imout


class AVOSegmenter:

    DEFAULT_MODEL_URL = 'https://github.com/jaburke166/OCTolyzer/releases/download/v1.0/avosegmenter_weights.pth'
    DEFAULT_THRESHOLD = 0.5
    DEFAULT_MODEL_PATH = None
    # DEFAULT_MODEL_PATH = os.path.join(SCRIPT_PATH, r"weights/avosegmenter_weights.pth")  
    
    def __init__(self, model_path=DEFAULT_MODEL_URL, 
                 threshold=DEFAULT_THRESHOLD, 
                 local_model_path=DEFAULT_MODEL_PATH,
                 postprocess_opticdisc=True):
        """
        Core inference class for SLO segmentation model
        """
        self.transform = get_default_img_transforms()
        self.threshold = threshold
        self.postprocess_OD = postprocess_opticdisc
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
            print("Artery-Vein-Optic disc detection has been loaded with GPU acceleration!")
        self.model.eval()

    @torch.inference_mode()
    def predict_img(self, img, vbinmap=None, location=None, soft_pred=False):
        """Inference on a single image"""
        if isinstance(img, (str, PurePath, PosixPath)):
            img = ImageOps.grayscale(Image.open(img))
            img_shape = (img.height, img.width)
        elif isinstance(img, np.ndarray):
            img_shape = img.shape
            img = ImageOps.grayscale(Image.fromarray(img))

        # Predict segmentation map and post-process
        RESIZE = T.Resize(size=img_shape, antialias=True)
        with torch.no_grad():
            img = self.transform(img)
            img = img.unsqueeze(0).to(self.device)
            pred = self.model(img).squeeze(0).sigmoid().cpu()

            # Resize back to native resolution
            if img_shape != (768,768):
                pred = RESIZE(pred)

            # Return if soft_pred, otherwise post-process
            if soft_pred:
                return pred.numpy()

            # Assuming a binary vessel map from binary SLO segmenter,
            # i.e. original setup
            if vbinmap is None:
                pred = (pred > self.threshold).numpy()
    
                # Work out vessel class
                imAV,imA,imV,imOD = pred
                imA = process_slomap(imA)
                imV = process_slomap(imV)
                imAV = process_slomap(imAV) 
            
            # If you in put the original vessel binary map, we select artery/vein class
            # dependent on highest probability from each class' probability map
            else:
                imOD = (pred[-1] > self.threshold).astype(int)
                imAV = process_slomap((pred[0] > self.threshold).astype(int))
                im_A_V1, im_A_V2 = np.zeros(img_shape), np.zeros(img_shape)
                im_A_V1[vbinmap.astype(bool)] = pred[1:3][:, vbinmap.astype(bool)].argmax(axis=0)+1
                im_A_V2[imAV.astype(bool)] += pred[1:3][:, imAV.astype(bool)].argmax(axis=0)+1
                imA = ((im_A_V1 == 1) + (im_A_V2 == 1)).astype(int)
                imV = ((im_A_V1 == 2) + (im_A_V2 == 2)).astype(int)
                imAV = (vbinmap + imA + imV).astype(bool).astype(int)

            # Create combined class-wise image
            imVClass = np.zeros(imAV.shape)
            imVClass[imAV == 1] = imA[imAV == 1] - imV[imAV == 1]  
            all_pred = (imAV,imA,imV,imOD,imVClass)
            imout = combine_classes(all_pred, location, self.postprocess_OD)

            # get optic disc centre
            od_centre = _get_od_centre(imOD)
            
            return imout, od_centre

    def predict_list(self, img_list, vbinmap_list=None, location_list=None, soft_pred=False):
        """Inference on a list of images without batching"""
        preds = []
        imouts = []
        N = len(img_list)
        if vbinmap_list is None:
            vbinmap_list = N*[None]
        if location_list is None:
            location_list = N*[None]
        with torch.no_grad():
            for img, vbmap in tqdm(zip(img_list, vbinmap_list), total=N, desc='Predicting', leave=False):
                pred, imout = self.predict_img(img, vbmap, soft_pred=soft_pred)
                preds.append(pred)
                imouts.append(imout)
        return preds, imouts

    # Mixed resolution will not work here 
    # Currently does not support resizing to common (768,768) resolution
    def _predict_loader(self, loader, soft_pred=False):
        """Inference from a DataLoader"""
        preds = []
        with torch.no_grad():
            for img, _ in tqdm(loader, desc='Predicting', leave=False):
                img = img.to(self.device)
                pred = self.model(img).sigmoid().squeeze().cpu().numpy()
                if not soft_pred:
                    pred = (pred > self.threshold).astype(np.int64)
                preds.append(pred)
        return preds

    def predict_batch(self, img_list, soft_pred=False, batch_size=16, num_workers=0, pin_memory=False):
        """Wrapper for DataLoader inference"""
        loader = get_img_list_dataloader(img_list, batch_size=batch_size, num_workers=num_workers,pin_memory=pin_memory)
        preds = self._predict_loader(loader, soft_pred=soft_pred)
        return preds
    
    def __call__(self, x):
        """Direct call for inference on single  image"""
        return self.predict_img(x)

    def __repr__(self):
        return f'{self.__class__.__name__}(threshold={self.threshold})'
