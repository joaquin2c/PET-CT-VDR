import os
import numpy as np

from scipy.ndimage import rotate
import torch
from tqdm import tqdm
import pandas as pd
import argparse
from swin3d.swin3d import swin_3D
import nrrd
import h5py
import torchvision.transforms as T
from torchvision.ops.misc import Permute

def save_features(filename, all_features, all_masks, patient_id):
    """ Save features and mask in a .hdf5 dataset file

    Args:
        filename (str): dataset.hdf5 path.
        all_features (list(np.array)): list of feature maps of each slice.
        all_masks (list(np.array)): list of nodule maks of each slice.
        patient_id (str): id of the patient.

    """
    with h5py.File(filename, 'a') as h5f:
        if patient_id in h5f:
            print(f'features for {patient_id} already exists')
            del h5f[patient_id]
        patient_group = h5f.create_group(patient_id)
        for i, (feature, mask) in enumerate(zip(all_features, all_masks)):
            patient_group.create_dataset(f'features/{i}',
                                         compression="lzf",
                                         data=feature,
                                         chunks=feature.shape)
            patient_group.create_dataset(f'masks/{i}',
                                         compression="lzf",
                                         data=mask,
                                         chunks=mask.shape)

def windowing_ct(width, level):
    lower_bound = level - width/2
    upper_bound = level + width/2
    return lower_bound, upper_bound

def apply_window_ct(ct, width, level):
    """ Normalize CT image using a window in the HU scale

    Args:
        ct (np.array): ct image.
        width (int): window width in the HU scale.
        level (int): center of the windows in the HU scale.

    Returns:
        ct (np.array): Normalized image in a range 0-1.

    """
    ct_min_val, ct_max_val = windowing_ct(width, level)
    ct_range = ct_max_val - ct_min_val
    ct = (ct - ct_min_val) / ct_range
    ct = np.clip(ct, 0, 1)
    return ct

def prepare_image(img, modality,model_name='swin3D'):
    """ Resize image and convert to torch cuda tensor

    Args:
        img (np.array): image of a slice with shape (h, w, ch) in a range (0-1).

    Returns:
        img_tensor (torch.tensor): image as cuda tensor with shape (batch, h, w, ch).

    """
    if model_name == 'swin3D':
        if modality=="ct":
            normalize = T.Normalize(mean=[0.1683], std=[0.2302]) #CT
        else:
            normalize = T.Normalize(mean=[0.0660], std=[0.2874]) #PET

        image = torch.tensor(img,dtype=torch.float32).unsqueeze(0) #[H,W,D]  -->  [C,H,W,D]
        image = image.permute(3, 0, 1, 2)         #[D,C,H,W] 
        
        permute = Permute([1, 0, 2, 3]) #[D,C,H,W] --> [C,D,H,W]
        transform = T.Compose([
            T.Resize((32,32)), #32
            #T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.)),
            permute,
            normalize])
    
        #img_tensor = (img - mu) / std 
        img_tensor = transform(image)
        """
        img_tensor=img
        img_tensor = img_tensor.transpose((2, 0, 1))
        print("after trans",np.shape(img_tensor))
        img_tensor = np.expand_dims(np.expand_dims(img_tensor, axis=0),axis=0)
        print("after expand",np.shape(img_tensor))
        """
    device = torch.cuda.current_device()
    #img_tensor = torch.as_tensor(img_tensor, dtype=torch.float32).to(f"cuda:{device}")
    img_tensor = img_tensor.to(f"cuda:{device}")
    #print(np.shape(img_tensor))
    return img_tensor

def get_dense_descriptor(model3d, img, modality):
    """ Use swin3d to extract patch embeddings

    Args:
        model (torch.nn.Module): 3D image encoder
        img (np.array): image3d of an volume with shape (D,H,W).

    Returns:
        features (np.array): slice feature maps with shape (N//patch_size, M//patch_size, feature_dim).

    """
    img_tensor = prepare_image(img, modality,model_name=model.model_name)
    if model.model_name == 'swin3D':
        with torch.no_grad():
            features_tensor = np.squeeze(model3d(img_tensor.unsqueeze(0)).cpu().detach().numpy())
        #print(np.shape(features_tensor))
    return features_tensor

def crop_image(img, xmin, ymin, zmin, xmax, ymax, zmax):
    cropped = img[ymin:ymax, xmin:xmax,zmin:zmax]
    return cropped

    
def extract_coords_limit(mask):
    indices = np.array(np.where(mask))
    ymin = np.min(indices[0, :])
    xmin = np.min(indices[1, :])
    zmin = np.min(indices[2, :])
    ymax = np.max(indices[0, :])+1
    xmax = np.max(indices[1, :])+1
    zmax = np.max(indices[2, :])
    
    d = zmax - zmin
    
    dsize=(d//32+1)*32
    ddiff=(dsize-d)//2
    zmax = zmin+dsize
    return xmin, ymin, zmin, xmax, ymax, zmax
    
def extract_coords(mask):
    indices = np.array(np.where(mask))
    ymin = np.min(indices[0, :])
    xmin = np.min(indices[1, :])
    zmin = np.min(indices[2, :])
    ymax = np.max(indices[0, :])
    xmax = np.max(indices[1, :])
    zmax = np.max(indices[2, :])
    
    h = ymax - ymin
    w = xmax - xmin
    d = zmax - zmin
    
    sizemax=np.max((h,w))
    hwsize=(sizemax//32+1)*32
    dsize=(d//32+1)*32
    
    hwdiff=(hwsize-sizemax)//2
    ddiff=(dsize-d)//2
    
    ymin=max(ymin-hwdiff,0)
    xmin=max(xmin-hwdiff,0)
    zmin=max(zmin-ddiff,0)
    ymax = ymin+hwsize
    xmax = xmin+hwsize
    zmax = zmin+dsize
    return xmin, ymin, zmin, xmax, ymax, zmax

def generate_features(model, img_3d, mask_3d, modality,tqdm_text):
    """ Extract feature map of each slice and crop them to focus on nodule region.

    Args:
        model (torch.nn.Module): ViT image encoder
        img_3d (np.array): CT or PET 3D data with shape (H, W, slices, Ch).
        mask_3d (np.array): 3D nodule boolean mask with shape (H, W, slices).
        tqdm_text (str): description to display in the tqdm loading bar.
        display (bool, optional): To visualize images and extracted features. Defaults to False.

    Returns:
        features_list (List(np.array)): featuremap of each slice cropped to the nodule region.
        mask_list (List(np.array)):  binary mask of each slice cropped to the nodule region.

    """
    limit=True
    if limit:
        xmin, ymin, zmin, xmax, ymax, zmax = extract_coords_limit(mask_3d) ##LIMITE
    else:
        xmin, ymin, zmin, xmax, ymax, zmax = extract_coords(mask_3d) ##LIMITE
    img_3d = crop_image(img_3d, xmin, ymin, zmin, xmax, ymax, zmax)
    mask_3d = crop_image(mask_3d, xmin, ymin, zmin, xmax, ymax, zmax)
    mask_3d = mask_3d.transpose((2, 0, 1))
    #print(np.shape(img_3d),np.shape(mask_3d),np.shape(bigger_mask))

    features = get_dense_descriptor(model, img_3d, modality)
    return features, mask_3d

def rotate_image(image, mask, angle, axes=(0, 1)):
    """ Rotates a 3D image and mask in the plane XY

    Args:
        image (np.array): CT or PET 3D data with shape (H, W, slices, Ch).
        mask (np.array): 3D nodule boolean mask with shape (H, W, slices).
        angle (int): rotation angle in degrees.
        axes (tuple, optional): rotation axis. Defaults to (0, 1).

    Returns:
        image_rot (TYPE): rotated 3D image.
        mask_rot (TYPE): rotated 3D mask.

    """
    image_rot = image.copy()
    mask_rot = mask.copy()
    if angle == 0:
        return image_rot, mask_rot
    image_rot = rotate(image_rot, angle, axes=axes, reshape=False, mode='nearest')
    #image_rot = np.clip(image_rot, 0, 1)
    mask_rot = rotate(mask_rot, angle, axes=axes, reshape=False, mode='nearest')
    mask_rot = mask_rot > 0
    return image_rot, mask_rot

def flip_image(image, mask, flip_type):
    """ Flip a 3D image and mask horizontal or vertically

    Args:
        image (np.array): CT or PET 3D data with shape (H, W, slices, Ch).
        mask (np.array): 3D nodule boolean mask with shape (H, W, slices).
        flip_type (str): None, 'horizontal' or 'vertical'.

    Returns:
        image_flip (np.array): flipped 3D image.
        mask_flip (np.array): flipped 3D mask.

    """
    image_flip = image.copy()
    mask_flip = mask.copy()
    if flip_type == 'horizontal':
        return image_flip[:, ::-1, ...], mask_flip[:, ::-1, ...]
    elif flip_type == 'vertical':
        return image_flip[::-1, ...], mask_flip[::-1, ...]
    return image_flip, mask_flip


def nrrd2voxels(data_info, dataset_path, modality):
    """ Get and stack patient slices into volumetrics arrays

    Args:
        data_info (pd.DataFrame): Info of patient.
        dataset_path (str): path to images.
        pet (bool, optional): pet images are divided by the mean of the liver pet. Defaults to False.

    Returns:
        img (np.array): CT or PET 3D data with shape (H, W, slices).
        mask (np.array): 3D nodule mask with shape (H, W, slices).
        label (np.array): nodule EGFR class 0: wildtype, 1: mutant, 2: unknown 3: not collected.

    """
    patient_id=data_info["patient_id"].values[0]
    path_img = os.path.join(dataset_path,patient_id,modality,data_info[f"{modality}_img_name"].values[0])
    path_mask = os.path.join(dataset_path,patient_id,modality,data_info[f"{modality}_seg_name"].values[0])
    img, _ = nrrd.read(path_img)
    msk, _ = nrrd.read(path_mask)
    label = data_info["label"].values[0]
    pet_liver_mean = 1
   
    if modality=="pet":
        pet_liver_mean = data_info["pet_mean"].values[0] + 1e-10
        img = img / pet_liver_mean
    else:
        img = apply_window_ct(img, width=1800, level=40)
    
    return img, msk, label



def load_swin3D(model_path):
    """ Load swin3D model

    Returns:
        model (torch.nn.Module): loaded torch model.

    """
    model3d=swin_3D(patch_size=(4,4,4))#return_all_tokens=True)
    device = torch.cuda.current_device()
    pesos=torch.load(model_path, weights_only=False,map_location=f"cuda:{device}")
    

    pesos_corrected = {}
    
    for key in pesos["teacher"]:
        if key[:8]=="backbone":
            pesos_corrected[key[9:]] = pesos["teacher"][key]
        
    model3d.load_state_dict(pesos_corrected)#,strict=False)
    model3d.to(f"cuda:{device}")
    model3d.eval()
    print("pesos cargados correctamente")
    return model3d

def load_model(model_name, model_path=None):
    """ Load a model as image encoder

    Args:
        model_name (str): 'swin3D'.
        model_path (str, optional): path to the .pth file. Defaults to None.

    Returns:
        model (torch.nn.Module): loaded torch model.

    """
    if model_name == 'swin3D':
        model = load_swin3D(model_path)
    else:
        model = None
    model.model_name = model_name
    
    
    
    return model




#python tfds_dense_descriptor_3D.py --model_path ../../../../shared_data/PET-CT/weights/ct_stanford/checkpoint.pth -f ../../../../shared_data/PET-CT/data/feature_name --modality ct -gpu 1
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Obtener ViT patch embeddings de los dataset lung_radiomics")

    parser.add_argument("-mn", "--model_name", type=str, default="swin3D",
                        help="backbone ViT of swin3D")
    
    parser.add_argument("-mp", "--model_path", type=str,
                        default="../model.pth",help="path del archivo model.pth")
    
    parser.add_argument("-d", "--dataset_path", type=str, default="../../../data/NSCLC_Radiogenomics/images/",
                        help="path de los imagenes CT y PET")
    
    parser.add_argument("-f", "--feature_folder", type=str, default=os.path.join('../../../Data/PET-CT','data', 'features_norm_correct'),
                        help="carpeta de salida donde se guardaran los features")
    
    parser.add_argument("-df", "--df_path", type=str, default="../../../data/NSCLC_Radiogenomics/info_dataset.csv",
                        help="path a los metadatos del dataset")
    
    parser.add_argument("-mod", "--modality", type=str, default='ct',
                        help="path a los metadatos del dataset")
    
    parser.add_argument("-gpu", "--gpu_device", type=int, default=1, help="cuda device number")

    args = parser.parse_args()
    model_name = args.model_name
    model_path = args.model_path
    dataset_path = args.dataset_path
    feature_folder = args.feature_folder
    df_metdata_path = args.df_path
    modality = args.modality
    #pet_liver=args.petliver
    gpu_device = args.gpu_device
    torch.cuda.set_device(int(gpu_device))

    model = load_model(model_name, model_path)    
    model.to(f"cuda:{gpu_device}")
    datasets = ['stanford_dataset'] #'santa_maria_dataset'
    
    df_metadata = pd.read_csv(df_metdata_path)
    df_metadata=df_metadata[df_metadata["egfr_exist"]==1]
    df_metadata=df_metadata[df_metadata["pet_segmentation"]==True]
    df_metadata['label'] = (df_metadata['egfr'] == 'Mutant').astype(int)
    #patient2label = dict(zip(df_metadata['patient_id'], df_metadata['label']))
    if modality == 'ct':
        df_metadata = df_metadata[df_metadata["ct_segmentation"]==True]
    df_metadata.reset_index(inplace=True, drop=True)

    for dataset_name in datasets:
        features_dir = os.path.join(feature_folder, dataset_name)
        os.makedirs(features_dir, exist_ok=True)
        
        patient_ids = list(df_metadata["patient_id"])  
    
        for patient_id in tqdm(patient_ids, desc=dataset_name):
            data_info=df_metadata[df_metadata["patient_id"]==patient_id]
            #img_path = os.path.join(dataset_path, patient_id,file_data[f"{modality}_img_name"])
            df_path = os.path.join(features_dir, f'{patient_id}_{modality}.parquet')
            features_file = os.path.join(feature_folder, f'features_masks_{modality}.hdf5') 
            if not os.path.exists(df_path):
                
                img_raw, mask_raw, label = nrrd2voxels(data_info, dataset_path, modality) 
                
                # normalize pixel values
                
                # extract patch features of each slice
                df = {'angle': [],
                      'flip': []}
    
                all_features = []
                all_masks = []
                angles = []
                flips = []
                slices = []
                # apply flip and rotation to use them as offline data augmentation
                for flip_type in [None, 'horizontal', 'vertical']:
                    image_flip, mask_flip = flip_image(img_raw, mask_raw, flip_type) 
                    for angle in [0, 90]:#range(0, 180, 45):
                        image, mask = rotate_image(image_flip, mask_flip, angle)
                        features, features_mask = generate_features(model=model, 
                                                                    img_3d=image,
                                                                    mask_3d=mask,
                                                                    modality=modality,
                                                                    tqdm_text=f'{modality} {patient_id}')
    
                        all_masks.append(features_mask)
                        all_features.append(features)
    
                        df['angle'] += [angle] 
                        df['flip'] += [flip_type] 
    
                # store metadata of each featuremap in a dataframe
                df = pd.DataFrame(df)
                df.reset_index(drop=False, inplace=True)
                df = df.rename(columns={'index': 'feature_id'})
                df['patient_id'] = patient_id
                df['label'] = label
                df['dataset'] = dataset_name.replace('_dataset', '')
                df['modality'] = modality
                df['augmentation'] = np.logical_not(np.logical_and(df['flip'] is None,  df['angle'] == 0))
                df.to_parquet(df_path)
                save_features(features_file, all_features, all_masks, patient_id)
