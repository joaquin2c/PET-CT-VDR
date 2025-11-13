# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import math
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
import nrrd
import utils

import os
import PIL
import SimpleITK as sitk
import torch
from PIL import Image, ImageOps
from torchvision.transforms.functional import pil_to_tensor
from torch.utils.data import Dataset

import pydicom as dicom

class HistoricalDocuments(Dataset):
    def __init__(self, pkl_path,
                 old_path="/media/chr/Datasets/HORAE/imgs/",
                 new_path="/home/data/cstears/horae/imgs/",
                 max_size=224, transform=None):
        try:
            self.df = pd.read_pickle(pkl_path)
        except Exception as e:
            print(f"No se pudo cargar el archivo pickle: {e}")
            raise e
        self.max_size = max_size
        self.transform = transform

        # Cambiamos el path de las imágenes
        self.change_path(old_path, new_path)
        # self.validation()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        df_idx = self.df.iloc[idx]
        crop = self.crop_image(df_idx)
        return self.transform_image(crop)

    def validation(self):
        for idx in tqdm(range(len(self.df))):
            df_idx = self.df.iloc[idx]

            x1, y1, x2, y2 = map(int, [df_idx['x1'], df_idx['y1'], df_idx['x2'], df_idx['y2']])
            if (x2 - x1) > 500 and (y2 - y1) > 500:
                tqdm.write(f"Se descartó el crop {df_idx['x1']},{df_idx['y1']},{df_idx['x2']},{df_idx['y2']} por ser muy grande")
                self.df.drop(idx, inplace=True)

    def crop_image(self, df_idx):
        path_image = df_idx['filename'].iloc[0] if isinstance(df_idx['filename'], pd.Series) else df_idx['filename']

        if not os.path.exists(path_image):
            raise FileNotFoundError(f"No se encontró el archivo de imagen: {path_image}")

        img = Image.open(path_image).convert('RGB')
        x1, y1, x2, y2 = map(int, [df_idx['x1'], df_idx['y1'], df_idx['x2'], df_idx['y2']])
        return img.crop((x1, y1, x2, y2))

    def transform_image(self, image):
        padded_image = ImageOps.pad(image, size=(self.max_size, self.max_size))
        return self.transform(padded_image)

    def save_sample(self, original, positive, negative):

        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(original.permute(1, 2, 0))
        ax[0].set_title('Original')
        ax[0].axis('off')

        ax[1].imshow(positive.permute(1, 2, 0))
        ax[1].set_title('Positive')
        ax[1].axis('off')

        ax[2].imshow(negative.permute(1, 2, 0))
        ax[2].set_title('Negative')
        ax[2].axis('off')

        plt.show()

    def change_path(self, old_path, new_path):
        self.df['filename'] = self.df['filename'].apply(lambda x: x.replace(old_path, new_path))

class HistoricalDocFolderMask(HistoricalDocuments):
    def __init__(self, *args, patch_size, pred_ratio, pred_ratio_var, pred_aspect_ratio,
                 pred_shape='block', pred_start_epoch=0, **kwargs):
        super(HistoricalDocFolderMask, self).__init__(*args, **kwargs)
        self.psz = patch_size
        self.pred_ratio = pred_ratio[0] if isinstance(pred_ratio, list) and \
            len(pred_ratio) == 1 else pred_ratio
        self.pred_ratio_var = pred_ratio_var[0] if isinstance(pred_ratio_var, list) and \
            len(pred_ratio_var) == 1 else pred_ratio_var
        if isinstance(self.pred_ratio, list) and not isinstance(self.pred_ratio_var, list):
            self.pred_ratio_var = [self.pred_ratio_var] * len(self.pred_ratio)
        self.log_aspect_ratio = tuple(map(lambda x: math.log(x), pred_aspect_ratio))
        self.pred_shape = pred_shape
        self.pred_start_epoch = pred_start_epoch

    def get_pred_ratio(self):
        if hasattr(self, 'epoch') and self.epoch < self.pred_start_epoch:
            return 0

        if isinstance(self.pred_ratio, list):
            pred_ratio = []
            for prm, prv in zip(self.pred_ratio, self.pred_ratio_var):
                assert prm >= prv
                pr = random.uniform(prm - prv, prm + prv) if prv > 0 else prm
                pred_ratio.append(pr)
            pred_ratio = random.choice(pred_ratio)
        else:
            assert self.pred_ratio >= self.pred_ratio_var
            pred_ratio = random.uniform(self.pred_ratio - self.pred_ratio_var, self.pred_ratio + \
                self.pred_ratio_var) if self.pred_ratio_var > 0 else self.pred_ratio

        return pred_ratio

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        output = super(HistoricalDocFolderMask, self).__getitem__(index)

        masks = []
        for img in output:
            try:
                H, W = img.shape[1] // self.psz, img.shape[2] // self.psz
            except:
                # skip non-image
                continue

            high = self.get_pred_ratio() * H * W

            if self.pred_shape == 'block':
                # following BEiT (https://arxiv.org/abs/2106.08254), see at
                # https://github.com/microsoft/unilm/blob/b94ec76c36f02fb2b0bf0dcb0b8554a2185173cd/beit/masking_generator.py#L55
                mask = np.zeros((H, W), dtype=bool)
                mask_count = 0
                while mask_count < high:
                    max_mask_patches = high - mask_count

                    delta = 0
                    for attempt in range(10):
                        low = (min(H, W) // 3) ** 2
                        target_area = random.uniform(low, max_mask_patches)
                        aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                        h = int(round(math.sqrt(target_area * aspect_ratio)))
                        w = int(round(math.sqrt(target_area / aspect_ratio)))
                        if w < W and h < H:
                            top = random.randint(0, H - h)
                            left = random.randint(0, W - w)

                            num_masked = mask[top: top + h, left: left + w].sum()
                            if 0 < h * w - num_masked <= max_mask_patches:
                                for i in range(top, top + h):
                                    for j in range(left, left + w):
                                        if mask[i, j] == 0:
                                            mask[i, j] = 1
                                            delta += 1

                        if delta > 0:
                            break

                    if delta == 0:
                        break
                    else:
                        mask_count += delta

            elif self.pred_shape == 'rand':
                mask = np.hstack([
                    np.zeros(H * W - int(high)),
                    np.ones(int(high)),
                ]).astype(bool)
                np.random.shuffle(mask)
                mask = mask.reshape(H, W)

            else:
                # no implementation
                assert False

            masks.append(mask)

        return (output, masks)


class CTDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_files[idx])
        image = PIL.Image.open(img_path)
        image = pil_to_tensor(image).float().unsqueeze(0)
        if self.transform:
            image = self.transform(image)
        return image

class CTFolderMask(CTDataset):
    def __init__(self, *args, patch_size, pred_ratio, pred_ratio_var, pred_aspect_ratio,
                 pred_shape='block', pred_start_epoch=0, **kwargs):
        super(CTFolderMask, self).__init__(*args, **kwargs)
        self.psz = patch_size
        self.pred_ratio = pred_ratio[0] if isinstance(pred_ratio, list) and \
            len(pred_ratio) == 1 else pred_ratio
        self.pred_ratio_var = pred_ratio_var[0] if isinstance(pred_ratio_var, list) and \
            len(pred_ratio_var) == 1 else pred_ratio_var
        if isinstance(self.pred_ratio, list) and not isinstance(self.pred_ratio_var, list):
            self.pred_ratio_var = [self.pred_ratio_var] * len(self.pred_ratio)
        self.log_aspect_ratio = tuple(map(lambda x: math.log(x), pred_aspect_ratio))
        self.pred_shape = pred_shape
        self.pred_start_epoch = pred_start_epoch

    def get_pred_ratio(self):
        if hasattr(self, 'epoch') and self.epoch < self.pred_start_epoch:
            return 0

        if isinstance(self.pred_ratio, list):
            pred_ratio = []
            for prm, prv in zip(self.pred_ratio, self.pred_ratio_var):
                assert prm >= prv
                pr = random.uniform(prm - prv, prm + prv) if prv > 0 else prm
                pred_ratio.append(pr)
            pred_ratio = random.choice(pred_ratio)
        else:
            assert self.pred_ratio >= self.pred_ratio_var
            pred_ratio = random.uniform(self.pred_ratio - self.pred_ratio_var, self.pred_ratio + \
                self.pred_ratio_var) if self.pred_ratio_var > 0 else self.pred_ratio

        return pred_ratio

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        output = super(CTFolderMask, self).__getitem__(index)

        masks = []
        for img in output:
            try:
                H, W = img.shape[1] // self.psz, img.shape[2] // self.psz
            except:
                # skip non-image
                continue

            high = self.get_pred_ratio() * H * W

            if self.pred_shape == 'block':
                # following BEiT (https://arxiv.org/abs/2106.08254), see at
                # https://github.com/microsoft/unilm/blob/b94ec76c36f02fb2b0bf0dcb0b8554a2185173cd/beit/masking_generator.py#L55
                mask = np.zeros((H, W), dtype=bool)
                mask_count = 0
                while mask_count < high:
                    max_mask_patches = high - mask_count

                    delta = 0
                    for attempt in range(10):
                        low = (min(H, W) // 3) ** 2
                        target_area = random.uniform(low, max_mask_patches)
                        aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                        h = int(round(math.sqrt(target_area * aspect_ratio)))
                        w = int(round(math.sqrt(target_area / aspect_ratio)))
                        if w < W and h < H:
                            top = random.randint(0, H - h)
                            left = random.randint(0, W - w)

                            num_masked = mask[top: top + h, left: left + w].sum()
                            if 0 < h * w - num_masked <= max_mask_patches:
                                for i in range(top, top + h):
                                    for j in range(left, left + w):
                                        if mask[i, j] == 0:
                                            mask[i, j] = 1
                                            delta += 1

                        if delta > 0:
                            break

                    if delta == 0:
                        break
                    else:
                        mask_count += delta

            elif self.pred_shape == 'rand':
                mask = np.hstack([
                    np.zeros(H * W - int(high)),
                    np.ones(int(high)),
                ]).astype(bool)
                np.random.shuffle(mask)
                mask = mask.reshape(H, W)

            else:
                # no implementation
                assert False

            masks.append(mask)

        return (output, masks)

class CTDataset3D(Dataset):
    def __init__(self, image_folder, transform=None, batch_size=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith(('.nii', '.nii.gz'))]
        self.batch_size = batch_size

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_files[idx])
        image = sitk.ReadImage(img_path)
        image = sitk.GetArrayFromImage(image)
        if self.batch_size:
            image = self.process_img(image)
        image = torch.tensor(image).unsqueeze(0) #[C,D,H,W]
        image = image.permute(1, 0, 2, 3)        #[D,C,H,W]
        if self.transform:
            image = self.transform(image)
        return image

    def process_img(self, img):
        D, _, _ = img.shape
        while D != self.batch_size:
            if D < self.batch_size:
                padding_needed = self.batch_size - D
                img = np.pad(img, ((0, padding_needed), (0, 0), (0, 0)), mode='constant')
            elif D > 250 and self.batch_size < 125:
                img = img[::2, :, :]
            else:
                crop_size = (D - self.batch_size) // 2
                img = img[crop_size:-crop_size, :, :]
            D, _, _ = img.shape
        return img

class CTFolderMask3D(CTDataset3D):
    def __init__(self, *args, patch_size, pred_ratio, pred_ratio_var, pred_aspect_ratio,
                 pred_shape='block', pred_start_epoch=0, **kwargs):
        super(CTFolderMask3D, self).__init__(*args, **kwargs)
        self.psz = patch_size
        self.pred_ratio = pred_ratio[0] if isinstance(pred_ratio, list) and \
            len(pred_ratio) == 1 else pred_ratio
        self.pred_ratio_var = pred_ratio_var[0] if isinstance(pred_ratio_var, list) and \
            len(pred_ratio_var) == 1 else pred_ratio_var
        if isinstance(self.pred_ratio, list) and not isinstance(self.pred_ratio_var, list):
            self.pred_ratio_var = [self.pred_ratio_var] * len(self.pred_ratio)
        self.log_aspect_ratio = tuple(map(lambda x: math.log(x), pred_aspect_ratio))
        self.pred_shape = pred_shape
        self.pred_start_epoch = pred_start_epoch

    def get_pred_ratio(self):
        if hasattr(self, 'epoch') and self.epoch < self.pred_start_epoch:
            return 0

        if isinstance(self.pred_ratio, list):
            pred_ratio = []
            for prm, prv in zip(self.pred_ratio, self.pred_ratio_var):
                assert prm >= prv
                pr = random.uniform(prm - prv, prm + prv) if prv > 0 else prm
                pred_ratio.append(pr)
            pred_ratio = random.choice(pred_ratio)
        else:
            assert self.pred_ratio >= self.pred_ratio_var
            pred_ratio = random.uniform(self.pred_ratio - self.pred_ratio_var, self.pred_ratio + \
                self.pred_ratio_var) if self.pred_ratio_var > 0 else self.pred_ratio

        return pred_ratio

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        output = super(CTFolderMask3D, self).__getitem__(index)

        masks = []
        for img in output:
            try:
                D, H, W = math.ceil(img.shape[-3] / self.psz), img.shape[-2] // self.psz, img.shape[-1] // self.psz
            except:
                # skip non-image
                continue

            high = self.get_pred_ratio() * H * W

            if self.pred_shape == 'block':
                # following BEiT (https://arxiv.org/abs/2106.08254), see at
                # https://github.com/microsoft/unilm/blob/b94ec76c36f02fb2b0bf0dcb0b8554a2185173cd/beit/masking_generator.py#L55
                mask = np.zeros((D, H, W), dtype=bool)
                mask_count = 0
                while mask_count < high:
                    max_mask_patches = high - mask_count

                    delta = 0
                    for attempt in range(10):
                        low = (min(H, W) // 3) ** 2 
                        target_area = random.uniform(low, max_mask_patches)
                        aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                        h = int(round(math.sqrt(target_area * aspect_ratio)))
                        w = int(round(math.sqrt(target_area / aspect_ratio)))
                        if w < W and h < H:
                            top = random.randint(0, H - h)
                            left = random.randint(0, W - w)

                            num_masked = mask[:, top: top + h, left: left + w].sum()
                            if 0 < h * w - num_masked <= max_mask_patches:
                                for i in range(top, top + h):
                                    for j in range(left, left + w):
                                        if not mask[:, i, j].any():
                                            mask[:, i, j] = 1
                                            delta += 1

                        if delta > 0:
                            break

                    if delta == 0:
                        break
                    else:
                        mask_count += delta

            elif self.pred_shape == 'rand':
                mask = np.hstack([
                    np.zeros(H * W - int(high)),
                    np.ones(int(high)),
                ]).astype(bool)
                np.random.shuffle(mask)
                mask = mask.reshape(H, W)
                mask = np.tile(mask, (D, 1, 1))
            else:
                # no implementation
                assert False

            masks.append(mask)

        return (output, masks)


def get_dcm(path):
    pat_files=os.listdir(path)
    pat_files.sort()
    img_3d=[]
    for i in pat_files:
        dicom_img=dicom.dcmread(os.path.join(path,i))
        slope=dicom_img[(0x0028, 0x1053)].value
        intercept=dicom_img[(0x0028, 0x1052)].value
        img_3d.append(dicom_img.pixel_array*slope + intercept)
    array_3d = np.array(img_3d)
    return array_3d.transpose(1,2,0)


class PETCTDataset3D(Dataset):
    def __init__(self, image_folder, csv_path, mode, transform=None, batch_size=None):
        self.image_folder = image_folder
        dataframe = pd.read_csv(csv_path)
        self.transform = transform
        self.type = mode
        assert mode in ["ct","pet","chest_ct"],'Mode must be "ct","pet" or "chest_ct".'
        self.dataframe_mode=dataframe[dataframe[f"{mode}_segmentation"]==True]
        #self.dataframe_mode=self.dataframe_mode[self.dataframe_mode["in_range"]==1] 
        self.batch_size = batch_size

    def __len__(self):
        return len(self.dataframe_mode)

    def __getitem__(self, idx):
        file_data=self.dataframe_mode.iloc[idx]
        img_path = os.path.join(self.image_folder,file_data["patient_id"],self.type,file_data[f"{self.type}_img_name"])
        image, _ = nrrd.read(img_path)
        if self.type=="ct":
            image = utils.apply_window_ct(image, width=1800, level=40)
        else:
            image = image/file_data["pet_mean"]
        if self.batch_size:
            image = self.process_img(image)
        image = torch.tensor(image,dtype=torch.float32).unsqueeze(0) #[C,H,W,D]
        image = image.permute(3, 0, 1, 2)        #[D,C,H,W]
        if self.transform:
            image = self.transform(image)
        return image, file_data["patient_id"]

    def process_img(self, img):
        _, _ , D= img.shape
        if D < self.batch_size:
            padding_needed = self.batch_size - D
            img = np.pad(img, ((0, 0), (0, 0), (0, padding_needed)), mode='constant')
        else:
            crop_size = (D - self.batch_size) // 2
            img = img[:, :, crop_size:crop_size+self.batch_size]
        return img





class PETCTFolderMask3D(PETCTDataset3D):
    def __init__(self, *args, patch_size, pred_ratio, pred_ratio_var, pred_aspect_ratio,
                 pred_shape='block', pred_start_epoch=0, **kwargs):
        super(PETCTFolderMask3D, self).__init__(*args, **kwargs)
        self.psz = patch_size
        self.pred_ratio = pred_ratio[0] if isinstance(pred_ratio, list) and \
            len(pred_ratio) == 1 else pred_ratio
        self.pred_ratio_var = pred_ratio_var[0] if isinstance(pred_ratio_var, list) and \
            len(pred_ratio_var) == 1 else pred_ratio_var
        if isinstance(self.pred_ratio, list) and not isinstance(self.pred_ratio_var, list):
            self.pred_ratio_var = [self.pred_ratio_var] * len(self.pred_ratio)
        self.log_aspect_ratio = tuple(map(lambda x: math.log(x), pred_aspect_ratio))
        self.pred_shape = pred_shape
        self.pred_start_epoch = pred_start_epoch

    def get_pred_ratio(self):
        if hasattr(self, 'epoch') and self.epoch < self.pred_start_epoch:
            return 0

        if isinstance(self.pred_ratio, list):
            pred_ratio = []
            for prm, prv in zip(self.pred_ratio, self.pred_ratio_var):
                assert prm >= prv
                pr = random.uniform(prm - prv, prm + prv) if prv > 0 else prm
                pred_ratio.append(pr)
            pred_ratio = random.choice(pred_ratio)
        else:
            assert self.pred_ratio >= self.pred_ratio_var
            pred_ratio = random.uniform(self.pred_ratio - self.pred_ratio_var, self.pred_ratio + \
                self.pred_ratio_var) if self.pred_ratio_var > 0 else self.pred_ratio

        return pred_ratio

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        output, patient_id = super(PETCTFolderMask3D, self).__getitem__(index)

        masks = []
        for img in output:
            try:
                D, H, W = math.ceil(img.shape[-3] / self.psz), img.shape[-2] // self.psz, img.shape[-1] // self.psz
            except:
                # skip non-image
                continue

            high = self.get_pred_ratio() * H * W

            if self.pred_shape == 'block':
                # following BEiT (https://arxiv.org/abs/2106.08254), see at
                # https://github.com/microsoft/unilm/blob/b94ec76c36f02fb2b0bf0dcb0b8554a2185173cd/beit/masking_generator.py#L55
                mask = np.zeros((D, H, W), dtype=bool)
                mask_count = 0
                while mask_count < high:
                    max_mask_patches = high - mask_count

                    delta = 0
                    for attempt in range(10):
                        low = (min(H, W) // 3) ** 2 
                        target_area = random.uniform(low, max_mask_patches)
                        aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                        h = int(round(math.sqrt(target_area * aspect_ratio)))
                        w = int(round(math.sqrt(target_area / aspect_ratio)))
                        if w < W and h < H:
                            top = random.randint(0, H - h)
                            left = random.randint(0, W - w)

                            num_masked = mask[:, top: top + h, left: left + w].sum()
                            if 0 < h * w - num_masked <= max_mask_patches:
                                for i in range(top, top + h):
                                    for j in range(left, left + w):
                                        if not mask[:, i, j].any():
                                            mask[:, i, j] = 1
                                            delta += 1

                        if delta > 0:
                            break

                    if delta == 0:
                        break
                    else:
                        mask_count += delta

            elif self.pred_shape == 'rand':
                mask = np.hstack([
                    np.zeros(H * W - int(high)),
                    np.ones(int(high)),
                ]).astype(bool)
                np.random.shuffle(mask)
                mask = mask.reshape(H, W)
                mask = np.tile(mask, (D, 1, 1))
            else:
                # no implementation
                assert False

            masks.append(mask)

        return (output, masks, patient_id)
        
class ImageFolderInstance(ImageFolder):
    def __getitem__(self, index):
        img, target = super(ImageFolderInstance, self).__getitem__(index)
        return img, target, index

class ImageFolderMask(ImageFolder):
    def __init__(self, *args, patch_size, pred_ratio, pred_ratio_var, pred_aspect_ratio,
                 pred_shape='block', pred_start_epoch=0, **kwargs):
        super(ImageFolderMask, self).__init__(*args, **kwargs)
        self.psz = patch_size
        self.pred_ratio = pred_ratio[0] if isinstance(pred_ratio, list) and \
            len(pred_ratio) == 1 else pred_ratio
        self.pred_ratio_var = pred_ratio_var[0] if isinstance(pred_ratio_var, list) and \
            len(pred_ratio_var) == 1 else pred_ratio_var
        if isinstance(self.pred_ratio, list) and not isinstance(self.pred_ratio_var, list):
            self.pred_ratio_var = [self.pred_ratio_var] * len(self.pred_ratio)
        self.log_aspect_ratio = tuple(map(lambda x: math.log(x), pred_aspect_ratio))
        self.pred_shape = pred_shape
        self.pred_start_epoch = pred_start_epoch

    def get_pred_ratio(self):
        if hasattr(self, 'epoch') and self.epoch < self.pred_start_epoch:
            return 0

        if isinstance(self.pred_ratio, list):
            pred_ratio = []
            for prm, prv in zip(self.pred_ratio, self.pred_ratio_var):
                assert prm >= prv
                pr = random.uniform(prm - prv, prm + prv) if prv > 0 else prm
                pred_ratio.append(pr)
            pred_ratio = random.choice(pred_ratio)
        else:
            assert self.pred_ratio >= self.pred_ratio_var
            pred_ratio = random.uniform(self.pred_ratio - self.pred_ratio_var, self.pred_ratio + \
                self.pred_ratio_var) if self.pred_ratio_var > 0 else self.pred_ratio

        return pred_ratio

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        output = super(ImageFolderMask, self).__getitem__(index)

        masks = []
        for img in output[0]:
            try:
                H, W = img.shape[1] // self.psz, img.shape[2] // self.psz
            except:
                # skip non-image
                continue

            high = self.get_pred_ratio() * H * W

            if self.pred_shape == 'block':
                # following BEiT (https://arxiv.org/abs/2106.08254), see at
                # https://github.com/microsoft/unilm/blob/b94ec76c36f02fb2b0bf0dcb0b8554a2185173cd/beit/masking_generator.py#L55
                mask = np.zeros((H, W), dtype=bool)
                mask_count = 0
                while mask_count < high:
                    max_mask_patches = high - mask_count

                    delta = 0
                    for attempt in range(10):
                        low = (min(H, W) // 3) ** 2
                        target_area = random.uniform(low, max_mask_patches)
                        aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                        h = int(round(math.sqrt(target_area * aspect_ratio)))
                        w = int(round(math.sqrt(target_area / aspect_ratio)))
                        if w < W and h < H:
                            top = random.randint(0, H - h)
                            left = random.randint(0, W - w)

                            num_masked = mask[top: top + h, left: left + w].sum()
                            if 0 < h * w - num_masked <= max_mask_patches:
                                for i in range(top, top + h):
                                    for j in range(left, left + w):
                                        if mask[i, j] == 0:
                                            mask[i, j] = 1
                                            delta += 1

                        if delta > 0:
                            break

                    if delta == 0:
                        break
                    else:
                        mask_count += delta

            elif self.pred_shape == 'rand':
                mask = np.hstack([
                    np.zeros(H * W - int(high)),
                    np.ones(int(high)),
                ]).astype(bool)
                np.random.shuffle(mask)
                mask = mask.reshape(H, W)

            else:
                # no implementation
                assert False

            masks.append(mask)

        return output + (masks,)


class PETCTFolderMask3D_MLNS(PETCTFolderMask3D):
    def __init__(self, *args, patch_size, pred_ratio, pred_ratio_var, pred_aspect_ratio,
                 pred_shape='block', pred_start_epoch=0, **kwargs):
        
        super(PETCTFolderMask3D_MLNS, self).__init__(*args,**kwargs, patch_size=patch_size, pred_ratio=pred_ratio, pred_ratio_var=pred_ratio_var, pred_aspect_ratio=pred_aspect_ratio, pred_shape=pred_shape, pred_start_epoch=pred_start_epoch)

        self.psz = patch_size
        self.pred_ratio = pred_ratio[0] if isinstance(pred_ratio, list) and \
            len(pred_ratio) == 1 else pred_ratio
        self.pred_ratio_var = pred_ratio_var[0] if isinstance(pred_ratio_var, list) and \
            len(pred_ratio_var) == 1 else pred_ratio_var
        if isinstance(self.pred_ratio, list) and not isinstance(self.pred_ratio_var, list):
            self.pred_ratio_var = [self.pred_ratio_var] * len(self.pred_ratio)
        self.log_aspect_ratio = tuple(map(lambda x: math.log(x), pred_aspect_ratio))
        self.pred_shape = pred_shape
        self.pred_start_epoch = pred_start_epoch


    def __getitem__(self, idx):
        file_data=self.dataframe_mode.iloc[idx]
        img_path = os.path.join(self.image_folder,file_data["patient_id"],file_data["path"])
        image = get_dcm(img_path)
        image = utils.apply_window_ct(image, width=1800, level=40)

        if self.batch_size:
            image = self.process_img(image)
        image = torch.tensor(image,dtype=torch.float32).unsqueeze(0) #[C,H,W,D]
        image = image.permute(3, 0, 1, 2)        #[D,C,H,W]
        if self.transform:
            image = self.transform(image)
        output=image
        patient_id=file_data["patient_id"]

        masks = []
        for img in output:
            try:
                D, H, W = math.ceil(img.shape[-3] / self.psz), img.shape[-2] // self.psz, img.shape[-1] // self.psz
            except:
                # skip non-image
                continue

            high = self.get_pred_ratio() * H * W

            if self.pred_shape == 'block':
                # following BEiT (https://arxiv.org/abs/2106.08254), see at
                # https://github.com/microsoft/unilm/blob/b94ec76c36f02fb2b0bf0dcb0b8554a2185173cd/beit/masking_generator.py#L55
                mask = np.zeros((D, H, W), dtype=bool)
                mask_count = 0
                while mask_count < high:
                    max_mask_patches = high - mask_count

                    delta = 0
                    for attempt in range(10):
                        low = (min(H, W) // 3) ** 2 
                        target_area = random.uniform(low, max_mask_patches)
                        aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                        h = int(round(math.sqrt(target_area * aspect_ratio)))
                        w = int(round(math.sqrt(target_area / aspect_ratio)))
                        if w < W and h < H:
                            top = random.randint(0, H - h)
                            left = random.randint(0, W - w)

                            num_masked = mask[:, top: top + h, left: left + w].sum()
                            if 0 < h * w - num_masked <= max_mask_patches:
                                for i in range(top, top + h):
                                    for j in range(left, left + w):
                                        if not mask[:, i, j].any():
                                            mask[:, i, j] = 1
                                            delta += 1

                        if delta > 0:
                            break

                    if delta == 0:
                        break
                    else:
                        mask_count += delta

            elif self.pred_shape == 'rand':
                mask = np.hstack([
                    np.zeros(H * W - int(high)),
                    np.ones(int(high)),
                ]).astype(bool)
                np.random.shuffle(mask)
                mask = mask.reshape(H, W)
                mask = np.tile(mask, (D, 1, 1))
            else:
                # no implementation
                assert False

            masks.append(mask)

        return (output, masks, patient_id)
