from pathlib import Path
import torchvision.transforms as T
from torchvision.ops.misc import Permute
from torchvision import models as torchvision_models
from tensorboardX import SummaryWriter
from models.head import iBOTHead
from loader import CTFolderMask, PETCTFolderMask3D, HistoricalDocFolderMask
from models.swin3D import swin_3D
from models.vit_lora import vit_lora

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


class PETCTDataset3D(Dataset):
    def __init__(self, image_folder, csv_path, mode, transform=None, batch_size=None):
        self.image_folder = image_folder
        dataframe = pd.read_csv(csv_path)
        self.transform = transform
        self.type = mode
        assert mode in ["ct","pet","chest_ct"],'Mode must be "ct","pet" or "chest_ct".'
        self.dataframe_mode=dataframe[dataframe[f"{mode}_segmentation"]==True]
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

class DataAugmentationiBOT(object):
    def __init__(self, global_crops_scale, local_crops_scale, global_crops_number, local_crops_number,mode,bcsh):
        flip_and_color_jitter = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(
                [T.ColorJitter(brightness=bcsh[0], contrast=bcsh[1], saturation=bcsh[2], hue=bcsh[3])],
                p=0.8
            ),
            # T.RandomGrayscale(p=0.2),
        ]) 
        #normalize = T.Normalize(mean=[0.07126], std=[0.13697])
        if mode=="ct":
            normalize = T.Normalize(mean=[0.1218], std=[0.2063]) #CT
        else:
            normalize = T.Normalize(mean=[0.066], std=[0.3504]) #PET
        permute = Permute([1, 0, 2, 3]) #[D,C,H,W] --> [C,D,H,W]
        # normalize = T.Compose([
        #     T.ToTensor(),
        #     T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # ])

        self.global_crops_number = global_crops_number
        # transformation for the first global crop
        self.global_transfo1 = T.Compose([
            T.RandomResizedCrop(256, scale=global_crops_scale, interpolation=3), #128 224
            flip_and_color_jitter,
            #T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.)),
            permute,
            normalize,
        ])
        """
        # transformation for the rest of global crops
        self.global_transfo2 = T.Compose([
            T.RandomResizedCrop(256, scale=global_crops_scale, interpolation=3),
            flip_and_color_jitter,
            T.RandomApply(
                [T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.))],
                p=0.1
            ),
            T.RandomSolarize(threshold=0.5, p=0.2),
            permute,
            normalize,
        ])
        # transformation for the local crops
        self.local_crops_number = local_crops_number
        self.local_transfo = T.Compose([
            T.RandomResizedCrop(96, scale=local_crops_scale, interpolation=3),
            flip_and_color_jitter,
            T.RandomApply(
                [T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.))],
                p=0.5
            ),
            permute,
            normalize
        ])
        """
    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        """
        for _ in range(self.global_crops_number - 1):
            crops.append(self.global_transfo2(image))
            
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        """
        return crops


if __name__=="__main__":
    print("\n\nBegining Code!!\n\n")
    brightness=[0.1,0.2,0.4,0.7,0.8,1]
    contrast=[0.1,0.2,0.4,0.7,0.8,1]
    saturation=[0.1,0.2,0.4,0.7,0.8,1]
    hue=[0,0.1,0.2,0.3,0.5]
    dict_info_all=[]
    image_folder="../../../shared_data/NSCLC_Radiogenomics/images"
    csv_path="../../../shared_data/NSCLC_Radiogenomics/info_dataset.csv"
    
    for mode in ["ct","pet"]:
        for bb in brightness:
            for cc in contrast:
                for ss in saturation:
                    for hh in hue:
                        print(f"mode: {mode} , brightness: {bb} , contrast: {cc} , saturation: {ss} , hue: {hh}")
                        transform = DataAugmentationiBOT(
                            [0.4,1.0],
                            [0.05,0.4],
                            2,
                            10,
                            mode,
                            [bb,cc,ss,hh]
                        )
                        dataset=PETCTDataset3D(image_folder, csv_path, mode, transform=transform, batch_size=256)
                        name_all=[]
                        mean_all=[]
                        name_all=[]
                        median_all=[]
                        var_all=[]
                        std_all=[]
                        max_all=[]
                        min_all=[]
                        for ind in range(2):
                            for i in tqdm(range(len(dataset))):
                                data,name=dataset[i]
                                data=np.array(data)[0,0]
                                mean=np.mean(data)
                                median=np.median(data)
                                var=np.var(data)
                                std=np.std(data)
                                mx=np.max(data)
                                mn=np.min(data)
                            
                                name_all.append(name)
                                mean_all.append(mean)
                                median_all.append(median)
                                var_all.append(var)
                                std_all.append(std)
                                max_all.append(mx)
                                min_all.append(mn)
                            dict_info={}
                            dict_info["mode"]=mode
                            dict_info["bcsh"]=[bb,cc,ss,hh]
                            dict_info["results"]= {"names":name_all,"mean":mean_all,"median":median_all,"var":var_all,"std":std_all,"max":max_all,"min":min_all}
                            dict_info_all.append(dict_info)
    
                        np.save("mods_dataset.npy",dict_info_all)