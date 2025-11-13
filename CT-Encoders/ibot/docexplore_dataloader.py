import os
import pandas as pd

from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image, ImageOps


class DocExplore(Dataset):
    def __init__(self, path, max_size=224, transform=None):
        try:
            self.df = pd.read_pickle(path)
        except Exception as e:
            print(f"No se pudo cargar el archivo pickle: {e}")
            raise e 
        self.max_size = max_size
        self.transform = transform

        # Cambiamos el path de las imágenes
        self.change_path()
        print(len(self.df))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        df_idx = self.df.iloc[idx]

        crop_anchor = self.crop_image(df_idx)
        crop_anchor_tensor = self.transform_image(crop_anchor)

        transform_dino = self.transforms_dino() 
        crop_positive_tensor = transform_dino(crop_anchor)

        return crop_anchor_tensor, crop_positive_tensor 

    def crop_image(self, df_idx):
        path_image = df_idx['filename'].iloc[0] if isinstance(df_idx['filename'], pd.Series) else df_idx['filename']

        if not os.path.exists(path_image):
            raise FileNotFoundError(f"No se encontró el archivo de imagen: {path_image}")

        img = Image.open(path_image).convert('RGB')
        x1, y1, x2, y2 = map(int, [df_idx['x1'], df_idx['y1'], df_idx['x2'], df_idx['y2']])
        return img.crop((x1, y1, x2, y2))

    def transform_image(self, image):
        padded_image = ImageOps.pad(image, size=(self.opts.max_size, self.opts.max_size))
        return self.transform(padded_image)

    def change_path(self, new_path='/home/data/cstears/DocExplore_images/'):
        self.df['filename'] = self.df['filename'].apply(lambda x: x.replace('/home/cloyola/datasets/DocExplore/DocExplore_images/', new_path))

    @staticmethod
    def data_transform(opts):
        dataset_transforms = transforms.Compose([
            transforms.Resize((opts.max_size, opts.max_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return dataset_transforms

    def transforms_dino(self):
        return transforms.Compose([
            transforms.Resize((self.opts.max_size, self.opts.max_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomResizedCrop(self.opts.max_size, scale=(0.7, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])