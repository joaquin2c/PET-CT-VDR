# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 15:25:38 2024

@author: Mico
"""

import os
import pandas as pd
import numpy as np

if __name__ == "__main__":
    datasets = ['stanford_dataset']#'santa_maria_dataset', 
    #feature_dir = os.path.join('../../../Data/PET-CT', 'data', 'features_norm_pet_correct')
    feature_dir = os.path.join('../../../../shared_data/PET-CT', 'data', 'swin_3d_384mlp_s256_aug_1111_limit')
    output_path = os.path.join(feature_dir, 'petct.parquet')
    print("PATH: ",feature_dir)
    df = []
    for dataset in datasets:
        dataset_features_dir = os.path.join(feature_dir, dataset)
        if os.path.exists(dataset_features_dir):
            df_fns = os.listdir(dataset_features_dir)
            for df_fn in df_fns:
                df_path = os.path.join(dataset_features_dir, df_fn)
                df_aux = pd.read_parquet(df_path)
                df.append(df_aux)
    df = pd.concat(df)
    df['feature_id'] = df['feature_id'].astype(int)
    df['angle'] = df['angle'].astype(int)
    df['label'] = df['label'].astype(int)
    df['flip'] = df['flip'].astype(str)
    augmentation=np.logical_not(np.logical_and(df['flip'] == 'None', df['angle'] == 0))
    df['augmentation'] = augmentation
    df.reset_index(drop=True, inplace=True)
    print(df)
    print(np.unique(df["label"],return_counts=True))
    df.to_parquet(output_path)
