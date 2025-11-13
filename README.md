Instalar librerias necesarias
```
conda create -n petct python=3.10
conda activate petct
pip install -r requirements.txt
```

Entrenar encoder en ibot
```
python main_ibot.py --arch swin3D --batch_size_per_gpu 4 --epochs 400 --data_path ../../../../data/Mediastinal-Lymph-Node-SEG/Mediastinal-Lymph-Node-SEG-DA-RAD/Mediastinal-Lymph-Node-SEG --csv_path ../MLNS.csv --mode ct --saveckp_freq 40 --clip_grad 3.0 --global_crops_scale 0.4 1.0 --local_crops_scale 0.05 0.4 --lr 0.0005 --min_lr 1e-06 --norm_last_layer False --patch_size 8 --pred_ratio 0.0 0.3 --pred_ratio_var 0.0 0.2 --pred_shape block --pred_start_epoch 50 --warmup_teacher_temp_epochs 30 --window_size 7 --type_dataset dcm --output_dir ../outputs/output_test --gpu 0
```
Entrenar modelo:

```
python tfds_dense_descriptor_3D.py --model_path ../../CT-Encoders/output_MLNS_ps4_V2/checkpoint.pth -f ../../../../data/PET-CT/data/test_git -d ../../../../data/NSCLC_Radiogenomics/images -df ../../../../data/NSCLC_Radiogenomics/info_dataset.csv --modality ct -gpu 0
```
