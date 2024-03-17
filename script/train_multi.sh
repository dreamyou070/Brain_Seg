# !/bin/bash
#
port_number=50678
category="medical"
obj_name="brain"
benchmark="BraTS2020_Segmentation_multisegment"
trigger_word='necrotic, ederma, tumor'
layer_name='layer_3'
sub_folder="up_16_32_64"
file_name="8_multi_segmentation_BraTS2020_only_anomal_use_multiclassification_focal_loss_do_class_weight"

accelerate launch --config_file ../../../gpu_config/gpu_0_1_2_3_4_5_config \
 --main_process_port $port_number ../train_multi.py --log_with wandb \
 --output_dir "../../result/${category}/${obj_name}/${layer_name}/${sub_folder}/${file_name}" \
 --train_unet --train_text_encoder --start_epoch 0 --max_train_epochs 100 \
 --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --data_path "/home/dreamyou070/MyData/anomaly_detection/medical/brain/BraTS2020_Segmentation_multisegment" \
 --trigger_word "${trigger_word}" \
 --obj_name "${obj_name}" \
 --do_map_loss \
 --trg_layer_list "['up_blocks_1_attentions_2_transformer_blocks_0_attn2',
                    'up_blocks_2_attentions_2_transformer_blocks_0_attn2',
                    'up_blocks_3_attentions_2_transformer_blocks_0_attn2',]" \
 --do_attn_loss --attn_loss_weight 1.0 --do_cls_train --normal_weight 1 \
 --resize_shape 512 --latent_res 64 --multiclassification_focal_loss --do_class_weight