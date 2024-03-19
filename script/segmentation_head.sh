# !/bin/bash
#
port_number=50644
category="medical"
obj_name="brain"
benchmark="BraTS2020_Segmentation_128"
layer_name='layer_3'
sub_folder="up_16_32_64"
file_name="21_segmentation_model_128"

accelerate launch --config_file ../../../gpu_config/gpu_0_1_2_3_4_5_config \
 --main_process_port $port_number ../segmentation_head.py --log_with wandb \
 --output_dir "../../result/${category}/${obj_name}/${layer_name}/${sub_folder}/${file_name}" \
 --network_weights "../../result/${category}/${obj_name}/${layer_name}/${sub_folder}/2_binary_segmentation_BraTS2020/models/epoch-000033.safetensors" \
 --position_embedder_weights "../../result/${category}/${obj_name}/${layer_name}/${sub_folder}/2_binary_segmentation_BraTS2020/position_embedder/position_embedder_33.safetensors" \
 --train_unet --train_text_encoder --start_epoch 0 --max_train_epochs 100 \
 --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --data_path "/home/dreamyou070/MyData/anomaly_detection/medical/brain/${benchmark}" \
 --trigger_word "brain" \
 --obj_name "${obj_name}" \
 --do_map_loss \
 --trg_layer_list "['up_blocks_1_attentions_2_transformer_blocks_0_attn2',
                    'up_blocks_2_attentions_2_transformer_blocks_0_attn2',
                    'up_blocks_3_attentions_2_transformer_blocks_0_attn2',]" \
 --do_attn_loss --do_cls_train \
 --resize_shape 512 --latent_res 64 --multiclassification_focal_loss --multiclassification_focal_loss \
 --train_segmentation \
 --use_position_embedder \
 --n_classes 4 \
 --pretrained_segmentation_model "../../result/${category}/${obj_name}/${layer_name}/${sub_folder}/${file_name}/segmentation_seg_based_lora/segmentation_87.safetensors"