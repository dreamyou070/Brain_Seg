# !/bin/bash

port_number=50033
category="medical"
obj_name="brain"
benchmark="BraTS2020_Segmentation_class_3"
trigger_word='brain'
layer_name='layer_3'
sub_folder="up_16_32_64"
file_name="10_class3_single_BraTS2020_text_truncate"

accelerate launch --config_file ../../../gpu_config/gpu_0_config \
 --main_process_port $port_number ../feature_extraction.py \
 --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --network_dim 64 --network_alpha 4 \
 --data_path "/home/dreamyou070/MyData/anomaly_detection/${category}/${obj_name}/${benchmark}/val" \
 --network_folder "../../result/${category}/${obj_name}/${layer_name}/${sub_folder}/${file_name}/models" \
 --obj_name "${obj_name}" \
 --prompt "${trigger_word}" \
 --latent_res 64 \
 --trg_layer_list "['up_blocks_1_attentions_2_transformer_blocks_0_attn2',
                    'up_blocks_2_attentions_2_transformer_blocks_0_attn2',
                    'up_blocks_3_attentions_2_transformer_blocks_0_attn2',]" \
 --threds [0.5] --use_position_embedder --text_truncate