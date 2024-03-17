# !/bin/bash
#
port_number=50072
category="medical"
obj_name="brain"
benchmark="BraTS2020_Segmentation_class_2"
trigger_word='brain'
layer_name='layer_3'
sub_folder="up_16_32_64"
file_name="10_class2_single_BraTS2020_text_truncate"

accelerate launch --config_file ../../../gpu_config/gpu_0_1_config \
 --main_process_port $port_number ../train_single.py --log_with wandb \
 --output_dir "../../result/${category}/${obj_name}/${layer_name}/${sub_folder}/${file_name}" \
 --network_weights "../../result/${category}/${obj_name}/${layer_name}/${sub_folder}/${file_name}/models/epoch-000004.safetensors" \
 --position_embedder_weights "../../result/${category}/${obj_name}/${layer_name}/${sub_folder}/${file_name}/position_embedder_4.safetensors" \
 --train_unet --train_text_encoder --start_epoch 4 --max_train_epochs 96 \
 --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --data_path "/home/dreamyou070/MyData/anomaly_detection/medical/brain/${benchmark}" \
 --trigger_word "${trigger_word}" \
 --obj_name "${obj_name}" \
 --do_map_loss \
 --trg_layer_list "['up_blocks_1_attentions_2_transformer_blocks_0_attn2',
                    'up_blocks_2_attentions_2_transformer_blocks_0_attn2',
                    'up_blocks_3_attentions_2_transformer_blocks_0_attn2',]" \
 --do_attn_loss --attn_loss_weight 1.0 --do_cls_train --normal_weight 1 --text_truncate --train_single