# !/bin/bash
#
port_number=58897
category="medical"
obj_name="leader_polyp"
benchmark="bkai-igh-neopolyp_sy"
layer_name='layer_3'
sub_folder="up_16_32_64"
file_name="12_segmentation_model_a_crossentropy_focal_loss_data_sy_layer_norm_nonlinearity_dice_loss"
#--use_position_embedder \
#--aggregation_model_b
accelerate launch --config_file ../../../gpu_config/gpu_0_1_2_3_4_5_config \
 --main_process_port $port_number ../train.py --log_with wandb \
 --output_dir "../../result/${category}/${obj_name}/${layer_name}/${sub_folder}/${file_name}" \
 --train_unet --train_text_encoder --start_epoch 0 --max_train_epochs 200 \
 --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --data_path "/home/dreamyou070/MyData/anomaly_detection/medical/${obj_name}/${benchmark}" \
 --resize_shape 512 \
 --latent_res 64 \
 --trigger_word "polyp" \
 --obj_name "${obj_name}" \
 --trg_layer_list "['up_blocks_1_attentions_2_transformer_blocks_0_attn2',
                    'up_blocks_2_attentions_2_transformer_blocks_0_attn2',
                    'up_blocks_3_attentions_2_transformer_blocks_0_attn2',]" \
 --n_classes 3 \
 --mask_res 256 --cross_entropy_focal_loss_both \
 --use_nonlinearity \
 --do_dice_loss