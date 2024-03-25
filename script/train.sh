# !/bin/bash
#
port_number=58814
category="medical"
obj_name="leader_polyp"
benchmark="bkai-igh-neopolyp_sy"
layer_name='layer_3'
sub_folder="up_16_32_64"
file_name="2_segmentation_model_a_with_binary_instance_norm_relu_crossentropy_focal_loss"
# #--do_attn_loss
#
accelerate launch --config_file ../../../gpu_config/gpu_0_1_2_3_4_5_config \
 --main_process_port $port_number ../train.py --log_with wandb \
 --output_dir "../../result/${category}/${obj_name}/${layer_name}/${sub_folder}/${file_name}" \
 --train_data_path "/home/dreamyou070/MyData/anomaly_detection/medical/${obj_name}/${benchmark}/train" \
 --test_data_path "/home/dreamyou070/MyData/anomaly_detection/medical/${obj_name}/${benchmark}/test" \
 --latent_res 64 \
 --trigger_word "polyp" \
 --obj_name "${obj_name}" \
 --resize_shape 512 \
 --n_classes 3 \
 --mask_res 256 \
 --start_epoch 0 --max_train_epochs 200 \
 --use_position_embedder \
 --train_unet --train_text_encoder \
 --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --trg_layer_list "['up_blocks_1_attentions_2_transformer_blocks_0_attn2',
                    'up_blocks_2_attentions_2_transformer_blocks_0_attn2',
                    'up_blocks_3_attentions_2_transformer_blocks_0_attn2',]" \
 --norm_type "instance_norm" \
 --nonlinearity_type "leaky_relu" \
 --do_binary