# !/bin/bash
#
port_number=50056
category="medical"
obj_name="brain"
benchmark="NFBS_SkullStripped"
trigger_word='brain'
layer_name='layer_3'
sub_folder="up_16_32_64"
file_name="8_unsupervised_skull_stripped_with_prepared_anomal"
# --unsupervised
# --anomal_source_path "../../../MyData/anomal_source_l_mode" \
# --anomal_position_source_path "../../../MyData/random_shape/${obj_name}"

accelerate launch --config_file ../../../gpu_config/gpu_0_1_2_3_4_5_config \
 --main_process_port $port_number ../train_brain.py --log_with wandb \
 --output_dir "/home/dreamyou070/SupervisedMED/result/${category}/${obj_name}/${layer_name}/${sub_folder}/${file_name}" \
 --train_unet --train_text_encoder --start_epoch 0 --max_train_epochs 60 \
 --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --data_path "/home/dreamyou070/MyData/anomaly_detection/${category}/${obj_name}/${benchmark}" \
 --trigger_word "${trigger_word}" \
 --obj_name "${obj_name}" \
 --do_map_loss \
 --trg_layer_list "['up_blocks_1_attentions_2_transformer_blocks_0_attn2',
                    'up_blocks_2_attentions_2_transformer_blocks_0_attn2',
                    'up_blocks_3_attentions_2_transformer_blocks_0_attn2',]" \
 --do_attn_loss --attn_loss_weight 1.0 --do_cls_train --normal_weight 1 \
 --do_self_aug \
 --do_normal_sample \
 --trigger_word "${trigger_word}" \
 --obj_name "${obj_name}"