# !/bin/bash


port_number=50056
category="medical"
obj_name="brain"
benchmark="NFBS_preprocess"
trigger_word='brain'
layer_name='layer_3'
sub_folder="up_16_32_64"
file_name="7_unsupervised_skull_stripped_with_normal_sample_anomal_position"
#--do_self_aug

python ../data_check.py --log_with wandb \
 --output_dir "../../result/${category}/${obj_name}/${layer_name}/${sub_folder}/${file_name}/data_check_anomal_kth_trg_beta_0.4_blur_2" \
 --train_unet --train_text_encoder --start_epoch 0 --max_train_epochs 30 \
 --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --data_path "../../../MyData/anomaly_detection/${category}/${obj_name}/${benchmark}" \
 --anomal_position_source_path "../../../MyData/random_shape/braTS2020" \
 --anomal_source_path "../../../MyData/kth" \
 --trigger_word "${trigger_word}" \
 --obj_name "${obj_name}" \
 --trg_beta 0.4 \
 --unsupervised