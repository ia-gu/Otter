#/bin/bash
export PYTHONPATH=.
function terminate() {
  exit
}
trap 'terminate' {1,2,3,15}

# accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
# pipeline/train/instruction_following.py \
# --pretrained_model_name_or_path="./weights/OTTER-Image-MPT7B" \
# --mimicit_ic_path="/data/dataset/yyama_dataset/tasks/PC2AC/PC_train_instructions.json" \
# --images_ic_path="/data/dataset/yyama_dataset/tasks/PC2AC/PC_train.json" \
# --train_config_ic_path="/data/dataset/yyama_dataset/tasks/PC2AC/PC_train_train.json" \
# --external_save_dir="./log/PC2AC" \
# --batch_size=128 \
# --num_epochs=1 \
# --report_to_wandb \
# --wandb_entity=ia-gu \
# --run_name=PC \
# --wandb_project=PC \
# --workers=1 \
# --lr_scheduler=cosine \
# --learning_rate=1e-5 \
# --warmup_steps_ratio=0.01 \

accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
pipeline/train/instruction_following.py \
--pretrained_model_name_or_path="./weights/OTTER-Image-MPT7B" \
--mimicit_ic_path="/data/dataset/yyama_dataset/tasks/PC2AC/AC_train_instructions.json" \
--images_ic_path="/data/dataset/yyama_dataset/tasks/PC2AC/AC_train.json" \
--train_config_ic_path="/data/dataset/yyama_dataset/tasks/PC2AC/AC_train_train.json" \
--external_save_dir="./log/PC2AC" \
--batch_size=128 \
--num_epochs=1 \
--report_to_wandb \
--wandb_entity=ia-gu \
--run_name=AC \
--wandb_project=AC \
--workers=1 \
--lr_scheduler=cosine \
--learning_rate=1e-5 \
--warmup_steps_ratio=0.01 \
--trained_ckpt="./log/PC2AC/PC/final_weights.pt"

accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
pipeline/train/instruction_following.py \
--pretrained_model_name_or_path="./weights/OTTER-Image-MPT7B" \
--mimicit_ic_path="/data/dataset/yyama_dataset/tasks/PC2AC/train_instructions.json" \
--images_ic_path="/data/dataset/yyama_dataset/tasks/PC2AC/train_images.json" \
--train_config_ic_path="/data/dataset/yyama_dataset/tasks/PC2AC/train_pairs25_train.json" \
--external_save_dir="./log/PC2AC" \
--batch_size=128 \
--num_epochs=1 \
--report_to_wandb \
--wandb_entity=ia-gu \
--run_name=PC2AC \
--wandb_project=VI \
--workers=1 \
--lr_scheduler=cosine \
--learning_rate=1e-5 \
--warmup_steps_ratio=0.01 \
--trained_ckpt="./log/PC2AC/AC/final_weights.pt"

cd pipline/demo
python evaluate_VI_MVTec.py
# 2 Otter fine tune AC �� VI loss=context+query (product)
# accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
# pipeline/train/instruction_following.py \
# --pretrained_model_name_or_path="/home/data/MIMIC-IT/weights/OTTER-Image-MPT7B" \
# --trained_ckpt="./log/AC_full_otter/batch128_epoch10_lr-5_yesno/final_weights.pt" \
# --mimicit_path="./data/VI_full_jsons/VI_train_product_instructions.json" \
# --images_path="/home/data/MIMIC-IT/VI_full_jsons/VI_train.json" \
# --train_config_path="/home/data/MIMIC-IT/VI_full_jsons/VI_train_pairs25_train.json" \
# --val_mimicit_path="./data/VI_full_jsons/VI_val_product_instructions.json" \
# --val_images_path="/home/data/MIMIC-IT/VI_full_jsons/VI_val.json" \
# --val_config_path="/home/data/MIMIC-IT/VI_full_jsons/VI_val_pairs1_train.json" \
# --external_save_dir="./log/AC_VI_full_otter_product" \
# --batch_size=128 \
# --num_epochs=1 \
# --include_context_loss="True" \
# --ratio_extra_token_loss=0.2 \
# --report_to_wandb \
# --wandb_entity=katlab_otter \
# --run_name=batch128_epoch1_lr-5_true \
# --wandb_project=AC_VI_full_otter_product \
# --workers=1 \
# --lr_scheduler=cosine \
# --learning_rate=1e-5 \
# --warmup_steps_ratio=0.01 \

# 1 Otter fine tune AC yesno
# accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
# pipeline/train/instruction_following.py \
# --pretrained_model_name_or_path="/home/data/MIMIC-IT/weights/OTTER-Image-MPT7B" \
# --mimicit_path="/home/data/MIMIC-IT/AC_full_jsons/AC_train_yesno_instructions.json" \
# --images_path="/home/data/MIMIC-IT/AC_full_jsons/AC_train.json" \
# --train_config_path="/home/data/MIMIC-IT/AC_full_jsons/AC_train_train.json" \
# --val_mimicit_path="/home/data/MIMIC-IT/AC_full_jsons/AC_val_yesno_instructions.json" \
# --val_images_path="/home/data/MIMIC-IT/AC_full_jsons/AC_val.json" \
# --val_config_path="/home/data/MIMIC-IT/AC_full_jsons/AC_val_train.json" \
# --external_save_dir="./log/AC_full_otter" \
# --batch_size=128 \
# --num_epochs=10 \
# --include_context_loss="False" \
# --ratio_extra_token_loss=0.2 \
# --report_to_wandb \
# --wandb_entity=katlab_otter \
# --run_name=batch128_epoch10_lr-5_yesno \
# --wandb_project=AC_full_otter \
# --workers=1 \
# --lr_scheduler=cosine \
# --learning_rate=1e-5 \
# --warmup_steps_ratio=0.01 \

# 2 Otter fine tune VI loss=context+query
# accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
# pipeline/train/instruction_following.py \
# --pretrained_model_name_or_path="/home/data/MIMIC-IT/weights/OTTER-Image-MPT7B" \
# --mimicit_path="/home/data/MIMIC-IT/VI_full_jsons/VI_train_instructions.json" \
# --images_path="/home/data/MIMIC-IT/VI_full_jsons/VI_train.json" \
# --train_config_path="/home/data/MIMIC-IT/VI_full_jsons/VI_train_pairs25_train.json" \
# --val_mimicit_path="/home/data/MIMIC-IT/VI_full_jsons/VI_val_instructions.json" \
# --val_images_path="/home/data/MIMIC-IT/VI_full_jsons/VI_val.json" \
# --val_config_path="/home/data/MIMIC-IT/VI_full_jsons/VI_val_pairs1_train.json" \
# --external_save_dir="./log/VI_full_otter" \
# --batch_size=128 \
# --num_epochs=1 \
# --include_context_loss="True" \
# --ratio_extra_token_loss=0.2 \
# --report_to_wandb \
# --wandb_entity=katlab_otter \
# --run_name=batch128_epoch1_lr-5_true \
# --wandb_project=VI_full_otter \
# --workers=1 \
# --lr_scheduler=cosine \
# --learning_rate=1e-5 \
# --warmup_steps_ratio=0.01 \

# 3 Otter fine tune loss=query
# accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
# pipeline/train/instruction_following.py \
# --pretrained_model_name_or_path="/home/data/MIMIC-IT/weights/OTTER-Image-MPT7B" \
# --mimicit_path="/home/data/MIMIC-IT/VI_full_jsons/VI_train_instructions.json" \
# --images_path="/home/data/MIMIC-IT/VI_full_jsons/VI_train.json" \
# --train_config_path="/home/data/MIMIC-IT/VI_full_jsons/VI_train_pairs25_train.json" \
# --val_mimicit_path="/home/data/MIMIC-IT/VI_full_jsons/VI_val_instructions.json" \
# --val_images_path="/home/data/MIMIC-IT/VI_full_jsons/VI_val.json" \
# --val_config_path="/home/data/MIMIC-IT/VI_full_jsons/VI_val_pairs1_train.json" \
# --external_save_dir="./log/VI_full_otter" \
# --batch_size=128 \
# --num_epochs=1 \
# --include_context_loss="False" \
# --ratio_extra_token_loss=0.2 \
# --report_to_wandb \
# --wandb_entity=katlab_otter \
# --run_name=batch128_epoch1_lr-5_false \
# --wandb_project=VI_full_otter \
# --workers=1 \
# --lr_scheduler=cosine \
# --learning_rate=1e-5 \
# --warmup_steps_ratio=0.01 \

# 4 Otter fine tune AC-->VI loss=context+query
# accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
# pipeline/train/instruction_following.py \
# --pretrained_model_name_or_path="/home/data/MIMIC-IT/weights/OTTER-Image-MPT7B" \
# --trained_ckpt="./log/AC_full_otter/batch128_epoch10_lr-5_yesno/final_weights.pt" \
# --mimicit_path="/home/data/MIMIC-IT/VI_full_jsons/VI_train_instructions.json" \
# --images_path="/home/data/MIMIC-IT/VI_full_jsons/VI_train.json" \
# --train_config_path="/home/data/MIMIC-IT/VI_full_jsons/VI_train_pairs25_train.json" \
# --val_mimicit_path="/home/data/MIMIC-IT/VI_full_jsons/VI_val_instructions.json" \
# --val_images_path="/home/data/MIMIC-IT/VI_full_jsons/VI_val.json" \
# --val_config_path="/home/data/MIMIC-IT/VI_full_jsons/VI_val_pairs1_train.json" \
# --external_save_dir="./log/AC_VI_full_otter" \
# --batch_size=128 \
# --num_epochs=1 \
# --include_context_loss="True" \
# --ratio_extra_token_loss=0.2 \
# --report_to_wandb \
# --wandb_entity=katlab_otter \
# --run_name=batch128_epoch1_lr-5_true \
# --wandb_project=AC_VI_full_otter \
# --workers=1 \
# --lr_scheduler=cosine \
# --learning_rate=1e-5 \
# --warmup_steps_ratio=0.01 \

# 5 Otter fine tune AC-->VI loss=query
# ��������
# accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
# pipeline/train/instruction_following.py \
# --pretrained_model_name_or_path="/home/data/MIMIC-IT/weights/OTTER-Image-MPT7B" \
# --trained_ckpt="./log/AC_full_otter/batch128_epoch10_lr-5_yesno/final_weights.pt" \
# --mimicit_path="/home/data/MIMIC-IT/VI_full_jsons/VI_train_instructions.json" \
# --images_path="/home/data/MIMIC-IT/VI_full_jsons/VI_train.json" \
# --train_config_path="/home/data/MIMIC-IT/VI_full_jsons/VI_train_pairs25_train.json" \
# --val_mimicit_path="/home/data/MIMIC-IT/VI_full_jsons/VI_val_instructions.json" \
# --val_images_path="/home/data/MIMIC-IT/VI_full_jsons/VI_val.json" \
# --val_config_path="/home/data/MIMIC-IT/VI_full_jsons/VI_val_pairs1_train.json" \
# --external_save_dir="./log/AC_VI_full_otter" \
# --batch_size=128 \
# --num_epochs=1 \
# --include_context_loss="False" \
# --ratio_extra_token_loss=0.2 \
# --report_to_wandb \
# --wandb_entity=katlab_otter \
# --run_name=batch128_epoch1_lr-5_false \
# --wandb_project=AC_VI_full_otter \
# --workers=1 \
# --lr_scheduler=cosine \
# --learning_rate=1e-5 \
# --warmup_steps_ratio=0.01 \

# 6 Otter fine tune AC yesno
# accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
# pipeline/train/instruction_following.py \
# --pretrained_model_name_or_path="/home/data/MIMIC-IT/weights/OTTER-Image-MPT7B" \
# --mimicit_path="/home/data/MIMIC-IT/AC_full_jsons/AC_train_yesno_instructions.json" \
# --images_path="/home/data/MIMIC-IT/AC_full_jsons/AC_train.json" \
# --train_config_path="/home/data/MIMIC-IT/AC_full_jsons/AC_train_train.json" \
# --val_mimicit_path="/home/data/MIMIC-IT/AC_full_jsons/AC_val_yesno_instructions.json" \
# --val_images_path="/home/data/MIMIC-IT/AC_full_jsons/AC_val.json" \
# --val_config_path="/home/data/MIMIC-IT/AC_full_jsons/AC_val_train.json" \
# --external_save_dir="./log/AC_full_otter" \
# --batch_size=4 \
# --num_epochs=1 \
# --include_context_loss="False" \
# --ratio_extra_token_loss=0.2 \
# --report_to_wandb \
# --wandb_entity=katlab_otter \
# --run_name=batch4_epoch1_lr-4_yesno \
# --wandb_project=AC_full_otter \
# --workers=1 \
# --lr_scheduler=cosine \
# --learning_rate=1e-4 \
# --warmup_steps_ratio=0.01 \

# 7 Otter fine tune loss=query
# accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
# pipeline/train/instruction_following.py \
# --pretrained_model_name_or_path="/home/data/MIMIC-IT/weights/OTTER-Image-MPT7B" \
# --mimicit_path="/home/data/MIMIC-IT/VI_full_jsons/VI_train_instructions.json" \
# --images_path="/home/data/MIMIC-IT/VI_full_jsons/VI_train.json" \
# --train_config_path="/home/data/MIMIC-IT/VI_full_jsons/VI_train_pairs25_train.json" \
# --val_mimicit_path="/home/data/MIMIC-IT/VI_full_jsons/VI_val_instructions.json" \
# --val_images_path="/home/data/MIMIC-IT/VI_full_jsons/VI_val.json" \
# --val_config_path="/home/data/MIMIC-IT/VI_full_jsons/VI_val_pairs1_train.json" \
# --external_save_dir="./log/VI_full_otter" \
# --batch_size=4 \
# --num_epochs=1 \
# --include_context_loss="False" \
# --ratio_extra_token_loss=0.2 \
# --report_to_wandb \
# --wandb_entity=katlab_otter \
# --run_name=batch4_epoch1_lr-4_false \
# --wandb_project=VI_full_otter \
# --workers=1 \
# --lr_scheduler=cosine \
# --learning_rate=1e-4 \
# --warmup_steps_ratio=0.01 \

# 8 Otter fine tune AC-->VI loss=query
# accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
# pipeline/train/instruction_following.py \
# --pretrained_model_name_or_path="/home/data/MIMIC-IT/weights/OTTER-Image-MPT7B" \
# --trained_ckpt="./log/AC_full_otter/batch128_epoch10_lr-5_yesno/final_weights.pt" \
# --mimicit_path="/home/data/MIMIC-IT/VI_full_jsons/VI_train_instructions.json" \
# --images_path="/home/data/MIMIC-IT/VI_full_jsons/VI_train.json" \
# --train_config_path="/home/data/MIMIC-IT/VI_full_jsons/VI_train_pairs25_train.json" \
# --val_mimicit_path="/home/data/MIMIC-IT/VI_full_jsons/VI_val_instructions.json" \
# --val_images_path="/home/data/MIMIC-IT/VI_full_jsons/VI_val.json" \
# --val_config_path="/home/data/MIMIC-IT/VI_full_jsons/VI_val_pairs1_train.json" \
# --external_save_dir="./log/AC_VI_full_otter" \
# --batch_size=4 \
# --num_epochs=1 \
# --include_context_loss="False" \
# --ratio_extra_token_loss=0.2 \
# --report_to_wandb \
# --wandb_entity=katlab_otter \
# --run_name=batch4_epoch1_lr-4_false \
# --wandb_project=AC_VI_full_otter \
# --workers=1 \
# --lr_scheduler=cosine \
# --learning_rate=1e-4 \
# --warmup_steps_ratio=0.01 \