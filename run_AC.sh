#/bin/bash
export PYTHONPATH=.
function terminate() {
  exit
}
trap 'terminate' {1,2,3,15}

accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
pipeline/train/instruction_following.py \
--pretrained_model_name_or_path="/home/data/MIMIC-IT/weights/OTTER-Image-MPT7B" \
--mimicit_path="/home/data/MIMIC-IT/AC/AC_train_instructions.json" \
--images_path="/home/data/MIMIC-IT/AC/AC_train.json" \
--train_config_path="/home/data/MIMIC-IT/AC/AC_train_train.json" \
--val_mimicit_path="/home/data/MIMIC-IT/AC/AC_val_instructions.json" \
--val_images_path="/home/data/MIMIC-IT/AC/AC_val.json" \
--val_config_path="/home/data/MIMIC-IT/AC/AC_val_train.json" \
--external_save_dir="./log/AC" \
--batch_size=128 \
--num_epochs=5 \
--wandb_entity=ia-gu \
--run_name=debug \
--wandb_project=debug \
--workers=1 \
--lr_scheduler=cosine \
--learning_rate=1e-5 \
--warmup_steps_ratio=0.01 \
--val_times_per_epoch=1 \
# --report_to_wandb \

# accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
# pipeline/train/instruction_following.py \
# --pretrained_model_name_or_path="/home/data/MIMIC-IT/weights/OTTER-Image-MPT7B" \
# --mimicit_path="/home/data/MIMIC-IT/AC/AC_train_instructions.json" \
# --images_path="/home/data/MIMIC-IT/AC/AC_train.json" \
# --train_config_path="/home/data/MIMIC-IT/AC/AC_train_train.json" \
# --val_mimicit_path="/home/data/MIMIC-IT/AC/AC_val_instructions.json" \
# --val_images_path="/home/data/MIMIC-IT/AC/AC_val.json" \
# --val_config_path="/home/data/MIMIC-IT/AC/AC_val_train.json" \
# --external_save_dir="./log/AC" \
# --batch_size=128 \
# --num_epochs=5 \
# --report_to_wandb \
# --wandb_entity=katlab_otter \
# --run_name=batch128_epoch5_lr-4 \
# --wandb_project=AC \
# --workers=1 \
# --lr_scheduler=cosine \
# --learning_rate=1e-4 \
# --warmup_steps_ratio=0.01 \

# accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
# pipeline/train/instruction_following.py \
# --pretrained_model_name_or_path="/home/data/MIMIC-IT/weights/OTTER-Image-MPT7B" \
# --mimicit_path="/home/data/MIMIC-IT/AC/AC_train_instructions.json" \
# --images_path="/home/data/MIMIC-IT/AC/AC_train.json" \
# --train_config_path="/home/data/MIMIC-IT/AC/AC_train_train.json" \
# --val_mimicit_path="/home/data/MIMIC-IT/AC/AC_val_instructions.json" \
# --val_images_path="/home/data/MIMIC-IT/AC/AC_val.json" \
# --val_config_path="/home/data/MIMIC-IT/AC/AC_val_train.json" \
# --external_save_dir="./log/AC" \
# --batch_size=128 \
# --num_epochs=5 \
# --report_to_wandb \
# --wandb_entity=katlab_otter \
# --run_name=batch128_epoch5_lr-3 \
# --wandb_project=AC \
# --workers=1 \
# --lr_scheduler=cosine \
# --learning_rate=1e-3 \
# --warmup_steps_ratio=0.01 \

# accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
# pipeline/train/instruction_following.py \
# --pretrained_model_name_or_path="/home/data/MIMIC-IT/weights/OTTER-Image-MPT7B" \
# --mimicit_path="/home/data/MIMIC-IT/AC/AC_train_instructions.json" \
# --images_path="/home/data/MIMIC-IT/AC/AC_train.json" \
# --train_config_path="/home/data/MIMIC-IT/AC/AC_train_train.json" \
# --val_mimicit_path="/home/data/MIMIC-IT/AC/AC_val_instructions.json" \
# --val_images_path="/home/data/MIMIC-IT/AC/AC_val.json" \
# --val_config_path="/home/data/MIMIC-IT/AC/AC_val_train.json" \
# --external_save_dir="./log/AC" \
# --batch_size=128 \
# --num_epochs=5 \
# --report_to_wandb \
# --wandb_entity=katlab_otter \
# --run_name=batch128_epoch5_lr-6 \
# --wandb_project=AC \
# --workers=1 \
# --lr_scheduler=cosine \
# --learning_rate=1e-6 \
# --warmup_steps_ratio=0.01 \

# accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
# pipeline/train/instruction_following.py \
# --pretrained_model_name_or_path="/home/data/MIMIC-IT/weights/OTTER-Image-MPT7B" \
# --mimicit_path="/home/data/MIMIC-IT/AC/AC_train_instructions.json" \
# --images_path="/home/data/MIMIC-IT/AC/AC_train.json" \
# --train_config_path="/home/data/MIMIC-IT/AC/AC_train_train.json" \
# --val_mimicit_path="/home/data/MIMIC-IT/AC/AC_val_instructions.json" \
# --val_images_path="/home/data/MIMIC-IT/AC/AC_val.json" \
# --val_config_path="/home/data/MIMIC-IT/AC/AC_val_train.json" \
# --external_save_dir="./log/AC" \
# --batch_size=128 \
# --num_epochs=5 \
# --report_to_wandb \
# --wandb_entity=katlab_otter \
# --run_name=batch128_epoch5_lr-5_warmup0 \
# --wandb_project=AC \
# --workers=1 \
# --lr_scheduler=cosine \
# --learning_rate=1e-5 \
# --warmup_steps_ratio=0 \

# accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
# pipeline/train/instruction_following.py \
# --pretrained_model_name_or_path="/home/data/MIMIC-IT/weights/OTTER-Image-MPT7B" \
# --mimicit_path="/home/data/MIMIC-IT/AC/AC_train_instructions.json" \
# --images_path="/home/data/MIMIC-IT/AC/AC_train.json" \
# --train_config_path="/home/data/MIMIC-IT/AC/AC_train_train.json" \
# --val_mimicit_path="/home/data/MIMIC-IT/AC/AC_val_instructions.json" \
# --val_images_path="/home/data/MIMIC-IT/AC/AC_val.json" \
# --val_config_path="/home/data/MIMIC-IT/AC/AC_val_train.json" \
# --external_save_dir="./log/AC" \
# --batch_size=128 \
# --num_epochs=5 \
# --report_to_wandb \
# --wandb_entity=katlab_otter \
# --run_name=batch128_epoch5_lr-5_warmup0.05 \
# --wandb_project=AC \
# --workers=1 \
# --lr_scheduler=cosine \
# --learning_rate=1e-5 \
# --warmup_steps_ratio=0.05 \

# accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
# pipeline/train/instruction_following.py \
# --pretrained_model_name_or_path="/home/data/MIMIC-IT/weights/OTTER-Image-MPT7B" \
# --mimicit_path="/home/data/MIMIC-IT/AC/AC_train_instructions.json" \
# --images_path="/home/data/MIMIC-IT/AC/AC_train.json" \
# --train_config_path="/home/data/MIMIC-IT/AC/AC_train_train.json" \
# --val_mimicit_path="/home/data/MIMIC-IT/AC/AC_val_instructions.json" \
# --val_images_path="/home/data/MIMIC-IT/AC/AC_val.json" \
# --val_config_path="/home/data/MIMIC-IT/AC/AC_val_train.json" \
# --external_save_dir="./log/AC" \
# --batch_size=128 \
# --num_epochs=5 \
# --report_to_wandb \
# --wandb_entity=katlab_otter \
# --run_name=batch128_epoch5_lr-5_warmup0.1 \
# --wandb_project=AC \
# --workers=1 \
# --lr_scheduler=cosine \
# --learning_rate=1e-5 \
# --warmup_steps_ratio=0.1 \