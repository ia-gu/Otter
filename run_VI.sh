#/bin/bash
export PYTHONPATH=.
function terminate() {
  exit
}
trap 'terminate' {1,2,3,15}
accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
pipeline/train/instruction_following.py \
--pretrained_model_name_or_path="./weights/OTTER-Image-MPT7B" \
--mimicit_ic_path="/home/data/MIMIC-IT/VI/train_VI_long_instructions.json" \
--images_ic_path="/home/data/MIMIC-IT/VI/train_VI.json" \
--train_config_ic_path="/home/data/MIMIC-IT/VI/train_VI_pairs25_train.json" \
--val_mimicit_ic_path="/home/data/MIMIC-IT/VI/val_VI_long_instructions.json" \
--val_images_ic_path="/home/data/MIMIC-IT/VI/val_VI.json" \
--val_config_ic_path="/home/data/MIMIC-IT/VI/val_VI_pairs25_train.json" \
--external_save_dir="./log" \
--batch_size=16 \
--logging_steps=10 \
--num_epochs=5 \
--run_name=context_false \
--wandb_entity=ia-gu \
--wandb_project=debug \
--workers=1 \
--lr_scheduler=cosine \
--learning_rate=1e-5 \
--warmup_steps_ratio=0.01 \
--include_context_loss=False \
--val_times_per_epoch=10 \
--report_to_wandb \

# export PYTHONPATH=.
# function terminate() {
#   exit
# }
# trap 'terminate' {1,2,3,15}
# accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
# pipeline/train/instruction_following.py \
# --pretrained_model_name_or_path="./weights/OTTER-Image-MPT7B" \
# --mimicit_ic_path="/home/data/MIMIC-IT/VI/train_VI_long_instructions.json" \
# --images_ic_path="/home/data/MIMIC-IT/VI/train_VI.json" \
# --train_config_ic_path="/home/data/MIMIC-IT/VI/train_VI_pairs25_train.json" \
# --val_mimicit_ic_path="/home/data/MIMIC-IT/VI/val_VI_long_instructions.json" \
# --val_images_ic_path="/home/data/MIMIC-IT/VI/val_VI.json" \
# --val_config_ic_path="/home/data/MIMIC-IT/VI/val_VI_pairs25_train.json" \
# --external_save_dir="./log" \
# --batch_size=16 \
# --logging_steps=10 \
# --num_epochs=5 \
# --run_name=context_true \
# --wandb_entity=ia-gu \
# --wandb_project=debug \
# --workers=1 \
# --lr_scheduler=cosine \
# --learning_rate=1e-5 \
# --warmup_steps_ratio=0.01 \
# --include_context_loss=True \
# --val_times_per_epoch=10 \
# --report_to_wandb \