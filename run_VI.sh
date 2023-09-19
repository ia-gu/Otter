#/bin/bash
export PYTHONPATH=.
function terminate() {
  exit
}
trap 'terminate' {1,2,3,15}

# accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
# pipeline/train/instruction_following.py \
# --pretrained_model_name_or_path="./weights/OTTER-Image-MPT7B" \
# --mimicit_ic_path="/home/data/MIMIC-IT/VI/train_VI_long_instructions.json" \
# --images_ic_path="/home/data/MIMIC-IT/VI/train_VI.json" \
# --train_config_ic_path="/home/data/MIMIC-IT/VI/train_VI_pairs25_train.json" \
# --external_save_dir="./log" \
# --batch_size=16 \
# --num_epochs=1 \
# --report_to_wandb \
# --wandb_entity=katlab_otter \
# --run_name=VI_batch16_long_pairs25 \
# --wandb_project=OTTER-Image-MPT7B \
# --workers=1 \
# --lr_scheduler=cosine \
# --learning_rate=1e-5 \
# --warmup_steps_ratio=0.01 \

# accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
# pipeline/train/instruction_following.py \
# --pretrained_model_name_or_path="./weights/OTTER-Image-MPT7B" \
# --mimicit_ic_path="/home/data/MIMIC-IT/VI/train_VI_short_instructions.json" \
# --images_ic_path="/home/data/MIMIC-IT/VI/train_VI.json" \
# --train_config_ic_path="/home/data/MIMIC-IT/VI/train_VI_pairs25_train.json" \
# --external_save_dir="./log" \
# --batch_size=16 \
# --num_epochs=1 \
# --report_to_wandb \
# --wandb_entity=katlab_otter \
# --run_name=VI_batch16_short_pairs25 \
# --wandb_project=OTTER-Image-MPT7B \
# --workers=1 \
# --lr_scheduler=cosine \
# --learning_rate=1e-5 \
# --warmup_steps_ratio=0.01 \

accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
pipeline/train/instruction_following.py \
--pretrained_model_name_or_path="./weights/OTTER-Image-MPT7B" \
--mimicit_ic_path="/home/data/MIMIC-IT/VI/train_VI_long_instructions.json" \
--images_ic_path="/home/data/MIMIC-IT/VI/train_VI.json" \
--train_config_ic_path="/home/data/MIMIC-IT/VI/train_VI_pairs5_train.json" \
--external_save_dir="./log" \
--batch_size=16 \
--num_epochs=1 \
--report_to_wandb \
--wandb_entity=katlab_otter \
--run_name=VI_batch16_long_pairs5 \
--wandb_project=OTTER-Image-MPT7B \
--workers=1 \
--lr_scheduler=cosine \
--learning_rate=1e-5 \
--warmup_steps_ratio=0.01 \

accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
pipeline/train/instruction_following.py \
--pretrained_model_name_or_path="./weights/OTTER-Image-MPT7B" \
--mimicit_ic_path="/home/data/MIMIC-IT/VI/train_VI_short_instructions.json" \
--images_ic_path="/home/data/MIMIC-IT/VI/train_VI.json" \
--train_config_ic_path="/home/data/MIMIC-IT/VI/train_VI_pairs5_train.json" \
--external_save_dir="./log" \
--batch_size=16 \
--num_epochs=1 \
--report_to_wandb \
--wandb_entity=katlab_otter \
--run_name=VI_batch16_short_pairs5 \
--wandb_project=OTTER-Image-MPT7B \
--workers=1 \
--lr_scheduler=cosine \
--learning_rate=1e-5 \
--warmup_steps_ratio=0.01 \