#/bin/bash
export PYTHONPATH=.
function terminate() {
  exit
}
trap 'terminate' {1,2,3,15}

accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
pipeline/train/instruction_following.py \
--pretrained_model_name_or_path="/home/data/MIMIC-IT/weights/OTTER-Image-MPT7B" \
--mimicit_ic_path="/home/data/MIMIC-IT/MI_for_debug/MI_train_instructions.json" \
--images_ic_path="/home/data/MIMIC-IT/MiniImagenet_jsons/MI_train.json" \
--train_config_ic_path="/home/data/MIMIC-IT/MiniImagenet_jsons/MI_train_pairs4_train.json" \
--external_save_dir="./log/test" \
--batch_size=16 \
--num_epochs=1 \
--report_to_wandb \
--wandb_entity=ia-gu \
--run_name=debug \
--wandb_project=debug \
--workers=1 \
--lr_scheduler=cosine \
--learning_rate=1e-5 \
--warmup_steps_ratio=0.01 \