#/bin/bash
export PYTHONPATH=.
function terminate() {
  exit
}
trap 'terminate' {1,2,3,15}

accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
pipeline/train/instruction_following.py \
--pretrained_model_name_or_path="./weights/OTTER-Image-MPT7B" \
--mimicit_path="./data/SD/SD_instructions.json" \
--images_path="./data/SD/SD.json" \
--train_config_path="./data/SD/SD_train.json" \
--batch_size=4 \
--num_epochs=9 \
--report_to_wandb \
--wandb_entity=katlab_otter \
--run_name=OTTER-Image-MPT7B-SD \
--wandb_project=OTTER-Image-MPT7B \
--workers=1 \
--lr_scheduler=cosine \
--learning_rate=1e-5 \
--warmup_steps_ratio=0.01 \