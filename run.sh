#/bin/bash
function terminate() {
  exit
}
trap 'terminate' {1,2,3,15}

accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
pipeline/train/instruction_following.py \
--pretrained_model_name_or_path=luodian/OTTER-MPT7B-Init \
--mimicit_path="./data/DC/DC_instruction.json" \
--images_path="./data/DC/DC.json" \
--train_config_path="./data/DC/DC_train.json" \
--batch_size=4 \
--num_epochs=9 \
--report_to_wandb \
--wandb_entity=ntu-slab \
--run_name=OTTER-LLaMA7B-densecaption \
--wandb_project=OTTER-LLaMA7B \
--workers=1 \
--lr_scheduler=cosine \
--learning_rate=1e-5 \
--warmup_steps_ratio=0.01 \