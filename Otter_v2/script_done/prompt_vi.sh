# #/bin/bash
export PYTHONPATH=.
function terminate() {
  exit
}
trap 'terminate' {1,2,3,15}
accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
pipeline/train/instruction_following.py \
--pretrained_model_name_or_path="./weights/OTTER-Image-MPT7B" \
--mimicit_ic_path="/data/yyama_dataset/tasks/VI_various_prompt/AC_train_instructions.json" \
--images_ic_path="/data/yyama_dataset/tasks/VI_various_prompt/AC_train.json" \
--train_config_ic_path="/data/yyama_dataset/tasks/VI_various_prompt/AC_train_train.json" \
--external_save_dir="./log" \
--batch_size=64 \
--logging_steps=10 \
--num_epochs=30 \
--run_name=prompt_comparison_30e/various_vi \
--wandb_entity=ia-gu \
--wandb_project=various_vi \
--workers=1 \
--lr_scheduler=cosine \
--learning_rate=1e-5 \
--warmup_steps_ratio=0.01 \
--report_to_wandb \

#/bin/bash
export PYTHONPATH=.
function terminate() {
  exit
}
trap 'terminate' {1,2,3,15}
accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
pipeline/train/instruction_following.py \
--pretrained_model_name_or_path="./weights/OTTER-Image-MPT7B" \
--mimicit_ic_path="/data/yyama_dataset/tasks/VI_simple_prompt/VI_train_instructions.json" \
--images_ic_path="/data/yyama_dataset/tasks/VI_simple_prompt/VI_train.json" \
--train_config_ic_path="/data/yyama_dataset/tasks/VI_simple_prompt/VI_train_train.json" \
--external_save_dir="./log" \
--batch_size=64 \
--logging_steps=10 \
--num_epochs=5 \
--run_name=prompt_comparison_30e/simple_vi \
--wandb_entity=ia-gu \
--wandb_project=simple_vi \
--workers=1 \
--lr_scheduler=cosine \
--learning_rate=1e-5 \
--warmup_steps_ratio=0.01 \
--report_to_wandb \