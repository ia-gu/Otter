#/bin/bash
export PYTHONPATH=.
function terminate() {
  exit
}
trap 'terminate' {1,2,3,15}
accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
pipeline/train/instruction_following.py \
--pretrained_model_name_or_path="./weights/OTTER-Image-MPT7B" \
--mimicit_ic_path="/data/yyama_dataset/tasks/VI_ICL/various_20fix_rand/train_instructions.json" \
--images_ic_path="/data/yyama_dataset/tasks/VI_ICL/various_20fix_rand/train_images.json" \
--train_config_ic_path="/data/yyama_dataset/tasks/VI_ICL/various_20fix_rand/train_pairs25_train.json" \
--external_save_dir="./log" \
--batch_size=32 \
--gradient_accumulation_steps=4 \
--logging_steps=300 \
--num_epochs=1 \
--run_name=prompt_comparison/various_20fix_rand \
--wandb_entity=ia-gu \
--wandb_project=various_20fix_rand \
--workers=1 \
--lr_scheduler=cosine \
--learning_rate=1e-5 \
--warmup_steps_ratio=0.01 \
--report_to_wandb \

# #/bin/bash
export PYTHONPATH=.
function terminate() {
  exit
}
trap 'terminate' {1,2,3,15}
accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
pipeline/train/instruction_following.py \
--pretrained_model_name_or_path="./weights/OTTER-Image-MPT7B" \
--mimicit_ic_path="/data/yyama_dataset/tasks/VI_ICL/various_20fix/train_instructions.json" \
--images_ic_path="/data/yyama_dataset/tasks/VI_ICL/various_20fix/train_images.json" \
--train_config_ic_path="/data/yyama_dataset/tasks/VI_ICL/various_20fix/train_pairs25_train.json" \
--external_save_dir="./log" \
--batch_size=32 \
--gradient_accumulation_steps=4 \
--logging_steps=300 \
--num_epochs=1 \
--run_name=prompt_comparison/various_20fix \
--wandb_entity=ia-gu \
--wandb_project=various_20fix \
--workers=1 \
--lr_scheduler=cosine \
--learning_rate=1e-5 \
--warmup_steps_ratio=0.01 \
--report_to_wandb \

cd pipeline/demo
python eval_fix.py