# #/bin/bash
# export PYTHONPATH=.
# function terminate() {
#   exit
# }
# trap 'terminate' {1,2,3,15}
# accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
# pipeline/train/instruction_following.py \
# --pretrained_model_name_or_path="./weights/OTTER-Image-MPT7B" \
# --mimicit_ic_path="/data/yyama_dataset/tasks/VI_ICL/simple/train_instructions.json" \
# --images_ic_path="/data/yyama_dataset/tasks/VI_ICL/simple/train_images.json" \
# --train_config_ic_path="/data/yyama_dataset/tasks/VI_ICL/simple/train_pairs25_train.json" \
# --external_save_dir="./log" \
# --batch_size=64 \
# --gradient_accumulation_steps=2 \
# --logging_steps=300 \
# --num_epochs=11 \
# --run_name=prompt_comparison/simple_icl \
# --wandb_entity=ia-gu \
# --wandb_project=simple_icl \
# --workers=1 \
# --lr_scheduler=cosine \
# --learning_rate=1e-5 \
# --warmup_steps_ratio=0.01 \
# --report_to_wandb \

# #/bin/bash
# export PYTHONPATH=.
# function terminate() {
#   exit
# }
# trap 'terminate' {1,2,3,15}
# accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
# pipeline/train/instruction_following.py \
# --pretrained_model_name_or_path="./weights/OTTER-Image-MPT7B" \
# --mimicit_ic_path="/data/yyama_dataset/tasks/VI_ICL/various/train_instructions.json" \
# --images_ic_path="/data/yyama_dataset/tasks/VI_ICL/various/train_images.json" \
# --train_config_ic_path="/data/yyama_dataset/tasks/VI_ICL/various/train_pairs25_train.json" \
# --external_save_dir="./log" \
# --batch_size=64 \
# --gradient_accumulation_steps=2 \
# --logging_steps=300 \
# --num_epochs=26 \
# --run_name=prompt_comparison/various_icl \
# --wandb_entity=ia-gu \
# --wandb_project=various_icl \
# --workers=1 \
# --lr_scheduler=cosine \
# --learning_rate=1e-5 \
# --warmup_steps_ratio=0.01 \
# --report_to_wandb \

#/bin/bash
# export PYTHONPATH=.
# function terminate() {
#   exit
# }
# trap 'terminate' {1,2,3,15}
# accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
# pipeline/train/instruction_following.py \
# --pretrained_model_name_or_path="./weights/OTTER-Image-MPT7B" \
# --mimicit_ic_path="/data/yyama_dataset/tasks/VI_ICL/various_30/train_instructions.json" \
# --images_ic_path="/data/yyama_dataset/tasks/VI_ICL/various_30/train_images.json" \
# --train_config_ic_path="/data/yyama_dataset/tasks/VI_ICL/various_30/train_pairs25_train.json" \
# --external_save_dir="./log" \
# --batch_size=32 \
# --gradient_accumulation_steps=4 \
# --logging_steps=100 \
# --num_epochs=26 \
# --run_name=prompt_comparison/various_icl_more \
# --wandb_entity=ia-gu \
# --wandb_project=various_icl_more \
# --workers=1 \
# --lr_scheduler=cosine \
# --learning_rate=1e-5 \
# --warmup_steps_ratio=0.01 \
# --report_to_wandb \