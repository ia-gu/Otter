#/bin/bash
export PYTHONPATH=.
function terminate() {
  exit
}
trap 'terminate' {1,2,3,15}
accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
pipeline/train/instruction_following.py \
--pretrained_model_name_or_path="./weights/OTTER-Image-MPT7B" \
--mimicit_ic_path="/data/yyama_dataset/rename_full_rearranged/train_instructions_json_based.json" \
--images_ic_path="/data/yyama_dataset/rename_full_rearranged/train_images_json_based.json" \
--train_config_ic_path="/data/yyama_dataset/rename_full_rearranged/train_pairs25_train_json_based.json" \
--external_save_dir="./log" \
--batch_size=16 \
--gradient_accumulation_steps=8 \
--logging_steps=300 \
--num_epochs=26 \
--run_name=rename_full/json_based_with_locaation_20_rand \
--wandb_entity=ia-gu \
--wandb_project=rename_full \
--workers=1 \
--lr_scheduler=cosine \
--learning_rate=1e-5 \
--warmup_steps_ratio=0.01 \
--report_to_wandb \

# cd pipeline/demo
# python eval37.py