import requests
import torch
import transformers
from PIL import Image
import matplotlib.pyplot as plt
import sys
import os
import textwrap
import ijson
import orjson
import random
import base64
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

sys.path.append("../..")
from otter.modeling_otter import OtterForConditionalGeneration

model = OtterForConditionalGeneration.from_pretrained("/home/data/MIMIC-IT/weights/OTTER-Image-MPT7B/", device_map="auto")
tokenizer = model.text_tokenizer
image_processor = transformers.CLIPImageProcessor()

def write_text_file(file_path, text):
    with open(file_path, mode="a") as f:
        f.write(text+"\n")
        
def test(images,cache_train_config,instructions,model_name,data_type,num=100,initialize_file=False):
    # 正解率
    folder_name = f'./result/{model_name}'
    os.makedirs(folder_name, exist_ok=True)
    if initialize_file:
        with open(f'{folder_name}/VI.txt', mode='w') as f:
            f.close()
    write_text_file(f'{folder_name}/VI.txt',f'-----{data_type} start-----')
    write_text_file(f'{folder_name}/VI.txt',"")

    trained_ckpt_path = f'../../log/{model_name}/final_weights.pt'

    train_ckpt = torch.load(trained_ckpt_path, map_location="cpu")
    if train_ckpt.get("model_state_dict", None) is not None:
        train_ckpt = train_ckpt["model_state_dict"]
    _ = model.load_state_dict(train_ckpt, strict=False)
    
    keys = list(cache_train_config.keys())
    random.seed(42)
    random.shuffle(keys)
    yesno_count = 0
    reason_count = 0
    both_count = 0
    NUM = num
    for i in range(len(keys[:NUM])):
        context1 = cache_train_config[keys[i]][0]
        context2 = cache_train_config[keys[i]][1]
        query = keys[i].split('=')[0]
        
        str_data1 = images[context1] # コンテキスト1
        str_data2 = images[context2] # コンテキスト2
        str_data3 = images[query] # クエリ
        
        # デコードしたバイトデータをImageオブジェクトに変換
        demo_image_one = Image.open(BytesIO(base64.urlsafe_b64decode(str_data1))).convert("RGB")
        demo_image_two = Image.open(BytesIO(base64.urlsafe_b64decode(str_data2))).convert("RGB")
        query_image = Image.open(BytesIO(base64.urlsafe_b64decode(str_data3))).convert("RGB")

        vision_x = image_processor.preprocess([demo_image_one, demo_image_two, query_image], return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0)
        model.text_tokenizer.padding_side = "left"
        
        inputs = textwrap.dedent(f"""
            <image>User:{instructions["data"][context1]["instruction"]} GPT:<answer>{instructions["data"][context1]["answer"]}<|endofchunk|>
            <image>User:{instructions["data"][context2]["instruction"]} GPT:<answer>{instructions["data"][context2]["answer"]}<|endofchunk|>
            <image>User:{instructions["data"][query]["instruction"]} GPT:<answer>
        """)
        inputs = "".join(inputs.split("\n"))
        lang_x = model.text_tokenizer(
            [
                inputs
            ],
            return_tensors="pt",
        )

        # Get the data type from model's parameters
        model_dtype = next(model.parameters()).dtype

        # Convert tensors to the model's data type
        vision_x = vision_x.to(dtype=model_dtype)
        lang_x_input_ids = lang_x["input_ids"]
        lang_x_attention_mask = lang_x["attention_mask"]

        bad_words_id = model.text_tokenizer(["User:", "GPT1:", "GFT:", "GPT:"], add_special_tokens=False).input_ids
        generated_text = model.generate(
            vision_x=vision_x.to(model.device),
            lang_x=lang_x_input_ids.to(model.device),
            attention_mask=lang_x_attention_mask.to(model.device),
            max_new_tokens=512,
            num_beams=3,
            no_repeat_ngram_size=3,
            bad_words_ids=bad_words_id,
        )

        parsed_output = (
            model.text_tokenizer.decode(generated_text[0]).split("<answer>")[-1].lstrip().rstrip().split("<|endofchunk|>")[0].lstrip().rstrip().lstrip('"').rstrip('"')
        )
        
        if instructions["data"][query]["answer"].split(" ")[0].lower()==parsed_output.split(" ")[0].lower():
            yesno_count += 1
            if len(parsed_output.split(" ")) > 1:
                if instructions["data"][query]["answer"].split(" ")[1].lower()==parsed_output.split(" ")[1].lower():
                    both_count += 1
        if len(parsed_output.split(" ")) > 1:
            if instructions["data"][query]["answer"].split(" ")[1].lower()==parsed_output.split(" ")[1].lower():
                    reason_count += 1
    
    sentence_yes = f"yesno correct: {yesno_count}, total: {NUM}, acc: {(yesno_count / NUM) * 100:.2f}%"
    sentence_no = f"reason correct: {reason_count}, total: {NUM}, acc: {(reason_count / NUM) * 100:.2f}%"
    sentence_both = f"both correct: {both_count}, total: {NUM}, acc: {(both_count / NUM) * 100:.2f}%"
    print(sentence_yes)
    print(sentence_no)
    print(sentence_both)
    write_text_file(f'{folder_name}/VI.txt',sentence_yes)
    write_text_file(f'{folder_name}/VI.txt',sentence_no)
    write_text_file(f'{folder_name}/VI.txt',sentence_both)
        

# 2
model_name = "VI_full_otter/batch128_epoch1_lr-5_true"

print("train")
images = {}
images_path="/home/data/MIMIC-IT/VI_full_jsons/VI_train.json"
with open(images_path, "rb") as f:
    for key, value in ijson.kvitems(f, "", use_float=True):
        images[key] = value

train_config_path="/home/data/MIMIC-IT/VI_full_jsons/VI_train_pairs25_train.json"
with open(train_config_path, "rb") as f:
    cache_train_config = orjson.loads(f.read())

mimicit_path="/home/data/MIMIC-IT/VI_full_jsons/VI_train_instructions.json"
with open(mimicit_path, "rb") as f:
    instructions = orjson.loads(f.read())
test(images, cache_train_config,instructions,model_name,data_type="train",initialize_file=True)

print("val")
images = {}
images_path="/home/data/MIMIC-IT/VI_full_jsons/VI_val.json"
with open(images_path, "rb") as f:
    for key, value in ijson.kvitems(f, "", use_float=True):
        images[key] = value

train_config_path="/home/data/MIMIC-IT/VI_full_jsons/VI_val_pairs1_train.json"
with open(train_config_path, "rb") as f:
    cache_train_config = orjson.loads(f.read())

mimicit_path="/home/data/MIMIC-IT/VI_full_jsons/VI_val_instructions.json"
with open(mimicit_path, "rb") as f:
    instructions = orjson.loads(f.read())
test(images, cache_train_config,instructions,model_name,data_type="val")

# 3
model_name = "VI_full_otter/batch128_epoch1_lr-5_false"

print("train")
images = {}
images_path="/home/data/MIMIC-IT/VI_full_jsons/VI_train.json"
with open(images_path, "rb") as f:
    for key, value in ijson.kvitems(f, "", use_float=True):
        images[key] = value

train_config_path="/home/data/MIMIC-IT/VI_full_jsons/VI_train_pairs25_train.json"
with open(train_config_path, "rb") as f:
    cache_train_config = orjson.loads(f.read())

mimicit_path="/home/data/MIMIC-IT/VI_full_jsons/VI_train_instructions.json"
with open(mimicit_path, "rb") as f:
    instructions = orjson.loads(f.read())
test(images, cache_train_config,instructions,model_name,data_type="train",initialize_file=True)

print("val")
images = {}
images_path="/home/data/MIMIC-IT/VI_full_jsons/VI_val.json"
with open(images_path, "rb") as f:
    for key, value in ijson.kvitems(f, "", use_float=True):
        images[key] = value

train_config_path="/home/data/MIMIC-IT/VI_full_jsons/VI_val_pairs1_train.json"
with open(train_config_path, "rb") as f:
    cache_train_config = orjson.loads(f.read())

mimicit_path="/home/data/MIMIC-IT/VI_full_jsons/VI_val_instructions.json"
with open(mimicit_path, "rb") as f:
    instructions = orjson.loads(f.read())
test(images, cache_train_config,instructions,model_name,data_type="val")

# # 4
# model_name = "AC_VI_full_otter/batch128_epoch1_lr-5_true"

# print("train")
# images = {}
# images_path="/home/data/MIMIC-IT/VI_full_jsons/VI_train.json"
# with open(images_path, "rb") as f:
#     for key, value in ijson.kvitems(f, "", use_float=True):
#         images[key] = value

# train_config_path="/home/data/MIMIC-IT/VI_full_jsons/VI_train_pairs25_train.json"
# with open(train_config_path, "rb") as f:
#     cache_train_config = orjson.loads(f.read())

# mimicit_path="/home/data/MIMIC-IT/VI_full_jsons/VI_train_instructions.json"
# with open(mimicit_path, "rb") as f:
#     instructions = orjson.loads(f.read())
# test(images, cache_train_config,instructions,model_name,data_type="train",initialize_file=True)

# print("val")
# images = {}
# images_path="/home/data/MIMIC-IT/VI_full_jsons/VI_val.json"
# with open(images_path, "rb") as f:
#     for key, value in ijson.kvitems(f, "", use_float=True):
#         images[key] = value

# train_config_path="/home/data/MIMIC-IT/VI_full_jsons/VI_val_pairs1_train.json"
# with open(train_config_path, "rb") as f:
#     cache_train_config = orjson.loads(f.read())

# mimicit_path="/home/data/MIMIC-IT/VI_full_jsons/VI_val_instructions.json"
# with open(mimicit_path, "rb") as f:
#     instructions = orjson.loads(f.read())
# test(images, cache_train_config,instructions,model_name,data_type="val")

# # 5
# model_name = "AC_VI_full_otter/batch128_epoch1_lr-5_false"

# print("train")
# images = {}
# images_path="/home/data/MIMIC-IT/VI_full_jsons/VI_train.json"
# with open(images_path, "rb") as f:
#     for key, value in ijson.kvitems(f, "", use_float=True):
#         images[key] = value

# train_config_path="/home/data/MIMIC-IT/VI_full_jsons/VI_train_pairs25_train.json"
# with open(train_config_path, "rb") as f:
#     cache_train_config = orjson.loads(f.read())

# mimicit_path="/home/data/MIMIC-IT/VI_full_jsons/VI_train_instructions.json"
# with open(mimicit_path, "rb") as f:
#     instructions = orjson.loads(f.read())
# test(images, cache_train_config,instructions,model_name,data_type="train",initialize_file=True)

# print("val")
# images = {}
# images_path="/home/data/MIMIC-IT/VI_full_jsons/VI_val.json"
# with open(images_path, "rb") as f:
#     for key, value in ijson.kvitems(f, "", use_float=True):
#         images[key] = value

# train_config_path="/home/data/MIMIC-IT/VI_full_jsons/VI_val_pairs1_train.json"
# with open(train_config_path, "rb") as f:
#     cache_train_config = orjson.loads(f.read())

# mimicit_path="/home/data/MIMIC-IT/VI_full_jsons/VI_val_instructions.json"
# with open(mimicit_path, "rb") as f:
#     instructions = orjson.loads(f.read())
# test(images, cache_train_config,instructions,model_name,data_type="val")