import os
import csv
import torch
import transformers
import textwrap
from PIL import Image
import sys
sys.path.append("../..")
from otter.modeling_otter_attn import OtterForConditionalGeneration
import matplotlib.pyplot as plt

model = OtterForConditionalGeneration.from_pretrained("/home/ueno/Otter/weights/OTTER-Image-MPT7B", device_map="auto")
tokenizer = model.text_tokenizer
image_processor = transformers.CLIPImageProcessor()
model.eval()

def get_image_paths(folder_path):
    image_extensions = ['.jpg', '.jpeg', '.png']
    all_files = sorted(os.listdir(folder_path))
    image_paths = [os.path.join(folder_path, file) for file in all_files if os.path.splitext(file)[1].lower() in image_extensions]
    return image_paths

def generate_list_string(items):
    items = [item.replace('_', ' ') for item in items]
    if len(items) == 1:
        return items[0]
    elif len(items) == 2:
        return f"{items[0]} and {items[1]}"
    else:
        return ", ".join(items[:-1]) + f", and {items[-1]}"
    
def test_ok(folder, sub_folder, GTs, gt_idx, output_dir):
    folder__ = folder.replace('_', ' ')
    folder_name = f'./{output_dir}/{folder}/{sub_folder}'
    os.makedirs(folder_name, exist_ok=True)
    demo_image_one = Image.open(f"/data/mvtec/{folder}/test/good/000.png").resize((224, 224)).convert("RGB")
    demo_image_two = Image.open(f"/data/mvtec/{folder}/test/{sub_folder}/000.png").resize((224, 224)).convert("RGB")
    query_folder_path = f"/data/mvtec/{folder}/test/good"
    query_image_paths = get_image_paths(query_folder_path)
    subfolder_string = generate_list_string(GTs)
    inputs = textwrap.dedent(f"""
        <image>User: This is an image of {folder__}. Does this {folder__} have any defects such as {subfolder_string}? GPT:<answer> No. This {folder__} does not have any defects such as {subfolder_string}, so it is non-defective.<|endofchunk|>
        <image>User: This is an image of {folder__}. Does this {folder__} have any defects such as {subfolder_string}? GPT:<answer> Yes. This {folder__} has some {GTs[gt_idx]}, so it is defective.<|endofchunk|>
        <image>User: This is an image of {folder__}. Does this {folder__} have any defects such as {subfolder_string}? GPT:<answer>
    """)
    inputs = "".join(inputs.split("\n"))
    lang_x = model.text_tokenizer([inputs], return_tensors="pt")
    sentence = f"{sub_folder} --> {GTs[gt_idx]}"
    sentence = f"context1: OK, context2: NG, query: OK"
    sentence = f'-----{sub_folder} start-----'
    for i, query_image_path in enumerate(query_image_paths[1:]):
        query_image = Image.open(query_image_path).resize((224, 224)).convert("RGB") 
        vision_x = image_processor.preprocess([demo_image_one, demo_image_two, query_image], return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0)
        model.text_tokenizer.padding_side = "left"
        # Get the data type from model's parameters
        model_dtype = next(model.parameters()).dtype
        # Convert tensors to the model's data type
        vision_x = vision_x.to(dtype=model_dtype)
        lang_x_input_ids = lang_x["input_ids"]
        print(f'input length: {lang_x_input_ids.shape}')
        lang_x_attention_mask = lang_x["attention_mask"]
        bad_words_id = model.text_tokenizer(["User:", "GPT1:", "GFT:", "GPT:"], add_special_tokens=False).input_ids
        generated_text, attention_maps = model.generate(
            vision_x=vision_x.to(model.device),
            lang_x=lang_x_input_ids.to(model.device),
            attention_mask=lang_x_attention_mask.to(model.device),
            max_new_tokens=512,
            num_beams=3,
            no_repeat_ngram_size=3,
            bad_words_ids=bad_words_id,
        )
        print('---------------------------------------------------')
        parsed_output = (model.text_tokenizer.decode(generated_text[0]).split("<answer>")[-1].lstrip().rstrip().split("<|endofchunk|>")[0].lstrip().rstrip().lstrip('"').rstrip('"'))
        print(query_image_path)
        print("GPT: ", parsed_output)
        for idx in range(len(attention_maps)):
            plt.imshow(query_image)
            plt.imshow(attention_maps[idx], cmap='hot', alpha=0.5)
            plt.axis('off')
            plt.savefig(f'{folder_name}/map_{idx}.png', bbox_inches='tight', pad_inches=0)
            plt.close()
        break

def test_ng(folder, sub_folder, GTs, gt_idx, output_dir):
    if folder=="grid":
        folder__ = "metal grid"
    else:
        folder__ = folder
    folder__ = folder__.replace('_', ' ')
    
    # save log
    folder_name = f'./{output_dir}/{folder}/{sub_folder}'
    os.makedirs(folder_name, exist_ok=True)
    log_name = "query_ng_attn.txt"
    with open(f'{folder_name}/{log_name}', mode='w') as f:
        f.close()
    
    # context images
    demo_image_one = Image.open(f"/data/mvtec/{folder}/test/good/000.png").resize((224, 224)).convert("RGB")
    demo_image_two = Image.open(f"/data/mvtec/{folder}/test/{sub_folder}/000.png").resize((224, 224)).convert("RGB")
    
    # query images
    query_folder_path = f"/data/mvtec/{folder}/test/{sub_folder}"
    query_image_paths = get_image_paths(query_folder_path)
    
    # input prompt
    subfolder_string = generate_list_string(GTs)
    # inputs = textwrap.dedent(f"""
    #                 <image>User: Is there any evidence of defects like {subfolder_string} in this image of the {folder__}? GPT:<answer>
    # """)
    inputs = textwrap.dedent(f"""
        <image>User: This is an image of {folder__}. Does this {folder__} have any defects such as {subfolder_string}? GPT:<answer> No. This {folder__} does not have any defects such as {subfolder_string}, so it is non-defective.<|endofchunk|>
        <image>User: This is an image of {folder__}. Does this {folder__} have any defects such as {subfolder_string}? GPT:<answer> Yes. This {folder__} has some {GTs[gt_idx]}, so it is defective.<|endofchunk|>
        <image>User: This is an image of {folder__}. Does this {folder__} have any defects such as {subfolder_string}? GPT:<answer>
    """)
    inputs = "".join(inputs.split("\n"))
    lang_x = model.text_tokenizer(
        [
            inputs
        ],
        return_tensors="pt",
    )
    
    sentence = f"{sub_folder} --> {GTs[gt_idx]}"
    sentence = f"context1: OK, context2: NG, query: NG"
    sentence = f'-----{sub_folder} start-----'
    
    for i, query_image_path in enumerate(query_image_paths[1:]):
        query_image = Image.open(query_image_path).resize((224, 224)).convert("RGB") 
        vision_x = image_processor.preprocess([demo_image_one, demo_image_two, query_image], return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0)
        model.text_tokenizer.padding_side = "left"

        # Get the data type from model's parameters
        model_dtype = next(model.parameters()).dtype

        # Convert tensors to the model's data type
        vision_x = vision_x.to(dtype=model_dtype)
        lang_x_input_ids = lang_x["input_ids"]
        lang_x_attention_mask = lang_x["attention_mask"]

        bad_words_id = model.text_tokenizer(["User:", "GPT1:", "GFT:", "GPT:"], add_special_tokens=False).input_ids
        generated_text, attention_maps = model.generate(
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
        
        print(query_image_path)
        print("GPT: ", parsed_output)
        for idx in range(len(attention_maps)):
            plt.imshow(query_image)
            plt.imshow(attention_maps[idx], cmap='hot', alpha=0.5)
            plt.axis('off')
            plt.savefig(f'{folder_name}/map_{idx}.png', bbox_inches='tight', pad_inches=0)
            plt.close()
        break

model_names = '../../log/rename_view_20/json_based_20_rand/checkpoint_21.pt'
output_dirs = 'debug'
trained_ckpt_path = model_names
train_ckpt = torch.load(trained_ckpt_path, map_location="cpu")
# model.load_state_dict(train_ckpt, strict=True)
if train_ckpt.get("model_state_dict", None) is not None:
    train_ckpt = train_ckpt["model_state_dict"]
_ = model.load_state_dict(train_ckpt, strict=False)

# log save path
output_dir = output_dirs
os.makedirs(f'./{output_dir}', exist_ok=True)

folder = "cable"
ngs = ["bent_wire","cable_swap","cut_inner_insulation","cut_outer_insulation","missing_cable","missing_wire","poke_insulation"]
GTs = ['bent','swap','crack','missing','hole']
test_ok(folder, ngs[0], GTs, 0, output_dir)
# test_ng(folder, ngs[0], GTs, 0, output_dir)
# test_ok(folder, ngs[2], GTs, 2, output_dir)
# test_ng(folder, ngs[2], GTs, 2, output_dir)
# test_ok(folder, ngs[3], GTs, 2, output_dir)
# test_ng(folder, ngs[3], GTs, 2, output_dir)

# folder = "capsule"
# ngs = ["crack","faulty_imprint","poke","scratch","squeeze"]
# GTs = ['crack','misprint','hole','scratch','misshapen']
# test_ok(folder, ngs[0], GTs, 0, output_dir)
# test_ng(folder, ngs[0], GTs, 0, output_dir)
# test_ok(folder, ngs[2], GTs, 2, output_dir)
# test_ng(folder, ngs[2], GTs, 2, output_dir)

# folder = "hazelnut"
# ngs = ["crack","cut","hole","print"]
# GTs = ['crack','scratch','hole','misprint']
# test_ok(folder, ngs[0], GTs, 0, output_dir)
# test_ng(folder, ngs[0], GTs, 0, output_dir)