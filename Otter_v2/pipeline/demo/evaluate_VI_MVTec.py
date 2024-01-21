import os
import requests
import torch
import transformers
import textwrap
import matplotlib.pyplot as plt
from PIL import Image
import sys

sys.path.append("../..")
from otter.modeling_otter import OtterForConditionalGeneration

model = OtterForConditionalGeneration.from_pretrained("/home/ueno/Otter/weights/OTTER-Image-MPT7B", device_map="auto")
tokenizer = model.text_tokenizer
image_processor = transformers.CLIPImageProcessor()
model.eval()

def get_image_paths(folder_path):
    image_extensions = ['.jpg', '.jpeg', '.png']
    all_files = sorted(os.listdir(folder_path))
    image_paths = [os.path.join(folder_path, file) for file in all_files if os.path.splitext(file)[1].lower() in image_extensions]
    return image_paths

def write_text_file(file_path, text):
    with open(file_path, mode="a") as f:
        f.write(text+"\n")
        
def generate_list_string(items):
    items = [item.replace('_', ' ') for item in items]
    if len(items) == 1:
        return items[0]
    elif len(items) == 2:
        return f"{items[0]} and {items[1]}"
    else:
        return ", ".join(items[:-1]) + f", and {items[-1]}"
    
def test_ok(folder, sub_folder, GTs, gt_idx, model_name):
    if folder=="grid":
        folder__ = "metal grid"
    else:
        folder__ = folder
    folder__ = folder__.replace('_', ' ')
    
    # save log
    folder_name = f'./result_last_ffn/{model_name}/{folder}/{sub_folder}'
    os.makedirs(folder_name, exist_ok=True)
    log_name = "query_ok.txt"
    with open(f'{folder_name}/{log_name}', mode='w') as f:
        f.close()
    
    # context images
    demo_image_one = Image.open(f"/data/dataset/mvtec/{folder}/test/good/000.png").resize((224, 224)).convert("RGB")
    demo_image_two = Image.open(f"/data/dataset/mvtec/{folder}/test/{sub_folder}/000.png").resize((224, 224)).convert("RGB")
    
    # query images
    query_folder_path = f"/data/dataset/mvtec/{folder}/test/good"
    query_image_paths = get_image_paths(query_folder_path)
    
    # input prompt
    subfolder_string = generate_list_string(GTs)
    inputs = textwrap.dedent(f"""
                    <image>User: Are there any visible faults like {subfolder_string} in this {folder__} as shown in the image? GPT:<answer> No. This {folder__} does not display any visible faults such as {subfolder_string}. It appears to be in non-defective condition.<|endofchunk|>
                    <image>User: Does this image of the {folder__} reveal any damage like {subfolder_string}? GPT:<answer> Yes. This {folder__} exhibits {GTs[gt_idx]}, indicating damage.<|endofchunk|>
                    <image>User: Is there any evidence of defects like {subfolder_string} in this image of the {folder__}? GPT:<answer>
    """)
    inputs = "".join(inputs.split("\n"))
    lang_x = model.text_tokenizer(
        [
            inputs
        ],
        return_tensors="pt",
    )
    
    sentence = f"{sub_folder} --> {GTs[gt_idx]}"
    print(sentence)
    write_text_file(f'{folder_name}/{log_name}',sentence)
    write_text_file(f'{folder_name}/{log_name}',"")
    
    sentence = f"context1: OK, context2: NG, query: OK"
    print(sentence)
    write_text_file(f'{folder_name}/{log_name}',sentence)
    write_text_file(f'{folder_name}/{log_name}',"")

    print(inputs)
    write_text_file(f'{folder_name}/{log_name}',inputs)
    write_text_file(f'{folder_name}/{log_name}',"")

    sentence = f'-----{sub_folder} start-----'
    write_text_file(f'{folder_name}/{log_name}',sentence)
    write_text_file(f'{folder_name}/{log_name}',"")
    
    yesno_count = 0
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
        
        if parsed_output.split(".")[0].lower()=="no":
                yesno_count += 1
                
        print(query_image_path)
        print("GPT: ", parsed_output)
        write_text_file(f'{folder_name}/{log_name}',query_image_path)
        write_text_file(f'{folder_name}/{log_name}',parsed_output)
        write_text_file(f'{folder_name}/{log_name}',"")
    
    yesno_acc = f"correct: {yesno_count}, total: {len(query_image_paths)-1}, yesno acc: {(yesno_count / (len(query_image_paths)-1)) * 100:.2f}%"
    
    write_text_file(f'{folder_name}/{log_name}',f'-----{sub_folder} end-----')
    write_text_file(f'{folder_name}/{log_name}',yesno_acc)
    print(yesno_acc)
    
def test_ng(folder, sub_folder, GTs, gt_idx, model_name):
    if folder=="grid":
        folder__ = "metal grid"
    else:
        folder__ = folder
    folder__ = folder__.replace('_', ' ')
    
    # save log
    folder_name = f'./result_last_ffn/{model_name}/{folder}/{sub_folder}'
    os.makedirs(folder_name, exist_ok=True)
    log_name = "query_ng.txt"
    with open(f'{folder_name}/{log_name}', mode='w') as f:
        f.close()
    
    # context images
    demo_image_one = Image.open(f"/data/dataset/mvtec/{folder}/test/good/000.png").resize((224, 224)).convert("RGB")
    demo_image_two = Image.open(f"/data/dataset/mvtec/{folder}/test/{sub_folder}/000.png").resize((224, 224)).convert("RGB")
    
    # query images
    query_folder_path = f"/data/dataset/mvtec/{folder}/test/{sub_folder}"
    query_image_paths = get_image_paths(query_folder_path)
    
    # input prompt
    subfolder_string = generate_list_string(GTs)
    inputs = textwrap.dedent(f"""
                    <image>User: Are there any visible faults like {subfolder_string} in this {folder__} as shown in the image? GPT:<answer> No. This {folder__} does not display any visible faults such as {subfolder_string}. It appears to be in non-defective condition.<|endofchunk|>
                    <image>User: Does this image of the {folder__} reveal any damage like {subfolder_string}? GPT:<answer> Yes. This {folder__} exhibits {GTs[gt_idx]}, indicating damage.<|endofchunk|>
                    <image>User: Is there any evidence of defects like {subfolder_string} in this image of the {folder__}? GPT:<answer>
    """)
    inputs = "".join(inputs.split("\n"))
    lang_x = model.text_tokenizer(
        [
            inputs
        ],
        return_tensors="pt",
    )
    
    sentence = f"{sub_folder} --> {GTs[gt_idx]}"
    print(sentence)
    write_text_file(f'{folder_name}/{log_name}',sentence)
    write_text_file(f'{folder_name}/{log_name}',"")
    
    sentence = f"context1: OK, context2: NG, query: NG"
    print(sentence)
    write_text_file(f'{folder_name}/{log_name}',sentence)
    write_text_file(f'{folder_name}/{log_name}',"")

    print(inputs)
    write_text_file(f'{folder_name}/{log_name}',inputs)
    write_text_file(f'{folder_name}/{log_name}',"")

    sentence = f'-----{sub_folder} start-----'
    write_text_file(f'{folder_name}/{log_name}',sentence)
    write_text_file(f'{folder_name}/{log_name}',"")
    
    yesno_count = 0
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
        
        if parsed_output.split(".")[0].lower()=="yes":
                yesno_count += 1
                
        print(query_image_path)
        print("GPT: ", parsed_output)
        write_text_file(f'{folder_name}/{log_name}',query_image_path)
        write_text_file(f'{folder_name}/{log_name}',parsed_output)
        write_text_file(f'{folder_name}/{log_name}',"")
    
    yesno_acc = f"correct: {yesno_count}, total: {len(query_image_paths)-1}, yesno acc: {(yesno_count / (len(query_image_paths)-1)) * 100:.2f}%"
    
    write_text_file(f'{folder_name}/{log_name}',f'-----{sub_folder} end-----')
    write_text_file(f'{folder_name}/{log_name}',yesno_acc)
    print(yesno_acc)

# load weight
model_name = '../../log/PC2AC/VI'
trained_ckpt_path = f'{model_name}/final_weights.pt'
train_ckpt = torch.load(trained_ckpt_path, map_location="cpu")
if train_ckpt.get("model_state_dict", None) is not None:
    train_ckpt = train_ckpt["model_state_dict"]
_ = model.load_state_dict(train_ckpt, strict=False)

# log save path
save_name = "VI_full_otter_product/debug"

folder = "wood"
ngs = ["color","scratch","liquid","hole"]
GTs = ["stain","scratch","hole"]
# GTs = ["stained wood","scratched wood","wood with holes"]
test_ok(folder, ngs[0], GTs, 0, save_name)
test_ng(folder, ngs[0], GTs, 0, save_name)
test_ok(folder, ngs[1], GTs, 1, save_name)
test_ng(folder, ngs[1], GTs, 1, save_name)
test_ok(folder, ngs[2], GTs, 0, save_name)
test_ng(folder, ngs[2], GTs, 0, save_name)
test_ok(folder, ngs[3], GTs, 2, save_name)
test_ng(folder, ngs[3], GTs, 2, save_name)

folder = "bottle"
ngs = ["broken_large","broken_small","contamination"]
GTs = ['broken','contamination']
# GTs = ['broken bottle','bottle with contamination']
test_ok(folder, ngs[0], GTs, 0, save_name)
test_ng(folder, ngs[0], GTs, 0, save_name)
test_ok(folder, ngs[1], GTs, 0, save_name)
test_ng(folder, ngs[1], GTs, 0, save_name)
test_ok(folder, ngs[2], GTs, 1, save_name)
test_ng(folder, ngs[2], GTs, 1, save_name)

folder = "cable"
ngs = ["bent_wire","cable_swap","cut_inner_insulation","cut_outer_insulation","missing_cable","missing_wire","poke_insulation"]
GTs = ['bent','swapp','crack','missing','hole']
# GTs = ['bent cable','swapped cable','cracked cable','missing cable','cable with holes']
test_ok(folder, ngs[0], GTs, 0, save_name)
test_ng(folder, ngs[0], GTs, 0, save_name)
test_ok(folder, ngs[1], GTs, 1, save_name)
test_ng(folder, ngs[1], GTs, 1, save_name)
test_ok(folder, ngs[2], GTs, 2, save_name)
test_ng(folder, ngs[2], GTs, 2, save_name)
test_ok(folder, ngs[3], GTs, 2, save_name)
test_ng(folder, ngs[3], GTs, 2, save_name)
test_ok(folder, ngs[4], GTs, 3, save_name)
test_ng(folder, ngs[4], GTs, 3, save_name)
test_ok(folder, ngs[5], GTs, 3, save_name)
test_ng(folder, ngs[5], GTs, 3, save_name)
test_ok(folder, ngs[6], GTs, 4, save_name)
test_ng(folder, ngs[6], GTs, 4, save_name)

folder = "capsule"
ngs = ["crack","faulty_imprint","poke","scratch","squeeze"]
GTs = ['crack','misprint','hole','scratch','misshapen']
# GTs = ['cracked capsule','misprinted capsule','capsule with holes','scratched capsule', 'misshapen capsule']
test_ok(folder, ngs[0], GTs, 0, save_name)
test_ng(folder, ngs[0], GTs, 0, save_name)
test_ok(folder, ngs[1], GTs, 1, save_name)
test_ng(folder, ngs[1], GTs, 1, save_name)
test_ok(folder, ngs[2], GTs, 2, save_name)
test_ng(folder, ngs[2], GTs, 2, save_name)
test_ok(folder, ngs[3], GTs, 3, save_name)
test_ng(folder, ngs[3], GTs, 3, save_name)
test_ok(folder, ngs[4], GTs, 4, save_name)
test_ng(folder, ngs[4], GTs, 4, save_name)

folder = "carpet"
ngs = ["color","cut","hole","metal_contamination","thread"]
GTs = ['stain','cut','hole','contamination']
# GTs = ['stained carpet','cut carpet','carpet with holes','carpet with contamination']
test_ok(folder, ngs[0], GTs, 0, save_name)
test_ng(folder, ngs[0], GTs, 0, save_name)
test_ok(folder, ngs[1], GTs, 1, save_name)
test_ng(folder, ngs[1], GTs, 1, save_name)
test_ok(folder, ngs[2], GTs, 2, save_name)
test_ng(folder, ngs[2], GTs, 2, save_name)
test_ok(folder, ngs[3], GTs, 3, save_name)
test_ng(folder, ngs[3], GTs, 3, save_name)
test_ok(folder, ngs[4], GTs, 3, save_name)
test_ng(folder, ngs[4], GTs, 3, save_name)

folder = "grid"
ngs = ["bent","broken","glue","metal_contamination","thread"]
GTs = ['bent','broken','contamination']
# GTs = ['bent metal grid','broken metal grid','metal grid with contamination']
test_ok(folder, ngs[0], GTs, 0, save_name)
test_ng(folder, ngs[0], GTs, 0, save_name)
test_ok(folder, ngs[1], GTs, 1, save_name)
test_ng(folder, ngs[1], GTs, 1, save_name)
test_ok(folder, ngs[2], GTs, 2, save_name)
test_ng(folder, ngs[2], GTs, 2, save_name)
test_ok(folder, ngs[3], GTs, 2, save_name)
test_ng(folder, ngs[3], GTs, 2, save_name)
test_ok(folder, ngs[4], GTs, 2, save_name)
test_ng(folder, ngs[4], GTs, 2, save_name)

folder = "hazelnut"
ngs = ["crack","cut","hole","print"]
GTs = ['crack','scratch','hole','misprint']
# GTs = ['cracked hazelnut','scratched hazelnut','hazelnut with holes','hazelnut with white marks']
test_ok(folder, ngs[0], GTs, 0, save_name)
test_ng(folder, ngs[0], GTs, 0, save_name)
test_ok(folder, ngs[1], GTs, 1, save_name)
test_ng(folder, ngs[1], GTs, 1, save_name)
test_ok(folder, ngs[2], GTs, 2, save_name)
test_ng(folder, ngs[2], GTs, 2, save_name)
test_ok(folder, ngs[3], GTs, 3, save_name)
test_ng(folder, ngs[3], GTs, 3, save_name)

folder = "leather"
ngs = ["color","cut","fold","glue","poke"]
GTs = ['stain','scratch','wrinkle','hole']
# GTs = ['stained leather','cut leather','wrinkle leather','leather with holes']
test_ok(folder, ngs[0], GTs, 0, save_name)
test_ng(folder, ngs[0], GTs, 0, save_name)
test_ok(folder, ngs[1], GTs, 1, save_name)
test_ng(folder, ngs[1], GTs, 1, save_name)
test_ok(folder, ngs[2], GTs, 2, save_name)
test_ng(folder, ngs[2], GTs, 2, save_name)
test_ok(folder, ngs[3], GTs, 0, save_name)
test_ng(folder, ngs[3], GTs, 0, save_name)
test_ok(folder, ngs[4], GTs, 3, save_name)
test_ng(folder, ngs[4], GTs, 3, save_name)

folder = "metal_nut"
ngs = ["bent","color","flip","scratch"]
GTs = ['bent','stain','flip','scratch']
# GTs = ['bent metal nut','stained metal nut','flipped metal nut','scratched metal nut']
test_ok(folder, ngs[0], GTs, 0, save_name)
test_ng(folder, ngs[0], GTs, 0, save_name)
test_ok(folder, ngs[1], GTs, 1, save_name)
test_ng(folder, ngs[1], GTs, 1, save_name)
test_ok(folder, ngs[2], GTs, 2, save_name)
test_ng(folder, ngs[2], GTs, 2, save_name)
test_ok(folder, ngs[3], GTs, 3, save_name)
test_ng(folder, ngs[3], GTs, 3, save_name)

folder = "pill"
ngs = ["color","contamination","crack","faulty_imprint","scratch","pill_type"]
GTs = ['stain','contamination','crack','misprint','scratch']
# GTs = ['stained pill','pill with contamination','cracked pill','misprinted pill','scratched pill']
test_ok(folder, ngs[0], GTs, 0, save_name)
test_ng(folder, ngs[0], GTs, 0, save_name)
test_ok(folder, ngs[1], GTs, 1, save_name)
test_ng(folder, ngs[1], GTs, 1, save_name)
test_ok(folder, ngs[2], GTs, 2, save_name)
test_ng(folder, ngs[2], GTs, 2, save_name)
test_ok(folder, ngs[3], GTs, 3, save_name)
test_ng(folder, ngs[3], GTs, 3, save_name)
test_ok(folder, ngs[4], GTs, 4, save_name)
test_ng(folder, ngs[4], GTs, 4, save_name)
test_ok(folder, ngs[5], GTs, 0, save_name)
test_ng(folder, ngs[5], GTs, 0, save_name)

folder = "screw"
ngs = ["manipulated_front","scratch_head","scratch_neck","thread_side","thread_top"]
GTs = ['strip','chip']
# GTs = ['stripped screw','chipped screw']
test_ok(folder, ngs[0], GTs, 0, save_name)
test_ng(folder, ngs[0], GTs, 0, save_name)
test_ok(folder, ngs[1], GTs, 1, save_name)
test_ng(folder, ngs[1], GTs, 1, save_name)
test_ok(folder, ngs[2], GTs, 1, save_name)
test_ng(folder, ngs[2], GTs, 1, save_name)
test_ok(folder, ngs[3], GTs, 1, save_name)
test_ng(folder, ngs[3], GTs, 1, save_name)
test_ok(folder, ngs[4], GTs, 1, save_name)
test_ng(folder, ngs[4], GTs, 1, save_name)

folder = "tile"
ngs = ["crack","glue_strip","gray_stroke","oil","rough"]
GTs = ['crack','contamination','stain']
# GTs = ['cracked tile','tile with contamination','stained tile']
test_ok(folder, ngs[0], GTs, 0, save_name)
test_ng(folder, ngs[0], GTs, 0, save_name)
test_ok(folder, ngs[1], GTs, 1, save_name)
test_ng(folder, ngs[1], GTs, 1, save_name)
test_ok(folder, ngs[2], GTs, 2, save_name)
test_ng(folder, ngs[2], GTs, 2, save_name)
test_ok(folder, ngs[3], GTs, 2, save_name)
test_ng(folder, ngs[3], GTs, 2, save_name)
test_ok(folder, ngs[4], GTs, 2, save_name)
test_ng(folder, ngs[4], GTs, 2, save_name)

folder = "toothbrush"
ngs = ["defective"]
GTs = ['broken']
# GTs = ['broken toothbrush']
test_ok(folder, ngs[0], GTs, 0, save_name)
test_ng(folder, ngs[0], GTs, 0, save_name)

folder = "transistor"
ngs = ["bent_lead","cut_lead","damaged_case","misplaced"]
GTs = ['bent','cut','broken','misalignment']
# GTs = ['bent transistor','cut transistor','broken transistor','misplaced transistor']
test_ok(folder, ngs[0], GTs, 0, save_name)
test_ng(folder, ngs[0], GTs, 0, save_name)
test_ok(folder, ngs[1], GTs, 1, save_name)
test_ng(folder, ngs[1], GTs, 1, save_name)
test_ok(folder, ngs[2], GTs, 2, save_name)
test_ng(folder, ngs[2], GTs, 2, save_name)
test_ok(folder, ngs[3], GTs, 3, save_name)
test_ng(folder, ngs[3], GTs, 3, save_name)

folder = "zipper"
ngs = ["broken_teeth","fabric_border","fabric_interior","rough","split_teeth","squeezed_teeth"]
GTs = ['broken','tear','frayed','misshapen']
# GTs = ['broken zipper','torn zipper','frayed zipper','misshapen zipper']
test_ok(folder, ngs[0], GTs, 0, save_name)
test_ng(folder, ngs[0], GTs, 0, save_name)
test_ok(folder, ngs[1], GTs, 1, save_name)
test_ng(folder, ngs[1], GTs, 1, save_name)
test_ok(folder, ngs[2], GTs, 2, save_name)
test_ng(folder, ngs[2], GTs, 2, save_name)
test_ok(folder, ngs[3], GTs, 2, save_name)
test_ng(folder, ngs[3], GTs, 2, save_name)
test_ok(folder, ngs[4], GTs, 3, save_name)
test_ng(folder, ngs[4], GTs, 3, save_name)
test_ok(folder, ngs[5], GTs, 3, save_name)
test_ng(folder, ngs[5], GTs, 3, save_name)