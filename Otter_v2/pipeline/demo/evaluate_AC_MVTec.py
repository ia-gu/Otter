import requests
import torch
import transformers
from PIL import Image
import matplotlib.pyplot as plt
import sys
import os
import textwrap

sys.path.append("../..")
from otter.modeling_otter import OtterForConditionalGeneration

model = OtterForConditionalGeneration.from_pretrained("/home/data/MIMIC-IT/weights/OTTER-Image-MPT7B/", device_map="auto")
tokenizer = model.text_tokenizer
image_processor = transformers.CLIPImageProcessor()

def get_image_paths(folder_path):
    image_extensions = ['.jpg', '.jpeg', '.png']  # 画像の拡張子リスト
    all_files = sorted(os.listdir(folder_path)) # フォルダ内の全てのファイルを取得
    image_paths = [os.path.join(folder_path, file) for file in all_files if os.path.splitext(file)[1].lower() in image_extensions] # 画像のパスを抽出してリストに格納
    return image_paths

def write_text_file(file_path, text):
    with open(file_path, mode="a") as f:
        f.write(text+"\n")
        
def generate_list_string(items):
    # アンダースコアをスペースに変換
    items = [item.replace('_', ' ') for item in items]
    if len(items) == 1:
        return items[0]
    elif len(items) == 2:
        return f"{items[0]} and {items[1]}"
    else:
        return ", ".join(items[:-1]) + f", and {items[-1]}"
    
def test_category(folder, sub_folder, GTs, model_name):
    # load check point
    trained_ckpt_path = f'../../log/{model_name}/final_weights.pt'

    train_ckpt = torch.load(trained_ckpt_path, map_location="cpu")
    if train_ckpt.get("model_state_dict", None) is not None:
        train_ckpt = train_ckpt["model_state_dict"]
    _ = model.load_state_dict(train_ckpt, strict=False)
    
    # acc = []
    plus_name = model_name.split("/")[0]
    for j, (sub,gt) in enumerate(zip(sub_folder,GTs)):
        folder_name = f'./{plus_name}_result/{folder}/{sub}/{model_name}'
        os.makedirs(folder_name, exist_ok=True)
        with open(f'{folder_name}/AC_MVTec_category.txt', mode='w') as f:
            f.close()
        
        model.text_tokenizer.padding_side = "left"
        
        sentence = f"{sub} --> {gt}"
        # print(sentence)
        write_text_file(f'{folder_name}/AC_MVTec_category.txt',sentence)
        inputs = textwrap.dedent(f"""
           <image>User: What are the defects present in this image? If there are none, please say None. GPT:<answer>
        """)
        
        inputs = "".join(inputs.split("\n"))
        lang_x = model.text_tokenizer(
            [
                inputs
            ],
            return_tensors="pt",
        )
        
        write_text_file(f'{folder_name}/AC_MVTec_category.txt',f'-----{sub} start-----')
        write_text_file(f'{folder_name}/AC_MVTec_category.txt',"")
        write_text_file(f'{folder_name}/AC_MVTec_category.txt',f'{inputs}')
        write_text_file(f'{folder_name}/AC_MVTec_category.txt',"")
            
        query_folder_path = f"/home/dataset/mvtec/{folder}/test/{sub}"
        query_image_paths = get_image_paths(query_folder_path)
        count = 0
        for i, query_image_path in enumerate(query_image_paths[:]):
            # print(query_image_path)
            query_image = Image.open(query_image_path).resize((224, 224)).convert("RGB")
            vision_x = image_processor.preprocess([query_image], return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0)
        
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
            
            if parsed_output.lower()==gt.lower():
                count += 1
                    
            write_text_file(f'{folder_name}/AC_MVTec_category.txt',query_image_path)
            write_text_file(f'{folder_name}/AC_MVTec_category.txt',parsed_output)
            write_text_file(f'{folder_name}/AC_MVTec_category.txt',"")
            
        accuracy = f"correct: {count}, total: {len(query_image_paths)}, acc: {(count / (len(query_image_paths))) * 100:.2f}%"
        # acc.append((sub,accuracy))
        
        write_text_file(f'{folder_name}/AC_MVTec_category.txt',accuracy)
        write_text_file(f'{folder_name}/AC_MVTec_category.txt',f'-----{sub} end-----')
        
def test_yesno(folder, sub_folder, GTs, model_name):
    # load check point
    trained_ckpt_path = f'../../log/{model_name}/final_weights.pt'

    train_ckpt = torch.load(trained_ckpt_path, map_location="cpu")
    if train_ckpt.get("model_state_dict", None) is not None:
        train_ckpt = train_ckpt["model_state_dict"]
    _ = model.load_state_dict(train_ckpt, strict=False)
    
    plus_name = model_name.split("/")[0]
    for j, (sub,gt) in enumerate(zip(sub_folder,GTs)):
        folder_name = f'./{plus_name}_result/{folder}/{sub}/{model_name}'
        os.makedirs(folder_name, exist_ok=True)
        with open(f'{folder_name}/AC_MVTec_yesno.txt', mode='w') as f:
            f.close()
        
        model.text_tokenizer.padding_side = "left"
        
        sentence = f"{sub} --> {gt}"
        # print(sentence)
        write_text_file(f'{folder_name}/AC_MVTec_yesno.txt',sentence)
        inputs = textwrap.dedent(f"""
           <image>User: Does this image have any defects? If there are any defects, please provide the defect name. If not, please say None. GPT:<answer>
        """)
        
        inputs = "".join(inputs.split("\n"))
        lang_x = model.text_tokenizer(
            [
                inputs
            ],
            return_tensors="pt",
        )
        
        write_text_file(f'{folder_name}/AC_MVTec_yesno.txt',f'-----{sub} start-----')
        write_text_file(f'{folder_name}/AC_MVTec_yesno.txt',"")
        write_text_file(f'{folder_name}/AC_MVTec_yesno.txt',f'{inputs}')
        write_text_file(f'{folder_name}/AC_MVTec_yesno.txt',"")
            
        query_folder_path = f"/home/dataset/mvtec/{folder}/test/{sub}"
        query_image_paths = get_image_paths(query_folder_path)
        yesno_count = 0
        reason_count = 0
        both_count = 0
        for i, query_image_path in enumerate(query_image_paths[:]):
            # print(query_image_path)
            query_image = Image.open(query_image_path).resize((224, 224)).convert("RGB")
            vision_x = image_processor.preprocess([query_image], return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0)
        
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
            
            if sub == "good":
                if "no"==parsed_output.split(" ")[0].lower():
                    yesno_count += 1
                    if len(parsed_output.split(" ")) > 1:
                        if sub.lower()==parsed_output.split(" ")[1].lower():
                            both_count += 1
            else:
                if "yes"==parsed_output.split(" ")[0].lower():
                    yesno_count += 1
                    if len(parsed_output.split(" ")) > 1:
                        if sub.lower()==parsed_output.split(" ")[1].lower():
                            both_count += 1
            
            if len(parsed_output.split(" ")) > 1:
                if sub.lower()==parsed_output.split(" ")[1].lower():
                        reason_count += 1
                    
            write_text_file(f'{folder_name}/AC_MVTec_yesno.txt',query_image_path)
            write_text_file(f'{folder_name}/AC_MVTec_yesno.txt',parsed_output)
            write_text_file(f'{folder_name}/AC_MVTec_yesno.txt',"")
        
        yesno_accuracy = f"yesno correct: {yesno_count}, total: {len(query_image_paths)}, acc: {(yesno_count / len(query_image_paths)) * 100:.2f}%"
        reason_accuracy = f"reason correct: {reason_count}, total: {len(query_image_paths)}, acc: {(reason_count / len(query_image_paths)) * 100:.2f}%"
        both_accuracy = f"both correct: {both_count}, total: {len(query_image_paths)}, acc: {(both_count / len(query_image_paths)) * 100:.2f}%"
        
        write_text_file(f'{folder_name}/AC_MVTec_yesno.txt',f'-----{sub} end-----')
        write_text_file(f'{folder_name}/AC_MVTec_yesno.txt',yesno_accuracy)
        write_text_file(f'{folder_name}/AC_MVTec_yesno.txt',reason_accuracy)
        write_text_file(f'{folder_name}/AC_MVTec_yesno.txt',both_accuracy)





# model_name = "AC_previous/batch128_epoch5_lr-5_AC_previous"

# folder = "bottle"
# sub_folder = ["good","broken_large","broken_small","contamination"]
# GTs = ['None','broken','broken','contamination']
# test_yesno(folder, sub_folder, GTs, model_name)

# folder = "cable"
# sub_folder = ["good","bent_wire","cable_swap","cut_inner_insulation","cut_outer_insulation","missing_cable","missing_wire","poke_insulation"]
# GTs = ["None",'bent','swapp','crack','crack','missing','missing','hole']
# test_yesno(folder, sub_folder, GTs, model_name)

# folder = "capsule"
# sub_folder = ["good","crack","faulty_imprint","poke","scratch","squeeze"]
# GTs = ["None",'crack','misprint','hole','scratch','misshapen']
# test_yesno(folder, sub_folder, GTs, model_name)

# folder = "carpet"
# sub_folder = ["good","color","cut","hole","metal_contamination","thread"]
# GTs = ["None",'stain','cut','hole','contamination','contamination']
# test_yesno(folder, sub_folder, GTs, model_name)

# folder = "grid"
# sub_folder = ["good","bent","broken","glue","metal_contamination","thread"]
# GTs = ["None","bent","broken","contamination","contamination","contamination"]
# test_yesno(folder, sub_folder, GTs, model_name)

# folder = "hazelnut"
# sub_folder = ["good","crack","cut","hole","print"]
# GTs = ['None','crack','scratch','hole','misprint']
# test_yesno(folder, sub_folder, GTs, model_name)

# folder = "leather"
# sub_folder = ["good","color","cut","fold","glue","poke"]
# GTs = ['None','stain','scratch','wrinkle','stain','hole']
# test_yesno(folder, sub_folder, GTs, model_name)

# folder = "metal_nut"
# sub_folder = ["good","bent","color","flip","scratch"]
# GTs = ['None','bent','stain','flip','scratch']
# test_yesno(folder, sub_folder, GTs, model_name)

# folder = "pill"
# sub_folder = ["good","color","contamination","crack","faulty_imprint","scratch","pill_type"]
# GTs = ['None','stain','contamination','crack','misprint','scratch','stain']
# test_yesno(folder, sub_folder, GTs, model_name)

# folder = "screw"
# sub_folder = ["good","manipulated_front","scratch_head","scratch_neck","thread_side","thread_top"]
# GTs = ['None','strip','chip','chip','chip','chip']
# test_yesno(folder, sub_folder, GTs, model_name)

# folder = "tile"
# sub_folder = ["good","crack","glue_strip","gray_stroke","oil","rough"]
# GTs = ['None','crack','contamination','stain','stain','contamination']
# test_yesno(folder, sub_folder, GTs, model_name)

# folder = "toothbrush"
# sub_folder = ["good","defective"]
# GTs = ["None","broken"]
# test_yesno(folder, sub_folder, GTs, model_name)

# folder = "transistor"
# sub_folder = ["good","bent_lead","cut_lead","damaged_case","misplaced"]
# GTs = ["None","bent","cut","broken","misalignment"]
# test_yesno(folder, sub_folder, GTs, model_name)

# folder = "wood"
# sub_folder = ["good","color","scratch","liquid","hole"]
# GTs = ["None","stain","scratch","stain","hole"]
# test_yesno(folder, sub_folder, GTs, model_name)

# folder = "zipper"
# sub_folder = ["good","broken_teeth","fabric_border","fabric_interior","rough","split_teeth","squeezed_teeth"]
# GTs = ["None","broken","tear","frayed","frayed","misshapen","misshapen"]
# test_yesno(folder, sub_folder, GTs, model_name)




model_name = "AC_newtext/batch128_epoch5_lr-5_AC_new_text"

folder = "bottle"
sub_folder = ["good","broken_large","broken_small","contamination"]
GTs = ['None','broken','broken','contamination']
test_yesno(folder, sub_folder, GTs, model_name)

# folder = "cable"
# sub_folder = ["good","bent_wire","cable_swap","cut_inner_insulation","cut_outer_insulation","missing_cable","missing_wire","poke_insulation"]
# GTs = ["None",'bent','swapp','crack','crack','missing','missing','hole']
# test_yesno(folder, sub_folder, GTs, model_name)

# folder = "capsule"
# sub_folder = ["good","crack","faulty_imprint","poke","scratch","squeeze"]
# GTs = ["None",'crack','misprint','hole','scratch','misshapen']
# test_yesno(folder, sub_folder, GTs, model_name)

# folder = "carpet"
# sub_folder = ["good","color","cut","hole","metal_contamination","thread"]
# GTs = ["None",'stain','cut','hole','contamination','contamination']
# test_yesno(folder, sub_folder, GTs, model_name)

# folder = "grid"
# sub_folder = ["good","bent","broken","glue","metal_contamination","thread"]
# GTs = ["None","bent","broken","contamination","contamination","contamination"]
# test_yesno(folder, sub_folder, GTs, model_name)

# folder = "hazelnut"
# sub_folder = ["good","crack","cut","hole","print"]
# GTs = ['None','crack','scratch','hole','misprint']
# test_yesno(folder, sub_folder, GTs, model_name)

# folder = "leather"
# sub_folder = ["good","color","cut","fold","glue","poke"]
# GTs = ['None','stain','scratch','wrinkle','stain','hole']
# test_yesno(folder, sub_folder, GTs, model_name)

# folder = "metal_nut"
# sub_folder = ["good","bent","color","flip","scratch"]
# GTs = ['None','bent','stain','flip','scratch']
# test_yesno(folder, sub_folder, GTs, model_name)

# folder = "pill"
# sub_folder = ["good","color","contamination","crack","faulty_imprint","scratch","pill_type"]
# GTs = ['None','stain','contamination','crack','misprint','scratch','stain']
# test_yesno(folder, sub_folder, GTs, model_name)

# folder = "screw"
# sub_folder = ["good","manipulated_front","scratch_head","scratch_neck","thread_side","thread_top"]
# GTs = ['None','strip','chip','chip','chip','chip']
# test_yesno(folder, sub_folder, GTs, model_name)

# folder = "tile"
# sub_folder = ["good","crack","glue_strip","gray_stroke","oil","rough"]
# GTs = ['None','crack','contamination','stain','stain','contamination']
# test_yesno(folder, sub_folder, GTs, model_name)

# folder = "toothbrush"
# sub_folder = ["good","defective"]
# GTs = ["None","broken"]
# test_yesno(folder, sub_folder, GTs, model_name)

# folder = "transistor"
# sub_folder = ["good","bent_lead","cut_lead","damaged_case","misplaced"]
# GTs = ["None","bent","cut","broken","misalignment"]
# test_yesno(folder, sub_folder, GTs, model_name)

# folder = "wood"
# sub_folder = ["good","color","scratch","liquid","hole"]
# GTs = ["None","stain","scratch","stain","hole"]
# test_yesno(folder, sub_folder, GTs, model_name)

# folder = "zipper"
# sub_folder = ["good","broken_teeth","fabric_border","fabric_interior","rough","split_teeth","squeezed_teeth"]
# GTs = ["None","broken","tear","frayed","frayed","misshapen","misshapen"]
# test_yesno(folder, sub_folder, GTs, model_name)