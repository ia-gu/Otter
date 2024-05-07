import os
import csv
import json
import torch
import transformers
import textwrap
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

def generate_list_string(items):
    items = [item.replace('_', ' ') for item in items]
    if len(items) == 1:
        return items[0]
    elif len(items) == 2:
        return f"{items[0]} and {items[1]}"
    else:
        return ", ".join(items[:-1]) + f", and {items[-1]}"
    
def test_ok(folder, sub_folder, output_dir, annotations):
    if folder=="grid": folder__ = "metal grid"
    else: folder__ = folder
    folder__ = folder__.replace('_', ' ')
    
    # context images
    demo_image_one = Image.open(f"/data/mvtec/{folder}/test/good/000.png").resize((224, 224)).convert("RGB")
    demo_image_two = Image.open(f"/data/mvtec/{folder}/test/{sub_folder}/000.png").resize((224, 224)).convert("RGB")
    mode_loc_of_demo = ''
    for annotation in annotations[f"/data/mvtec/{folder}/test/{sub_folder}/000.png"]["bboxes"]:
        mode_loc_of_demo += '{defect} on the {loc}'.format(defect=annotation['mode'], loc=annotation['location'])
        mode_loc_of_demo += ', '
    mode_loc_of_demo = mode_loc_of_demo[:-2]
    GTs = []
    for i in annotations:
        if i.startswith(f"/data/mvtec/{folder}/test/"):
            for j in range(len(annotations[i]["bboxes"])):
                GTs.append(annotations[i]["bboxes"][j]['mode'])
    GTs = list(set(GTs))

    # query images
    query_folder_path = f"/data/mvtec/{folder}/test/good"
    query_image_paths = get_image_paths(query_folder_path)
    # input prompt
    subfolder_string = generate_list_string(GTs)
    inputs = textwrap.dedent(f"""
        <image>User: This is an image of {folder__}. Does this {folder__} have any defects such as {subfolder_string}? GPT:<answer> No. This {folder__} does not have any defects such as {subfolder_string}, so it is non-defective.<|endofchunk|>
        <image>User: This is an image of {folder__}. Does this {folder__} have any defects such as {subfolder_string}? GPT:<answer> Yes. This {folder__} has {mode_loc_of_demo}, so it is defective.<|endofchunk|>
        <image>User: This is an image of {folder__}. Does this {folder__} have any defects such as {subfolder_string}? GPT:<answer>
    """)
    inputs = "".join(inputs.split("\n"))
    lang_x = model.text_tokenizer([inputs], return_tensors="pt")
    
    sentence = f"context1: OK, context2: NG, query: OK"
    print(sentence)
    print(inputs)
    sentence = f'-----{sub_folder} start-----'
    
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
        with open(f'./{output_dir}/result.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([query_image_path, parsed_output])
    
    yesno_acc = f"{yesno_count}/{len(query_image_paths)-1}"
    with open(f'./{output_dir}/result.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([yesno_acc])
    with open(f'./{output_dir}/only_result.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([query_image_path.split('/')[5], query_image_path.split('/')[5], yesno_acc])

def test_ng(folder, sub_folder, output_dir, annotations):
    if folder=="grid":
        folder__ = "metal grid"
    else:
        folder__ = folder
    folder__ = folder__.replace('_', ' ')
    
    # context images
    demo_image_one = Image.open(f"/data/mvtec/{folder}/test/good/000.png").resize((224, 224)).convert("RGB")
    demo_image_two = Image.open(f"/data/mvtec/{folder}/test/{sub_folder}/000.png").resize((224, 224)).convert("RGB")
    mode_loc_of_demo = ''
    for annotation in annotations[f"/data/mvtec/{folder}/test/{sub_folder}/000.png"]["bboxes"]:
        mode_loc_of_demo += '{defect} on the {loc}'.format(defect=annotation['mode'], loc=annotation['location'])
        mode_loc_of_demo += ', '
    mode_loc_of_demo = mode_loc_of_demo[:-2]
    GTs = []
    for i in annotations:
        if i.startswith(f"/data/mvtec/{folder}/test/"):
            for j in range(len(annotations[i]["bboxes"])):
                GTs.append(annotations[i]["bboxes"][j]['mode'])
    GTs = list(set(GTs))

    # query images
    query_folder_path = f"/data/mvtec/{folder}/test/{sub_folder}"
    query_image_paths = get_image_paths(query_folder_path)
    
    # input prompt
    subfolder_string = generate_list_string(GTs)

    inputs = textwrap.dedent(f"""
        <image>User: This is an image of {folder__}. Does this {folder__} have any defects such as {subfolder_string}? GPT:<answer> No. This {folder__} does not have any defects such as {subfolder_string}, so it is non-defective.<|endofchunk|>
        <image>User: This is an image of {folder__}. Does this {folder__} have any defects such as {subfolder_string}? GPT:<answer> Yes. This {folder__} has {mode_loc_of_demo}, so it is defective.<|endofchunk|>
        <image>User: This is an image of {folder__}. Does this {folder__} have any defects such as {subfolder_string}? GPT:<answer>
    """)
    inputs = "".join(inputs.split("\n"))
    lang_x = model.text_tokenizer([inputs], return_tensors="pt")
    
    sentence = f"context1: OK, context2: NG, query: NG"
    print(sentence)
    print(inputs)
    sentence = f'-----{sub_folder} start-----'
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
        with open(f'./{output_dir}/result.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([query_image_path, parsed_output])
    
    yesno_acc = f"{yesno_count}/{len(query_image_paths)-1}"
    with open(f'./{output_dir}/result.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([yesno_acc])
    with open(f'./{output_dir}/only_result.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([query_image_path.split('/')[5], query_image_path.split('/')[5], yesno_acc])

model_names = ['../../log/rename_full/4_templates/checkpoint_1.pt', '../../log/rename_full/4_templates/checkpoint_6.pt', '../../log/rename_full/4_templates/checkpoint_11.pt', '../../log/rename_full/4_templates/checkpoint_16.pt', '../../log/rename_full/4_templates/checkpoint_21.pt', '../../log/rename_full/4_templates/checkpoint_26.pt']
output_dirs = ['rename_full/4_templates_1', 'rename_full/4_templates_6', 'rename_full/4_templates_11', 'rename_full/4_templates_16', 'rename_full/4_templates_21', 'rename_full/4_templates_26']

with open('/home/ueno/Otter/Otter_v2/pipeline/demo/mvtec_annotations.json', 'r') as f:
    annotations = json.load(f)

for id in range(len(model_names)):
    # load weight
    trained_ckpt_path = model_names[id]
    train_ckpt = torch.load(trained_ckpt_path, map_location="cpu")
    # model.load_state_dict(train_ckpt, strict=True)
    if train_ckpt.get("model_state_dict", None) is not None:
        train_ckpt = train_ckpt["model_state_dict"]
    _ = model.load_state_dict(train_ckpt, strict=False)

    # log save path
    output_dir = output_dirs[id]
    os.makedirs(f'./{output_dir}', exist_ok=True)
    with open(f'./{output_dir}/result.csv', 'w') as f:
        pass
    with open(f'./{output_dir}/result_category.csv', 'w') as f:
        pass
    with open(f'./{output_dir}/only_result.csv', 'w') as f:
        pass

    folder = "wood"
    ngs = ["color","scratch","liquid","hole"]
    for ng in ngs:
        test_ok(folder, ng, output_dir, annotations)
        test_ng(folder, ng, output_dir, annotations)

    folder = "bottle"
    ngs = ["broken_large","broken_small","contamination"]
    for ng in ngs:  
        test_ok(folder, ng, output_dir, annotations)
        test_ng(folder, ng, output_dir, annotations)

    folder = "cable"
    ngs = ["bent_wire","cable_swap","cut_inner_insulation","cut_outer_insulation","missing_cable","missing_wire","poke_insulation"]
    for ng in ngs:
        test_ok(folder, ng, output_dir, annotations)
        test_ng(folder, ng, output_dir, annotations)

    folder = "capsule"
    ngs = ["crack","faulty_imprint","poke","scratch","squeeze"]
    for ng in ngs:
        test_ok(folder, ng, output_dir, annotations)
        test_ng(folder, ng, output_dir, annotations)

    folder = "carpet"
    ngs = ["color","cut","hole","metal_contamination","thread"]
    for ng in ngs:
        test_ok(folder, ng, output_dir, annotations)
        test_ng(folder, ng, output_dir, annotations)

    folder = "grid"
    ngs = ["bent","broken","glue","metal_contamination","thread"]
    for ng in ngs:
        test_ok(folder, ng, output_dir, annotations)
        test_ng(folder, ng, output_dir, annotations)

    folder = "hazelnut"
    ngs = ["crack","cut","hole","print"]
    for ng in ngs:
        test_ok(folder, ng, output_dir, annotations)
        test_ng(folder, ng, output_dir, annotations)

    folder = "leather"
    ngs = ["color","cut","fold","glue","poke"]
    for ng in ngs:
        test_ok(folder, ng, output_dir, annotations)
        test_ng(folder, ng, output_dir, annotations)

    folder = "metal_nut"
    ngs = ["bent","color","flip","scratch"]
    for ng in ngs:
        test_ok(folder, ng, output_dir, annotations)
        test_ng(folder, ng, output_dir, annotations)

    folder = "pill"
    ngs = ["color","contamination","crack","faulty_imprint","scratch","pill_type"]
    for ng in ngs:
        test_ok(folder, ng, output_dir, annotations)
        test_ng(folder, ng, output_dir, annotations)

    folder = "screw"
    ngs = ["manipulated_front","scratch_head","scratch_neck","thread_side","thread_top"]
    for ng in ngs:
        test_ok(folder, ng, output_dir, annotations)
        test_ng(folder, ng, output_dir, annotations)

    folder = "tile"
    ngs = ["crack","glue_strip","gray_stroke","oil","rough"]
    for ng in ngs:
        test_ok(folder, ng, output_dir, annotations)
        test_ng(folder, ng, output_dir, annotations)

    folder = "toothbrush"
    ngs = ["defective"]
    for ng in ngs:
        test_ok(folder, ng, output_dir, annotations)
        test_ng(folder, ng, output_dir, annotations)

    folder = "transistor"
    ngs = ["bent_lead","cut_lead","damaged_case","misplaced"]
    for ng in ngs:
        test_ok(folder, ng, output_dir, annotations)
        test_ng(folder, ng, output_dir, annotations)

    folder = "zipper"
    ngs = ["broken_teeth","fabric_border","fabric_interior","rough","split_teeth","squeezed_teeth"]
    for ng in ngs:
        test_ok(folder, ng, output_dir, annotations)
        test_ng(folder, ng, output_dir, annotations)