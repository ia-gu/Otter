import os
import csv
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
    
def test_ok(folder, sub_folder, GTs, gt_idx, output_dir):
    if folder=="grid": folder__ = "metal grid"
    else: folder__ = folder
    folder__ = folder__.replace('_', ' ')
    
    # save log
    folder_name = f'./{output_dir}/{folder}/{sub_folder}'
    os.makedirs(folder_name, exist_ok=True)
    log_name = "query_ok.txt"
    with open(f'{folder_name}/{log_name}', mode='w') as f:
        f.close()
    
    # context images
    demo_image_one = Image.open(f"/data/mvtec/{folder}/test/good/000.png").resize((224, 224)).convert("RGB")
    demo_image_two = Image.open(f"/data/mvtec/{folder}/test/{sub_folder}/000.png").resize((224, 224)).convert("RGB")
    # query images
    query_folder_path = f"/data/mvtec/{folder}/test/good"
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
    lang_x = model.text_tokenizer([inputs], return_tensors="pt")
    
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
        write_text_file(f'{folder_name}/{log_name}', query_image_path)
        write_text_file(f'{folder_name}/{log_name}', parsed_output)
        write_text_file(f'{folder_name}/{log_name}', "")
        with open(f'./{output_dir}/result.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([query_image_path, parsed_output])
    
    yesno_acc = f"{yesno_count}/{len(query_image_paths)-1}"
    write_text_file(f'{folder_name}/{log_name}',f'-----{sub_folder} end-----')
    write_text_file(f'{folder_name}/{log_name}',yesno_acc)
    with open(f'./{output_dir}/result.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([yesno_acc])
    with open(f'./{output_dir}/only_result.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([query_image_path.split('/')[5], query_image_path.split('/')[5], yesno_acc])

def test_ng(folder, sub_folder, GTs, gt_idx, output_dir):
    if folder=="grid":
        folder__ = "metal grid"
    else:
        folder__ = folder
    folder__ = folder__.replace('_', ' ')
    
    # save log
    folder_name = f'./{output_dir}/{folder}/{sub_folder}'
    os.makedirs(folder_name, exist_ok=True)
    log_name = "query_ng.txt"
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
        with open(f'./{output_dir}/result.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([query_image_path, parsed_output])
    
    yesno_acc = f"{yesno_count}/{len(query_image_paths)-1}"
    write_text_file(f'{folder_name}/{log_name}',f'-----{sub_folder} end-----')
    write_text_file(f'{folder_name}/{log_name}',yesno_acc)
    with open(f'./{output_dir}/result.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([yesno_acc])
    with open(f'./{output_dir}/only_result.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([query_image_path.split('/')[5], query_image_path.split('/')[5], yesno_acc])


def test_category(folder, sub_folder, output_dir):
    if folder=="grid":
        folder__ = "metal grid"
    else:
        folder__ = folder
    folder__ = folder__.replace('_', ' ')

    # acc = []
    folder_name = f'./{output_dir}/{folder}/{sub_folder}'
    os.makedirs(folder_name, exist_ok=True)
    log_name = "category_guess.txt"
    with open(f'{folder_name}/{log_name}', mode='w') as f:
        f.close()

    query_folder_path = f"/data/mvtec/{folder}/test/{sub_folder}"
    query_image_paths = get_image_paths(query_folder_path)
    inputs = textwrap.dedent(f"""<image> User: What does this image describe? GPT:<answer>""")
    inputs = "".join(inputs.split("\n"))
    lang_x = model.text_tokenizer([inputs], return_tensors="pt")
    write_text_file(f'{folder_name}/{log_name}', f'-----{sub_folder} start-----')
    write_text_file(f'{folder_name}/{log_name}', "")
    write_text_file(f'{folder_name}/{log_name}', f'{inputs}')
    write_text_file(f'{folder_name}/{log_name}', "")
        
    count = 0
    for i, query_image_path in enumerate(query_image_paths[1:]):
        # print(query_image_path)
        query_image = Image.open(query_image_path).resize((224, 224)).convert("RGB")
        vision_x = image_processor.preprocess([query_image], return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0)
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
        
        if parsed_output.lower()==folder.lower():
            count += 1
                
        write_text_file(f'{folder_name}/{log_name}',query_image_path)
        write_text_file(f'{folder_name}/{log_name}',parsed_output)
        write_text_file(f'{folder_name}/{log_name}',"")
        with open(f'./{output_dir}/result_category.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([query_image_path, parsed_output])

    accuracy = f"correct: {count}, total: {len(query_image_paths)}, acc: {(count / (len(query_image_paths))) * 100:.2f}%"
    # acc.append((sub,accuracy))
    
    write_text_file(f'{folder_name}/{log_name}',f'-----{sub_folder} end-----')
    write_text_file(f'{folder_name}/{log_name}',accuracy)


model_names = ['../../log/rename_view_20/json_based_20_rand/checkpoint_1.pt', '../../log/rename_view_20/json_based_20_rand/checkpoint_6.pt', '../../log/rename_view_20/json_based_20_rand/checkpoint_11.pt', '../../log/rename_view_20/json_based_20_rand/checkpoint_16.pt', '../../log/rename_view_20/json_based_20_rand/checkpoint_21.pt', '../../log/rename_view_20/json_based_20_rand/checkpoint_26.pt']
# model_names = ['../../log/prompt_comparison/simple_icl/checkpoint_1.pt', '../../log/prompt_comparison/simple_icl/checkpoint_6.pt', '../../log/prompt_comparison/simple_icl/checkpoint_11.pt']
output_dirs = ['rename_view_20/json_based_20_rand_1', 'rename_view_20/json_based_20_rand_6', 'rename_view_20/json_based_20_rand_11', 'rename_view_20/json_based_20_rand_16', 'rename_view_20/json_based_20_rand_21', 'rename_view_20/json_based_20_rand_26']
# output_dirs = ['prompt_comparison/simple_icl_1', 'prompt_comparison/simple_icl_6', 'prompt_comparison/simple_icl_11']
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
    # output_dir = "vi_full_simple"

    folder = "wood"
    ngs = ["color","scratch","liquid","hole"]
    GTs = ["stain","scratch","hole"]
    # GTs = ["stained wood","scratched wood","wood with holes"]
    test_ok(folder, ngs[0], GTs, 0, output_dir)
    test_ng(folder, ngs[0], GTs, 0, output_dir)
    test_ok(folder, ngs[1], GTs, 1, output_dir)
    test_ng(folder, ngs[1], GTs, 1, output_dir)
    test_ok(folder, ngs[2], GTs, 0, output_dir)
    test_ng(folder, ngs[2], GTs, 0, output_dir)
    test_ok(folder, ngs[3], GTs, 2, output_dir)
    test_ng(folder, ngs[3], GTs, 2, output_dir)
    # test_category(folder, 'good', output_dir)
    # for ng in ngs:
    #     test_category(folder, ng, output_dir)

    folder = "bottle"
    ngs = ["broken_large","broken_small","contamination"]
    GTs = ['broken','contamination']
    # GTs = ['broken bottle','bottle with contamination']
    test_ok(folder, ngs[0], GTs, 0, output_dir)
    test_ng(folder, ngs[0], GTs, 0, output_dir)
    test_ok(folder, ngs[1], GTs, 0, output_dir)
    test_ng(folder, ngs[1], GTs, 0, output_dir)
    test_ok(folder, ngs[2], GTs, 1, output_dir)
    test_ng(folder, ngs[2], GTs, 1, output_dir)
    # test_category(folder, 'good', output_dir)
    # for ng in ngs:
    #     test_category(folder, ng, output_dir)

    folder = "cable"
    ngs = ["bent_wire","cable_swap","cut_inner_insulation","cut_outer_insulation","missing_cable","missing_wire","poke_insulation"]
    GTs = ['bent','swapp','crack','missing','hole']
    # GTs = ['bent cable','swapped cable','cracked cable','missing cable','cable with holes']
    test_ok(folder, ngs[0], GTs, 0, output_dir)
    test_ng(folder, ngs[0], GTs, 0, output_dir)
    test_ok(folder, ngs[1], GTs, 1, output_dir)
    test_ng(folder, ngs[1], GTs, 1, output_dir)
    test_ok(folder, ngs[2], GTs, 2, output_dir)
    test_ng(folder, ngs[2], GTs, 2, output_dir)
    test_ok(folder, ngs[3], GTs, 2, output_dir)
    test_ng(folder, ngs[3], GTs, 2, output_dir)
    test_ok(folder, ngs[4], GTs, 3, output_dir)
    test_ng(folder, ngs[4], GTs, 3, output_dir)
    test_ok(folder, ngs[5], GTs, 3, output_dir)
    test_ng(folder, ngs[5], GTs, 3, output_dir)
    test_ok(folder, ngs[6], GTs, 4, output_dir)
    test_ng(folder, ngs[6], GTs, 4, output_dir)
    # test_category(folder, 'good', output_dir)
    # for ng in ngs:
    #     test_category(folder, ng, output_dir)

    folder = "capsule"
    ngs = ["crack","faulty_imprint","poke","scratch","squeeze"]
    GTs = ['crack','misprint','hole','scratch','misshapen']
    # GTs = ['cracked capsule','misprinted capsule','capsule with holes','scratched capsule', 'misshapen capsule']
    test_ok(folder, ngs[0], GTs, 0, output_dir)
    test_ng(folder, ngs[0], GTs, 0, output_dir)
    test_ok(folder, ngs[1], GTs, 1, output_dir)
    test_ng(folder, ngs[1], GTs, 1, output_dir)
    test_ok(folder, ngs[2], GTs, 2, output_dir)
    test_ng(folder, ngs[2], GTs, 2, output_dir)
    test_ok(folder, ngs[3], GTs, 3, output_dir)
    test_ng(folder, ngs[3], GTs, 3, output_dir)
    test_ok(folder, ngs[4], GTs, 4, output_dir)
    test_ng(folder, ngs[4], GTs, 4, output_dir)
    # test_category(folder, 'good', output_dir)
    # for ng in ngs:
    #     test_category(folder, ng, output_dir)

    folder = "carpet"
    ngs = ["color","cut","hole","metal_contamination","thread"]
    GTs = ['stain','cut','hole','contamination']
    # GTs = ['stained carpet','cut carpet','carpet with holes','carpet with contamination']
    test_ok(folder, ngs[0], GTs, 0, output_dir)
    test_ng(folder, ngs[0], GTs, 0, output_dir)
    test_ok(folder, ngs[1], GTs, 1, output_dir)
    test_ng(folder, ngs[1], GTs, 1, output_dir)
    test_ok(folder, ngs[2], GTs, 2, output_dir)
    test_ng(folder, ngs[2], GTs, 2, output_dir)
    test_ok(folder, ngs[3], GTs, 3, output_dir)
    test_ng(folder, ngs[3], GTs, 3, output_dir)
    test_ok(folder, ngs[4], GTs, 3, output_dir)
    test_ng(folder, ngs[4], GTs, 3, output_dir)
    # test_category(folder, 'good', output_dir)
    # for ng in ngs:
    #     test_category(folder, ng, output_dir)

    folder = "grid"
    ngs = ["bent","broken","glue","metal_contamination","thread"]
    GTs = ['bent','broken','contamination']
    # GTs = ['bent metal grid','broken metal grid','metal grid with contamination']
    test_ok(folder, ngs[0], GTs, 0, output_dir)
    test_ng(folder, ngs[0], GTs, 0, output_dir)
    test_ok(folder, ngs[1], GTs, 1, output_dir)
    test_ng(folder, ngs[1], GTs, 1, output_dir)
    test_ok(folder, ngs[2], GTs, 2, output_dir)
    test_ng(folder, ngs[2], GTs, 2, output_dir)
    test_ok(folder, ngs[3], GTs, 2, output_dir)
    test_ng(folder, ngs[3], GTs, 2, output_dir)
    test_ok(folder, ngs[4], GTs, 2, output_dir)
    test_ng(folder, ngs[4], GTs, 2, output_dir)
    # test_category(folder, 'good', output_dir)
    # for ng in ngs:
    #     test_category(folder, ng, output_dir)

    folder = "hazelnut"
    ngs = ["crack","cut","hole","print"]
    GTs = ['crack','scratch','hole','misprint']
    # GTs = ['cracked hazelnut','scratched hazelnut','hazelnut with holes','hazelnut with white marks']
    test_ok(folder, ngs[0], GTs, 0, output_dir)
    test_ng(folder, ngs[0], GTs, 0, output_dir)
    test_ok(folder, ngs[1], GTs, 1, output_dir)
    test_ng(folder, ngs[1], GTs, 1, output_dir)
    test_ok(folder, ngs[2], GTs, 2, output_dir)
    test_ng(folder, ngs[2], GTs, 2, output_dir)
    test_ok(folder, ngs[3], GTs, 3, output_dir)
    test_ng(folder, ngs[3], GTs, 3, output_dir)
    # test_category(folder, 'good', output_dir)
    # for ng in ngs:
    #     test_category(folder, ng, output_dir)

    folder = "leather"
    ngs = ["color","cut","fold","glue","poke"]
    GTs = ['stain','scratch','wrinkle','hole']
    # GTs = ['stained leather','cut leather','wrinkle leather','leather with holes']
    test_ok(folder, ngs[0], GTs, 0, output_dir)
    test_ng(folder, ngs[0], GTs, 0, output_dir)
    test_ok(folder, ngs[1], GTs, 1, output_dir)
    test_ng(folder, ngs[1], GTs, 1, output_dir)
    test_ok(folder, ngs[2], GTs, 2, output_dir)
    test_ng(folder, ngs[2], GTs, 2, output_dir)
    test_ok(folder, ngs[3], GTs, 0, output_dir)
    test_ng(folder, ngs[3], GTs, 0, output_dir)
    test_ok(folder, ngs[4], GTs, 3, output_dir)
    test_ng(folder, ngs[4], GTs, 3, output_dir)
    # test_category(folder, 'good', output_dir)
    # for ng in ngs:
    #     test_category(folder, ng, output_dir)

    folder = "metal_nut"
    ngs = ["bent","color","flip","scratch"]
    GTs = ['bent','stain','flip','scratch']
    # GTs = ['bent metal nut','stained metal nut','flipped metal nut','scratched metal nut']
    test_ok(folder, ngs[0], GTs, 0, output_dir)
    test_ng(folder, ngs[0], GTs, 0, output_dir)
    test_ok(folder, ngs[1], GTs, 1, output_dir)
    test_ng(folder, ngs[1], GTs, 1, output_dir)
    test_ok(folder, ngs[2], GTs, 2, output_dir)
    test_ng(folder, ngs[2], GTs, 2, output_dir)
    test_ok(folder, ngs[3], GTs, 3, output_dir)
    test_ng(folder, ngs[3], GTs, 3, output_dir)
    # test_category(folder, 'good', output_dir)
    # for ng in ngs:
    #     test_category(folder, ng, output_dir)

    folder = "pill"
    ngs = ["color","contamination","crack","faulty_imprint","scratch","pill_type"]
    GTs = ['stain','contamination','crack','misprint','scratch']
    # GTs = ['stained pill','pill with contamination','cracked pill','misprinted pill','scratched pill']
    test_ok(folder, ngs[0], GTs, 0, output_dir)
    test_ng(folder, ngs[0], GTs, 0, output_dir)
    test_ok(folder, ngs[1], GTs, 1, output_dir)
    test_ng(folder, ngs[1], GTs, 1, output_dir)
    test_ok(folder, ngs[2], GTs, 2, output_dir)
    test_ng(folder, ngs[2], GTs, 2, output_dir)
    test_ok(folder, ngs[3], GTs, 3, output_dir)
    test_ng(folder, ngs[3], GTs, 3, output_dir)
    test_ok(folder, ngs[4], GTs, 4, output_dir)
    test_ng(folder, ngs[4], GTs, 4, output_dir)
    test_ok(folder, ngs[5], GTs, 0, output_dir)
    test_ng(folder, ngs[5], GTs, 0, output_dir)
    # test_category(folder, 'good', output_dir)
    # for ng in ngs:
    #     test_category(folder, ng, output_dir)

    folder = "screw"
    ngs = ["manipulated_front","scratch_head","scratch_neck","thread_side","thread_top"]
    GTs = ['strip','chip']
    # GTs = ['stripped screw','chipped screw']
    test_ok(folder, ngs[0], GTs, 0, output_dir)
    test_ng(folder, ngs[0], GTs, 0, output_dir)
    test_ok(folder, ngs[1], GTs, 1, output_dir)
    test_ng(folder, ngs[1], GTs, 1, output_dir)
    test_ok(folder, ngs[2], GTs, 1, output_dir)
    test_ng(folder, ngs[2], GTs, 1, output_dir)
    test_ok(folder, ngs[3], GTs, 1, output_dir)
    test_ng(folder, ngs[3], GTs, 1, output_dir)
    test_ok(folder, ngs[4], GTs, 1, output_dir)
    test_ng(folder, ngs[4], GTs, 1, output_dir)
    # test_category(folder, 'good', output_dir)
    # for ng in ngs:
    #     test_category(folder, ng, output_dir)

    folder = "tile"
    ngs = ["crack","glue_strip","gray_stroke","oil","rough"]
    GTs = ['crack','contamination','stain']
    # GTs = ['cracked tile','tile with contamination','stained tile']
    test_ok(folder, ngs[0], GTs, 0, output_dir)
    test_ng(folder, ngs[0], GTs, 0, output_dir)
    test_ok(folder, ngs[1], GTs, 1, output_dir)
    test_ng(folder, ngs[1], GTs, 1, output_dir)
    test_ok(folder, ngs[2], GTs, 2, output_dir)
    test_ng(folder, ngs[2], GTs, 2, output_dir)
    test_ok(folder, ngs[3], GTs, 2, output_dir)
    test_ng(folder, ngs[3], GTs, 2, output_dir)
    test_ok(folder, ngs[4], GTs, 2, output_dir)
    test_ng(folder, ngs[4], GTs, 2, output_dir)
    # test_category(folder, 'good', output_dir)
    # for ng in ngs:
    #     test_category(folder, ng, output_dir)

    folder = "toothbrush"
    ngs = ["defective"]
    GTs = ['broken']
    # GTs = ['broken toothbrush']
    test_ok(folder, ngs[0], GTs, 0, output_dir)
    test_ng(folder, ngs[0], GTs, 0, output_dir)

    folder = "transistor"
    ngs = ["bent_lead","cut_lead","damaged_case","misplaced"]
    GTs = ['bent','cut','broken','misalignment']
    # GTs = ['bent transistor','cut transistor','broken transistor','misplaced transistor']
    test_ok(folder, ngs[0], GTs, 0, output_dir)
    test_ng(folder, ngs[0], GTs, 0, output_dir)
    test_ok(folder, ngs[1], GTs, 1, output_dir)
    test_ng(folder, ngs[1], GTs, 1, output_dir)
    test_ok(folder, ngs[2], GTs, 2, output_dir)
    test_ng(folder, ngs[2], GTs, 2, output_dir)
    test_ok(folder, ngs[3], GTs, 3, output_dir)
    test_ng(folder, ngs[3], GTs, 3, output_dir)
    # test_category(folder, 'good', output_dir)
    # for ng in ngs:
    #     test_category(folder, ng, output_dir)

    folder = "zipper"
    ngs = ["broken_teeth","fabric_border","fabric_interior","rough","split_teeth","squeezed_teeth"]
    GTs = ['broken','tear','frayed','misshapen']
    # GTs = ['broken zipper','torn zipper','frayed zipper','misshapen zipper']
    test_ok(folder, ngs[0], GTs, 0, output_dir)
    test_ng(folder, ngs[0], GTs, 0, output_dir)
    test_ok(folder, ngs[1], GTs, 1, output_dir)
    test_ng(folder, ngs[1], GTs, 1, output_dir)
    test_ok(folder, ngs[2], GTs, 2, output_dir)
    test_ng(folder, ngs[2], GTs, 2, output_dir)
    test_ok(folder, ngs[3], GTs, 2, output_dir)
    test_ng(folder, ngs[3], GTs, 2, output_dir)
    test_ok(folder, ngs[4], GTs, 3, output_dir)
    test_ng(folder, ngs[4], GTs, 3, output_dir)
    test_ok(folder, ngs[5], GTs, 3, output_dir)
    test_ng(folder, ngs[5], GTs, 3, output_dir)
    # test_category(folder, 'good', output_dir)
    # for ng in ngs:
    #     test_category(folder, ng, output_dir)