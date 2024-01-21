import os
from PIL import Image
import os
import textwrap

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
    

def test_context(folder, sub_folder, GTs, model, model_name, image_processor, order=True):
    acc = []
    if folder=="grid":
        folder__ = "metal grid"
    else:
        folder__ = folder
    folder__ = folder__.replace('_', ' ')
    for sub,gt in zip(sub_folder,GTs):
        subfolder_string = generate_list_string(GTs)
        model.text_tokenizer.padding_side = "left"
        sentence = f"{sub} --> {gt}"
        print(sentence)
        # write_text_file(f'{folder_name}/detective.txt',sentence)
        
        """ クエリ：不良品 """
        if order: # demo_image_one: 良品, demo_image_two: 不良品
            sentence = f"context1: OK, context2: NG, query: NG"
            # write_text_file(f'{folder_name}/detective.txt',sentence)
            demo_image_one = Image.open(f"/data/dataset/mvtec/{folder}/test/good/000.png").resize((224, 224)).convert("RGB")
            demo_image_two = Image.open(f"/data/dataset/mvtec/{folder}/test/{sub}/000.png").resize((224, 224)).convert("RGB")
            
            # inputs = textwrap.dedent(f"""
            #     <image>User: This is an image of {folder__}. Does this {folder__} have any defects such as {subfolder_string}? If there are any defects, please provide the defect name. If not, please say None. GPT:<answer>No None<|endofchunk|>
            #     <image>User: This is an image of {folder__}. Does this {folder__} have any defects such as {subfolder_string}? If there are any defects, please provide the defect name. If not, please say None. GPT:<answer>Yes {gt}<|endofchunk|>
            #     <image>User: This is an image of {folder__}. Does this {folder__} have any defects such as {subfolder_string}? If there are any defects, please provide the defect name. If not, please say None. GPT:<answer>
            # """)
            inputs = textwrap.dedent(f"""
                <image>User: This is an image of {folder__}. Does this {folder__} have any defects such as {subfolder_string}? GPT:<answer>No. This {folder__} does not have any defects such as {subfolder_string}, so it is non-defective.<|endofchunk|>
                <image>User: This is an image of {folder__}. Does this {folder__} have any defects such as {subfolder_string}? GPT:<answer>Yes. This {folder__} has some {gt}, so it is defective.<|endofchunk|>
                <image>User: This is an image of {folder__}. Does this {folder__} have any defects such as {subfolder_string}? GPT:<answer>
            """)
        
        else: # demo_image_one: 不良品, demo_image_two: 良品
            sentence = f"context1: NG, context2: OK, query: NG"
            # write_text_file(f'{folder_name}/detective.txt',sentence)
            demo_image_one = Image.open(f"/data/dataset/mvtec/{folder}/test/{sub}/000.png").resize((224, 224)).convert("RGB")
            demo_image_two = Image.open(f"/data/dataset/mvtec/{folder}/test/good/000.png").resize((224, 224)).convert("RGB")
            
            # inputs = textwrap.dedent(f"""
            #     <image>User: This is an image of {folder__}. Does this {folder__} have any defects such as {subfolder_string}? If there are any defects, please provide the defect name. If not, please say None. GPT:<answer>Yes {gt}<|endofchunk|>
            #     <image>User: This is an image of {folder__}. Does this {folder__} have any defects such as {subfolder_string}? If there are any defects, please provide the defect name. If not, please say None. GPT:<answer>No None<|endofchunk|>
            #     <image>User: This is an image of {folder__}. Does this {folder__} have any defects such as {subfolder_string}? If there are any defects, please provide the defect name. If not, please say None. GPT:<answer>
            # """)
            inputs = textwrap.dedent(f"""
                <image>User: This is an image of {folder__}. Does this {folder__} have any defects? GPT:<answer>Yes. This {folder__} has some {gt}, so it is defective.<|endofchunk|>
                <image>User: This is an image of {folder__}. Does this {folder__} have any defects? GPT:<answer>No. This {folder__} does not have any defects, so it is non-defective.<|endofchunk|>
                <image>User: This is an image of {folder__}. Does this {folder__} have any defects? GPT:<answer>
            """)
        
        inputs = "".join(inputs.split("\n"))
        lang_x = model.text_tokenizer(
            [
                inputs
            ],
            return_tensors="pt",
        )
        
        # write_text_file(f'{folder_name}/detective.txt',f'-----{sub} start-----')
        # write_text_file(f'{folder_name}/detective.txt',"")
            
        query_folder_path = f"/data/dataset/mvtec/{folder}/test/{sub}"
        query_image_paths = get_image_paths(query_folder_path)
        yesno_count = 0
        reason_count = 0
        both_count = 0
        for i, query_image_path in enumerate(query_image_paths[1:]):
            # print(query_image_path)
            query_image = Image.open(query_image_path).resize((224, 224)).convert("RGB")
            vision_x = image_processor.preprocess([demo_image_one, demo_image_two, query_image], return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0)
        
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
            
            # print(inputs)
            # print(parsed_output)
            
            if parsed_output.split(".")[0].lower()=="yes":
                yesno_count += 1
                if len(parsed_output.split(" ")) > 1:
                    if parsed_output.split(" ")[1].lower()==f"{gt}":
                        both_count += 1
            if len(parsed_output.split(" ")) > 1:
                if parsed_output.split(" ")[1].lower()==f"{gt}":
                        reason_count += 1
                    
            # write_text_file(f'{folder_name}/detective.txt',query_image_path)
            # write_text_file(f'{folder_name}/detective.txt',parsed_output)
            # write_text_file(f'{folder_name}/detective.txt',"")
            
            print(inputs)
            print("Generated text: ", parsed_output)
            
        yesno_acc = f"correct: {yesno_count}, total: {len(query_image_paths)-1}, yesno acc: {(yesno_count / (len(query_image_paths)-1)) * 100:.2f}%"
        # print(yesno_acc)
        reason_acc = f"correct: {reason_count}, total: {len(query_image_paths)-1}, reason acc: {(reason_count / (len(query_image_paths)-1)) * 100:.2f}%"
        # print(reason_acc)
        both_acc = f"correct: {both_count}, total: {len(query_image_paths)-1}, both acc: {(both_count / (len(query_image_paths)-1)) * 100:.2f}%"
        # print(both_acc)
        acc.append((sub,yesno_acc))
        acc.append((sub,reason_acc))
        acc.append((sub,both_acc))
        print(yesno_acc)
        
        """ クエリ：良品 """
        sentence = f"{sub} --> {gt}"
        print("good")
        print(sentence)
        # write_text_file(f'{folder_name}/non-detective.txt',sentence)
        if order: # demo_image_one: 良品, demo_image_two: 不良品
            sentence = f"context1: OK, context2: NG, query: OK"
            # write_text_file(f'{folder_name}/non-detective.txt',sentence)
        
        else: # demo_image_one: 不良品, demo_image_two: 良品
            sentence = f"context1: NG, context2: OK, query: OK"
        #     write_text_file(f'{folder_name}/non-detective.txt',sentence)
        
        query_folder_path = f"/data/dataset/mvtec/{folder}/test/good"
        query_image_paths = get_image_paths(query_folder_path)
        yesno_count = 0
        reason_count = 0
        both_count = 0
        for i, query_image_path in enumerate(query_image_paths[1:]):
            # print(query_image_path)
            query_image = Image.open(query_image_path).resize((224, 224)).convert("RGB")
            vision_x = image_processor.preprocess([demo_image_one, demo_image_two, query_image], return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0)
        
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
            
            # print(inputs)
            # print(parsed_output)
            
            if parsed_output.split(".")[0].lower()=="no":
                yesno_count += 1
                if len(parsed_output.split(" ")) > 1:
                    if parsed_output.split(" ")[1].lower()=="none":
                        both_count += 1
            if len(parsed_output.split(" ")) > 1:
                if parsed_output.split(" ")[1].lower()=="none":
                        reason_count += 1
                
            # write_text_file(f'{folder_name}/non-detective.txt',query_image_path)
            # write_text_file(f'{folder_name}/non-detective.txt',parsed_output)
            # write_text_file(f'{folder_name}/non-detective.txt',"")
            
            print(inputs)
            print("Generated text: ", parsed_output)
            
        yesno_acc = f"correct: {yesno_count}, total: {len(query_image_paths)-1}, yesno acc: {(yesno_count / (len(query_image_paths)-1)) * 100:.2f}%"
        # print(yesno_acc)
        reason_acc = f"correct: {reason_count}, total: {len(query_image_paths)-1}, reason acc: {(reason_count / (len(query_image_paths)-1)) * 100:.2f}%"
        # print(reason_acc)
        both_acc = f"correct: {both_count}, total: {len(query_image_paths)-1}, both acc: {(both_count / (len(query_image_paths)-1)) * 100:.2f}%"
        # print(both_acc)
        acc.append(("good",yesno_acc))
        acc.append(("good",reason_acc))
        acc.append(("good",both_acc))
        print(yesno_acc)
        
        
    # for a in acc:
    #     print(a)

#     acc = []
#     if folder=="grid":
#         folder__ = "metal grid"
#     else:
#         folder__ = folder
#     folder__ = folder__.replace('_', ' ')
#     for sub,gt in zip(sub_folder,GTs):
#         folder_name = f'./result/{folder}/{sub}/{model_name}'
#         os.makedirs(folder_name, exist_ok=True)
#         with open(f'{folder_name}/detective.txt', mode='w') as f:
#             f.close()
#         with open(f'{folder_name}/non-detective.txt', mode='w') as f:
#             f.close()
        
#         subfolder_string = generate_list_string(GTs)
#         model.text_tokenizer.padding_side = "left"
#         sentence = f"{sub} --> {gt}"
#         write_text_file(f'{folder_name}/detective.txt',sentence)
        
#         """ クエリ：不良品 """
#         if order: # demo_image_one: 良品, demo_image_two: 不良品
#             sentence = f"context1: OK, context2: NG, query: NG"
#             write_text_file(f'{folder_name}/detective.txt',sentence)
#             demo_image_one = Image.open(f"/data/data/mvtec/{folder}/test/good/000.png").resize((224, 224)).convert("RGB")
#             demo_image_two = Image.open(f"/data/data/mvtec/{folder}/test/{sub}/000.png").resize((224, 224)).convert("RGB")
            
#             inputs = textwrap.dedent(f"""
#                 <image>User: This is an image of {folder__}. Does this {folder__} have any defects such as {subfolder_string}? If there are any defects, please provide the defect name. If not, please say None. GPT:<answer>No None<|endofchunk|>
#                 <image>User: This is an image of {folder__}. Does this {folder__} have any defects such as {subfolder_string}? If there are any defects, please provide the defect name. If not, please say None. GPT:<answer>Yes {gt}<|endofchunk|>
#                 <image>User: This is an image of {folder__}. Does this {folder__} have any defects such as {subfolder_string}? If there are any defects, please provide the defect name. If not, please say None. GPT:<answer>
#             """)
        
#         else: # demo_image_one: 不良品, demo_image_two: 良品
#             sentence = f"context1: NG, context2: OK, query: NG"
#             write_text_file(f'{folder_name}/detective.txt',sentence)
#             demo_image_one = Image.open(f"/data/data/mvtec/{folder}/test/{sub}/000.png").resize((224, 224)).convert("RGB")
#             demo_image_two = Image.open(f"/data/data/mvtec/{folder}/test/good/000.png").resize((224, 224)).convert("RGB")
            
#             inputs = textwrap.dedent(f"""
#                 <image>User: This is an image of {folder__}. Does this {folder__} have any defects such as {subfolder_string}? If there are any defects, please provide the defect name. If not, please say None. GPT:<answer>Yes {gt}<|endofchunk|>
#                 <image>User: This is an image of {folder__}. Does this {folder__} have any defects such as {subfolder_string}? If there are any defects, please provide the defect name. If not, please say None. GPT:<answer>No None<|endofchunk|>
#                 <image>User: This is an image of {folder__}. Does this {folder__} have any defects such as {subfolder_string}? If there are any defects, please provide the defect name. If not, please say None. GPT:<answer>
#             """)
        
#         inputs = "".join(inputs.split("\n"))
#         lang_x = model.text_tokenizer(
#             [
#                 inputs
#             ],
#             return_tensors="pt",
#         )
        
#         write_text_file(f'{folder_name}/detective.txt',f'-----{sub} start-----')
#         write_text_file(f'{folder_name}/detective.txt',"")
            
#         query_folder_path = f"/data/data/mvtec/{folder}/test/{sub}"
#         query_image_paths = get_image_paths(query_folder_path)
#         yesno_count = 0
#         reason_count = 0
#         both_count = 0
#         for i, query_image_path in enumerate(query_image_paths[1:]):
#             # print(query_image_path)
#             query_image = Image.open(query_image_path).resize((224, 224)).convert("RGB")
#             vision_x = image_processor.preprocess([demo_image_one, demo_image_two, query_image], return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0)
        
#             # Get the data type from model's parameters
#             model_dtype = next(model.parameters()).dtype

#             # Convert tensors to the model's data type
#             vision_x = vision_x.to(dtype=model_dtype)
#             lang_x_input_ids = lang_x["input_ids"]
#             lang_x_attention_mask = lang_x["attention_mask"]

#             bad_words_id = model.text_tokenizer(["User:", "GPT1:", "GFT:", "GPT:"], add_special_tokens=False).input_ids
#             generated_text = model.generate(
#                 vision_x=vision_x.to(model.device),
#                 lang_x=lang_x_input_ids.to(model.device),
#                 attention_mask=lang_x_attention_mask.to(model.device),
#                 max_new_tokens=512,
#                 num_beams=3,
#                 no_repeat_ngram_size=3,
#                 bad_words_ids=bad_words_id,
#             )

#             parsed_output = (
#                 model.text_tokenizer.decode(generated_text[0]).split("<answer>")[-1].lstrip().rstrip().split("<|endofchunk|>")[0].lstrip().rstrip().lstrip('"').rstrip('"')
#             )
            
#             print(inputs)
#             print(parsed_output)
            
#             if parsed_output.split(" ")[0].lower()=="yes":
#                 yesno_count += 1
#                 if len(parsed_output.split(" ")) > 1:
#                     if parsed_output.split(" ")[1].lower()==f"{gt}":
#                         both_count += 1
#             if len(parsed_output.split(" ")) > 1:
#                 if parsed_output.split(" ")[1].lower()==f"{gt}":
#                         reason_count += 1
                    
#             write_text_file(f'{folder_name}/detective.txt',query_image_path)
#             write_text_file(f'{folder_name}/detective.txt',parsed_output)
#             write_text_file(f'{folder_name}/detective.txt',"")
            
#             # print(inputs)
#             # print("GPT:", parsed_output)
            
#             # fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10,5))
#             # axes[0].imshow(demo_image_one)
#             # axes[0].axis('off')
#             # axes[1].imshow(demo_image_two)
#             # axes[1].axis('off')
#             # axes[2].imshow(query_image)
#             # axes[2].axis('off')
#             # plt.show()
#             print(i,sub)
#             clear_output(wait=True)
            
#         yesno_acc = f"correct: {yesno_count}, total: {len(query_image_paths)-1}, yesno acc: {(yesno_count / (len(query_image_paths)-1)) * 100:.2f}%"
#         # print(yesno_acc)
#         reason_acc = f"correct: {reason_count}, total: {len(query_image_paths)-1}, reason acc: {(reason_count / (len(query_image_paths)-1)) * 100:.2f}%"
#         # print(reason_acc)
#         both_acc = f"correct: {both_count}, total: {len(query_image_paths)-1}, both acc: {(both_count / (len(query_image_paths)-1)) * 100:.2f}%"
#         # print(both_acc)
#         acc.append((sub,yesno_acc))
#         acc.append((sub,reason_acc))
#         acc.append((sub,both_acc))
        
#         write_text_file(f'{folder_name}/detective.txt',f'-----{sub} end-----')
#         write_text_file(f'{folder_name}/detective.txt',yesno_acc)
#         write_text_file(f'{folder_name}/detective.txt',reason_acc)
#         write_text_file(f'{folder_name}/detective.txt',both_acc)
        
        
#         """ クエリ：良品 """
#         sentence = f"{sub} --> {gt}"
#         write_text_file(f'{folder_name}/non-detective.txt',sentence)
#         if order: # demo_image_one: 良品, demo_image_two: 不良品
#             sentence = f"context1: OK, context2: NG, query: OK"
#             write_text_file(f'{folder_name}/non-detective.txt',sentence)
        
#         else: # demo_image_one: 不良品, demo_image_two: 良品
#             sentence = f"context1: NG, context2: OK, query: OK"
#             write_text_file(f'{folder_name}/non-detective.txt',sentence)
        
#         write_text_file(f'{folder_name}/non-detective.txt',f'-----{sub} start-----')
#         write_text_file(f'{folder_name}/non-detective.txt',"")
            
#         query_folder_path = f"/data/data/mvtec/{folder}/test/good"
#         query_image_paths = get_image_paths(query_folder_path)
#         yesno_count = 0
#         reason_count = 0
#         both_count = 0
#         for i, query_image_path in enumerate(query_image_paths[1:]):
#             # print(query_image_path)
#             query_image = Image.open(query_image_path).resize((224, 224)).convert("RGB")
#             vision_x = image_processor.preprocess([demo_image_one, demo_image_two, query_image], return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0)
        
#             # Get the data type from model's parameters
#             model_dtype = next(model.parameters()).dtype

#             # Convert tensors to the model's data type
#             vision_x = vision_x.to(dtype=model_dtype)
#             lang_x_input_ids = lang_x["input_ids"]
#             lang_x_attention_mask = lang_x["attention_mask"]

#             bad_words_id = model.text_tokenizer(["User:", "GPT1:", "GFT:", "GPT:"], add_special_tokens=False).input_ids
#             generated_text = model.generate(
#                 vision_x=vision_x.to(model.device),
#                 lang_x=lang_x_input_ids.to(model.device),
#                 attention_mask=lang_x_attention_mask.to(model.device),
#                 max_new_tokens=512,
#                 num_beams=3,
#                 no_repeat_ngram_size=3,
#                 bad_words_ids=bad_words_id,
#             )

#             parsed_output = (
#                 model.text_tokenizer.decode(generated_text[0]).split("<answer>")[-1].lstrip().rstrip().split("<|endofchunk|>")[0].lstrip().rstrip().lstrip('"').rstrip('"')
#             )
            
#             print(inputs)
#             print(parsed_output)
            
#             if parsed_output.split(" ")[0].lower()=="no":
#                 yesno_count += 1
#                 if len(parsed_output.split(" ")) > 1:
#                     if parsed_output.split(" ")[1].lower()=="none":
#                         both_count += 1
#             if len(parsed_output.split(" ")) > 1:
#                 if parsed_output.split(" ")[1].lower()=="none":
#                         reason_count += 1
                
#             write_text_file(f'{folder_name}/non-detective.txt',query_image_path)
#             write_text_file(f'{folder_name}/non-detective.txt',parsed_output)
#             write_text_file(f'{folder_name}/non-detective.txt',"")
            
#             # print(inputs)
#             # print("GPT:", parsed_output)
            
#             # fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10,5))
#             # axes[0].imshow(demo_image_one)
#             # axes[0].axis('off')
#             # axes[1].imshow(demo_image_two)
#             # axes[1].axis('off')
#             # axes[2].imshow(query_image)
#             # axes[2].axis('off')
#             # plt.show()
#             print(i,sub)
#             clear_output(wait=True)
            
#         yesno_acc = f"correct: {yesno_count}, total: {len(query_image_paths)-1}, yesno acc: {(yesno_count / (len(query_image_paths)-1)) * 100:.2f}%"
#         # print(yesno_acc)
#         reason_acc = f"correct: {reason_count}, total: {len(query_image_paths)-1}, reason acc: {(reason_count / (len(query_image_paths)-1)) * 100:.2f}%"
#         # print(reason_acc)
#         both_acc = f"correct: {both_count}, total: {len(query_image_paths)-1}, both acc: {(both_count / (len(query_image_paths)-1)) * 100:.2f}%"
#         # print(both_acc)
#         acc.append(("good",yesno_acc))
#         acc.append(("good",reason_acc))
#         acc.append(("good",both_acc))
        
#         write_text_file(f'{folder_name}/non-detective.txt',f'-----{sub} end-----')
#         write_text_file(f'{folder_name}/non-detective.txt',yesno_acc)
#         write_text_file(f'{folder_name}/non-detective.txt',reason_acc)
#         write_text_file(f'{folder_name}/non-detective.txt',both_acc)
        
#     for a in acc:
#         print(a)

# def test_query(category, anormaly_reason, anormaly_type, model, model_name, image_processor):
#     if category=="grid":
#         category__ = "metal grid"
#     else:
#         category__ = category
#     category__ = category__.replace('_', ' ')
#     for j, (ano_type,ano_reason) in enumerate(zip(anormaly_type,anormaly_reason)):
#         folder_name = f'./result/{category}/{ano_type}/{model_name}'
#         os.makedirs(folder_name, exist_ok=True)
#         with open(f'{folder_name}/detective_1.txt', mode='w') as f:
#             f.close()
#         with open(f'{folder_name}/non-detective_1.txt', mode='w') as f:
#             f.close()
        
#         subfolder_string = generate_list_string(anormaly_reason)
#         model.text_tokenizer.padding_side = "left"
        
#         """ クエリ：不良品 """
#         sentence = f"query: NG"
#         # print(sentence)
#         write_text_file(f'{folder_name}/detective_1.txt',sentence)
#         # long
#         inputs = textwrap.dedent(f"""
#            <image>User: This is an image of {category__}. Does this wood have any defects such as {subfolder_string}? GPT:<answer>
#         """)
#         # short
#         # inputs = textwrap.dedent(f"""
#         #     <image>User: This is an image of {category__}. Does this wood have any defects? GPT:<answer>
#         # """)        
        
#         inputs = "".join(inputs.split("\n"))
#         lang_x = model.text_tokenizer(
#             [
#                 inputs
#             ],
#             return_tensors="pt",
#         )
        
#         write_text_file(f'{folder_name}/detective_1.txt',f'-----{ano_type} start-----')
#         write_text_file(f'{folder_name}/detective_1.txt',"")
            
#         query_folder_path = f"/data/data/mvtec/{category}/test/{ano_type}"
#         query_image_paths = get_image_paths(query_folder_path)
#         count = 0
#         for i, query_image_path in enumerate(query_image_paths[1:]):
#             # print(query_image_path)
#             query_image = Image.open(query_image_path).resize((224, 224)).convert("RGB")
#             vision_x = image_processor.preprocess([query_image], return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0)
        
#             # Get the data type from model's parameters
#             model_dtype = next(model.parameters()).dtype

#             # Convert tensors to the model's data type
#             vision_x = vision_x.to(dtype=model_dtype)
#             lang_x_input_ids = lang_x["input_ids"]
#             lang_x_attention_mask = lang_x["attention_mask"]

#             bad_words_id = model.text_tokenizer(["User:", "GPT1:", "GFT:", "GPT:"], add_special_tokens=False).input_ids
#             generated_text = model.generate(
#                 vision_x=vision_x.to(model.device),
#                 lang_x=lang_x_input_ids.to(model.device),
#                 attention_mask=lang_x_attention_mask.to(model.device),
#                 max_new_tokens=512,
#                 num_beams=3,
#                 no_repeat_ngram_size=3,
#                 bad_words_ids=bad_words_id,
#             )

#             parsed_output = (
#                 model.text_tokenizer.decode(generated_text[0]).split("<answer>")[-1].lstrip().rstrip().split("<|endofchunk|>")[0].lstrip().rstrip().lstrip('"').rstrip('"')
#             )
            
#             if parsed_output.split(".")[0].lower()=="yes":
#                 count += 1
            
#             write_text_file(f'{folder_name}/detective_1.txt',query_image_path)
#             write_text_file(f'{folder_name}/detective_1.txt',parsed_output)
#             write_text_file(f'{folder_name}/detective_1.txt',"")
            
#             # print(inputs)
#             # print("GPT:", parsed_output)
            
#             # fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10,5))
#             # axes[0].imshow(demo_image_one)
#             # axes[0].axis('off')
#             # axes[1].imshow(demo_image_two)
#             # axes[1].axis('off')
#             # axes[2].imshow(query_image)
#             # axes[2].axis('off')
#             # plt.show()
            
#         acc = f"correct: {count}, total: {len(query_image_paths)-1}, acc: {(count / (len(query_image_paths)-1)) * 100:.2f}%"
#         print(acc)
        
#         write_text_file(f'{folder_name}/detective_1.txt',f'-----{ano_type} end-----')
#         write_text_file(f'{folder_name}/detective_1.txt',acc)
        
#         """ クエリ：良品 """
#         if j==0:
#             sentence = f"query: OK"
#             # print(sentence)
#             write_text_file(f'{folder_name}/non-detective_1.txt',sentence)
#             write_text_file(f'{folder_name}/non-detective_1.txt',f'-----{ano_type} start-----')
#             write_text_file(f'{folder_name}/non-detective_1.txt',"")
                
#             query_folder_path = f"/data/data/mvtec/{category}/test/good"
#             query_image_paths = get_image_paths(query_folder_path)
#             count = 0
#             for i, query_image_path in enumerate(query_image_paths[1:]):
#                 # print(query_image_path)
#                 query_image = Image.open(query_image_path).resize((224, 224)).convert("RGB")
#                 vision_x = image_processor.preprocess([query_image], return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0)
            
#                 # Get the data type from model's parameters
#                 model_dtype = next(model.parameters()).dtype

#                 # Convert tensors to the model's data type
#                 vision_x = vision_x.to(dtype=model_dtype)
#                 lang_x_input_ids = lang_x["input_ids"]
#                 lang_x_attention_mask = lang_x["attention_mask"]

#                 bad_words_id = model.text_tokenizer(["User:", "GPT1:", "GFT:", "GPT:"], add_special_tokens=False).input_ids
#                 generated_text = model.generate(
#                     vision_x=vision_x.to(model.device),
#                     lang_x=lang_x_input_ids.to(model.device),
#                     attention_mask=lang_x_attention_mask.to(model.device),
#                     max_new_tokens=512,
#                     num_beams=3,
#                     no_repeat_ngram_size=3,
#                     bad_words_ids=bad_words_id,
#                 )

#                 parsed_output = (
#                     model.text_tokenizer.decode(generated_text[0]).split("<answer>")[-1].lstrip().rstrip().split("<|endofchunk|>")[0].lstrip().rstrip().lstrip('"').rstrip('"')
#                 )
                
#                 if parsed_output.split(".")[0].lower()=="no":
#                     count += 1
                
#                 write_text_file(f'{folder_name}/non-detective_1.txt',query_image_path)
#                 write_text_file(f'{folder_name}/non-detective_1.txt',parsed_output)
#                 write_text_file(f'{folder_name}/non-detective_1.txt',"")
                
#                 # print(inputs)
#                 # print("GPT:", parsed_output)
                
#                 # fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10,5))
#                 # axes[0].imshow(demo_image_one)
#                 # axes[0].axis('off')
#                 # axes[1].imshow(demo_image_two)
#                 # axes[1].axis('off')
#                 # axes[2].imshow(query_image)
#                 # axes[2].axis('off')
#                 # plt.show()
                
#             acc = f"correct: {count}, total: {len(query_image_paths)-1}, acc: {(count / (len(query_image_paths)-1)) * 100:.2f}%"
#             print(acc)
            
#             write_text_file(f'{folder_name}/non-detective_1.txt',f'-----{ano_type} end-----')
#             write_text_file(f'{folder_name}/non-detective_1.txt',acc)