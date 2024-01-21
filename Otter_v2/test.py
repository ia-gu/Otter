
# # /home/yyamada/Otter_の直下にあるファイル，フォルダ名をすべて取得
# import os
# import glob
# import shutil
# import sys
# import subprocess



# # 任意の拡張子を持つファイルのみにマッチするようにパターンを変更
# files = glob.glob("/home/yyamada/Otter_/*.*")
# # ファイルのみを含む新しいリストを作成
# files_only = [f for f in files if not os.path.isdir(f)]

# # フォルダ名を取得
# folders = glob.glob("/home/yyamada/Otter_/*/")
# print(folders)
# for f in ['/home/yyamada/Otter_/log_/','/home/yyamada/Otter_/log/','/home/yyamada/Otter_/wandb/','/home/yyamada/Otter_/weights/']:
#     folders.remove(f)
# print(folders)

# # カラントディレクトリに上記のファイル，フォルダをコピー
# # ただしフォルダに関しては再帰的にコピー
# for f in files_only:
#     shutil.copy(f, "./")
# for f in folders:
#     shutil.copytree(f, "./"+f.split("/")[-2])

import random

question_lines = [
    ['Are there any defects visible? Please list them if so.'],
    ['Can you spot any defects? Kindly provide the names of any defects.'],
    ['Are any defects present? If so, please specify the defect(s).'],
    ["Is there a defect noticeable? Please mention the defect's name if present."],
    ['Could you point out any defects? If yes, name the defect(s), please.'],
    ['Are there imperfections to be noted? Please identify the defect(s) by name.'],
    ['Could you identify defects? Please specify any defects identified.'],
    ["Is the image showing any defects? If that's the case, please state the defect."],
    ['Are defects evident in this image? Please provide the name of any such defects.'],
    ['Can you detect any defects here? Please detail the defect(s) found.'],
    ['Is this picture defect-free? Please give the name of any defects found.'],
    ['Are there flaws in this image? If any, please provide the defect identification.'],
    ['Does this picture show any defects? Please provide the defect details if applicable.'],
    ['Can you see any defects in this image? Please cite the defect(s) if any.'],
    ['Are defects apparent in this picture? If present, please provide the defect name.'],
    ['Are there any issues with this image? Please state the defect(s) if they exist.'],
    ['Do you find any defects in this image? Please name any defects observed.'],
    ['Could you inspect the image for defects? Please declare any defects noticed.'],
    ['Is the image marred by defects? Please call out any defects detected.'],
    ['Are there any defects that stand out? Please enumerate the defect(s) if any.'],
    ['Is there anything amiss with this image? Please provide the defect name(s) if evident.'],
    ['Would you say there are defects in the image? Please indicate the defect(s) if any are found.'],
    ['Could there be any defects in this image? Please report any defects identified.'],
    ['Are imperfections present in this image? Please name the defect(s) if any are visible.'],
    ['Does this image exhibit any defects? Please describe the defect(s) if present.'],
    ['Might there be defects in this image? Please reveal any defect names if found.'],
    ['Are there noticeable defects in this image? Please disclose any defects if detected.'],
    ['Is anything incorrect with this image? Please point out the defect(s) if any.'],
    ['Can you confirm if there are defects in the image? Please announce the defect(s) if observed.'],
    ['Is the image flawed in any way? Please articulate any defects if any.']
]

yes_responses_array = [
    ['Yes, the product shown has {defect}.'],
    ["Yes, indeed, there's {defect} evident in the product."],
    ['Yes, absolutely, {defect} is present in this item.'],
    ['Yes, certainly, the product displays {defect}.'],
    ['Yes, affirmative, the item exhibits {defect}.'],
    ['Yes, correct, {defect} can be seen on the product.'],
    ['Yes, undoubtedly, the product is marked by {defect}.'],
    ['Yes, confirmed, we can observe {defect} in the product.'],
    ['Yes, {defect} is visible on this item.'],
    ['Yes, surely, the product carries {defect}.'],
    ['Yes, the product shown has {defect}.'],
    ["Yes, indeed, there's {defect} evident in the product."],
    ['Yes, absolutely, {defect} is present in this item.'],
    ['Yes, certainly, the product displays {defect}.'],
    ['Yes, affirmative, the item exhibits {defect}.'],
    ['Yes, correct, {defect} can be seen on the product.'],
    ['Yes, undoubtedly, the product is marked by {defect}.'],
    ['Yes, confirmed, we can observe {defect} in the product.'],
    ['Yes, {defect} is visible on this item.'],
    ['Yes, surely, the product carries {defect}.'],
    ['Yes, the product shown has {defect}.'],
    ["Yes, indeed, there's {defect} evident in the product."],
    ['Yes, absolutely, {defect} is present in this item.'],
    ['Yes, certainly, the product displays {defect}.'],
    ['Yes, affirmative, the item exhibits {defect}.'],
    ['Yes, correct, {defect} can be seen on the product.'],
    ['Yes, undoubtedly, the product is marked by {defect}.'],
    ['Yes, confirmed, we can observe {defect} in the product.'],
    ['Yes, {defect} is visible on this item.'],
    ['Yes, surely, the product carries {defect}.']
]
random_idx = random.randint(0, len(question_lines)-1)
print(random_idx)
print(question_lines[random_idx][0])
print(yes_responses_array[random_idx][0])
defect = 'scratch'
# yes_responses_array[random_idx][0]に含まれる{defect}をdefectに置換
print(yes_responses_array[random_idx][0].format(defect=defect))