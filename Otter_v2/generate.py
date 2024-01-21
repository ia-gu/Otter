import os
import shutil
import re




# ベースパス
base_folder = "/home/dataset/yyama_dataset/AC_images"

def delete_folder(folder_path):
    try:
        shutil.rmtree(folder_path)
    except Exception as e:
        print(f"エラー：{e}")

def count_image_files(folder_path):
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_count = 0
    for file in os.listdir(folder_path):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_count += 1
    return image_count

if os.path.isdir(base_folder):  # ベースフォルダが存在するかチェック
    for parent_folder in os.listdir(base_folder):
        parent_folder_path = os.path.join(base_folder, parent_folder)
        if os.path.isdir(parent_folder_path):  # parent_folder_pathがディレクトリであるかチェック
            for child_folder in os.listdir(parent_folder_path):
                child_folder_path = os.path.join(parent_folder_path, child_folder)
                if os.path.isdir(child_folder_path):  # child_folder_pathがディレクトリであるかチェック
                    num_images = count_image_files(child_folder_path)
                    if num_images == 0:
                        print(f"削除フォルダ名: {child_folder_path}, 枚数: {num_images}")
                        delete_folder(child_folder_path)
            if len(os.listdir(parent_folder_path)) == 0:
                print(f"削除フォルダ名: {parent_folder_path}")
                delete_folder(parent_folder_path)


# ベースフォルダが存在するかどうかを確認
if os.path.isdir(base_folder):
    # 親フォルダ内のサブフォルダを取得
    for folder_name in os.listdir(base_folder):
        subfolder_path = os.path.join(base_folder, folder_name)
        
        if os.path.isdir(subfolder_path):  # サブフォルダであるか確認
            # サブフォルダ内の子フォルダを取得
            for subfolder_name in os.listdir(subfolder_path):
                child_folder_path = os.path.join(subfolder_path, subfolder_name)
                
                if os.path.isdir(child_folder_path):  # 子フォルダであるか確認
                    # 子フォルダ内のファイル名を取得
                    files = os.listdir(child_folder_path)
                    
                    # ファイル名から数字部分を取得し、リストに保存
                    numbers = [int(re.search(r'\d+', file).group()) for file in files if re.search(r'\d+', file)]
                    
                    # ファイル名の重複チェックと変更
                    for file in files:
                        match = re.search(r'\d+', file)
                        if match:
                            number = int(match.group())
                            
                            # 同じ番号が複数存在する場合
                            if numbers.count(number) > 1:
                                new_number = 0
                                
                                # 重複していない番号を探す
                                while new_number in numbers:
                                    new_number += 1
                                
                                # 新しいファイル名を作成
                                new_file = file.replace(str(number), str(new_number))
                                new_file_path = os.path.join(child_folder_path, new_file)
                                
                                # ファイル名を変更
                                os.rename(
                                    os.path.join(child_folder_path, file), 
                                    new_file_path
                                )
                                
                                # ファイル名の変更を表示
                                print(f'{child_folder_path}/{file} -> {new_file_path}')
                                
                                # リストを更新
                                numbers.remove(number)
                                numbers.append(new_number)



import os
import shutil
import random

# ディレクトリのパスを指定

src_dir = "/home/dataset/yyama_dataset/AC_images"
train_dir = '/home/dataset/yyama_dataset/tasks/AC/train/'
val_dir = '/home/dataset/yyama_dataset/tasks/AC/val/'

# trainとvalの割合
ratio = 0.8

# trainとvalのディレクトリを作成
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# src_dir内の各フォルダに対して
for root, dirs, files in os.walk(src_dir):
    # 画像のリストを取得し、シャッフル
    images = [f for f in files if os.path.isfile(os.path.join(root, f))]
    random.shuffle(images)
    
    # trainとvalに分割
    train_images = images[:int(ratio * len(images))]
    val_images = images[int(ratio * len(images)):]
    # print(len(train_images))
    # print(len(val_images))
    
    # trainとvalのサブディレクトリを作成
    # print(os.path.join(train_dir, os.path.relpath(root, src_dir)))
    train_folder = os.path.join(train_dir, os.path.relpath(root, src_dir))
    val_folder = os.path.join(val_dir, os.path.relpath(root, src_dir))
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    
    # # 画像をコピー
    for img in train_images:
        shutil.copy(os.path.join(root, img), os.path.join(train_folder, img))
        # print(os.path.join(root, img))
        # print(os.path.join(train_folder, img))
    for img in val_images:
        shutil.copy(os.path.join(root, img), os.path.join(val_folder, img))
        # print(os.path.join(root, img))
        # print(os.path.join(val_folder, img))

print('Images copied to train and val folders.')


import os
import shutil
from pathlib import Path
# 子フォルダを移動させる関数
def move_child_folder(src, dst):
    """
    :param src: 移動するフォルダのパス
    :param dst: 移動先のパス
    """ 
    if os.path.exists(dst):
        # 移動先のディレクトリが存在する場合、ファイルのみを移動
        for file_name in os.listdir(src):
            full_file_name = os.path.join(src, file_name)
            if os.path.isfile(full_file_name):
                shutil.move(full_file_name, dst)
        # もとのフォルダを削除
        os.rmdir(src)
    else:
        # 移動先のフォルダが存在しない場合、フォルダごと移動
        if not os.path.exists(os.path.dirname(dst)):
            os.makedirs(os.path.dirname(dst))
        shutil.move(src, os.path.join(os.path.dirname(dst), os.path.basename(src)))

# 親フォルダごと移動させる関数
def move_parent_folder(src, dst):
    """
    :param src: 移動するフォルダのパス
    :param dst: 移動先のパス
    """
    for src_dir, dirs, files in os.walk(src):
        dst_dir = src_dir.replace(src, dst, 1)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for file_ in files:
            src_file = os.path.join(src_dir, file_)
            dst_file = os.path.join(dst_dir, file_)
            if os.path.exists(dst_file):
                os.remove(dst_file)
            shutil.move(src_file, dst_dir)
    shutil.rmtree(src)
    
# 移動先のパスを生成する関数
def change_path(path, new_base):
    # パスを分割
    parts = path.split('/')
    # 新しいベースでパスを再構築
    parts[1] = new_base
    return '/'.join(parts)

# 与えられたフォルダ内の画像ファイル数を数える関数
def count_image_files(folder_path):
    image_extensions = ['.jpg', '.jpeg', '.png']  # 画像ファイルの拡張子を追加
    image_count = 0
    for file in os.listdir(folder_path):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_count += 1
    return image_count

def val(val_dir, val_base):
    if os.path.isdir(val_dir):  # val_dirがディレクトリであるか確認
        for parent_folder in os.listdir(val_dir):
            parent_folder_path = os.path.join(val_dir, parent_folder)
            parent_dir_name = Path(parent_folder_path).name
            if os.path.isdir(parent_folder_path):  # ディレクトリであるか確認
                for child_folder in os.listdir(parent_folder_path):
                    child_folder_path = os.path.join(parent_folder_path, child_folder)
                    if os.path.isdir(child_folder_path):  # ディレクトリであるか確認
                        num_images = count_image_files(child_folder_path)
                        if num_images == 1:
                            # print(f'子フォルダ:{child_folder_path}')
                            print(f'親:{Path(child_folder_path).parent.name}    子:{Path(child_folder_path).name}')
                            print(f"移動フォルダ名: {child_folder_path}, 移動先: {val_base+Path(child_folder_path).parent.name+'/'+Path(child_folder_path).name}, 枚数: {num_images}")
                            move_child_folder(child_folder_path, val_base+Path(child_folder_path).parent.name+'/'+Path(child_folder_path).name)

val_dir = '/home/dataset/yyama_dataset/tasks/AC/val'
val_base = '/home/dataset/yyama_dataset/tasks/AC/train/'
val(val_dir, val_base)


# 空のフォルダを削除
def delete_empty_folders(base_folder):
    if os.path.isdir(base_folder):
        for parent_folder in os.listdir(base_folder):
            parent_folder_path = os.path.join(base_folder, parent_folder)
            if os.path.isdir(parent_folder_path):  # ディレクトリであるか確認
                for child_folder in os.listdir(parent_folder_path):
                    child_folder_path = os.path.join(parent_folder_path, child_folder)
                    if os.path.isdir(child_folder_path):  # ディレクトリであるか確認
                        num_images = count_image_files(child_folder_path)
                        # 画像枚数が2枚未満の場合、その子フォルダを削除
                        if num_images==0:
                            print(f"削除フォルダ名: {child_folder_path}, 枚数: {num_images}")
                            delete_folder(child_folder_path)
                # 親フォルダ内の子フォルダ数が0の場合、親フォルダを削除
                if len(os.listdir(parent_folder_path))==0 :
                    print(f"削除フォルダ名: {parent_folder_path}")
                    delete_folder(parent_folder_path)
                    
train_dir = '/home/dataset/yyama_dataset/tasks/AC/train'
print(train_dir)
delete_empty_folders(train_dir)
val_dir = '/home/dataset/yyama_dataset/tasks/AC/val'
print(val_dir)
delete_empty_folders(val_dir)


import os

def find_missing_folders(folder1, folder2):
    """
    指定された2つのフォルダ間で存在しないフォルダを見つけて出力します。
    Args:
        folder1 (str): 最初のフォルダのパス
        folder2 (str): 2番目のフォルダのパス
    """
    # フォルダ1内のフォルダリストを取得
    folders1 = os.listdir(folder1)

    # フォルダ2内のフォルダリストを取得
    folders2 = os.listdir(folder2)

    # フォルダ1に存在し、フォルダ2に存在しないフォルダを見つける
    missing_folders = [folder for folder in folders1 if folder not in folders2]

    # 結果を出力
    print(f"フォルダ '{folder1}' にあって '{folder2}' に存在しないフォルダ:")
    for folder in missing_folders:
        print(folder)
    print('FINISH')

# 使用例
folder1_path = '/home/dataset/yyama_dataset/tasks/AC/train'
folder2_path = '/home/dataset/yyama_dataset/tasks/AC/val'
find_missing_folders(folder1_path, folder2_path)
find_missing_folders(folder2_path, folder1_path)