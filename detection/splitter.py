import glob
import random
import os
import shutil
from dotenv import load_dotenv 
load_dotenv() 

PATH = os.getenv("BASE_PATH_TRAIN_DATA")

imgs_folder = PATH + "/images/"
txt_folder = PATH + "/labels/"

img_paths = glob.glob(imgs_folder + '*.jpg')
txt_paths = glob.glob(txt_folder + '*.txt')

data_size = len(img_paths)
train_coeff, val_coeff, test_coeff = 0.7, 0.2, 0.1

train_size = int(data_size * train_coeff)
val_size = int(data_size * val_coeff)

img_txt = list(zip(img_paths, txt_paths))
random.shuffle(img_txt)
img_paths, txt_paths = zip(*img_txt)

train_img_paths, train_txt_paths = img_paths[:train_size], txt_paths[:train_size]
valid_img_paths, valid_txt_paths = img_paths[train_size:train_size + val_size], txt_paths[train_size:train_size + val_size]
test_img_paths, test_txt_paths = img_paths[train_size + val_size:], txt_paths[train_size + val_size:]

folders = {
    'train': ("/train/images/", "/train/labels/"),
    'valid': ("/valid/images/", "/valid/labels/"),
    'test': ("/test/images/", "/test/labels/")
}

for key, (img_folder, label_folder) in folders.items():
    img_path = PATH + img_folder
    label_path = PATH + label_folder
    os.makedirs(img_path, exist_ok=True)
    os.makedirs(label_path, exist_ok=True)

def moving_data():
    for key, (img_folder, label_folder) in folders.items():
        move(eval(f"{key}_img_paths"), PATH + img_folder)
        move(eval(f"{key}_txt_paths"), PATH + label_folder)

def move(paths, folder):
    for p in paths:
        shutil.move(p, folder)
        
def create_yaml_file():
    classes_file = PATH + '/classes.txt'
    output_file = PATH + '/data.yaml'
    with open(classes_file, 'r') as f:
        class_names = [line.strip() for line in f.readlines() if line.strip()]

    nc = len(class_names)
    names = class_names

    with open(output_file, 'w') as f:
        f.write(f"train: ../train/images\n")
        f.write(f"val: ../val/images\n")
        f.write(f"test: ../test/images\n")
        f.write(f"nc: {nc}\n")
        f.write(f"names: {names}\n")

moving_data()
create_yaml_file()

os.rmdir(imgs_folder)
os.rmdir(txt_folder)