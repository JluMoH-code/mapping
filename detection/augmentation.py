import albumentations as A
import cv2
import os
import random
import uuid
from dotenv import load_dotenv 
load_dotenv() 

COLOR_RED = (255, 0, 0)
COLOR_WHITE = (255, 255, 255)

def visualize_bbox(img, bbox, class_name, color=COLOR_RED, thickness=2):
    """Visualizes a single bounding box on the image"""
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), COLOR_RED, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=COLOR_WHITE,
        lineType=cv2.LINE_AA,
    )
    return img

def visualize(image, bboxes, category_ids, category_id_to_name):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        bbox_yolo = bbox_from_yolo_to_coco(img, bbox)
        img = visualize_bbox(img, bbox_yolo, class_name)
    cv2.imshow("img_with_bbox", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def bbox_from_yolo_to_coco(image, bbox):
    img_height, img_width, _ = image.shape
    x_center, y_center, width, height = bbox

    x_min = int((x_center - width / 2) * img_width)
    y_min = int((y_center - height / 2) * img_height)
    w = int(width * img_width)
    h = int(height * img_height)
    
    return [x_min, y_min, w, h]

def load_data_from_folders(data_folder):
    images_folder = os.path.join(data_folder, 'images')
    labels_folder = os.path.join(data_folder, 'labels')

    image_files = [f for f in os.listdir(images_folder) if f.endswith('.jpg')]

    category_id_to_name = {0: 'car', 1: 'people'} 

    for image_file in image_files:
        image_path = os.path.join(images_folder, image_file)
        image = cv2.imread(image_path)
        
        label_file = image_file.replace('.jpg', '.txt')
        label_path = os.path.join(labels_folder, label_file)

        bboxes = []
        category_ids = []

        if os.path.exists(label_path):
            with open(label_path, 'r') as file:
                annotations = file.readlines()
                for annotation in annotations:
                    parts = list(map(float, annotation.strip().split()))
                    category_id = int(parts[0]) 

                    bboxes.append(parts[1:])
                    category_ids.append(category_id)

        yield image, bboxes, category_ids, category_id_to_name, image_file
      
def save_img(image, bboxes, category_ids, image_file):
    images_folder = data_folder_path + 'images'
    labels_folder = data_folder_path + 'labels'
    
    uuid_add_name = str(uuid.uuid4())
    label_file = image_file[:-4] + ".txt"
    image_path = images_folder + "\\" + uuid_add_name + image_file
    label_path = labels_folder + "\\" + uuid_add_name + label_file
    
    combined = [[category_id, *bbox] for category_id, bbox in zip(category_ids, bboxes)]
    cv2.imwrite(image_path, image)
    
    with open(label_path, 'w') as file:
        for item in combined:
            file.write(' '.join(map(str, item)) + '\n')
      
def get_pipeline(p):
    return A.Compose(
        [
            A.Affine(p=p),
            A.CLAHE(p=p),
            A.ChannelShuffle(p=p),
            A.ChromaticAberration(p=p),
            A.CoarseDropout(p=p, max_holes=100, max_height=10, max_width=10),
            A.ColorJitter(p=p, brightness=(0.6, 0.8), contrast=(0.6, 1), saturation=(0.6, 1), hue=(-0.0, 0.0)),
            A.D4(p=p),
            A.Downscale(p=p, scale_min=0.2, scale_max=0.99),
            A.HueSaturationValue(p=p, hue_shift_limit=15, sat_shift_limit=15, val_shift_limit=15),
            A.ISONoise(p=p, intensity=(0.4, 0.9)),
            A.Morphological(p=p),
            A.MotionBlur(p=p),
            A.OpticalDistortion(p=p),
            A.Perspective(p=p),
            A.PixelDropout(p=p),
            A.RGBShift(p=p, r_shift_limit=(-30, 30), g_shift_limit=(-30, 30), b_shift_limit=(-30, 30)),
            A.RandomBrightnessContrast(p=p),
            A.RandomGamma(p=p),
            A.RandomGravel(p=p),
            A.RandomRain(p=p, blur_value=1, brightness_coefficient=0.8),
            A.Sharpen(p=p),
            A.Spatter(p=p, std=(0.05, 0.2), intensity=(0.2, 0.6)),
        ],
        bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']),
    )

data_folder_path = os.getenv("BASE_PATH_TRAIN_DATA") + "\\"
data_generator = load_data_from_folders(data_folder_path)
       
for image, bboxes, category_ids, category_id_to_name, image_file in data_generator:    
    transformed = get_pipeline(p=0.2)(image=image, bboxes=bboxes, category_ids=category_ids)
    visualize(transformed['image'], transformed['bboxes'], transformed['category_ids'], category_id_to_name)
    save_img(transformed['image'], transformed['bboxes'], transformed['category_ids'], image_file)