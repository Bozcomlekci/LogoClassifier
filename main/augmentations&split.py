import cv2
import os
import albumentations as A
from albumentations import ( 
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    GaussNoise, MotionBlur, MedianBlur, PiecewiseAffine,
    Sharpen, Emboss, RandomBrightnessContrast, Flip, OneOf, Compose, RandomGamma, ElasticTransform, ChannelShuffle,RGBShift, Rotate,
    Resize, RandomCrop, Rotate, HorizontalFlip
)
import shutil
import glob
import random
from PIL import Image
import numpy as np
import os
import shutil
from tqdm import tqdm


# # Augmentations

def augmentor():
    return Compose([
        Resize(512, 512),
        Compose([
            RandomRotate90(),
            Flip(),
            Transpose(),
        ], p=0.15),        
        OneOf([
            Rotate(limit=[-60,60]),
        ], p=0.5),
        Compose([
            ShiftScaleRotate()
        ], p=0.05),
        OneOf([
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=.2),
            MedianBlur(blur_limit=3, p=.1),
            Blur(blur_limit=3, p=.1),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=.2),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=.1),
            PiecewiseAffine(p=0.2),
        ], p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            Sharpen(),
            Emboss(),
            RandomBrightnessContrast()
        ], p=0.3),
        OneOf([
            HueSaturationValue(p=0.3)
        ], p=0.05),
        Compose([
            RandomCrop(500, 500),
            Resize(512, 512)
        ], p=0.05)
    ], p=0.90)


# In[ ]:


logodir = 'logo_data'
subdirs = os.walk(logodir)
subdirs = [x[0] for x in subdirs]
subdirs = subdirs[1:]
print(subdirs)

augoffset = 500
augment = augmentor()
random.seed(42)

extracted_path = 'logo_data_augments'
for path in subdirs:
    print(path, 'started')
    images=[]
    for filename in os.listdir(path):
        imagepath = os.path.join(path,filename)
        img = cv2.imread(imagepath)
        if img is not None:
            images.append(img) 
    logo_brand = path[10:]
    branddir = os.path.join(extracted_path,logo_brand)
    if not os.path.exists(branddir):
        os.mkdir(branddir)
    for i in range(augoffset):
        imageindex = random.randint(0, len(images)-1)
        img = images[imageindex]
        augmented = augment(image=img)
        augmentedimg = augmented['image']
        newimgpath = os.path.join(branddir, logo_brand + str(i) + '.jpg')
        out = cv2.imwrite(newimgpath, augmentedimg)
    print(path, 'finished')


# #### Export LogoDet files as other_class

#Export other_class type of logos&images for traning from LogoDet-3K dataset
#Path to LogoDet-3K
logodetpath = 'datasets/LogoDet-3K'
dest_root = 'logo_data_augments/other_class'
if not os.path.exists(dest_root):
    os.makedirs(dest_root)
for i, file in enumerate(glob.glob(os.path.join(logodetpath, '**/*.jpg'), recursive=True)):
    if file.endswith(".jpg") and random.randint(1, 25) == 1:
        destination = os.path.join(dest_root, str(i) + '.jpg')
        print('Moved from: {} to {}'.format(file,destination) )
        shutil.copyfile(file, destination)


# # Train-Val Split

TRAIN_VAL_RATIO = 0.99
data_dir = 'data/logo_data'

images_dir = 'logo_data_augments'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')

classes = os.listdir(images_dir)

for label in tqdm(classes):
    cls_label = str(label).replace(' ', '_')
    class_dir = os.path.join(images_dir, label)

    images = os.listdir(class_dir)

    n_train = int(len(images) * TRAIN_VAL_RATIO)

    train_images = images[:n_train]
    test_images = images[n_train:]

    os.makedirs(os.path.join(train_dir, cls_label), exist_ok = True)
    os.makedirs(os.path.join(val_dir, cls_label), exist_ok = True)

    for image in train_images:
        image_src = os.path.join(class_dir, image)
        image_dst = os.path.join(train_dir, cls_label, image.replace(' ', '_')) 
        shutil.copyfile(image_src, image_dst)

    for image in test_images:
        image_src = os.path.join(class_dir, image)
        image_dst = os.path.join(val_dir, cls_label, image.replace(' ', '_')) 
        shutil.copyfile(image_src, image_dst)

val_data_dir = 'data/logo_data/val'
train_data_dir = 'data/logo_data/train'

os.makedirs('data/logo_data/meta', exist_ok = True)
val_annotation_output = 'data/logo_data/meta/val.txt' 
train_annotation_output = 'data/logo_data/meta/train.txt' 
class_output = 'data/logo_data/meta/class_mapping.txt'
counter = 0

for subdir, dirs, files in os.walk(val_data_dir):
  for dir_ in dirs:
    class_name = str(dir_).replace(' ', '_')
    folder_path = os.path.join(val_data_dir, class_name)

    files = glob.glob(folder_path + '/**/*.png', recursive=True)
    files_jpg = glob.glob(folder_path + '/**/*.jpg', recursive=True)
    files.extend(files_jpg)

    f=open(val_annotation_output,'a')
    for file in files:
        f.write(os.path.join(class_name,os.path.basename(file))+" "+str(counter)+'\n')
        
    f.close()
    
    f=open(class_output,'a')
    f.write(class_name+" "+str(counter)+'\n')

    f.close()

    counter += 1
    
for subdir, dirs, files in os.walk(train_data_dir):
  for dir_ in dirs:
    class_name = str(dir_).replace(' ', '_')
    folder_path = os.path.join(train_data_dir, class_name)

    files = glob.glob(folder_path + '/**/*.png', recursive=True)
    files_jpg = glob.glob(folder_path + '/**/*.jpg', recursive=True)
    files.extend(files_jpg)
    
    f = open(class_output,'r')
    text = f.read()
    parsed_text = text.split()
    indexOfDir = parsed_text.index(class_name)
    label = parsed_text[indexOfDir+1]
    f.close()
    
    f=open(train_annotation_output,'a')
    for file in files:
        f.write(os.path.join(class_name,os.path.basename(file))+" "+str(label)+'\n')
    f.close()

