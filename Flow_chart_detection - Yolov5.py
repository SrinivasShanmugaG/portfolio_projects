#!/usr/bin/env python
# coding: utf-8

# #***Importing libraries***

# In[1]:


import os
import glob as glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import cv2
import requests
import random
import numpy as np
import pandas as pd
from IPython.display import Image, clear_output

get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')

np.random.seed(369)


# #***Installing Roboflow & Numba packages***

# In[2]:


get_ipython().system('pip install roboflow')
get_ipython().system('pip install numba')
clear_output()

from roboflow import Roboflow
from numba import cuda
device = cuda.get_current_device()
device.reset()


# #***Setting the directory***

# In[3]:


os.makedirs('/content/group_project', exist_ok=True)
os.chdir('/content/group_project')
os.getcwd()


# # **Importing dataset**

# In[4]:


# using Roboflow library importing the dataset
rf = Roboflow(api_key="V26IKEBkpPUKk9Ch9O4D", model_format="yolov5")
dataset = rf.workspace("object-detection-gol2i").project("flow-chart-detection").version(1).download(location="/content/group_project")


# In[5]:


# Setting Class Name
class_names = ['action', 'activity', 'commeent', 'control_flow', 'control_flowcontrol_flow', 'decision_node', 'exit_node', 'final_flow_node', 'final_node', 'fork', 'merge', 'merge_noode', 'null', 'object', 'object_flow', 'signal_recept', 'signal_send', 'start_node', 'text']
colors = np.random.uniform(0, 255, size=(len(class_names), 3))


# #**Exploratory Data Analysis**

# In[ ]:


image_paths='train/images/*'
label_paths='train/labels/*'
all_training_images = glob.glob(image_paths)
all_training_labels = glob.glob(label_paths)
all_training_images.sort()
all_training_labels.sort()

def images_info(images):
    processed_images = []
    for imgs in images:
        image = mpimg.imread(imgs)
        # Apply image pre-processing steps here
        # Example: Resize the image to a specific size
        # image = cv2.resize(image, (224, 224))
        processed_images.append(image)
    return processed_images

n_images = len(all_training_images)
img_shape = [x.shape[:2] for x in images_info(all_training_images)]

total_labels = 0
annot = []
image_class = []

for j in range(n_images):
    image = cv2.imread(all_training_images[j])
    with open(all_training_labels[j], 'r') as f:
        bboxes = []
        labels = []
        label_lines = f.readlines()
        for label_line in label_lines:
            label = label_line[0]
            bbox_string = label_line[2:]
            if len(bbox_string.split(' ')) == 4:
                x_c, y_c, w, h = bbox_string.split(' ')
                x_c = float(x_c)
                y_c = float(y_c)
                w = float(w)
                h = float(h)
                bboxes.append([x_c, y_c, w, h])
                labels.append(label)
                image_class.append(class_names[int(label)])

        total_labels += len(labels)
        annot.append(len(labels))

print('Dataset Name: ',dataset.name)
print('Number of images: ',n_images)
# print('Images shape: ',img_shape)
print("Number of Labels per image", set(annot))
print("Image Labels", image_class)
print(f"Total number of labels: {total_labels}")

class_df = pd.DataFrame({'Categories': image_class})

# plotting
plt.figure(figsize=(7, 7))

plt.subplot(2,1,1)
plt.hist(annot, bins=10)
plt.xlabel('Number of labels per image')
plt.ylabel('Frequency')
plt.title('Histogram: Class distribution')


plt.subplot(2,1,2)
sns.countplot(x='Categories',data=class_df,order=class_df['Categories'].value_counts().index)
sns.set(style="darkgrid")  # Optional: Set the plot style
plt.xlabel('Categories')
plt.ylabel('Frequency')
plt.title('Image Labels Frequency')
plt.xticks(rotation=75)  # Rotate x-axis labels by 45 degrees
plt.tight_layout()


# In[6]:


def download_file(url, save_name):
    url = url
    if not os.path.exists(save_name):
        file = requests.get(url)
        open(save_name, 'wb').write(file.content)
    else:
        print('File already present, skipping download...')


# In[7]:


# Function to convert bounding boxes in YOLO format to xmin, ymin, xmax, ymax.
def yolo2bbox(bboxes):
    xmin, ymin = bboxes[0]-bboxes[2]/2, bboxes[1]-bboxes[3]/2
    xmax, ymax = bboxes[0]+bboxes[2]/2, bboxes[1]+bboxes[3]/2
    return xmin, ymin, xmax, ymax


# In[8]:


def plot_box(image, bboxes, labels):
    # Need the image height and width to denormalize
    # the bounding box coordinates
    h, w, _ = image.shape
    for box_num, box in enumerate(bboxes):
        x1, y1, x2, y2 = yolo2bbox(box)
        # denormalize the coordinates
        xmin = int(x1*w)
        ymin = int(y1*h)
        xmax = int(x2*w)
        ymax = int(y2*h)
        width = xmax - xmin
        height = ymax - ymin

        class_name = class_names[int(labels[box_num])]

        cv2.rectangle(
            image,
            (xmin, ymin), (xmax, ymax),
            color=colors[class_names.index(class_name)],
            thickness=2
        )

        font_scale = min(1,max(0.45,int(w/500)))
        font_thickness = min(1, max(2.5,int(w/50)))

        p1, p2 = (int(xmin), int(ymin)), (int(xmax), int(ymax))
        # Text width and height
        tw, th = cv2.getTextSize(
            class_name,
            0, fontScale=font_scale, thickness=font_thickness
        )[0]
        p2 = p1[0] + tw, p1[1] + -th - 10
        # cv2.rectangle(
        #     image,
        #     p1, p2,
        #     color=colors[class_names.index(class_name)],
        #     thickness=-2,
        # )
        # cv2.putText(
        #     image,
        #     class_name,
        #     (xmin+1, ymin-10),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     font_scale,
        #     (255, 255, 255),
        #     font_thickness
        # )
    return image


# In[9]:


# Function to plot images with the bounding boxes.
def plot(image_paths, label_paths, num_samples):
    all_training_images = glob.glob(image_paths)
    all_training_labels = glob.glob(label_paths)
    all_training_images.sort()
    all_training_labels.sort()

    num_images = len(all_training_images)

    plt.figure(figsize=(10, 12))
    for i in range(num_samples):
        j = random.randint(0,num_images-1)
        image = cv2.imread(all_training_images[j])
        with open(all_training_labels[j], 'r') as f:
            bboxes = []
            labels = []
            label_lines = f.readlines()
            for label_line in label_lines:
                label = label_line[0]
                bbox_string = label_line[2:]
                if len(bbox_string.split(' ')) == 4:
                  x_c, y_c, w, h = bbox_string.split(' ')
                  x_c = float(x_c)
                  y_c = float(y_c)
                  w = float(w)
                  h = float(h)
                  bboxes.append([x_c, y_c, w, h])
                  labels.append(label)
        result_image = plot_box(image, bboxes, labels)
        plt.subplot(2, 2, i+1)
        plt.imshow(result_image[:, :, ::-1])
        plt.axis('off')
    plt.subplots_adjust(wspace=0)
    plt.tight_layout()
    plt.show()


# In[ ]:


# Visualize a few training images.
plot(
    image_paths='train/images/*',
    label_paths='train/labels/*',
    num_samples=4,
)


# In[10]:


import cv2

def extract_objects(image, bboxes):
    objects = []
    h, w, _ = image.shape

    for bbox in bboxes:
        x1, y1, x2, y2 = yolo2bbox(bbox)
        xmin = int(x1 * w)
        ymin = int(y1 * h)
        xmax = int(x2 * w)
        ymax = int(y2 * h)

        object_image = image[ymin:ymax, xmin:xmax]
        objects.append(object_image)

    return objects

def extract_objects_from_images(image_paths, label_paths):
    all_training_images = glob.glob(image_paths)
    all_training_labels = glob.glob(label_paths)
    all_training_images.sort()
    all_training_labels.sort()

    num_images = len(all_training_images)
    extracted_objects = []

    for j in range(num_images):
        image = cv2.imread(all_training_images[j])
        with open(all_training_labels[j], 'r') as f:
            bboxes = []
            labels = []
            label_lines = f.readlines()
            for label_line in label_lines:
                label = label_line[0]
                bbox_string = label_line[2:]
                if len(bbox_string.split(' ')) == 4:
                    x_c, y_c, w, h = bbox_string.split(' ')
                    x_c = float(x_c)
                    y_c = float(y_c)
                    w = float(w)
                    h = float(h)
                    bboxes.append([x_c, y_c, w, h])
                    labels.append(label)

            objects = extract_objects(image, bboxes)
            extracted_objects.extend(objects)

    return extracted_objects


# In[11]:


# Extracting each object from the image
for extracted_images in extract_objects_from_images(image_paths='train/images/109_png_jpg.rf.2821a1475546ee9beba7a18c204863d9.jpg',
    label_paths='train/labels/109_png_jpg.rf.2821a1475546ee9beba7a18c204863d9.txt'):
    print('*')
    plt.figure(figsize=[25, 15])
    nodes = len(extracted_images)
    for i, img in enumerate(extracted_images):
        plt.subplot(1, nodes, i+1)
        plt.imshow(img)
plt.show()


# # **Data Pre-processing**

# In[12]:


# Image resizing and leveling
# # Define the paths to the train, test, and validation folders
# train_folder = 'train/'
# test_folder = 'test/'
# valid_folder = 'valid/

# leveled_train_folder = 'train/leveled_'
# leveled_test_folder = 'test/leveled_'
# leveled_valid_folder = 'valid/leveled_'

# for fold in ['/images','/labels']:
#   os.makedirs(leveled_train_folder + fold, exist_ok=True)
#   os.makedirs(leveled_test_folder, exist_ok=True)
#   os.makedirs(leveled_valid_folder, exist_ok=True)

# # Iterate over the images in the train folder
# image_path = os.path.join(train_folder, 'images/*')
# label_path = os.path.join(train_folder, 'labels/*')
# leveled_image_path = os.path.join(leveled_train_folder, 'images/*')
# leveled_label_path = os.path.join(leveled_train_folder, 'labels/*')

def perform_object_leveling(image_path, label_path, leveled_image_path, leveled_label_path):
    all_training_images = glob.glob(image_path)
    all_training_labels = glob.glob(label_path)
    all_training_images.sort()
    all_training_labels.sort()
    # Define the desired size for the resized images
    desired_size = (640, 640)
    num_images = len(all_training_images)

    for j in range(num_images):
        image = cv2.imread(all_training_images[j])
        try:
          # Resize the image
          resized_image = cv2.resize(image, desired_size)
          # Convert the image to grayscale
          gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
          # equalized = cv2.equalizeHist(gray_image)
          # Normalize the pixel values
          # normalized_image = gray_image / 255.0
          # # # Subtract the mean value
          # # mean_value = np.mean(normalized_image)
          # # centered_image = normalized_image - mean_value
          # plt.imshow(centered_image)
          # plt.show()
          blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
          # Thresholding - Applying adaptive threshold
          # threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
          # Apply Canny edge detection
          edges = cv2.Canny(blurred, 50, 150)
          # Save the leveled image to a new file
          leveled_image = cv2.normalize(edges, dst=None, alpha=255, beta=0,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
          # cv2.imwrite(leveled_image_path + all_training_images[j].split('/')[2],leveled_image)
          cv2.imwrite(all_training_images[j],leveled_image)

        except:
          raise Exception('Please check the fomart of the image')
          break
          height, width , layers = centered_image.shape
          size=(width,height)
          print(size)


# In[13]:


# Process images in the train folder
print("Processing train images...")
perform_object_leveling('train/images/*', 'train/labels/*', 'train/leveled_/images/', 'train/leveled_/labels/')


# In[14]:


# Process images in the test folder
print("Processing test images...")
perform_object_leveling('test/images/*', 'test/labels/*', 'test/leveled_/images/', 'test/leveled_/labels/')


# In[15]:



# Process images in the validation folder
print("Processing validation images...")
perform_object_leveling('valid/images/*', 'valid/labels/*', 'valid/leveled_/images/', 'valid/leveled_/labels/')


# In[16]:


TRAIN = True
def set_res_dir():
    # Directory to store results
    res_dir_count = len(glob.glob('runs/train/*'))
    print(f"Current number of result directories: {res_dir_count}")
    if TRAIN:
        RES_DIR = f"results_{res_dir_count+1}"
        print(RES_DIR)
    else:
        RES_DIR = f"results_{res_dir_count}"
    return RES_DIR


# In[17]:


def monitor_tensorboard():
    get_ipython().run_line_magic('load_ext', 'tensorboard')
    get_ipython().run_line_magic('tensorboard', '--logdir runs/train')


# #***Cloning YOLOV5 & Installing required packages through requirements***

# In[18]:


if not os.path.exists('yolov5'):
    get_ipython().system('git clone https://github.com/ultralytics/yolov5.git')


# In[19]:


get_ipython().run_line_magic('cd', 'yolov5/')
get_ipython().system('pwd')


# In[20]:


get_ipython().system('pip install -r requirements.txt')


# #***Training the YOLOV5m6 model***

# In[21]:


# RES_DIR = set_res_dir()
# if TRAIN:
#     !python train.py --data ../data.yaml --weights yolov5s.pt \
#     --img 640 --epochs {EPOCHS} --batch-size 16 --name {RES_DIR}

RES_DIR = set_res_dir()
if TRAIN:
    get_ipython().system('python train.py --data ../data.yaml --weights yolov5m6.pt     --img 640 --epochs {100} --batch-size 25 --name {RES_DIR}')


# In[22]:


monitor_tensorboard()


# In[23]:


# Function to show validation predictions saved during training.
def show_valid_results(RES_DIR):
    get_ipython().system('ls runs/train/{RES_DIR}')
    EXP_PATH = f"runs/train/{RES_DIR}"
    validation_pred_images = glob.glob(f"{EXP_PATH}/*_pred.jpg")
    print(validation_pred_images)
    for pred_image in validation_pred_images:
        image = cv2.imread(pred_image)
        plt.figure(figsize=(19, 16))
        plt.imshow(image[:, :, ::-1])
        plt.axis('off')
        plt.show()


# In[24]:


# Helper function for inference on images.
def inference(RES_DIR, data_path):
    # Directory to store inference results.
    infer_dir_count = len(glob.glob('runs/detect/*'))
    print(f"Current number of inference detection directories: {infer_dir_count}")
    INFER_DIR = f"inference_{infer_dir_count+1}"
    print(INFER_DIR)
    # Inference on images.
    get_ipython().system('python detect.py --weights runs/train/{RES_DIR}/weights/last.pt     --img 640 --conf 0.4 --source {dataset.location}/test/images --name {INFER_DIR}')
    return INFER_DIR


# In[25]:


def visualize(INFER_DIR):
# Visualize inference images.
    INFER_PATH = f"runs/detect/{INFER_DIR}"
    infer_images = glob.glob(f"{INFER_PATH}/*.jpg")
    print(infer_images)
    for pred_image in infer_images:
        image = cv2.imread(pred_image)
        plt.figure(figsize=(19, 16))
        plt.imshow(image[:, :, ::-1])
        plt.axis('off')
        plt.show()


# In[26]:


show_valid_results(RES_DIR)


# #***Image Inferencing***

# In[27]:


IMAGE_INFER_DIR = inference(RES_DIR, 'inference_images')
visualize(IMAGE_INFER_DIR)


# In[ ]:


os.getcwd()

# from google.colab import drive
# drive.mount('/content/drive')
# %cd /content/drive/MyDrive/Colab Notebooks/
get_ipython().run_line_magic('cp', '/content/drive/MyDrive/group_project/yolov5 /content/drive/MyDrive/Colab Notebooks/')


# !jupyter nbconvert --to html Team_A_Capstone_Project.ipynb


# # ***Text Detection: Google's Tesseract***

# In[ ]:


# # !pip install opencv-python pytesseract
# !sudo apt install tesseract-ocr


# In[ ]:


# !pip install pytesseract


# In[ ]:


# import cv2
# import os,argparse
# import pytesseract
# from PIL import Image
# from google.colab.patches import cv2_imshow


# In[ ]:


# def read_text_Img():
#     EXP_PATH = "runs/detect/inference_1"
#     test_images = glob.glob(f"{EXP_PATH}/*.jpg")
#     print(test_images)
#     for pred_image in test_images:
#         images = cv2.imread(pred_image)

#     # #We then Construct an Argument Parser
#     # ap=argparse.ArgumentParser()
#     # ap.add_argument("-i","--image",
#     #                 required=True,
#     #                 help="Path to the image folder")
#     # ap.add_argument("-p","--pre_processor",
#     #                 default="thresh",
#     #                 help="the preprocessor usage")
#     # args=vars(ap.parse_args())

#     # #We then read the image with text
#     # images=cv2.imread(args["image"])

#     #convert to grayscale image
#     gray=cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)


#     #memory usage with image i.e. adding image to memory
#     filename = "{}.jpg".format(os.getpid())
#     cv2.imwrite(filename, gray)
#     text = pytesseract.image_to_data(Image.open(filename))
#     os.remove(filename)
#     n_boxes = len(text['level'])
#     for i in range(n_boxes):
#         (x, y, w, h) = (text['left'][i], text['top'][i], text['width'][i], text['height'][i])
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

#     cv2_imshow(img)
#     cv2.waitKey(0)
#     print('Test:',text)

#     # show the output images
#     # cv2.imshow("Image Input", images)
#     # cv2.imshow("Output In Grayscale", gray)
#     # cv2.waitKey(0)


# In[ ]:


# read_text_Img()


# In[ ]:




