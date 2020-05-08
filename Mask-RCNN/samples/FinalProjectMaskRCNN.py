#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Needed libraries
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
import warnings
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from samples.coco import coco

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Ignore the filter warnings
warnings.filterwarnings("ignore")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Local path to trained weights file
COCO_MODEL_PATH = os.path.join('', "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


# In[3]:


# Used for batch size
# How many gpus the computer has
NUM_GPUS = 1
# How many images to detect per a gpu
NUM_IMGS_PER_GPU = 10

# Set options for batch sizes
class InferenceConfig(coco.CocoConfig):
    # Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = NUM_GPUS
    IMAGES_PER_GPU = NUM_IMGS_PER_GPU

# Configure Mask RCNN to run on above settings
config = InferenceConfig()
config.display()


# In[4]:


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir='mask_rcnn_coco.hy', config=config)

# Load weights trained on MS-COCO
model.load_weights('mask_rcnn_coco.h5', by_name=True)


# In[5]:


# COCO Class names
# Only used when displaying the resulting images
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


# In[6]:


# Load all the images from the images folder
images = []
# Iterate through each image in the images directory
for filename in os.listdir('images'):
    # Get the image
    img = cv2.imread(os.path.join('images',filename))
    # Append the image to the array
    if img is not None:
        images.append(img)


# In[7]:


# Run detection on images
# Holds results
results = []
# Holds image batch
imgs = []
# Iterate through each image
for image in images:
    # Append image to the batch
    imgs.append(image)
    # If batch size is hit, run detection
    if len(imgs) == NUM_GPUS * NUM_IMGS_PER_GPU:
        # Run detection
        r = model.detect(imgs, verbose=1)
        # Save results
        for res in r:
            results.append(res)
        # Reset the batch
        imgs = []


# In[8]:


# Visualize Results
for image, result in zip(images, results):
    visualize.display_instances(image, result['rois'], result['masks'], result['class_ids'], class_names, result['scores'])


# In[9]:


total = 0
i = 0
imgCnt = 0
for result in results:
    imgCnt += 1
    for score in result['scores']:
        i += 1
        total += score
print('Average confidence score:    {}%'.format(round(total/i, 4)*100))
print('Total class instances found: {} instances in {} images'.format(i, imgCnt))


# In[ ]:




