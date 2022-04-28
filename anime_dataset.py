import random 
import numpy as np
from tqdm import tqdm
import os
import cv2 
import matplotlib.pyplot as plt 

# folder = "/mnt/local/amacke26/images"
# folder = "/mnt/local/amacke26/portraits"

SIZE = 8196
RES = 32

image_names = os.listdir(folder)

def center_crop(img):
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return cv2.resize(img, (RES, RES))
        # return cv2.resize(img, (128, 128))
    except Exception as e:
        print(e)


def get_dataset():
    dataset = []
    filenames = os.listdir(folder)
    random.shuffle(filenames)
    filenames = filenames[:SIZE] 
    
    for i in tqdm(filenames):
      x = center_crop(cv2.imread(folder+"/"+i))
      if x is not None:
        dataset.append(x)
    
    dataset = np.array(dataset, dtype='float32')
    return 2*(dataset[:SIZE] - 255.0/2)/255.0
