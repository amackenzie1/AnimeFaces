import random 
import numpy as np
from tqdm import tqdm
import os
import cv2 
import matplotlib.pyplot as plt 
from time_embedding import embedding

folder = "/mnt/local/amacke26/images"
# folder = "/mnt/local/amacke26/portraits"

SIZE = 8196
RES = 64
T = 1000
s = 0.008

def f(t):
    return np.cos((t/T + s)/(1+s) * np.pi/2)**2

def alpha(t):
    return f(t)/f(0)

def beta(t):
    return min(1 - alpha(t)/alpha(t-1), 0.999)

def center_crop(img):
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return cv2.resize(img, (RES, RES))
    except Exception as e:
        print(e)

def generator():
    filenames = os.listdir(folder)
    while True:
        filename = random.choice(filenames)
        x = center_crop(cv2.imread(folder+"/"+filename)) 
        if x is None:
            continue
        if x.shape[0] < RES or x.shape[1] < RES:
            continue
        x = 2*(x - 255.0/2)/255.0
        e = np.random.normal(size=(RES, RES, 3))
        t = np.random.randint(1, T)
        x = np.sqrt(alpha(t)) * x + np.sqrt(1 - alpha(t)) * e
        t = embedding(t)
        yield {"input": x, "time": t}, e
