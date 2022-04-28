import matplotlib.pyplot as plt
import tensorflow as tf 
import numpy as np
from tqdm import tqdm
from anime_model import get_model
from time_embedding import embedding

T = 1000
s = 0.008
ROOT_NUM = 5
RES = 64

def f(t):
    return np.cos((t/T + s)/(1+s) * np.pi/2)**2

def alpha(t):
    return min(f(t)/f(0), 0.9999)

def beta(t):
    return min(1 - alpha(t)/alpha(t-1), 0.9999)

model = get_model()

# model.load_weights("weights/v3").expect_partial()
model = tf.keras.models.load_model("tmp/ckpt")

x = np.random.normal(size=(ROOT_NUM**2, RES, RES, 3))

fig, ax = plt.subplots(ROOT_NUM, ROOT_NUM)

for t in tqdm(range(T-2, 1, -1)):
    x = 1/np.sqrt(1 - beta(t)) * (x - ((beta(t))/np.sqrt(1 - alpha(t))*model([x, np.array([embedding(t)])])))
    if t > 20:
       x += np.random.normal(size=(ROOT_NUM**2, RES, RES, 3)) * np.sqrt(beta(t))

for i in range(ROOT_NUM**2):
    image = (255.0/2) + (255.0/2) * x[i]
    image = np.array(image, dtype='int') 
    print(image)
    ax[i//ROOT_NUM, i%ROOT_NUM].imshow(image)

fig.savefig("test_image.png")
