from os import listdir
from PIL import Image
import numpy as np
import pandas as pd
from matplotlib import pyplot
import keras
import tensorflow as tf

# Image Path
dir = "archive/img_align_celeba/img_align_celeba/"

# Open CSVs
attr = pd.read_csv("archive/list_attr_celeba.csv")
# bbox = pd.read_csv("archive/list_bbox_celeba.csv")
# eval = pd.read_csv("archive/list_eval_partition.csv")

# Print Entire Array
# np.set_printoptions(threshold=np.inf)


# Open single image, return as nparray
# Resize to 100x122
def open_img(fn):
    img = Image.open(fn)
    img = img.resize((80,80))
    img_as_array = np.asarray(img)
    return img_as_array

# Open all images, return as nparray
# Filter out Glasses and Hats in the process
def open_imgs(dir):
    imgs = []
    i = 0
    for image in listdir(dir):
        if (attr.iloc[i]["Eyeglasses"] == 1 or attr.iloc[i]["Wearing_Hat"] == 1):
            i += 1
            continue
        img = open_img(dir + image)
        imgs.append(img)
        print(i)
        i += 1
    imgs = np.asarray(imgs)
    return imgs

# Generate dataset as array
dataset = open_imgs(dir)
# Save dataset arrayfile
np.save('dataset_processed', dataset)

# Load Dataset Array
dataset = np.load("dataset_processed.npy")

# Create Discriminator
def create_dis():
    model = keras.models.Sequential()

    model.add(keras.layers.Conv2D(128, (5,5), padding='same', input_shape=(122,100,3)))
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Conv2D(128, (5,5), strides=(2,2), padding='same'))
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Conv2D(128, (5, 5), strides=(2,2), padding='same'))
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Conv2D(128, (5, 5), strides=(2,2), padding='same'))
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Conv2D(128, (5, 5), strides=(2,2), padding='same'))
    model.add(keras.layers.LeakyReLU())

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    opti = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opti, metrics=['accuracy'])
    return model

# Create Generator
def create_gen(dim_lat):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(128 * 5 * 5, input_dim=dim_lat))
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Reshape((5,5,128)))

    model.add(keras.layers.Conv2DTranspose(128, (4, 4), strides=(2,2), padding="same"))
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Conv2DTranspose(128, (4, 4), strides=(2,2), padding="same"))
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Conv2DTranspose(128, (4, 4), strides=(2,2), padding="same"))
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Conv2DTranspose(128, (4, 4), strides=(2,2), padding="same"))
    model.add(keras.layers.LeakyReLU())

    model.add((keras.layers.Conv2D(3, (5,5), activation='tanh', padding='same')))
    return model