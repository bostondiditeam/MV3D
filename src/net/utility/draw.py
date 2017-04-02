import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

image_sava_dir='/home/stu/Development/MV3D/data/image_output/'
os.makedirs(image_sava_dir ,exist_ok=True)

def imshow(name, image, resize=1):
    H,W,_ = image.shape
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image.astype(np.uint8))
    cv2.resizeWindow(name, round(resize*W), round(resize*H))


def normalise(image, limit=255.0):
    image -= image.min()
    image *= (limit/image.max())
    return image

def imsave(name, image):
    plt.imsave(image_sava_dir+name,image)

def npsave(name,numpy_array):
    np.save(image_sava_dir+name,numpy_array)