import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import net.utility.file as file
from config import cfg


file.makedirs(cfg.LOG_DIR)

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
    plt.imsave(os.path.join(cfg.LOG_DIR,name) ,image)

def npsave(name,numpy_array):
    np.save(os.path.join(cfg.LOG_DIR,name),numpy_array)