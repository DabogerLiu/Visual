import os
from skimage.transform import resize
import numpy as np
from morph import dilate, erode, adapt
import keras
from keras.datasets import mnist
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Conv2D, Activation
from keras import optimizers
from keras.models import Model
import matplotlib.pyplot as plt


def get_y(x):
    y = []
    kernel1=np.zeros((3,3), np.uint8)
    kernel1[1,0]=1
    kernel1[1,1]=1
    kernel1[1,2]=1
    kernel1[0,1]=1
    kernel1[2,1]=1

    kernel3 = np.zeros((3,3), np.uint8)
    kernel3[0,1]=1
    kernel3[1,1]=1
    kernel3[2,1]=1

    kernel2=np.zeros((3,3),np.uint8)*10
    kernel2[1,0]=1
    kernel2[1,1]=1
    kernel2[1,2]=1

    kernel2 = np.ones((5,5), np.uint8)
    kernel2[0,0]=0
    kernel2[0,1]=0
    kernel2[1,0]=0
    kernel2[0,3]=0
    kernel2[0,4]=0
    kernel2[1,4]=0
    #kernel1 = kernel2
    #kernel2[3,0]=kernel2[3,4]=0
    #kernel2[4,0]=kernel2[4,1]=kernel2[4,3]=kernel2[4,4]=0

    #kernel3 = np.zeros((5,5), np.uint8)
    #kernel3[0,4]=kernel3[1,3]=kernel3[2,2]=kernel3[3,1]=kernel3[4,0]=1
    #kernel3[0,0]=kernel3[1,1]=kernel3[3,3]=kernel3[4,4]=1

    #kernel1 = np.ones((1,5), np.uint8)

    #kernel3[1,1]=1
    #kernel3[2,1]=1
    #kernel1=np.random.random((3,3))
    #print(kernel1)
    #kernel2=np.random.random((3,3))
    #print(kernel2)
    #kernel3=np.random.random((3,3))
    #print(kernel3)
    #kernel4 = np.random.random((3,3))
    #print(kernel4)
    #print(kernel)
    #kernel = np.zeros((3,3), np.uint8)
    #kernel[0,0]=kernel[1,1]=kernel[2,2]=1
    #print(kernel2)
    #print(kernel2)
    #print(kernel3)
    #gaussian_kernel = cv2.getGaussianKernel(5,5)
    #print(gaussian_kernel)
    for imgno in range(x.shape[0]):
        #x1 = cv2.erode(x[imgno],kernel1, iterations=1)
        #x2 = cv2.dilate(x1,kernel2,iterations=1)
        #x3 = cv2.dilate(x2,kernel3,iterations=1)
        #y.append(cv2.morphologyEx(x[i, cv2.MORPH_OPEN, kernel2))
        #print(x[imgno][:, :, 0].shape)
        #y.append(gray_dilate(x[imgno][:,:,0], kernel1)[:, :, np.newaxis])
        x1 = cv2.dilate(x[imgno], kernel1)
        y.append(cv2.erode(x1, kernel1))
        #x1 = cv2.filter2D(x[imgno], -1, kernel1)
        #x2 = cv2.filter2D(x1, -1, kernel2)
        #y.append(cv2.filter2D(x2, -1, kernel3))
    y = np.array(y)
    y = y.reshape(y.shape[0], 28, 28, 1)
    #y = y.astype('float32')/ 255.
    return y, kernel1

def indicator(x):
    result = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    return result
