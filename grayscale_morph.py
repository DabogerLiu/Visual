import cv2
import numpy as np
import time

def gray_dilate(img, kernel):
    '''
    Args:
        img: input image
        kernel: structuring element
    Return:
        Result image
    '''
    operated_img = img.copy()
    for (x, y, window) in sliding_window(img, stepSize = 1, windowSize=(kernel.shape[0], kernel.shape[1])):
        r = kernel.shape[0]//2
        if y == 0 and x == 0:
            window = np.pad(window, ((r, 0), (r, 0)), 'minimum')
        elif y == 0:
            window = np.pad(window, ((r, 0), (0, 0)), 'minimum')
        elif x == 0:
            window = np.pad(window, ((0, 0), (r, 0)), 'minimum')

        if y == img.shape[0]-1 and x == img.shape[1]-1:
            window = np.pad(window, ((0, r), (0, r)), 'minimum')
        elif y == img.shape[0]-1:
            window = np.pad(window, ((0, r), (0, 0)), 'minimum')
        elif x == img.shape[1]-1:
            window = np.pad(window, ((0, 0), (0, r)), 'minimum')

        #print(kernel.shape)
        #print(window.shape)
        re_this = np.max(kernel + window)
        operated_img[y, x] = re_this

    return operated_img

def gray_erode(img, kernel):
    '''
    Args:
        img: input image
        kernel: structuring element
    Return:
        Result image
    '''
    operated_img = img.copy()
    for (x, y, window) in sliding_window(img, stepSize = 1, windowSize=(kernel.shape[0], kernel.shape[1])):
        r = kernel.shape[0]//2
        if y == 0 and x == 0:
            window = np.pad(window, ((r, 0), (r, 0)), 'minimum')
        elif y == 0:
            window = np.pad(window, ((r, 0), (0, 0)), 'minimum')
        elif x == 0:
            window = np.pad(window, ((0, 0), (r, 0)), 'minimum')

        if y == img.shape[0]-1 and x == img.shape[1]-1:
            window = np.pad(window, ((0, r), (0, r)), 'minimum')
        elif y == img.shape[0]-1:
            window = np.pad(window, ((0, r), (0, 0)), 'minimum')
        elif x == img.shape[1]-1:
            window = np.pad(window, ((0, 0), (0, r)), 'minimum')

        #print(kernel.shape)
        #print(window.shape)
        re_this = np.min(kernel + window)
        operated_img[y, x] = re_this

    return operated_img


def sliding_window(image, stepSize, windowSize):
    radius = windowSize[0] // 2
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            y_start, x_start = y-radius, x-radius
            y_end, x_end = y+radius+1, x+radius+1
            if y_start<0 and x_start<0:
                yield (x, y, image[y:y_end, x:x_end])
            elif y_start<0:
                yield (x, y, image[y:y_end, x_start:x_end])
            elif x_start<0:
                yield (x, y, image[y_start:y_end, x:x_end])
            else:
                yield (x, y, image[y_start:y_end, x_start:x_end])

#kernel = np.arange(9).reshape((3,3)).astype(np.uint8)
#img = ((np.arange(784).reshape((28, 28))+100)*1.0).astype(np.float32)
#dilated_img = gray_dilate(img, kernel)
#print(dilated_img)
