import cv2
import numpy as np

'''
All image preprocessing functions to be used before passing in images
to pytesseract for text extraction. 
'''

# adaptive threshold, good for shadows
def adaptive_threshold(img):
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 2)

# noise removal
def denoise(img, blur = 5):
    return cv2.medianBlur(img,blur)

# convert to greyscale 
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# increase image contrast
def increase_contrast(img, grid_size = 30):
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(grid_size,grid_size))

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l, a, b = cv2.split(lab)                    # split on 3 different channels

    l2 = clahe.apply(l)                         # apply CLAHE to the L-channel

    lab = cv2.merge((l2,a,b))                   # merge channels
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
    return img

# morph image
def morph(img):
    kernel = np.ones((3,3), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

# open image
def open(img):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

# threshold
def threshold(img):
    return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# workds for most standard images
def process_standard(img):
    img = denoise(img)
    img = grayscale(img)
    img = threshold(img)
    img = denoise(img)
    return img

# workds for images with a low text/background contrast
def process_contrast(img):
    img = denoise(img)
    img = increase_contrast(img)
    img = grayscale(img)
    img = threshold(img)
    img = denoise(img)
    return img

# workds for images with whose lighting needs to be evened out
def process_gradient(img):
    img = grayscale(img)
    img = denoise(img, 9)
    img = open(img)
    img = adaptive_threshold(img)
    img = threshold(img)
    img = denoise(img,9)
    return img