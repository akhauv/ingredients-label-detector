import numpy as np
import cv2
import pytesseract
from PIL import Image

img = cv2.imread('data/raw_images/14.jpeg')

# convert to greyscale 
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# noise removal
def denoise(img, blur = 5):
    return cv2.medianBlur(img,blur)

def morph(img):
    kernel = np.ones((3,3), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

# threshold
def threshold(img):
    return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# dilation
def dilate(img):
    kernel = np.ones((5,5), np.uint8)
    return cv2.dilate(img, kernel, iterations = 1)

# erosion
def erode(img):
    kernel = np.ones((5,5), np.uint8)
    return cv2.dilate(img, kernel, iterations = 1)

def increase_contrast(img, grid_size = 30):
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(grid_size,grid_size))

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l, a, b = cv2.split(lab)  # split on 3 different channels

    l2 = clahe.apply(l)  # apply CLAHE to the L-channel

    lab = cv2.merge((l2,a,b))  # merge channels
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
    return img

def show_image(img):
    cv2.imshow('Image', img)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()  # Close the window

def adaptive_threshold(img):
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 2)

def opening(img):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def invert(img):
    return cv2.bitwise_not(img)


# workds for most standard images
def attempt_one(img):
    img = denoise(img)
    # show_image(img)
    img = grayscale(img)
    # show_image(img)
    img = threshold(img)
    # show_image(img)
    img = denoise(img)
    show_image(img)
    return img

# workds for images with a low text/background contrast
def attempt_two(img):
    # show_image(img)
    img = denoise(img)
    # show_image(img)
    img = increase_contrast(img)
    # show_image(img)
    img = grayscale(img)
    # show_image(img)
    img = threshold(img)
    # show_image(img)
    img = denoise(img)
    show_image(img)
    return img

# workds for images with a low text/background contrast
def attempt_two(img):
    # show_image(img)
    img = denoise(img)
    # show_image(img)
    img = increase_contrast(img)
    # show_image(img)
    img = grayscale(img)
    # show_image(img)
    img = threshold(img)
    # show_image(img)
    img = denoise(img)
    show_image(img)
    return img

# workds for images with whose lighting needs to be evened out
def attempt_three(img):
    # show_image(img)
    img = grayscale(img)
    # show_image(img)
    img = denoise(img, 9)
    # show_image(img)
    img = opening(img)
    # show_image(img)
    img = adaptive_threshold(img)
    # show_image(img)
    img = threshold(img)
    # show_image(img)
    img = denoise(img,9)
    show_image(img)
    return img

# workds for images with a low text/background contrast and need to be inverted
def attempt_four(img):
    # show_image(img)
    img = denoise(img)
    # show_image(img)
    img = increase_contrast(img, grid_size=50)
    # show_image(img)
    img = grayscale(img)
    # show_image(img)
    img = dilate(img)
    # show_image(img)
    img = adaptive_threshold(img)
    # show_image(img)
    img = invert(img)
    # show_image(img)
    img = morph(img)
    # show_image(img)
    img = erode(img)
    # show_image(img)
    return img



# img = increase_contrast(img)
# show_image(img)

# img = grayscale(img)
# show_image(img)

# img = threshold(img)
# show_image(img)

show_image(img)
print(pytesseract.image_to_string(Image.fromarray(attempt_one(img))))
print("\n\n\n-----------!!!!!------\n\n\n")
print(pytesseract.image_to_string(Image.fromarray(attempt_two(img))))
print("\n\n\n-----------!!!!!------\n\n\n")
print(pytesseract.image_to_string(Image.fromarray(attempt_three(img))))
print("\n\n\n-----------!!!!!------\n\n\n")
print(pytesseract.image_to_string(Image.fromarray(attempt_four(img))))