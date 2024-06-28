import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile    # used to read the frozen model file 

from text_detection_class import TextDetector

if __name__ == '__main__':
    # load model 
    model = TextDetector()

    # take in image path to analyze
    print("Enter image path:")
    img_path = input()