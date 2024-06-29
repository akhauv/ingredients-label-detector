import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile    # used to read the frozen model file 
import sys
import os

sys.path.append(os.getcwd())
from lib.fast_rcnn.config import cfg, cfg_from_file
from lib.fast_rcnn.test import _get_blobs
from lib.text_connector.detectors import TextDetector
from lib.text_connector.text_connect_cfg import Config as TextLineCfg
from lib.rpn_msr.proposal_layer_tf import proposal_layer

from text_detection_class import LabelDetector

# DELETE THIS METHOD LATER
def return_blobs_tuple(boxes, scale):
    blob_list = []
    for box in boxes:
        min_x = min(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
        min_y = min(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))
        max_x = max(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
        max_y = max(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale)) 

        blob_list.append((min_x, min_y, max_x, max_y))

    return(tuple(blob_list))

def resize_img(img, scale, max_scale=None):
    f = float(scale) / min(img.shape[0], img.shape[1])
    if max_scale != None and f * max(img.shape[0], img.shape[1]) > max_scale:
        f = float(max_scale) / max(img.shape[0], img.shape[1])
    return cv2.resize(img, None, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR), f

def filter_boxes(boxes):
    # Initialize a boolean mask with true values
    mask = np.ones(len(boxes), dtype=bool)
    
    # Set mask to False for boxes to be filtered out
    for i, box in enumerate(boxes):
        if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
            mask[i] = False
    
    # Apply the mask to filter the boxes
    filtered_boxes = boxes[mask]
    return filtered_boxes

def get_coords(box):
    min_x = min(box[0], box[2], box[4], box[6])
    min_y = min(box[1], box[3], box[5], box[7])
    max_x = max(box[0], box[2], box[4], box[6])
    max_y = max(box[1], box[3], box[5], box[7])
    return min_x, min_y, max_x, max_y

def has_overlap(box_one, box_two):
    # vertical uncertainty is permissible becasue of thin lines
    uncertainty = 8

    one_x_min, one_y_min, one_x_max, one_y_max = get_coords(box_one)
    two_x_min, two_y_min, two_x_max, two_y_max = get_coords(box_two)

    # check to see if they are overlapping
    if (one_x_max < two_x_min) or (two_x_max < one_x_min):
        return False
    if (one_y_max + uncertainty < two_y_min) or (two_y_max + uncertainty < one_y_min):
        return False

    return True

def merge_boxes(box_one, box_two):
    one_x_min, one_y_min, one_x_max, one_y_max = get_coords(box_one)
    two_x_min, two_y_min, two_x_max, two_y_max = get_coords(box_two)

    new_x_min = min(one_x_min, two_x_min)
    new_x_max = max(one_x_max, two_x_max)
    new_y_min = min(one_y_min, two_y_min)
    new_y_max = max(one_y_max, two_y_max)

    confidence_score = max(box_one[8], box_two[8])

    return np.array([new_x_min, new_y_min, new_x_max, new_y_min, new_x_min, new_y_max, new_x_max, new_y_max, confidence_score])

def merge_overlapping_boxes(boxes):
    # the loop will reach its end once no overlap occurs 
    i = 0
    while i < len(boxes):
        j = 0
        while j < len(boxes):
            print(i, j)
            if i == j:
                j += 1
                continue
            
            if has_overlap(boxes[i], boxes[j]):
                print("foudn overlap")
                # set the first box to the merged box
                boxes[i] = merge_boxes(boxes[i], boxes[j])

                # delte the second box 
                mask = np.ones(len(boxes), dtype=bool)
                mask[j] = False
                boxes = boxes[mask]

                # reset to start fro mbeginning again
                i = 0
                j = 0
                continue
            j += 1
        i += 1
    
    return boxes

def draw_boxes(img, image_name, boxes, scale):
    # extract name of img file from full path 
    base_name = image_name.split('/')[-1]

    # open a text file to write bounding box coordinates 
    with open('data/results/' + 'res_{}.txt'.format(base_name.split('.')[0]), 'w') as f:
        # filter out very small boxes 
        boxes = filter_boxes(boxes)

        # merge overlapping boxes
        boxes = merge_overlapping_boxes(boxes)

        for box in boxes:
            # assign colors based on confidence score
            if box[8] >= 0.9:
                color = (0, 255, 0)
            elif box[8] >= 0.8:
                color = (255, 0, 0)
            
            # draw box
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 1)
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[4]), int(box[5])), color, 1)
            cv2.line(img, (int(box[6]), int(box[7])), (int(box[2]), int(box[3])), color, 1)
            cv2.line(img, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), color, 1)

            # calculate coords based on original image scale to revert to original image dimesions 
            min_x = min(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
            min_y = min(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))
            max_x = max(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
            max_y = max(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))

            # write coordinates to result file 
            line = ','.join([str(min_x), str(min_y), str(max_x), str(max_y)]) + '\r\n'
            f.write(line)

    # resize image back to regular scale and save results image
    img = cv2.resize(img, None, None, fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join("data/results", base_name), img)

def detect_text(img_path):
    # load image and resize
    img = cv2.imread(img_path)
    img, scale = resize_img(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)

    # blobs: contains processed image data // img_scales: list of scaling foactors for the image
    blobs, img_scales = _get_blobs(img, None)

    # checks if model has a Region Proposal Network (RPN).
    if cfg.TEST.HAS_RPN:
        # extract blob and add to blobs dictionary
        img_blob = blobs['data']
        blobs['img_info'] = np.array(
            [[img_blob.shape[1],
              img_blob.shape[2],
              img_scales[0]]],
            dtype=np.float32)
    
    # run a tensorflow session to compute class probabilities and box predictions
    cls_prob, box_pred = model.extract_text_info(blobs)

    # generate region proposals based on classification probabilities and box predictions
    rois, _ = proposal_layer(cls_prob, box_pred, blobs['img_info'], 'TEST', anchor_scales=cfg.ANCHOR_SCALES)

    # get confidence scores and bounding box coordinates
    scores = rois[:, 0]
    boxes = rois[:, 1:5] / img_scales[0]

    # refine and get bounding boxes based on confidence scores. 
    textdetector = TextDetector()
    boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])

    # draw boxes and return information
    draw_boxes(img, img_path, boxes, scale)
    return(return_blobs_tuple(boxes, scale))

if __name__ == '__main__':
    # load model 
    global model
    model = LabelDetector()

    # take in image path to analyze
    print("Enter image path:")
    img_path = input()
    print("analyzing")

    detect_text(img_path)
    print("done!")