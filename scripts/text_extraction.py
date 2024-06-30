from text_detection import load_model, detect_text
import pytesseract
from PIL import Image
import cv2
import preprocess

'''
Preprocesses and extracts text from a given image
    Returns: a list of each string extraction from the preprocessed images. 
'''
def extract_text(img_path):
    # open image
    base_img = cv2.imread(img_path)

    # retrieve blobs from image
    blobs = detect_text(img_path)

    # create array of preprocessed images and an array of their data 
    preprocessed_imgs = [preprocess.process_standard(base_img),
                         preprocess.process_contrast(base_img),
                         preprocess.process_gradient(base_img)]
    text_data = []
    for i in range(len(preprocessed_imgs)):
        text_data.append([])

    # apply ocr for each blob
    for blob in blobs:
        for i in range(len(preprocessed_imgs)):
            img = preprocessed_imgs[i]

            # crop image. slicing: y_min:y_max, x_min:x_max
            y_min = (blob[1] - 8) if (blob[1] - 8 >= 0) else 0
            y_max = (blob[3] + 8) if (blob[3] + 8 <= img.shape[0]) else (img.shape[0])
            cropped_img = img[y_min:y_max, blob[0]:blob[2]]

            # process and append to text data
            text_data[i].append(pytesseract.image_to_string(Image.fromarray(cropped_img)))
    
    # return text data
    extracted_text = []
    for i in range(len(preprocessed_imgs)):
        extracted_text.append('\n\n'.join(text_data[i]))
    return extracted_text

'''
main
'''
if __name__ == '__main__':
    # load model 
    load_model()

    # take in image path to analyze
    print("Enter image path:")
    img_path = input()

    all_text = extract_text(img_path)
    for text in all_text:
        print(text)
        print("\n\n-----------------\n\n")