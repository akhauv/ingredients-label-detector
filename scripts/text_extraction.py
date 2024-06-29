from text_detection import load_model, detect_text
import pytesseract
from PIL import Image
import cv2
import preprocess

'''
main
'''
if __name__ == '__main__':
    # load model 
    load_model()

    # take in image path to analyze
    print("Enter image path:")
    img_path = input()
    img = cv2.imread(img_path)

    # retrieve blobs from image
    blobs = detect_text(img_path)

    # apply ocr for each blob
    for blob in blobs:
        cropped_img = img[blob[1]:blob[3], blob[0]:blob[2]]
        cv2.imshow('Image', cropped_img)
        cv2.waitKey(0)  # Wait for a key press to close the window
        cv2.destroyAllWindows()  # Close the window

        print(pytesseract.image_to_string(Image.fromarray(preprocess.process_standard(cropped_img))))


    # print(pytesseract.image_to_string(Image.fromarray(attempt_one(img))))
    # print("\n\n\n-----------!!!!!------\n\n\n")
    # print(pytesseract.image_to_string(Image.fromarray(attempt_two(img))))
    # print("\n\n\n-----------!!!!!------\n\n\n")
    # print(pytesseract.image_to_string(Image.fromarray(attempt_three(img))))
    # print("\n\n\n-----------!!!!!------\n\n\n")
    # print(pytesseract.image_to_string(Image.fromarray(attempt_four(img))))