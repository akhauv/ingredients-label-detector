import os
import sys

sys.path.append(os.getcwd())
from scripts.text_detection import load_detection_model
from scripts.text_determination import predict, load_determination_model, process_text_chunk
from scripts.text_extraction import extract_text
from scripts.text_correction import initialize_symspell, clean_ingredients, correct_ingredients_list

'''
Ingredients detection is split into three parts:
    1) Text detection: determining locations of text chunks in the image
    2) Text extraction: pulling text from text chunks into strings
    3) Text determination: determining which of the strings corresponds to the ingredietns list.
This script combines each part of the pipeline.

When run, user can input an image path and have a list of ingredients returned to them. 
'''

'''
Extracts ingredients list from an image
'''
def get_ingredients_list(img_path):
    # extract text 
    all_text = extract_text(img_path)

    # review the text chunks from each image.
    # if there are multiple results, 
    best_ingredients = ""
    best_len = 0
    for text in all_text:
        # split text based on lines and max # of tokens
        text_arr = process_text_chunk(text)

        # get predictions 
        ingredients, _ = predict(text_arr)
        if len(ingredients) > best_len:
            best_ingredients = " ".join(ingredients)
            best_len = len(ingredients)
    
    # strip string to clean it
    cleaned_ingredients = clean_ingredients(best_ingredients)

    # run through spellcheck
    final_result = correct_ingredients_list(cleaned_ingredients)
    return final_result

'''
main
'''
if __name__ == '__main__':
    # load text detection and determination models
    load_detection_model()
    load_determination_model()
    initialize_symspell()

    # take in image path to analyze
    print("Enter image path:")
    img_path = input()

    ingredients = get_ingredients_list(img_path)
    print(ingredients)