from text_detection import load_detection_model
from text_extraction import extract_text
from text_determination import predict, load_determination_model, process_text_chunk
import re

'''
Ingredients detection is split into three parts:
    1) Text detection: determining locations of text chunks in the image
    2) Text extraction: pulling text from text chunks into strings
    3) Text determination: determining which of the strings corresponds to the ingredietns list.
This script combines each part of the pipeline.
'''

def clean_ingredients(ingredients):
    cleaned_ingredients = ingredients.lower()

    # get rid of any newlines 
    cleaned_ingredients = cleaned_ingredients.replace("-\n", "")
    cleaned_ingredients = cleaned_ingredients.replace("\n", " ")

    # strip anything after the "may contain" and "contains" keyword
    contains_ind = cleaned_ingredients.find("may contain")
    if contains_ind != -1:
        cleaned_ingredients = cleaned_ingredients[:contains_ind]
    contains_ind = cleaned_ingredients.find("contains")
    if contains_ind != -1:
        cleaned_ingredients = cleaned_ingredients[:contains_ind]
    
    # strip any unwanted characters 
    pattern = r'[^A-Za-z0-9\-()\[\]{}/%:,.\s]'
    cleaned_ingredients = re.sub(pattern, '', cleaned_ingredients)
    cleaned_ingredients = cleaned_ingredients.replace("-", "")
    cleaned_ingredients = cleaned_ingredients.replace(".", ",")

    # get rid of trailing commas
    cleaned_ingredients = cleaned_ingredients.rstrip(',')

    # add spaces after commas which don't have them
    cleaned_ingredients = re.sub(r',(?=\S)', ', ', cleaned_ingredients)

    return cleaned_ingredients

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
            best_ingredients = "".join(ingredients)
            best_len = len(ingredients)
    
    # strip string to clean it
    cleaned_ingredients = clean_ingredients(best_ingredients)

    return cleaned_ingredients

if __name__ == '__main__':
    # load text detection and determination models
    load_detection_model()
    load_determination_model()

    # take in image path to analyze
    print("Enter image path:")
    img_path = input()

    ingredients = get_ingredients_list(img_path)
    print(ingredients)

