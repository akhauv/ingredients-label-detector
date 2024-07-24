from symspellpy.symspellpy import SymSpell, Verbosity
import re

def initialize_symspell():
    # Create SymSpell object
    max_edit_distance_dictionary = 2    # determines how many character changes are allowed
    prefix_length = 7                   # first 7 characters of each term to build the prefix-based index
    global symspell
    symspell = SymSpell(max_edit_distance_dictionary, prefix_length)

    # Load the dictionary
    dictionary_path = "./scripts/ingredients_dictionary.txt"
    term_index = 0  # Column index for the term
    count_index = 1  # Column index for the term frequency
    separator = '$'
    symspell.load_dictionary(dictionary_path, term_index, count_index, separator)

def correct_ingredients_list(text):
    phrases = re.split(r',\s*', text)
    string_builder = []

    for phrase in phrases:
        corrected_phrase = correct_phrase(phrase)
        string_builder.append(corrected_phrase)
    
    corrected_text = ', '.join(string_builder)
    return corrected_text

def correct_phrase(phrase):
    words = phrase.split()
    corrected_words = []
    
    for word in words:
        if len(word) <= 3:
            corrected_words.append(word)
            continue

        # Look up the word in the dictionary
        suggestions = symspell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
        if suggestions:
            # Use the most likely correction
            corrected_words.append(suggestions[0].term)
        else:
            # If no suggestions, use the original w ord
            corrected_words.append(word)
    
    corrected_phrase = ' '.join(corrected_words)
    return corrected_phrase

if __name__ == '__main__':
    # load model 
    initialize_symspell()

    # take in text
    print("Enter text:")
    text = input()

    # request boundign boxes
    print(correct_ingredients_list(text))