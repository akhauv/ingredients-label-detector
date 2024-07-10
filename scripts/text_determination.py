import torch
from transformers import MobileBertForSequenceClassification, MobileBertTokenizer

'''
Determines whether inputted text classifies as an ingredients list or not using a fine-tuned
mobileBert model. To use this, ensure that you have a mobileBert model and tokenizer trained on the 
given data/training_data dataset in a models/trained folder.  

When run, user is prompted for a text line for which the model will output the ingredients list 
(if it is so) and confidence scores.
'''

'''
Loads model
    Returns: nothing
'''
def load_determination_model():
    global determination_model
    determination_model = MobileBertForSequenceClassification.from_pretrained('./models/trained')
    
    global determination_tokenizer
    determination_tokenizer = MobileBertTokenizer.from_pretrained('./models/trained')

'''
Predicts probability of texts being ingredients list
    Returns: a list of the confidence scores of each text 
'''
def predict(texts):
    # tokenizes the input texts 
    encodings = determination_tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors='pt')

    # prevents gradient calculation
    with torch.no_grad():
        outputs = determination_model(**encodings)
    
    # extracts raw output scores 
    logits = outputs.logits

    # converts logits into probabilities and extracts maximum probability from each
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    confidence_scores, predictions = torch.max(probabilities, dim=-1)

    # masks texts and confidnce scroes based on probability 
    mask = predictions == 1
    filtered_confidence_scores = confidence_scores[mask]
    filtered_texts = [text for text, include in zip(texts, mask) if include]
    return filtered_texts, filtered_confidence_scores.tolist()

'''
processes a given text chunk into analyzable lines
    Returns: processed text list
'''
def process_text_chunk(text):
    # split text based on lines and max # of tokens
    text_arr = text.splitlines()
    text_arr = split_to_tokensize(text_arr)
    return text_arr

'''
Splits the text if it exceeds the number of tokens.
    Returns: split text list 
'''
def split_to_tokensize(texts, max_length = 512):
    # the final list of all split texts 
    new_texts = []

    for text in texts:
        # get the number of tokens for each
        encodings = determination_tokenizer(text, return_tensors='pt')
        tokens = encodings['input_ids'][0]

        # add text to new_texts and break if text satisfies tokens 
        if len(tokens) <= max_length:
            new_texts.append(text)
            continue

        # split text into chunks of managemable tokens 
        chunks = []
        for i in range(0, len(tokens), max_length):
            chunk_tokens = tokens[i:i + max_length]
            chunk_text = determination_tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)
        new_texts.extend(chunks)

    return new_texts

'''
main
'''
if __name__ == '__main__':
    # load model
    load_determination_model()

    # take in image path to analyze
    print("Enter text to determine:")
    text = input()
    text_arr = process_text_chunk(text)

    # Get predictions
    ingredients, confidence_scores = predict(text_arr)
    print(ingredients)
    print(confidence_scores)

