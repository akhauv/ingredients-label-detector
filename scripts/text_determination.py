from transformers import MobileBertForSequenceClassification, MobileBertTokenizer
import torch

'''
Loads model
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
Splits the text if it exceeds the number of tokens 
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

def process_text_chunk(text):
    # split text based on lines and max # of tokens
    text_arr = text.splitlines()
    text_arr = split_to_tokensize(text_arr)
    return text_arr

if __name__ == '__main__':
    # load model
    load_determination_model()

    text = "Cultures, enzymes, annatto extract color).\nContains 2% or less of salt, whey, paprika, monosodium\nglutamate, buttermilk, parmesan cheese (milk, cheese cultures,\nSalt, enzymes), cheddar cheese (milk, cheese cuttures, salt,\nenzymes), garlic powder, yeast, tomato powder, sugar, onion"
    text_arr = process_text_chunk(text)

    # Get predictions
    ingredients, confidence_scores = predict(text_arr)
    print(confidence_scores)
    print(ingredients)

