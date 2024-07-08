from transformers import MobileBertForSequenceClassification, MobileBertTokenizer
import torch

def load_model():
    global determination_model
    determination_model = MobileBertForSequenceClassification.from_pretrained('./models/trained')
    global determination_tokenizer
    determination_tokenizer = MobileBertTokenizer.from_pretrained('./models/trained')

def predict(texts):
    encodings = determination_tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
    with torch.no_grad():
        outputs = determination_model(**encodings)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    return predictions

if __name__ == '__main__':
    # load model
    load_model()

    testing_texts = ['butter, flour, eggs, milk, cocao powde, sladfjhlskdjfnnvnvnm, sdfska, aaa']

    # Get predictions
    predictions = predict(testing_texts)
    print(predictions)