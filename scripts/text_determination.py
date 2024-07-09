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
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    confidence_scores, _ = torch.max(probabilities, dim=-1)
    return confidence_scores

if __name__ == '__main__':
    # load model
    load_model()

    testing_texts = ['Contains 2% or less of salt, whey, paprika, monosodium',
                     'glutamate, buttermilk, parmesan cheese (milk, cheese cultures']

    # Get predictions
    confidence_scores = predict(testing_texts)
    print(confidence_scores)
