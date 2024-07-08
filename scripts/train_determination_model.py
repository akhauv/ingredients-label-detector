from transformers import MobileBertTokenizer, MobileBertForSequenceClassification, Trainer, TrainingArguments
import sys
import os
import torch
from text_dataset_class import TextDataset

sys.path.append(os.getcwd())
from data.training_data.training_data import text, num_valid

def train_dataset():
    # format label 
    train_labels = [1] * num_valid
    train_labels.extend([0] * (len(text) - num_valid))

    # establish model name and tokenizer
    model_name = "google/mobilebert-uncased"
    tokenizer = MobileBertTokenizer.from_pretrained(model_name)

    # create training dataset
    train_encodings = tokenizer(text, truncation=True, padding=True, max_length=128)
    train_dataset = TextDataset(train_encodings, train_labels)
    print("\nCreated dataset!\n\n\n")

    # load model
    model = MobileBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    device = torch.device("cpu") # "mps" if torch.backends.mps.is_available() else 
    model.to(device)
    print("\nloaded model!\n\n\n")

    # training parameters
    training_args = TrainingArguments(
        output_dir='./models',          # output directory
        num_train_epochs=3,              # total number of training epochs
        per_device_train_batch_size=16,  # batch size for training
        per_device_eval_batch_size=64,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./models/logs',     # directory for storing logs
        logging_steps=10,
        use_cpu=True
    )

    # initialize trainer
    trainer = Trainer(
        model=model,                         # the instantiated Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
    )
    print("\nCreated trainer!\n\n\n")

    # train model
    trainer.train()
    print("\ntrained model!\n\n\n")

    # Save the trained model and tokenizer
    model.save_pretrained('./models/trained')
    tokenizer.save_pretrained('./models/trained')


if __name__ == '__main__':
    train_dataset()