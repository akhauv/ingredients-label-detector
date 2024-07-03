from transformers import MobileBertForSequenceClassification, Trainer, TrainingArguments

# Load the model
model = MobileBertForSequenceClassification.from_pretrained("google/mobilebert-uncased", num_labels=2)

# Define the training arguments
training_args = TrainingArguments(
    output_dir='../models',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='../models/logs',
    logging_steps=10,
)

# load dataset 
# Assume dataset is a Hugging Face dataset with columns 'text' and 'label'
train_dataset = ...
eval_dataset = ...

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Fine-tune the model
trainer.train()