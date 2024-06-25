# ingredients-label-detector

A machine learning model which recognizes and extracts the ingredients list from images of food product labels.

Intended workflow:

1. Collect data
2. LabelImg to manually draw bounding boxes around ingredients lists from dataset and label them

New intended workflow:

1. extract all text from an image
2. use an NLP to identify the ingredients
