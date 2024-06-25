# ingredients-label-detector

A machine learning model which recognizes and extracts the ingredients list from images of food product labels.

Intended workflow:

1. Collect data
2. LabelImg to manually draw bounding boxes around ingredients lists from dataset and label them

New intended workflow:

I planned to train a Faster-RCNN model on my own data, but looking into how Faster-RCNNS work, I realized how this was not possible. Ingredients labels are not easily distinguishable from other text in an image, and they cannot be identified as potential 'objects' to further analyze w/ the faster-RCNN technique of objet reason proposals. Now, I am moving forward with a new workflow:

1. extract all text from an image
2. use an NLP to identify the ingredients
