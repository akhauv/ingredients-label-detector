# ingredients-label-detector

## How to Run

1. Clone this repository
2. Download the frozen cptn file for text line recognition <a href="https://github.com/eragonruan/text-detection-ctpn/releases/download/untagged-48d74c6337a71b6b5f87/ctpn.pb">here</a>
3. Save the model in `./data`
4. Add any desired testing images in `./data/raw_images`
4. Run `python ./scripts/run_ingredients_detection.py` and answer terminal
   prompts.

If you want to run individual stages of the ingrededients detection, you can
also run: 
- `python ./scripts/text_detection.py` for text detection
- `python ./scripts/text_extraction.py` for text extraction
- `python ./scripts/text_determination.py` for text determination: the classification of text lines as
  components of an ingredients list.

## Pipeline

There are three parts to the ingredients extraction process: text detection, 
text extraction (OCR), and ingredients identification.

### Text Detection

I am using a frozen <a
href="https://github.com/eragonruan/text-detection-ctpn">connectionist text
proposal network (CTPN)</a> to detect the locations of text lines within a
given image.

Doing so with nutrition labels introduced a new issue: individual lines within large
blocks of text such as ingredients lists would sometimes remain undetected. After
initial text line detection, I added the additional step of merging nearby text
boxes so the resulting bounding boxes would encompass text paragraphs. This
prevents these initially skipped lines from being ignored.

### Text Extraction (OCR)

I am using OpenCV for image preprocessing Tesseract OCR for text
recognition. The varied stylization of ingredients lists (dark text on light background,
light text on light background, 'gradient' lighting on cylindrical containers, etc.)
is handled by passing the image through three different preprocessing variations.
Each variation is then cropped using the bounding box specifications of the
text detection step and passed through the OCR.

The result is three large strings containing the text information of each
preprocessed image, one of which is the best. This depends on which
preprocessing filter was most suited to the given image.

### Text Determination

Given each of text detected in the image, it is neccesary to determine which
lines correspond to the ingredients list and which lines do not. To do so, I
fine-tuned a <a
href="https://huggingface.co/docs/transformers/en/model_doc/mobilebert">MobileBERT</a>
model on a custom dataset (`./data/training_data`) to predict whether a given
line of text belongs to an ingredients list. MobileBert is a compressed and
accelerated version of the BERT language model.

The results from each image variant are then compared, and the one with the
most ingredients identified is chosen as the most accurate.

### Final Result

## Requirements

1. Tensorflow
2. Torch
3. Transformers
4. OpenCV
5. PyTesseract
6. Pillow
7. Numpy
8. Regex
