from flask import Flask, request, jsonify
import os
import sys

sys.path.append(os.getcwd())
from scripts.text_detection import load_detection_model
from scripts.text_determination import load_determination_model
from scripts.text_correction import initialize_symspell
from scripts.run_ingredients_detection import get_ingredients_list

app = Flask(__name__)

def load_all():
    load_detection_model()
    load_determination_model()
    initialize_symspell()

# load model when server starts
load_all()

@app.route('/extract', methods=['POST'])
def extract():
    try:
        # handle image upload here
        img_path = 'placeholder'

        ingredients = get_ingredients_list(img_path)

        return jsonify({'ingredients': ingredients})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)