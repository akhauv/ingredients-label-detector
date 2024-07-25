from flask import Flask, request, jsonify
import uuid
import os
import sys

sys.path.append(os.getcwd())
from scripts.text_detection import load_detection_model
from scripts.text_determination import load_determination_model
from scripts.text_correction import initialize_symspell
from scripts.run_ingredients_detection import get_ingredients_list

app = Flask(__name__)

def load_all():
    global has_loaded
    load_detection_model()
    load_determination_model()
    initialize_symspell()
    has_loaded = True

load_all()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'}), 200

@app.route('/extract', methods=['POST'])
def extract():
    try:
        if not has_loaded:
            return jsonify({'error': 'server error'}), 400
        
        # Check if a file is included in the request (required image)
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']

        # Check if a file was actually uploaded
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # save file
        file_path = os.path.join('/tmp', str(uuid.uuid4()) + '-' + file.filename)
        file.save(file_path)

        # run ingredients extraction 
        ingredients = get_ingredients_list(file_path)

        # remove file after analysis
        os.remove(file_path)

        return jsonify({'ingredients': ingredients})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)