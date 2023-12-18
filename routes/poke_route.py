from flask import Flask, request, jsonify
import pandas as pd
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)

from services.inference.find_poke import Inference

app = Flask(__name__)

# Assume the 'Inference' class and its methods are defined here

# Create an instance of the Inference class
inference = Inference('data/datasets/pokemon_alopez247.csv')

@app.route('/find_similar', methods=['POST'])
def find_similar():
    if request.method == 'POST':
        data = request.get_json()  # Get the JSON data from the request
        
        if data:
            # Normalize the input dictionary
            normalized_dict = inference.get_normalized(data)

            # Find k nearest neighbors for the given dictionary
            nearest_neighbors = inference.find_k_nearest_neighbors(normalized_dict, 1)

            # Remove the found nearest neighbor from the dataset temporarily
            inference.dataset = inference.dataset.drop(nearest_neighbors.index)

            return jsonify(nearest_neighbors.to_dict())  # Respond with name and attributes as JSON

    return jsonify({'error': 'Invalid data'}), 400  # Respond with an error for invalid data

if __name__ == '__main__':
    app.run(debug=True)
