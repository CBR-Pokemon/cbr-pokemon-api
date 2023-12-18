from numpy import NaN
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)

from db_connection.poke_db import DatasetHandler


class Inference:
    def __init__(self, dataset_file):
        self.dataset = pd.read_csv(dataset_file)
        self.weights = {
            'Total': 1, 'HP': 1, 'Attack': 1, 'Defense': 1, 'Sp_Atk': 1,
            'Sp_Def': 1, 'Speed': 1, 'Generation': 5, 'isLegendary': 5,
            'hasGender': 2, 'Pr_Male': 1, 'hasMegaEvolution': 2,
            'Height_m': 1, 'Weight_kg': 1, 'Catch_Rate': 1
        }
        self.column_order = [
            'Total', 'HP', 'Attack', 'Defense', 'Sp_Atk', 'Sp_Def', 'Speed',
            'Generation', 'isLegendary', 'hasGender', 'Pr_Male', 'hasMegaEvolution',
            'Height_m', 'Weight_kg', 'Catch_Rate'
        ]
    
    def local_similarity(self, dictionary_value, row_value):
        try:
            dict_val_float = float(dictionary_value)
            row_val_float = float(row_value)
            return 1 - abs(dict_val_float - row_val_float) / 10
        except (ValueError, TypeError):
            return 0 
    
    def global_similarity(self, dict_values, row_values):
        global_sim = 0
        for i, (dict_val, row_val) in enumerate(zip(dict_values, row_values)):
            weight = self.weights.get(self.column_order[i], 1)
            local_sim = self.local_similarity(dict_val, row_val)
            global_sim += local_sim * weight

        # Normalize the global similarity by dividing by the total weight sum
        total_weight = sum(self.weights.values())
        normalized_global_sim = global_sim / total_weight if total_weight > 0 else 0

        return normalized_global_sim
    
    def find_k_nearest_neighbors(self, dictionary, k=1):
        # Convert dictionary to DataFrame row
        dict_df = pd.DataFrame(dictionary, index=[0])

        # Calculate global similarities for all rows
        global_similarities = self.dataset.apply(
            lambda row: self.global_similarity(dict_df.values[0], row[self.column_order].values), axis=1
        )

        # Find k nearest neighbors based on global similarity
        nearest_indices = global_similarities.nlargest(k).index.tolist()

        return self.dataset.iloc[nearest_indices]
    
    def get_normalized(self, dictionary):

        # Convert dictionary to DataFrame
        dict_df = pd.DataFrame(dictionary, index=[0])

        dataset = DatasetHandler('data/datasets/pokemon_alopez247.csv')
        dataset.data = dataset.data._append(dict_df)
        dataset = dataset.get_normalized()

        def scale_using_rule_of_three(column):
            max_val = column.max()
            conversion_factor = 10 / max_val
            scaled_values = column * conversion_factor
            return scaled_values

        # Apply rule of three scaling to each column from the third column to the last one
        scaled_data = dataset.iloc[:, 2:].apply(scale_using_rule_of_three, axis=0)

        return scaled_data.to_dict(orient='records')[len(scaled_data) - 1]



