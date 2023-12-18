import pandas as pd
from sklearn.preprocessing import LabelEncoder

class DatasetHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = pd.read_csv(file_path)  # Load dataset initially
    
    def get_row_by_id(self, row_id):
        return self.data.iloc[row_id]
    
    def add_record(self, new_record):
        # Assuming new_record is a dictionary with 23 values for each column
        self.data = self.data.append(new_record, ignore_index=True)
    
    def save_changes(self):
        self.data.to_csv(self.file_path, index=False)

    def get_normalized(self):
        df_norm = self.data

        lbl_encod = LabelEncoder()

        for column in df_norm.columns:
            if column == 'Number' or column == 'Name':
                continue
            df_norm[column] = lbl_encod.fit_transform(df_norm[column])
        
        def scale_using_rule_of_three(column):
            max_val = column.max()
            conversion_factor = 10 / max_val
            scaled_values = column * conversion_factor
            return scaled_values

        # Apply rule of three scaling to each column from the third column to the last one
        scaled_data = df_norm.iloc[:, 2:].apply(scale_using_rule_of_three, axis=0)

        return scaled_data
