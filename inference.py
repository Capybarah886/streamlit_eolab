import torch.nn as nn
import torch
import pandas as pd
from transformer import TransformerModel
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter



class InferenceModel:
    def __init__(self, model, left_csv, right_csv):
        self.left_csv = left_csv
        self.right_csv = right_csv
        self.label_dict = {'Freestyle': 0, 'BackStroke': 1, 'Butterfly': 2, 'BreastStroke': 3}
        self.model = model
    
    # def load_model(self):
    #     model = torch.load(self.model_path)
    #     model.eval()
    #     return model

    def merge_file(self, left, right):
        # add suffix to columns to differentiate between the two dataframes
        left.columns = [str(col) + '_1' for col in left.columns]
        right.columns = [str(col) + '_2' for col in right.columns]
        merged_dataframe = pd.concat([left, right], axis=1)
        merged_dataframe.drop(columns=["timestamp_1", "timestamp_2", 'quaternion.accuracy_1', 'quaternion.accuracy_2', 'quaternion.real_1', 'quaternion.real_2'], inplace=True)
        return merged_dataframe

    def split_data(self, data, batch_size=32):
        data_lst = [] 
        data = data.apply(pd.to_numeric, errors='coerce')

        # Slice the data and convert it to tensors
        for i in range(0, len(data) - 200, 100):
            data_tensor = torch.tensor(data.values[i:i + 200]).float()
            data_lst.append(data_tensor)

        # Stack the data into a single tensor and create a TensorDataset
        data_tensor = torch.stack(data_lst)  # Shape: (num_samples, sequence_length, input_size)

        # If your model requires a batch size, make sure the dataset is batched accordingly
        dataset = TensorDataset(data_tensor)

        # Create a DataLoader with the specified batch size
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        return data_loader

    def inference(self):
        batch_count = 1
        predictions_dic = {}
        predictions_dic['batch_result'] = []        

        # Load and merge files
        file_left = (self.left_csv)
        file_right = (self.right_csv)
        merged_file = self.merge_file(file_left, file_right)
        print("There are", len(merged_file), "data points in the merged file")

        # Prepare data for prediction
        data_loader = self.split_data(merged_file)
        print("The data is ready for prediction")
        print("There are", len(data_loader), "batches")

        predictions_dic['batch_count'] = len(data_loader)
        predictions_dic['total_data_points'] = len(merged_file)

        # Iterate over data loader to make predictions
        for data_batch in data_loader:
            # Extract the batch (since DataLoader returns tuples)
            data_batch = data_batch[0]

            # Make predictions
            prediction = self.model(data_batch)
            predicted_indices = prediction.argmax(dim=1)  # Get the predicted indices
            predicted_labels = [list(self.label_dict.keys())[list(self.label_dict.values()).index(idx.item())] for idx in predicted_indices]

            # Print the batch id and the most possible label
            # find the most frequent label in the batch
            most_common_label = Counter(predicted_labels).most_common(1)[0][0]
            print("In the batch", batch_count, "The most possible result is", most_common_label)
            batch_count += 1
            predictions_dic['batch_result'].append(most_common_label)

        return predictions_dic

if __name__ == '__main__':
    model_path = './model.pth'
    model=torch.load(model_path)
    left_csv = pd.read_csv('left2.csv')
    right_csv = pd.read_csv('right2.csv')
    inference_model = InferenceModel(model, left_csv, right_csv)
    predictions = inference_model.inference()
    print(predictions)