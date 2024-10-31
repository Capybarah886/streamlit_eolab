import torch.nn as nn
import torch
import pandas as pd
from torch_models import TransformerModel, LSTMModel
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter

class InferenceModel:
    def __init__(self, model, left_csv, right_csv):
        self.left_csv = left_csv
        self.right_csv = right_csv
        self.label_dict = {'Freestyle': 0, 'BackStroke': 1, 'Butterfly': 2, 'BreastStroke': 3}
        self.model = model
    
    def merge_file(self, left, right):
        left.columns = [str(col) + '_1' for col in left.columns]
        right.columns = [str(col) + '_2' for col in right.columns]
        merged_dataframe = pd.concat([left, right], axis=1)
        merged_dataframe.drop(columns=["timestamp_1", "timestamp_2",'quaternion.real_1', 'quaternion.real_2'], inplace=True)
        return merged_dataframe
# 'quaternion.accuracy_1', 'quaternion.accuracy_2' ,
    def split_data(self, data, batch_size=32):
        data_lst = [] 
        data = data.apply(pd.to_numeric, errors='coerce')

        for i in range(0, len(data) - 200, 100):
            data_tensor = torch.tensor(data.values[i:i + 200]).float()
            data_lst.append(data_tensor)

        data_tensor = torch.stack(data_lst)
        dataset = TensorDataset(data_tensor)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        return data_loader

    def inference(self):
        batch_count = 1
        predictions_dic = {}
        predictions_dic['batch_result'] = []        

        file_left = (self.left_csv)
        file_right = (self.right_csv)
        merged_file = self.merge_file(file_left, file_right)
        print("There are", len(merged_file), "data points in the merged file")

        data_loader = self.split_data(merged_file)
        print("The data is ready for prediction")
        print("There are", len(data_loader), "batches")

        predictions_dic['batch_count'] = len(data_loader)
        predictions_dic['total_data_points'] = len(merged_file)

        for data_batch in data_loader:
            data_batch = data_batch[0]
            prediction = self.model(data_batch)
            predicted_indices = prediction.argmax(dim=1)
            predicted_labels = [list(self.label_dict.keys())[list(self.label_dict.values()).index(idx.item())] for idx in predicted_indices]

            most_common_label = Counter(predicted_labels).most_common(1)[0][0]
            print("In the batch", batch_count, "The most possible result is", most_common_label)
            batch_count += 1
            predictions_dic['batch_result'].append(most_common_label)

        return predictions_dic

if __name__ == '__main__':
    model_path = './models/lstm_model.pth'
    model = LSTMModel(18,256,3)  # Initialize the model instance
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # Load the state dictionary
    model.eval()  # Set the model to evaluation mode

    left_csv = pd.read_csv('left.csv')
    right_csv = pd.read_csv('right.csv')
    inference_model = InferenceModel(model, left_csv, right_csv)
    predictions = inference_model.inference()
    print(predictions)