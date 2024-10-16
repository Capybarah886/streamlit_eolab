import pandas as pd
import numpy as np
# build the funciont to accept the data and return the prediction
def merge_files(left, right):
    left_copy = left.copy()
    right_copy = right.copy()
    left_copy.columns = [str(col) + '_1' for col in left_copy.columns]
    right_copy.columns = [str(col) + '_2' for col in right_copy.columns]
    merged_df = pd.concat([left_copy, right_copy], axis=1)
    merged_df.drop(columns=["timestamp_1", "timestamp_2", 'quaternion.accuracy_1', 'quaternion.accuracy_2', 'quaternion.real_1', 'quaternion.real_2'], inplace=True)
    merged_df = merged_df.dropna()

    return merged_df


def split_data(df):
    count=0
    data_lst = []
    features = ['fPressureFront_1', 'fPressureSide_1',
       'acceleration.x_1', 'acceleration.y_1', 'acceleration.z_1',
       'quaternion.i_1', 'quaternion.j_1', 'quaternion.k_1',
        'fPressureFront_2',
       'fPressureSide_2', 'acceleration.x_2', 'acceleration.y_2',
       'acceleration.z_2', 'quaternion.i_2', 'quaternion.j_2',
       'quaternion.k_2' ]
    data = df[features].values
    for i in range(200, min(7000,len(data)-200), 100):
        data_lst.append(data[i:i+200])
        count += 1
    return count, data_lst

def predict_stroke(data_lst, model):
    data_lst = np.transpose(data_lst, (0, 2, 1))
    y_pred = model.predict(data_lst)
    # y_proba = model.predict_proba(data_lst)

    return y_pred

def combined_flow(left,right,model):
    df = merge_files(left,right)
    count, data_lst = split_data(df)
    predicts = predict_stroke(data_lst,model)
    return predicts


