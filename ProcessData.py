import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import matplotlib.pyplot as plt

import os

def generate_time_series_dataset(data_array, time_chunk, num_samples):
    x_data = []
    y_data = []
    scaler = MinMaxScaler()
    for i in range(num_samples):
        start_index = np.random.randint(0, data_array.shape[0] - time_chunk)

        x = data_array[start_index:start_index+time_chunk]
        x = np.nan_to_num(x,nan=0, posinf=0, neginf=0)
        x = scaler.fit_transform(X=x)
        

        x_data.append(x)
        y_data.append(data_array[start_index+time_chunk])
    return (np.array(x_data), np.array(y_data))

def preprocess_time_series(df):
    # Forward fill and backfill NaNs
    df.fillna(method='ffill', axis=0, inplace=True)
    df.fillna(method='bfill', axis=0, inplace=True)

    # Calculate % change from previous row's values
    df = df.pct_change()

    # Fill remaning NA with 0
    df.fillna(0, axis=0,inplace=True)
    df.fillna(0, axis=1,inplace=True)

    # Removing outliers
    z_scores = (df - df.mean()) / df.std()
    threshold = 3
    outliers = np.abs(z_scores) > threshold
    df[outliers]=0

    return df

def main():
    input_path = "./Data/"
    output_path = "./ProcessedData/"
    
    chunks = [1000,5000,10000]

    for chunk in chunks:
        time_chunk = chunk
        sample_percent = 1.2 #how much of the dataset we want to sample
        
        save_dir = os.path.join(output_path,str(time_chunk)+"hr")

        x_path = os.path.join(save_dir,'x')
        y_path = os.path.join(save_dir,'y')

        os.makedirs(x_path,exist_ok=True)
        os.makedirs(y_path,exist_ok=True)   

        for file in os.listdir(input_path):
            file_path = input_path+file

            df = pd.read_csv(file_path)
            preprocessed_df = preprocess_time_series(df)
            preprocessed_df.drop(columns=["time"],inplace=True)

            preprocessed_array = preprocessed_df.to_numpy()

            num_samples =  (len(preprocessed_array)/time_chunk) * sample_percent
            num_samples = int(num_samples)

            x,y = generate_time_series_dataset(preprocessed_array,time_chunk,num_samples)

            file_name = os.path.splitext(file)[0]

            np.save(os.path.join(x_path,file_name),x)
            np.save(os.path.join(y_path,file_name),y)
            
            print(file_name)

def test():
    x = np.load("./ProcessedData/1000hr/x/BATS_AAL, 60.npy")
    y = np.load("./ProcessedData/1000hr/y/BATS_AAL, 60.npy")

    print(x)
    
    print(x.shape)
    print(y.shape)

if __name__=="__main__":
    # test()
    main()