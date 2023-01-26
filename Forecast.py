import darts
import pandas as pd

import ProcessData

file_path = "./Data/BATS_AAL, 60.csv"
# loading df
df = pd.read_csv(file_path)
preprocessed_df = ProcessData.preprocess_time_series(df)
preprocessed_df.drop(columns=["time"],inplace=True)

# Define the target variable
target_variable = "close"

# Fit the model
model = darts.TSF(preprocessed_df[target_variable], forecast_length=10)
model.fit()

# Forecast the next 10 steps
forecast = model.predict()

print("Forecasted values: ", forecast)