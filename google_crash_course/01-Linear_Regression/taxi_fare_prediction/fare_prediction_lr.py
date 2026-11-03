
# data
import numpy as np
import pandas as pd

# machine learning
import keras
import ml_edu.experiment
import ml_edu.results

# data visualization
import plotly.express as px


chicago_taxi_dataset = pd.read_csv("./chicago_taxi_train.csv")
#print(chicago_taxi_dataset.head())

training_df = chicago_taxi_dataset.loc[:, ('TRIP_MILES', 'TRIP_SECONDS', 'FARE', 'COMPANY', 'PAYMENT_TYPE', 'TIP_RATE')]

training_df['TRIP_MINUTES'] = training_df['TRIP_SECONDS'] / 60
print('Read dataset completed successfully.')
print('Total number of rows: {0}\n\n'.format(len(training_df.index)))
#print(training_df.head(200))

print(training_df.describe(include='all'))

#max_fare = training_df['FARE'].max()
#mean_distance = training_df['TRIP_MILES'].mean()
#unique_cab_companies = training_df['COMPANY'].nunique()
#most_frequent_payment_type = training_df['PAYMENT_TYPE'].value_counts().idxmax()
#missing_values = training_df.isnull().sum().sum()

#print(max_fare)
#print(mean_distance)
#print(unique_cab_companies)
#print(most_frequent_payment_type)
#print("Are any features missing data? Answer:", "No" if missing_values == 0 else "Yes")

#print(training_df.corr(numeric_only = True))

#px.scatter_matrix(training_df, dimensions=['FARE', 'TRIP_MILES', 'TRIP_SECONDS']).show()


