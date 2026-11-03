import numpy as np 
import pandas as pd

my_columns = ["Eleanor", "Chidi", "Tahani", "Jason"]
my_data = np.random.randint(low=0, high=101, size=(3,4))

my_dataframe = pd.DataFrame(data=my_data, columns=my_columns)

print(my_dataframe["Eleanor"].iloc[[1]])

my_dataframe["Janet"] = my_dataframe["Tahani"] + my_dataframe["Jason"]
print(my_dataframe)

cp_my_dataframe = my_dataframe.copy()

cp_my_dataframe.at[1, 'Eleanor'] = 200

print("cp: %d\n df: %d\n" % (cp_my_dataframe['Eleanor'][1],  my_dataframe['Eleanor'][1]))

px.scatter_matrix(training_df, dimensions=['FARE', 'TRIP_MILES', 'TRIP_SECONDS'])
