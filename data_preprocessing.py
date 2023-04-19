import numpy as np
import pickle

# Specify the file path of the pickle file
file_path = "Dataset/ntu60_3danno.pkl"

# Load the data from the pickle file
with open(file_path, "rb") as f:
    data = pickle.load(f)

# Print the loaded data
print(data)
