import numpy as np
import pickle
from collections import OrderedDict
import pandas as pd

def extract_prefix(imageid):
    return imageid.split('_')[0]

# Load the CSV file to inspect its contents and structure
data = pd.read_csv('/localmount/volume-hd/users/uline/data_sets/bio-bank-sample/wmh_overall.csv')

# Filter data for 'train' and 'validation' splits
train_cases_np = data[data['wmh_split'] == 'train']['imageid'].tolist()
val_cases_np = data[data['wmh_split'] == 'val']['imageid'].tolist()

# Function to extract the part of the string before the first underscore

# Apply the function to the lists
train_cases_np = [extract_prefix(id) for id in train_cases_np]
val_cases_np = [extract_prefix(id) for id in val_cases_np]

# Now train_cases_np and val_cases_np contain the modified strings
print("Train Cases:", train_cases_np)
print("Validation Cases:", val_cases_np)

# Output the first few elements of each list to verify

# Step 3: Prepare the split ordered dictionary
splits = [
    OrderedDict(
    [
        ("train", train_cases_np),
        ("val", val_cases_np)
    ]),
    OrderedDict(
        [
            ("train", train_cases_np),
            ("val", val_cases_np)
        ]),
    OrderedDict(
        [
            ("train", train_cases_np),
            ("val", val_cases_np)
        ]),
    OrderedDict(
        [
            ("train", train_cases_np),
            ("val", val_cases_np)
        ]),
    OrderedDict(
        [
            ("train", train_cases_np),
            ("val", val_cases_np)
        ])
]

# Step 4: Serialize and save the splits to a .pkl file
splits_file = "splits_final.pkl"
with open(splits_file, 'wb') as outfile:
    pickle.dump(splits, outfile)


