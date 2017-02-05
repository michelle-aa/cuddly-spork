import numpy as np

train_data_filename = 'train_2008.csv'
test_data_filename = 'test_2008.csv'

# field_indices = [5, 6, 7, 11, 17, 37, 41, 43, 45, 48, 49, 62]
field_indices = [5, 11, 17, 37, 41, 45, 48, 170, 375]

# PES1 is the last column of the training data.
# It is the column we are trying to predict.
label_index = -1

# Screens for invalid/blank answers to questions
def process_value(val):
    return int(val) if val >= 0 else np.nan

# Reads all the data in a CSV file
def read_raw_data(filename):
    with open(filename) as f:
        # Skip the header line
        next(f)
        # Convert the rest of the rows into numerical values
        return np.array([map(process_value, line.split(',')) for line in f])

# Read the training data.
# Returns a tuple (input_data, labels).
# The shape of the input_data is (num_samples, num_fields)
# The shape of labels is (num_samples, 1)
def read_train_data():
    raw_data = read_raw_data(train_data_filename)
    return np.take(raw_data, field_indices, axis=1), raw_data[:, label_index]

# Reads the test data file.
# The test data file does not have labels, we need to predict them.
def read_test_data():
    raw_data = read_raw_data(test_data_filename)
    return np.take(raw_data, field_indices, axis=1)
