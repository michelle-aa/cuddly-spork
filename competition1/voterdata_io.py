import numpy as np
# from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder

train_2008_filename = 'train_2008.csv'
test_2008_filename = 'test_2008.csv'
test_2012_filename = '../competition2/test_2012.csv'

# field_indices = [5, 6, 7, 11, 17, 37, 41, 43, 45, 48, 49, 62]
# field_indices = [5, 11, 17, 37, 41, 45, 48, 170, 375]

# field_indices, is_categorical = zip(*
#     [   (5, False),
#         (6, True),
#         (7, True),
#         (8, False),
#         (11, False),
#         (17, False),
#         (18, True),
#         (24, False),
#         (37, False),
#         (41, False),
#         (43, True),
#         (45, False),
#         (47, False),
#         (48, False),
#         (49, True),
#         (62, True),
#         (67, True),
#         (186, True),
#         (247, False),
#         (375, True)
#     ])

# List of fields to use as features.
# Each element in this list is a tuple of
# (field index, is_categorical, num_categories)

# Not sure how to handle binary data with missing values. Currently treating
# them as categorical with 3 categories.
field_indices, is_categorical, num_categories = zip(*
    [   (5, False, 2),
        (6, True, 4),
        (7, True, 13),
        (8, True, 3),
        (11, False, 17),
        (17, False, 17),
        (18, True, 11),
        (24, True, 3),
        (30, True, 51), # state
        #(34, True, 4),
        (37, False, 8),
        (41, False, 86),
        (43, True, 7),
        (45, True, 3),
        #(46, True, 3),
        (47, True, 3),
        (48, False, 47),
        (49, True, 27),
        (62, True, 6),
        (67, True, 6),
        (186, True, 9),
        (247, False, 12),
        (375, True, 3)
    ])
field_indices = np.array(field_indices)
is_categorical = np.array(is_categorical)
num_categories = np.array(num_categories)[is_categorical]

# PES1 is the last column of the training data.
# It is the column we are trying to predict.
label_index = -1

# Screens for invalid/blank answers to questions
def process_value(val):
    return int(val) if int(val) > 0 else 0

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
def read_train_data(filename):
    raw_data = read_raw_data(filename)
    X = np.take(raw_data, field_indices, axis=1)
    return np.take(raw_data, field_indices, axis=1), raw_data[:, label_index]

# Reads the test data file.
# The test data file does not have labels, we need to predict them.
def read_test_data(filename):
    raw_data = read_raw_data(filename)
    return np.take(raw_data, field_indices, axis=1)

# Processes feature data using one-hot encoding.
# Returns (processed_data, one_hot_encoder)
def process_data(X, one_hot_encoder=None):
    if one_hot_encoder == None:
        one_hot_encoder = OneHotEncoder(n_values=num_categories,
                                        categorical_features=is_categorical,
                                        sparse=False).fit(X)
    return one_hot_encoder.transform(X), one_hot_encoder

# Processes labels to convert from {1, 2} -> {0, 1} so a neural network
# can classify it correctly.
def process_labels(y):
    return y - 1


# Helper functions that get the data from each year
def get_2008_train():
    return read_train_data(train_2008_filename)
def get_2008_test():
    return read_test_data(test_2008_filename)
def get_2012_test():
    return read_test_data(test_2012_filename)

# Outputs predictions to a file with given filename, following the format
# specified in sample_submission.csv:
# Predictions should be a 1D array with only the PES1 column.
# id, PES1
def write_predictions(filename, predictions):
    # Assume that the index of a prediction is the id.
    labeled_data = np.column_stack((range(len(predictions)), predictions))
    np.savetxt(filename, labeled_data,
        delimiter=',', header='id,PES1', fmt='%d,%d', comments='')
