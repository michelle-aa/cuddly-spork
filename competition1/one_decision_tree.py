import numpy as np
from sklearn.model_selection import KFold
from sklearn import tree
import matplotlib.pyplot as plt


train_data_filename = 'train_2008.csv'
test_data_filename = 'test_2008.csv'

# field_indices = [5, 6, 7, 11, 17, 37, 41, 43, 45, 48, 49, 62]
field_indices = [5, 11, 17, 37, 41, 45, 48, 170, 375]

# PES1 is the last column of the training data
label_index = -1

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
# The first dimension of input_data and labels match and are the number of
# of samples.
def read_train_data():
    raw_data = read_raw_data(train_data_filename)
    return np.take(raw_data, field_indices, axis=1), raw_data[:, label_index]

# Reads the training data file.
# The training data file
def read_test_data():
    raw_data = read_raw_data(test_data_filename)
    return np.take(raw_data, field_indices, axis=1)

print "loading..."

x, y = read_train_data()
x_test = read_test_data()

print x[:10]
print y[:10]
print x_test[:10]

print "done loading"

NUM_FOLDS = 2
kf = KFold(n_splits=NUM_FOLDS)
# Plots accuracy vs some number
def plot_accs(train_accs, test_accs, indices, label):
    # make teh graph
    plt.figure()
    colors = ['crimson', 'darkcyan']
    plt.plot(indices, train_accs, marker='o', color='crimson', label='train')
    plt.plot(indices, test_accs, marker='o', color='darkcyan', label='test')
    plt.xlabel(label)
    plt.ylabel('average accuracy')
    plt.legend(loc='best')

# Play around with tree depth and minimum leaf samples
for train_index, test_index in kf.split(x):
    # do the train thing
    # Decision tree time! Max tree depth
    train_acc = []
    test_acc = []
    for i in range(2, 21):
        t = tree.DecisionTreeClassifier(criterion='gini',
                                              splitter='random',
                                              random_state=None,
                                              max_depth=i)
        t.fit(x[train_index], y[train_index])
        train_acc.append(t.score(x[train_index], y[train_index]))
        test_acc.append(t.score(x[test_index], y[test_index]))
    plot_accs(train_acc, test_acc, range(2, 21), 'maximum tree depth')

    # Decision tree time! Part A
    train_acc = []
    test_acc = []
    for i in range(1, 50):
        t = tree.DecisionTreeClassifier(criterion='gini',
                                          splitter='random',
                                          random_state=None,
                                          min_samples_leaf=i)
        t.fit(x[train_index], y[train_index])
        train_acc.append(t.score(x[train_index], y[train_index]))
        test_acc.append(t.score(x[test_index], y[test_index]))
    plot_accs(train_acc, test_acc, range(1, 50), 'minimum leaf samples')


# Actually use the training data to predict the test data
t_real = tree.DecisionTreeClassifier(criterion='gini',
                                              splitter='random',
                                              random_state=None,
                                              max_depth=8)
t_real.fit(x, y)
print "Test score:", t_real.score(x, y)
pred = t_real.predict(x_test)
print "Predictions:", pred[:10]

def write_predictions(filename, predictions):
    # Assume that the index of a prediction is the id.
    labeled_data = np.column_stack((range(len(predictions)), predictions))
    np.savetxt(filename, labeled_data,
        delimiter=',', header='id,PES1', fmt='%d,%d', comments='')

write_predictions('cuddlyspork_test1.csv', pred)

try:
    #plt.show()
    pass
except:
    pass
