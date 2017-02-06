import numpy as np
from sklearn.model_selection import KFold
from sklearn import tree
import matplotlib.pyplot as plt

import voterdata_io as vdio

TESTING = False
PRINT_LEVEL = 1

if (PRINT_LEVEL >= 1):
    print "loading..."

x, y = vdio.get_2008_train()
x, enc = vdio.process_data(x)
if (PRINT_LEVEL >= 2):
    print x[:10]
    print y[:10]

x_2008 = vdio.get_2008_test()
x_2008, _ = vdio.process_data(x_2008, enc)
if (PRINT_LEVEL >= 2):
    print x_2008[:10]

x_2012 = vdio.get_2012_test()
x_2012, _ = vdio.process_data(x_2012, enc)
if (PRINT_LEVEL >= 2):
    print x_2012[:10]

if (PRINT_LEVEL >= 1):
    print "done loading"

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

if (TESTING):
    NUM_FOLDS = 2
    kf = KFold(n_splits=NUM_FOLDS)

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
pred_08 = t_real.predict(x_2008)
print "Predictions 2008:", pred_08[:10]
pred_12 = t_real.predict(x_2012)
print "Predictions 2012:", pred_12[:10]

vdio.write_predictions('2008_test2.csv', pred_08)
vdio.write_predictions('../competition2/2012_test2.csv', pred_12)


try:
    plt.show()
    pass
except:
    pass
