import numpy as np
from sklearn.model_selection import KFold
from sklearn import tree
import matplotlib.pyplot as plt

import voterdata_io as vdio
import plot

TESTING = True
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

# Play around with tree depth and minimum leaf samples

if TESTING:
    NUM_FOLDS = 5
    kf = KFold(n_splits=NUM_FOLDS)

    train_acc = []
    test_acc = []

    for i in range(2, 21):
        avg_test_acc = 0.0
        avg_train_acc = 0.0
        # do the train thing
        # Decision tree time! Max tree depth
        for train_index, test_index in kf.split(x):

            t = tree.DecisionTreeClassifier(criterion='gini',
                                                  splitter='random',
                                                  random_state=None,
                                                  max_depth=i)
            t.fit(x[train_index], y[train_index])

            avg_train_acc += 1.0 / NUM_FOLDS * t.score(x[train_index], y[train_index])
            avg_test_acc += 1.0 / NUM_FOLDS * t.score(x[test_index], y[test_index])

        train_acc.append(avg_train_acc)
        test_acc.append(avg_test_acc)
    
    plot.plot_accs(train_acc, test_acc, range(2, 21), 'maximum tree depth')


    train_acc = []
    test_acc = []
    
    for i in range(1, 50):
        # Decision tree time! Part A
        avg_test_acc = 0.0
        avg_train_acc = 0.0
        for train_index, test_index in kf.split(x):

            t = tree.DecisionTreeClassifier(criterion='gini',
                                              splitter='random',
                                              random_state=None,
                                              min_samples_leaf=i)
            t.fit(x[train_index], y[train_index])

            avg_train_acc += 1.0 / NUM_FOLDS * t.score(x[train_index], y[train_index])
            avg_test_acc += 1.0 / NUM_FOLDS * t.score(x[test_index], y[test_index])

        train_acc.append(avg_train_acc)
        test_acc.append(avg_test_acc)
    
    plot.plot_accs(train_acc, test_acc, range(1, 50), 'minimum leaf samples')

else:
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
        #plt.show()
        pass
    except:
        pass
