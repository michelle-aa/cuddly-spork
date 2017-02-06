import numpy as np
from sklearn.model_selection import KFold
from sklearn import ensemble
#import matplotlib.pyplot as plt

import voterdata_io as vdio
import plot

TESTING = True
PRINT_LEVEL = 2

if (PRINT_LEVEL >= 1):
    print "loading..."

x, y = vdio.get_2008_train()
if (PRINT_LEVEL >= 2):
    print x[:10]
    print y[:10]

x_2008 = vdio.get_2008_test()
if (PRINT_LEVEL >= 2):
    print x_2008[:10]

x_2012 = vdio.get_2012_test()
if (PRINT_LEVEL >= 2):
    print x_2012[:10]

if (PRINT_LEVEL >= 1):
    print "done loading"

if TESTING:
    # estimating test error with cross validation over different max depths
    NUM_FOLDS = 5
    kf = KFold(n_splits=NUM_FOLDS)
    
    depths = range(2,21)
    train_acc = []
    test_acc = []

    for d in depths:
        avg_test_acc = 0.0
        avg_train_acc = 0.0

        for train_index, test_index in kf.split(x):

            t = ensemble.RandomForestClassifier(n_estimators=101,
                                                        criterion='gini',
                                                          random_state=None,
                                                          max_depth=d,
                                                          n_jobs=-1)
            t.fit(x[train_index], y[train_index])
            avg_train_acc += 1.0 / NUM_FOLDS * t.score(x[train_index], y[train_index])
            avg_test_acc += 1.0 / NUM_FOLDS * t.score(x[test_index], y[test_index])

        train_acc.append(avg_train_acc)
        test_acc.append(avg_test_acc)
    
    plot.plot_accs(train_acc, test_acc, depths, 'maximum tree depth')

    '''
    leaf_fractions = [l / 100000.0 for l in range(1, 31)]#(31)]
    train_acc = []
    test_acc = []

    for f in leaf_fractions:
        avg_test_acc = 0.0
        avg_train_acc = 0.0

        for train_index, test_index in kf.split(x):

            t = ensemble.RandomForestClassifier(n_estimators=101,
                                                        criterion='gini',
                                                          random_state=None,
                                                          #max_depth=8,
                                                          min_samples_leaf=f,
                                                          n_jobs=-1)
            t.fit(x[train_index], y[train_index])
            avg_train_acc += 1.0 / NUM_FOLDS * t.score(x[train_index], y[train_index])
            avg_test_acc += 1.0 / NUM_FOLDS * t.score(x[test_index], y[test_index])

        train_acc.append(avg_train_acc)
        test_acc.append(avg_test_acc)

        print('leaf size ' + str(f) + ' done')
    
    plot.plot_accs(train_acc, test_acc, leaf_fractions, 'min leaf fraction')'''

else: # not testing
    # Actually use the training data to predict the test data
    t_real = ensemble.RandomForestClassifier(n_estimators=101,
                                                criterion='gini',
                                                  random_state=None,
                                                  max_depth=8,
                                                  n_jobs=-1)
    t_real.fit(x, y)
    print "Test score:", t_real.score(x, y)
    pred_08 = t_real.predict(x_2008)
    print "Predictions 2008:", pred_08[:10]
    pred_12 = t_real.predict(x_2012)
    print "Predictions 2012:", pred_12[:10]

    vdio.write_predictions('2008_test4.csv', pred_08)
    vdio.write_predictions('../competition2/2012_test4.csv', pred_12)

