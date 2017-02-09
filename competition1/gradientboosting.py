import numpy as np
from sklearn.model_selection import KFold
from sklearn import ensemble
#import matplotlib.pyplot as plt

import voterdata_io as vdio
import plot

TESTING = False
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
    # try one hot encoding:
    #x, encoder = vdio.process_data(x)

    # estimating test error with cross validation over different max depths
    NUM_FOLDS = 2
    kf = KFold(n_splits=NUM_FOLDS)

    #for k in range(3, 5):
    k = 2 # best

    depths = range(20, 35)
    train_acc = []
    test_acc = []

    #for d in depths:
    for d in depths:
        avg_test_acc = 0.0
        avg_train_acc = 0.0

        for train_index, test_index in kf.split(x):

            t = ensemble.GradientBoostingClassifier(n_estimators=d * 20, max_depth=k, warm_start=True)
            t.fit(x[train_index], y[train_index])
            avg_train_acc += 1.0 / NUM_FOLDS * t.score(x[train_index], y[train_index])
            avg_test_acc += 1.0 / NUM_FOLDS * t.score(x[test_index], y[test_index])

        train_acc.append(avg_train_acc)
        test_acc.append(avg_test_acc)
        print "done with depth", d
    
    plot.plot_accs(train_acc, test_acc, depths, 'num classifiers for depth ' + str(k))


else: # not testing
    # Actually use the training data to predict the test data
    t_real = ensemble.GradientBoostingClassifier(n_estimators=800, max_depth=2)
    t_real.fit(x, y)
    print "Test score:", t_real.score(x, y)
    pred_08 = t_real.predict(x_2008)
    print "Predictions 2008:", pred_08[:10]
    pred_12 = t_real.predict(x_2012)
    print "Predictions 2012:", pred_12[:10]

    vdio.write_predictions('2008_test_gradboost2.csv', pred_08)
    vdio.write_predictions('../competition2/2012_test_gradboost2.csv', pred_12)

print "dooooone!"
