import numpy as np
from sklearn.model_selection import KFold
from sklearn import ensemble
import matplotlib.pyplot as plt

import voterdata_io as vdio

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


try:
    #plt.show()
    pass
except:
    pass
