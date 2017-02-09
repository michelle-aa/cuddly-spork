import voterdata_io as vdio

import numpy as np

x, y = vdio.get_2008_train()
x_2008 = vdio.get_2008_test()
x_2012 = vdio.get_2012_test()

print x_2008.shape[0]
print x_2012.shape[0]

pred_08 = np.ones(x_2008.shape[0])
print "Predictions 2008:", pred_08[:10]
pred_12 = np.ones(x_2012.shape[0])
print "Predictions 2012:", pred_12[:10]

vdio.write_predictions('2008_test3.csv', pred_08)
vdio.write_predictions('../competition2/2012_test3.csv', pred_12)