import numpy as np 
import tensorflow as tf 
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from sklearn.model_selection import KFold


import voterdata_io as vdio
import plot

TESTING = True

x, y = vdio.get_2008_train()
#print x.shape()
x, encoder = vdio.process_data(x)
y = vdio.process_labels(y)
#x = np.matrix(x)
#print x.shape
#print y.shape
#print y[:10]

if TESTING:
    model = Sequential()

    model.add(Dense(105, input_shape=(105,)))
    '''model.add(Activation('relu'))
    model.add(Dropout(0.15))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(0.15))'''

    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    NUM_FOLDS = 5
    kf = KFold(n_splits=NUM_FOLDS)

    avg_test_acc = 0.0
    avg_train_acc = 0.0

    for train_index, test_index in kf.split(x):
        # fit = 
        model.fit(x[train_index], y[train_index], batch_size=100, nb_epoch=10, verbose=1)

        avg_train_acc += 1.0 / NUM_FOLDS * model.evaluate(x[train_index], y[train_index], verbose=0)
        avg_test_acc += 1.0 / NUM_FOLDS * model.evaluate(x[test_index], y[test_index], verbose=0)

    #train_acc.append(avg_train_acc)
    #test_acc.append(avg_test_acc)

    print "Training accuracy: " + str(avg_train_acc)
    print "Testing accuracy: " + str(avg_test_acc)
    
    #plot.plot_accs(train_acc, test_acc, depths, 'maximum tree depth')

else:
    pass