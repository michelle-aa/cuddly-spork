import numpy as np
import tensorflow as tf 
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from sklearn.model_selection import KFold

import voterdata_io as vdio
import plot

x, y = vdio.get_2008_train()
x, enc = vdio.process_data(x)

# {1, 2} -> {0, 1}
y -= 1

print np.shape(x[0])

def create_model():
    model = Sequential()
    model.add(Dense(105, input_shape=np.shape(x[0])))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(105))
    model.add(Activation('relu'))
    model.add(Dropout(0.15))
    model.add(Dense(105))
    model.add(Activation('relu'))
    model.add(Dropout(0.15))
    model.add(Dense(105))
    model.add(Activation('relu'))

    #model.add(Dense(20))
    #model.add(Activation('relu'))
    # model.add(Dropout(0.2))
    #model.add(Dense(5))
    #model.add(Activation('relu'))

    # Output layer
    model.add(Dense(1, init='normal', activation='sigmoid'))

    model.summary()

    model.compile(loss='binary_crossentropy', optimizer='adam',
        metrics=['accuracy'])
    return model

# estimating test error with cross validation over different max depths
NUM_FOLDS = 4
kf = KFold(n_splits=NUM_FOLDS)

avg_train_acc = 0
avg_test_acc = 0
for train_index, test_index in kf.split(x):
    model = create_model()
    model.fit(x[train_index], y[train_index], batch_size=32, nb_epoch=50, verbose=1)
    avg_train_acc += 1.0 / NUM_FOLDS * model.evaluate(x[train_index], y[train_index])[1]
    avg_test_acc += 1.0 / NUM_FOLDS * model.evaluate(x[test_index], y[test_index])[1]

print 'Train accuracy:', avg_train_acc
print 'Test accuracy:', avg_test_acc





