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
y = vdio.process_labels(y)

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


model = create_model()
model.fit(x, y, batch_size=256, nb_epoch=50, validation_split=0.2, verbose=1)
score = model.evaluate(x, y)

# Predictions!
x_2008 = vdio.get_2008_test()
x_2008, enc = vdio.process_data(x_2008)
x_2012 = vdio.get_2012_test()
x_2012, enc = vdio.process_data(x_2012)

pred_08 = model.predict(x_2008)
print "Predictions 2008:", pred_08[:10]
pred_12 = model.predict(x_2012)
print "Predictions 2012:", pred_12[:10]

vdio.write_predictions('2008_test_nn.csv', pred_08)
vdio.write_predictions('../competition2/2012_test_nn.csv', pred_12)

print ''
print 'Training Loss:', score[0]
print 'Accuracy:', score[1]
