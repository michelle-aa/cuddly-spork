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

print ''
print 'Training Loss:', score[0]
print 'Accuracy:', score[1]
