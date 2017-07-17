#!/usr/bin/python3

from keras.models import Sequential 
from keras.layers import Dense
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU, ThresholdedReLU
from keras.layers.recurrent import GRU, LSTM
import numpy as np
from numpy.linalg import norm
from PIL import Image

seed = 2
np.random.seed(seed)

    
path = 'images/image.jpg'
img = Image.open(path)
rows = np.shape(img)[0]
cols = np.shape(img)[1]

#get training data
def get_geometry(rows,cols):
    minside = min(rows,cols)
    npixels = rows*cols	
    X = np.empty([npixels,3],dtype=np.float32);
    for x in range(0,cols):
            for y in range(0,rows):
                    point = np.array([(y-0.5*rows)/minside, (x-0.5*cols)/minside])
                    X[y*cols+x,0] = point[0]
                    X[y*cols+x,1] = point[1]
                    X[y*cols+x,2] = norm(point)
    X = np.reshape(X, X.shape+(1,))
    return X
X = get_geometry(rows,cols)
Y = np.empty([rows*cols,3],dtype=np.float32)
Y = np.reshape(np.asarray(img)/255,[rows*cols,3])

#define model
def get_model():
    width = 50
    depth = 6
    model = Sequential()
    model.add(Dense(width,input_shape = (3,1)))
    model.add(PReLU())
    for d in range(depth):
            model.add(Dense(width))
            model.add(PReLU())
    model.add(GRU(16))
    for d in range(2):
            model.add(Dense(width))
            model.add(PReLU())
    model.add(Dense(3))
    model.compile(loss='mae', optimizer='rmsprop')
    return model

#train model
model = get_model()
for epoch in range(10):
    model.fit(X, Y, epochs = 1, batch_size=64)


#sample sequence
X[:,2] = X[:,2]+1.25
for i in range(100):
    predictions = model.predict(X,batch_size = 100000)
    predictions = predictions.reshape([rows,cols,3])*255
    np.clip(predictions, 0, 255, out=predictions)
    j = Image.fromarray(predictions.astype('uint8')) 
    j.save('prediction_'+str(i).zfill(2)+'.jpg')
    X[:,2] = X[:,2]-0.025
