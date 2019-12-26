"""
Created on 16 Jun 2016
@author: @beinghorizontal
"""

import numpy as np
np.random.seed(7)  # for reproducibility
import pandas as pd
#from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers import LSTM,Dropout
from keras import optimizers
from keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint

file='wave_memory.csv'
df=pd.read_csv(file)
df=df.dropna()
numfeatures=(len(df.columns))-1    
features=list(df.columns[:numfeatures])
print(' number of features ', numfeatures)
print(' features: ', features)

dataset=df.values
y=dataset[:,numfeatures]
x=dataset[:,0:numfeatures]
train_size = int(len(x) * 0.50)
print( ' number of train : test samples ', train_size)

"Train_size-2 and train_size+1 for train and test is used so that number of samples can be equally devisiable by batch size" 
X_train, X_test = x[0:train_size-2,:], x[train_size+1:len(dataset)-1,:]
y_train, y_test = y[0:train_size-2], y[train_size+1:len(dataset)-1]

print('before rehsape x train, y train x test y test'+'\n',
      X_train.shape,y_train.shape,X_test.shape,y_test.shape)


"LSTM requires 3D array [number of samples, time steps, number of features]"
def reshapexy(xarray,yarray,seq): #xarray is x_train or x_test and yarray is y_train or y_test
    nb_samples = xarray.shape[0] - seq

    xarray_reshaped = np.zeros((nb_samples, seq,xarray.shape[1] ))
    yarray_reshaped = np.zeros((nb_samples))
    print('sample size', nb_samples)
    for i in range(nb_samples):
        "for continuous sliding window"
        #y_position = x_position + seq
        y_position = i + seq
        xarray_reshaped[i] = xarray[i:y_position]
        yarray_reshaped[i] = yarray[y_position-1]
    return(xarray_reshaped, yarray_reshaped)

"Always split first and then re-scale"
scalertr=MinMaxScaler(feature_range=(-1,1))
X_train=scalertr.fit_transform(X_train)
scalerte=MinMaxScaler(feature_range=(-1,1))
X_test=scalerte.fit_transform(X_test)

"This will be the time step we are interested in. 15 means 15 rows model will feed to NN" 
seq=15
"Reshape to 3d array"
X_train,y_train=reshapexy(X_train,y_train,seq=seq)
X_test,y_test=reshapexy(X_test,y_test,seq=seq)
#X_train[1]
#y_train[1]

print('After rehsape x train, y train x test y test'+'\n',
      X_train.shape,y_train.shape,X_test.shape,y_test.shape)

"Convert target variables to various different classes"
def ynputil(yarray):
    dfy=pd.DataFrame(yarray)
    #print(dfy[0:10])
    dfy.columns=['y']
    conditions = [
        (dfy['y'] >4) & (dfy['y'] <8),
        (dfy['y'] >14) & (dfy['y'] <18),
        (dfy['y'])>19,
        (dfy['y'] <-3) & (dfy['y'] >-9),
        (dfy['y'] <-13) & (dfy['y'] >-19),
        (dfy['y'])<-19]
    
    choices = [0,1,2,3,4,5]
    
    dfy['y'] = np.select(conditions, choices, default=5)
    
    "verbose..................................................................................."
    net = len(dfy)
    c1 = (dfy['y']==0).sum()
    c2 = (dfy['y']==1).sum()
    c3 = (dfy['y']==2).sum()
    c4 = (dfy['y']==3).sum()
    c5 = (dfy['y']==4).sum()
    c6 = (dfy['y']==5).sum()
        
    print('condition 1: ', c1 ,' out of ' , net , ' percentage ', round(100*(c1/net),2), '%')
    print('condition 2: ', c2 ,' out of ' , net , ' percentage ', round(100*(c2/net),2), '%')
    print('condition 3 (outliers): ', c3 ,' out of ' , net , ' percentage ', round(100*(c3/net),2), '%')
    print('condition 4: ', c4 ,' out of ' , net , ' percentage ', round(100*(c4/net),2), '%')
    print('condition 5: ', c5 ,' out of ' , net , ' percentage ', round(100*(c5/net),2), '%')
    print('condition 6 (outliers): ', c6 ,' out of ' , net , ' percentage ', round(100*(c6/net),2), '%')
    "................................."
    
    y_util=dfy.values
    y_util = np_utils.to_categorical(y_util, 6)
    
    return(y_util)

y_train=ynputil(y_train)    
y_test=ynputil(y_test)    

batch_size = 49 
"""
To find batch size which can divide samples in equal numbers. For stateful LSTM in Keras this is kind of headache
Replace 502 with whatever is your training sample size after transforming X to 3 dimensional array.  X_train.shape[0]
"""
 
#for i in range(1,1400):
#    xy=(502/i)
#    if (xy).is_integer()==True:
#        print('fond factor', i)



# =============================================================================
print('LSTM...')

model = Sequential()
model.add(LSTM(64, kernel_initializer='glorot_uniform', activation='sigmoid',
               batch_input_shape=(batch_size,X_train.shape[1],X_train.shape[2]),stateful=True,
                return_sequences=True))
model.add(Activation('relu'))
#model.add(Dropout(0.3))
#Block 2
model.add(LSTM(128, return_sequences=False))
model.add(Activation('relu'))
#model.add(Dropout(0.3))

model.add(Dense(48))
model.add(Activation('relu'))
#model.add(Dropout(0.2))
model.add(Dense(y_train.shape[1],kernel_initializer='glorot_uniform'))
model.add(Activation('softmax'))
rms=optimizers.RMSprop(lr=0.001, decay=0.0)

"loss........................................................................................."
    
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])#, fmeasure])
model.summary()

checkpointer = (ModelCheckpoint(filepath="lstm_toy.h5", monitor='val_acc',
                                verbose=1, save_best_only=True))
earlystopper = EarlyStopping(monitor='val_acc', patience=355, verbose=1, mode='max')

k=0
trainac=[]
valac=[]
#class_weights={0:1.,1:1.}
history=model.fit(X_train, y_train, epochs=100, batch_size=batch_size, verbose=1, 
                      validation_data=(X_test,y_test),callbacks=[earlystopper,checkpointer],
                      shuffle=False)#,class_weight=class_weights)

trainl=(history.history['acc']) #, label='train')
vall=(history.history['val_acc'])

df_fit=pd.DataFrame({'trainac':trainl,'valac':vall})
df_fit.plot()
plt.show()
scores = model.evaluate(X_test, y_test,batch_size=batch_size, verbose=1)

#scores = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print('scores evaluation ',scores)
print('LSTM test score:', scores[0])
print('LSTM test accuracy:', (scores[1])*100, '%')
