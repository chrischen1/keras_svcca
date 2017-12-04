from keras import backend as K
import numpy as np
import pandas as pd
import cca_core as cc
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.datasets import cifar10


np.random.seed(2017)     
(train_features, train_labels), (test_features, test_labels) = cifar10.load_data()
input_shape=(32, 32,3)
num_classes = len(np.unique(train_labels))

x_train = test_features.astype('float32')/255
y_train = np_utils.to_categorical(test_labels, num_classes)


model1 = cc.get_cnn_model(input_shape,num_classes)
model2 = cc.get_dense_model(input_shape,num_classes)

history1 = model1.fit(x_train, y_train,epochs=60,batch_size=256,validation_split=0.1)
history2 = model2.fit(x_train, y_train,epochs=60,batch_size=256,validation_split=0.1)

model_performance = cc.output_model(history1)
model_performance.to_csv('performance1.csv')
model_performance = cc.output_model(history2)
model_performance.to_csv('performance2.csv')

act1 = cc.get_acts_from_model(model1,x_train[range(600)])
act2 = cc.get_acts_from_model(model2,x_train[range(600)])

cca = cc.get_cca_similarity(act1, act2,compute_dirns=False)
pd.DataFrame(cca['cca_dirns1']).round(3).to_csv('cca1.csv')
pd.DataFrame(cca['cca_dirns2']).round(3).to_csv('cca2.csv')


#pd.DataFrame(act1).to_csv('acts1.csv')
# history = model4.fit(x_train, y_train,epochs=12,batch_size=128,validation_data=(x_test,y_test))
# model_info = cc.output_model(history)
# model_info.to_csv('model_history.csv')
