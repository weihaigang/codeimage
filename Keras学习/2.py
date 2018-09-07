import numpy as np
np.random.seed(1337)
from keras.datasets import mnist
from  keras.utils import  np_utils
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.optimizers import RMSprop


(x_train,y_train),(x_test,y_test)=mnist.load_data()



x_train = x_train.reshape(x_train.shape[0],-1)
x_test =  x_test.reshape(x_test.shape[0],-1)
y_train = np_utils.to_categorical(y_train,num_classes=10)
y_test = np_utils.to_categorical(y_test,num_classes=10)


model = Sequential([
    Dense(output_dim=32,input_dim=784),
    Activation('relu'),
    Dense(output_dim=10),
    Activation('softmax')
])

rmsprop =RMSprop(lr=0.001,rho=0.9,decay=0.0)
model.compile(
    optimizer=rmsprop,
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)


model.fit(x_train,y_train,nb_epoch=50,batch_size=32)


loss,accuracy = model.evaluate(x_test,y_test)


print('test loss=',loss)
print('acciracy=',accuracy)




