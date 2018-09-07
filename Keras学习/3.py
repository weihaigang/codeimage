from keras.datasets import mnist
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten,Conv2D
from keras.optimizers import Adam
from keras.utils import np_utils
from keras import Sequential


(x_train,y_train),(x_test,y_test)=mnist.load_data()



x_train = x_train.reshape(-1,1,28,28)
x_test =  x_test.reshape(-1,1,28,28)
y_train = np_utils.to_categorical(y_train,num_classes=10)
y_test = np_utils.to_categorical(y_test,num_classes=10)


model =Sequential()

model.add(Conv2D(
    filters=32,
    kernel_size=(5,5),
    padding='same',
    input_shape=(1,28,28)
))
model.add(Activation('relu'))
model.add(MaxPooling2D(
    pool_size=(2,2),
    strides=2,
    padding='same'
))
model.add(Conv2D(
    filters=64,
    kernel_size=(5,5),
    padding='same',
))
model.add(Activation('relu'))
model.add(MaxPooling2D(
    pool_size=(2,2),
    strides=2,
    padding='same'
))


model.add(Flatten())

model.add(Dense(1024))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('softmax'))

adam=Adam(lr=1e-4)

model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])



model.fit(x_train,y_train,nb_epoch=1,batch_size=32)

loass,accuracy=model.evaluate(x_test,y_test)

print('loss',loass)
print('\n',accuracy)
