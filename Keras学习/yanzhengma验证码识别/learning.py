from pathlib import Path
from keras.models import Sequential
from keras.layers import Dense, InputLayer, Activation
from keras.layers.core import Reshape, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Input, concatenate
from keras.models import Model
from keras.optimizers import Adam

import phpcode_data











model = Sequential()


# 三组卷积逻辑，每组包括两个卷积层及一个池化层

model.add(Conv2D(
    filters=32,
    kernel_size=5,
    strides=(1, 1),
    padding='same',
    use_bias=True,
    input_shape=(70, 150, 3)
))
model.add(Activation('relu'))
model.add(MaxPooling2D(
    pool_size=2,
    strides=2,
    padding='same'
))


model.add(Conv2D(
    filters=64,
    kernel_size=5,
    strides=(1, 1),
    padding='same',
    use_bias=True
))
model.add(Activation('relu'))
model.add(MaxPooling2D(
    pool_size=2,
    strides=2,
    padding='same'
))





model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))

model.add(Dense(phpcode_data.MAX_CAPTCHA * phpcode_data.CHAR_SET_LEN))
model.add(Activation('softmax'))

adam = Adam(lr=1e-4)
# 编译模型，损失函数使用categorical_crossentropy， 优化函数使用adadelta，每一次epoch度量accuracy
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

if Path('kaptcha_recognition.h5').is_file():
    model.load_weights('kaptcha_recognition.h5')
batch_size = 64

for epoch in range(1000):
    print("epoch {}...".format(epoch))
    (x_train, y_train) = phpcode_data.get_batch_data(batch_size)

    x_train = x_train.reshape(-1, 70, 150, 3)
    train_result = model.train_on_batch(x=x_train, y=y_train)
    print(' loss: %.6f, accuracy: %.6f' % (train_result[0], train_result[1]))
    if epoch % 5 == 0:
        # 保存模型的权值`
        model.save_weights('kaptcha_recognition.h5')
    # 当准确率大于0.5时，说明学习到的模型已经可以投入实际使用，停止计算
