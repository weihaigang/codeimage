import numpy as np
from PIL import Image
from keras import Sequential
from keras.layers import Flatten, Dense, Activation, Conv2D, MaxPooling2D
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
    input_shape=(70, 150, 4)
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

def get_single_image(filename):
    images = []
    with open(filename, 'rb') as f:
        image = Image.open(f)
        # image = image.convert('L')
        images.append(np.array(image)/255)
    return np.array(images)
# 计算某一张图片的验证码
predicts = model.predict(get_single_image('../data/0_0593.png'), batch_size=1)

print('predict: %s' % phpcode_data.vec2text1(predicts[0]))