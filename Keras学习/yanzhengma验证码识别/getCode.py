from pathlib import Path

import numpy as np
from PIL import Image
from keras import Sequential, Input, Model
from keras.engine import InputLayer
from keras.layers import Flatten, Dense, Activation, Conv2D, MaxPooling2D, concatenate, Reshape
from keras.optimizers import Adam

import phpcode_data



model = Sequential()
model.add(InputLayer(input_shape=(phpcode_data.IMAGE_HEIGHT, phpcode_data.IMAGE_WIDTH)))
model.add(Reshape((phpcode_data.IMAGE_HEIGHT, phpcode_data.IMAGE_WIDTH, 1)))

# 三组卷积逻辑，每组包括两个卷积层及一个池化层

model.add(Conv2D(
    filters=32,
    kernel_size=5,
    strides=(1, 1),
    padding='same',
    use_bias=True,
    input_shape=(phpcode_data.IMAGE_HEIGHT, phpcode_data.IMAGE_WIDTH, 1)
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

model.add(Conv2D(
    filters=128,
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


# 全连接层，输出维数是kaptcha_data.MAX_CAPTCHA * kaptcha_data.CHAR_SET_LEN
image_input = Input(shape=(phpcode_data.IMAGE_HEIGHT, phpcode_data.IMAGE_WIDTH))
encoded_image = model(image_input)

encoded_softmax = []
for i in range(phpcode_data.MAX_CAPTCHA):
    out1 = Dense(128, activation="relu")(encoded_image)
    output1 = Dense(phpcode_data.CHAR_SET_LEN, activation="softmax")(out1)
    encoded_softmax.append(output1)
output = concatenate(encoded_softmax)

output1 = Dense(128,activation='relu')(output)
output2 = Dense(phpcode_data.MAX_CAPTCHA * phpcode_data.CHAR_SET_LEN,activation='softmax')(output1)

model = Model(inputs=[image_input], outputs=output2)


adam = Adam(lr=1e-4)
# 编译模型，损失函数使用categorical_crossentropy， 优化函数使用adadelta，每一次epoch度量accuracy
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])


if Path('kaptcha_recognition.h5').is_file():
    model.load_weights('kaptcha_recognition.h5')


def get_single_image(filename):
    images = []
    with open(filename, 'rb') as f:
        image = Image.open(f)
        image = image.convert('L')
        images.append(np.array(image)/255)
    return np.array(images)
# 计算某一张图片的验证码
predicts = model.predict(get_single_image('data/82_1010.png'), batch_size=1)

print(predicts)
print('predict: %s' % phpcode_data.vec2text1(predicts[0]))