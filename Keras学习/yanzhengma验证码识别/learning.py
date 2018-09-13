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


# model.add(Dense(1024))
# model.add(Activation('relu'))
#
# model.add(Dense(phpcode_data.MAX_CAPTCHA * phpcode_data.CHAR_SET_LEN))
# model.add(Activation('softmax'))

adam = Adam(lr=1e-4)
# 编译模型，损失函数使用categorical_crossentropy， 优化函数使用adadelta，每一次epoch度量accuracy
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# 模型可视化
from keras.utils import plot_model
plot_model(model, to_file='captcha_recognition_model.png')


if Path('kaptcha_recognition.h5').is_file():
    model.load_weights('kaptcha_recognition.h5')
batch_size = 100

for epoch in range(10000):
    print("epoch {}...".format(epoch))
    (x_train, y_train) = phpcode_data.get_batch_data(batch_size)
    train_result = model.train_on_batch(x=x_train, y=y_train)
    print(' loss: %.6f, accuracy: %.6f' % (train_result[0], train_result[1]))
    if epoch % 5 == 0:
        # 保存模型的权值`
        model.save_weights('kaptcha_recognition.h5')
        # 当准确率大于0.5时，说明学习到的模型已经可以投入实际使用，停止计算
