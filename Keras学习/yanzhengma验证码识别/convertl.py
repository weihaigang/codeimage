#处理数据第二部 把所有图片转换成灰度图片

import os

from PIL import Image

pics_dir='../../data'
processed_pics_dir='../data'
# 将图片灰度化以减少计算压力
def preprocess_pics():
    for (dirpath, dirnames, filenames) in os.walk(pics_dir):
        for filename in filenames:
            if filename.endswith('.png'):
                with open(pics_dir + '/' + filename, 'rb') as f:
                    image = Image.open(f)
                    # 直接使用convert方法对图片进行灰度操作
                    image = image.convert('L')
                    with open(processed_pics_dir + '/' + filename, 'wb') as of:
                        image.save(of)


# preprocess_pics()  会覆盖掉手动标注的数据请慎用
