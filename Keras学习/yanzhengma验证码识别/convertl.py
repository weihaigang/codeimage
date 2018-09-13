#处理数据第二部 把所有图片转换成灰度图片

import os

from PIL import Image

pics_dir='../data'
processed_pics_dir='./data'
# 将图片灰度化以减少计算压力
def preprocess_pics():
    for (dirpath, dirnames, filenames) in os.walk(pics_dir):
        for filename in filenames:
            if filename.endswith('.png'):
                with open(pics_dir + '/' + filename, 'rb') as f:
                    image = Image.open(f)
                    # 直接使用convert方法对图片进行灰度操作
                    image = image.convert('L')
                    image = convert_Image(image)
                    with open(processed_pics_dir + '/' + filename, 'wb') as of:
                        image.save(of)

def convert_Image(img, standard=127.5):
    '''
    【灰度转换】
    '''
    image = img.convert('L')

    '''
    【二值化】
    根据阈值 standard , 将所有像素都置为 0(黑色) 或 255(白色), 便于接下来的分割
    '''
    pixels = image.load()
    for x in range(image.width):
        for y in range(image.height):
            if pixels[x, y] > standard:
                pixels[x, y] = 255
            else:
                pixels[x, y] = 0
    return image


preprocess_pics()  #会覆盖掉手动标注的数据请慎用
