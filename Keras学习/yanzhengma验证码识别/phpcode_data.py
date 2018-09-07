import os
import random

import numpy as np
from PIL import Image


IMAGE_HEIGHT=150
IMAGE_WIDTH=70

# captcha_chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
captcha_chars = '0123456789'
char_idx_mappings = {}
idx_char_mappings = {}

for idx, c in enumerate(list(captcha_chars)):
    char_idx_mappings[c] = idx
    idx_char_mappings[idx] = c

MAX_CAPTCHA = 4 #验证码长度
CHAR_SET_LEN = len(captcha_chars)
# 验证码转化为向量
def text2vec(text):
    text_len = len(text)
    if text_len > MAX_CAPTCHA:
        raise ValueError('验证码最长%d个字符'%MAX_CAPTCHA)
    vector = np.zeros(MAX_CAPTCHA*CHAR_SET_LEN)
    for i, c in enumerate(text):
        idx = i * CHAR_SET_LEN + char_idx_mappings[c]
        vector[idx] = 1
    return vector
# 向量转化为验证码
def vec2text(vec):
    text = []
    vec[vec<0.5] = 0
    char_pos = vec.nonzero()[0]
    for i, c in enumerate(char_pos):
        char_idx = c % CHAR_SET_LEN
        text.append(idx_char_mappings[char_idx])
    return ''.join(text)


# 向量转化为验证码
def vec2text1(vec):
    text = []
    for i in range(MAX_CAPTCHA):
        data = vec[i*CHAR_SET_LEN:(i+1)*CHAR_SET_LEN]
        index = data.tolist().index(max(data))
        text.append(idx_char_mappings[index])
    return ''.join(text)







processed_pics_dir='../data'

img_idx_filename_mappings = {}
img_idx_text_mappings = {}
img_idxes = []
# 首先遍历目录，根据文件名初始化idx->filename, idx->text的映射，同时初始化idx列表
for (dirpath, dirnames, filenames) in os.walk(processed_pics_dir):
    for filename in filenames:
        if filename.endswith('.png'):
            idx = int(filename[0:filename.index('_')])
            text = filename[int(filename.index('_')+1):int(filename.index('.'))]
            img_idx_filename_mappings[idx] = filename
            img_idx_text_mappings[idx] = text
            img_idxes.append(idx)


# 为避免频繁读入文件，将images及labels缓存起来
sample_idx_image_mappings = {}
sample_idx_label_mappings = {}
# 提供给外部取得一批训练数据的接口
def get_batch_data(batch_size):
    images = []
    labels = []
    target_idxes = random.sample(img_idxes, batch_size)
    for target_idx in target_idxes:
        image = None
        if target_idx in sample_idx_image_mappings:
            image = sample_idx_image_mappings[target_idx]
        else:
            with open(processed_pics_dir + '/' + img_idx_filename_mappings[target_idx], 'rb') as f:
                image = Image.open(f)
                # 对数据正则化，tensorflow处理时更高效
                image = np.array(image)/255
            sample_idx_image_mappings[target_idx] = image
        label = None
        if target_idx in sample_idx_label_mappings:
            label = sample_idx_label_mappings[target_idx]
        else:
            label = text2vec(img_idx_text_mappings[target_idx])
            sample_idx_label_mappings[target_idx] = label
        images.append(image)
        labels.append(label)
    x = np.array(images)
    y = np.array(labels)
    return (x, y)


(x, y)= get_batch_data(10)

print(x.shape)