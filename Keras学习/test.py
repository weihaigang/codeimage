import numpy as np

captcha_chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

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


