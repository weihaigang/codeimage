
# 对训练数据预处理
# 验证码的向量化
# 验证码是形如ncn34之类的字符串，而机器学习时用到的标签也必须是向量，
# 因此写两个方法，分别完成验证码字符串的向量化及反向量化。
# 我们知道一个字符很容易向量化，采用one-hot encoding,
# 那么一个字符串向量化可以简单地把字符one-hot encoding得到的向量拼起来。

# 验证码的可选字符是
import os
import random

import numpy as np
from PIL import Image

















