import os
from urllib import request


for idx in range(1000):
    urlString ='http://www.code.com/public/index.php/index'
    response =  request.urlopen(urlString)
    imageName ='../data/'+ str(idx)+'_'+response.headers['code']+'.png'

    with open(imageName,'wb') as img:
        img.write(response.read())