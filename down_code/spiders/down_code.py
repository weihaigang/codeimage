import json
import random

import scrapy
from scrapy import Request


from down_code.items import DownCodeItem


class DownCodeSpider(scrapy.Spider):
    name = "downcode"

    start_urls = [
        'https://m.yubook.net/tool/getVerify/',
    ]
    urlss=[]
    for i in range(10):
        urlss.append( 'https://m.yubook.net/tool/getVerify/')
    
    def parse(self, response):
        downcode = DownCodeItem()
        downcode['file_urls']=self.urlss
        yield downcode


