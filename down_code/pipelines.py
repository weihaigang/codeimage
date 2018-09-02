# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html
import logging

import scrapy
from scrapy import Request
from scrapy.pipelines.images import ImagesPipeline

import random

from scrapy.utils.defer import mustbe_deferred, defer_result
from scrapy.utils.log import failure_to_exc_info
from scrapy.utils.request import request_fingerprint
from twisted.internet.defer import Deferred

logger = logging.getLogger(__name__)
class DownCodePipeline(ImagesPipeline):
    #def process_item(self, item, spider):
     #   return item
    index=0
    def _process_request(self, request, info):
        fp = request_fingerprint(request)
        cb = request.callback or (lambda _: _)
        eb = request.errback
        request.callback = None
        request.errback = None

        # Return cached result if request was already seen
        # if fp in info.downloaded:
        #     return defer_result(info.downloaded[fp]).addCallbacks(cb, eb)
        #
        # # Otherwise, wait for result
        wad = Deferred().addCallbacks(cb, eb)
        info.waiting[fp].append(wad)
        #
        # # Check if request is downloading right now to avoid doing it twice
        # if fp in info.downloading:
        #     return wad

        # Download request checking media_to_download hook output first
        info.downloading.add(fp)
        dfd = mustbe_deferred(self.media_to_download, request, info)
        dfd.addCallback(self._check_media_to_download, request, info)
        dfd.addBoth(self._cache_result_and_execute_waiters, fp, info)
        dfd.addErrback(lambda f: logger.error(
            f.value, exc_info=failure_to_exc_info(f), extra={'spider': info.spider})
        )
        return dfd.addBoth(lambda _: wad)  # it must return wad at last

    def get_media_requests(self, item, info):
        for image_url in item['file_urls']:
            yield Request(image_url,dont_filter=True)
    def file_path(self, request, response=None, info=None):
        self.index=self.index+1
        down_file_name = '{0}.png'.format(str(self.index))
        return down_file_name

