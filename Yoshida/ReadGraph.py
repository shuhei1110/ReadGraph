#Call 
import os
import re
import glob
import errno
import configparser

import cv2
import pyocr
import numpy as np
import pandas as pd
import japanize_matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from natsort import natsorted
from datetime import date, timedelta


from template_module import temp_PDF as temp


class ReadGraph:

    _defaults = {
        "input_dir": "./src/",
        "output_dir": "./dst/",
        "config_path": "template_module/temp_config_PDF.ini"
    }
    
    @classmethod
    def get_defaults(cls, key):
        if key in cls._defaults:
            return cls._defaults[key]
        else:
            return "Unrecognized attribute name '" + key + "'"
        
    def __init__(self, location, target, **kwargs):
        self.__dict__.update(self._defaults)
        self.__dict__.update(kwargs)
        self.location = location
        self.target = target
        self.section = location + "_" + target
        self.config_ini, self.default_conf = self.set_config()
        
    def set_config(self):
        """
        設定ファイルの読み込み

        Parameters
        ----------
       
        Returns
        --------

        """
        config_ini = configparser.ConfigParser()
        config_ini_path = self.__dict__["config_path"]

        # 指定したiniファイルが存在しない場合、エラー発生
        if not os.path.exists(config_ini_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), config_ini_path)

        config_ini.read(config_ini_path, encoding='utf-8')
        DEFAULT_conf = config_ini["DEFAULT"]
        
        return config_ini, DEFAULT_conf
       
    @classmethod
    def get_image_path(self):
        """
        入力画像のパスを取得
        """
        image_paths = natsorted(glob.glob(self.__dict__["input_dir"] + "*.png"))
        
        return image_paths
    
    @classmethod
    def get_image_size(self, image_path):
        """
        入力画像のサイズを取得
        """        
        img = cv2.imread(image_path)
        size = img.shape
        
        return size
    
    @classmethod
    def get_type(self):
        """
        グラフのタイプを取得
        """       
        img_type = self.config_ini[self.section]["type"]
        
        return img_type
    
    def read_graph(self, image_path):
        """
        mainの関数

        Parameters
        ----------
        image_path : 入力画像
       
        Returns
        --------
        dfObj : グラフの読み取り結果（pandas.DataFrame）
        
        Flow
        --------
        1. 年月日の取得＆設定したタイプが合っているか確認
        ↓
        2. タイプが合っていなければ変更して再度年月日を読み取り
        ↓
        3. テンプレートマッチング
        ↓
        4. 縦軸の目盛読み取り
        ↓
        5. 目盛の直線を検出
        ↓
        6. マーカーの値を算出
        ↓
        7. データフレームを作成
        ↓
        8. データフレームを出力
        
        """  
        #1
        img_type = self.get_type()
        crop_range = eval(self.config_ini[self.section]["date_range"])
        padding_range = eval(self.config_ini[self.section]["padding_range"])
        #タイプAとタイプBで前処理が異なる
        if img_type == "A":
            pil_image = temp.img_crop(image_path, crop_range)
        else:
            pil_image = temp.crop2ocr(image_path, crop_range, padding_range)
        ocr_date = temp.OCR(pil_image, lang='jpn')
        
        date_list, type_check = temp.get_date_type(ocr_date)
   
        #2
        if type_check:
            pass
        
        else:
            self.section = self.section + "_B"
            crop_range = eval(self.config_ini[self.section]["date_range"])
            padding_range = eval(self.config_ini[self.section]["padding_range"])
            pil_image = temp.crop2ocr(image_path, crop_range, padding_range)
            ocr_date = temp.OCR(pil_image, lang='jpn')

            date_list, type_check = temp.get_date_type(ocr_date)
            print("change typeA to typeB")
        
        #3
        img = cv2.imread(image_path)
        y1, y2, x1, x2 = eval(self.config_ini[self.section]["match_range"])
        img_cropped = img[y1:y2, x1:x2, :]
        
        fill_color = eval(self.default_conf["fill_color"])
        min_list = eval(self.config_ini[self.get_type()]["min_list"])
        

        list_1, h1, w1 = temp.template_matching(img_cropped, eval(self.default_conf["red_mark_path"]),
                                                [0, 0, 255], fill_color, eval(self.default_conf["red_list"]), min_list)
        list_2, h2, w2 = temp.template_matching(img_cropped, eval(self.default_conf["blue_mark_path"]),
                                                [255, 0, 0], fill_color, eval(self.default_conf["blue_list"]), min_list)
        list_3, h3, w3 = temp.template_matching(img_cropped, eval(self.default_conf["black_mark_path"]),
                                                [0, 0, 0], fill_color, eval(self.default_conf["black_list"]), min_list)

        #template matching 書き出し
        """
        temp.draw(img_cropped, list_1, h1, w1, [0, 0, 255])
        temp.draw(img_cropped, list_2, h2, w2, [255, 0, 0])
        temp.draw(img_cropped, list_3, h3, w3, [0, 255, 0])
        cv2.imwrite(self.__dict__["output_dir"] + os.path.split(image_path)[1], img_cropped)
        """

        #4
        pil_image_axis = temp.img_crop(image_path, eval(self.config_ini[self.section]["axis_range"]))
        ocr_result = temp.OCR(pil_image_axis, lang='digits')
        ticks_list = temp.filter_yaxis(ocr_result)
        
        #5
        y_list, mean = temp.read_pixel(image_path, eval(self.config_ini[self.section]["match_range"]),
                                       eval(self.default_conf["threshold"]))

        y_base = y_list[-2]
        y_min = ticks_list[-1]        
        
        #6
        y1 = [None if list_1[i][1]==None else y_min + (y_base-list_1[i][1])/mean for i in range(len(list_1))]
        y2 = [None if list_2[i][1]==None else y_min + (y_base-list_2[i][1])/mean for i in range(len(list_2))]
        y3 = [None if list_3[i][1]==None else y_min + (y_base-list_3[i][1])/mean for i in range(len(list_3))]
        
        #7
        dfObj = pd.DataFrame(columns=['year', 'month', 'day', 'value1', 'value2', 'value3'])
        for i in range(7):
            dfObj = dfObj.append({'year': date_list[i].year, 'month': date_list[i].month,
                                  'day': date_list[i].day, 'value1': y1[i], 'value2': y2[i],
                                  'value3': y3[i]}, ignore_index=True)
        #8
        return dfObj
