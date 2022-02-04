#
import os
import re
import glob

import cv2
import pyocr
import japanize_matplotlib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from natsort import natsorted
from datetime import date, timedelta


def view(img):
    """
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(16, 9))
    plt.imshow(img)

def crop(img, crop_range):
    """
    """
    x1, x2, y1, y2 = crop_range
    return img[x1:x2, y1:y2, :]

def search(match, sort, min_, max_):
    """
    テンプレートマッチングでmatchを算出した後、マークが決められた場所に存在しているかを確認する
    
    Parameters
    ----------
    match : cv2.matchTemplate()の返り値
    sort : sortしたmatch
    min_ : マーカーが存在する左端
    max_ : マーカーが存在する右端
        
    Returns
    --------
    p : match度（0~1）
    sort[0][n] : y座標
    sort[1][n] : x座標
    """
    p = 0
    d = np.count_nonzero(sort[1] < min_)
    for i in range(np.count_nonzero((min_ <= sort[1]) & (sort[1] <= max_))):
        if p <= match[sort[0][i + d]][sort[1][i + d]] :
            p = match[sort[0][i + d]][sort[1][i + d]]
            n = i + d   
            
    if p == 0: 
        return p, None, None
    
    else:
        return p, sort[0][n], sort[1][n]

def template_matching(img, temp_path, after_color, fill_color, color_list, min_list):
    """
    テンプレートマッチング
    
    Parameters
    ----------
    img : 対象画像
    temp_path : テンプレート画像のパス
    after_color : マーカーの色（RGB）
    fill_color : マーカー以外を塗りつぶす色（RGB）
    color_list : マーカーの色（RGB）
    min_list : マーカーの存在するピクセル値
        
    Returns
    --------
    list_ : マーカーの座標とmatch度
    h : マーカーの縦幅
    w : マーカーの横幅
    """
    img_cp = img.copy()
    temp = cv2.imread(temp_path)
    h, w = temp.shape[:2]
    
    for RGB in color_list:
        img_cp[np.where((img_cp == RGB).all(axis=2))] = after_color
        
    img_cp[np.where((img_cp != after_color).any(axis=2))] = fill_color
    match = cv2.matchTemplate(img_cp, temp, cv2.TM_CCOEFF_NORMED)
    
    if after_color == [255, 0, 0]:
        loc = np.where(match >= 0)
    else:
        loc = np.where(match >=0.15)

    loc_stack = np.stack([loc[0], loc[1]], 0)

    loc_sort = loc_stack[:, loc_stack[1, :].argsort()]

    list_ = [search(match, loc_sort, min_list[i]-10, min_list[i]+3)[:] for i in range(len(min_list))]

    return list_ , h, w

def draw(img, list_, h, w, color):
    """
    テンプレートマッチングした結果を描画
    
    Parameters
    ----------
    img : 
    list_ : 
    h : 
    w : 
    color :
       
    Returns
    --------

    """
    for  i in range(len(list_)):
        try:
            cv2.rectangle(img, (list_[i][2] - 1, list_[i][1] - 1), (list_[i][2] + w, list_[i][1] + h), color=color, thickness=1)
        except TypeError:
            pass
        
def read_pixel(image_path, crop_range, threshold):
    """
    横軸の直線を検出
    
    Parameters
    ----------
    image_path : 対象画像
    crop_range : グラフが描画してある座標
    threshold : 二値化の閾値
       
    Returns
    --------
    y_list : 検出した直線の座標
    mean : 一区間あたりのピクセル値

    """
    img = cv2.imread(image_path)
    y1, y2, x1, x2 = crop_range
    img_cropped = img[y1:y2, x1:x2, :]
    gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    ret, img_thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    lines = cv2.HoughLinesP(img_thresh, rho=1, theta=np.pi/360, threshold=threshold,
                            minLineLength=550, maxLineGap=0)

    y_list = [lines[i, 0, 1] for i in range(len(lines))]
    y_list.sort()

    y_sum = 0
    
    #PDF画像は必ずx軸が入るので―2
    
    for i in range((len(y_list) - 2)):
        y_sum += y_list[i + 1] - y_list[i]

    mean = y_sum / (len(y_list) - 2)
    
    return y_list, mean

def crop2ocr(image_path, crop_range, padding_range):
    """
    年月日を読むための前処理
    
    Parameters
    ----------
    image_path : 入力画像
    crop_range : 年月日の書いてある座標
    padding_range : 読み取りに邪魔な部分を塗りつぶす
       
    Returns
    --------
    pil_image : OCR()の引数


    """
    img = cv2.imread(image_path)
    left_p, upper_p, right_p, lower_p = padding_range
    cv2.rectangle(img, (left_p, upper_p), (right_p, lower_p), (255, 255, 255), -1)
    left_c, upper_c, right_c, lower_c = crop_range
    img_cropped = img[upper_c:lower_c, left_c:right_c]
    
    ret, tho = cv2.threshold(img_cropped, 100, 255, cv2.THRESH_BINARY)
    kernel = np.ones((2,2),np.uint8)
    dst = cv2.erode(tho, kernel, iterations=1)
    pil_image = Image.fromarray(dst)
    
    return pil_image

def img_crop(image_path, crop_range):
    """
    画像のトリミング
    
    Parameters
    ----------
    image_path : 
       
    Returns
    --------
    pil_image : OCR()の引数

    """
    img = cv2.imread(image_path)
    left_c, upper_c, right_c, lower_c = crop_range
    img_cropped = img[upper_c:lower_c, left_c:right_c]
    pil_image = Image.fromarray(img_cropped)
    
    return pil_image
    
def OCR(pil_image, lang):
    """
    光学文字認識
    
    Parameters
    ----------
    pil_image : 入力画像
    lang : 読み取る言語 
       
    Returns
    --------
    result : 読み取り結果

    """    
    #pyocr.tesseract.TESSERACT_CMD = tesseract_install_path
    tools = pyocr.get_available_tools()
    tool = tools[0]
    builder = pyocr.builders.TextBuilder()
    result = tool.image_to_string(pil_image, lang=lang, builder=pyocr.builders.DigitBuilder(tesseract_layout=6))
    
    return result

def filter_yaxis(ocr_result):
    """
    OCRの読み取り結果を切り分け
    
    Parameters
    ----------
    ocr_result : OCRの読み取り結果 
       
    Returns
    --------
    ticks_list : 切り分けた読み取り結果

    """
    ticks = re.sub(r'\D', ',', ocr_result)
    ticks_list_str = ticks.split(',')
    ticks_list_filter = list(filter(lambda x: x != '', ticks_list_str))
    ticks_list = list(map(int, ticks_list_filter))
    
    return ticks_list

def get_date_type(ocr_date):
    """
    明石水温のタイプAとタイプBを判断する
    
    Parameters
    ----------
    ocr_date : OCRの読み取り結果 
       
    Returns
    --------
    date_list : 横軸の日付
    type_check : 想定したタイプが合っているかどうか（bool値）

    """    
    try:
        year = ocr_date.split("年")
        month = year[1].split("月")
        day = month[1].split("日")
        d = date(int(year[0]), int(month[0]), int(day[0]))
        date_list = list(reversed([d - timedelta(days=i) for i in range(32)]))
        
        type_check = True
        
        return date_list, type_check
    
    except ValueError:
        print("The date could not be read")
        
        date_list = None
        type_check = True
        
        return date_list, type_check
    
    except IndexError:
        
        date_list = None
        type_check = False
        
        return date_list, type_check
        

        
       