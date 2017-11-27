#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 14:52:18 2017

@author: cooperjack
"""
#import tensorflow as tf
#tf.
import numpy as np
import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFilter

from numpy import *

#import pygame
#from pygame.locals import * 

import codecs

import math
#from PIL import imageOps

import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')

import cv2
import os 
import csv
import glob
import re
import string


#from zhon.hanzo import punctuation


pathToImage = ''
data_dir = '/Users/cooperjack/Downloads/SynthText_Chinese_version-master/data/'
#pathToFontList = '/Users/cooperjack/Downloads/SynthText_Chinese_version-master/data/fonts/ubuntu/Ubuntu-Bold.ttf'
pathToFontList = os.path.join(data_dir, 'fonts/fontlist.txt')
fontsTable = [os.path.join(data_dir,'fonts',f.strip()) for f in open(pathToFontList)]

def get_images(path):
    files = []
    for ext in ['jpg', 'png', 'jpeg', 'JPG']:
        files.extend(glob.glob(
        os.path.join(path, '*.{}'.format(ext))))
    return files

def save_annoataionCH(finalPos,text,textPath,textName,times):
    finalPath = textPath
    finalPath += '/'
    finalPath += (textName)
    finalPath += ('.txt')
    
    text_polys = []
    text_tags = []
#    if not os.path.exists(finalPath):
#        os.mknod(finalPath)
      
    with open(finalPath, 'wb') as f:
        for index in range(times):
            for item in range(len(finalPos[index])):
                f.write(str(finalPos[index][item]))
                if item <= len(finalPos[index])-2:
                    f.write(',')
            f.write(',')         
            f.write((text[index]).encode('utf8'))
            f.write('\n')
def save_annoataionEN(finalPos,text,textPath,textName,times):
    finalPath = textPath
    #finalPath += '/'
    finalPath += (textName)
    finalPath += ('.txt')
    
    text_polys = []
    text_tags = []

    with open(finalPath, 'w') as f:
        for index in range(times):
            b = len(finalPos[index])
            for item in range(b):
                f.write(str(int(finalPos[index][item])))
                if item <= b-2:
                    f.write(',')
            f.write(',')         
            f.write(text[index])
            f.write('\n')
def load_annoataion(p):
    '''
    load annotation from the text file
    :param p:
    :return:
    '''
    text_polys = []
    text_tags = []
    if not os.path.exists(p):
        return np.array(text_polys, dtype=np.float32)

    with open(p, 'r') as f:
        reader = csv.reader(f) 
        for line in reader:
            label = line[-1]
            # strip BOM. \ufeff for python3,  \xef\xbb\bf for python2
            line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]
            x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, line[:8]))
            text_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            if label == '*' or label == '###':
                text_tags.append(True)
            else:
                text_tags.append(False)
        return np.array(text_polys, dtype=np.float32), np.array(text_tags, dtype=np.bool)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

def GenerateFontPosByOpenCV(img):
    im, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print("num of contours: {}".format(len(contours)))
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)  # threshold
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    dilated = cv2.dilate(thresh, kernel, iterations = 1)  # dilate
    _,contours,_ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # get contours

    mult = 1.2   # I wanted to show an area slightly larger than my min rectangle set this to one if you don't
    img_box = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img_box, [box], 0, (0,255,0), 2) # this was mostly for debugging you may omit
    plt.imshow(img_box)
    plt.show()
    
def GenerateRotatedPnt(origin,angle,pnt):
    tW = pnt[0]-origin[0]
    tH = pnt[1]-origin[1]
    temp = []
    anglePi = angle * math.pi/180.0
    cosA = math.cos(anglePi)
    sinA = math.sin(anglePi)
    temp.append(int((tW*cosA-tH*sinA)+origin[0]) ) 
    temp.append(int((tW*sinA+tH*cosA)+origin[1]) ) 
    pnt[0]=temp[0]
    pnt[1]=temp[1]

def GenerateRotationFontPos(img,size,angle):
    #print "size",size
    tempImg = img;
    input = array(tempImg)
    floodFill = input.copy()
    
    h, w = floodFill.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    # Floodfill from point (0, 0)
    cv2.floodFill(floodFill, mask, (0,0), 255);
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(floodFill)
 
# Combine the two images to get the foreground.
    im_out = input | im_floodfill_inv
    
    debug = 0
    if debug == 1:
        cv2.imshow("im_out", im_out)
        cv2.waitKey(0)
    for index in range(10):
    #print 'dilated:', dilated
        eleSize = max(int(size),2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(eleSize,eleSize))
        dilated = cv2.dilate(array(im_out), kernel)
        if debug == 1:
            cv2.imshow("dilated2", dilated)
            cv2.waitKey(0)

#    eleErose = max(int(size*1),1)
#    kernelErose = np.ones((eleErose,eleErose),'uint8')
#    erosed = cv2.erode(array(dilated), kernelErose)
    
#    eleErose = max(int(size*0.5),1)
#    kernelErose = np.ones((eleErose,eleErose),'uint8')
#    erosed = cv2.erode(array(dilated), kernelErose)
       
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # get contours


    if(len(np.nonzero(dilated))==0):
        print 'Dilated image is empty'
    flattened_list =[]
    mult = 1.2   # I wanted to show an area slightly larger than my min rectangle set this to one if you don't
    img_box = cv2.cvtColor(dilated.copy(), cv2.COLOR_GRAY2BGR)
    rect = []
    if len(contours)==0:
        print 'len of contours is zeros'
        rows,cols = np.nonzero(input)
        minY = min(rows)
        maxY = max(rows)
        minX = min(cols)
        maxX = max(cols)
        
        rect = [minX,minY,maxX-minX,maxY-minY]
        
        flattened_list = [minX, minY, maxX,minY,minX,maxY,maxX,maxY]
    else:
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            box = cv2.cv.BoxPoints(rect)
            box = np.int0(box)
            
            cen= rect[0]; wh=[]; wh.append(rect[1][1]); wh.append(rect[1][0]);
            
            bottomRight=[cen[0]+ 0.5*wh[0],cen[1]+ 0.5*wh[1]];
            rightTop=[cen[0]+ 0.5*wh[0],cen[1]- 0.5*wh[1]];
            leftTop=[cen[0]- 0.5*wh[0],cen[1]- 0.5*wh[1]];
            bottomLeft=[cen[0]- 0.5*wh[0],cen[1]+ 0.5*wh[1]];
            
            GenerateRotatedPnt(cen,angle,bottomLeft)
            GenerateRotatedPnt(cen,angle,bottomRight)
            GenerateRotatedPnt(cen,angle,rightTop)
            GenerateRotatedPnt(cen,angle,leftTop)
            
            rows,cols = np.nonzero(dilated)
            minY = min(rows)
            maxY = max(rows)
            minX = min(cols)
            maxX = max(cols)
            
            rect = [minX,minY,maxX-minX,maxY-minY]
        
            vertice =[]
            vertice.append(rightTop)
            vertice.append(bottomRight )
            vertice.append(bottomLeft)
            vertice.append(leftTop)
            flattened_list = [y for x in vertice for y in x]

#    cv2.drawContours(dilated, [box], 0, (0,255,0), 2) # this was mostly for debugging you may omit
#    cv2.imshow("dilated", dilated)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    return flattened_list, rect

def GenerateFontPos(img):
    #print'nonzero', len(np.nonzero(img))
    temp = np.nonzero(img)
    rows = temp[0]
    cols = temp[1]
    #rows,cols,_ = np.nonzero(img)
    minY = min(rows)
    maxY = max(rows)
    minX = min(cols)
    maxX = max(cols)
    
    rect = [minX,minY,maxX-minX,maxY-minY]
    coord = [minX,minY,maxX,minY,maxX,maxY,minX,maxY]    
    return coord,rect

def GenerateRandomGrayValue(image):
    meanGray = np.mean(np.asarray(image))
    tR = 250
    
    grayH = 100; grayL = 10;
    if meanGray > 200 and meanGray<255 :
        grayL = 10; grayH = 100;
    elif meanGray >50 and meanGray <=200: 
        grayL = 220; grayH = 250;
    elif  meanGray <=50: 
        grayL = 200; grayH = 250;
        
    tR = np.random.randint(grayL,grayH)
    return tR
    
def polygon_area(poly):
    '''
    compute area of a polygon
    :param poly:
    :return:
    '''
    edge = [
        (poly[1][0] - poly[0][0]) * (poly[1][1] + poly[0][1]),
        (poly[2][0] - poly[1][0]) * (poly[2][1] + poly[1][1]),
        (poly[3][0] - poly[2][0]) * (poly[3][1] + poly[2][1]),
        (poly[0][0] - poly[3][0]) * (poly[0][1] + poly[3][1])
    ]
    return np.sum(edge)/2.
def draw_rotated_textSimplify(image, angle, text,leftTopPnt,textSize, fontSize,byLine, *args, **kwargs):
    """ Draw text at an angle into an image, takes the same arguments
        as Image.text() except for:
    :param image: Image to write text into
    :param angle: Angle to write text at
    """
    cen = leftTopPnt;wh = textSize;
#    bottomRight=[origin[0]+ wh[0],origin[1]+ wh[1]];
#    rightTop=[origin[0]+ wh[0],origin[1]];
#    leftTop=[origin[0],origin[1]];
#    bottomLeft=[origin[0],origin[1]+ wh[1]];
    #cen = [origin[0]+0.5*wh[0],origin[1]+0.5*wh[1]]
    bottomRight=[cen[0]+ 0.5*wh[0],cen[1]+ 0.5*wh[1]];
    rightTop=[cen[0]+ 0.5*wh[0],cen[1]- 0.5*wh[1]];
    leftTop=[cen[0]- 0.5*wh[0],cen[1]- 0.5*wh[1]];
    bottomLeft=[cen[0]- 0.5*wh[0],cen[1]+ 0.5*wh[1]];

    GenerateRotatedPnt(cen,angle,bottomRight)
    GenerateRotatedPnt(cen,angle,rightTop)
    GenerateRotatedPnt(cen,angle,leftTop)
    GenerateRotatedPnt(cen,angle,bottomLeft)
    
    polyLine = [(leftTop[0], leftTop[1]),(rightTop[0], rightTop[1]),(bottomRight[0],bottomRight[1]),(bottomLeft[0], bottomLeft[1])]
    
    cor = [leftTop[0], leftTop[1],rightTop[0], rightTop[1],bottomRight[0],bottomRight[1],bottomLeft[0], bottomLeft[1]]
    #print'Geometry Pnt',cor

    # get the size of our image
    width, height = image.size
    max_dim = max(width, height)

    # build a transparency mask large enough to hold the text
    mask_size = (max_dim * 2, max_dim * 2)
    mask = Image.new('L', mask_size, 0)
    #print'mask info:',np.nonzero(array(mask))[0],np.nonzero(array(mask))[1]

    # add text to mask
    draw = ImageDraw.Draw(mask)
    draw.text((max_dim, max_dim), unicode(text), 255, *args, **kwargs)

    if angle % 90 == 0:
        # rotate by multiple of 90 deg is easier
        rotated_mask = mask.rotate(angle)
    else:
        # rotate an an enlarged mask to minimize jaggies
        bigger_mask = mask.resize((max_dim*8, max_dim*8),
                                  resample=Image.BICUBIC)
        rotated_mask = bigger_mask.rotate(angle).resize(
                                  mask_size, resample=Image.LANCZOS)
    # crop the mask to match image

    mask_xy = (int(max_dim - leftTopPnt[0]), int(max_dim - leftTopPnt[1]))
    b_box = mask_xy + (int(mask_xy[0] + width), int(mask_xy[1] + height) )
    mask = rotated_mask.crop(b_box)
    #mask.show()
    #print 'b_Box',b_box
    #print 'mask',(mask.size)		
    #print 'mask size', len(np.nonzero(mask))
    rows,cols = np.nonzero(mask)
	
    #print 'mask rows',len(rows)
    #print 'mask cols',len(cols)
    if len(rows)==0 or len(cols)==0:
        print 'nonzero(mask) is not exist:'
        return [];


    p_area=  polygon_area((polyLine))
    if abs(p_area) < 1:
        print poly
        print('invalid poly')
    dr = ImageDraw.Draw(mask)
    #dr.rectangle(rect,outline='green')
    #dr.polygon(polyLine,outline='green')
    dr.polygon(polyLine,fill=255)
    mask.show()

    fontColor = int(random.randint(10,50))
    color_image = Image.new('RGBA', image.size, fontColor)
    image.paste(color_image, mask)
    
    return cor
def pntIsInRect(pnt,finalTextRect):
    flag = 0
    for item in range(len(finalTextRect)):
        rect = finalTextRect[item]
        if pnt[0]>rect[0]-5 and pnt[0]<rect[0]+rect[2]+10\
            and  pnt[1]>rect[1]-5 and pnt[1]<=rect[1]+rect[3]+10:
            flag =  1
    return flag 
    
def GenerateDrawPos(img,finalTextRect,first,newTextSize,angle):
    [imgW,imgH]= img.size

    blkW = newTextSize[0]
    blkH = newTextSize[1]
    wh=[]
    wh.append( newTextSize[0])
    wh.append( newTextSize[1])

    blkTLX = 30
    blkTLY = 30
    
    if imgW>blkW+90:
        blkTLX = np.random.randint(30,imgW-blkW-50)
    if imgH>blkH+90:
        blkTLY = np.random.randint(30,imgH-blkH-50)
        
    origin=[]
    origin.append((blkTLX))
    origin.append((blkTLY))
    
    newTextIsOut = 0
    iteration = 0;

    while(newTextIsOut==0 and first == 0):

        if imgW>blkW+80:
            blkTLX = np.random.randint(20,imgW-blkW-50)
        else:
            blkTLX = np.random.randint(10,imgW-blkW)
        if imgH>blkH+80:
            blkTLY = np.random.randint(20,imgH-blkH-50)
        else:
            blkTLY = np.random.randint(10,imgH-blkH)
        origin=[]
        origin.append((blkTLX))
        origin.append((blkTLY))

        bottomRight=[origin[0]+ wh[0],origin[1]+ wh[1]];
        rightTop=[origin[0]+ wh[0],origin[1]];
        leftTop=[origin[0],origin[1]];
        bottomLeft=[origin[0],origin[1]+ wh[1]];
        center = [origin[0]+0.5* wh[0],origin[1]+0.5* wh[1]]
        
        GenerateRotatedPnt(origin,angle,bottomRight)
        GenerateRotatedPnt(origin,angle,rightTop)
        GenerateRotatedPnt(origin,angle,leftTop)
        GenerateRotatedPnt(origin,angle,bottomLeft)
        GenerateRotatedPnt(origin,angle,center)
        
        con1 =  pntIsInRect(bottomRight,finalTextRect)
        con2 =  pntIsInRect(rightTop,finalTextRect)
        con3 =  pntIsInRect(leftTop,finalTextRect)
        con4 =  pntIsInRect(bottomLeft,finalTextRect)
        con5 =  pntIsInRect(center,finalTextRect)
        
        if con1==0 and con2==0 and con3 == 0 and con4 == 0 and con5 == 0:
            newTextIsOut = 1
           # print'pntIsOutRect'
            return origin
        else:
            newTextIsOut = 0
        iteration += 1
        if iteration >100:
            break;

    return origin
def pntIsInRectNew(pnt,finalTextRect):
    flag = 0
    for item in range(len(finalTextRect)):
        rect = finalTextRect[item]
        dist = (pnt[0]-rect[0]-0.5*rect[2])*(pnt[0]-rect[0]-0.5*rect[2])\
                    +(pnt[1]-rect[1]-0.5*rect[3])*(pnt[1]-rect[1]-0.5*rect[3])
        dist = np.sqrt(dist)
        thresh = min(rect[2],rect[3])*0.5
        thresh = max(thresh-5,5)
        if dist < thresh:
            flag =  1
    return flag 
def GenerateDrawPosNew(img,finalTextRect,first,newTextSize,angle):
    [imgW,imgH]= img.size

    blkW = newTextSize[0]
    blkH = newTextSize[1]
    wh=[]
    wh.append( newTextSize[0])
    wh.append( newTextSize[1])

    blkTLX = 30
    blkTLY = 30
    
    if imgW>blkW+90:
        blkTLX = np.random.randint(30,imgW-blkW-50)
    if imgH>blkH+90:
        blkTLY = np.random.randint(30,imgH-blkH-50)
        
    origin=[]
    origin.append((blkTLX))
    origin.append((blkTLY))
    
    newTextIsOut = 0
    iteration = 0;

    while(newTextIsOut==0 and first == 0):

        if imgW>blkW+80:
            blkTLX = np.random.randint(20,imgW-blkW-50)
        else:
            blkTLX = np.random.randint(10,imgW-blkW)
        if imgH>blkH+80:
            blkTLY = np.random.randint(20,imgH-blkH-50)
        else:
            blkTLY = np.random.randint(10,imgH-blkH)
        origin=[]
        origin.append((blkTLX))
        origin.append((blkTLY))
        
        topX = list(np.linspace(origin[0],(origin[0]+wh[0]),3))
        topY = list(np.linspace(origin[1],(origin[1]),3))
        
        leftX = list(np.linspace(origin[0],(origin[0]),3))
        leftY = list(np.linspace(origin[1],(origin[1]+wh[1]),3))
        
        rightX = list(np.linspace((origin[0]+wh[0]),(origin[0]+wh[0]),3))
        rightY = list(np.linspace(origin[1],(origin[1]+wh[1]),3))
        
        bottomX = list(np.linspace((origin[0]),(origin[0]+wh[0]),3))
        bottomY = list(np.linspace((origin[1]+wh[1]),(origin[1]+wh[1]),3))
        
        ## 下面这段代码有逻辑bug;
        coordX = []; coordY = [];
        coordX.append(topX); coordX.append(leftX); coordX.append(rightX);coordX.append(bottomX);
        coordY.append(topY); coordY.append(leftY); coordY.append(rightY);coordY.append(bottomY);
        
        flattened_listX = [y for x in coordX for y in x]
        flattened_listY = [y for x in coordY for y in x]
        
        rightVote = 0
        for pntX in range(len(flattened_listX)):
            for pntY in range(len(flattened_listY)):
                pnt=[]
                pnt.append(flattened_listX[pntX])
                pnt.append(flattened_listY[pntY])
                GenerateRotatedPnt(origin,angle,pnt)
                
                if(pntIsInRectNew(pnt,finalTextRect) == 0):
                    rightVote+=1
                if rightVote == 36:
                    return origin
        iteration += 1
        if iteration >100:
            break;
    return origin
def GenerateDrawPosByBlocks(img,txt,pathToFont,row,col,splitNum):
    
    [imgW,imgH]= img.size
    blockW = int(ceil(imgW/splitNum))
    blockH = int(ceil(imgH/splitNum))
    
    ratioMax = 0.5/(len(txt)+1);
    ratioMin = 0.2/(len(txt)+1);
    curRatio = np.random.uniform(ratioMin,ratioMax)
    
    minSize = min(blockW,blockH)
    fontSize = int(curRatio*minSize)
    font = ImageFont.truetype(pathToFont, fontSize)
    
    img_fraction = 0.8
#    print 'font,getSize',font.getsize(txt)
#    print 'threshold',img_fraction*blockW,img_fraction*blockH
#    print 'Font.get',font.getmask(txt).getbbox()
    while (font.getsize(txt)[0] < img_fraction*blockW and font.getsize(txt)[1] < img_fraction*blockH ):
              fontSize += 1
              font = ImageFont.truetype(pathToFont, fontSize)
#              print 'font,getSize',font.getsize(txt)
#              print 'threshold',img_fraction*blockW,img_fraction*blockH
#              print 'Font.get',font.getmask(txt).getbbox()
              
    fontSize -= 1 
    font = ImageFont.truetype(pathToFont, fontSize)
    
    cenX = blockW*(col+0.5)
    cenY = blockH*(row+0.5)
    
    origin=[]
    origin.append(int(cenX-font.getsize(txt)[0]*0.5))
    origin.append(int(cenY-font.getsize(txt)[1]*0.5))
    
    #print'fontSize', fontSize
    return origin,fontSize;
#GenerateDrawPosOnText(emptyImg,chooseString,pathToFont,3,0,splitNum*splitNum,1)
def NewGenerateDrawPosOnText(img,txt,pathToFont,angle, row,col,splitNumR,splitNumC):
    [imgW,imgH]= img.size
    blockW = int(ceil(imgW/splitNumC))
    blockH = int(ceil(imgH/splitNumR))
    
    ratioMax = 0.5/(len(txt)+1);
    ratioMin = 0.2/(len(txt)+1);
    #curRatio = np.random.uniform(ratioMin,ratioMax)
    
    curRatio= 2/(len(txt)+1)
    #minSize = min(blockW,blockH)
    fontSize = int(curRatio*blockW)
    font = ImageFont.truetype(pathToFont, fontSize)
    
    rect=[]
    rect[0]=(blockW*col); rect[1]=(blockH*row); rect[2]=blockW; rect[3]= blockH;
    
    
    img_fraction = 0.8
    center=[]
    count = 0;
    while (count!=4):
        
        cenX = blockW*(col+0.5)
        cenY = blockH*(row+0.5)
        
        wh = font.getsize(txt)
        startX = max(0, int(cenX-wh[0]*0.5))
        startY = max(0, int(cenY-wh[1]*0.5))
        
        leftTop=[cenX-wh[0]*0.5,cenY-wh[1]*0.5]
        rightTop=[cenX+wh[0]*0.5,cenY-wh[1]*0.5];
        bottomRight=[cenX+ wh[0]*0.5,cenY+wh[1]*0.5];
        bottomLeft=[cenX-wh[0]*0.5,cenY+wh[1]*0.5];
        
        center = [cenX,cenY]
        GenerateRotatedPnt(center,angle,bottomRight)
        GenerateRotatedPnt(center,angle,rightTop)
        GenerateRotatedPnt(center,angle,leftTop)
        GenerateRotatedPnt(center,angle,bottomLeft)
        
        count = 0
        con1 =  pntIsInRect(bottomRight,rect)
        con2 =  pntIsInRect(rightTop,rect)
        con3 =  pntIsInRect(leftTop,rect)
        con4 =  pntIsInRect(bottomLeft,rect)
        count = con1+con2+con3+con4
        
        fontSize -= 1
        font = ImageFont.truetype(pathToFont, fontSize)
        
        print 'font,getSize',font.getsize(txt)
        print 'threshold',img_fraction*blockW,img_fraction*blockH
        print 'Font.get',font.getmask(txt).getbbox()

    fontSize += 1 
    font = ImageFont.truetype(pathToFont, fontSize)
    
    #print'fontSize', fontSize
    return center,fontSize;
def GenerateDrawPosOnText(img,txt,pathToFont,row,col,splitNumR,splitNumC):
    
    [imgW,imgH]= img.size
    blockW = int(ceil(imgW/splitNumC))
    blockH = int(ceil(imgH/splitNumR))
    
    ratioMax = 0.5/(len(txt)+1);
    ratioMin = 0.2/(len(txt)+1);
    curRatio = np.random.uniform(ratioMin,ratioMax)
    
    minSize = min(blockW,blockH)
    fontSize = int(curRatio*minSize)
    font = ImageFont.truetype(pathToFont, fontSize)
    
    img_fraction = 0.8
#    print 'font,getSize',font.getsize(txt)
#    print 'threshold',img_fraction*blockW,img_fraction*blockH
#    print 'Font.get',font.getmask(txt).getbbox()
    while (font.getsize(txt)[0] < img_fraction*blockW and font.getsize(txt)[1] < img_fraction*blockH ):
              fontSize += 1
              font = ImageFont.truetype(pathToFont, fontSize)
#              print 'font,getSize',font.getsize(txt)
#              print 'threshold',img_fraction*blockW,img_fraction*blockH
#              print 'Font.get',font.getmask(txt).getbbox()
    fontSize -= 1 
    font = ImageFont.truetype(pathToFont, fontSize)
    
    cenX = blockW*(col+0.5)
    cenY = blockH*(row+0.5)
    
    startX = max(0, int(cenX-font.getsize(txt)[0]*0.6))
    startY = max(0, int(cenY-font.getsize(txt)[1]*0.6))
    origin=[]
    origin.append(int(startX))
    origin.append(int(startY))
    
    #print'fontSize', fontSize
    return origin,fontSize;
def GenerateDrawPosByBlocks(img,txt,pathToFont,row,col,splitNum):
    
    [imgW,imgH]= img.size
    blockW = int(ceil(imgW/splitNum))
    blockH = int(ceil(imgH/splitNum))
    
    ratioMax = 0.5/(len(txt)+1);
    ratioMin = 0.2/(len(txt)+1);
    curRatio = np.random.uniform(ratioMin,ratioMax)
    
    minSize = min(blockW,blockH)
    fontSize = int(curRatio*minSize)
    font = ImageFont.truetype(pathToFont, fontSize)
    
    img_fraction = 0.8
#    print 'font,getSize',font.getsize(txt)
#    print 'threshold',img_fraction*blockW,img_fraction*blockH
#    print 'Font.get',font.getmask(txt).getbbox()
    while (font.getsize(txt)[0] < img_fraction*blockW and font.getsize(txt)[1] < img_fraction*blockH ):
              fontSize += 1
              font = ImageFont.truetype(pathToFont, fontSize)
#              print 'font,getSize',font.getsize(txt)
#              print 'threshold',img_fraction*blockW,img_fraction*blockH
#              print 'Font.get',font.getmask(txt).getbbox()
              
              
    fontSize -= 1 
    font = ImageFont.truetype(pathToFont, fontSize)
    
    cenX = blockW*(col+0.5)
    cenY = blockH*(row+0.5)
    
    origin=[]
    origin.append(int(cenX-font.getsize(txt)[0]*0.5))
    origin.append(int(cenY-font.getsize(txt)[1]*0.5))
    
    #print'fontSize', fontSize
    return origin,fontSize;
def GenerateDrawPosOnNextLine(img,finalTextRect,first,newTextSize,angle):
    
    [imgW,imgH]= img.size
    origin=[]

    blkW = newTextSize[0]
    blkH = newTextSize[1]
    wh=[]
    wh.append( newTextSize[0]) 
    wh.append( newTextSize[1])

    blkTLX = 30
    blkTLY = 30
    
    if imgW>blkW+90:
        blkTLX = np.random.randint(30,int(0.5*imgW))
    if imgH>blkH+90:
        blkTLY = np.random.randint(30,int(0.5*imgH))
        
    origin=[]
    
    if first == 1:
        origin.append((blkTLX))
        origin.append((blkTLY))
    else:
        origin.append((finalTextRect[0]))
        origin.append((finalTextRect[1]+finalTextRect[3]*1.1+8))

    return origin

def GenerateDrawPosbyLines(img,stringLen,rect, first):
    [imgW,imgH]= img.size
    
    ## generate the text region 
    ratioMax = 0.1;
    ratioMin = 0.01;
    curRatio = np.random.uniform(ratioMin,ratioMax)
    
    minSize = min(imgW,imgH)
    fontSize = int(curRatio*minSize)
    
    blkW = fontSize*stringLen;
    blkH = int(fontSize*1.1)
    
    blkTLX = 30
    blkTLY = 30
    if imgW>blkW+90:
        blkTLX = np.random.randint(30,imgW-blkW-50)
    if imgH>blkH+90:
        blkTLY = np.random.randint(30,imgH-blkH-50)
    
    # Generate the text four coordinate
    origin=[]
    if first == 1:
        origin.append(blkTLX)
        origin.append(blkTLY)
    else:
        origin.append(rect[0])
        origin.append(rect[5])
    return origin,fontSize
def GenerateStringCHByLine(filename,stringNum):
    f = codecs.open(filename, 'rb', encoding="utf8")
    data = f.read()##.decode("gbk").encode("utf-8")

    callNumber = np.arange(len(data))
    np.random.shuffle(callNumber)
    pune = '!"#$%&\'()*+,  -./:;<=>?@[\\]^_`{|}~"'
    pune = pune.decode("utf-8")
    punc = "！？｡。＂＃＄％＆＇（）＊＋，－／： ；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
    punc = punc.decode("utf-8")
    chooseString = u''
    index = 0
    strLen = 0
    #stringNum = np.random.randint(4,15)
    while strLen <stringNum:
        strLen = len(chooseString)
        character = data[callNumber[index]]
        d=re.findall(u'[\u4e00-\u9fa5_a-zA-Z0-9！？｡。＂＃＄％＆＇（）＊＋，－／： ；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」]',character)
        if d!='\s'and d!='\n'and d!= '0':
            chooseString += ''.join(d)
        index += 1
    #print chooseString
    return chooseString
## Generate the string to render
def GenerateStringCH(filename,stringNum):
    f = codecs.open(filename, 'rb', encoding="utf8")
    data = f.read()##.decode("gbk").encode("utf-8")

    callNumber = np.arange(len(data))
    np.random.shuffle(callNumber)
    pune = '!"#$%&\'()*+,  -./:;<=>?@[\\]^_`{|}~"'
    pune = pune.decode("utf-8")
    punc = "！？｡。＂＃＄％＆＇（）＊＋，－／： ；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
    punc = punc.decode("utf-8")
    chooseString = u''
    index = 0
    strLen = 0
    #stringNum = np.random.randint(4,15)
    while strLen <stringNum:
        strLen = len(chooseString)
        character = data[callNumber[index]]
        d=re.findall(u'[\u4e00-\u9fa5_a-zA-Z0-9！？｡。＂＃＄％＆＇（）＊＋，－／： ；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“]',character)
        if d!='\s'and d!='\n'and d!= ' ':
            chooseString += ''.join(d)
        index += 1
    #print chooseString
    return chooseString

def GenerateStringEN(path,stringNum):
    
    try:
        fileOpen = open(path,'r')
    except IOError:
        print 'Open text failed!'
        
    r='^[A-Za-z0-9]+$'
    chooseString = '';
    allContent = fileOpen.readlines()
    
    srcLen = len(allContent)
    test  =len(set(chooseString))
    index = 0
    while index <stringNum-1:
            s = np.random.randint(0,srcLen-1)
            tempLen = len(allContent[s])
            if tempLen == 1:
                continue;
            tempIndex = np.random.randint(0,tempLen-1)
            d=re.findall(u'[\u4e00-\u9fa5_a-zA-Z0-9!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]',allContent[s][tempIndex])
            if d !='\t'and d != ' ':
                chooseString +=''.join(d)
                index += 1
    #print 'length', len(set(chooseString))
    xx = chooseString.replace('\n','')
    yy = xx.replace('\\','')
    yy = yy.replace('','')
    
    return yy

from time import clock
def GenerateAnotationImgBlock(startIndex,novelPath,dataSetPath,dataSetNum, outputPath,english,byLine,splitNum):
    files = get_images(dataSetPath)
    ## for circle 
    index = 1+startIndex;
    viz = 1
    tempS = ['']
    maxNum = min(len(files),dataSetNum)

    indexlist   = range(len(files))
    np.random.shuffle(indexlist)
    
    #splitNum = 3
    writeTimes = splitNum*splitNum
    for item in (indexlist):
        img=Image.open(files[item])
        [imgW,imgH]= img.size
        if imgW<=0 or imgH <=0:
            print'invalid input!', files[item]
            continue;
        if imgW== []or imgH ==[]:
            print'invalid input!', files[item]
            continue;
        tR = np.random.randint(200,250)
        emptyImg = Image.new('L', img.size, int(tR))
        finalCoord = []
        finalText = []
        newTextSize=[];
        curAngle = 0;
        textRect = [(0,0)]
        finalTextRect= []
        
        maxAngle = 2
        curAngle = np.random.randint(-maxAngle,maxAngle)

        first = 1
        stringNum = np.random.randint(15,45)
        for times in range(writeTimes):
            
            row= int(floor(times/splitNum))
            col = int(times%splitNum)
            #print row,col
            start = clock()
            if english == 1:
                chooseString = GenerateStringEN(novelPath,stringNum)
            else:
                chooseString = GenerateStringCH(novelPath,stringNum)
            finish= clock()
            print('GenerateString:{0}'.format(finish-start))
            
            start = clock()
            pathToFont = fontsTable[np.random.randint(0,len(fontsTable)-1)]
            
            fontSize=10
            font = ImageFont.truetype(pathToFont, fontSize)
            curAngle = np.random.randint(-maxAngle,maxAngle)
            
            origin,fontSize = GenerateDrawPosOnText(emptyImg,chooseString,pathToFont,times,0,splitNum*splitNum,1);
            #origin,fontSize = GenerateDrawPosByBlocks(emptyImg,chooseString,pathToFont,row,col,splitNum)
            font = ImageFont.truetype(pathToFont, fontSize)
            
            newTextSize = font.getsize(chooseString)

            finish= clock()
            #print('GenerateDrawPosOnText:{0}'.format(finish-start))
            pnt=[]
            pnt.append(origin[0]+newTextSize[0])
            pnt.append(origin[1]+newTextSize[1])
            
            start = clock()
            cor = draw_rotated_textSimplify(emptyImg, curAngle, chooseString, origin,newTextSize,byLine,fontSize, font=font)
            if cor == []:
                continue;
                
            finish= clock()
            #print('draw_rotated_textSimplify:{0}'.format(finish-start))
            #finalTextRect.append(textRect);
            #finalCoord.append(cor)
            finalCoord.append(cor)
            finalText.append(chooseString)
            
            if first == 1:
                first = 0
            
        textPath = outputPath
        textName = 'img_'+str(index)
        
        start = clock()
        print "Num", index
        if english == 1:
            save_annoataionEN(finalCoord,finalText,textPath,textName,writeTimes)
        else:
            save_annoataionCH(finalCoord,finalText,textPath,textName,writeTimes)
            
        finish= clock()
        #print('save_annoataion:{0}'.format(finish-start))
        
        if viz==1:
            #dr = ImageDraw.Draw(emptyImg)
            #dr.polygon(polyLine,outline='green')
            emptyImg.show()
        
        ## Generate the Position, size, angle of the string 
        outputImgPath = outputPath 
        outputImgPath += 'img_'+str(index)
        outputImgPath += '.jpg'
        emptyImg.save(outputImgPath)
        
        index = index+1
        if index-startIndex>maxNum+1:
            break; 


def testImageAlphaSet(imgPath):
    formalImg = PIL.Image.open(imgPath)
    
    im = formalImg.convert("L")
    array = np.array(im)
    
    formalImg = formalImg.convert('RGBA')
    
    mask = np.ones(array.shape,dtype=np.uint8)*255
    rect= [5,10,6,400]
    mask[0:100,0:200] = 0
    maskImg = PIL.Image.fromarray(mask)
 #   formalImg[:,:,:,0] = formalImg[np.where(mask==1)if formalImg[:,:,:,0] else 0]
    formalImg.putalpha(maskImg);
    #formalImg[:,:,:,0] = mask
    
    tempRegion = formalImg[100:300,200:500,:,:]
    r,g,b,a= formalImg.split()
    a.show();
    tempRegion.save("formalImg.png")
    
    
    
dataSetPath = '/Users/cooperjack/Documents/TestImage/Image/'
outputPath = '/Users/cooperjack/Documents/TestImage/Output/'

projPath = '/Users/cooperjack/Documents'
pathToCHText = projPath+'/EAST/chCh.txt' #Test
pathToENText = projPath+'/EAST/Alice.txt'

novelPath = pathToENText
dataSetNum = 1
english = 1


imgPath = dataSetPath + 'img_1.jpg'
testImageAlphaSet(imgPath)

#splitNum = 2
startIndex = 0
#GenerateAnotationImgBlock(startIndex,novelPath,dataSetPath,dataSetNum,outputPath,english,1,splitNum)
#
#splitNum = 3
#startIndex = startIndex+100
#GenerateAnotationImgBlock(startIndex,novelPath,dataSetPath,dataSetNum,outputPath,english,1,splitNum)
#
#splitNum = 2
#startIndex = startIndex+100
#GenerateAnotationImgBlock(startIndex,novelPath,dataSetPath,dataSetNum, outputPath,english,0,splitNum)
#
#startIndex = startIndex+100
#splitNum = 3
#GenerateAnotationImgBlock(startIndex,novelPath,dataSetPath,dataSetNum, outputPath,english,0,splitNum)
##

### Chinese
#novelPath = pathToCHText
#startIndex = startIndex+2
#english = 0
#
#startIndex = startIndex+2
#splitNum = 3
#GenerateAnotationImgBlock(startIndex,novelPath,dataSetPath,dataSetNum,outputPath,english,1,splitNum)
#
#startIndex = startIndex+2
#splitNum = 3
#GenerateAnotationImgBlock(startIndex,novelPath,dataSetPath,dataSetNum,outputPath,english,1,splitNum)

#startIndex = startIndex+2
#splitNum = 2
#GenerateAnotationImgBlock(startIndex,novelPath,dataSetPath,dataSetNum,outputPath,english,0,splitNum)
#
#startIndex = startIndex+2
#splitNum = 3
#GenerateAnotationImgBlock(startIndex,novelPath,dataSetPath,dataSetNum,outputPath,english,0,splitNum)









