
# coding: utf-8

# In[30]:


# coding=utf-8
import uuid
import sys
import cv2
import numpy as np
import os

is_cut = False
class plateCut:
    def __int__(self):
        self.is_cut = False
    
    def preprocess(self,gray,iterations):
        #gaussian smoothing
        gaussian = cv2.GaussianBlur(gray,(3,3),0,0,cv2.BORDER_DEFAULT)
        #median smoothing
        median = cv2.medianBlur(gaussian,5)
        #sobel operator
        sobel = cv2.Sobel(median,cv2.CV_8U,1,0,ksize=3)
        #binaryzation, threshold amount=170
        ret,binary = cv2.threshold(sobel,170,255,cv2.THRESH_BINARY)
        e1 = cv2.getStructuringElement(cv2.MORPH_RECT,(9,1))
        e2 = cv2.getStructuringElement(cv2.MORPH_RECT,(9,7))
        dilation = cv2.dilate(binary,e2,iterations = 1)
        erosion = cv2.erode(dilation,e1,iterations = 1)
        dilation2 = cv2.dilate(erosion,e2,iterations = iterations)
        return dilation2

    def the_area_plate(self,img):
        region = []
        binary,contours,hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for i in range(len(contours)):
            ct = contours[i]
            area = cv2.contourArea(ct)
            if (area<2000):
                continue
            epsilon = 0.001 * cv2.arcLength(ct,True)
            approx = cv2.approxPolyDP(ct, epsilon, True)
            srect = cv2.minAreaRect(ct)
            box = cv2.cv2.boxPoints(srect)
            box = np.int0(box)
            h = abs(box[0][1]-box[2][1])
            w = abs(box[0][0]-box[2][0])
            ratio = float(w)/float(h)
            if (ratio>5 or ratio<2):
                continue
            region.append(box)
        return region

    def detect(self,img,iterations, is_infer=False):
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        dilation = self.preprocess(gray,iterations)
        region = self.the_area_plate(dilation)
        for box in region:
            cv2.drawContours(img,[box],0,(0,0,255),2)
        if len(region)>0:
            box = region[0]
            ys = [box[0,1],box[1,1],box[2,1],box[3,1]]
            xs = [box[0,0],box[1,0],box[2,0],box[3,0]]
            ys_sorted_index = np.argsort(ys)
            xs_sorted_index = np.argsort(xs) 
            x1 = box[xs_sorted_index[0],0]
            x2 = box[xs_sorted_index[3],0]
            y1 = box[ys_sorted_index[0],1]
            y2 = box[ys_sorted_index[3],1]
            img_cy = img.copy()
            plate_img = img_cy[y1:y2,x1:x2]
            if is_infer:
                cv2.imwrite('../images/infer.jpg',plate_img)
            else:
                cv2.imwrite('../data/%s.jpg'% self.img_name,plate_img)
        else:
            if self.is_cut:
                pass
            else:
                self.is_cut = True
                self.detect(img,3)
                
    def strat_crop(self,imagePath, is_infer=False,name=None):
        self.is_cut = False
        if not is_infer:
            self.img_name = name.split('.')[0]
        img = cv2.imread(imagePath)
        self.detect(img=img, iterations=6, is_infer=is_infer)

if __name__=='__main__':
    plateCut = plateCut()
    img_path = 'images/tem/'
    imgs = os.listdir(img_path)
    for img in imgs:
        plateCut.strat_crop(img_path + img, False,img)

