##practica 4
import numpy as np
from matplotlib import pyplot as plt
import math as ma
import cv2 #opencv

drawing = False 
mode = True 
ix,iy = -1,-1

def segmentacion(m1):
    img = m1
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cv2.imshow('image', thresh)
    kernel = np.ones((3, 3), np.uint8) 
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations = 2) 
    bg = cv2.dilate(closing, kernel, iterations = 1) 
    dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 0) 
    ret, fg = cv2.threshold(dist_transform, 0.02* dist_transform.max(), 255, 0) 
    cv2.imshow('image', fg)
    cv2.waitKey(0)


def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode


    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    if event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(img,(x,y),5,(0,0,255),-1)

    if event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(img,(x,y),5,(0,0,255),-1)

    


        
img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)
ima1 = cv2.imread('coin-detection.jpg')

while(1):

    cv2.imshow('image',img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('j'):
        segmentacion(ima1)
        break

cv2.destroyAllWindows()
