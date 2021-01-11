import os
import cv2
import time

img_before = r'D:\Dataset\computer_vision\good_before'
img_after = r'D:\Dataset\computer_vision\good_after'
img_groundtruth = r'D:\Dataset\computer_vision\groundtruth'

def rgb2gray_resize(img):
    img = cv2.bilateralFilter(img,20,60,60) #雙邊濾波
    img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_r = cv2.resize(img_g, (1295,971))
    return img_r

def Area(img):
    area = 0
    height, width = img.shape
    for i in range(height):
        for j in range(width):
            if img[i, j] == 255:
                area += 1
    return area

total_iou = 0
total_time = 0
for i in range(7):
    img_1 = cv2.imread(os.path.join(img_before, (str(i+1) + '_before.jpg')))
    img_2 = cv2.imread(os.path.join(img_after, (str(i+1) + '_after.jpg')))
    img_3 = cv2.imread(os.path.join(img_groundtruth, (str(i+1) + '_correct.jpg')))

    area = 0
    #計算執行時間
    start=time.time()
    print(('----------img' + str(i+1) + '----------'))
    #轉灰階 + resize
    img_1r = rgb2gray_resize(img_1)
    img_2r = rgb2gray_resize(img_2)

    #img_2r - img_1r
    img_sub = cv2.subtract(img_2r, img_1r)
    img_sub = cv2.GaussianBlur(img_sub,(3,3),0)
    ret,img_sub = cv2.threshold(img_sub,40,255,cv2.THRESH_BINARY)
    cv2.imshow('img_sub',img_sub)
    cv2.waitKey(0)

    #處理groundtruth
    img_3r = rgb2gray_resize(img_3)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    

    #and
    AND = cv2.bitwise_and(img_sub, img_3r)
    and1 = Area(AND)
    # print('and面積 = ',and1)
   
    #or
    OR = cv2.bitwise_or(img_sub, img_3r)
    or1 = Area(OR)
    # print('or面積 = ',or1)

    #iou = and1/or1
    print('iou = %.4f' %(and1/or1))
    end=time.time()
    print('執行時間：%.2f ' %(end-start))
    total_iou += (and1/or1)
    total_time +=  (end-start)
    # cv2.waitKey(0)

print('----------result----------')
print('平均iou = %.4f '  %(total_iou/7))
print('平均執行時間 = %.2f '  %(total_time/7))
cv2.waitKey(0)