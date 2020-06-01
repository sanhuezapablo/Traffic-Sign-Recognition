import cv2
import numpy as np
import glob
import bisect

one = cv2.imread('Good_Images/1.jpg')
one = cv2.resize(one,(64,64))
fourteen = cv2.imread('Good_Images/14.PNG')
fourteen = cv2.resize(fourteen,(64,64))
seventeen = cv2.imread('Good_Images/17.jpg')
seventeen = cv2.resize(seventeen,(64,64))
nineteen = cv2.imread('Good_Images/19.jpg')
nineteen = cv2.resize(nineteen,(64,64))
twenty1 = cv2.imread('Good_Images/21.jpg')
twenty1 = cv2.resize(twenty1,(64,64))
thirty5 = cv2.imread('Good_Images/35.jpg')
thirty5 = cv2.resize(thirty5,(64,64))
thirty8 = cv2.imread('Good_Images/38.jpg')
thirty8 = cv2.resize(thirty8,(64,64))
fourty5 = cv2.imread('Good_Images/45.jpg')
fourty5 = cv2.resize(fourty5,(64,64))


def getBlue(roi_label, roi):
    global thirty5
    global thirty8
    global fourty5
    
    if roi_label == 35.0:
        return thirty5
    if roi_label == 38.0:
        return thirty8
    if roi_label == 45.0:
        return fourty5
    
def getRed(roi_label, roi):
    global one
    global fourteen
    global seventeen
    global nineteen
    global twenty1
    
    if roi_label == 1.0:
        return one
    if roi_label == 14.0:
        return fourteen
    if roi_label == 17.0:
        return seventeen
    if roi_label == 19.0:
        return nineteen
    if roi_label == 21.0:
        return twenty1
     

def hsv_test(image):
    image_hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        
    low_blue = np.array([100, 130, 100])
    high_blue = np.array([130, 255, 255])

    frame_threshed = cv2.inRange(image_hsv, low_blue, high_blue)
    opening = cv2.morphologyEx(frame_threshed, cv2.MORPH_OPEN, (10,10))
    opening = cv2.morphologyEx(frame_threshed ,cv2.MORPH_CLOSE, (10,10))
    res = cv2.bitwise_and(image,image, mask= opening)

    res=cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    contours= cv2.findContours(res,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

    for i in contours:
        area=cv2.contourArea(i)
        x,y,w,h = cv2.boundingRect(i)
        aspect_ratio = float(h)/w
        

        if aspect_ratio>0.8 and aspect_ratio<2  and area>100:
            x,y,w,h = cv2.boundingRect(i)
            roi_blue = frame[y:y+h, x:x+w]
            roi_blue = cv2.resize(roi_blue, (64,64))
            roi_label = svm_predict(roi_blue, svm)
            
            if x>64:
                if roi_label == 35 or roi_label == 38 or roi_label == 45:
                    frame[y:y+64, x-64:x] = getBlue(roi_label, roi_blue)
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                    cv2.putText(frame,str(roi_label),(x+20,y+10), font, 1,(255,255,255),2,cv2.LINE_AA)
                    

def hsv_red(image):
    image_hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    image = cv2.GaussianBlur(image, (5, 5), 0)

    low_blue = np.array([0, 15, 70])#good
    high_blue = np.array([10, 250, 250])#good
    frame_threshed = cv2.inRange(image_hsv, low_blue, high_blue)
    opening = cv2.morphologyEx(frame_threshed ,cv2.MORPH_CLOSE, (25,25))
    res = cv2.bitwise_and(image,image, mask= opening)

    
    res=cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    contours= cv2.findContours(res,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

    for i in contours:
        area=cv2.contourArea(i)
        x,y,w,h = cv2.boundingRect(i)
        aspect_ratio = float(h)/w
        rect_area = w*h
        neww=float(w)/(w+h)
        newh=float(h)/(w+h)
        
        extent = area/rect_area
        
        k=cv2.isContourConvex(i)
        perimeter = cv2.arcLength(i,True)
        ratio = perimeter/(np.sqrt(x**2 + y**2))
                
        if area>150 and area<1500 and abs(neww-newh)<0.3 and extent>.23 and extent<.6:# and ratio > 3.5 and ratio < 10:
            if aspect_ratio<.98 and aspect_ratio>0.95:
#                cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
                x,y,w,h = cv2.boundingRect(i)
                roi_blue = frame[y:y+h, x:x+w]
                #roi_blue = deskew(roi_blue)
                roi_label = svm_predict(roi_blue, svm)
            
                if x>64:
                    if roi_label == 1.0 or roi_label == 14.0 or roi_label == 17.0 or roi_label == 19.0 or roi_label == 21.0:
                        print(extent)
                        frame[y:y+64, x-64:x] = getRed(roi_label, roi_blue)
                        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                        cv2.putText(frame,str(roi_label),(x+20,y+10), font, 1,(255,255,255),2,cv2.LINE_AA)
  
def process_image(frame):
    
	median_r = cv2.medianBlur(frame[:,:,2],5)
	median_g = cv2.medianBlur(frame[:,:,1],5)
	median_b = cv2.medianBlur(frame[:,:,0],5)

	norm_r=imadjust(median_r)    
	norm_g=imadjust(median_g)    
	norm_b=imadjust(median_b)    
	
	total_norm=norm_r+norm_g+norm_b
	id_red=np.maximum(0,(np.minimum((norm_r-norm_b),(norm_r-norm_g))/(total_norm)))
	#id_red=np.maximum(0,(np.minimum((norm_r-norm_b),(norm_r-norm_g))))
	id_blue=np.maximum(0,(norm_b-norm_r)/(total_norm))
	#id_blue=np.maximum(0,(norm_b-norm_r))
	
	return id_red, id_blue

def imadjust(src, tol=1, vin=[0,255], vout=(0,255)):
    # src : input one-layer image (numpy array)
    # tol : tolerance, from 0 to 100.
    # vin  : src image bounds
    # vout : dst image bounds
    # return : output img

    assert len(src.shape) == 2 ,'Input image should be 2-dims'

    tol = max(0, min(100, tol))

    if tol > 0:
        # Compute in and out limits
        # Histogram
        hist = np.histogram(src,bins=list(range(256)),range=(0,255))[0]

        # Cumulative histogram
        cum = hist.copy()
        for i in range(0, 255): cum[i] = cum[i - 1] + hist[i]

        # Compute bounds
        total = src.shape[0] * src.shape[1]
        low_bound = total * tol / 100
        upp_bound = total * (100 - tol) / 100
        vin[0] = bisect.bisect_left(cum, low_bound)
        vin[1] = bisect.bisect_left(cum, upp_bound)

    # Stretching
    scale = (vout[1] - vout[0]) / (vin[1] - vin[0])
    vs = src-vin[0]
    vs[src<vin[0]]=0
    vd = vs*scale+0.5 + vout[0]
    vd[vd>vout[1]] = vout[1]
    dst = vd

    return dst
            
            
#def process_img(img):
#    blur = cv2.GaussianBlur(img,(5,5),0)
#    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
#    
#    low_blue = np.array([100, 120, 100])
#    high_blue = np.array([130, 255, 255])
#    blue_threshed = cv2.inRange(hsv, low_blue, high_blue)
#    blue = cv2.morphologyEx(blue_threshed, cv2.MORPH_OPEN, (5,5))
#
#    
#    low_red = np.array([114, 50, 60])
#    high_red = np.array([179, 150, 255])
#    red_threshed = cv2.inRange(hsv, low_red, high_red)
#    red = cv2.morphologyEx(red_threshed, cv2.MORPH_OPEN, (5,5))
#    
#    return blue, red

def svm_predict(image, svm):

    img = cv2.resize(image, (64,64))
    
#    winSize = (60,60)
#    blockSize = (10,10)
#    blockStride = (5,5)
#    cellSize = (10,10)
    
    winSize = (64,64)
    blockSize = (16,16)
    blockStride = (8,8)
    cellSize = (8,8)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                            histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
    
    descriptor_img = hog.compute(img)
    
    testResponse = svm.predict(np.array([descriptor_img]))[1].ravel()
    
    #Gives back the label type (which is the name of the folder of the sign)
    return testResponse[0]

    
svm = cv2.ml.SVM_load('SVM_TEST.xml')    
input_path = glob.glob("input/*.jpg")
N = len(input_path)
font = cv2.FONT_HERSHEY_SIMPLEX


for i in range(500,N):
    frame = cv2.imread(input_path[i])
    frame = cv2.resize(frame, (0,0), fx = .7, fy = .7)
    hsv_test(frame)
    hsv_red(frame)

#    red , blue = process_image(frame)
#    ret,thresh1 = cv2.threshold(red,0.105,255,cv2.THRESH_BINARY)
#    
#    new_red = thresh1.astype(np.uint8)
#    cv2.imshow("new_red",new_red)
#
#    contours= cv2.findContours(new_red,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
#    
#    # cv2.drawContours(frame, contours, -1, (0, 255,0 ), 3)
#    
#
#    for i in contours:
#        area=cv2.contourArea(i)
#        x,y,w,h = cv2.boundingRect(i)
#        aspect_ratio = float(h)/w
#        neww=float(w)/(w+h)
#        newh=float(h)/(w+h)
#        
#        k=cv2.isContourConvex(i)
#        
#        if area>100 and area<3000 and abs(neww-newh)<0.3 :
#            if aspect_ratio<1.1 and aspect_ratio>0.9:
#                roi_red = frame[y:y+h, x:x+w]
#                red_label = svm_predict(roi_red, svm)
#                
#                if x>64:
#                    if red_label == 1.0 or red_label == 14.0 or red_label == 17.0 or red_label == 19.0 or red_label == 21.0:
#                        frame[y:y+64, x-64:x] = getRed(red_label, roi_red)
#                        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
#                        cv2.putText(frame,str(red_label),(x+20,y+10), font, 1,(255,255,255),2,cv2.LINE_AA)
#

    
#    cv2.imshow('blue', blue)
#    cv2.imshow('red', red)
    cv2.imshow('key', frame)
    cv2.waitKey(1)
