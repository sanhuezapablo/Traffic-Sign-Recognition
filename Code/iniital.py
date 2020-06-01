import numpy as np
import cv2
import glob
import bisect
import copy
import matplotlib.pyplot as plt


one = cv2.imread('Good_Images/1.jpg')
one= cv2.resize(one, (64,64))
fourteen = cv2.imread('Good_Images/14.png')
fourteen= cv2.resize(fourteen, (64,64))
seventeen = cv2.imread('Good_Images/17.jpg')
seventeen = cv2.resize(seventeen, (64,64))
nineteen = cv2.imread('Good_Images/19.jpg')
nineteen = cv2.resize(nineteen, (64,64))
twenty1 = cv2.imread('Good_Images/21.jpg')
twenty1 = cv2.resize(twenty1, (64,64))
stop=cv2.cvtColor(twenty1, cv2.COLOR_BGR2GRAY)
stop = cv2.Canny(stop,100,200)
_,stop_contours,_= cv2.findContours(stop,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
stop_contours=stop_contours[2]


thirty5 = cv2.imread('Good_Images/35.jpg')
thirty5 = cv2.resize(thirty5, (64,64))
thirty8 = cv2.imread('Good_Images/38.jpg')
thirty8 = cv2.resize(thirty8, (64,64))
fourty5 = cv2.imread('Good_Images/45.jpg')
fourty5 = cv2.resize(fourty5, (64,64))

font=cv2.FONT_HERSHEY_SIMPLEX

def getBlue(roi_label):
	global thirty5,thirty8,fourty5

	if roi_label == 35.0:
		#x_s,y_s = roi.shape[1], roi.shape[0]
		#thirty5 = cv2.resize(thirty5, (x_s,y_s))
		return thirty5
	if roi_label == 38.0:
		#x_s,y_s = roi.shape[1], roi.shape[0]
		#thirty8 = cv2.resize(thirty8, (x_s,y_s))
		return thirty8
	if roi_label == 45.0:
		#x_s,y_s = roi.shape[1], roi.shape[0]
		#fourty5 = cv2.resize(fourty5, (x_s,y_s))
		return fourty5

def getRed(roi_label):
	global one,fourteen,seventeen,nineteen,twenty1

	if roi_label == 1.0:
		#x_s,y_s = roi.shape[1], roi.shape[0]
		#thirty5 = cv2.resize(thirty5, (x_s,y_s))
		return one
	if roi_label == 14.0:
		#x_s,y_s = roi.shape[1], roi.shape[0]
		#thirty8 = cv2.resize(thirty8, (x_s,y_s))
		return fourteen
	if roi_label == 17.0:
		#x_s,y_s = roi.shape[1], roi.shape[0]
		#fourty5 = cv2.resize(fourty5, (x_s,y_s))
		return seventeen
	if roi_label == 19.0:
		#x_s,y_s = roi.shape[1], roi.shape[0]
		#fourty5 = cv2.resize(fourty5, (x_s,y_s))
		return nineteen
	if roi_label == 21.0:
		#x_s,y_s = roi.shape[1], roi.shape[0]
		#fourty5 = cv2.resize(fourty5, (x_s,y_s))
		return twenty1

def svm_predict(image, svm):

    img = cv2.resize(image, (64,64))
    winSize = (64,64)
    blockSize = (16,16)
    blockStride = (8,8)
    cellSize = (4,4)
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
    return testResponse
    
svm = cv2.ml.SVM_load('SVM_PROJECT5.xml')

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

def process_image(frame):

	median_r = cv2.GaussianBlur(frame[:,:,2],(5,5),0)
	median_g = cv2.GaussianBlur(frame[:,:,1],(5,5),0)
	median_b = cv2.GaussianBlur(frame[:,:,0],(5,5),0)

	norm_r=imadjust(median_r)    
	norm_g=imadjust(median_g)    
	norm_b=imadjust(median_b)    
	
	
	total_norm=norm_r+norm_g+norm_b
	
	id_red=np.maximum(0,(np.minimum((norm_r-norm_b),(norm_r-norm_g))))
	#id_red=np.maximum(0,(np.minimum((norm_r-norm_b),(norm_r-norm_g)))/total_norm)
	
	id_blue=np.maximum(0,(np.minimum((norm_b-norm_r),(norm_b-norm_g))))
	#id_blue=np.maximum(0,(np.minimum((norm_b-norm_r),(norm_b-norm_g)))/total_norm)

	
	return id_red, id_blue

	
def do_MSER(frame,stop_contours):

	id_red,id_blue=process_image(frame)
	red_frame=frame[:,:,2]
	blue_frame=frame[:,:,0]
	height=np.shape(id_red)[0]
	width=np.shape(id_red)[1]

	new_red=cv2.rectangle(id_red,(np.float32(0),np.float32((height/2))),(np.float32(width),np.float32(height)),(0,0,0),-1)
	new_red=cv2.rectangle(new_red,(np.float32(0),np.float32((0))),(np.float32(width/2),np.float32(250)),(0,0,0),-1)
	new_blue=cv2.rectangle(id_blue,(np.float32(0),np.float32(height/2)),(np.float32(width),np.float32(height)),(0,0,0),-1)

	new_red = new_red.astype(np.uint8)
	new_blue = new_blue.astype(np.uint8)
	
	mser_blue = cv2.MSER_create(5,300,9000,0.25,0.2,200,1.01,0.003,5)
	mser_red = cv2.MSER_create(7,400,7000,0.25,0.2,200,1.01,0.003,5)
	red_region,red_box = mser_red.detectRegions(new_red)
	blue_region,blue_box = mser_blue.detectRegions(new_blue)
	
	red_hull = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in red_region]
	blue_hull = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in blue_region]
	
	mask = np.zeros((frame.shape[0], frame.shape[1], 1), dtype=np.uint8)
	mask_new = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
	
	roi_blue=[]
	for contour in blue_hull:
		x_blue,y_blue,w,h = cv2.boundingRect(contour)
		aspect_ratio = float(w)/h
		if aspect_ratio<1:
			blue_binary=cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
			roi_blue = frame[y_blue:y_blue+h, x_blue:x_blue+w] 
	
	roi_red=[]
	for contour in red_hull:
		area=cv2.contourArea(contour)
		x_red,y_red,w,h = cv2.boundingRect(contour)
		aspect_ratio = float(w)/h
		norm_w=w/(w+h)
		norm_h=h/(w+h)
		compare = cv2.matchShapes(contour,stop_contours,1,0.0)
		if aspect_ratio>1 and abs(norm_w-norm_h)<0.3 and area>1000 and area<2000 and cv2.isContourConvex(contour) and compare<0.1:
			red_binary=cv2.drawContours(mask_new, [contour], -1, (255, 255, 255), -1)
			roi_red = frame[y_red:y_red+h, x_red:x_red+w] 

	    	

	return mask,mask_new,roi_blue,roi_red

frame_width=1628
frame_height=1236
#out = cv2.VideoWriter('output_final.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

input_path = glob.glob("input/*.jpg")
N=len(input_path)
for i in range(2275,N):
    frame = cv2.imread(input_path[i])

    mask,mask_new,roi_blue,roi_red=do_MSER(frame,stop_contours)
    
    if roi_blue!=[]:
    	testResponse = svm_predict(roi_blue, svm)  
    	if testResponse==45.0 or testResponse==38.0 or testResponse==35.0:
    		_,cnt,_=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    		cv2.drawContours(frame, cnt, -1, (255,0,0), 3)
    		#epsilon = cv2.arcLength(cnt[0],True)
    		#approx = cv2.approxPolyDP(cnt[0],epsilon,True)
    		x_image,y_image,w,h = cv2.boundingRect(cnt[0])
    		image_blue=getBlue(testResponse)
    		if x_image>64:
    			frame[y_image:y_image+64, x_image-64:x_image] = image_blue
    		

    if roi_red!=[]:
    	testResponse = svm_predict(roi_red, svm)
    	if testResponse==21.0 or testResponse==17.0 or testResponse==1.0 or testResponse==14.0 or testResponse==19.0:
    		_,cnt,_=cv2.findContours(mask_new,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    		cv2.drawContours(frame, cnt, -1, (0,0,255), 3)

    		x_image_red,y_image_red,w,h = cv2.boundingRect(cnt[0])
    		image_red=getRed(testResponse)
    		#if x_image_red>64:
    		frame[y_image_red:y_image_red+64, x_image_red-64:x_image_red] = image_red
    		

    #out.write(frame)
    #mask=cv2.resize(mask, (0,0), fx=0.5, fy=0.5)
    #cv2.imshow('mask_new',mask)
    frame=cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
    cv2.imshow('Frame', frame)
    print(i)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break