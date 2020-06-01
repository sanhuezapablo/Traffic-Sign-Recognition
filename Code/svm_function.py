import numpy as np
import cv2
import glob

def svm_predict(image, svm):

    img = cv2.resize(image, (64,64))
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
    return testResponse
    
svm = cv2.ml.SVM_load('SVM_TEST.xml')
testingImgs = glob.glob('Testing/*/*.ppm')
test_labels = []
results = []

count1=0
count2=0

for i in testingImgs:
    
    img = cv2.imread(i)
    testResponse = svm_predict(img, svm)
    results.append(testResponse)
    test_labels.append(int(i.split("\\")[1]))
    print('Original Label: ', int(i.split("\\")[1]))
    print('SVM Prediction: ', testResponse)
    print('==========================')
    if (int(i.split("\\")[1])==int(testResponse[0])):
        count1+=1
    else:
        count2+=1
print(count1/(count2+count1))