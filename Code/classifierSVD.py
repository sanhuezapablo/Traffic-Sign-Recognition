import glob
import numpy as np
import cv2

trainingImgs = glob.glob('Training/*/*.ppm')
testingImgs = glob.glob('Testing/*/*.ppm')

count_training = np.size(trainingImgs)

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

trainingData = []
labels = []

C = 1
gamma = 0

for i in trainingImgs:
    
    img = cv2.imread(i)
    img = cv2.resize(img, (64,64))
    trainingData.append(hog.compute(img))
    labels.append(int(i.split("\\")[1]))


labels = np.array(labels).reshape(-1)    
hog_descriptors = np.squeeze(trainingData)

svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setC(C)
svm.setGamma(gamma)
 
svm.train(hog_descriptors, cv2.ml.ROW_SAMPLE, labels)
svm.save("SVM_TEST.xml")
    