# Traffic-Sign-Recognition

## **PROJECT DESCRIPTION**

The aim of this project is to do Traffic Sign Recognition. This is a two step process of detection and classification.
The Traffic Sign Dataset used in this project is by courtesy of Radu Timofte from ETH-Zurich Vision Lab

Please refer to [Project Report](https://github.com/adheeshc/Traffic-Sign-Recognition/blob/master/Report/Report.pdf) for further description

### Detection Stage

In the Detection stage we aim to extract possible regions which contain a traffic sign. 

<p align="center">
  <img src="/Images/input.png" alt="pre">
</p>

First, I have added a black mask in the bottom half as a traffic sign detected will only be near the top half the image and not on the ground.

There are two approaches I have attempted to detect the traffic sign - Thresholding in HSV Color Space and Using MSER Algorithm

#### Thresholding in HSV Color Space

Here the idea is, that any traffic sign will be of a typical color composition. The HSV Color Space serves better to identify appropriate bands for H, S, V channels to model the color composition of a traffic sign. Also it isolates intensity/brightness (unlike RGB) which helps with robustness to illumination.

<p align="center">
  <img src="/Images/blue_hsv.png" alt="blue_hsv">
</p>

<p align="center">
  <img src="/Images/red_hsv.png" alt="red_hsv">
</p>

- Noise Removal is done
- Model is thresholded in the HSV color space to extract blobs for traffic signs
- Properties of each blob (e.g., size, aspect-ratio) are analyzed to determine if it corresponds to a traffic sign.
- The bounding box is extracted 

This algorithm howeer involves a lot of tweaking and I was not able to get a good detection for Red signs. Hence, I tried the next approach. 

#### MSER Algorithm

A trivial intuition is that MSER gives regions of similar intensity given a grayscale image, and we know that a traffic sign is mostly a uniform intensity region, be it red or blue.


<p align="center">
  <img src="/Images/video.gif" alt="blue_hsv">
</p>

- Noise Removal is done
- Contrast normalization over each channel is performed
- An appropriate grayscale image that best highlights the sign that you want to detect is created. The objective is that for example for the red sign the generated grayscale image should be brighter (towards white) in the red region and darker elsewhere.
- the intensity of the image is normalized 
- MSER region is extracted from the image.
- The bounding box is fit on the MSER region

Please look at the references for more information about MSER algorithm

### Traffic Sign Classification

In the Classification stage, we go over each Region of Interest (ROI) extracted previously and try to identify a traffic sign

<p align="center">
  <img src="/Images/images.png" alt="images">
</p>

- I resize the images to a standard 64x64 extract HOG features
- A multi-class SVM is trained for various signs
- The performance is then tested using the test data to check model created.

The ROI previously extracted in the detection stage is then used as an input to the model and if there is a match, an image (from Good_Image folder) is displayed next to the detection. 

## **DEPENDANCIES**

- Python 3
- OpenCV
- Numpy
- Matplotlib
- Glob (built-in)
- Copy (built-in)
- Bisect (built-in)

## **FILE DESCRIPTION**

- Code Folder/[iniital.py](https://github.com/adheeshc/Traffic-Sign-Recognition/blob/master/Code/iniital.py) - The code using MSER algorithm
- Code Folder/[project6.py](https://github.com/adheeshc/Traffic-Sign-Recognition/blob/master/Code/project6.py) - The code using HSV Color Space
- Code Folder/[classifierSVD.py](https://github.com/adheeshc/Traffic-Sign-Recognition/blob/master/Code/classifierSVD.py) - used for training the SVD
- Code Folder/[svm_function.py](https://github.com/adheeshc/Traffic-Sign-Recognition/blob/master/Code/svm_function.py) - used for testing the SVD

- Dataset folder - Contains Good_Images folder containing all good images to be displayed next to detected signs, link to Input images folder, Training Images and Testing Images. Also has link to 2 xml files that have trained classification model

- Images folder - Contains images for github use (can be ignored)

- Output folder - Contains output video
  
- References folder - Contains supplementary documents that aid in understanding

- Report folder - Contains [Project Report](https://github.com/adheeshc/Traffic-Sign-Recognition/blob/master/Report/Report.pdf)

## **RUN INSTRUCTIONS**

- Make sure all dependancies are met
- Ensure the location of the input video files are correct in the code you're running
- Comment/Uncomment as reqd

- RUN iniital.py for running code using MSER algorithm
- (OPTIONAL) RUN project6.py for code using HSV color Space
- (OPTIONAL) RUN classifierSVD.py to train a new model
