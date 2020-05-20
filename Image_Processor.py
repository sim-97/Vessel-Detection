import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt

def binaryThresh(im):
    (thresh, im_bw) = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # No more noise, binary image
    return im_bw,thresh

def cleancontours(image):
    cnts = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 1000:
            cv2.drawContours(image, [c], -1, (0, 0, 0), -1)
    return image

def readImage(filename):
    image = cv2.imread(filename)
    image = cv2.resize(image,(224,224))
    return image

def morph(image, kernel):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    erodedimage = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    image = binaryThresh(erodedimage)[0]
    image = cleancontours(image)
    return image

def intensity(matrix):
    return matrix.sum(axis=1)

def plotCircles(image, resultingArray):
    for x in range(0, len(resultingArray)):
        for y in range(0,1):
            cv2.circle(image, (resultingArray[x][y], res[x][y+1]), 11, (0, 0, 255), 2)

#Edit this function for input Paths
def inputParser():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", help = "path to the image file")
    args = vars(ap.parse_args())
    bw = "outputImages/" + args["image"]
    fundus = "outputImages-Fundus/" + args["image"]
    return [bw,fundus]

def slidingWindowTop4(image, stepSize, w_width, w_height):
    square = image
    for x in range(0, 224-16, stepSize):
        for y in range(0, 224-16, stepSize):
            window = image[y:y+w_width, x:x+w_height]
            list.append([x,y,np.sum(window)])
            cv2.rectangle(square, (x,y), (x+w_width, y+w_height), (255, 0, 0), 2) #draw the window on images

    list.sort(key = lambda x:x[2])
    res = list[-4:]
    return res

def bestGuess(sortedArray, medians):
    if sortedArray[3] > 160:
        if 200 < sortedArray[3] < 300 :
            return medians.index(sortedArray[3])
        if sortedArray[3] <200:
            return medians.index(sortedArray[2])
    if sortedArray[3] > 160 and sortedArray[3] > 300 and sortedArray[2]>350:
        return medians.index(sortedArray[1])
    if sortedArray[3] <20:
        return medians.index(sortedArray[3])
    else:
        return medians.index(sortedArray[2])


inputImages = inputParser() #Paths for inputImages

resizedimg = readImage(inputImages[0]) #Read & resize image

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,1))
filteredImage = morph(resizedimg, kernel)

cv2.imwrite('erosionTest.jpg', filteredImage)
erodedimage = cv2.imread("erosionTest.jpg", 1)
list = []
#Returns the top 4 brightest based on white pixel count of a 16X16 window
res = slidingWindowTop4(erodedimage, 16, 16, 16)

image = readImage(inputImages[1])
image_rbg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_rbg[:,:,0]=0
image_rbg[:,:,1]=0

gray = cv2.cvtColor(image_rbg, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3,5), 0)
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
brightness_x, brightness_y = maxLoc

plotCircles(image_rbg,res)

top_4 = gray[res[0][0]:res[0][0]+width, res[0][1]:res[0][1]+height]
top_3 = gray[res[1][0]:res[1][0]+width, res[1][1]:res[1][1]+height]
top_2 = gray[res[2][0]:res[2][0]+width, res[2][1]:res[2][1]+height]
top_1 = gray[res[3][0]:res[3][0]+width, res[3][1]:res[3][1]+height]

intensity_4 = intensity(top_4)
medians = [np.median(intensity_4)]

intensity_3 = intensity(top_3)
medians.append(np.median(intensity_3))

intensity_2 = intensity(top_2)
medians.append(np.median(intensity_2))

intensity_1 = intensity(top_1)
medians.append(np.median(intensity_1))

sorted = medians.copy()
sorted.sort()

ind = bestGuess(sorted, medians)
cv2.rectangle(image, (res[ind][0]-16, res[ind][1]-16), (res[ind][0]+32, res[ind][1]+32), (255, 0, 0), 2) #draw the window on images

f = plt.figure()
f.add_subplot(1,3,1)
plt.imshow(erodedimage)
f.add_subplot(1,3,2)
plt.imshow(cv2.cvtColor(image_rbg, cv2.COLOR_BGR2RGB))
f.add_subplot(1,3,3)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show(block=True)
