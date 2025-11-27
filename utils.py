import cv2
import numpy as np

def preProcess(img):
    # convert to greyscale
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # finds edges using Canny edge detection
    # Gaussian Blur -> Sobel filter -> "thins" edges using gradiant (if not local max in direction of grad then remove)
    # some other stuff that removes more noise, often leaves a very thin and clean edge
    imgCanny = cv2.Canny(imgGray, 50, 150)
    # in case the image quality isn't super good, dilating ensures the outline of the paper is connected/solid
    # and eroding then thins it back down. 
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=2)
    imgThres = cv2.erode(imgDial, kernel, iterations=1)
    return imgThres

#returns the largest white space, treats that as the paper
def findPaper(img):
    biggest = np.array([])
    maxArea = 0
    # gets all the white spaces ("shapes")
    shapes, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for i in shapes:
        area = cv2.contourArea(i)
        # ignore all the small shapes 
        if area > 2000:
            # calculates perimeter, second perimeter is it's an open or closed shape
            peri = cv2.arcLength(i, True)
            # approximates how many points the polygon for the shape would have
            # inputs: shape, epsilon (max distance between original and new shape, essentially degree of accuracy), open/closed shape
            approx = cv2.approxPolyDP(i, 0.05 * peri, True)
            # updates max value if the shape is a rectangle
            if area > maxArea and len(approx) == 4:
                # checks if the shape is has interior angles < 180 degrees, it kept picking up random quadrilaterals
                if cv2.isContourConvex(approx):
                    # check aspect ratio to avoid weird long shapes
                    x, y, w, h = cv2.boundingRect(approx)
                    aspectRatio = float(w) / h
                    # paper is usually between 0.5 and 2.0 (portrait or landscape)
                    if 0.5 < aspectRatio < 2.0:
                        biggest = approx
                        maxArea = area
    return biggest, maxArea

# reorders the points of the paper to be in order: TL, TR, BL, BR. 
def reorder(myPoints):
    # reshapes to a 4x2 array, 4 points with x and y cords each. 
    # opencv arrays r usually 4x1x2, just make it easier to work with 
    myPoints = myPoints.reshape((4, 2))
    # output array should match opencv format
    myPointsNew = np.zeros((4, 1, 2), np.int32)
    # adds up x and y cords of each point
    add = myPoints.sum(1)
    # finds the value of y - x for each point
    diff = np.diff(myPoints, axis=1)
    # algorithm for ordering points: 
    # TL will always have the smallest sum of x + y
    # BR will always have largest sum of x + y
    # TR will have smallest value of y - x (most negative, since small y and big x)
    # BL will have the largest value of y - x (since big y small x)
    
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

# warps the image so that it's "facing" the camera (approximates what it would look like if we were looking at it dead on)
def getWarp(img, biggest, widthImg=28, heightImg=28):
    # corners of the paper can be in any order
    biggest = reorder(biggest)
    # coordinates in the current frame
    pts1 = np.float32(biggest)
    # coordinates in the frame we want ("facing the camera"), with sizes adjusted
    #sizes of 28x28 since that's the size of pictures in the dataset used for training
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    # calculates the transformation matrix to go from first frame to the second (p much just linalg)
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    # applies the transformation matrix on the original image
    imgOutput = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    
    # Crop slightly to remove borders (2 pixels on each side)
    imgCropped = imgOutput[2:imgOutput.shape[0]-2, 2:imgOutput.shape[1]-2]
    imgCropped = cv2.resize(imgCropped, (widthImg, heightImg))
    
    return imgCropped
