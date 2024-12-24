# todo: CONVERT PIXEL INTO CM
import cv2
import numpy as np
cap = cv2.VideoCapture(0)
while(True):
    # frame = cv2.imread('C:\\Users\\DELL\\Desktop\\yash\\python\\cv\\sizeCalc\\newCards.jpg')
    # frame = cv2.resize(frame,(0,0),fx=0.5,fy=0.4)
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # ret, threshold = cv2.threshold(gray,127,255,0) # thresholding for removing bg 2nd arg is min val of threshold and 3rd arg is max val to be given to pixel

    threshold = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((5,5))
    imgDial = cv2.dilate(edges,kernel, iterations = 3) #Expands white regions (foreground) by adding pixels to object boundarie Why it's used: Helps to close small gaps in the thresholded image, such as noise or disconnected parts of the card edges.
    imgThre = cv2.erode(imgDial,kernel, iterations = 2) #Shrinks white regions by removing pixels from object boundaries. Why it's used: Removes small noise regions and smooths object edges.
    contours, hierarchy = cv2.findContours(imgThre,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #second is contour retrieval mode, third is contour approximation method.
    # contours is the py list of all (x,y) cods of outine 
    # cv2.drawContours(frame,contours,-1,(0,255,255),4) # 2nd arg is the list of points , 3rd arg is to specify which point to be drawn -1 means all 

    for cont in contours:
        perimeter = cv2.arcLength(cont,True)

        approx = cv2.approxPolyDP(cont,0.02*perimeter,True) # reduces number of vertices and simplify the polygon 
        # print(approx)
        area = cv2.contourArea(cont,True)
        font = cv2.FONT_HERSHEY_SIMPLEX
        if abs(area)>500:
                # rect = cv2.minAreaRect(cont)  # rect contains (center, (width, height), angle)
                # box = cv2.boxPoints(rect)    # Get the 4 corners of the rectangle
                # box = np.int0(box) 
                # Draw the rectangle on the image
            if len(approx) ==4:
                x,y,w,h = cv2.boundingRect(approx)
                cv2.drawContours(frame, [approx], 0, (0, 255, 0), 2)
                cv2.putText(frame, f"Rectangle: {int(abs(area))}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            elif len(approx) == 3:
                x,y,w,h = cv2.boundingRect(approx)
                cv2.drawContours(frame, [approx], 0, (0, 255, 0), 2)
                cv2.putText(frame, f"Triangle: {int(abs(area))}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            elif len(approx) >20:
                x,y,w,h = cv2.boundingRect(approx)
                cv2.drawContours(frame, [approx], 0, (0, 255, 0), 2)
                cv2.putText(frame, f"Circle: {int(abs(area))}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow('frame',frame)
    if cv2.waitKey(1)==ord('q'):
        break

cv2.destroyAllWindows()

