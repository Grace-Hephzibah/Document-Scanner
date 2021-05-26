import cv2

import  ScannerFunctions as sf

widthImg, heightImg = 480, 640

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(10, 150)

while True:
    success, img = cap.read()
    img = cv2.resize(img,(widthImg,heightImg))
    imgContour = img.copy()
    imgThres = sf.preProcessing(img)
    biggest = sf.getContours(imgThres, imgContour)

    if biggest.size !=0:
        imgWarped = sf.getWarp(img,biggest)
        imageArray = ([img,imgThres], [imgContour,imgWarped])
        cv2.imshow("ImageWarped", imgWarped)
    else:
        imageArray = ([img, imgThres], [img, img])

    stackedImages = sf.stackImages(0.4,imageArray)
    cv2.imshow("WorkFlow", stackedImages)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break