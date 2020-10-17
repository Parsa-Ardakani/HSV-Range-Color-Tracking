import numpy as np
from cv2 import cv2
from matplotlib import pyplot as plt

#--Capture Camera
cap = cv2.VideoCapture(0)

#--Font of Text
font = cv2.FONT_HERSHEY_SIMPLEX

#--Pass the bar (it is for trackbar)
def nothing (x):
    pass

#--Creating a Blank Window for Finding HSV of BGR and Vice Versa
window_bgr_hsv = np.zeros((100,512,3), np.uint8)
cv2.namedWindow('HSV BGR')

#--Creating a Blank Window for Changing HSV Range
window_hsv_rang = np.zeros((1,512,3), np.uint8)
cv2.namedWindow('HSV Range')


#--Trackbars for Finding HSV of BGR and Vice Versa
cv2.createTrackbar('B', 'HSV BGR', 0, 255, nothing)
cv2.createTrackbar('G', 'HSV BGR', 0, 255, nothing)
cv2.createTrackbar('R', 'HSV BGR', 0, 255, nothing)
cv2.createTrackbar('H', 'HSV BGR', 0, 255, nothing)
cv2.createTrackbar('S', 'HSV BGR', 0, 255, nothing)
cv2.createTrackbar('V', 'HSV BGR', 0, 255, nothing)

#--Switch between BGR and HSV
switch = '0 : BGR\n1 : HSV'
cv2.createTrackbar(switch, 'HSV BGR',0,1,nothing)

#--Trackbars for Changing HSV Range
cv2.createTrackbar('H Lower', 'HSV Range', 0, 255, nothing)
cv2.createTrackbar('S Lower', 'HSV Range', 0, 255, nothing)
cv2.createTrackbar('V Lower', 'HSV Range', 0, 255, nothing)
cv2.createTrackbar('H Upper', 'HSV Range', 0, 255, nothing)
cv2.createTrackbar('S Upper', 'HSV Range', 0, 255, nothing)
cv2.createTrackbar('V Upper', 'HSV Range', 0, 255, nothing)




#--Strting the Program
while True:
    
    # Save each frame of video and store it in fram.
    # Then convert frame BGR color type to HSV
    # Note: We need to use '_,' behind of frame because we are convert it to HSV
    _,frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define lower and upper element as a numpy array
    lower_element = np.array([0,0,0])
    upper_element = np.array([0,0,0])

    #-----------------Read down blew text
    """
    Track The Current Position of All Trackbars
    and Switch.
    """
    r = cv2.getTrackbarPos('R', 'HSV BGR')
    g = cv2.getTrackbarPos('G', 'HSV BGR')
    b = cv2.getTrackbarPos('B', 'HSV BGR')
    h = cv2.getTrackbarPos('H', 'HSV BGR')
    s = cv2.getTrackbarPos('S', 'HSV BGR')
    v = cv2.getTrackbarPos('V', 'HSV BGR')
    hsv_bgr_switch = cv2.getTrackbarPos(switch, 'HSV BGR')
    h_lower = cv2.getTrackbarPos('H Lower', 'HSV Range')
    s_lower = cv2.getTrackbarPos('S Lower', 'HSV Range')
    v_lower = cv2.getTrackbarPos('V Lower', 'HSV Range')
    h_upper = cv2.getTrackbarPos('H Upper', 'HSV Range')
    s_upper = cv2.getTrackbarPos('S Upper', 'HSV Range')
    v_upper = cv2.getTrackbarPos('V Upper', 'HSV Range')


    #Convert BGR to HSV
    bgr_code = np.uint8([[[b,g,r]]])
    bgr_hsv = cv2.cvtColor(bgr_code, cv2.COLOR_BGR2HSV)

    #Convert HSV to BGR
    hsv_code = np.uint8([[[h,s,v]]])
    hsv_bgr = cv2.cvtColor(hsv_code, cv2.COLOR_HSV2BGR)
    


    #--/Range of HSV: Upper is maximum and Lower is minimum
    lower_element = np.array([h_lower,s_lower,v_lower])
    upper_element = np.array([h_upper,s_upper,v_upper])
    

    #--Creating Mask
    mask = cv2.inRange(hsv, lower_element, upper_element)

    # Bitwise 'AND' the mask and original image and store it in 'res'(Result)
    # For more inforamtion search 'Bitwise Operations'
    res = cv2.bitwise_and(frame,frame, mask=mask)

    # Change scale of all frame, mask , and res to 640*480
    # Note: changing the scale and resolution are different.
    # Resolution changes the quality but scale changes the window size 
    frame = cv2.resize(frame, (640,480), interpolation = cv2.INTER_AREA)
    mask = cv2.resize(mask, (640,480), interpolation = cv2.INTER_AREA)
    res = cv2.resize(res, (640,480), interpolation = cv2.INTER_AREA)

    # Show All Windows
    cv2.imshow('frame',frame)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)
    cv2.imshow('HSV BGR', window_bgr_hsv)
    cv2.imshow('HSV Range', window_hsv_rang)

    # Using if for show BGR code or HSV
    if hsv_bgr_switch == 0:
        window_bgr_hsv[:] = [b,g,r]
        cv2.putText(window_bgr_hsv,str(bgr_hsv),(0,50), font, 1,(0,0,0),2,cv2.LINE_AA)
    elif hsv_bgr_switch == 1:
        window_bgr_hsv[:] = [hsv_bgr[0,0,0],hsv_bgr[0,0,1],hsv_bgr[0,0,2]]
        cv2.putText(window_bgr_hsv,str(hsv_bgr),(0,50), font, 1,(0,0,0),2,cv2.LINE_AA)
    
    # Close all windows when 'Esc' button is pressed
    k= cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()

