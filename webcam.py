import cv2
import numpy as np
from kernals import *


cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    with_kernel_x = cv2.filter2D(gray, -1, sobelX)
    with_kernel_y = cv2.filter2D(gray, -1, sobelY)

    # Display the resulting frame
    cv2.imshow('B&W', with_kernel_x+with_kernel_y)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
