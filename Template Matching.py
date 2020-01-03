#Template Matching
'''Basic Version of Object Detection.
The idea here is to find identical regions of an image that 
match a template we provide, giving a certain threshold.'''


import cv2
import numpy as np

img = cv2.imread('input/picture.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

template = cv2.imread('input/template.jpg', 0)

w, h = template.shape[::-1]

res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.5 #Vary using trial & error
loc = np.where(res >= threshold)

for pt in zip(*loc[::-1]):
	cv2.rectangle(img, pt, (pt[0]+w, pt[1]+h), (0, 255, 255), 1)
	
cv2.imshow('Detected', img)
cv2.imwrite('output/detected.jpg', img)
cv2.waitKey(0)
cv2.destroyAllWindows()



'''  methods: 
 
cv2.TM_CCOEFF
cv2.TM_CCOEFF_NORMED
cv2.TM_CCORR
cv2.TM_CCORR_NORMED
cv2.TM_SQDIFF
cv2.TM_SQDIFF_NORMED

'''
