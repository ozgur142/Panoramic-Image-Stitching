import cv2
import random
import numpy as np


n = 2
img = cv2.imread('img/img{}.jpg'.format(n))
height, width = img.shape[:2]

# divide point
divide_point1 = int((width / 2) + int(width / 2) * random.uniform(0, 0.5))
divide_point2 = int((width / 2) * random.uniform(.5, 1))


left_img = img[:, :divide_point1]
right_img = img[:, divide_point2:]

# Save the left and right images
cv2.imwrite('img/img{}l.jpg'.format(n), left_img)
cv2.imwrite('img/img{}r.jpg'.format(n), right_img)


#**********************************************


img1 = cv2.imread('img/img{}l.jpg'.format(n))
img2 = cv2.imread('img/img{}r.jpg'.format(n))

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# Match keypoints
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key = lambda x:x.distance)

# Draw matches
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:100],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Show result
cv2.imshow('sift', img3)
cv2.waitKey()
cv2.destroyAllWindows()
