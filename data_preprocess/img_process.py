import os
import cv2



files = os.listdir('train/')
minFirst = 10000000;
minSecond = 10000000;
maxFirst = 0;
maxSecond = 0;
for file in files:
	if file.endswith('png'):
		im = cv2.imread('train/'+file)
		if (im.shape[0]>maxFirst):
			maxFirst = im.shape[0]
		if (im.shape[0]<minFirst):
			minFirst = im.shape[0]
		if (im.shape[1]>maxSecond):
			maxSecond = im.shape[1]
		if (im.shape[1]<minSecond):
			minSecond = im.shape[1]
		print(im.shape)
print("minFirst ",minFirst)
print("minSecond ",minSecond)
print("maxFirst ",maxFirst)
print("maxSecond ",maxSecond)
