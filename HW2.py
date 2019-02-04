from __future__ import division
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import img_as_ubyte

'''
For each question, uncomment it and comment other questions' code
Because image display with cv2 cannot stop
'''


####print '*************** Problem 1 ***************'
####img = cv2.imread('2_1.bmp')
####
####val = np.arange(256)
####count = np.zeros(256)
####for p_val in range(256):  # 0-255
####    print p_val
####    for i in range(img.shape[0]):
		count[p_val] = np.sum((img==p_val))
####        for j in range(img.shape[1]):
####            if (img[i,j][0] == p_val):
####                count[p_val] = count[p_val] + 1
####                
####f1 = plt.figure()
####ax1 = f1.add_subplot(111)
####ax1.bar(val, count)
####plt.title('Histogram for gray scale picture')
####plt.xlabel('Pixel Value')
####plt.ylabel('Counts')
####plt.grid(True)
####plt.savefig('Histogram.png')
####
##### CDF
####count = count.astype(float)
####def cdf(data):
####    pdf = np.zeros(data.shape)
####    cdf = np.zeros(data.shape)
####    ttl = 0
####    n = len(data)
####    x = np.arange(n)
####    
####    for i in range(n):
####        pdf[i] = data[i]/(img.shape[0]*img.shape[1])
####    for j in range(n):
####        ttl = ttl + pdf[j]
####        cdf[j] = ttl
####    return x, cdf
####
####x_data, y_data = cdf(count)
####
####f2 = plt.figure()
####ax2 = f2.add_subplot(111)
####plt.title('CDF of picture')
####ax2.plot(x_data, y_data, marker= '.', linestyle= 'none')
####plt.ylabel("CDF")
####
####plt.grid(True)
####plt.savefig('cdf.png')
####plt.show()


####print '*************** Problem 2 ***************'
##### part(a)
####img = cv2.imread('2_2.bmp')
####b, g, r = cv2.split(img)
####
##### part(b)
####hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
####h, s, v = cv2.split(hsv_img)
####
####cv2.imwrite('Blue.png', b)
####cv2.imwrite('Green.png', g)
####cv2.imwrite('Red.png', r)
####cv2.imshow('blue', b)
####cv2.imshow('green', g)
####cv2.imshow('red', r)
####
####cv2.imwrite('Hue.png', h)
####cv2.imwrite('Saturation.png', s)
####cv2.imwrite('Value.png', v)
####cv2.imshow('hue', h)
####cv2.imshow('saturation', s)
####cv2.imshow('value', v)
####
####cv2.waitKey(0)
####cv2.destroyAllWindows()
####
print '*************** Problem 3 ***************'
img = cv2.imread('books.tif')

####for i in range(img.shape[0]):
####    for j in range(img.shape[1]):
####        # blue chanel      rggb
####        if (i%2 == 0 and j%2 == 0):
####            img[i,j][1] = img[i,j+1][1]
####            img[i,j][2] = img[i+1,j+1][2]
####
####        # red chanel
####        elif (i%2 == 1 and j%2 == 1):
####            img[i,j][0] = img[i-1,j-1][0]
####            img[i,j][1] = img[i,j-1][1]            
####
####        # green channel
####        elif (i%2 == 1 and j%2 == 0):
####            img[i,j][0] = img[i-1,j][0]
####            img[i,j][2] = img[i,j+1][2]
####
####        # green channel
####        else:
####            img[i,j][0] = img[i,j-1][0]
####            img[i,j][2] = img[i+1,j][2]
####
####cv2.imwrite('Demosaicing.png', img)
####cv2.imshow('Demosaicing', img)
####cv2.waitKey(0)
####cv2.destroyAllWindows()
####
























