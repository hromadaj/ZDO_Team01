import sys
import numpy as np
from scipy import ndimage, signal, misc
import skimage.data, skimage.io, skimage.feature
import matplotlib.pyplot as plt
from skimage.filters import gaussian as gaussian_filter
from skimage.filters import threshold_otsu
from skimage.morphology import label
from skimage.measure import regionprops
from skimage.color import label2rgb
from skimage.io import imread
import skimage.morphology
from skimage.filters import sobel, roberts
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage.draw import line as draw_line
from skimage import data
from matplotlib import cm
from numpy.ma.extras import unique


#if len(sys.argv) < 2:
#    print("Usage: python3 run.py <image_filename>")
#    sys.exit(1)

for i in range(2,len(sys.argv)):
    print(sys.argv[i])
    image_filename = sys.argv[i]
    image = skimage.io.imread(image_filename)
    img= gaussian_filter(image,sigma=1,channel_axis=None)
    
    plt.figure(figsize=[15,10])
    plt.subplot(1,2,1)
    plt.imshow(image)

    plt.subplot(1,2,2)
    plt.imshow(img)
    plt.show()




#image_filename = sys.argv[1]

#image = skimage.io.imread(image_filename)
#img= gaussian_filter(image,sigma=1,channel_axis=None)

#print(len(sys.argv))

