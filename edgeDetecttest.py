from edgeDetect import *

f = Filter('./testscale.jpg')
'''
# Canny edge
im = f.cannyEdge()
f.showImage(im)
'''
# Sobel filter
im, _ = f.sobelFilter()
f.showImage(im)
'''
# Prewitt filter
im, _ = f.prewittFilter()
f.showImage(im)
'''