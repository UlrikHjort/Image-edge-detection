########################################################################
#                                                                           
#                                                                    
#                          Image Edge Detection                                                           
#                           edgeDetect.py                                      
#                                                                           
#                                MAIN                                      
#                                                                           
#                 Copyright (C) 2010 Ulrik Hoerlyk Hjort                   
#                                                                        
#  Image Edge Detection is free software;  you can  redistribute it                          
#  and/or modify it under terms of the  GNU General Public License          
#  as published  by the Free Software  Foundation;  either version 2,       
#  or (at your option) any later version.                                   
#  Image Edge Detection is distributed in the hope that it will be                           
#  useful, but WITHOUT ANY WARRANTY;  without even the  implied warranty    
#  of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                  
#  See the GNU General Public License for  more details.                    
#  You should have  received  a copy of the GNU General                     
#  Public License  distributed with Yolk.  If not, write  to  the  Free     
#  Software Foundation,  51  Franklin  Street,  Fifth  Floor, Boston,       
#  MA 02110 - 1301, USA.                                                    
########################################################################        
from PIL import Image
import numpy as np
from scipy.signal import convolve2d
import math
import matplotlib.pyplot as plt

class Filter :
    prewittKernel = np.mgrid[-1:2, -1:2]

    ########################################################################
    #
    # 
    #
    ########################################################################
    def __init__(self, filename):
        self.im = Image.open(filename)
        self.im_height = self.im.height
        self.im_width = self.im.width
        self.im_buf = np.array(self.im)
        self.imGray = Image.new('L', (self.im_width, self.im_height))
        self.toGray()
        
    ########################################################################
    #
    # 
    #
    ########################################################################        
    def toGray(self):                
        for x in range (self.im_width):
            for y in range (self.im_height):
                r,g,b = self.im.getpixel((x,y))
                gray = int(0.2989 * r + 0.5870 * g + 0.1140 * b)                
                self.imGray.putpixel((x, y), gray)


    ########################################################################
    #
    # 
    #
    ########################################################################
    def gaussianKernel(self, dimension, sigma=1):
        lim = int(dimension) // 2        
        x, y = np.mgrid[-lim:lim+1, -lim:lim+1]
        normal = 1 / (2.0 * np.pi * sigma**2)
        g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
        return g

    ########################################################################
    #
    # 
    #
    ########################################################################
    def gaussianBlur(self, im):
        kernel = self.gaussianKernel(5, 1.4)
        return convolve2d(im, kernel, mode='same', boundary = 'symm', fillvalue=0)

    ########################################################################
    #
    # 
    #
    ########################################################################
    def nonMaximumSupression(self, im, phase):        
        gmax = np.zeros((im.shape), dtype=np.int32)
        phase = phase * 57.3
        phase[phase < 0] += 180

        for i in range(1,gmax.shape[0]-1):
            for j in range(1,gmax.shape[1]-1):
                a = 255
                b = 255
               
                if ((0 <= phase[i,j] < 22.5) or 
                   (157.5 <= phase[i,j] <= 180)):
                    a = im[i, j+1]
                    b = im[i, j-1]               
                elif (22.5 <= phase[i,j] < 67.5):
                    a = im[i+1, j-1]
                    b = im[i-1, j+1]               
                elif (67.5 <= phase[i,j] < 112.5):
                    a = im[i+1, j]
                    b = im[i-1, j]               
                elif (112.5 <= phase[i,j] < 157.5):
                    a = im[i-1, j-1]
                    b = im[i+1, j+1]

                if (im[i,j] >= a) and (im[i,j] >= b):
                    gmax[i,j] = im[i,j]
                else:
                    gmax[i,j] = 0
        return gmax

    ########################################################################
    #
    # 
    #
    ########################################################################
    def doubleThreshold(self, im, weakPixel=75, strongPixel=255, lowThreshold=0.05, highThreshold=0.15):
            highThreshold = im.max() * highThreshold;
            lowThreshold = highThreshold * lowThreshold;            
            imThr = np.zeros((im.shape), dtype=np.int32)
            imThr[np.where(im >= highThreshold)] = np.int32(strongPixel) 
            imThr[np.where((im <= highThreshold) & (im >= lowThreshold))] = np.int32(weakPixel) 

            return imThr

    ########################################################################
    #
    # 
    #
    ########################################################################
    def hysteresis(self, im, weakPixel=75, strongPixel=255):
        for i in range(1, im.shape[0]-1):
            for j in range(1, im.shape[1]-1):
                if (im[i,j] == weakPixel):
                        if ((im[i+1, j-1] == strongPixel) or 
                            (im[i+1, j] == strongPixel) or 
                            (im[i+1, j+1] == strongPixel) or 
                            (im[i, j-1] == strongPixel) or 
                            (im[i, j+1] == strongPixel) or 
                            (im[i-1, j-1] == strongPixel) or 
                            (im[i-1, j] == strongPixel) or 
                            (im[i-1, j+1] == strongPixel)):
                                im[i, j] = strongPixel
                        else:
                            im[i, j] = 0
        return im
    ########################################################################
    #
    # 
    #
    ########################################################################
    def cannyEdge(self, im=None):
        if im is None:
            im=self.imGray

        im = self.gaussianBlur(im)
        im, alpha = self.sobelFilter(im)
        im = self.nonMaximumSupression(im, alpha)
        im = self.doubleThreshold(im)
        im = self.hysteresis(im)
        return im

    ########################################################################
    #
    # 
    #
    ########################################################################
    def averageFilter(self, im=None, kernel = np.array([[1,2,1], [2,4,2], [1,2,1]]) *(1/16)):     
        if im is None:
            im=self.imGray
        return convolve2d(im, kernel, mode='same', boundary = 'symm', fillvalue=0)


    ########################################################################
    #
    # 
    #
    ########################################################################
    def sobelFilter(self, im=None,  kernelX = np.array([[1,0,-1], [2,0,-2], [1,0,-1]]), kernelY = np.array([[1,2,1], [0,0,0], [-1,-2,-1]])):
      if im is None:
        im=self.imGray
      return self.edgeDetect(im, kernelX,kernelY)
    
    ########################################################################
    #
    # 
    #
    ########################################################################    
    def prewittFilter(self, im=None, kernelX = np.array(prewittKernel[1]), kernelY = np.array(prewittKernel[0])):
      if im is None:
        im=self.imGray        
      return self.edgeDetect(im,kernelX,kernelY)

    ########################################################################
    #
    # 
    #
    ########################################################################
    def edgeDetect(self, im=None, kernelX = np.array(prewittKernel[1]), kernelY = np.array(prewittKernel[0])):
      if im is None:
        im=self.imGray

      gradX = convolve2d(im, kernelX, mode='same', boundary = 'symm', fillvalue=0)
      gradY = convolve2d(im, kernelY, mode='same', boundary = 'symm', fillvalue=0)
      grad = np.zeros(shape=gradX.shape)
            
      for x in range (grad.shape[0]):
          for y in range (grad.shape[1]):
              grad[x,y] = int(math.sqrt((int(gradX[x,y]) * int(gradX[x,y])) + (int(gradY[x,y]) * int(gradY[x,y]))))

      
      m = np.max(grad)
      for x in range (grad.shape[0]):
          for y in range (grad.shape[1]):
              grad[x,y] *= 255.0/m 
    
      alpha = np.arctan2(gradY, gradX)
      return (grad, alpha)

    ########################################################################
    #
    # 
    #
    ########################################################################        
    def plot(self):
        img = self.im
        for i in range(1000):
            img.putpixel((i, i), (255, 0, 0))
        img.show()

    ########################################################################
    #
    # 
    #
    ########################################################################        
    def showImage(self, im):
      PILimage = Image.fromarray(im.astype('uint8'), 'L')
      PILimage.show()

        


