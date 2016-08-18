# Gradient computation for grayscale/color Canny edge detection
# author: Muhammet Bastan, mubastan@gmail.com
# date: March 2012

import numpy, scipy, scipy.ndimage
from scipy.ndimage import filters
from numpy import *

# Sobel or Prewit gradient
def gradient(image, type=0):
    if type==0:
        Ix = filters.sobel(image, axis=1)
        Iy = filters.sobel(image, axis=0)
        return Ix, Iy
    else:
        Ix = filters.prewitt(image, axis=1)
        Iy = filters.prewitt(image, axis=0)
    return Ix, Iy

# gradient magnitudes and directions of a grayscale image, Gd: [-pi/2, +pi/2]
def gray_gradient(image):
    Gx, Gy = gradient(image)
    Gm = sqrt(Gx**2+Gy**2)
    Gd = arctan2(Gy, Gx)    
    Gd[Gd > 0.5*numpy.pi] -= numpy.pi
    Gd[Gd < -0.5*numpy.pi] += numpy.pi
    return Gm, Gd

# using the same approach of color gradient mag-ori computation for a gray image
def gray_gradient_tensor(image, sigma=0.5):
        
    g = image.astype('float32')
    #g = scipy.ndimage.filters.gaussian_filter(g, sigma)
    gx, gy = gradient(g)
    
    cxx = gx*gx
    cyy = gy*gy 
    cxy = 2*(gx*gy)     ## 2*cxy
    
    ## todo: smooth the color tensor
    cxx = scipy.ndimage.filters.gaussian_filter(cxx, sigma)
    cyy = scipy.ndimage.filters.gaussian_filter(cyy, sigma)
    cxy = scipy.ndimage.filters.gaussian_filter(cxy, sigma)
    
    cxx_cyy = cxx-cyy
    eps = 0.0000001
    d = sqrt( cxx_cyy**2 + cxy**2 + eps)
    
    ## largest eigenvalue -- derivative energy in the most prominent direction
    lambda1 = cxx + cyy + d          ## 0.5*... ?
    
    Gm = sqrt(lambda1 + eps)
    Gd = 0.5*arctan2(cxy, cxx_cyy)
    
    return Gm, Gd

# color gradient from multiple single channel images
# imgs: contains a list of single channel images (pixel values must be in the same range)
# output: Gm, gradient magnitude in the most prominent gradient direction (largest eigenvalue of color tensor)
# Gd: gradient directions [-pi/2, +pi/2]
# Reference: Van De Weijer, Joost and Gevers, Theo and Smeulders, Arnold WM, 'Robust Photometric Invariant Features from the Color Tensor', TIP 2006.
# This Py code was adapted from the Matlab code of the paper
def multi_gradient(imgs, gtype=0, sigma=0.5):
    if gtype==1: return multi_gradient_max(imgs)

    N=len(imgs)    
    # compute gradients
    # first channel    
    gx, gy = gradient(imgs[0])
    # structure tensor (color tensor in RGB images)
    cxx = gx*gx; cyy = gy*gy; cxy = gx*gy
    # remaining channels    
    for i in range(1, N):        
        gx, gy = gradient(imgs[i])
        cxx += gx*gx; cyy += gy*gy; cxy += gx*gy        
    cxy *= 2    ## 2*cxy
    
    ## smooth the structure/color tensor
    cxx = scipy.ndimage.filters.gaussian_filter(cxx, sigma)
    cyy = scipy.ndimage.filters.gaussian_filter(cyy, sigma)
    cxy = scipy.ndimage.filters.gaussian_filter(cxy, sigma)
    
    cxx_cyy = cxx-cyy
    eps = 1e-9
    d = sqrt( cxx_cyy**2 + cxy**2 + eps)
        
    ## largest eigenvalue -- derivative energy in the most prominent direction
    lambda1 = cxx + cyy + d          ## 0.5*... ?
        
    Gm = sqrt(lambda1 + eps)
    Gd = 0.5*arctan2(cxy, cxx_cyy)
    
    return Gm, Gd

# maximum gradient for each pixel in all the channels
def multi_gradient_max(imgs):
    N=len(imgs)      
    Gm,Gd = gray_gradient(imgs[0])
    for i in range(1, N):
        Gm2, Gd2 = gray_gradient(imgs[i])        
        ind= Gm2>Gm
        Gm[ind] = Gm2[ind]       
        Gd[ind] = Gd2[ind]
    return Gm, Gd

# gradient magnitude from an RGB image
# type: 0 (color tensor), else (max)
def rgb_gradient(image, gtype=0, sigma=0.5):
    r = image[:,:,0].astype('float32')
    g = image[:,:,1].astype('float32')
    b = image[:,:,2].astype('float32')
    
    imgs=[]
    imgs.append(r)
    imgs.append(g)
    imgs.append(b)
    
    if gtype == 0:
        return multi_gradient(imgs, sigma)
    else:
        return multi_gradient_max(imgs)
