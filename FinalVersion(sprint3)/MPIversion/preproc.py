#you may find this useful if you're working on ubuntu/linux mint. Packages should be similar on other linux distros. Otherwise they can be installed with pip
#sudo apt-get update
#sudo apt-get install python3-tk python3-numpy python3-pandas python3-dicom python3-skimage

#you also need these 2 python3-scipy python3-matplotlib, however I ran into a lot of erros with these and found things worked much better when installing them using pip3 (you will need python3-setuptools if you don't have it already)

#if you already had any of these packages installed and something doesn't work, try upgdating the package to the most recent version before troubleshooting further.

#other info

#Pixel spacing
#Section 10.7.1.3: Pixel Spacing
#The first value is the row spacing in mm, that is the spacing between the centers of adjacent rows, or vertical spacing. The second value is the column spacing in mm, that is the spacing between the centers of adjacent columns, or horizontal spacing.

import numpy as np
import pandas as pd
import dicom
import os
import scipy.ndimage
import matplotlib.pyplot as plot
import threading
import sys
from stl import mesh
from sklearn.cluster import KMeans

from skimage import measure, morphology, img_as_ubyte, img_as_float
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

MIN_BOUND = -1000.0
MAX_BOUND = 400.0
PIXEL_MEAN = 0.25

def sample_stack(stack, rows=6, cols=6, start_with=10, show_every=3):
    fig,ax = plot.subplots(rows,cols,figsize=[12,12])
    for i in range(rows*cols):
        ind = start_with + i*show_every
        ax[int(i/rows),int(i % rows)].set_title('slice %d' % ind)
        ax[int(i/rows),int(i % rows)].imshow(stack[ind],cmap='gray')
        ax[int(i/rows),int(i % rows)].axis('off')
    plot.show()

# Load the scans in given folder path
def load_scan(path,name):
    slices = []
    gotSomething=0
    for s in os.listdir(path):
        file=path+'/'+s
        if s!=name:
            slices.append(dicom.read_file(file))
            gotSomething=1

    if gotSomething==0:
        sys.exit()
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
    return slices

def get_pixels_hu2(slices):
    image = np.stack([s.pixel_array for s in slices])

    image = image.astype(np.int16)

    image[image == -2000] = 0
    
    for slice_number in range(len(slices)):
        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
        
    return np.array(image, dtype=np.int16)

def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))
    spacing = np.array(list(spacing))
    if spacing[0]<0.1:
        spacing[0]=1.92692
    #image[image<MIN_BOUND]=0
    #image[image>MAX_BOUND]=0

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    #image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    
    return image, new_spacing

def resize_to_dimensions(array,x,y,z):
    oldx = array.shape[0] 
    oldy = array.shape[1]
    oldz = array.shape[2]
    xRatio = x / oldx
    yRatio = y / oldy
    zRatio = z / oldz
    return scipy.ndimage.zoom(array, (xRatio,yRatio,zRatio));

scipy.seterr(all="ignore")

INPUT_FOLDER=''
OUTPUT_FOLDER=''
whichStage=int(sys.argv[2])
if whichStage==1:
    INPUT_FOLDER = '/home/gvuser/stage1/'
    OUTPUT_FOLDER = '/home/gvuser/MPIversion/KagglePre/'
if whichStage==2:
    INPUT_FOLDER = '/home/gvuser/stage2/'
    OUTPUT_FOLDER = '/home/gvuser/stage2Preprocessed/'

whichPatient=int(sys.argv[1])
patients = os.listdir(INPUT_FOLDER)
patients.sort()
#minFloat=float(sys.argv[2])
#maxFloat=float(sys.argv[3])
#maxDim=int(sys.argv[4])
maxDim=350
name=INPUT_FOLDER+patients[whichPatient]
patient = load_scan(name,patients[whichPatient])
patient_pixels = get_pixels_hu2(patient)
imgs_after_resamp, spacing = resample(patient_pixels, patient, [1,1,1])
seg2=resize_to_dimensions(imgs_after_resamp, maxDim,maxDim,maxDim)
#outputI=np.zeros(shape=seg2.shape)
outputI=[]
lowBound=-500
highBound=300
#index=0
for image in seg2:
    image[image > highBound]=-6000
    binary_image = np.array(image > lowBound, dtype=np.int8)+1
    labels = measure.label(binary_image)
    i=0
    for j in range(labels.shape[1]):
        background_label=labels[i,j]
        binary_image[background_label == labels] = 2
        background_label=labels[j,i]
        binary_image[background_label == labels] = 2
    i=labels.shape[0]-1
    for j in range(labels.shape[1]):
        background_label=labels[i,j]
        binary_image[background_label == labels] = 2
        background_label=labels[j,i]
        binary_image[background_label == labels] = 2
    for i in range(int(labels.shape[0]*0.85),labels.shape[0]):
        for j in range(labels.shape[1]):
            binary_image[i,j]=2
    binary_image -= 1
    binary_image = 1-binary_image
    image=image+1000
    image[image < 0]=0
    image=image*binary_image
    image=image/(-lowBound)
    output=image*255.0;
    output=output+0.5;
    outputI.append(np.uint8(output))
    #outputI[index]=np.uint8(output)
    #index+=1
#import gzip
#f = gzip.GzipFile("{0}{1}.dat".format(OUTPUT_FOLDER,patients[whichPatient]), "w")
#np.save(file=f, arr=outputI)
#f.close()
with open("{0}{1}.dat".format(OUTPUT_FOLDER,patients[whichPatient]), "wb") as f:
    for o in outputI:
        f.write(o)

print("done",end='')

