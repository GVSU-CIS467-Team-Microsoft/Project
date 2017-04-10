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

from skimage import measure, morphology, img_as_ubyte
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

MIN_BOUND = -700.0
MAX_BOUND = 400.0
PIXEL_MEAN = 0.25
avgSpacing=0

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

    print("shape of one slice: ")

    print(slices[0].pixel_array.shape)
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
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
    #image[image < MIN_BOUND] = MIN_BOUND
    #image[image > MAX_BOUND] = MAX_BOUND
    
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
    if spacing[0]<0.2:
        spacing[0]=1.25

    global avgSpacing
    avgSpacing=spacing[0]

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
    #image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    
    return image, new_spacing

def weightedAverage(pixel):
    return 0.299*pixel[0] + 0.587*pixel[1] + 0.114*pixel[2]

def getGrey(image):
    grey = np.zeros((image.shape[0], image.shape[1])) # init 2D numpy array
    # get row number
    for rownum in range(len(image)):
       for colnum in range(len(image[rownum])):
          grey[rownum][colnum] = weightedAverage(image[rownum][colnum])

    return grey

def plot_3d(outFilename, image, threshold=-300):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    
    verts, faces, normals, values = measure.marching_cubes(p, threshold)

    fig = plot.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh2 = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh2.set_facecolor(face_color)
    ax.add_collection3d(mesh2)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    # Create the mesh
    cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            cube.vectors[i][j] = verts[f[j],:]

    # Write the mesh to file "cube.stl"
    outFilename="/home/ron-patrick/Documents/Capstone/stage1Preproc/"+outFilename
    cube.save(outFilename+'.stl')

    #plot.show()
    # fig = plot.figure();
    # fig.savefig(filename);

def make_mesh(image, threshold=-300, step_size=1):

    #print("Transposing surface")
    p = image.transpose(2,1,0)
    
    #print("Calculating surface")
    verts, faces, norm, val = measure.marching_cubes(p, threshold, step_size=step_size, allow_degenerate=True) 
    return verts, faces

def plotly_3d(verts, faces):
    x,y,z = zip(*verts) 
    
    print("Drawing")
    
    # Make the colormap single color since the axes are positional not intensity. 
#    colormap=['rgb(255,105,180)','rgb(255,255,51)','rgb(0,191,255)']
    colormap=['rgb(236, 236, 212)','rgb(236, 236, 212)']
    
    fig = FF.create_trisurf(x=x,
                        y=y, 
                        z=z, 
                        plot_edges=False,
                        colormap=colormap,
                        simplices=faces,
                        backgroundcolor='rgb(64, 64, 64)',
                        title="Interactive Visualization")
    iplot(fig)

def plt_3d(verts, faces):
    #print("Drawing")
    x,y,z = zip(*verts) 
    fig = plot.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], linewidths=0.05, alpha=1)
    face_color = [1, 1, 0.9]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, max(x))
    ax.set_ylim(0, max(y))
    ax.set_zlim(0, max(z))
    ax.set_axis_bgcolor((0.7, 0.7, 0.7))
    plot.show()    

def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

#Standardize the pixel values
def make_lungmask(img, display=False):
    #display=True
    row_size= img.shape[0]
    col_size = img.shape[1]
    
    mean = np.mean(img)
    std = np.std(img)
    img = img-mean
    img = img/std
    # Find the average pixel value near the lungs
    # to renormalize washed out images
    middle = img[int(col_size/5):int(col_size/5*4),int(row_size/5):int(row_size/5*4)] 
    mean = np.mean(middle)  
    max = np.max(img)
    min = np.min(img)
    # To improve threshold finding, I'm moving the 
    # underflow and overflow on the pixel spectrum
    img[img==max]=mean
    img[img==min]=mean
    #
    # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)
    #
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image

    # First erode away the finer elements, then dilate to include some of the pixels surrounding the lung.  
    # We don't want to accidentally clip the lung.

    eroded = morphology.erosion(thresh_img,np.ones([3,3]))
    dilation = morphology.dilation(eroded,np.ones([8,8]))

    labels = measure.label(dilation) # Different labels are displayed in different colors
    label_vals = np.unique(labels)
    regions = measure.regionprops(labels)
    good_labels = []
    for prop in regions:
        B = prop.bbox
        if B[2]-B[0]<row_size/10*9 and B[3]-B[1]<col_size/10*9 and B[0]>row_size/5 and B[2]<col_size/5*4:
            good_labels.append(prop.label)
    mask = np.ndarray([row_size,col_size],dtype=np.int8)
    mask[:] = 0

    #
    #  After just the lungs are left, we do another large dilation
    #  in order to fill in and out the lung mask 
    #
    for N in good_labels:
        mask = mask + np.where(labels==N,1,0)
    mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation
    #mask2=np.ndarray(mask>0)
    #plot.imshow(mask, cmap='gray')
    #plot.show()

    if (display):
        #fig, ax = plot.subplots(1,1, figsize=[12, 12])
        #ax[0, 0].set_title("Original")
        #ax[0, 0].imshow(mask, cmap='gray')
        #ax[0, 0].axis('off')
        #'''
        fig, ax = plot.subplots(3, 2, figsize=[15, 15])
        ax[0, 0].set_title("Original")
        ax[0, 0].imshow(img, cmap='gray')
        ax[0, 0].axis('off')
        ax[0, 1].set_title("Threshold")
        ax[0, 1].imshow(thresh_img, cmap='gray')
        ax[0, 1].axis('off')
        ax[1, 0].set_title("After Erosion and Dilation")
        ax[1, 0].imshow(dilation, cmap='gray')
        ax[1, 0].axis('off')
        ax[1, 1].set_title("Color Labels")
        ax[1, 1].imshow(labels)
        ax[1, 1].axis('off')
        ax[2, 0].set_title("Final Mask")
        ax[2, 0].imshow(mask, cmap='gray')
        ax[2, 0].axis('off')
        ax[2, 1].set_title("Apply Mask on Original")
        ax[2, 1].imshow(mask*img, cmap='gray')
        ax[2, 1].axis('off')
        #'''
        
        plot.show()
    return mask*img
    #return mask

def segment_lung_mask(image, name, fill_lung_structures=True):
    
    # not actually binary, but 1 and 2. 
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -500, dtype=np.int8)+1

    labels = measure.label(binary_image)
    
    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air 
    #   around the person in half
    background_label = labels[0,0,0]
    
    #Fill the air around the person
    binary_image[background_label == labels] = 2
    
    
    # Method of filling the lung structures (that is superior to something like 
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
            
            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    
    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
    
    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0
        
    return binary_image

def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

def zero_center(image):
    image = image - PIXEL_MEAN
    return image

def resize_to_dimensions(array,x,y,z):
    oldx = array.shape[0] 
    oldy = array.shape[1]
    oldz = array.shape[2]
    xRatio = x / oldx
    yRatio = y / oldy
    zRatio = z / oldz
    return scipy.ndimage.zoom(array, (xRatio,yRatio,zRatio));

scipy.seterr(all="ignore")

INPUT_FOLDER = '/home/gvuser/stage1/'
whichPatient=int(sys.argv[1])
patients = os.listdir(INPUT_FOLDER)
patients.sort()
name=INPUT_FOLDER+patients[whichPatient]
patient = load_scan(name,patients[whichPatient])
patient_pixels = get_pixels_hu(patient)

imgs_after_resamp, spacing = resample(patient_pixels, patient, [1,1,1])
imMax=imgs_after_resamp
imMax[imMax > MAX_BOUND] = -10000
imMax[imMax == 0] = -10000
maxFloat=imMax.max()
imMin=imgs_after_resamp
imMin[imMin < MIN_BOUND] = 10000
imMin[imMin == 0] = 10000
minFloat=imMin.min()
#imgs_after_resamp[imgs_after_resamp < MIN_BOUND] = 0#MIN_BOUND
#imgs_after_resamp[imgs_after_resamp > MAX_BOUND] = 0#MAX_BOUND
#maxFloat=imgs_after_resamp.max()
#minFloat=imgs_after_resamp.min()

print("output: maxDim={0} maxFloat={1} minFloat={2} avgSpacing={3} :end".format(max(imgs_after_resamp.shape),maxFloat,minFloat,avgSpacing), end='')
