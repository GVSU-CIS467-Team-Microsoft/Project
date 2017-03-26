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

from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

INPUT_FOLDER = '../temp_data/'
patients = os.listdir(INPUT_FOLDER)
patients.sort()

MIN_BOUND = -1000.0
MAX_BOUND = 400.0
PIXEL_MEAN = 0.25


# Load the scans in given folder path
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
    return slices

def get_pixels_hu(slices):
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

def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing

def plot_3d(image, threshold=-300):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    
    verts, faces = measure.marching_cubes(p, threshold)

    fig = plot.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plot.show()
    # fig = plot.figure();
    # fig.savefig(filename);

def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def segment_lung_mask(image, fill_lung_structures=True):
    
    # not actually binary, but 1 and 2. 
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8)+1

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

# def 3dArrayToOneByThreeArray(array):
#     returnArr = []
#     for axis in array:
#         for current_slice in axis:
#             for hu_value in current_slice:
                

for index, patient in enumerate(patients):
    patient = load_scan(INPUT_FOLDER + patient)
    patient_pixels = get_pixels_hu(patient)
    
    #pixels_resampled, spacing = resample(patient_pixels, patient, [1,1,1])
    pixels_resampled = resize_to_dimensions(patient_pixels, 335, 335, 335);
    segmented_lungs = segment_lung_mask(pixels_resampled, False)
    #if you want to visualize the data uncomment the below 2 lines
    #plot_3d(segmented_lungs, 0) 
    #break;
    
    with open("patient_data_0/patient_{0}.dat".format(index),"w+") as f:
        for axis in segmented_lungs:
            f.write("/1\n")
            for current_slice in axis:
                f.write("/2\n")
                for hu_value in current_slice:
                    f.write(str(hu_value))
                    f.write(",")
                f.write('\n')
    
    with open("patient_data_0/patient_{0}.bin".format(index),"w+b") as f:
        counter = 7;
        byte = 0;
        for axis in segmented_lungs:
            for current_slice in axis:
                #add 8 bits into a byte and write it out to file
                for hu_value in current_slice:
                    bit = hu_value * pow(2, counter);
                    byte += bit;
                    if counter == 0:
                        f.write(byte.item().to_bytes(1, byteorder='big',signed=False))
                        byte = 0
                        counter = 7
                    else:
                        counter -= 1
            
#---------------------------------------------------------------------------------------------------------- load_array.cpp will be based off this                
# overall_array = []
# outer_index = -1;
# inner_index = -1;
# with open("patient_0.dat", "r") as f:
#     for line in f:
#         for s in line.split(','):
#             s = s.rstrip()
#             if s == "/1":
#                 outer_index += 1
#                 inner_index = -1
#                 overall_array.append([])
#             elif s == "/2":
#                 inner_index += 1
#                 overall_array[outer_index].append([])
#             else:
#                 if s != '':
#                     overall_array[outer_index][inner_index].append(int(s))
#---------------------------------------------------------------------------------------------------------- other testing


# print(len(overall_array))
# print(len(overall_array[0]))
# print(len(overall_array[0][0]))
            # first_patient = load_scan(INPUT_FOLDER + patients[0])
                # first_patient_pixels = get_pixels_hu(first_patient)
                # plot.hist(first_patient_pixels.flatten(), bins=80, color='c')
                # plot.xlabel("Hounsfield Units (HU)")
                # plot.ylabel("Frequency")
                # plot.show()

# # Show some slice in the middle
# plot.imshow(first_patient_pixels[80], cmap=plot.cm.gray)
# plot.show()

# pix_resampled, spacing = resample(first_patient_pixels, first_patient, [1,1,1])
# print("Shape before resampling\t", first_patient_pixels.shape)
# print("Shape after resampling\t", pix_resampled.shape)

# # plot_3d(pix_resampled, 400)

# segmented_lungs = segment_lung_mask(pix_resampled, False)
# segmented_lungs_fill = segment_lung_mask(pix_resampled, True)


# print(segmented_lungs)
# # np.savetxt("foo.csv", segmented_lungs, delimiter=",");
# segmented_lungs.tofile('foo.csv', sep=',', format='%10.5f')
# # df = pd.DataFrame(segmented_lungs)
# print(segmented_lungs.ndim)
# print(segmented_lungs[0].ndim)
# print(segmented_lungs[1].ndim)


    

# # plot_3d(segmented_lungs, 0)
