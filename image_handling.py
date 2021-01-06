import numpy as np

def grab_postage_stamp(image,cropsize,xcentroid,ycentroid):
    '''
        Given an image, desired downsize, and centroid positions, crop out a postage stamp
        returns the cropped image, and the centroids in the new image.
    '''
    dif =int(cropsize/2)
    x,y=int(np.round(xcentroid)),int(np.round(ycentroid))
    ##2 Cases for even or odd number of pixels required
    if dif%2 == 1: ##odd case
        xmin = np.max([x-dif,0])
        xmax = np.min([x+dif+1,image.shape[0]])
        ymin = np.max([y-dif,0])
        ymax = np.min([y+dif+1,image.shape[1]])
    else : ##even case
        xmin = np.max([x-dif,0])
        xmax = np.min([x+dif,image.shape[0]])
        ymin = np.max([y-dif,0])
        ymax = np.min([y+dif,image.shape[1]])
    #import pdb
    #pdb.set_trace()
    return image[ymin:ymax,xmin:xmax]
    

    