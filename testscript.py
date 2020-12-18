import numpy as np
import pdb,os,sys,glob,pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
from astropy.io import fits
from scipy.io import readsav
import pyoifits
import pandas as pd
import image_handling
##data location
datadir = os.getenv("HOME")+'/data_local/nirc2/ADAM/1204/'


import numpy as np
import glob,os,sys

infofile = 'targlistcentroids.txt'

ff = open(infofile,'r')
tnum = []
xcen = []
ycen = []
fprefix=[]
tname = []
for line in ff:
    tnum.append(int(line[0:2]))
    xcen.append(float(line[4:10]))
    ycen.append(float(line[13:19]))
    fprefix.append(line[20:37])
    tname.append(line[37:-1].lstrip().rstrip())
    #pdb.set_trace()    
ff.close()
xcen=np.array(xcen)
ycen=np.array(ycen)
tname=np.array(tname)
tnum=np.array(tnum)
fprefix=np.array(fprefix)



##Select a target, this is a random FFTau image
targ = np.where(tname == 'FF-Tau')[0]
qwe= 99578 ##This happens to be n0392.fits
LDIF = glob.glob(datadir+fprefix[qwe]+'.LDIF.fits')
hdu = fits.open(LDIF[0])
target_image = hdu[0].data
target_header = hdu[0].header

'''
    PLAN
    0. Read header and determine correct mask template file
    1. Crop the image
    2. Read mask template file 
    3. Calc Bispec
    4. Save as OIFITS
    

'''

##MASK SETUP
mask_setup = target_header['FILTER']
mflookup = pd.read_csv('mask_template_lookup.txt',delimiter=',').to_records()
mftemplatefile = 'nrm_mask_templates/nirc2/'+mflookup[np.where(mflookup.filter == mask_setup)[0]].templatefile[0]
maskinfo = readsav(mftemplatefile)

pdb.set_trace()




##crop out the postage stamp as per qbe_nirc2
cropsize=256
cropped_image=image_handling.grab_postage_stamp(target_image,cropsize,xcen[qwe],ycen[qwe])

##This is where calc_bispec really starts 






pdb.set_trace()
print('Done')