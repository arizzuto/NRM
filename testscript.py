import numpy as np
import pdb,os,sys,glob,pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
from astropy.io import fits
from scipy.io import readsav
import pyoifits
import pandas as pd
import image_handling
import mask_functions
import bispec
##data location: Replace with you own file locations
datadir = os.getenv("HOME")+'/data_local/nirc2/ADAM/1204/' 



import numpy as np
import glob,os,sys

## Centroid of all targets file produced by Adam Kraus' preprocessing code. Replace with newest version if applicable
infofile = 'targlistcentroids.txt'

##find an example datafile and read in the centroid information
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

#pdb.set_trace()

##Select a target, this is a random FFTau image
qwe= 99578-4 ##This happens to be n0388.fits, the first in a set of 8 for FFTau in future this has to be automated but that's up to the users setup
targetprefix = fprefix[qwe]


fftau_index = [x for x in range(qwe,qwe+8)]
iptau_index = [x for x in range(qwe+8,qwe+8+8)]
runindex = fftau_index + iptau_index
reprocess = True ##This will turn off the file-by-file part ofthe bispectrum calculation to save some time in testing
if reprocess == True:
    #pdb.set_trace()
    ##This runs everything bispec related that can be done on a single frame, except compute the closure-phase triple product because 
    ##That can be done later and is super simple once you have the bispectrum
    outfiles = []
    for i,ind in enumerate(runindex):
        print('Processing Frame ' + str(i+1) + ' of ' + str(len(runindex)))
        outputfilename = bispec.run_bispec(fprefix[ind],datadir='/Users/acr2877/data_local/nirc2/ADAM',dontsave=False,giveoutput=False)
        outfiles.append(outputfilename)
    
    ##following this, we need groupings of target runs to do statistics on the bispectrum/visibility products
    ##right here we need some code to match outputs from the above to target file names
    fftau_nrmfiles = outfiles[0:8]
    iptau_nrmfiles = outfiles[8:]

### Now do the group run statistics calculations: Here I've group two examples which are a target-calibrator pair by hand, this should be done in a more automated way but that will depend on setup
fftau_nrmfiles = ['/Users/acr2877/data_local/nirc2/ADAM/NRM/2012/1204/N2.20121204.33780.LDIFNRM.pkl', '/Users/acr2877/data_local/nirc2/ADAM/NRM/2012/1204/N2.20121204.33807.LDIFNRM.pkl', '/Users/acr2877/data_local/nirc2/ADAM/NRM/2012/1204/N2.20121204.33835.LDIFNRM.pkl', '/Users/acr2877/data_local/nirc2/ADAM/NRM/2012/1204/N2.20121204.33862.LDIFNRM.pkl', '/Users/acr2877/data_local/nirc2/ADAM/NRM/2012/1204/N2.20121204.33890.LDIFNRM.pkl', '/Users/acr2877/data_local/nirc2/ADAM/NRM/2012/1204/N2.20121204.33918.LDIFNRM.pkl', '/Users/acr2877/data_local/nirc2/ADAM/NRM/2012/1204/N2.20121204.33946.LDIFNRM.pkl', '/Users/acr2877/data_local/nirc2/ADAM/NRM/2012/1204/N2.20121204.33973.LDIFNRM.pkl']
iptau_nrmfiles = ['/Users/acr2877/data_local/nirc2/ADAM/NRM/2012/1204/N2.20121204.34094.LDIFNRM.pkl', '/Users/acr2877/data_local/nirc2/ADAM/NRM/2012/1204/N2.20121204.34122.LDIFNRM.pkl', '/Users/acr2877/data_local/nirc2/ADAM/NRM/2012/1204/N2.20121204.34151.LDIFNRM.pkl', '/Users/acr2877/data_local/nirc2/ADAM/NRM/2012/1204/N2.20121204.34179.LDIFNRM.pkl', '/Users/acr2877/data_local/nirc2/ADAM/NRM/2012/1204/N2.20121204.34206.LDIFNRM.pkl', '/Users/acr2877/data_local/nirc2/ADAM/NRM/2012/1204/N2.20121204.34234.LDIFNRM.pkl', '/Users/acr2877/data_local/nirc2/ADAM/NRM/2012/1204/N2.20121204.34261.LDIFNRM.pkl', '/Users/acr2877/data_local/nirc2/ADAM/NRM/2012/1204/N2.20121204.34293.LDIFNRM.pkl']

## This will run the final group biscpetrum calculations for the first of the two groups, and save it in a standard location inside of your data directory according to Adam's naming formats
## see bispec.pro for options on changing the save location.
dummy = bispec.bispectrum_runstats(fftau_nrmfiles)

##At this point you would do the calibration, which for closure phase is just subtracting the calibrator closure phase from the targets, and dividiong the visibilities. idlNRM has 
## fancy features here but they aren't all useful.

print("Done")
