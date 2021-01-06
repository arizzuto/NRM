import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.io import readsav
import pyoifits
import pandas as pd
import image_handling
import mask_functions
import os,pickle


def run_bispec(LDIF_prefix,targetcentroidfile = 'targlistcentroids.txt',datadir='/data/03344/akraus/NIRC2',savedir=None,dontsave=False,giveoutput=False):
    '''
        Handler that does all the steps needed for single frame bispectrum calculation
        INPUTS:
        ---------
            LDIF_prefix: the nirc2 archive file prefix (e.g. N2.20180701.50403) to look up in the targetlistcentroids.txt file
        OPTIONALS:
        ----------
            datadir:A Root Location that the data is stored in, default is Adam's wrangler drive
                if datadir is set to None, will assume LDIF_prefix is the actual complete filename 
            targetcentroidfile: text file containing the outpus from Adam Kraus' cleaning/preprocessing stuff 
            savedir: Optional save target location, otherwise a directory structure conforming to ALK's setup will be created
                     from datadir variable.
            dontsave: default False, if True will not write output to disk, if False will write and return written filename
            giveoutput: default False, it True will return output variables, wont do anything if dontsave is False
                
        
    '''    
    
    if datadir is None: 
        LDIF_fname = LDIF_prefix
    else:
        ##Start by stripping the date information out of the LDIF_prefix string
        if datadir[-1] != '/': datadir += '/'

        datestr = LDIF_prefix.split('.')[1]
        Yr,Mn,Dy      = datestr[0:4],datestr[4:6],datestr[6:]
    
        ##construct the filename
        LDIF_fname = datadir + 'LDIF/' + Yr + 'c/' + Mn+Dy + '/' + LDIF_prefix + '.LDIF.fits'
    
    ##Read the fits file for this frame
    if os.path.isfile(LDIF_fname) == False:
        print('LDIF filename and path extrapolated from prefix does not exist!')
        print(LDIF_fname)
        import pdb
        pdb.set_trace()
        
    hdu = fits.open(LDIF_fname)
    target_image = hdu[0].data
    target_header = hdu[0].header
    hdu.close()
    ##Grab MASK SETUP and read the mask file
    mask_setup = target_header['FILTER']
    minfo = mask_functions.NRM_mask(mask_setup)
    minfo.find_maskfile()
    minfo.read_IDL_maskfile()
    if hasattr(minfo,'maskfile') == False:
        print('This filter setup was not in the lookup table:')
        print(mask_setup)
        print('Either its wrong, or you need to add a new line to the file \n mask_template_lookup.txt')
        import pdb
        pdb.set_trace()
        
    ##Now read the centroids file and find the target frame
    ff = open(targetcentroidfile,'r')
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
    
    ff.close()
    xcen=np.array(xcen)
    ycen=np.array(ycen)
    tname=np.array(tname)
    tnum=np.array(tnum)
    fprefix=np.array(fprefix)
    
    match = np.where(fprefix == LDIF_prefix)[0]
    if len(match) == 0: 
        print("Can't find this LDIF prefix in centroid file, either remake centroid files if this is new data, or make sure the prefix is correct")
        print(LDIF_file)
        print(LDIF_prefix)
        print(targetcentroidfile)
        import pdb
        pdb.set_trace()
    if len(match) > 1: 
        print("Multiple entries for this LDIF prefix in target centroid file, check to make sure everything is fine")
        print(LDIF_file)
        print(LDIF_prefix)
        print(targetcentroidfile)
        import pdb
        pdb.set_trace()
   
    ##crop out the postage stamp as per qbe_nirc2
    cropsize=256
    cropped_image=image_handling.grab_postage_stamp(target_image,cropsize,xcen[match[0]],ycen[match[0]])

    ##This is where calc_bispec really starts 

    phase, V2,bispectrum,flux,complex_vis,phase_slope,phase_slope_err,ftimage    = calc_bispec(cropped_image,minfo)
    
    if dontsave == False:
        ##build the output:
        outputdir = datadir+'NRM/'+ Yr + '/' + Mn+Dy + '/'
        outfilename = outputdir + LDIF_prefix + '.LDIFNRM.pkl'
        if os.path.isdir(outputdir) == False:
            if os.path.isdir(datadir +'NRM/') == False:os.mkdir(datadir+'NRM/')
            if os.path.isdir(datadir+'NRM/' + Yr) == False:  os.mkdir(datadir+'NRM/' + Yr)
            os.mkdir(outputdir)
    
        if os.path.exists(outfilename) == True:
            print('Overwriting existing file with bispec output')
    
        ffout = open(outfilename,'wb')
        pickle.dump((phase,V2,bispectrum,flux,complex_vis,phase_slope,phase_slope_err,ftimage,cropped_image,minfo,LDIF_fname,tname[match[0]]),ffout)
        ffout.close()
        return outfilename

    if giveoutput == True:    
        return phase,V2,bispectrum,flux,complex_vis,phase_slope,phase_slope_err,ftimage,cropped_image,minfo,LDIF_fname,tname[match[0]]

def dist(NAXIS):
    if NAXIS % 2 != 0 :
        print('Warning, this function fails on odd numbered sizes!!!!')
        print('Cropped image size should be even (256x256) pixels')
        import pdb
        pdb.set_trace()
    """Returns a rectangular array in which the value of each element is proportional to its frequency.

    >>> dist(3)
    array([[ 0.        ,  1.        ,  1.        ],
           [ 1.        ,  1.41421356,  1.41421356],
           [ 1.        ,  1.41421356,  1.41421356]])

    >>> dist(4)
    array([[ 0.        ,  1.        ,  2.        ,  1.        ],
           [ 1.        ,  1.41421356,  2.23606798,  1.41421356],
           [ 2.        ,  2.23606798,  2.82842712,  2.23606798],
           [ 1.        ,  1.41421356,  2.23606798,  1.41421356]])

    """
    NA2 = int(np.ceil((NAXIS/2)))
    axis = np.linspace(-NA2+1, NA2, NAXIS)
    result = np.sqrt(axis**2 + axis[:,np.newaxis]**2)
   # pdb.set_trace()
    return np.roll(result, NA2+1, axis=(0,1))


def calc_bispec(image,maskinfo,windowsize=0.8,window_type='SupGauss',coherent_vis=True,coherent_bs=True):
    '''
        Function for running the bispectrum calculation, generally follows the idlNRM version
        from Mike Ireland's github page. This is just the frame-by-frame stuff.
        
        INPUTS:
        -------
            image: the cropped image
            maskinfo: the NRMmask class object for the correct mask for this image
        OPTIONALS:
        ----------
            windowsize=0.8: window half-max radius multiplier (set by hole size) 
            window_type='SupGauss': Either SupGauss=Super-Gaussian or 'none'=nothing
            coheret_vis=True: Do we coherently integrate the visibility over a splodge
            coheret_bs=True: Do we coherently integrate the bispectrum over a splodge
    '''
    
    
    ##Calculate the default image size:
    imsize = maskinfo.filter[0]/maskinfo.hole_diam/maskinfo.rad_pixel
    
    dimx,dimy = image.shape[0],image.shape[1]

    '''
    Adjust the window sizing
    This is the radius from the image centroid that the window goes down to half its
    peak value (half max radius)
    1.3 supresses, bit still includes the first airy ring, 
    0.8 (default) is slightly larget than S/N optimal
    The window multiple kewords from idlNRM are currently not implemented
    '''
    window_size = windowsize*imsize 
    
    ##make the window
    window=np.zeros((dimx,dimy),dtype=float)
    if window_type == 'SupGauss':
       ### dist(3)
        window = np.exp(-(dist(min([dimx,dimy]))/window_size*0.91244)**4)
    else: 
        window=np.zeros((dimx,dimy),dtype=float)+1.0
    
    ##This is where the actual calculations start happening

    
    ##Do the Fourier transform
    #pdb.set_trace()
    ftimage = np.fft.ifft2(np.roll(image,(int(-dimx/2),int(-dimy/2)),axis=(0,1))*window)*dimx*dimy
    ##The above line seems like it's putting splodges into the right places, the overall bias
    ##level and scaling is different from idlNRM though, probably because Adam has 
    ## done a different sort of bias-sub/cleaning preprocess. 
    
    ##Now the bispectrum calculations
    n_holes = maskinfo.n_holes*1.0 ##casting to float
    n_baselines = maskinfo.bl2bs_ix.shape[1]
    n_bispect = maskinfo.bs2bl_ix.shape[0]
    n_cov = maskinfo.bscov2bs_ix.shape[0]  
    
    ##initialize some things
    cvis = np.zeros(n_baselines,dtype=complex)
    phs_arr=np.zeros((n_baselines,2),dtype=float)
    phserr_arr=np.zeros((n_baselines,2),dtype=float)
    ##power spectrum
    ps = np.abs(ftimage)**2
    
    ##loop over baselines:
    for j in range(n_baselines):
        ##complex visibilities:
        pix=maskinfo.mf_pvct[maskinfo.mf_ix[j,0]:maskinfo.mf_ix[j,1]+1]
        cvis[j] = np.sum(maskinfo.mf_gvct[maskinfo.mf_ix[j,0]:maskinfo.mf_ix[j,1]+1]*ftimage.flatten()[pix])
        ##compute phase slope arrays
        ftf1 = np.roll(ftimage,-1,axis=0)
        ftf2 = np.roll(ftimage,1,axis=0)
        dummy = np.sum(ftimage.flatten()[pix]*np.conjugate(ftf1).flatten()[pix] + np.conjugate(ftimage).flatten()[pix]*ftf2.flatten()[pix])
        phs_arr[j,0] = np.arctan2(dummy.imag,dummy.real)
        phserr_arr[j,0] = 1/np.abs(dummy)
        ftf1 = np.roll(ftimage,-1,axis=1)
        ftf2 = np.roll(ftimage,1,axis=1)
        dummy = np.sum(ftimage.flatten()[pix]*np.conjugate(ftf1).flatten()[pix] + np.conjugate(ftimage).flatten()[pix]*ftf2.flatten()[pix])
        phs_arr[j,1] = np.arctan2(dummy.imag,dummy.real)
        phserr_arr[j,1] = 1/np.abs(dummy)

    ##overlapping baseline corrections
    rvis =   np.dot(maskinfo.mf_rmat,cvis.real)
    ivis =   np.dot(maskinfo.mf_imat,cvis.imag)
    cvis.real=rvis
    cvis.imag=ivis
    
    ##phases
    ph_arr = np.arctan2(cvis.imag,cvis.real)
    ##V-squared
    V2_arr = np.abs(cvis)**2
    ##Bispectrum cvis[bs2bl_ix[0,*]]*cvis[bs2bl_ix[1,*]]*conj(cvis[bs2bl_ix[2,*]])
    bs_arr = cvis[maskinfo.bs2bl_ix[:,0]]*cvis[maskinfo.bs2bl_ix[:,1]]*np.conjugate(cvis[maskinfo.bs2bl_ix[:,2]])
    ##Flux, this is just the peak power on the fourier transformed image
    flux =  np.abs(ftimage[0,0])

    ##closure phase, probably don't calculate this here
    #clph = np.arctan2(bs_arr.imag,bs_arr.real)*180.0/np.pi    
    #pdb.set_trace()
    
    return ph_arr, V2_arr,bs_arr,flux,cvis,phs_arr,phserr_arr,ftimage    
    
def bispectrum_runstats(NRMfilenames):
    '''
        Function for doing statistics on the bispectrum products for a target run of multiple frames, needs at least 3 frames.
        
        INPUTS:
        -------
        NRMfilenames: list of filenames for a target run, if target names/filters don't match there will be a problem.
    '''
    nframes=len(NRMfilenames)
    target_names = np.zeros(nframes,dtype='|U50')
    LDIFs = np.zeros(nframes,dtype='|U200')
    masks = []
    flux_arr = np.zeros(nframes,dtype=float)
    mask_setups = []
    for i in range(nframes):
        fff = open(NRMfilenames[i],'rb')
        phase,V2,bispectrum,flux,complex_vis,phase_slope,phase_slope_err,ftimage,cropped_image,minfo,LDIF_fname,tname = pickle.load(fff)
        fff.close()
        target_names[i] = tname
        LDIFs[i]        = LDIF_fname
        masks.append(minfo)
        mask_setups.append(minfo.id)
        flux_arr[i] = flux
        if i == 0: ##On the first pass, figure out array sizes and initialize, things should be consistent, if they aren't something bad has happened upstream.
            image_arr   = np.zeros((cropped_image.shape[0],cropped_image.shape[1],nframes),dtype=float)
            ftimage_arr = np.zeros((ftimage.shape[0],ftimage.shape[1],nframes),dtype=float)
            phserr_arr  = np.zeros((phase_slope_err.shape[0],phase_slope_err.shape[1],nframes),dtype=float)
            phs_arr     = np.zeros((phase_slope.shape[0],phase_slope.shape[1],nframes),dtype=float)
            cvis_arr    = np.zeros((complex_vis.shape[0],nframes),dtype=float)   
            bs_arr      = np.zeros((bispectrum.shape[0],nframes),dtype=float)
            v2_arr      = np.zeros((V2.shape[0],nframes),dtype=float)
            ph_arr      = np.zeros((phase.shape[0],nframes),dtype=float)
            
        image_arr[:,:,i]   = cropped_image.copy()
        ftimage_arr[:,:,i] = ftimage.copy()
        phserr_arr[:,:,i]  = phase_slope_err.copy()
        phs_arr[:,:,i]     = phase_slope.copy()
        cvis_arr[:,i]      = complex_vis.copy()
        bs_arr[:,i]        = bispectrum.copy()
        v2_arr[:,i]        = V2.copy()
        ph_arr[:,i]        = phase.copy()
            

        if len(np.unique(mask_setups)) != 1:
            print('There are multiple mask setups in this reduction block, your target groupings are likely wrong!')
            import pdb
            pdb.set_trace()
            

        ##From here, we can carry on doing the bispectrum multi-frame stats and covariance calculations from bispect.pro    
        
            
            
            
            
            

    import pdb
    pdb.set_trace()

