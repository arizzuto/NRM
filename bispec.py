import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.io import readsav
import pyoifits
import pandas as pd
import image_handling
import mask_functions
import os,pickle
from scipy.optimize import minimize


def cov2cor(cov):

	dim1 = cov.shape[0]
	dim2 = cov.shape[1]
	corr = cov.copy()
	for i in range(0,dim1):
		sig_i = np.sqrt(cov[i,i])
		for j in range(0,dim2):
			sig_j = np.sqrt(cov[j,j])
			corr[i,j] = cov[i,j]/sig_i/sig_j
			
	sig = np.sqrt(np.diag(cov))
			
	return corr,sig

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
    import pdb
    pdb.set_trace()
    
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

    phase, V2,bispectrum,flux,complex_vis,phase_slope,phase_slope_err,ps,ftimage    = calc_bispec(cropped_image,minfo)
    
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
        pickle.dump((phase,V2,bispectrum,flux,complex_vis,phase_slope,phase_slope_err,ps,ftimage,cropped_image,minfo,LDIF_fname,tname[match[0]]),ffout)
        ffout.close()
        return outfilename

    if giveoutput == True:    
        return phase,V2,bispectrum,flux,complex_vis,phase_slope,phase_slope_err,ps,ftimage,cropped_image,minfo,LDIF_fname,tname[match[0]]

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
    
    return ph_arr, V2_arr,bs_arr,flux,cvis,phs_arr,phserr_arr,ps,ftimage    
    
def phase_chi2(p,fitmat,ph_mn,ph_err):

    
    dummy = np.concatenate((ph_mn,[0]))
    dummy2 = np.zeros(dummy.shape,dtype=complex)
    dummy2.imag = dummy
    varvar = dummy2 - np.dot(p,fitmat)
    var3   = np.concatenate((ph_err,[0.01]))**2
    return np.sum(np.abs(1.0 - np.exp(varvar))**2/var3)
    

def regress_noc(x,y,weights):	
    from numpy.linalg import inv
    ##On_error,2              ;Return to caller if an error occurs 
    #SY = SIZE(Y)            ;Get dimensions of x and y.  
    #SX = SIZE(X)
    #IF (N_ELEMENTS(Weights) NE SY[1]) OR (SX[0] NE 2) OR (SY[1] NE SX[2]) THEN $
    #  message, 'Incompatible arrays.'
    
   # NTERM = SX[1]           ;# OF TERMS
    #NPTS = SY[1]            ;# OF OBSERVATIONS
    
    nterm = x.shape[0]# OF TERMS
    npts = y.shape[0]# OF OBSERVATIONS
    xwy = np.dot(x,(weights*y))	
    wx= np.zeros((npts, nterm),dtype=float)
    for i in range(npts): wx[i,:] = x[:,i]*weights[i]
    xwx = np.dot(x,wx)
    cov = inv(xwx)				##;Neter et al pg 402-403
    coeff = np.dot(cov,xwy)
    yfit = np.dot(x.T,coeff)
    if npts != nterm: MSE = np.sum(weights*(yfit-y)**2)/float(npts-nterm)
    var_yfit = np.zeros(npts,dtype=float)
    for i in range(npts): var_yfit[i] = np.dot(np.dot(x[:,i].T,cov),x[:,i]) ###  ;Neter et al pg 233

    return coeff,cov,yfit, MSE, var_yfit


def bispectrum_runstats(NRMfilenames,n_blocks=0,subtract_bs_bias=False,output_cp_cov=True):
    '''
        Function for doing statistics on the bispectrum products for a target run of multiple frames, needs at least 3 frames.
        
        INPUTS:
        -------
        NRMfilenames: list of filenames for a target run, if target names/filters don't match there will be a problem.
        OPTIONALS:
        -----------
        n_blocks=0: number of chunks to use when calculating variances/covars, if default of 0 it will make each frame a block
        subtract_bs_bias=False: Do we subtract the bispectrum bias?
        output_cp_cov=True: Calculate the closure-phase covariances, will not do this for masks with more than 21 holes.      
    '''
    nframes=len(NRMfilenames)
    target_names = np.zeros(nframes,dtype='|U50')
    LDIFs = np.zeros(nframes,dtype='|U200')
    masks = []
    flux_arr = np.zeros(nframes,dtype=float)
    mask_setups = []
    
    for i in range(nframes):
        fff = open(NRMfilenames[i],'rb')
        phase,V2,bispectrum,flux,complex_vis,phase_slope,phase_slope_err,ps,ftimage,cropped_image,minfo,LDIF_fname,tname = pickle.load(fff)    
        fff.close()
        target_names[i] = tname
        LDIFs[i]        = LDIF_fname
        masks.append(minfo)
        mask_setups.append(minfo.id)
        flux_arr[i] = flux
        if i == 0: ##On the first pass, figure out array sizes and initialize, things should be consistent, if they aren't something bad has happened upstream.
            image_arr   = np.zeros((cropped_image.shape[0],cropped_image.shape[1],nframes),dtype=float)
            ftimage_arr = np.zeros((ftimage.shape[0],ftimage.shape[1],nframes),dtype=complex)
            phserr_arr  = np.zeros((phase_slope_err.shape[0],phase_slope_err.shape[1],nframes),dtype=float)
            phs_arr     = np.zeros((phase_slope.shape[0],phase_slope.shape[1],nframes),dtype=float)
            cvis_arr    = np.zeros((complex_vis.shape[0],nframes),dtype=complex)   
            bs_arr      = np.zeros((bispectrum.shape[0],nframes),dtype=complex)
            v2_arr      = np.zeros((V2.shape[0],nframes),dtype=float)
            ph_arr      = np.zeros((phase.shape[0],nframes),dtype=float)
            ps_arr      = np.zeros((ps.shape[0],ps.shape[1],nframes),dtype=float)
            
        image_arr[:,:,i]   = cropped_image.copy()
        ftimage_arr[:,:,i] = ftimage.copy()
        phserr_arr[:,:,i]  = phase_slope_err.copy()
        phs_arr[:,:,i]     = phase_slope.copy()
        cvis_arr[:,i]      = complex_vis.copy()
        bs_arr[:,i]        = bispectrum.copy()
        v2_arr[:,i]        = V2.copy()
        ph_arr[:,i]        = phase.copy()
        ps_arr[:,:,i]      = ps.copy()
        if len(np.unique(mask_setups)) != 1:
            print('There are multiple mask setups in this reduction block, your target groupings are likely wrong!')
            import pdb
            pdb.set_trace()
            

    ##From here, we can carry on doing the bispectrum multi-frame stats and covariance calculations from bispect.pro    
    
    ##Start by finding regions of the FT image that have signal
    ##This is just making a mask of all the splodges (the signals)
    dimx,dimy = ftimage_arr.shape[0],ftimage_arr.shape[1]
    signal = np.zeros((dimx,dimy),dtype=float)
    mf_pvct_ur = np.unravel_index(minfo.mf_pvct,signal.shape)
    signal[mf_pvct_ur[0],mf_pvct_ur[1]] = 1.0
    signal += np.roll(np.rot90(signal,k=2),(1,1),(0,1))
    ##This adds the central peak at the corners out to 16 sigmas
    signal[np.where(dist(dimx) < dimx/16)] = 1.0
    
    ##Average Power Spectrum
    ps = np.mean(ps_arr,axis=2)
    n_ps = ps_arr.shape[2]
    ##From the signal-free regions, get the median
    ##This seems hacky but hey it works
    for dummy in range(4):
        ww = np.where(signal == 0)
        bias = np.median(ps[ww[0],ww[1]])
        signal[np.where(ps > 3.0*bias)] = 1.0  
    ww = np.where(signal != 0) ##Signal free points
            
    ##Now we know where the signal is, we can find the 'real' bias that
    ##includes the correlation between neighbouring terms in each
    ##ft_frame...       
    autocor_noise = np.zeros((dimx,dimy),dtype=float)
    
    ##Zero the signal points in the FT images
    ##ifft them, modsquare, the fft again, i.e. autocorrelation via FFTs
    for i in range(n_ps):
        ft_frame = ftimage_arr[:,:,i]
        ft_frame[ww[0],ww[1]] = 0.0
        autocor_noise += np.fft.fft2(np.abs(np.fft.ifft2(ft_frame))**2).real
    autocor_noise /= n_ps
        
    biasmn = np.zeros(minfo.n_baselines,dtype=float)
    for j in range(minfo.n_baselines):
        mf     = np.zeros((dimx,dimy),dtype=float)
        mark1  = np.unravel_index(minfo.mf_pvct[minfo.mf_ix[j,0]:minfo.mf_ix[j,1]+1],mf.shape)
        result = minfo.mf_gvct[minfo.mf_ix[j,0]:minfo.mf_ix[j,1]+1]
        mf[mark1]  = result
        autocor_mf = np.fft.fft2(np.abs(np.fft.ifft2(mf))**2).real
        biasmn[j] = np.sum(autocor_mf*autocor_noise)*bias/autocor_noise[0,0]*dimx*dimy
        v2_arr[j,:] -= biasmn[j]
    
    
    ##At this point, since there's no dark bias or dark noise we don't have to worry about adding it back in 
    ##to deal with the double subtraction issues.

    ftimage_arr = 0
    
    ##Now convert arrays of interesting variables into means/covariances over frames.
    if n_blocks == 0: n_blocks = n_ps ##default to the number of frames
    
    v2_cov    = np.zeros((minfo.n_baselines,minfo.n_baselines),dtype=float)
    bs        = np.zeros((minfo.n_baselines,minfo.n_baselines),dtype=complex)
    bs_var    = np.zeros((2,minfo.n_bispect),dtype=float)
    bs_cov    = np.zeros((2,minfo.n_cov),dtype=float)
    bs_v2_cov = np.zeros((minfo.n_baselines,minfo.n_holes-2),dtype=float)

    print('Calculating mean V2 and Variances')
    v2        = np.mean(v2_arr,1)
    v2diff    = np.zeros((n_blocks,minfo.n_baselines),dtype=float) ##differences in V2 average in each block compared to global average
    for j in range(minfo.n_baselines):
        for k in range(n_blocks):            
            v2diff[k,j] = np.mean(v2_arr[j,int(k*n_ps/n_blocks):int((k+1)*n_ps/n_blocks)]) - v2[j]
        
    for j in range(minfo.n_baselines):
        for k in range(minfo.n_baselines):
            v2_cov[j,k] = np.sum(v2diff[:,j]*v2diff[:,k])/(n_blocks-1.0)/float(n_blocks)
            
    ##Now, for the case where we have coherent integration over splodges,
    ##it is easy to calculate the bias in the variance. (note that the
    ##variance of the bias is simply the square of it's mean)
    x = np.arange(minfo.n_baselines,dtype=int)        
    avar = v2_cov[x,x]*n_ps - (1+2.0*v2/biasmn)*biasmn**2
    err_avar = np.sqrt(2.0*n_ps*v2_cov[x,x]**2 + 4.0*v2_cov[x,x]*biasmn*82) ##Assumes no error in biasmnerr_avar = sqrt(2.0/n_ps*double(v2_cov[x,x])^2*n_ps^2 + 4.0*v2_cov[x,x]*biasmn^2) ;Assumes no error in biasmn...
            
   
    print('Calculating mean biscpetrum and variance')        
    bs = np.mean(bs_arr,axis=1)
    ##Bispectral bias subtraction. This assumes that the matched filter has been
    ##correctly normalised...
    if subtract_bs_bias == True:
        bs_bias = v2[minfo.bs2bl_ix[:,0]] + v2[minfo.bs2bl_ix[:,1]] + v2[minfo.bs2bl_ix[:,2]] + np.mean(flux_arr)
        bs -= bs_bias
        print('Maximum bispectrum bias (as fraction of bispectra amplitude): ' + str(np.max(bs_bias)/np.abs(bs)))
    
    temp = np.zeros(n_blocks,dtype=complex)
    for j in range(minfo.n_bispect):
        ##temp is the complex difference from the mean, shifted so that real axis corresponds to amplitude, and imaginary axis to phase.
        temp2 = (bs_arr[j,:]-bs[j])*np.conjugate(bs[j])
        for k in range(n_blocks): temp[k] = np.mean(temp2[int(k*n_ps/n_blocks):int((k+1)*n_ps/n_blocks)])
        bs_var[0,j] = np.sum(temp.real**2)/float(n_blocks)/(n_blocks-1.0)/np.abs(bs[j])**2
        bs_var[1,j] = np.sum(temp.imag**2)/float(n_blocks)/(n_blocks-1.0)/np.abs(bs[j])**2
            
        
    print("Calculating Bispectral Covariances")
    for j in range(minfo.n_cov):
        temp1 = (bs_arr[minfo.bscov2bs_ix[j,0],:] - bs[minfo.bscov2bs_ix[j,0]])*np.conjugate(bs[minfo.bscov2bs_ix[j,0]])
        temp2 = (bs_arr[minfo.bscov2bs_ix[j,1],:] - bs[minfo.bscov2bs_ix[j,1]])*np.conjugate(bs[minfo.bscov2bs_ix[j,1]])
        denom = np.abs(bs[minfo.bscov2bs_ix[j,0]])*np.abs(bs[minfo.bscov2bs_ix[j,1]])*(n_ps-1.0)*n_ps
        bs_cov[0,j] = np.sum(temp1.real*temp2.real)/denom
        bs_cov[1,j] = np.sum(temp1.imag*temp2.imag)/denom
        
    print("Calculating Covariance Between Power and Bispectral Amplitude")
#     This complicated thing below calculates the dot product between the
#     bispectrum point and its error term ie (x . del_x)/|x| and
#     multiplies this by the power error term. Note that this is not the
#     same as using absolute value, and that this sum should be zero where
#     |bs| is zero within errors.
    for j in range(minfo.n_baselines):
        for k in range(minfo.n_holes-2):
            first_term     = ((bs_arr[minfo.bl2bs_ix[k,j],:]-bs[minfo.bl2bs_ix[k,j]])*np.conjugate(bs[minfo.bl2bs_ix[k,j]])).real
            second_term    = v2_arr[j,:]-v2[j]
            third_term     = np.abs(bs[minfo.bl2bs_ix[k,j]])*(n_ps-1.0)*float(n_ps)
            bs_v2_cov[j,k] = np.sum(first_term*second_term)/third_term
    
    if (minfo.n_holes < 22) & (output_cp_cov == True):
        cp_cov = np.zeros((minfo.n_bispect,minfo.n_bispect),dtype=float)
        for i in range(minfo.n_bispect):
            for j in range(minfo.n_bispect):
                temp1 = (bs_arr[i,:] - bs[i])*np.conjugate(bs[i])
                temp2 = (bs_arr[j,:] - bs[j])*np.conjugate(bs[j])
                denom = (n_ps-1.0)*float(n_ps)*np.abs(bs[i])**2*np.abs(bs[j])**2
                cp_cov[i,j] = np.sum(temp1.imag*temp2.imag)/denom
        
    else: cp_cov=-1
    
    ##Normalise Return Variables
    smallfloat = 1.0000000e-16
    bs_all   = bs_arr/np.mean(flux_arr**3)*minfo.n_holes**3
    v2_all   = v2_arr.T/np.mean(flux_arr**2)*minfo.n_holes**2
    cvis_all = cvis_arr.T/np.mean(flux_arr)*minfo.n_holes
    v2       = v2/np.mean(flux_arr**2)*minfo.n_holes**2
    bs       = bs/np.mean(flux_arr**3)*minfo.n_holes**3
    v2_cov   = v2_cov/np.mean(flux_arr**4)*minfo.n_holes**4
    qwe      =  np.where(v2_cov < smallfloat)[0]
    if len(qwe) != 0: v2_cov[qwe[0],qwe[1]] = smallfloat
    avar     = avar/np.mean(flux_arr**4)*minfo.n_holes**4
    qwe      = np.where(avar < smallfloat)
    avar[qwe[0]] = smallfloat
    err_avar = err_avar/np.mean(flux_arr**4)*minfo.n_holes**4
    qwe      = np.where(err_avar < smallfloat)
    err_avar[qwe[0]] = smallfloat
    fluxes = flux_arr/10000.00 ##prevents overflows
    bs_v2_cov = (bs_v2_cov/np.mean(fluxes**5)*minfo.n_holes**5).real/10000.0**5
    bs_cov    = bs_cov/np.mean(fluxes**6)*minfo.n_holes**6/10000.0**6
    bs_var    = bs_var/np.mean(fluxes**6)*minfo.n_holes**6/10000.0**6
    qwe = np.where(bs_var < smallfloat)[0]
    bs_var[qwe] = smallfloat
    
#     Finally, convert baseline variables to hole variables...
#     1) In the MAPPIT-style, we define the relationship between hole phases 
#     (or phase slopes) and baseline phases (or phase slopes) by the
#     use of a matrix, fitmat. 
    fitmat = np.zeros((minfo.n_holes,minfo.n_baselines+1),dtype=float)
    for j in range(minfo.n_baselines): fitmat[minfo.bl2h_ix[j,0],j] = 1.0
    for j in range(minfo.n_baselines): fitmat[minfo.bl2h_ix[j,1],j] = -1.0
    fitmat[0,minfo.n_baselines] = 1.0
    
    ##Start with the fit to the phases by doing a weighted LS fit to baseline phasors
    dummy = np.zeros(ph_arr.T.shape,dtype=complex)
    dummy.imag = ph_arr.T
    phasors = np.exp(dummy)
    ph_mn = np.arctan2(np.sum(phasors,axis=0).imag,np.sum(phasors,axis=0).real)
    ph_err = np.ones(len(ph_mn),dtype=float)
    for j in range(minfo.n_baselines): ph_err[j] = np.std(np.mod((ph_arr[j,:]-ph_mn[j]+3*np.pi),np.pi*2)-np.pi)
    ph_err = ph_err/np.sqrt(n_ps)
    result = minimize(phase_chi2, np.zeros(minfo.n_holes),args=(fitmat,ph_mn,ph_err),tol=1e-3)
    hole_piston = result['x']
    if hole_piston[0] != -1:
        print('Phase Chi^2: ' + str(phase_chi2(hole_piston,fitmat,ph_mn,ph_err)/(minfo.n_baselines-minfo.n_holes+1)))
    else: print("Error Calculating Hole Pistons")
    
#     fit to the phase slopes using weighted linear regression.
#     Normalisation:  hole_phs was in radians per Fourier pixel.
#     Convert to phase slopes in pixels.
    phs_arr      = np.transpose(phs_arr,(1,2,0))/2.0/np.pi*dimx
    phserr_arr   = np.transpose(phserr_arr,(1,2,0))
    hole_phs     = np.zeros((2,n_ps,minfo.n_holes),dtype=float)
    hole_err_phs = np.zeros((2,n_ps,minfo.n_holes),dtype=float)
    for j in range(minfo.n_baselines): fitmat[minfo.bl2h_ix[j,1],j] = 1.0
    fitmat = fitmat/2.0
    fitmat=fitmat[:,0:minfo.n_baselines]
    err2 = v2_arr.T.copy() ##just a same size matrix
    err2_bias = err2.copy()
    for j in range(n_ps):
        #import pdb
        #pdb.set_trace()
        hole_phs[0,j,:],cov,YFIT,MSE,var_yfit = regress_noc(fitmat,phs_arr[0,j,:],phserr_arr[0,j,:])
        dummy,sig = cov2cor(cov)
        hole_err_phs[0,j,:] = sig*np.sqrt(MSE)
    
        hole_phs[1,j,:],cov,YFIT,MSE,var_yfit = regress_noc(fitmat,phs_arr[1,j,:],phserr_arr[1,j,:])
        dummy,sig = cov2cor(cov)
        hole_err_phs[1,j,:] = sig*np.sqrt(MSE)
        
        err2[j,:] = (hole_phs[0,j,minfo.bl2h_ix[:,0]]-hole_phs[0,j,minfo.bl2h_ix[:,1]])**2 + (hole_phs[1,j,minfo.bl2h_ix[:,0]]-hole_phs[1,j,minfo.bl2h_ix[:,1]])**2
        err2_bias[j,:] = (hole_err_phs[0,j,minfo.bl2h_ix[:,0]]-hole_err_phs[0,j,minfo.bl2h_ix[:,1]])**2 + (hole_err_phs[1,j,minfo.bl2h_ix[:,0]]-hole_err_phs[1,j,minfo.bl2h_ix[:,1]])**2
    
    hole_mnphs = np.sum(hole_phs,axis=1)/n_ps
    mnerr2 = (hole_mnphs[0,minfo.bl2h_ix[:,0]]-hole_mnphs[0,minfo.bl2h_ix[:,1]])**2 + (hole_mnphs[1,minfo.bl2h_ix[:,0]]-hole_mnphs[1,minfo.bl2h_ix[:,1]])**2
    phs_v2corr = np.zeros(minfo.n_baselines,dtype=float)
    predictor = v2_arr.T.copy()
    for j in range(minfo.n_baselines): predictor[:,j] = err2[:,j] - np.mean(err2_bias[:,j])
    
#      imsize is \lambda/hole_diameter in pixels. A factor of 3.0 was only
#      roughly correct based on simulations.
#      2.5 seems to be better based on real data. 
#      NB there is no window size adjustment here.
    imsize = minfo.filter[0]/minfo.hole_diam/minfo.rad_pixel
    for j in range(minfo.n_baselines): phs_v2corr[j] = np.mean(np.exp(-2.5*predictor[:,j]/imsize**2))
    
    
    ###!!!DONE!!!###
    ##THis is where bispect.pro outputs and calc_bispect.pro keeps going, doing it all in here though

    ##Here are the corresponding outputs from bispect.pro
    ##The transpose on the v2_arr is to make is the same shape as the IDL version.
    ##return v2, v2_cov, bs,bs_var, bs_cov, bs_v2_cov, cp_var, bs_all, v2_all, cvis_all,fluxes,cp_cov,avar,err_avar,v2_arr.T,phs_v2corr

    ##here we've run into a problem, we need the olog data, go back and reverse engineer that.


    import pdb
    pdb.set_trace()
