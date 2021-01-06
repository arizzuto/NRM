import numpy as np
import pandas as pd
from scipy.io import readsav
import os,sys,pickle,pdb

'''
    This python file contains functions for dealing with NRM mask files
'''

##THESE ARE THE VARIABLES IN THE MASK FILES FROM IDL, UNPACKED INTO THE PYTHON CLASS
##dict_keys(['mf_pvct', 'mf_gvct', 'mf_ix', 'mf_rmat', 'mf_imat', 'bl2h_ix', 'h2bl_ix', 'bl2bs_ix', 
#'bs2bl_ix', 'bscov2bs_ix', 'u', 'v', 'filter', 'n_holes', 'n_baselines', 'n_bispect', 'n_cov', 
#'hole_diam', 'rad_pixel'])

class NRM_mask:
    def __init__(self,label):
        '''
        Provide the label, which should be the FILTER header item from the original NIRC2 file
        (For masking this looks like 'CH4_short + 9holeMsk'
        '''
        self.id= label  
        
    def find_maskfile(self,templatelocation='nrm_mask_templates/nirc2/'):
        '''
        This function takes the above label, which is the FILTER header item from the original nirc2 data files
        and use a lookup table to find the appropriate mask template file
        This calls lookup_maskfile below
        '''
        mftemplatefile = lookup_maskfile(self.id,templatelocation=templatelocation)
        if mftemplatefile != '-1' : 
            print('Mask file found')
            self.maskfile=mftemplatefile
        else: print('Mask file not found')
     
    def read_IDL_maskfile(self,filename=''):
        '''
            This functions will read an idl mask file and unpack it into class attributes
            by default will use self.maskfile as the file to read, otherwise if 
            filename variable is provided it will use that.
        '''
        
        if filename =='':
            if hasattr(self,'maskfile') == True:
                filename=self.maskfile
            else: 
                print('No mask file attribute or input filename, returning')
                return 
        if os.path.isfile(filename) == False:
            print('Mask file provided does not exist!')
            return
        
        maskinfo = readsav(filename)    
        for nn,vv in maskinfo.items(): setattr(self,nn,vv)


def lookup_maskfile(filterstring,templatelocation='nrm_mask_templates/nirc2/'):
    '''
    This function will find a mask template file based on the filter header string in the nirc2 data
    (looks like 'CH4_short + 9holeMsk'), by reading a lookup table and searching templatelocation for the appropriate file
    '''
    
    if templatelocation[-1] != '/': templatelocation += '/'
    
    mflookup = pd.read_csv('mask_template_lookup.txt',delimiter=',').to_records()
    
    ##Do some error handling for different fail cases
    ##lookuptable entry mismatch
    try:
        mftemplatefile = templatelocation+mflookup[np.where(mflookup.filter == filterstring)[0]].templatefile[0]
    except: 
        print('Mask setup string' + filterstring +' not in lookup table:')
        print('mask_template_lookup.txt')
        print('Either figure out whats wrong with your filter setup, or add a new lookup entry')
        return '-1'
    ##Mask file not in expected place, but lookup entry present
    if os.path.isfile(mftemplatefile) == False:
        print('Mask file ')
        print(mftemplatefile)
        print('Does not exist in templatelocation directory but is in lookup table, figure out why')
        return '-1'
    ##ALL passed
    return mftemplatefile