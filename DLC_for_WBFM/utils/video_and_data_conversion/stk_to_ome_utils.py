import numpy as np
import tifffile


##
## Creating ome metadata
##

def ome_metadata_from_array_and_stk(stk_full_fname, dat_sz):
    """
    Writes an xml file for use as metadata in an ome-tiff
    
    Parameters
    ----------
    stk_full_fname : full filename of one of the original stk files that is being converted
    dat_sz : size of the final array, in XYCZT
    
    """