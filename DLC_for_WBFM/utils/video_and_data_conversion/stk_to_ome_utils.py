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

    ##
    ## First, set up the ome-tiff template
    ##
    ome_pixels_dict = {}

    def get_full_ome_dict(ome_pixels_dict):
        return {'Instrument':'',
                'Image': {'Pixels':ome_pixels_dict},
                'Creator':'',
                'UUID':''}


    ##
    ## Second, deal with fields from the stk file
    ##

    # Dictionary to translate field names
    stk_to_ome = {'NumberPlanes' : 'SizeZ',
                  'XCalibration' : 'PhysicalSizeX',
                  'YCalibration' : 'PhysicalSizeY',
                  'CalibrationUnits' : 'PhysicalSizeXUnit',
                  '' : ''}
    # Some fields are doubled
    stk_to_ome2 = {'CalibrationUnits' : 'PhysicalSizeYUnit'}

    # Get the actual stk metadata xml
    stk_meta = tifffile.TiffFile(data_file, multifile=False).stk_metadata
