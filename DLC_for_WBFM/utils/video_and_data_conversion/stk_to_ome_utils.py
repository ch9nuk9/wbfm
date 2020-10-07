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
    def get_full_ome_dict(ome_pixels_dict):
        return {'Instrument':'',
                'Image': {'Pixels':ome_pixels_dict},
                'Creator':'',
                'UUID':''}

    ##
    ## Second, deal with fields from the stk file
    ##

    # Dictionary to translate field names
    stk_to_ome_strings = {'SizeZ': 'NumberPlanes',
                          'PhysicalSizeX' : 'XCalibration',
                          'PhysicalSizeY' : 'YCalibration',
                          'PhysicalSizeXUnit' : 'CalibrationUnits',
                          'PhysicalSizeYUnit' : 'CalibrationUnits',
                          'PhysicalSizeZUnit' : 'CalibrationUnits'}

    # Get the actual stk metadata xml
    stk_meta = tifffile.TiffFile(stk_full_fname, multifile=False).stk_metadata

    # Convert above strings to value using the stk_meta
    stk_to_ome = {}
    for key, value in stk_to_ome_strings.items():
        stk_to_ome[key] = stk_meta[value]

    ##
    ## Third, deal with fields from the final array size
    ##
    array_to_ome = {'SizeX' : dat_sz[2],
                    'SizeY' : dat_sz[3],
                    'SizeT' : dat_sz[0],
                    'Type' : 'uint16'}

    ##
    ## Finally, put together the inner dict and return
    ##
    merged_tmp = {**stk_to_ome, **array_to_ome}
    return get_full_ome_dict(merged_tmp)
