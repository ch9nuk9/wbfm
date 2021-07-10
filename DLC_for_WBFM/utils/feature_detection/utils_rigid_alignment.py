# Based off code by Lukas Hille
# See originals in: Y:\shared_projects\notebooks\LukasNotebooks
import cv2
import numpy as np
from tqdm import tqdm


##
## Preprocessing
##

def filter_image(img_to_filter, high_freq, low_freq):
    # get Image size
    (x, y) = img_to_filter.shape
    # get image avarage
    # get some Background aproximation:
    min, max, minPt, maxPt = cv2.minMaxLoc(img_to_filter)
    avr = min
    if x > y:
        tmp_img = np.full((x, x), avr)
        tmp_img[:,int((x-y)/2):y+int((x-y)/2)] = img_to_filter
        size = x
    elif y > x:
        tmp_img = np.full((y, y), avr)
        tmp_img[int((y-x)/2):x+int((y-x)/2)] = img_to_filter
        size = y
    else:
        tmp_img = img_to_filter
        size = x
    # set filter values
    rhigh = high_freq # how narrower the window is
    rlow = low_freq # how broad the window is
    # function to generate the mask (window)
    ham = np.hamming(size)[:,None] # 1D hamming
    ham2dhigh = np.sqrt(np.dot(ham, ham.T)) ** rhigh # expand to 2D hamming
    ham2dlow = np.sqrt(np.dot(ham, ham.T)) ** rlow
    ham2d = ham2dhigh - ham2dlow
    # the mask determines the filter in frequency space.
    # with different filter regular image distorgen can be removed
    # see fourier space filters
    # calculate fourier transform
    f = cv2.dft(tmp_img.astype('float32'), flags=cv2.DFT_COMPLEX_OUTPUT)
    # reorder result quarters (the 4 quardrants are in wrong order)
    f_shifted = np.fft.fftshift(f)
    f_complex = f_shifted[:,:,0]*1j + f_shifted[:,:,1]
    # apply filter mask
    f_filtered = ham2d * f_complex
    # reorder result
    f_filtered_shifted = np.fft.fftshift(f_filtered)
    # inverse F.T.
    inv_img = np.fft.ifft2(f_filtered_shifted)
    # take absolut values
    filtered_img = np.abs(inv_img)
    # return result
    if x > y:
        return filtered_img[:,int((x-y)/2):y+int((x-y)/2)]
    if y > x:
        return filtered_img[int((y-x)/2):x+int((y-x)/2)]

    filtered_img = cv2.bilateralFilter(filtered_img.astype('float32'), d=9, sigmaColor=50, sigmaSpace=3).astype('uint8')
    return filtered_img


def filter_stack(stack_to_align, filter_opt=None):
    # First, filter all planes
    if filter_opt is None:
        filter_opt = {'high_freq': 2.0, 'low_freq': 5000.0}
    filtered_stack = np.array(stack_to_align)
    for i in range(stack_to_align.shape[0]):
        filtered_stack[i, ...] = filter_image(filtered_stack[i, ...], **filter_opt)
    return filtered_stack


##
## Utilities
##

# this function calculates the overall transformation of two given warp matrix
# the warp matrix which combines both given warp matrix is given back
def merge_matrices(mat1, mat2):
    # creaty identity matrix of size 3x3
    firstMul = np.identity(3)
    secondMul = np.identity(3)
    # copy the given matrix into the 3x3 matrix
    # it is necessary to distinguishe by size
    (x, y) = mat1.shape
    affine = True
    if x < 3:
        firstMul[0:2,:] = mat1
    else:
        affine = False
        firstMul = mat1
    (x, y) = mat2.shape
    if x < 3:
        secondMul[0:2,:] = mat2
    else:
        affine = False
        firstMul = mat2
    # calculate the dot product of mat1 and mat2
    mat_combine = secondMul.dot(firstMul)
    # return the affine matrix
    if affine:
        return mat_combine[0:2,:]
    # return the homography matrix
    return mat_combine


def get_warp_mat(im_prev, im_next, warp_mat):
    warp_new = calc_warp_ECC(im_prev, im_next)
    # if warp_new is None:
    #     print(f"Skipping plane {i}, error {error_mode}")
    #     warp_new = np.identity(3)[0:2,:]

    return merge_matrices(warp_mat, warp_new)

##
## Alignment
##

def calc_warp_ECC(im1_gray, im2_gray, warp_mode=cv2.MOTION_EUCLIDEAN,
                  termination_eps=2e-2,
                  number_of_iterations=10000):
    # prefilter Images, helps to correlate
    # for best performance a filter which keeps good features to align should be used.
    # for example: with strong differences in brightnes but identical edges an edge dector can be used to filter.

    # Find size of image1
    sz = im1_gray.shape

    # Define the motion model (more freedom needs more computational time)
    # Translation, Euclidean, Affine, Homography
    # given by Parameter
    # warp_mode = cv2.MOTION_EUCLIDEAN
    # (as function parameter)

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    # number_of_iterations = 10000

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    # (precision vs computational time)
    # termination_eps = 1e-4;
    # (as function parameter)

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    # Depends on version of cv2
    try:
        (cc, warp_matrix) = cv2.findTransformECC(im1_gray.astype('float32'), im2_gray.astype('float32'), warp_matrix, warp_mode, criteria, None)
    except TypeError:
        (cc, warp_matrix) = cv2.findTransformECC(im1_gray.astype('float32'), im2_gray.astype('float32'), warp_matrix, warp_mode, criteria, inputMask=None, gaussFiltSize=1)

    # example to execute the transformation
    #if warp_mode == cv2.MOTION_HOMOGRAPHY :
        # Use warpPerspective for Homography
        #im2_aligned = cv2.warpPerspective (im2_gray, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    #else :
        # Use warpAffine for Translation, Euclidean and Affine
        #im2_aligned = cv2.warpAffine(im2_gray, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    return warp_matrix


def align_stack(stack_to_align, hide_progress=True):
    """
    Takes a z stack (format: ZXY) and rigidly aligns planes sequentially
    """
    # Settings for the actual warping
    sz = stack_to_align[0].shape
    sz = (sz[1], sz[0])
    flags = cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
    # align volumes, starting from the center
    i_center_plane = int(stack_to_align.shape[0]/2)
    stack_aligned = np.empty_like(stack_to_align)
    stack_aligned[i_center_plane] = stack_to_align[i_center_plane]
    warp_matrices = {}  # (i_prev, i_next) -> matrix

    # From i_center_plane to 0
    # Calculates the matrix per plane, and cumulatively multiplies them
    warp_mat = np.identity(3)[0:2, :]
    for i in tqdm(range(i_center_plane, 0, -1), disable=hide_progress):
        im_next, im_prev = stack_to_align[i-1], stack_to_align[i]
        warp_mat = get_warp_mat(im_prev, im_next, warp_mat)
        warp_matrices[(i, i-1)] = warp_mat.copy()
        stack_aligned[i-1] = cv2.warpAffine(im_next, warp_mat, sz, flags=flags)

    # From i_center_plane to end (usually 33)
    warp_mat = np.identity(3)[0:2, :]
    for i in tqdm(range(i_center_plane, (stack_to_align.shape[0]-1)), disable=hide_progress):
        im_prev, im_next = stack_to_align[i], stack_to_align[i+1]
        warp_mat = get_warp_mat(im_prev, im_next, warp_mat)
        warp_matrices[(i, i+1)] = warp_mat.copy()
        stack_aligned[i+1] = cv2.warpAffine(im_next, warp_mat, sz, flags=flags)

    return stack_aligned, warp_matrices


def align_stack_using_previous_results(stack_to_align, previous_warp_matrices, hide_progress=True):
    """
    Takes a z stack (zxy) and a dictionary of previous alignment matrices, and performs the same alignment
    """
    # Settings for the actual warping
    sz = stack_to_align[0].shape
    sz = (sz[1], sz[0])
    flags = cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
    # align volumes, starting from the center
    i_center_plane = int(stack_to_align.shape[0]/2)
    stack_aligned = np.empty_like(stack_to_align)
    stack_aligned[i_center_plane] = stack_to_align[i_center_plane]

    # From i_center_plane to 0
    # Calculates the matrix per plane, and cumulatively multiplies them
    for i in tqdm(range(i_center_plane, 0, -1), disable=hide_progress):
        im_next = stack_to_align[i-1]
        warp_mat = previous_warp_matrices[(i, i-1)]
        stack_aligned[i-1] = cv2.warpAffine(im_next, warp_mat, sz, flags=flags)

    # From i_center_plane to end (usually 33)
    for i in tqdm(range(i_center_plane, (stack_to_align.shape[0]-1)), disable=hide_progress):
        im_next = stack_to_align[i+1]
        warp_mat = previous_warp_matrices[(i, i+1)]
        stack_aligned[i+1] = cv2.warpAffine(im_next, warp_mat, sz, flags=flags)

    return stack_aligned
