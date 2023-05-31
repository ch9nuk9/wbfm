# Based off code by Lukas Hille
# See originals in: Y:\shared_projects\notebooks\LukasNotebooks
import cv2
import numpy as np
from tqdm.auto import tqdm

##
## Utilities
##


def merge_matrices(mat1, mat2):
    # this function calculates the overall transformation of two given warp matrix
    # the warp matrix which combines both given warp matrix is given back

    # creaty identity matrix of size 3x3
    firstMul = np.identity(3)
    secondMul = np.identity(3)
    # copy the given matrix into the 3x3 matrix
    # it is necessary to distinguishe by size
    (x, y) = mat1.shape
    affine = True
    if x < 3:
        firstMul[0:2, :] = mat1
    else:
        affine = False
        firstMul = mat1
    (x, y) = mat2.shape
    if x < 3:
        secondMul[0:2, :] = mat2
    else:
        affine = False
        firstMul = mat2
    # calculate the dot product of mat1 and mat2
    mat_combine = secondMul.dot(firstMul)
    # return the affine matrix
    if affine:
        return mat_combine[0:2, :]
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
                  number_of_iterations=10000,
                  gauss_filt_sigma=None):
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
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    # Depends on version of cv2
    try:
        if gauss_filt_sigma is not None:
            kernel_sz = (5, 5)
            im1_gray = cv2.GaussianBlur(im1_gray.astype('float32'), kernel_sz, gauss_filt_sigma)
        (cc, warp_matrix) = cv2.findTransformECC(im1_gray.astype('float32'), im2_gray.astype('float32'), warp_matrix,
                                                 warp_mode, criteria, None)
    except TypeError:
        (cc, warp_matrix) = cv2.findTransformECC(im1_gray.astype('float32'), im2_gray.astype('float32'), warp_matrix,
                                                 warp_mode, criteria, inputMask=None, gaussFiltSize=1)

    # example to execute the transformation
    # if warp_mode == cv2.MOTION_HOMOGRAPHY :
    # Use warpPerspective for Homography
    # im2_aligned = cv2.warpPerspective (im2_gray, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    # else :
    # Use warpAffine for Translation, Euclidean and Affine
    # im2_aligned = cv2.warpAffine(im2_gray, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    return warp_matrix


def align_stack_to_middle_slice(stack_to_align, hide_progress=True):
    """
    Takes a z stack (format: ZXY) and rigidly aligns planes sequentially
    """
    # Settings for the actual warping
    sz = stack_to_align[0].shape
    sz = (sz[1], sz[0])
    flags = cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
    # align volumes, starting from the center
    i_center_plane = int(stack_to_align.shape[0] / 2)
    stack_aligned = np.empty_like(stack_to_align)
    stack_aligned[i_center_plane] = stack_to_align[i_center_plane]
    warp_matrices = {}  # (i_prev, i_next) -> matrix

    # From i_center_plane to 0
    # Calculates the matrix per plane, and cumulatively multiplies them
    warp_mat = np.identity(3)[0:2, :]
    for i in tqdm(range(i_center_plane, 0, -1), disable=hide_progress):
        im_next, im_prev = stack_to_align[i - 1], stack_to_align[i]
        warp_mat = get_warp_mat(im_prev, im_next, warp_mat)
        warp_matrices[(i, i - 1)] = warp_mat.copy()
        stack_aligned[i - 1] = cv2.warpAffine(im_next, warp_mat, sz, flags=flags)

    # From i_center_plane to end (usually 33)
    warp_mat = np.identity(3)[0:2, :]
    for i in tqdm(range(i_center_plane, (stack_to_align.shape[0] - 1)), disable=hide_progress):
        im_prev, im_next = stack_to_align[i], stack_to_align[i + 1]
        warp_mat = get_warp_mat(im_prev, im_next, warp_mat)
        warp_matrices[(i, i + 1)] = warp_mat.copy()
        stack_aligned[i + 1] = cv2.warpAffine(im_next, warp_mat, sz, flags=flags)

    return stack_aligned, warp_matrices


def calculate_alignment_matrix_two_stacks(stack_template: np.ndarray, stack_rotated: np.ndarray, hide_progress=True,
                                          use_only_first_pair=True, gauss_filt_sigma=2.5) -> np.ndarray:
    """
    Takes two z stacks (format: ZXY) and rigidly aligns plane, returning only the warp matrices

    Parameters
    ----------
    stack_template: stack to align to
    stack_rotated: stack to align
    hide_progress
    use_only_first_pair: flag for using the first pair only (stacks might be full videos)
    gauss_filt_sigma: sigma for gaussian filter to apply to images before alignment

    Returns
    -------

    """
    warp_matrices = []

    warp_mat = np.identity(3)[0:2, :]
    for i, (im0, im1) in enumerate(tqdm(zip(stack_template, stack_rotated), disable=hide_progress)):
        warp_mat = calc_warp_ECC(im0, im1, termination_eps=1e-6, warp_mode=cv2.MOTION_AFFINE,
                                 gauss_filt_sigma=gauss_filt_sigma)
        if use_only_first_pair:
            break
        else:
            warp_matrices.append(warp_mat)

    if use_only_first_pair:
        final_warp_mat = warp_mat
    else:
        final_warp_mat = np.mean(np.array(warp_matrices), axis=0)

    return final_warp_mat


def apply_alignment_matrix_to_stack(stack_to_align: np.ndarray, warp_mat: np.ndarray, hide_progress=True):
    """
    Takes a z stack (zxy) and a single previous alignment matrix, and performs the same alignment

    Parameters
    ----------
    stack_to_align
    warp_mat
    hide_progress

    Returns
    -------

    """
    # Settings for the actual warping
    sz = stack_to_align[0].shape
    sz = (sz[1], sz[0])
    flags = cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
    stack_aligned = np.empty_like(stack_to_align)

    for i_final, i_initial in enumerate(tqdm(range(stack_to_align.shape[0]), disable=hide_progress, leave=False)):
        img = stack_to_align[i_initial, ...]
        stack_aligned[i_final, ...] = cv2.warpAffine(img, warp_mat, sz, flags=flags)

    return stack_aligned


def cumulative_alignment_of_stack(stack_to_align: np.ndarray, previous_warp_matrices: dict, hide_progress=True):
    """
    Takes a z stack (zxy) and a dictionary of previous alignment matrices (indexed by z slice), and performs the same
    alignment

    Cumulative version of apply_alignment_matrix_to_stack, counting out from the center plane

    Parameters
    ----------
    stack_to_align
    previous_warp_matrices
    hide_progress

    Returns
    -------

    """
    # Settings for the actual warping
    sz = stack_to_align[0].shape
    sz = (sz[1], sz[0])
    flags = cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
    # align volumes, starting from the center
    i_center_plane = int(stack_to_align.shape[0] / 2)
    stack_aligned = np.empty_like(stack_to_align)
    stack_aligned[i_center_plane] = stack_to_align[i_center_plane]

    # From i_center_plane to 0
    # Calculates the matrix per plane, and cumulatively multiplies them
    for i in tqdm(range(i_center_plane, 0, -1), disable=hide_progress):
        im_next = stack_to_align[i - 1]
        warp_mat = previous_warp_matrices[(i, i - 1)]
        stack_aligned[i - 1] = cv2.warpAffine(im_next, warp_mat, sz, flags=flags)

    # From i_center_plane to end (usually 33)
    for i in tqdm(range(i_center_plane, (stack_to_align.shape[0] - 1)), disable=hide_progress):
        im_next = stack_to_align[i + 1]
        warp_mat = previous_warp_matrices[(i, i + 1)]
        stack_aligned[i + 1] = cv2.warpAffine(im_next, warp_mat, sz, flags=flags)

    return stack_aligned
