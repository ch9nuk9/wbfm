import numpy as np

def colorize_frame(frame, cmap, z_val, threshold_val=None):
    """
    Colors a frame according a colormap, separated by a threshold
    
    This will REMOVE information related to intensity (for now), and replace everything with cmap(z_val)
    """
    
    if threshold_val is None:
        threshold_val = np.median(frame)

    sz = frame.shape
    new_frame = np.zeros(sz[0:2] + (4,), dtype='uint8')
    mask = frame > threshold_val
    #mask_ind = np.where(mask)
    new_val = (np.array(cmap(z_val))*255).astype('uint8') # Everything else is zeros
    new_frame[mask] = new_val
    
    return new_frame, threshold_val

####
####
####
def colorize_stack(stack, cmap, threshold_val=None):
    """
    Colors a stack by taking a maximum projection, and then colorizing each entry according to the original z-stack
    If the intensity is less than a threshold, it will be set to 0
    
    This will REMOVE information related to intensity (for now), and replace everything with cmap(z_val)
    """
    #if threshold_val is None:
     #   threshold_val = np.max(stack[0,...])/1.1

    sz = stack.shape
    #print(sz)
    new_frame = np.zeros(sz[0:2] + (4,), dtype='uint8')
    #print(new_frame.shape)
    
    # First, do a max projection
    max_proj = np.max(stack, axis=-1)
    max_proj_ind = np.argmax(stack, axis=-1)
    #print(max_proj)
    #print(max_proj_ind)
    #print(max_proj_ind.shape)
    if threshold_val is None:
        threshold_val = 0.1*np.max(max_proj)
    
    # Colorize the projection
    for i_z in range(sz[-1]):
        mask_val = max_proj > threshold_val
        mask_z = i_z == max_proj_ind
        mask = mask_val & mask_z #np.where(mask_val & mask_z)
        #print(mask.shape)
        new_val = (np.array(cmap(i_z/sz[-1]))*255).astype('uint8') # Everything else is zeros
        new_frame[mask] = new_val
    
    #new_frame[:,:,-1] = max_proj.astype('uint8') # Replace alpha with intensity information
    
    return new_frame, threshold_val


####
####
####
def colorize_stack_by_channels(stack, threshold_val=None):
    """
    Colors a stack by taking a maximum projection, and then colorizing the RGB channels according thirds of the stack
    i.e. Red channel = max(stack[:,:,0:10]) etc.
    
    This will REMOVE information related to intensity (for now), and replace everything with cmap(z_val)
    """
    if threshold_val is None:
        threshold_val = np.max(stack[0,...])/1.1

    sz = stack.shape
    new_frame = 255*np.ones(sz[0:2] + (4,), dtype='uint8')
    #print(new_frame.shape)
    #print(stack.shape)
    
    # First, do 3 max projections
    max_proj0 = np.max(stack[...,0:10], axis=-1)
    #max_proj_ind1 = np.argmax(stack[0:10], axis=-1)
    max_proj1 = np.max(stack[...,11:20], axis=-1)
    #max_proj_ind2 = np.argmax(stack[11:20], axis=-1)
    max_proj2 = np.max(stack[...,21:30], axis=-1)
    #max_proj_ind3 = np.argmax(stack[21:30], axis=-1)
    #print(max_proj0.shape)
    
    # Colorize each channel
    new_frame[:,:,0] = max_proj0.astype('uint8')
    new_frame[:,:,1] = max_proj1.astype('uint8')
    new_frame[:,:,2] = max_proj2.astype('uint8')
    
    return new_frame, threshold_val



def colorize_3planes_keeping_intensity(stack, planes, alpha=1.0):
    """
    Colors 3 planes by mapping each to a color channel : RGB
    
    Expected input format: TZXY
    Outputs: TCXY
    """
#     if threshold_val is None:
#         threshold_val = np.max(stack[0,...])/1.1

#     sz = stack.shape
#     new_frame = np.ones((sz[0],) + (4,) + sz[1:-1], dtype='uint8')
    
    # Get the three planes
    all_slices = stack[:,planes[0]:(planes[-1]+1),...] * alpha
#     slice0 = stack[:,planes[0],...] * alpha
#     slice1 = stack[:,planes[1],...] * alpha
#     slice2 = stack[:,planes[2],...] * alpha
    
    #print(max_proj0.shape)
    
    # Colorize each channel
    colored_slices = all_slices
#     new_frame[:,0,...] = slice0.astype('uint8')
#     new_frame[:,1,...] = slice1.astype('uint8')
#     new_frame[:,2,...] = slice2.astype('uint8')
    
    return colored_slices