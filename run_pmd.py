#!/usr/bin/env python
# Adapted from "demos/Matrix_Decomposition/Demo PMD Compression & Denoising.ipynb", for use in Docker
# General Dependencies
import time
import os
import numpy as np
import h5py
#from scipy.io import loadmat, savemat
import argparse
import sys
import pickle

#Preprocessing Dependencies
from trefide.utils import psd_noise_estimate

# PMD Model Dependencies
from trefide.preprocess import detrend, flag_outliers
from trefide.pmd import batch_decompose
from trefide.pmd import batch_recompose
from trefide.pmd import overlapping_batch_decompose
from trefide.pmd import overlapping_batch_recompose
from trefide.pmd import determine_thresholds
from trefide.reformat import overlapping_component_reformat

# Plotting & Video Rendering Dependencies
import matplotlib.pyplot as plt
#from trefide.plot import pixelwise_ranks
#from trefide.extras.util_plot import comparison_plot
#from trefide.video import play_cv2

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--do-detrend', action='store_true', help='set this flag to run the detrending first')
    p.add_argument('--infile', required=True, help='Input movie *.mat. Must contain a field "data"')
    p.add_argument('--consec-failures', type=int, required=True, help='consecutive failures. More means more conservative, retaining a higher rank matrix')
    p.add_argument('--max-components', type=int, required=True, help='')
    p.add_argument('--max-iters-main', type=int, required=True, help='')
    p.add_argument('--max-iters-init', type=int, required=True, help='')
    p.add_argument('--block-width', type=int, required=True, help='Note - FOV width must be divisble by block width')
    p.add_argument('--block-height', type=int, required=True, help='Note - FOV height must be divisible by block height')
    p.add_argument('--d-sub', type=int, required=True, help='')
    p.add_argument('--t-sub', type=int, required=True, help='Note - temporal dimension must be divisible by t_sub')
    args = p.parse_args()

    time_info = {}
    intermediate_outputs = {}
    intermediate_outputs['args'] = args

    print("Begin with arguments: ", args)
    t00 = time.time()
    print("loading movie: {}...".format(args.infile))

    # NOTE - matlab produces a fortran-contiguous array (column major order)
    # Failure to fix the array memory layout results in:
    #Traceback (most recent call last):
    #  File "/usr/local/envs/trefide/lib/python3.6/site-packages/run_pmd.py", line 131, in <module>
    #    main()
    #  File "/usr/local/envs/trefide/lib/python3.6/site-packages/run_pmd.py", line 108, in main
    #    d_sub=d_sub, t_sub=t_sub)
    #  File "trefide/pmd.pyx", line 448, in trefide.pmd.overlapping_batch_decompose
    #    double[:, :, ::1] Y, 
    #  File "stringsource", line 654, in View.MemoryView.memoryview_cwrapper
    #  File "stringsource", line 349, in View.MemoryView.memoryview.__cinit__
    #ValueError: ndarray is not C-contiguous

    # NOTE - having an empty dimension somewhere seems to be the cause of the following error:
    #   BufferError: memoryview: underlying buffer is not C-contiguous
    # This occurs when the block size and the FOV size are the exact same (meaning no tiling is happening?). 
    # Probably setting overlapping=false might also help this

#    mov = np.ascontiguousarray(loadmat("/input/{}".format(args.infile))['inputData'], dtype='double') # TODO - hardcoded path within container
    t0 = time.time()
    with h5py.File("/input/{}".format(args.infile), 'r') as f:
        # Based on how MATLAB writes the file and how numpy reads it, we get reversed indices.
        # By reading it with  order='F'  , we get an array that is F_CONTIGUOUS = true
        # Then, after transposing, we have the desired order of indices and C_CONTIGUOUS = true
        mov = np.ascontiguousarray(f['inputData'].value.transpose([2,1,0])).astype('double')
        stim = np.ascontiguousarray(f['stim']).squeeze().astype('double')
        print("mov.shape: ", mov.shape)
        print("mov.flags: ", mov.flags)
        print("mov.dtype: ", mov.dtype)
        print("stim.shape: ", stim.shape)
        print("stim.flags: ", stim.flags)
        print("stim.dtype: ", stim.dtype)
        print("done")
    time_info['load_input'] = time.time() - t0

    fov_height, fov_width, num_frames = mov.shape
    print("fov_height: {}".format(fov_height))
    print("fov_width: {}".format(fov_width))
    intermediate_outputs['fov_height'] = fov_height
    intermediate_outputs['fov_width'] = fov_width
    intermediate_outputs['num_frames'] = num_frames

    
    if args.do_detrend:
        print("detrend movie")
        t0 = time.time()
        # TODO - unclear what "signal" is supposed to be?
        # should the "del_idx" return value from flag_outliers() be used also?
        # TODO - disc_idx is not working:
	#Traceback (most recent call last):
	#  File "/usr/local/envs/trefide/lib/python3.6/site-packages/run_pmd.py", line 214, in <module>
	#    main()
	#  File "/usr/local/envs/trefide/lib/python3.6/site-packages/run_pmd.py", line 111, in main
	#    disc_idx=disc_idx)
	#  File "/usr/local/envs/trefide/lib/python3.6/site-packages/trefide/preprocess.py", line 234, in detrend
	#    disc_idx[1:] = disc_idx[1:] - np.cumsum(np.ones(len(disc_idx) - 1) * 3)
	#ValueError: could not broadcast input array from shape (4,4) into shape (4,1)

        _, disc_idx = flag_outliers(signal=np.mean(mov, axis=(0,1))) 
        print(f"disc_idx.shape: {disc_idx.shape}")
        print(f"disc_idx.flags: {disc_idx.flags}")
        print(f"disc_idx.dtype: {disc_idx.dtype}")
        print(f"disc_idx: {disc_idx}")
        # TODO - any interest in the other return values from detrend()?
        # in particular, disc_idx gets returned with more problem areas - shouldn't these be computed in advance, before placing spline knots??
        t0 = time.time()
        mov, _, _, _ = detrend(mov=mov,
                               stim=stim,
                               disc_idx=disc_idx.squeeze())
        print(f"mov.shape: {mov.shape}")
        print(f"mov.flags: {mov.flags}")
        print(f"mov.dtype: {mov.dtype}")
        mov = np.ascontiguousarray(mov)
        time_info['detrend'] = time.time() - t0

    # Generous maximum of rank 50 blocks (safeguard to terminate early if this is hit)
    #max_components = 50

    # Enable Decimation 
    #max_iters_main = 10
    #max_iters_init = 40
    #d_sub=2
    #t_sub=2

    # Defaults
    #consec_failures = 3
    tol = 5e-3

    # Set Blocksize Parameters
    #block_height = 40
    #block_width = 40
    overlapping = True

    # Compress Video
    ## Simulate Critical Region Using Noise
    print("determine_thresholds...")
    t0 = time.time()
    spatial_thresh, temporal_thresh = determine_thresholds((fov_height, fov_width, num_frames),
                                                           (args.block_height, args.block_width),
                                                           args.consec_failures, args.max_iters_main, 
                                                           args.max_iters_init, tol, 
                                                           args.d_sub, args.t_sub, 5, True)
    time_info['determine_thresholds'] = time.time() - t0
    intermediate_outputs['spatial_thresh'] = spatial_thresh
    intermediate_outputs['temporal_thresh'] = temporal_thresh
    print("done")

    ## Decompose Each Block Into Spatial & Temporal Components

    # Blockwise Parallel, 4x Overlapping Tiling
    print("overlapping_batch_decompose...")
    t0 = time.time()
    spatial_components, temporal_components, block_ranks, block_indices, block_weights = overlapping_batch_decompose(
            fov_height, fov_width, num_frames,
            mov, args.block_height, args.block_width,
            spatial_thresh, temporal_thresh,
            args.max_components, args.consec_failures,
            args.max_iters_main, args.max_iters_init, tol,
            d_sub=args.d_sub, t_sub=args.t_sub)
    time_info['overlapping_batch_decompose'] = time.time() - t0
    intermediate_outputs['spatial_components'] = spatial_components
    intermediate_outputs['temporal_components'] = temporal_components
    intermediate_outputs['block_ranks'] = block_ranks
    intermediate_outputs['block_indices'] = block_indices
    intermediate_outputs['block_weights'] = block_weights
    print("done")


    print("overlapping_component_reformat...")
    t0 = time.time()
    U, V = overlapping_component_reformat(fov_height, fov_width, num_frames,
                                          args.block_height, args.block_width,
                                          spatial_components,
                                          temporal_components,
                                          block_ranks,
                                          block_indices,
                                          block_weights)
    time_info['overlapping_component_reformat'] = time.time() - t0
    print("done")

    #savemat(file_name='/output/pmd-output.mat', mdict={'U':U, 'V':V})

    # TODO - lots of changes in ordering. Could change conventions inside trefide and avoid this work
    #        though cost is probably relatively quite small
    print("U.shape: ", U.shape)
    print("U.flags: ", U.flags)
    print("U.dtype: ", U.dtype)
    print()
    print("V.shape: ", V.shape)
    print("V.flags: ", V.flags)
    print("V.dtype: ", V.dtype)
    t0 = time.time()
    with h5py.File('/output/pmd-output.h5', 'w') as outfile:
        outfile['U'] = U.transpose([2,1,0])
        outfile['V'] = V.transpose([1,0])
    time_info['save_output'] = time.time() - t0

    time_info['total_duration'] = time.time() - t00

    # TODO - hardcoded paths
    with open('/output/pmd-time-info.pkl', 'wb') as timefile:
        pickle.dump(time_info, timefile)
    # Can't pickle intermediate_outputs - seems to be lack of __reduce__ method for some of the Cython objects?
    # Traceback (most recent call last):
    #  File "/usr/local/envs/trefide/lib/python3.6/site-packages/run_pmd.py", line 184, in <module>
    #    main()
    #  File "/usr/local/envs/trefide/lib/python3.6/site-packages/run_pmd.py", line 180, in main
    #    pickle.dump(intermediate_outputs, intermed_file)
    #  File "stringsource", line 2, in View.MemoryView._memoryviewslice.__reduce_cython__
    #TypeError: no default __reduce__ due to non-trivial __cinit__ 
    #with open('/output/intermediate-outputs.pkl', 'wb') as intermed_file:
    #    pickle.dump(intermediate_outputs, intermed_file)
    print("done")

if __name__ == "__main__":
    main()
