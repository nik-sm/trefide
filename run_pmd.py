#!/usr/bin/env python
# Adapted from "demos/Matrix_Decomposition/Demo PMD Compression & Denoising.ipynb", for use in Docker
# General Dependencies
import os
import numpy as np
from scipy.io import loadmat, savemat
import argparse
import sys

#Preprocessing Dependencies
from trefide.utils import psd_noise_estimate

# PMD Model Dependencies
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', required=True, help='Input movie *.mat. Must contain a field "data"')
    parser.add_argument('--consec-failures', type=int, required=True, help='consecutive failures. More means more conservative, retaining a higher rank matrix')
    parser.add_argument('--max-components', type=int, required=True, help='')
    parser.add_argument('--max-iters-main', type=int, required=True, help='')
    parser.add_argument('--max-iters-init', type=int, required=True, help='')
    parser.add_argument('--block-width', type=int, required=True, help='')
    parser.add_argument('--block-height', type=int, required=True, help='')
    parser.add_argument('--d-sub', type=int, required=True, help='')
    parser.add_argument('--t-sub', type=int, required=True, help='')

    args = parser.parse_args()
    print("Begin with arguments: ", args)

    print("loading movie: {}...".format(args.infile))

    # TODO - justify ascontiguousarray or take it out
    # TODO - justify choice of dtype='double' ?
    mov = np.ascontiguousarray(loadmat("/input/{}".format(args.infile))['inputData'], dtype='double') # TODO - hardcoded path within container

    fov_height, fov_width, num_frames = mov.shape
    print("done")
    print("fov_height: {}".format(fov_height))
    print("fov_width: {}".format(fov_width))

    # Generous maximum of rank 50 blocks (safeguard to terminate early if this is hit)
    #max_components = 50
    max_components = args.max_components

    # Enable Decimation 
    #max_iters_main = 10
    #max_iters_init = 40
    #d_sub=2
    #t_sub=2
    max_iters_main = args.max_iters_main
    max_iters_init = args.max_iters_init
    d_sub=args.d_sub
    t_sub=args.t_sub

    # Defaults
    #consec_failures = 3
    consec_failures = args.consec_failures
    tol = 5e-3

    # Set Blocksize Parameters
    #block_height = 40
    #block_width = 40
    block_height = args.block_height
    block_width = args.block_width
    overlapping = True

    # Compress Video
    ## Simulate Critical Region Using Noise
    print("determine_thresholds...")
    spatial_thresh, temporal_thresh = determine_thresholds((fov_height, fov_width, num_frames),
                                                           (block_height, block_width),
                                                           consec_failures, max_iters_main, 
                                                           max_iters_init, tol, 
                                                           d_sub, t_sub, 5, True)
    print("done")

    ## Decompose Each Block Into Spatial & Temporal Components
    print("overlapping: {}".format(overlapping))


    # TODO - detrending before denoising, or begin with whole-frame bleach-correction
    if not overlapping:    # Blockwise Parallel, Single Tiling
        print("batch_decompose...")
        spatial_components, temporal_components, block_ranks, block_indices = batch_decompose(fov_height, fov_width, num_frames,
                                                                                              mov, block_height, block_width,
                                                                                              spatial_thresh, temporal_thresh,
                                                                                              max_components, consec_failures,
                                                                                              max_iters_main, max_iters_init, tol,
                                                                                              d_sub=d_sub, t_sub=t_sub)
        print("done")
    else:    # Blockwise Parallel, 4x Overlapping Tiling
        print("overlapping_batch_decompose...")
        spatial_components, temporal_components, block_ranks, block_indices, block_weights = overlapping_batch_decompose(fov_height, fov_width, num_frames,
                                                                                             mov, block_height, block_width,
                                                                                             spatial_thresh, temporal_thresh,
                                                                                             max_components, consec_failures,
                                                                                             max_iters_main, max_iters_init, tol,
                                                                                             d_sub=d_sub, t_sub=t_sub)
        print("done")

    # Reconstruct Denoised Video
    if not overlapping:  # Single Tiling (No need for reqweighting)
        print("batch_recompose...")
        mov_denoised = np.asarray(batch_recompose(spatial_components,
                                                  temporal_components,
                                                  block_ranks,
                                                  block_indices))
        print("done")
    else:   # Overlapping Tilings With Reweighting
        print("overlapping_batch_recompose...")
        mov_denoised = np.asarray(overlapping_batch_recompose(fov_height, fov_width, num_frames,
                                                              block_height, block_width,
                                                              spatial_components,
                                                              temporal_components,
                                                              block_ranks,
                                                              block_indices,
                                                              block_weights)) 
        print("done")

    # Produce Diagnostics
    ### Single Tiling Pixel-Wise Ranks
    #if overlapping:
    #    print("pixelwise_ranks...")
    #    pixelwise_ranks(block_ranks['no_skew']['full'], fov_height, fov_width, num_frames, block_height, block_width)
    #    print("done")
    #else:
    #    print("pixelwise_ranks (overlapping)...")
    #    pixelwise_ranks(block_ranks, fov_height, fov_width, num_frames, block_height, block_width)
    #    print("done")

    ### Correlation Images

    #print("comparison_plot")
    #comparison_plot([mov, mov_denoised + np.random.randn(np.prod(mov.shape)).reshape(mov.shape)*.01],
    #                plot_orientation="horizontal")

    # play_cv2 seems to be broken, crashes no matter what opencv python library is installed
    #print("play_cv2")
    #play_cv2(np.vstack([mov, mov_denoised, mov-mov_denoised]), magnification=2)

    print("overlapping_component_reformat...")
    U, V = overlapping_component_reformat(fov_height, fov_width, num_frames,
                                          block_height, block_width,
                                          spatial_components,
                                          temporal_components,
                                          block_ranks,
                                          block_indices,
                                          block_weights)
    print("done")

    #print("matmul to reconstruct Y...")
    #ushape = U.shape
    #vshape = V.shape
    ##savemat(file_name='/output/demo_results.UV.mat', mdict={'U':U, 'V':V})
    #print("U.shape:", ushape)
    #print("V.shape:", vshape)
    #U = np.reshape(U, (np.prod(ushape[:-1]),ushape[-1]))
    ## multiply the matrices, reshape back to a movie, and cast as uint16
    #Y = np.reshape(np.matmul(U, V), (*ushape[:-1], vshape[-1])).astype(np.uint16)
    #print("Y.shape:", Y.shape)
    #print("done")
    #print("np.save...")
    #np.savez(os.path.join(args.outdir, "demo_results.npz"), U, V)
    ##np.save("/output/demo_results.U.npy", U)
    ##np.save("/output/demo_results.V.npy", V)
    #print("scipy.io.savemat...")
    #savemat(file_name='/output/demo_results.Y.mat', mdict={'Y':Y, 'latent_dim':ushape[-1]})

    # TODO - can the latent dimension be read from the "spatial_components" or "temporal_components" objects?
    savemat(file_name='/output/pmd-output.mat', mdict={'Y':mov_denoised.astype(np.uint16), 'U':U, 'V':V})
    print("done")

if __name__ == "__main__":
    main()
