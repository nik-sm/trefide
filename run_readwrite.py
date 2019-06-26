#!/usr/bin/env python
# Test whether reading/writing causing any data corruption
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

    mov = np.ascontiguousarray(loadmat("/input/{}".format(args.infile))['inputData'], dtype='double') # TODO - hardcoded path within container
    savemat(file_name='/output/pmd-output.mat', mdict={'Y_raw':mov.astype(np.uint16)})

    print("done")

if __name__ == "__main__":
    main()
