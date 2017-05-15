#! /usr/bin/python
""" Generate a random public/private split of frame indices
"""

from __future__ import print_function, division
import argparse
import numpy as np
import os
import sys
import math


def main():
    parser = argparse.ArgumentParser(description='Generate public/private scoring split.')

    # group for mutually exclusive num_frames or input csv file options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-n', '--num_frames', type=int,
        help='Integer number of frames in source dataset (bag)')
    group.add_argument('-i', '--input_file', type=str,
        help='CSV filename to derive num_frames (input row count-header row)')

    parser.add_argument('-p', '--public_split', type=float, nargs='?', default=0.2,
        help='Public split percentage as float, 0.0 - 1.0 [default = 0.2 (20%%)]')
    parser.add_argument('-s', '--seed', type=int, nargs='?', default=None,
        help='Random number seed')
    parser.add_argument('-o', '--outdir', type=str, nargs='?', default='.',
        help='Output folder for split index CSV files')
    args = parser.parse_args()
    num_frames = args.num_frames
    input_csv_file = args.input_file
    output_dir = args.outdir
    public_split = args.public_split
    if args.seed:
        np.random.seed(args.seed)

    if not num_frames:
        if not input_csv_file or not os.path.exists(input_csv_file):
            sys.stderr.write('Error: Num frames not specified and csv file %s not found.\n' % input_csv_file)
            exit(-1)
        num_frames = sum(1 for line in open(input_csv_file) if line.strip()) - 1
    if num_frames <= 0:
        sys.stderr.write('Error: Invalid number of frames specified.\n')
        exit(-1)

    if public_split < 0 or public_split > 1.0:
        sys.stderr.write('Error: Invalid value for public split. Must be between 0.0 and 1.0\n')
        exit(-1)

    frame_indices = list(range(0, num_frames))
    np.random.shuffle(frame_indices)
    split_idx = int(math.ceil(num_frames * public_split))
    public_indices = frame_indices[:split_idx]
    private_indices = frame_indices[split_idx:]

    print('Public:', public_indices)
    print('Private:', private_indices)

    with open(os.path.join(output_dir, 'public_indices.csv'), 'w') as f:
        f.write('index\n')
        [f.write('%d\n' % x) for x in public_indices]

    with open(os.path.join(output_dir, 'private_indices.csv'), 'w') as f:
        f.write('index\n')
        [f.write('%d\n' % x) for x in private_indices]

if __name__ == '__main__':
    main()
