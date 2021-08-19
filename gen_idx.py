import numpy as np
import os
import argparse

def read_index_raw(fname):
    idx = np.loadtxt(fname, dtype=int)
    return idx.flatten()

def add_shift(fname, shift, new_fname=None):
    idx = read_index_raw(fname)
    idx += shift
    if new_fname:
        with open(new_fname, 'w') as f:
            for ii in idx:
                f.write(str(ii) + " ")
    return idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default="index.raw", help="Input index file")
    parser.add_argument("-s", "--shift", default=0, type=int, help="Shift")
    parser.add_argument("-o", "--output", default="index_shift.raw", help="Output index file")
    args = parser.parse_args()
    add_shift(args.input, args.shift, args.output)