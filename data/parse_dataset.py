
import argparse
import numpy as np
import os
import subprocess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ref', default='val.txt', type=str, help='file list')
    parser.add_argument('--gt_dir', default='', type=str)
    parser.add_argument('--result_dir', default='', type=str)
    parser.add_argument('--out_dir', type=str, default='./')
    args = parser.parse_args()

    num_list = np.loadtxt(args.ref, dtype=str)
    gt_file = [os.path.join(args.gt_dir, num + '.txt') for num in num_list]
    res_file = [os.path.join(args.result_dir, num + '.txt') for num in num_list]

    if args.gt_dir:
        subprocess.check_call(['mkdir', '-p', os.path.join(args.out_dir, 'gt')])
        out_file = [os.path.join(args.out_dir, 'gt', "{:06}.txt".format(i)) for i in range(len(num_list))]
        for gf, of in zip(gt_file, out_file):
            subprocess.check_call(['cp', gf, of])

    if args.result_dir:
        subprocess.check_call(['mkdir', '-p', os.path.join(args.out_dir, 'result')])
        out_file = [os.path.join(args.out_dir, 'result', "{:06}.txt".format(i)) for i in range(len(num_list))]
        for gf, of in zip(res_file, out_file):
            subprocess.check_call(['cp', gf, of])

if __name__ == "__main__":
    main()
