import argparse
import os
import re
import shutil
import subprocess
from itertools import product
from pathlib import Path

import numpy as np
import yaml


def main():
    parser = argparse.ArgumentParser(
        description="Signal Quality Assessment CLI (grid search)"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to a single data file or a folder containing \
            multiple data files."
    )
    parser.add_argument(
        "--res_dir",
        type=str,
        default="res",
        help="Path to results folder (default: 'res')."
    )
    parser.add_argument(
        "--window_lengths",
        type=str,  # int
        default='10000',
        help="Number of samples in each segment window (as range)."
    )
    parser.add_argument(
        "--steps",
        type=str,  # int
        default='20',
        help="Step size for sliding windows (as range)."
    )
    parser.add_argument(
        "--f_samplings",
        type=str,  # float
        default='10000.0',
        help="Sampling frequency of the data (as range)."
    )
    parser.add_argument(
        "--dbs",
        type=str,  # bool
        default='True',
        help="Convert FFT values to dB scale. (as range)"
    )
    parser.add_argument(
        "--cutoff_freqs",
        type=str,  # float
        default='250.0',
        help="Frequency cutoff for filtering FFT results (as range)."
    )
    parser.add_argument(
        "--window_shapes",
        type=str,
        default='rect',
        help="Shape of window (as range)."
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    res_dir = Path(args.res_dir)
    window_lengths = [win_len.strip() for win_len in args.window_lengths.split(',')]
    steps = [step.strip() for step in args.steps.split(',')]
    f_samplings = [f_samp.strip() for f_samp in args.f_samplings.split(',')]
    dbs = [db.strip() for db in args.dbs.split(',')]
    cutoff_freqs = [cutoff_freq.strip() for cutoff_freq in args.cutoff_freqs.split(',')]
    window_shapes = [win_shape.strip() for win_shape in args.window_shapes.split(',')]

    if 'as_window_length' in f_samplings:
        for params in product(window_lengths,
                              steps,
                              dbs,
                              cutoff_freqs,
                              window_shapes):
            subprocess.run(['python',
                            'main.py',
                            '--input', input_path,
                            '--res_dir', res_dir,
                            '--window_length', params[0],
                            '--step', params[1],
                            '--f_sampling', f'{params[0]}.0',
                            '--db', params[2],
                            '--cutoff_freq', params[3],
                            '--window_shape', params[4]])
    else:
        for params in product(window_lengths,
                              steps,
                              f_samplings,
                              dbs,
                              cutoff_freqs,
                              window_shapes):
            subprocess.run(['python',
                            'main.py',
                            '--input', input_path,
                            '--res_dir', res_dir,
                            '--window_length', params[0],
                            '--step', params[1],
                            '--f_sampling', params[2],
                            '--db', params[3],
                            '--cutoff_freq', params[4],
                            '--window_shape', params[5]])

    input_files = list(Path(input_path).glob('*.txt'))
    res_files = list(Path(res_dir).glob('*.yml'))
    for input_file in input_files:
        signal_name = input_file.stem.replace('data/', '')
        max_ratio = -np.inf
        min_ratio = np.inf
        max_params_file = ''
        min_params_file = ''
        for res_file in res_files:
            if re.match(f'metrics_{signal_name}', res_file.name) is not None:
                yml_name = f'{res_dir}/{res_file.name}'
                with open(yml_name) as stream:
                    results = yaml.safe_load(stream)
                    if results['Dominant Freq Ratio (50.0 Hz)'] < min_ratio:
                        min_ratio = results['Dominant Freq Ratio (50.0 Hz)']
                        min_params_file = yml_name
                    if results['Dominant Freq Ratio (50.0 Hz)'] > max_ratio:
                        max_ratio = results['Dominant Freq Ratio (50.0 Hz)']
                        max_params_file = yml_name

        if not os.path.exists(f'{res_dir}/max_params'):
            os.makedirs(f'{res_dir}/max_params')
        if not os.path.exists(f'{res_dir}/min_params'):
            os.makedirs(f'{res_dir}/min_params')
        shutil.copy(max_params_file, f'{res_dir}/max_params')
        shutil.copy(min_params_file, f'{res_dir}/min_params')


if __name__ == "__main__":
    main()
