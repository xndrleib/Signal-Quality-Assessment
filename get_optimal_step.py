import argparse
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.vis import plot_parameter_dependence


def main():
    parser = argparse.ArgumentParser(
        description="Signal Quality Assessment CLI (step depdndence plot)"
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
        "--start_value",
        type=int,
        default=10,
        help="Start value of Step range."
    )
    parser.add_argument(
        "--end_value",
        type=int,
        default=500,
        help="End value of Step range."
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    res_dir = Path(args.res_dir)

    for step in range(args.start_value, args.end_value, 10):
        subprocess.run(['python',
                        'main.py',
                        '--input', input_path,
                        '--res_dir', res_dir,
                        '--window_length', '10000',
                        '--step', str(step),
                        '--f_sampling', '10000',
                        '--db', 'True',
                        '--cutoff_freq', '500',
                        '--window_shape', 'blackman'])

    input_files = list(Path(input_path).glob('*.txt'))
    res_files = list(Path(res_dir).glob('*.yml'))
    for input_file in input_files:
        param_metric = {}
        max_ratio = -np.inf
        min_ratio = np.inf
        max_params_file = ''
        min_params_file = ''
        signal_name = input_file.stem.replace('data/', '')
        for res_file in res_files:
            if re.match(f'metrics_{signal_name}', res_file.name) is not None:
                yml_name = f'{res_dir}/{res_file.name}'
                with open(yml_name) as stream:
                    results = yaml.safe_load(stream)
                    param_metric[results['Step']
                                 ] = results['Dominant Freq Ratio (50.0 Hz)']
                    if results['Dominant Freq Ratio (50.0 Hz)'] < min_ratio:
                        min_ratio = results['Dominant Freq Ratio (50.0 Hz)']
                        min_params_file = yml_name
                    if results['Dominant Freq Ratio (50.0 Hz)'] > max_ratio:
                        max_ratio = results['Dominant Freq Ratio (50.0 Hz)']
                        max_params_file = yml_name
        fig, ax = plot_parameter_dependence('Step',
                                            'Dominant Freq Ratio (50.0 Hz)',
                                            signal_name,
                                            param_metric)
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        png_path = res_dir / f"Step_{signal_name}_{timestamp_str}.png"
        fig.savefig(png_path, bbox_inches='tight')
        plt.close(fig)
        if not os.path.exists(f'{res_dir}/max_params'):
            os.makedirs(f'{res_dir}/max_params')
        if not os.path.exists(f'{res_dir}/min_params'):
            os.makedirs(f'{res_dir}/min_params')
        shutil.copy(max_params_file, f'{res_dir}/max_params')
        shutil.copy(min_params_file, f'{res_dir}/min_params')


if __name__ == "__main__":
    main()
