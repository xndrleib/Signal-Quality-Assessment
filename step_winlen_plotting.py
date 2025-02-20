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
            multiple data files.",
    )
    parser.add_argument(
        "--res_dir_prefix",
        type=str,
        default="plotting",
        help="Path to results folder (default: 'res').",
    )
    parser.add_argument(
        "--start_fft_step_value",
        type=int,
        default=10,
        help="Start value of FFT step range.",
    )
    parser.add_argument(
        "--end_fft_step_value",
        type=int,
        default=500,
        help="End value of FFT step range.",
    )
    parser.add_argument(
        "--fft_step_step",
        type=int,
        default=10,
        help="FFT step range step.",
    )
    parser.add_argument(
        "--start_winlen_value",
        type=int,
        default=5000,
        help="Start value of Window length range.",
    )
    parser.add_argument(
        "--end_winlen_value",
        type=int,
        default=30000,
        help="End value of Window length range.",
    )
    parser.add_argument(
        "--winlen_step",
        type=int,
        default=100,
        help="Window length range step.",
    )
    parser.add_argument(
        "--f_sampling",
        type=str,
        default="as_window_length",
        help="F sampling (float or 'as_window_length').",
    )
    parser.add_argument(
        "--x_axis_parameter",
        type=str,
        default="window_length",
        help="Parameter for X axis of plots.",
    )

    args = parser.parse_args()
    input_path = Path(args.input)

    if args.x_axis_parameter == 'window_length':
        first = 'Step'
        second = 'Window length'
        start_first = args.start_fft_step_value
        end_first = args.end_fft_step_value
        step_first = args.fft_step_step
        start_second = args.start_winlen_value
        end_second = args.end_winlen_value
        step_second = args.winlen_step
    else:
        first = 'Window length'
        second = 'Step'
        start_first = args.start_winlen_value
        end_first = args.end_winlen_value
        step_first = args.winlen_step
        start_second = args.start_fft_step_value
        end_second = args.end_fft_step_value
        step_second = args.fft_step_step

    for first_value in range(start_first, end_first, step_first):
        res_dir = Path(
            f"{args.res_dir_prefix}_{first}_{first_value}_f-sampling_{args.f_sampling}"
        )
        for second_value in range(start_second, end_second, step_second):
            step = first_value if first == 'Step' else second_value
            window_length = first_value if first == 'Window length' else second_value
            f_sampling = (
                str(window_length)
                if (args.f_sampling == "as_window_length")
                else args.f_sampling
            )
            subprocess.run(
                [
                    "python", "main.py",
                    "--input", input_path,
                    "--res_dir", res_dir,
                    "--window_length", str(window_length),
                    "--step", str(step),
                    "--f_sampling", f_sampling,
                    "--db", "True",
                    "--cutoff_freq", "250.0",
                    "--window_shape", "blackman",
                ]
            )

        input_files = list(Path(input_path).glob("*.txt"))
        res_files = list(Path(res_dir).glob("*.yml"))
        for input_file in input_files:
            param_metric = {}
            max_ratio = -np.inf
            min_ratio = np.inf
            max_params_file = ""
            min_params_file = ""
            signal_name = input_file.stem.replace("data/", "")
            for res_file in res_files:
                if (
                    re.match(f"metrics_{signal_name}", res_file.name)
                    is not None
                ):
                    yml_name = f"{res_dir}/{res_file.name}"
                    with open(yml_name) as stream:
                        results = yaml.safe_load(stream)
                        param_metric[results[second]] = results[
                            "Dominant Freq Ratio (50.0 Hz)"
                        ]
                        if results["Dominant Freq Ratio (50.0 Hz)"] < min_ratio:
                            min_ratio = results["Dominant Freq Ratio (50.0 Hz)"]
                            min_params_file = yml_name
                        if results["Dominant Freq Ratio (50.0 Hz)"] > max_ratio:
                            max_ratio = results["Dominant Freq Ratio (50.0 Hz)"]
                            max_params_file = yml_name
            fig, ax = plot_parameter_dependence(
                f"{second} ({first}={first_value}, F sampling={args.f_sampling})",
                "Dominant Freq Ratio (50.0 Hz)",
                signal_name,
                param_metric,
            )
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            param_part = f'({first}={first_value}, F sampling={args.f_sampling})'
            png_path = (
                res_dir
                / f"{second} {param_part}_{signal_name}_{timestamp_str}.png"
            )
            fig.savefig(png_path, bbox_inches="tight")
            plt.close(fig)
            if not os.path.exists(f"{res_dir}/max_params"):
                os.makedirs(f"{res_dir}/max_params")
            if not os.path.exists(f"{res_dir}/min_params"):
                os.makedirs(f"{res_dir}/min_params")
            shutil.copy(max_params_file, f"{res_dir}/max_params")
            shutil.copy(min_params_file, f"{res_dir}/min_params")


if __name__ == "__main__":
    main()
