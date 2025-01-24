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
        "--fft_steps",
        type=str,
        default="10,20,50,100,250,500",
        help="FFT step manual range.",
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
        "--winlen_step", type=int, default=100, help="Window length range step."
    )
    parser.add_argument(
        "--f_sampling",
        type=str,
        default="as_window_length",
        help="F sampling (float or 'as_window_length').",
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    steps = [step.strip() for step in args.fft_steps.split(",")]

    for step in steps:
        res_dir = Path(
            f"{args.res_dir_prefix}_step_{step}_f-sampling_{args.f_sampling}"
        )
        for window_length in range(
            args.start_winlen_value, args.end_winlen_value, args.winlen_step
        ):
            f_sampling = (
                str(window_length)
                if (args.f_sampling == "as_window_length")
                else args.f_sampling
            )
            subprocess.run(
                [
                    "python",
                    "main.py",
                    "--input",
                    input_path,
                    "--res_dir",
                    res_dir,
                    "--window_length",
                    str(window_length),
                    "--step",
                    step,
                    "--f_sampling",
                    f_sampling,
                    "--db",
                    "True",
                    "--cutoff_freq",
                    "250",
                    "--window_shape",
                    "blackman",
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
                        param_metric[results["Window length"]] = results[
                            "Dominant Freq Ratio (50.0 Hz)"
                        ]
                        if results["Dominant Freq Ratio (50.0 Hz)"] < min_ratio:
                            min_ratio = results["Dominant Freq Ratio (50.0 Hz)"]
                            min_params_file = yml_name
                        if results["Dominant Freq Ratio (50.0 Hz)"] > max_ratio:
                            max_ratio = results["Dominant Freq Ratio (50.0 Hz)"]
                            max_params_file = yml_name
            fig, ax = plot_parameter_dependence(
                f"Window length (step={step}, F sampling={args.f_sampling})",
                "Dominant Freq Ratio (50.0 Hz)",
                signal_name,
                param_metric,
            )
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            param_part = f'(step={step}, F sampling={args.f_sampling})'
            png_path = (
                res_dir
                / f"Window length {param_part}_{signal_name}_{timestamp_str}.png"
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
