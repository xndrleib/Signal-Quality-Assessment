#!/usr/bin/env python
import argparse
import os
import time
import yaml
import numpy as np

# Local imports
from src.utils import read_data
from src.data_preprocessing import (
    segment_signal,
    normalize_segment,
    fft_segment
)
from src.vis import (
    plot_spectrum_with_uncertainty
)
from src.quality_assessment import spectral_entropy

import matplotlib.pyplot as plt


def process_single_file(file_path, res_dir, window_length=10000, step=20,
                        f_sampling=10000, db=True, cutoff_freq=250):
    """
    Process a single signal file: segment it, transform to FFT, normalize, 
    compute metrics, plot uncertainty, save to YAML in res_dir with a timestamp.
    """
    # Read the data
    df = read_data(file_path)
    signal = df['Data'].to_numpy()

    # Optional removal of DC offset
    signal_mean = np.mean(signal)
    signal -= signal_mean

    # Segment the signal
    segments = segment_signal(signal, segment_length=window_length, step=step)

    # Perform FFT on each segment
    fft_segments, freqs = fft_segment(segments, f_sampling=f_sampling, db=db, cutoff_freq=cutoff_freq)

    # Compute mean and standard deviation across segments
    spectrum_mean = np.mean(fft_segments, axis=0)
    spectrum_std = np.std(fft_segments, axis=0)

    # Compute metrics
    avg_std = np.mean(spectrum_std)
    spec_entropy = spectral_entropy(spectrum_mean, eps=1e-12)

    # Plot and save
    fig, ax = plot_spectrum_with_uncertainty(spectrum_mean=spectrum_mean, 
                                            spectrum_std=spectrum_std, 
                                            x_values=freqs, 
                                            n_std=3,
                                            title="Spectrum with Uncertainty")
    
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    basename = os.path.basename(file_path)
    filename_without_ext = os.path.splitext(basename)[0]

    os.makedirs(res_dir, exist_ok=True)
    png_path = os.path.join(res_dir, f"segment_with_uncertainty_{filename_without_ext}_{timestamp_str}.png")
    fig.savefig(png_path, bbox_inches='tight')
    plt.close(fig)

    # Prepare results dictionary
    results = {
        'Path': file_path,
        'Average Standard Deviation': float(avg_std),
        'Spectral Entropy': float(spec_entropy),
        'Timestamp': timestamp_str
    }

    # Save results to a YAML file with timestamp
    yml_filename = f"metrics_{filename_without_ext}_{timestamp_str}.yml"
    yml_path = os.path.join(res_dir, yml_filename)

    with open(yml_path, 'w') as yaml_file:
        yaml.dump(results, yaml_file, default_flow_style=False)

    print(f"Processed: {file_path}")
    print(f"Saved results to: {yml_path}")
    print(f"Saved plot to: {png_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Signal Quality Assessment CLI"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to a single data file or a folder containing multiple data files."
    )
    parser.add_argument(
        "--res_dir",
        type=str,
        default="res",
        help="Path to results folder (default: 'res')."
    )
    parser.add_argument(
        "--window_length",
        type=int,
        default=10000,
        help="Number of samples in each segment window (default: 10000)."
    )
    parser.add_argument(
        "--step",
        type=int,
        default=20,
        help="Step size for sliding windows (default: 20)."
    )
    parser.add_argument(
        "--f_sampling",
        type=float,
        default=10000.0,
        help="Sampling frequency of the data (default: 10000 Hz)."
    )
    parser.add_argument(
        "--db",
        action="store_true",
        default=True,
        help="Convert FFT values to dB scale if set. (default: True)"
    )
    parser.add_argument(
        "--cutoff_freq",
        type=float,
        default=250.0,
        help="Frequency cutoff for filtering FFT results (default: 250 Hz)."
    )

    args = parser.parse_args()

    # Check if input is a file or a directory
    if os.path.isfile(args.input):
        # Single file
        process_single_file(
            file_path=args.input,
            res_dir=args.res_dir,
            window_length=args.window_length,
            step=args.step,
            f_sampling=args.f_sampling,
            db=args.db,
            cutoff_freq=args.cutoff_freq
        )
    elif os.path.isdir(args.input):
        # Process multiple files in the folder
        for filename in os.listdir(args.input):
            if filename.lower().endswith(".txt"):
                full_path = os.path.join(args.input, filename)
                process_single_file(
                    file_path=full_path,
                    res_dir=args.res_dir,
                    window_length=args.window_length,
                    step=args.step,
                    f_sampling=args.f_sampling,
                    db=args.db,
                    cutoff_freq=args.cutoff_freq
                )
    else:
        print(f"Invalid input: {args.input}. Please provide a valid file or folder path.")

if __name__ == "__main__":
    main()