import argparse
import logging
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import yaml

# Local imports
from src.data_preprocessing import fft_segment, segment_signal
from src.quality_assessment import dominant_frequency_metric, spectral_entropy
from src.utils import read_data
from src.vis import plot_spectrum_with_uncertainty

# Suppress all logs at the root level
logging.basicConfig(
    level=logging.CRITICAL,  # Suppress all logs by default
    format='[%(levelname)s] %(asctime)s - %(name)s - %(message)s'
)

# Create a file handler
file_handler = logging.FileHandler("signal_quality.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(
    '[%(levelname)s] %(asctime)s - %(name)s - %(message)s'))

# Create a stream handler
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(logging.Formatter(
    '[%(levelname)s] %(asctime)s - %(name)s - %(message)s'))

# Enable logging for 'main'
main_logger = logging.getLogger("main")
main_logger.setLevel(logging.DEBUG)  # Enable logging for 'main'
main_logger.addHandler(file_handler)  # Attach handlers to 'main'
main_logger.addHandler(stream_handler)

# Enable logging for 'src'
src_logger = logging.getLogger("src")
src_logger.setLevel(logging.DEBUG)  # Enable logging for 'src'
src_logger.addHandler(file_handler)  # Attach handlers to 'src'
src_logger.addHandler(stream_handler)

# Ensure 'main' and 'src' loggers do not propagate to the root logger
main_logger.propagate = False
src_logger.propagate = False

# Use 'main' logger for this script
logger = logging.getLogger("main")


def process_single_file(
    file_path: Path,
    res_dir: Path,
    window_length: int = 10000,
    step: int = 20,
    f_sampling: float = 10000.0,
    db: bool = True,
    cutoff_freq: float = 250.0,
    window_shape: str = 'rect'
) -> None:
    """
    Process a single signal file: segment it, transform to FFT, compute metrics,
    plot uncertainty, and save results to a YAML file.

    :param file_path: Path to the input signal file.
    :param res_dir: Directory to store results.
    :param window_length: Number of samples in each segment window.
    :param step: Step size for sliding windows.
    :param f_sampling: Sampling frequency of the data.
    :param db: Whether to convert FFT amplitude to dB scale.
    :param cutoff_freq: Frequency cutoff for filtering FFT results.
    :param window_shape: Shape of window
    """
    logger.info(f"Processing file: {file_path}")

    # Read the data
    df = read_data(file_path)
    signal = df['Data'].to_numpy()

    # Optional removal of DC offset
    signal_mean = np.mean(signal)
    signal -= signal_mean
    logger.debug(f"Signal mean: {signal_mean:.3f} removed from signal.")

    # Segment the signal
    segments = segment_signal(
        signal, segment_length=window_length,
        step=step,
        window_shape=window_shape
    )
    if segments.size == 0:
        logger.warning(
            f"No valid segments produced from {file_path} with \
                window_length={window_length}, step={step}")
        return

    # Perform FFT on each segment
    fft_segments, freqs = fft_segment(
        segments,
        f_sampling=f_sampling,
        db=db,
        cutoff_freq=cutoff_freq
    )
    logger.debug(
        f"fft_segments shape: {fft_segments.shape}, \
            freqs length: {freqs.shape[0]}")

    # Compute mean and standard deviation across segments
    spectrum_mean = np.mean(fft_segments, axis=0)
    spectrum_std = np.std(fft_segments, axis=0)

    # Compute metrics
    avg_std: float = float(np.mean(spectrum_std))
    spec_entropy: float = float(spectral_entropy(spectrum_mean, eps=1e-12))

    target_freq = 50.0
    freq_tolerance = 1.0
    dom_freq_ratio: float = float(dominant_frequency_metric(
        fft_magnitude=spectrum_mean,
        freqs=freqs,
        target_freq=target_freq,
        freq_tolerance=freq_tolerance
    ))

    # Plot and save
    fig, ax = plot_spectrum_with_uncertainty(
        spectrum_mean=spectrum_mean,
        spectrum_std=spectrum_std,
        x_values=freqs,
        n_std=3,
        title=f"Spectrum with Uncertainty: {file_path.name}"
    )

    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    filename_without_ext = file_path.stem

    res_dir.mkdir(parents=True, exist_ok=True)
    png_path = res_dir / \
        f"spectrum_with_uncertainty_{filename_without_ext}_{timestamp_str}.png"
    fig.savefig(png_path, bbox_inches='tight')
    logger.info(f"Plot saved to {png_path}")
    plt.close(fig)

    # Prepare results dictionary
    results = {
        'Path': str(file_path),
        'Window length': window_length,
        'Step': step,
        'F sampling': f_sampling,
        'Db': db,
        'Cutoff freq': cutoff_freq,
        'Window shape': window_shape,
        'Average Standard Deviation': avg_std,
        'Spectral Entropy': spec_entropy,
        f'Dominant Freq Ratio ({target_freq} Hz)': dom_freq_ratio,
        'Timestamp': timestamp_str
    }

    # Save results to a YAML file with timestamp
    yml_filename = f"metrics_{filename_without_ext}_{timestamp_str}.yml"
    yml_path = res_dir / yml_filename

    with open(yml_path, 'w', encoding='utf-8') as yaml_file:
        yaml.dump(results, yaml_file,
                  default_flow_style=False, allow_unicode=True)

    logger.info(f"Results saved to {yml_path}")
    return yml_path


class Objective:
    def __init__(
            self,
            input_file,
            res_dir,
            window_length_range,
            step_range,
    ):
        self.input_file = input_file
        self.res_dir = res_dir
        self.window_length_range = window_length_range
        self.step_range = step_range

    def __call__(self, trial):
        window_length = trial.suggest_int('window_length',
                                          self.window_length_range[0],
                                          self.window_length_range[1]
                                          )
        step = trial.suggest_int('step',
                                 self.step_range[0],
                                 self.step_range[1]
                                 )
        f_sampling = 10000
        cutoff_freq = 250.0
        db = True
        window_shape = trial.suggest_categorical('window_shape', ['rect',
                                                                  'bartlett',
                                                                  'blackman',
                                                                  'hanning',
                                                                  'hamming',
                                                                  'kaiser'])

        # Check if input is a file or a directory
        if self.input_file.is_file():
            # Single file
            yml_path = process_single_file(
                file_path=self.input_file,
                res_dir=self.res_dir,
                window_length=window_length,
                step=step,
                f_sampling=f_sampling,
                db=db,
                cutoff_freq=cutoff_freq,
                window_shape=window_shape
            )
            with open(yml_path) as stream:
                results = yaml.safe_load(stream)
                return results['Dominant Freq Ratio (50.0 Hz)']
        else:
            logger.error(f"Invalid input path: {self.input_file}")
            return


def main():
    parser = argparse.ArgumentParser(
        description="Signal Quality Assessment CLI"
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
        "--window_length_range",
        type=str,
        default='5000,50000',
        help="Number of samples in each segment window (default: 10000)."
    )
    parser.add_argument(
        "--step_range",
        type=str,
        default='10,100',
        help="Step size for sliding windows (default: 20)."
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default='100',
        help="Number of optuna trials (default: 100)."
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    res_dir = Path(args.res_dir)
    window_length_range = [int(arg)
                           for arg in args.window_length_range.split(',')]
    step_range = [int(arg) for arg in args.step_range.split(',')]

    # Log input parameters
    logger.info("CLI Input Parameters:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")

    for input_file in input_path.glob('*.txt'):
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.MedianPruner()
        )
        study.optimize(Objective(input_file,
                                 res_dir,
                                 window_length_range,
                                 step_range),
                       n_trials=args.n_trials)
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        with open(f'{res_dir}/optuna_{input_file.stem}_{timestamp_str}.yml',
                  'w',
                  encoding='utf-8') as inf:
            best_signal = study.best_params.copy()
            best_signal['metric'] = study.best_value
            yaml.dump(best_signal, inf,
                      default_flow_style=False, allow_unicode=True)


if __name__ == "__main__":
    main()
