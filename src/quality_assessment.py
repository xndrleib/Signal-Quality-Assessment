from src.utils import read_data
import numpy as np
from src.data_preprocessing import (
    segment_signal,
    normalize_segment,
    fft_segment
)
from src.vis import (
    plot_segment_with_uncertainty
)
import matplotlib.pyplot as plt
import yaml

def weighted_average_metric(metrics, weights):
    """
    Compute a weighted average of a metric across segments.

    Args:
        metrics (np.ndarray): Array of metric values, one per segment.
        weights (np.ndarray): Array of weights, same length as metrics.

    Returns:
        float: Weighted average of the metric.
    """
    metrics = np.array(metrics)
    weights = np.array(weights)

    # Handle edge cases
    if len(metrics) == 0 or len(weights) == 0 or np.sum(weights) == 0:
        return 0.0

    return np.sum(metrics * weights) / np.sum(weights)

def spectral_entropy(fft_magnitude, eps=1e-12):
    """
    Compute the Spectral Entropy of the FFT magnitude spectrum.
    
    Spectral Entropy measures how spread out (uniform) the energy is
    across the spectrum. Lower values indicate one or few dominant frequencies;
    higher values indicate more uniform distribution of energy.
    
    Args:
        fft_magnitude (np.ndarray): Magnitudes of the FFT for each frequency bin.
        eps (float): A small constant to avoid log(0).
    
    Returns:
        float: The spectral entropy (in bits if log base 2, or nats if log base e).
    """
    fft_magnitude = np.array(fft_magnitude)
    
    # Convert magnitude to power
    power_spectrum = fft_magnitude ** 2
    
    # Normalize the power to form a probability distribution
    power_sum = np.sum(power_spectrum) + eps
    p = power_spectrum / power_sum
    
    # Calculate the spectral entropy
    spectral_ent = -np.sum(p * np.log(p + eps))
    
    return spectral_ent


if __name__ == '__main__':
    datapath = 'data/data-sample_motor-operating-at-100%-load.txt'
    df = read_data(datapath)

    noisy_signal = df['Data'].to_numpy()
    noisy_signal_mean = np.mean(noisy_signal)
    noisy_signal -= noisy_signal_mean

    segments = segment_signal(noisy_signal, segment_length=10000, step=20)
    fft_segments, freqs = fft_segment(segments, f_sampling=10000, db=True, cutoff_freq=250)

    for i in range(fft_segments.shape[0]):
        fft_segments[i, :] = normalize_segment(fft_segments[i, :], method='z-score')[0]

    spectrum_mean = np.mean(fft_segments, axis=0)
    spectrum_std = np.std(fft_segments, axis=0)

    fig, ax = plot_segment_with_uncertainty(spectrum_mean=spectrum_mean, spectrum_std=spectrum_std, 
                                  x_values=freqs,
                                  n_std=3)
    

    save_path = 'res/segment_with_uncertainty.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")

    avg_std = np.mean(spectrum_std)
    spectral_entropy = spectral_entropy(spectrum_mean, eps=1e-12)

    results = {
        'Path': datapath,
        'Average Standard Deviation': avg_std.tolist(),
        'Spectral Entropy': spectral_entropy.tolist()
    }

    # with open('res/result.yml', 'w') as yaml_file:
    #     yaml.dump(results, yaml_file, default_flow_style=False)

        