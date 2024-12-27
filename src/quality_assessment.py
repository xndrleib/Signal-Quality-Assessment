import logging
import numpy as np
from typing import Union

logger = logging.getLogger(__name__)

def weighted_average_metric(metrics: np.ndarray, weights: np.ndarray) -> float:
    """
    Compute a weighted average of a metric across segments.

    :param metrics: Array of metric values, one per segment.
    :param weights: Array of weights, same length as metrics.
    :return: Weighted average of the metric.
    """
    metrics = np.array(metrics)
    weights = np.array(weights)

    if len(metrics) == 0 or len(weights) == 0 or np.sum(weights) == 0:
        logger.warning("weighted_average_metric called with empty metrics or zero-sum weights.")
        return 0.0

    return float(np.sum(metrics * weights) / np.sum(weights))


def spectral_entropy(fft_magnitude: np.ndarray, eps: float = 1e-12) -> float:
    """
    Compute the Spectral Entropy of the FFT magnitude spectrum.

    Spectral Entropy measures how spread out (uniform) the energy is
    across the spectrum. Lower values indicate one or few dominant frequencies;
    higher values indicate more uniform distribution of energy.

    :param fft_magnitude: 1D array of FFT magnitudes (dB or linear).
    :param eps: Small constant to avoid log(0).
    :return: Spectral entropy (nats if natural log).
    """
    fft_magnitude = np.array(fft_magnitude)

    # Convert magnitude to power
    power_spectrum = fft_magnitude ** 2

    # Normalize the power to form a probability distribution
    power_sum = np.sum(power_spectrum) + eps
    p = power_spectrum / power_sum

    # Calculate the spectral entropy
    spectral_ent = -np.sum(p * np.log(p + eps))
    return float(spectral_ent)


def dominant_frequency_metric(
    fft_magnitude: np.ndarray,
    freqs: np.ndarray,
    target_freq: float = 50.0,
    freq_tolerance: float = 1.0
) -> float:
    """
    Calculates how dominant the target frequency (e.g., 50 Hz) is compared 
    to other frequencies. We find the amplitude near the target frequency 
    (within freq_tolerance) and then form a ratio with the average amplitude 
    of the rest of the spectrum.

    :param fft_magnitude: 1D array of FFT magnitudes (dB or linear).
    :param freqs: 1D array of frequency values.
    :param target_freq: The frequency of interest (Hz).
    :param freq_tolerance: Range +/- around the target freq to consider.
    :return: Ratio of amplitude near target freq to average amplitude of the rest.
    """
    fft_magnitude = np.array(fft_magnitude)
    freqs = np.array(freqs)

    mask_target = (freqs >= (target_freq - freq_tolerance)) & (freqs <= (target_freq + freq_tolerance))
    if not np.any(mask_target):
        logger.info(f"No frequency bins within {target_freq} ± {freq_tolerance} Hz found.")
        return 0.0

    target_amp = float(np.mean(fft_magnitude[mask_target]))
    mask_rest = ~mask_target

    if np.any(mask_rest):
        rest_amp = float(np.mean(fft_magnitude[mask_rest]))
    else:
        rest_amp = 1e-8  # Avoid zero division

    ratio = (target_amp + 1e-12) / (rest_amp + 1e-12)
    logger.debug(f"dominant_frequency_metric -> target_amp={target_amp:.3f}, rest_amp={rest_amp:.3f}, ratio={ratio:.3f}")
    return float(ratio)
