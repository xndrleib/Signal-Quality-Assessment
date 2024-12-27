import numpy as np

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
