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

def dominant_frequency_metric(fft_magnitude, freqs, target_freq=50.0, freq_tolerance=1.0):
    """
    Calculates how dominant the target frequency (e.g., 50 Hz) is compared 
    to other frequencies. We find the amplitude near the target frequency 
    (within freq_tolerance) and then form a ratio with the average amplitude 
    of the rest of the spectrum.

    Args:
        fft_magnitude (np.ndarray): 1D array of FFT amplitudes (already in dB or linear scale).
        freqs (np.ndarray): 1D array of corresponding frequency values.
        target_freq (float): The frequency of interest, e.g., 50 Hz.
        freq_tolerance (float): Range around target_freq to consider as "peak zone".

    Returns:
        float: A ratio where >1 means target frequency region is stronger 
               than the average of the rest.
    """
    # Ensure inputs are numpy arrays
    fft_magnitude = np.array(fft_magnitude)
    freqs = np.array(freqs)

    # Identify indices near the target frequency
    mask_target = (freqs >= (target_freq - freq_tolerance)) & (freqs <= (target_freq + freq_tolerance))
    
    if not np.any(mask_target):
        # If we don't find the target frequency, return something sensible (like 0.0)
        return 0.0

    # Mean amplitude in the band around target_freq
    target_amp = np.mean(fft_magnitude[mask_target])

    # Average amplitude of the rest
    mask_rest = ~mask_target
    if np.any(mask_rest):
        rest_amp = np.mean(fft_magnitude[mask_rest])
    else:
        rest_amp = 1e-8  # Avoid divide by zero
    
    # Ratio: if target_amp >> rest_amp, ratio >> 1
    ratio = (target_amp + 1e-12) / (rest_amp + 1e-12)
    target_amp = np.mean(fft_magnitude[mask_target])
    
    print(f'Mean amplitude in the band {target_freq} ± {freq_tolerance} Hz: {target_amp}')
    print(f'Average amplitude of the rest: {rest_amp}')
    return ratio