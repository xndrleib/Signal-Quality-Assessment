import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def segment_signal(
    signal: np.ndarray,
    segment_length: int,
    step: Optional[int] = None,
    overlap: Optional[float] = None,
    window_shape: Optional[str] = None,
) -> np.ndarray:
    """
    Segments the signal into windows.

    :param signal: 1D numpy array representing the time-series data.
    :param segment_length: Length of each segment in samples.
    :param step: Number of samples to shift the window for each \
        iteration (non-overlapping windows).
    :param overlap: Fraction of overlap between consecutive windows \
        (0 to 1). Ignored if `step` is provided.
    :param window_shape: Shape of window.
    :return: 2D numpy array of segmented windows.
    """
    if window_shape == 'barlett':
        window = np.bartlett(segment_length)
    elif window_shape == 'blackman':
        window = np.blackman(segment_length)
    elif window_shape == 'hamming':
        window = np.hamming(segment_length)
    elif window_shape == 'hanning':
        window = np.hanning(segment_length)
    elif window_shape == 'kaiser':
        window = np.kaiser(segment_length, 0.5)
    else:
        window = 1

    if step is not None:
        # Calculate segments using fixed step size
        segments = [
            signal[i:i + segment_length] * window
            for i in range(0, len(signal) - segment_length + 1, step)
        ]
        logger.debug(
            f"Segmented with step={step}, total segments={len(segments)}")
    elif overlap is not None:
        # Calculate segments using overlap
        step = int(segment_length * (1 - overlap))
        segments = [
            signal[i:i + segment_length]
            for i in range(0, len(signal) - segment_length + 1, step)
        ]
        logger.debug(
            f"Segmented with overlap={overlap:.2f}, \
                total segments={len(segments)}")
    else:
        raise ValueError("Either 'step' or 'overlap' must be specified.")

    return np.array(segments)


def time_to_freq_transform(
    data: np.ndarray,
    f_sampling: float,
    db: bool = True,
    cutoff_freq: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transforms time-series data to frequency domain using FFT (rFFT).
    Returns the magnitude spectrum (dB if db=True, otherwise linear).
    Optionally applies a cutoff frequency.

    :param data: 1D NumPy array of time-series data.
    :param f_sampling: Sampling frequency of the data.
    :param db: Whether to convert to dB scale.
    :param cutoff_freq: Optional cutoff frequency.
    :return: (yf, freqs) where yf is amplitude (dB or linear), \
        freqs is array of frequencies.
    """
    if data.ndim != 1:
        raise ValueError(
            f"Input data must be a 1D NumPy array. Got shape {data.shape}.")

    n = data.shape[0]  # Number of samples
    yf = np.fft.rfft(data)  # Perform FFT
    freqs = np.fft.rfftfreq(n, d=1 / f_sampling)  # Frequency bins

    yf = np.abs(yf)

    if db:
        # Convert to dB scale
        yf = 20 * np.log10(yf + 1e-12)  # Add small offset to avoid log(0)

    if cutoff_freq is not None:
        mask = freqs < cutoff_freq  # Apply cutoff filter
        yf = yf[mask]
        freqs = freqs[mask]

    return yf, freqs


def fft_segment(
    segments: np.ndarray,
    f_sampling: float,
    db: bool = True,
    cutoff_freq: float = 250.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform FFT on each segment in the provided array \
        and return the transformed data.

    :param segments: 2D NumPy array of segments (rows = segments).
    :param f_sampling: Sampling frequency of the data.
    :param db: Whether to convert amplitude to dB scale.
    :param cutoff_freq: Frequency cutoff for filtering FFT results.
    :return: (fft_segments, freqs)
    """
    num_segments = segments.shape[0]
    segment_length = segments.shape[1]

    # Determine the length of FFT output
    full_freqs = np.fft.rfftfreq(segment_length, d=1 / f_sampling)
    cutoff_mask = full_freqs < cutoff_freq
    fft_output_length = np.sum(cutoff_mask)

    fft_segments = np.zeros((num_segments, fft_output_length))
    freqs = full_freqs[cutoff_mask]

    # Perform FFT for each segment
    for i in range(num_segments):
        yf, _ = time_to_freq_transform(
            segments[i], f_sampling, db=db, cutoff_freq=cutoff_freq)
        fft_segments[i, :] = yf

    logger.debug(
        f"FFT computed on {num_segments} segments, \
            resulting shape={fft_segments.shape}")
    return fft_segments, freqs


def normalize_segment(
    segment: np.ndarray,
    method: str = 'z-score'
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """
    Normalizes a segment using the specified method.

    :param segment: Array representing the segment to normalize.
    :param method: 'z-score' or 'min-max'.
    :return: (normalized_segment, stats)
    """
    if method == 'z-score':
        mean = np.mean(segment)
        std = np.std(segment)
        normalized_segment = (segment - mean) / \
            (std + 1e-8)  # Avoid division by zero
        stats = (mean, std)
    elif method == 'min-max':
        min_val = np.min(segment)
        max_val = np.max(segment)
        normalized_segment = (segment - min_val) / \
            (max_val - min_val + 1e-8)  # Avoid division by zero
        stats = (min_val, max_val)
    else:
        raise ValueError(
            "Normalization method must be either 'z-score' or 'min-max'")

    return normalized_segment, stats
