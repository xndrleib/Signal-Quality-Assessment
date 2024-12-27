import os
import numpy as np
import pandas as pd


def segment_signal(signal, segment_length, step=None, overlap=None,):
    """
    Segments the signal into windows, with options for overlap or fixed step size.

    Parameters:
    - signal: 1D numpy array representing the time-series data.
    - segment_length: Length of each segment in samples.
    - step: Number of samples to shift the window for each iteration (used for non-overlapping windows).
             If None, overlap is used instead.
    - overlap: Fraction of overlap between consecutive windows (0 to 1). Ignored if `step` is provided.

    Returns:
    - numpy array of segmented windows.
    """
    if step is not None:
        # Calculate segments using fixed step size
        segments = [
            signal[i:i + segment_length]
            for i in range(0, len(signal) - segment_length + 1, step)
        ]
    elif overlap is not None:
        # Calculate segments using overlap
        step = int(segment_length * (1 - overlap))
        segments = [
            signal[i:i + segment_length]
            for i in range(0, len(signal) - segment_length + 1, step)
        ]
    else:
        raise ValueError("Either 'step' or 'overlap' must be specified.")

    return np.array(segments)


def time_to_freq_transform(data, f_sampling, db=True, cutoff_freq=None):
    """
    Transforms time-series data to frequency domain using FFT.
    Optionally applies a cutoff frequency.
    
    Parameters:
    - data: NumPy array containing the time series data.
    - f_sampling: Sampling frequency of the data.
    - db: Boolean flag to convert FFT values to decibel scale.
    - cutoff_freq: Optional cutoff frequency to filter the results.
    
    Returns:
    - yf: Transformed frequency domain values.
    - freqs: Frequency bins.
    """
    if data.ndim != 1:
        raise ValueError(f"Input data must be a 1D NumPy array. Got shape {data.shape}.")
    
    n = data.shape[0]  # Number of samples
    yf = np.fft.rfft(data)  # Perform FFT
    freqs = np.fft.rfftfreq(n, d=1/f_sampling)  # Frequency bins
    
    if db:
        yf = 20 * np.log10(np.abs(yf))  # Convert to dB scale
    
    if cutoff_freq is not None:
        mask = freqs < cutoff_freq  # Apply cutoff filter
        yf = yf[mask]
        freqs = freqs[mask]
    
    return yf, freqs


def fft_segment(segments, f_sampling, db=True, cutoff_freq=250):
    """
    Perform FFT on each segment in the provided array and return the transformed data.
    
    Parameters:
    - segments: 2D NumPy array of segments (each row is a segment).
    - f_sampling: Sampling frequency of the data.
    - db: Boolean flag to convert FFT values to decibel scale.
    - cutoff_freq: Frequency cutoff for filtering FFT results.
    
    Returns:
    - fft_segments: 2D NumPy array where each row is the FFT-transformed data of a segment.
    - freqs: Frequency bins (shared across all segments).
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
        yf, _ = time_to_freq_transform(segments[i], f_sampling, db=db, cutoff_freq=cutoff_freq)
        fft_segments[i, :] = yf
    
    return fft_segments, freqs


def process_time_series(input_data, output_dir, window_length=20000, shift=20, f_sampling=1.0, db=True, cutoff_freq=250):
    """
    Applies a sliding window over time series data, performs FFT on each window,
    and saves the transformed data.
    
    Parameters:
    - input_data: Path to the CSV file with time-series data or a pandas DataFrame.
    - output_dir: Directory to save the transformed window files.
    - window_length: Number of samples in each window.
    - shift: Number of samples to shift the window for each iteration.
    - f_sampling: Sampling frequency of the data.
    - db: Boolean flag to convert FFT values to decibel scale.
    - cutoff_freq: Frequency cutoff for filtering FFT results.
    """
    # Determine if input_data is a file path or a DataFrame
    if isinstance(input_data, str):
        df = pd.read_csv(input_data)
    elif isinstance(input_data, pd.DataFrame):
        df = input_data
    else:
        raise ValueError("input_data must be a file path (str) or a pandas DataFrame.")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Segment the signal
    segments = segment_signal(df['Data'].values, segment_length=window_length, step=shift)
    
    # Perform FFT on each segment and get the FFT data
    fft_segments, freqs = fft_segment(segments, f_sampling, db=db, cutoff_freq=cutoff_freq)
    
    # Save each FFT-transformed segment to a file
    for i, yf in enumerate(fft_segments):
        transformed_df = pd.DataFrame({
            'Frequency (Hz)': freqs,
            'Amplitude': yf
        })
        output_file_path = os.path.join(output_dir, f'Window_{i + 1}.csv')
        transformed_df.to_csv(output_file_path, index=False)
    
    print(f'Results saved to {output_dir}')
    return fft_segments, freqs

def normalize_segment(segment, method='z-score'):
    """
    Normalizes a segment using the specified method.
    
    Parameters:
    - segment: Array representing the segment to normalize.
    - method: Normalization method, either 'z-score' (standardization) or 'min-max'.
    
    Returns:
    - normalized_segment: The normalized segment array.
    - stats: Normalization statistics (mean and std for z-score, min and max for min-max).
    """
    if method == 'z-score':
        mean = np.mean(segment)
        std = np.std(segment)
        normalized_segment = (segment - mean) / (std + 1e-8)  # Avoid division by zero
        stats = (mean, std)
    elif method == 'min-max':
        min_val = np.min(segment)
        max_val = np.max(segment)
        normalized_segment = (segment - min_val) / (max_val - min_val + 1e-8)  # Avoid division by zero
        stats = (min_val, max_val)
    else:
        raise ValueError("Normalization method must be either 'z-score' or 'min-max'")

    return normalized_segment, stats
