import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)

def plot_random_segments(
    segments: np.ndarray,
    num_segments: int = 10,
    labels: Optional[List[str]] = None,
    axis_labels: Optional[Tuple[str, str]] = None,
    x_values: Optional[np.ndarray] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots random segments from the provided segments.

    :param segments: 2D array of shape (N_segments, segment_length).
    :param num_segments: How many random segments to plot.
    :param labels: Optional list of custom labels for segments.
    :param axis_labels: (x_label, y_label) for the axes.
    :param x_values: x-values for the plot. If 2D, it matches each segment; if 1D, truncated to segment length.
    :return: (fig, ax)
    """
    if len(segments) < num_segments:
        raise ValueError(f"Requested {num_segments} segments but only have {len(segments)}.")

    random_indices = np.random.choice(len(segments), num_segments, replace=False)
    selected_segments = segments[random_indices]

    if labels is not None:
        if len(labels) != len(segments):
            raise ValueError("Length of labels must match the length of segments.")
        selected_labels = [labels[i] for i in random_indices]
    else:
        selected_labels = [f"Segment {i}" for i in random_indices]

    rows = (num_segments + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(15, 5 * rows))
    axes = axes.flatten()

    if axis_labels is not None:
        x_label, y_label = axis_labels
    else:
        x_label, y_label = ('Frequency (Hz)', 'Power (dB)')

    for i, ax in enumerate(axes):
        if i >= num_segments:
            ax.set_visible(False)
            continue

        segment = selected_segments[i]
        if x_values is not None:
            if x_values.ndim == 1:
                x_vals = x_values[:len(segment)]
            else:
                x_vals = x_values[random_indices[i]]
        else:
            x_vals = np.linspace(0, len(segment), len(segment))

        ax.plot(x_vals, segment, linewidth=1.5, color='blue')
        ax.set_title(selected_labels[i], fontsize=10, fontweight='bold')
        ax.set_xlabel(x_label, fontsize=9)
        ax.set_ylabel(y_label, fontsize=9)
        ax.grid(True, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    return fig, axes


def plot_spectrum(
    df: pd.DataFrame,
    freq_col: str = "Frequency (Hz)",
    amp_col: str = "Amplitude",
    title: str = "Frequency Spectrum",
    xlabel: str = "Frequency (Hz)",
    ylabel: str = "Amplitude (dB)",
    highlight_freqs: Optional[List[float]] = None,
    method: str = "interpolate"
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots the frequency spectrum with optional highlighted points using interpolation or closest point methods.

    :param df: DataFrame containing frequency and amplitude columns.
    :param freq_col: Name of the frequency column.
    :param amp_col: Name of the amplitude column.
    :param title: Plot title.
    :param xlabel: X-axis label.
    :param ylabel: Y-axis label.
    :param highlight_freqs: Frequencies to highlight.
    :param method: 'interpolate' or 'closest' method for highlight points.
    :return: (fig, ax)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort the DataFrame by frequency to ensure correct order for interpolation
    df_sorted = df.sort_values(by=freq_col)
    freqs = df_sorted[freq_col].values
    amps = df_sorted[amp_col].values

    # Plot the base spectrum line
    ax.plot(freqs, amps, label='Spectrum', color='blue')

    # If highlight frequencies are provided, calculate or find their amplitude
    if highlight_freqs is not None and len(highlight_freqs) > 0:
        highlight_amps = []
        for hf in highlight_freqs:
            if hf in freqs:
                # Exact match found
                hf_amp = amps[freqs.tolist().index(hf)]
            else:
                # No exact match, handle according to method
                if method == "interpolate":
                    # Use np.interp to interpolate
                    hf_amp = float(np.interp(hf, freqs, amps))
                elif method == "closest":
                    # Find closest frequency
                    idx_closest = np.argmin(np.abs(freqs - hf))
                    hf_amp = amps[idx_closest]
                else:
                    raise ValueError("Invalid method. Use 'interpolate' or 'closest'.")

            highlight_amps.append(hf_amp)
        
        # Plot the highlight points
        ax.scatter(highlight_freqs, highlight_amps, color='red', s=50, marker='o', label='Highlighted Points')

    # Set titles and labels
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # Add grid lines
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    # Reference line at 0 dB
    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='--')
    # Add a legend
    ax.legend(loc='best')
    return fig, ax


def plot_spectrum_with_uncertainty(
    spectrum_mean: np.ndarray,
    spectrum_std: np.ndarray,
    x_values: Optional[np.ndarray] = None,
    n_std: int = 3,
    title: str = "Spectrum with Uncertainty",
    axis_labels: Optional[Tuple[str, str]] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots the average spectrum with uncertainty boundaries.

    :param spectrum_mean: Mean amplitude array.
    :param spectrum_std: Standard deviation array.
    :param x_values: X-axis values. If None, use array index.
    :param n_std: # of standard deviations for the shaded region.
    :param title: Plot title.
    :param axis_labels: (x_label, y_label) for axes.
    :return: (fig, ax)
    """
    if x_values is None:
        x_values = np.arange(len(spectrum_mean))

    x_label, y_label = axis_labels if axis_labels else ('Frequency (Hz)', 'Power (dB)')

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_values, spectrum_mean, linewidth=2.0, color='blue', label='Mean Spectrum')
    ax.fill_between(
        x_values,
        spectrum_mean - spectrum_std * n_std,
        spectrum_mean + spectrum_std * n_std,
        color='blue',
        alpha=0.2,
        label=f'Uncertainty (±{n_std} SD)'
    )
    # Add thicker boundary lines for the shaded region
    ax.plot(x_values, spectrum_mean - spectrum_std * n_std, linewidth=1.0, color='darkblue', linestyle='--')
    ax.plot(x_values, spectrum_mean + spectrum_std * n_std, linewidth=1.0, color='darkblue', linestyle='--')

    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel(x_label, fontsize=9)
    ax.set_ylabel(y_label, fontsize=9)
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.legend(fontsize=9)

    plt.tight_layout()
    return fig, ax
