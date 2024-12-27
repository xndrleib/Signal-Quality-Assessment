import matplotlib.pyplot as plt
import numpy as np


def plot_random_segments(segments, num_segments=10, labels=None, axis_labels=None, x_values=None):
    """
    Plots random segments from the provided segments.

    Parameters:
    - segments: Array of segments to plot.
    - num_segments: Number of random segments to plot.
    - labels: Optional list of custom labels for the segments.
    - axis_labels: Optional tuple (x_label, y_label) for x and y axis labels.
    - x_values: Optional array of x-values corresponding to the peak segments.
    """
    if len(segments) < num_segments:
        raise ValueError("Not enough segments to plot")

    random_indices = np.random.choice(len(segments), num_segments, replace=False)
    selected_segments = segments[random_indices]

    if labels is not None:
        if len(labels) != len(segments):
            raise ValueError("Length of labels must match the length of peak_segments")
        selected_labels = [labels[i] for i in random_indices]
    else:
        selected_labels = [f"Segment {i}" for i in random_indices]

    # Setup subplot grid
    fig, axes = plt.subplots(num_segments // 2, 2, figsize=(15, 5 * (num_segments // 2)))
    axes = axes.flatten()

    x_label, y_label = axis_labels if axis_labels else ('Frequency (Hz)', 'Power (dB)')

    for i, ax in enumerate(axes):
        segment = selected_segments[i]

        if x_values is not None:
            if len(x_values.shape) == 1:
                x_segment = x_values[:len(segment)]
            else:
                x_segment = x_values[random_indices[i]]
        else:
            x_segment = np.linspace(0, len(segment), len(segment))

        ax.plot(x_segment, segment, linewidth=1.5, color='blue')
        ax.set_title(selected_labels[i], fontsize=10, fontweight='bold')
        ax.set_xlabel(x_label, fontsize=9)
        ax.set_ylabel(y_label, fontsize=9)
        ax.grid(True, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    return fig, ax


def plot_spectrum(df, freq_col="Frequency (Hz)", amp_col="Amplitude",
                  title="Frequency Spectrum", xlabel="Frequency (Hz)",
                  ylabel="Amplitude (dB)", highlight_freqs=None, method="interpolate"):
    """
    Plots the frequency spectrum with optional highlighted points using interpolation or closest point methods.

    Parameters:
    - df: Pandas DataFrame containing frequency and amplitude data.
    - freq_col: Column name for frequency data (default: "Frequency (Hz)").
    - amp_col: Column name for amplitude data (default: "Amplitude").
    - title: Title of the plot (default: "Frequency Spectrum").
    - xlabel: Label for the x-axis (default: "Frequency (Hz)").
    - ylabel: Label for the y-axis (default: "Amplitude (dB)").
    - highlight_freqs: List of frequency values to highlight as points.
    - method: Method to handle frequencies not in DataFrame. Options: "interpolate" or "closest".
              "interpolate" uses linear interpolation between known points.
              "closest" finds the closest existing frequency in df.
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
                    hf_amp = np.interp(hf, freqs, amps)
                elif method == "closest":
                    # Find closest frequency
                    idx_closest = np.argmin(np.abs(freqs - hf))
                    hf_amp = amps[idx_closest]
                else:
                    raise ValueError("Invalid method. Use 'interpolate' or 'closest'.")

            highlight_amps.append(hf_amp)

        # Plot the highlight points
        ax.scatter(highlight_freqs, highlight_amps, color='red', s=50, marker='o', 
                   label='Highlighted Points')

    # Set titles and labels
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Add grid lines
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Reference line at 0 dB (optional)
    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='--')

    # Add a legend
    ax.legend(loc='best')

    return fig, ax


def plot_spectrum_with_uncertainty(spectrum_mean, spectrum_std, x_values=None, n_std=2, title="Spectrum with Uncertainty",
                                         axis_labels=None):
    """
    Plots the average spectrum with uncertainty boundaries.

    Parameters:
    - x_values: Optional x-axis values corresponding to the spectrum.
    - n_std: Number of standard deviations for the uncertainty boundary.
    - title: Title for the spectrum.
    - axis_labels: Optional tuple (x_label, y_label) for axis labels.
    """

    if x_values is None:
        x_values = np.linspace(0, len(spectrum_mean), len(spectrum_mean))

    # Plot
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