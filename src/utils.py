import logging
import pandas as pd
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

def load_preprocessed_data(file_path: str) -> pd.DataFrame:
    """
    Reads a preprocessed file.

    :param file_path: Path to the original input file.
    :return: DataFrame with Time as index and sensor channels as columns.
    """
    file_path = Path(file_path)

    # Load preprocessed data
    df = pd.read_csv(file_path, index_col=0)
    return df


def preprocess_txt_file(file_path: str, output_dir: str, n_channels: int = 1) -> None:
    """
    Preprocesses an oscilloscope .txt file:
    - Extracts metadata and saves it as a .txt file.
    - Extracts numerical data and saves it as a multi-channel .csv file.

    :param file_path: Path to the input .txt file.
    :param output_dir: Directory where the .csv and metadata.txt files will be saved.
    :param n_channels: Number of data channels in the file.
    """
    file_path = Path(file_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

    csv_path = output_dir / f"{file_path.stem}.csv"
    metadata_path = output_dir / f"{file_path.stem}_metadata.txt"

    # Skip if preprocessed files already exist
    if csv_path.exists() and metadata_path.exists():
        logger.info(f"Preprocessed files already exist for {file_path.stem}. Skipping processing.")
        return

    metadata = []
    time = []
    data_channels = [[] for _ in range(n_channels)]  # Create a list for each channel

    with open(file_path, 'r', encoding='cp1251') as file:
        lines = file.readlines()

    # Extract metadata (all lines before first occurrence of ';')
    for line in lines:
        if ";" not in line:
            metadata.append(line)
        else:
            break  # Stop collecting metadata once numerical data starts

    # Save metadata as a .txt file
    with open(metadata_path, "w", encoding="utf-8") as meta_file:
        meta_file.write("".join(metadata))

    logger.info(f"Metadata extracted and saved to {metadata_path}")

    # Find where numerical data starts
    start_index = lines.index("Data as Time Sequence:\n") + 4  # Skip headers

    # Extract numerical data
    for line in lines[start_index:]:
        parts = line.strip().split(";")
        try:
            time.append(float(parts[0].strip()))  # First column is always time
            for i in range(n_channels):
                data_channels[i].append(float(parts[i + 1].strip()))
        except ValueError:
            logger.warning(f"Skipping malformed line: {line.strip()}")

    # Create DataFrame with time as index
    column_names = ["Time"] + [f"Ch_{i+1}" for i in range(n_channels)]
    df = pd.DataFrame({'Time': time, **{col: data_channels[i] for i, col in enumerate(column_names[1:])}})
    df.set_index("Time", inplace=True)

    # Save processed numerical data as CSV
    df.to_csv(csv_path)

    logger.info(f"Processed numerical data saved to {csv_path}")


def preprocess_csv_file(file_path: str, output_dir: str) -> None:
    """
    Preprocesses a sensor CSV file:
    - Converts timestamps to relative time in seconds.
    - Saves the processed data as a structured CSV.

    :param file_path: Path to the input CSV file.
    :param output_dir: Directory where processed files will be saved.
    """
    file_path = Path(file_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

    csv_path = output_dir / f"{file_path.stem}.csv"

    # Skip if preprocessed files already exist
    if csv_path.exists():
        logger.info(f"Preprocessed files already exist for {file_path.stem}. Skipping processing.")
        return

    df = pd.read_csv(file_path)

    # Validate required columns
    if "timestamp" not in df.columns:
        logger.error(f"CSV file {file_path} missing 'timestamp' column.")
        return

    # Convert timestamp to relative time
    df['Time'] = (df['timestamp'] - df['timestamp'].iloc[0])
    df.set_index("Time", inplace=True)
    df.sort_index(inplace=True)

    # Drop original timestamp column
    df.drop(columns=["timestamp"], inplace=True)

    # Save processed numerical data as CSV
    df.to_csv(csv_path)

    logger.info(f"Processed numerical data saved to {csv_path}")