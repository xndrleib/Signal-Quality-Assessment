import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def read_data(
        file_path: str,
        output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Reads a text file with semicolon-separated time, data columns.

    :param file_path: Path to the input text file.
    :param output_path: Optional CSV output path.
    :return: DataFrame with columns ['Time', 'Data'].
    """
    time = []
    channel_1 = []

    with open(file_path, 'r', encoding='latin-1') as file:
        lines = file.readlines()

    for line in lines:
        if ";" in line:
            parts = line.strip().split(";")
            time.append(float(parts[0].strip()))
            channel_1.append(float(parts[1].strip()))

    df = pd.DataFrame({'Time': time, 'Data': channel_1})
    if output_path:
        df.to_csv(output_path, index=False)
        logger.info(f"Data saved to {output_path}")

    logger.debug(f"Read {len(df)} rows from {file_path}")
    return df
