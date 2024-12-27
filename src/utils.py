import numpy as np
import pandas as pd


def read_data(file_path, output_path=None):
    time = []
    channel_1 = []

    # Open the file and process lines
    with open(file_path, 'r', encoding='latin') as file:
        lines = file.readlines()

        for line in lines:
            # Check if the line contains data with a semicolon separator
            if ";" in line:
                # Split the line into parts
                parts = line.strip().split(";")

                time.append(float(parts[0].strip()))
                channel_1.append(float(parts[1].strip()))

    df = pd.DataFrame({'Time': time, 'Data': channel_1})
    if output_path:
        df.to_csv(output_path, index=False)
    return df
