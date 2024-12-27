# Signal-Quality-Assessment

This repository contains Python scripts to assess the quality of time-series signals. The main steps include:
1. **Reading** raw data from text files.
2. **Segmenting** signals into windows.
3. **Transforming** them to the frequency domain (FFT).
4. **Computing** quality metrics (Spectral Entropy, etc.).
5. **Visualizing** the mean spectrum with uncertainty (standard deviation).
6. **Saving** metrics into timestamped `.yml` files.

## Table of Contents
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Example Commands](#example-commands)
- [Details](#details)

## Directory Structure
```
.
├── .gitignore
├── data/
│   ├── 0702 без перекосов фаза A.txt
│   └── data-sample_motor-operating-at-100%-load.txt
├── environment.yml
├── pyproject.toml
├── res/
│   ├── result.yml
│   └── segment_with_uncertainty.png
├── setup.sh
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── quality_assessment.py
│   ├── utils.py
│   └── vis.py
└── main.py
└── README.md
```
- **data/**: Contains example `.txt` signals.
- **res/**: Output folder for results, plots, and metrics in `.yml` format.
- **src/**: All source code modules.
- **main.py**: Command-line interface script for running signal quality assessment.
- **environment.yml** / **pyproject.toml**: Dependencies configuration.
- **setup.sh**: Setup script.

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/xndrleib/Signal-Quality-Assessment.git
cd Signal-Quality-Assessment
```

### 2. Set Up the Conda Environment
```bash
bash setup.sh
conda activate py311_sqa
```

### 3. Install the Project in Editable Mode
Install the package using the pyproject.toml setup:
```bash
pip install -e .
```

## Usage

### Command-Line Interface
Run the CLI script `main.py` to process either a single file or an entire folder of signals.

```bash
python main.py --input <path_to_file_or_folder> [OPTIONS]
```

**Options**:
- `--res_dir`: Output directory for results (`default: "res"`).
- `--window_length`: Number of samples in each segment (`default: 10000`).
- `--step`: Step size for windowing (`default: 20`).
- `--f_sampling`: Sampling frequency (`default: 10000.0`).
- `--db`: Flag to convert FFT results to dB scale (`default: True`).
- `--cutoff_freq`: Frequency cutoff for filtering (`default: 250.0`).

## Example Commands

1. **Process a single file**:
   ```bash
   python main.py \
       --input data/data-sample_motor-operating-at-100%-load.txt \
       --res_dir res \
       --window_length 10000 \
       --step 20 \
       --f_sampling 10000 \
       --db \
       --cutoff_freq 250
   ```

2. **Process all `.txt` files in a folder**:
   ```bash
   python main.py --input data/
   ```

After completion, the script will generate:
- **Plot of Average Spectrum with Uncertainty** in `res/spectrum_with_uncertainty_data-sample_motor-operating-at-100%-load_20241227_101523.png`.
- **Result metrics** in `.yml` files (e.g., `res/data-sample_motor-operating-at-100%-load_20241227_101523.yml`).

## Details

- **Signal Reading**: Implemented in `src/utils.py`.
- **Segmentation & FFT**: Implemented in `src/data_preprocessing.py`.
- **Quality Metrics**: Spectral Entropy, etc. in `src/quality_assessment.py`.
- **Visualization**: Plot functions in `src/vis.py`.
