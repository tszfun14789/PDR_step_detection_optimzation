[README.md](https://github.com/user-attachments/files/23047396/README.md)
# PDR Step Detection Optimization

Step detection and distance estimation using smartphone IMU sensors for indoor positioning.

## Overview

This project detects walking steps from accelerometer data and estimates walking distance using optimized algorithms.

## Features

- Step detection from accelerometer data
- Parameter optimization (Grid Search, PSO, Bayesian, etc.)
- Real-time PDR processing
- Distance estimation with dynamic step length

## Installation

```bash
pip install pandas numpy matplotlib scipy joblib
```

## Quick Start

### 1. Segment Your Data

```bash
cd src
python Data_segmentation.py
```

### 2. Run Step Detection

```bash
cd src
python stepdetection.py
```

### 3. Run Real-time PDR

```bash
cd src
python run_pdr_realtime.py
```

## Results Example

![Step Detection Results](PDR_step_detection_optimization/assets/step_detection_example.png)

The visualization shows:
- **Top**: Raw and filtered acceleration with detected steps (red dots)
- **Bottom**: Step length distribution (short/medium/long steps)

## Project Structure

```
├── assets/                  # Raw IMU data
├── src/
│   ├── stepdetection.py                # Main step detection algorithm
│   ├── Data_segmentation.py            # Segment raw data
│   ├── pdr_realtime_processor.py       # Real-time PDR
│   ├── run_pdr_realtime.py            # Run PDR processing
│   ├── compare_algorithms.py           # Compare algorithms
│   ├── optimized_parameters_simple.json # Optimized parameters
│   ├── combined_path_segments/         # Processed segments
│   └── step_detection_results/         # Output results
```

## Optimized Parameters

Current optimized values:
- `step_threshold`: 0.599
- `min_peak_distance`: 59 samples
- `base_step_length`: 65.24 cm

## How It Works

1. **Pre-processing**: Remove gravity, apply filters
2. **Peak Detection**: Find step peaks in acceleration data
3. **Step Classification**: Categorize as short/medium/long steps
4. **Distance Estimation**: Calculate total distance using weighted step lengths

## Basic Usage

```python
from stepdetection import StepDetectionAlgorithm

# Initialize
detector = StepDetectionAlgorithm(
    step_threshold=0.599,
    min_peak_distance=59
)

# Process segments
results = detector.process_all_segments(
    segments_dir="combined_path_segments",
    output_dir="step_detection_results"
)
```

## Compare Algorithms

```bash
cd src
python compare_algorithms.py
```

This compares the original algorithm vs. real-time PDR and generates comparison plots.

## License

For research and educational purposes.

