# Graph neural network accelerated pressure Poisson solver

## Overview
This repository contains scripts for training, evaluating, and testing machine learning models for computational fluid dynamics (CFD) simulations. The code leverages PyTorch and OpenFOAM to analyze performance metrics across different sample meshes and configurations. The repository includes plotting utilities, training logs, and speed tests for different solver configurations.

## File Structure

### Plotting Scripts
- `plot_performance_per_sample.py` – Plots the performance of the trained model per sample.
- `plot_performance_per_sample_cfd.py` – Similar to the above, but includes CFD-based computations.
- `plot_train_log.py` – Visualizes training loss curves from log files.

### Testing Scripts
- `test_PerlinToCFDSpeedtestNIterationsMultipleMeshesCFDInclMag.py` – Tests performance of different solver iterations with Perlin noise-based initialization.
- `test_SpeedtestNIterationsMultipleMeshesCFDInclMag.py` – Runs speed tests with multiple meshes using OpenFOAM.
- `test_testNIterationMultipleMeshesPerlin.py` – Alternative test script for multiple meshes with Perlin noise initialization.

### Training Scripts
- `train_trainScriptMultipleMeshesAuto.py` – Trains a model on multiple meshes automatically.
- `train_trainScriptMultipleMeshesAutoPerlin.py` – Similar to the above, but includes Perlin noise augmentation.
- `train_trainScriptMultipleMeshesMagAuto.py` – Trains a model with magnitude-based configurations.

### Channel Naming Convention

This table describes the naming convention used for different channel sizes.

| Channels | Named              |
|----------|--------------------|
| 6        | veryVeryVerySmall  |
| 8        | veryVerySmall      |
| 12       | verySmall          |
| 16       | small              |
| 24       | medium             |
| 32       | fullscale          |



## Usage

### Training
To train a model, use:
```bash
python train_trainScriptMultipleMeshesAuto.py
```

Modify the `config_file_list` in the script to specify different configurations.

### Testing
Run a test script as follows:
```bash
python test_SpeedtestNIterationsMultipleMeshesCFDInclMag.py
```

Ensure that OpenFOAM is properly sourced before running.

### Plotting
To visualize training logs:
```bash
python plot_train_log.py
```

## Configuration
Configuration files are located in the `config_files/` directory. Adjust these to modify training parameters, test cases, and other settings.

## License
This project is for research purposes only. Please contact the author before distribution or modification.

## Contact
For questions, reach out via email or GitHub issues.

