# Graph neural network accelerated pressure Poisson solver

## Overview
This repository contains the files used during my master thesis about using graph neural networks to accelerate solving the pressure Poisson equation. The code leverages PyTorch and OpenFOAM to analyze performance metrics across different sample meshes and configurations. The repository includes plotting utilities, training logs, and speed tests for different solver configurations. To use the files

## Dependencies

- python 3.11 (pyfoam is not compatible with later versions)
- OpenFOAM-v2412 + Custom scripts. Add these scripts to your openfoam version.

## Scripts_final

From the maps, only trainTestModel and varyingMeshScripts contain files that are called to train and test the models. The other maps contain scripts that are called from these scripts.

### trainTestModel

##### Plotting Scripts
- `plot_performance_per_sample.py` – Plots the performance of the trained model per sample.
- `plot_performance_per_sample_cfd.py` – Similar to the above, but includes CFD-based computations.
- `plot_train_log.py` – Visualizes training loss curves from log files.

##### Testing Scripts
- `test_PerlinToCFDSpeedtestNIterationsMultipleMeshesCFDInclMag.py` – Tests performance of different solver iterations with Perlin noise-based initialization.
- `test_SpeedtestNIterationsMultipleMeshesCFDInclMag.py` – Runs speed tests with multiple meshes using OpenFOAM.
- `test_testNIterationMultipleMeshesPerlin.py` – Alternative test script for multiple meshes with Perlin noise initialization.

##### Training Scripts
- `train_trainScriptMultipleMeshesAuto.py` – Trains a model on multiple meshes automatically.
- `train_trainScriptMultipleMeshesAutoPerlin.py` – Similar to the above, but includes Perlin noise augmentation.
- `train_trainScriptMultipleMeshesMagAuto.py` – Trains a model with magnitude-based configurations.

### varyingMeshScripts

- `createMeshes.py` – Generates different types of CFD meshes (only called from other scripts).
- `setupCfdDataset.py` – Sets up a CFD dataset by generating meshes and running simulations.
- `setupPerlinNoiseDataset.py` – Generates datasets using Perlin Noise to create variable structures.
- `setupLinearTests.py` – Prepares linear solver tests for benchmarking.


#### Channel Naming Convention

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

## Configuration
Configuration files are located in the `config_files/` directory. Adjust these to modify training parameters, test cases, and other settings.


## Contact
For questions, reach out via email

