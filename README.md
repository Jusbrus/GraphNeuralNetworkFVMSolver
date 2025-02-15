# Graph neural network accelerated pressure Poisson solver

## Overview
This repository contains the files used during my master thesis about using graph neural networks to accelerate solving the pressure Poisson equation. The repository includes plotting utilities, training scripts, and performance tests. To use the code, one must first set up the datasets using the scripts in the "varyingMeshScripts" map.

## Dependencies

- python 3.11 (pyfoam is not compatible with later versions)
- OpenFOAM-v2412 + Custom scripts. Add these scripts to your openfoam version.

## Scripts_final

From the maps, only trainTestModel and varyingMeshScripts contain files that are called to train and test the models. The other maps contain scripts called from these scripts.

### trainTestModel

##### Plotting Scripts
- `plot_performance_per_sample.py` – Plots performance per sample for the Perlin noise data.
- `plot_performance_per_sample_cfd.py` – Similar to the above, but includes CFD-based computations.
- `plot_train_log.py` – Visualizes training loss curves from log files.

##### Testing Scripts
- `test_PerlinToCFDSpeedtestNIterationsMultipleMeshesCFDInclMag.py` –Tests Perlin Noise models on CFD data.
- `test_SpeedtestNIterationsMultipleMeshesCFDInclMag.py` – Tests CFD models on CFD data.
- `test_testNIterationMultipleMeshesPerlin.py` – Test Perlin noise models on Perlin noise data.

##### Training Scripts
- `train_trainScriptMultipleMeshesAuto.py` – Trains a model on multiple meshes automatically.
- `train_trainScriptMultipleMeshesAutoPerlin.py` – Similar to the above, but includes Perlin noise augmentation.
- `train_trainScriptMultipleMeshesMagAuto.py` – Trains a model with magnitude-based configurations.

### varyingMeshScripts

- `createMeshes.py` – Generates different types of CFD meshes (only called from other scripts).
- `setupCfdDataset.py` – Sets up a CFD dataset by generating meshes and running simulations.
- `setupPerlinNoiseDataset.py` – Generates datasets using Perlin Noise to create a diverse dataset.
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
- The root directory should be "/home/justinbrusche/"
- Modify the `config_file` in the script to specify different configurations.


### Testing
Run a test script as follows:
```bash
python test_SpeedtestNIterationsMultipleMeshesCFDInclMag.py
```

## Configuration
Configuration files are located in the `config_files/` directory. Adjust these to modify training parameters, test cases, and other settings.

## License
This project is part of a master thesis of the TU Delft.

## Contact
For questions, reach out via email

