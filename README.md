# Gym Air Quality

This project implements a custom Gymnasium environment for an energy-aware air-quality sensor transmission problem. It compares standard Q-learning with a structural-knowledge variant and saves the learned policies and score histories as NumPy files.

## Files

- `GymAirQuality.py` defines the `SensorTransmissionEnv` Gymnasium environment.
- `train.py` trains both learning methods and saves the output arrays.
- `air.npy` and `solar.npy` contain the transition data used by the environment.
- `policy1.npy` and `policy2.npy` are saved learned policies.
- `qlearning_scores.npy` and `structural_scores.npy` contain evaluation scores collected during training.
- `report.pdf` contains the project report.

## Setup

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Run

From this directory, run:

```bash
python train.py
```

The script trains both agents for 10,000 episodes and writes the learned policies and score arrays to `.npy` files.

## Requirements

- Python 3
- NumPy
- Gymnasium

## License

This project is licensed under the MIT License. See `LICENSE` for details.
