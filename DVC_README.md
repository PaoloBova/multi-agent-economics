# DVC Pipeline for Multi-Agent Economics Simulation

This document describes the DVC (Data Version Control) pipeline for reproducible multi-agent economics simulations.

## Pipeline Overview

Our DVC pipeline consists of three main stages:

### 1. Data Preparation (`prepare_data`)
- **Script**: `scripts/prepare_data.py`
- **Input**: Configuration from `simulation_config.yaml`
- **Output**: `data/processed/initial_agents.csv`
- **Purpose**: Prepare initial agent data based on simulation parameters

### 2. Simulation Execution (`run_simulation`)
- **Script**: `scripts/run_simulation.py`
- **Input**: Processed agent data, simulation configuration
- **Output**: 
  - `results/simulation_output.csv` - Market data over time
  - `results/agent_behaviors.json` - Individual agent performance
- **Purpose**: Run the multi-agent economics simulation using AutoGen

### 3. Results Analysis (`analyze_results`)
- **Script**: `scripts/analyze_results.py`
- **Input**: Simulation results
- **Output**: 
  - `results/analysis/summary.json` - Statistical summary
  - `results/plots/economic_indicators.png` - Market trends
  - `results/plots/agent_wealth_distribution.png` - Wealth distribution
- **Purpose**: Analyze results and generate visualizations

## Directory Structure

```
data/
├── raw/              # Raw input data (tracked by DVC)
├── interim/          # Intermediate processing results (tracked by DVC)
└── processed/        # Final processed data ready for simulation (tracked by DVC)

results/
├── analysis/         # Analysis outputs (tracked by DVC)
└── plots/           # Generated visualizations (tracked by DVC)

scripts/
├── prepare_data.py   # Data preparation script
├── run_simulation.py # Simulation execution script
└── analyze_results.py # Results analysis script
```

## Usage

### Running the Full Pipeline
```bash
# Run all stages
dvc repro

# Run specific stage
dvc repro run_simulation

# Force re-run (ignore cache)
dvc repro --force
```

### Checking Pipeline Status
```bash
# Check what stages need to be run
dvc status

# Show pipeline DAG
dvc dag

# Show pipeline metrics
dvc metrics show
```

### Data Management
```bash
# Track new data files
dvc add data/raw/new_dataset.csv

# Push data to remote storage (after configuring remote)
dvc push

# Pull data from remote storage
dvc pull
```

## Configuration

### Simulation Parameters
Edit `simulation_config.yaml` to modify:
- Number of agents
- Simulation duration
- Economic parameters
- Agent types and behaviors

### DVC Configuration
The pipeline is defined in `dvc.yaml` with:
- Dependencies tracking
- Output file management
- Parameter monitoring
- Plot generation rules

## Remote Storage Setup (Optional)

To share data and results across team members:

```bash
# Add remote storage (example with S3)
dvc remote add -d myremote s3://my-bucket/dvc-storage

# Add remote storage (example with local shared folder)
dvc remote add -d myremote /shared/dvc-storage

# Configure remote
dvc remote modify myremote access_key_id mykey
dvc remote modify myremote secret_access_key mysecret
```

## Best Practices

1. **Version Control**: Commit `dvc.yaml`, `dvc.lock`, and `.dvc` files to git
2. **Data Tracking**: Use `dvc add` for large data files
3. **Reproducibility**: Always run `dvc repro` after parameter changes
4. **Documentation**: Update this README when adding new pipeline stages

## Troubleshooting

### Common Issues

1. **Missing dependencies**: Ensure all Python packages are installed
   ```bash
   poetry install
   ```

2. **Permission errors**: Check file permissions for scripts
   ```bash
   chmod +x scripts/*.py
   ```

3. **Pipeline errors**: Check individual stage outputs
   ```bash
   dvc repro --verbose
   ```

4. **Data conflicts**: Reset and re-run pipeline
   ```bash
   dvc repro --force
   ```

## Next Steps

- [ ] Implement actual AutoGen agents in `run_simulation.py`
- [ ] Add more sophisticated economic models
- [ ] Configure remote storage for data sharing
- [ ] Add model training and evaluation stages
- [ ] Implement parameter sweeps for experiments
