# BlueSkySim RL Training

This project uses reinforcement learning algorithms to train air traffic control models using the BlueSkySim environment.

## Training Scripts

### Main Training Script

The `main.py` script trains three different RL algorithms (DDPG, PPO, and SAC) on the MergeEnv-v0 environment and collects performance metrics.

To run the training:

```bash
python main.py
```

This will:

1. Train all three algorithms for the specified number of steps
2. Track environment metrics (total reward, FAF reach, average drift, total intrusions)
3. Save model checkpoints and collected data

### Visualization Script

After training, use the `plot_results.py` script to generate detailed visualizations:

```bash
python plot_results.py --log_dir logs/ --save_dir figures/ --show
```

Arguments:

- `--log_dir`: Directory containing the training logs (default: `logs/`)
- `--save_dir`: Directory to save generated figures (default: `figures/`)
- `--show`: Flag to display plots interactively (optional)

## Generated Visualizations

The visualization script generates multiple plots:

1. **Individual Metric Comparisons**: Separate plots for each metric, comparing all models
2. **Metrics Dashboard**: A combined view of all metrics in a 2x2 grid
3. **Normalized Metrics**: Scale-independent comparison of metrics across models
4. **Per-Model Progression**: Multi-axis plots showing how all metrics evolve over time for each model

## Metrics Tracked

- **Episode Reward**: Total cumulative reward per episode
- **FAF Reach**: Final Approach Fix reach metric
- **Average Drift**: Average aircraft drift during the episode
- **Total Intrusions**: Count of airspace intrusions during the episode
