# Onboarding: Monitoring, Metrics, & Evaluation

> **Module 9**: A practical guide to monitoring training runs, analyzing experimental metrics, using reporting scripts, running evaluations, and navigating the comparison dashboard.

---

## 1. Monitoring Active Training Runs

Training PPO models on large MLIR datasets is computationally heavy. We monitor runs in three ways:

### A. Terminal / Standard Output
The stdout of `train.sh` prints iteration stats at every step:
```
Iter: 120 | Loss: 0.125 | Value Loss: 0.045 | Mean Reward: 0.450 | Speedup: 1.82x | Entropy: 1.250 | Clip Frac: 0.05
```
- **Entropy**: Watch this closely. A healthy run starts with high entropy ($1.5$ to $3.0$) and drops slowly. If entropy drops to $0.0$ or near-zero within the first few hundred iterations, it indicates **Entropy Collapse** (the agent has become deterministic and stopped exploring).
- **Mean Speedup**: Represents the speedup ratio relative to the unoptimized baseline. A value of $1.5$ means the compiled loops are running 50% faster on average.

### B. Neptune.ai Integration
If `"logging": true` is set in the configuration JSON, metrics are pushed to your Neptune dashboard in real time.
- Track curves for `train/reward`, `train/entropy`, `train/final_speedup`, `train_ppo/policy_loss`, and `train_ppo/value_loss`.
- Check `approx_kl` (Approximate Kullback-Leibler divergence). If KL spikes above $0.05$ consistently, the policy is changing too rapidly, and you may need to reduce the learning rate.

### C. FileLogger (Crash Resilience)
If the process is killed unexpectedly, `FileLogger` writes results incrementally to:
`results/<experiment>/<agent>/run_N/train/results.json`

---

## 2. Using Reporting Scripts (`scripts/utils/`)

Instead of parsing raw log files, we use reporting scripts to check training progress:

### A. Training Progress Summary
Run this to see a structured table of active and completed training runs:
```bash
python scripts/utils/report_training.py -v v4_6 v4_8 v4_9
```
- **Options**:
  * `-w <seconds>`: Runs the script in watch mode, auto-updating the console (e.g., `report_training.py -w 300`).

### B. Evaluation Progress
Once training is finished and you have run evaluations, check the performance of your checkpoints:
```bash
# Show summary of all evaluation runs
python scripts/utils/report_eval.py

# Find only the best performing checkpoint per agent
python scripts/utils/report_eval.py --best

# Find checkpoints that have not been evaluated yet
python scripts/utils/report_eval.py --missing
```

---

## 3. Running Standalone Evaluations (`eval.sh`)

Evaluations run deterministically (meaning the policy selects the most likely action instead of sampling randomly).

Submit evaluation jobs via Slurm:
```bash
# Evaluate checkpoint 500 of the v4_9_large configuration
sbatch --cpus-per-task=12 --mem=16G --time=04:00:00 \
  scripts/eval/eval.sh config/single_ops_dataset/eval/v4_9_large_eval.json --checkpoint 500
```

> [!TIP]
> Assigning `--cpus-per-task=12` is recommended because evaluation executes over the entire test set split (typically 1,600 to 2,100 benchmarks). Higher core count speeds up JIT runs.

---

## 4. Key Experimental Metrics

Our performance comparisons use several standard statistical metrics:

1. **CDF (Cumulative Distribution Function) Speedup Curve**:
   - Plots the percentage of benchmarks (y-axis) that achieve a speedup less than or equal to a given value (x-axis). A curve shifted to the right indicates superior scheduling.
2. **Win / Loss / Tie Matrix**:
   - Compares Agent A vs Agent B head-to-head across all benchmarks:
     * **Win**: Agent A's schedule runs faster than Agent B's by $>5\%$.
     * **Loss**: Agent A's schedule runs slower than Agent B's by $>5\%$.
     * **Tie**: Within $5\%$ margin.
3. **Wilcoxon Signed-Rank Test**:
   - A non-parametric statistical hypothesis test used to compare two paired groups. It determines if the speedup difference between Agent A and Agent B is statistically significant ($p < 0.05$) or if it is just measurement noise.

---

## 5. The Comparison Dashboard

We use a Streamlit-based dashboard to visualize and compare results.

```bash
# Start Streamlit dashboard
cd dashboard
streamlit run dashboard.py
```

### Key Dashboard Tabs

- **Exec Times**: Shows bar charts comparing execution times across the RL agent, the unoptimized MLIR baseline, and PyTorch (compile/jit).
- **Speedup CDF**: Displays the speedup cumulative distribution curves.
- **Head-to-Head**: Displays pairwise scatterplots comparing execution times between selected agents, calculating win/loss/tie counts and Wilcoxon p-values.
- **Schedules**: Lists the actual compiler schedules (the sequence of loop transforms) selected by the agent. This is useful to verify if the agent is finding interesting scheduling combinations or if it is just outputting basic tiling.

---

## 6. Lustre File Quota Safety

Our experiments run on HPC clusters using **Lustre high-performance filesystems** (typically mounted on `/scratch`). 

> [!CAUTION]
> Lustre filesystems enforce strict **file count (inode) limits** per user (usually a soft limit of 500,000 files and a hard limit of 1,000,000). 

Because our experiments generate many files:
- Each model checkpoint is a separate `.pt` file ($\approx 45\text{MB}$).
- Eval runs write individual execution time logs per benchmark.

Before submitting large training sweeps or evaluating multiple checkpoints, check your file quota limit:
```bash
lfs quota -u $USER /scratch
```
If you are close to the file limit, clean up old run folders or contact the user before starting new runs to avoid crashing the cluster storage node.
