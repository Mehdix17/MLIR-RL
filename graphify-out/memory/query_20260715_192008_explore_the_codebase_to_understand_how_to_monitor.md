---
type: "query"
date: "2026-07-15T19:20:08.786138+00:00"
question: "explore the codebase to understand how to monitor and submit jobs (training and eval)"
contributor: "graphify"
outcome: "useful"
source_nodes: ["Training and Evaluation Guide", "submit_eval.py", "sync_progress.py", "report_training.py", "submit_and_monitor.sh"]
---

# Q: explore the codebase to understand how to monitor and submit jobs (training and eval)

## Answer

Expanded from original query via vocab: ['submit', 'monitor', 'job', 'jobs', 'train', 'training', 'eval', 'evaluation', 'slurm', 'progress', 'report']

### MLIR-RL Job Submission & Monitoring Guide

#### 1. Job Submission
- **Training**:
  - Training is initiated via Slurm using .
  - To start training from scratch:
    
  - To resume training from an existing run:
    
  - Alternatively, training can be submitted and monitored interactively using:
    

- **Evaluation**:
  - Large-scale batch evaluations are submitted using the helper script :
    
    Example:
    Successfully synced progress to docs/results/eval_progress.md
Resolved Config Path: /scratch/mb10856/MLIR-RL/config/ops_and_blocks/eval/paper_transformer_small_eval.json
Agent Version: paper_transformer_small
Range: 7300 to 10200 with step 100
Submitted batch job 16742686
Registered Job 16742686 in active jobs cache.
    This script resolves the target configuration, invokes Submitted batch job 16742687 under the hood, registers the job in , and triggers progress synchronization.
  - Alternatively, you can directly launch the batch script:
    
  - You can also evaluate a single checkpoint using:
    

#### 2. Job Monitoring & Progress Reporting
- **Live Output Tail**:
  - You can submit jobs and automatically tail their outputs in the terminal using the  wrapper:
    
    This script prints the Slurm job details, waits for the log file to be created, and monitors the log stream in real time.

- **Monitoring Scripts**:
  - **Training Progress**: Use  to output a clean table of iterations, durations, average step time, and speedup trends of training runs.
  - **Evaluation Progress**: Run Successfully synced progress to docs/results/eval_progress.md to synchronize active Slurm jobs, check completed checkpoints, and output a detailed status to .
  - **Evaluation Performance**: Use  to compile speedup results across checkpoints and identify the best-performing checkpoints.
  - **Unified Report Slash Command**: Inside the Antigravity TUI, typing  executes the  skill which runs , , and queries Disk quotas for usr mb10856 (uid 4709778):
     Filesystem  kbytes   quota   limit   grace   files   quota   limit   grace
       /scratch 285181364  5242880000 5347737600       -  316399  500000 1000000       - to verify storage limit boundaries (500K soft file limit).


## Outcome

- Signal: useful

## Source Nodes

- Training and Evaluation Guide
- submit_eval.py
- sync_progress.py
- report_training.py
- submit_and_monitor.sh