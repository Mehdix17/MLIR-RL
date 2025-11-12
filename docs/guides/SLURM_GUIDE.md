# SLURM Job Management Guide

## Submit Training Job

```bash
sbatch scripts/train.sh
```

Output:
```
Submitted batch job 12345678
```

## Monitor Your Job

### Check job status
```bash
squeue -u $USER
```

Output:
```
JOBID    PARTITION  NAME          USER     ST  TIME  NODES
12345678 compute    p-original-c  mb10856  R   1:23  1
```

**Status codes:**
- `PD` = Pending (waiting for resources)
- `R` = Running
- `CG` = Completing
- `CD` = Completed

### Detailed job info
```bash
scontrol show job 12345678
```

### Check all your recent jobs
```bash
sacct -u $USER
```

### Check only today's jobs
```bash
sacct -u $USER --starttime=today
```

## View Job Output

### Real-time output (while running)
```bash
tail -f logs/p-original-c_12345678.out
```

Press `Ctrl+C` to stop following.

### View errors
```bash
tail -f logs/p-original-c_12345678.err
```

### View full output
```bash
less logs/p-original-c_12345678.out
```

## Cancel Job

### Cancel specific job
```bash
scancel 12345678
```

### Cancel all your jobs
```bash
scancel -u $USER
```

### Cancel by job name
```bash
scancel --name=p-original-c
```

## Common Issues

### Job stays in PENDING (PD)
**Cause**: Waiting for resources or reservation  
**Solution**: Check queue:
```bash
squeue -p compute
```

### Job fails immediately
**Cause**: Usually environment or path issues  
**Solution**: Check error log:
```bash
cat logs/p-original-c_<JOB_ID>.err
```

### Cannot find logs
**Cause**: Logs directory doesn't exist  
**Solution**: Create it:
```bash
mkdir -p logs
```

## Job Arrays (Run Multiple Training Runs)

To run multiple training runs with different configs:

```bash
sbatch --array=1-5 scripts/train.sh
```

This submits 5 jobs, each with `$SLURM_ARRAY_TASK_ID` set to 1, 2, 3, 4, 5.

## Useful Commands Summary

| Command | Purpose |
|---------|---------|
| `sbatch script.sh` | Submit job |
| `squeue -u $USER` | Check your jobs |
| `scancel JOBID` | Cancel job |
| `scontrol show job JOBID` | Job details |
| `sacct` | Job history |
| `tail -f logs/job.out` | Watch output |

## Email Notifications

Your `train.sh` is configured to email you at `mb10856@nyu.edu` when:
- Job fails (`--mail-type=FAIL`)
- Job hits time limit (`--mail-type=TIME_LIMIT`)

To get notified on completion too, modify `train.sh`:
```bash
#SBATCH --mail-type=FAIL,TIME_LIMIT,END
```

## Resource Modifications

### Change time limit
```bash
sbatch --time=2-00:00:00 scripts/train.sh  # 2 days
```

### Change memory
```bash
sbatch --mem=200G scripts/train.sh
```

### Change partition
```bash
sbatch --partition=gpu scripts/train.sh
```

## Tips

1. **Always check logs** after submission to ensure job started correctly
2. **Use `tail -f`** to monitor progress in real-time
3. **Save job ID** for later reference
4. **Check queue** if job doesn't start immediately

## Example Session

```bash
# Submit job
$ sbatch scripts/train.sh
Submitted batch job 12345678

# Check status
$ squeue -u $USER
JOBID    PARTITION  NAME          USER     ST  TIME  NODES
12345678 compute    p-original-c  mb10856  R   0:45  1

# Watch output
$ tail -f logs/p-original-c_12345678.out
Loading module 'miniconda-nobashrc/3-4.11.0'
11-09 16:45 - [INFO] Logging to: results/run_12
11-09 16:45 - [SUCCESS] Model initialized
...

# Disconnect and come back later
$ exit

# Reconnect and check
$ squeue -u $USER
JOBID    PARTITION  NAME          USER     ST  TIME  NODES
12345678 compute    p-original-c  mb10856  R   5:23  1

# Still running! Check progress
$ tail -20 logs/p-original-c_12345678.out
```

---

**Your training will continue even after you disconnect!**
