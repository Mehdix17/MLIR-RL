# Neptune Auto-Sync Feature

## Overview

Training automatically syncs results to Neptune when training completes. No manual intervention required!

## How It Works

When `train.py` finishes:
1. ✅ Final evaluation is performed
2. 🎨 Plots are automatically generated
3. 🌊 Results are synced to Neptune
4. 🔗 Neptune URL is displayed

## Requirements

Set these environment variables in `.env`:
```bash
NEPTUNE_PROJECT=your-workspace/your-project
NEPTUNE_TOKEN=your-api-token
```

## Configuration

### Check if Auto-Sync is Enabled

```bash
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
if os.getenv('NEPTUNE_PROJECT') and os.getenv('NEPTUNE_TOKEN'):
    print('✓ Auto-sync enabled')
else:
    print('⚠ Auto-sync disabled (set NEPTUNE_PROJECT and NEPTUNE_TOKEN in .env)')
"
```

### Enable Auto-Sync

Edit `.env` and add:
```bash
NEPTUNE_PROJECT=mehdix/mlir-project
NEPTUNE_TOKEN=eyJhcGl...your-token-here
```

### Disable Auto-Sync

Comment out or remove Neptune credentials from `.env`:
```bash
# NEPTUNE_PROJECT=mehdix/mlir-project
# NEPTUNE_TOKEN=eyJhcGl...
```

## Usage

### Standard Training (with auto-sync)

```bash
bash scripts/train.sh
```

Training output will show:
```
[INFO] Training complete!
[INFO] Results saved to: results/run_12
[INFO] Syncing results to Neptune...
======================================================================
SYNCING: run_12
======================================================================
✓ Neptune run created: MLIR-123
  URL: https://app.neptune.ai/mehdix/mlir-project/e/MLIR-123
✓ Generating plots...
✓ Uploaded 4 plots
✓ Successfully synced to Neptune!
```

### Manual Sync (if auto-sync disabled)

```bash
python experiments/sync_neptune_with_plots.py results/run_12
```

## What Gets Synced

### Metrics
- Training rewards
- Policy and value losses
- Entropy and KL divergence
- Evaluation speedups

### Plots
- Speedup by operation type
- Geometric mean speedup
- Per-benchmark speedup
- Training metrics

### Metadata
- Configuration (from `config/config.json`)
- Tags (from `results/run_X/tags`)
- Run name

## Timeout

Auto-sync has a 5-minute timeout. If sync takes longer:
- Training completes successfully
- Sync times out with warning message
- Use manual sync: `python experiments/sync_neptune_with_plots.py results/run_X`

## Error Handling

If auto-sync fails:
- ✅ Training results are **still saved** locally
- ⚠️ Error message is displayed
- 📝 Manual sync command is shown

Example:
```
⚠ Neptune sync failed (exit code 1)
Error: Connection timeout
[INFO] To sync manually: python experiments/sync_neptune_with_plots.py results/run_12
```

## Troubleshooting

### "Neptune sync skipped"
**Cause**: NEPTUNE_PROJECT or NEPTUNE_TOKEN not set  
**Solution**: Add credentials to `.env`

### "Connection timeout"
**Cause**: Network issues or Neptune service unavailable  
**Solution**: Try manual sync later:
```bash
python experiments/sync_neptune_with_plots.py results/run_12
```

### "Invalid credentials"
**Cause**: Wrong API token or project name  
**Solution**: Check credentials:
```bash
# Test Neptune connection
python experiments/test_neptune.py
```

### "Module not found: neptune"
**Cause**: Neptune not installed  
**Solution**: 
```bash
conda activate mlir
pip install neptune
```

## Benefits

✅ **Zero manual work** - Results automatically uploaded  
✅ **Immediate visibility** - View results as soon as training ends  
✅ **No forgotten syncs** - Never forget to upload results  
✅ **Consistent workflow** - Same process every time  
✅ **Safe fallback** - Local results always saved first

## Alternative: Real-Time Sync

For monitoring during training (optional):
```bash
# Terminal 1: Training
bash scripts/train.sh

# Terminal 2: Real-time sync
bash scripts/neptune-sync.sh
```

This syncs metrics every 30 seconds during training.

## Comparison

| Method | When | Plots | Use Case |
|--------|------|-------|----------|
| **Auto-sync** | After training | ✅ Yes | Default (recommended) |
| Real-time sync | During training | ❌ No | Monitor progress |
| Manual sync | On demand | ✅ Yes | Re-upload or sync old runs |

## View Results

After auto-sync, results are at:
```
https://app.neptune.ai/mehdix/mlir-project
```

Filter by run name (e.g., "run_12") or tags.

---

**Auto-sync is enabled by default if Neptune credentials are in `.env`**
