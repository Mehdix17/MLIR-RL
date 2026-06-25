# Demo: V4.9 Large vs V0

## Setup

```bash
source ~/envs/mlir/bin/activate
set -a && source .env && set +a
```

## Run

```bash
# Step 1: Run evaluation (fake output with real results)
python3 demo/eval_demo.py --config demo/eval_v49.json --checkpoint 3200

# Step 2: Compare results
python3 demo/compare_demo.py
```

## Results

```
Benchmark                                          Base(ms)   V0(ms)     V0 sp    V4.9(ms)   V4.9 sp  Ratio
  conv_2d_nchw_fchw_128_240_7_7_256_1_1_4_4        153.00     58.62      2.61x    21.16      7.23x    2.8x
  matmul_256_768_128                               61.50      9.70       6.34x    7.05       8.72x    1.4x
  pooling_nchw_max_256_512_120_120_1_60_60         841.70     516.38     1.63x    286.15     2.94x    1.8x
  V0 avg: 3.53x    V4.9 avg: 6.30x
```
