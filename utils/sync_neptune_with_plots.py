#!/usr/bin/env python3
"""
Enhanced Neptune sync with automatic plotting
Syncs metrics and generates/uploads comparison plots
"""

from dotenv import load_dotenv
load_dotenv(override=True)

import neptune
import os
import sys
import json
import subprocess
from pathlib import Path

def sync_run_with_plots(run_dir, project, api_token):
    """Sync a single run with Neptune including plots"""
    run_path = Path(run_dir)
    run_name = run_path.name
    
    print(f"\n{'='*70}")
    print(f"SYNCING: {run_name}")
    print(f"{'='*70}")
    
    # Load tags
    with open(run_path / 'tags') as f:
        tags = f.read().splitlines()
    
    # Load config
    with open('config/config.json') as f:
        config = json.load(f)
    
    # Initialize Neptune run
    print(f"Creating Neptune run...")
    neptune_run = neptune.init_run(
        project=project,
        api_token=api_token,
        tags=tags,
        name=f"{run_name}",
    )
    
    print(f"✓ Neptune run created: {neptune_run._sys_id}")
    print(f"  URL: {neptune_run.get_url()}")
    
    # Upload config
    neptune_run['config'] = config
    print(f"✓ Uploaded configuration")
    
    # Sync all log files
    logs_dir = run_path / 'logs'
    file_count = 0
    
    for root, _, files in os.walk(logs_dir):
        relative_root = Path(root).relative_to(logs_dir)
        for filename in files:
            file_path = Path(root) / filename
            neptune_path = str(relative_root / filename) if str(relative_root) != '.' else filename
            
            try:
                with open(file_path) as f:
                    values = [float(line.strip()) for line in f if line.strip()]
                    if values:
                        for value in values:
                            neptune_run[neptune_path].append(value)
                        file_count += 1
            except:
                pass
    
    print(f"✓ Uploaded {file_count} metric files")
    
    # Generate and upload plots
    plots_dir = run_path / 'plots'
    if plots_dir.exists():
        print(f"✓ Found existing plots directory")
        plot_count = 0
        for plot_file in plots_dir.glob('*.png'):
            neptune_run[f"plots/{plot_file.name}"].upload(str(plot_file))
            plot_count += 1
        print(f"✓ Uploaded {plot_count} plots")
    else:
        print(f"  Note: No plots directory found. Run 'python utils/plot_results.py {run_dir}' to generate plots.")
    
    # Upload execution data
    exec_data_file = run_path / 'exec_data.json'
    if exec_data_file.exists():
        with open(exec_data_file) as f:
            exec_data = json.load(f)
        neptune_run['exec_data_size'] = len(str(exec_data))
        neptune_run['num_benchmarks_executed'] = len(exec_data)
        print(f"✓ Logged execution data statistics")
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"✓ SYNC COMPLETE")
    print(f"{'='*70}")
    print(f"View at: {neptune_run.get_url()}")
    
    neptune_run.stop()
    return neptune_run._sys_id


def main():
    if len(sys.argv) < 2:
        print("Usage: python sync_neptune_with_plots.py <run_directory>")
        print("Example: python sync_neptune_with_plots.py results/run_9")
        sys.exit(1)
    
    run_dir = sys.argv[1]
    
    if not os.path.exists(run_dir):
        print(f"Error: Directory {run_dir} does not exist")
        sys.exit(1)
    
    # Get Neptune credentials
    project = os.getenv('NEPTUNE_PROJECT')
    api_token = os.getenv('NEPTUNE_TOKEN')
    
    if not project or not api_token:
        print("Error: NEPTUNE_PROJECT and NEPTUNE_TOKEN must be set in .env file")
        sys.exit(1)
    
    print(f"Neptune Project: {project}")
    
    # Generate plots first (if matplotlib is available)
    try:
        import matplotlib
        plots_dir = Path(run_dir) / 'plots'
        if not plots_dir.exists():
            print(f"\nGenerating plots...")
            result = subprocess.run(['python', 'utils/plot_results.py', run_dir], 
                                    capture_output=True, text=True)
            if result.returncode == 0:
                print(result.stdout)
            else:
                print(f"Warning: Could not generate plots: {result.stderr}")
    except ImportError:
        print("\nNote: matplotlib not installed. Plots will not be generated.")
        print("Install with: pip install matplotlib")
    
    # Sync to Neptune
    run_id = sync_run_with_plots(run_dir, project, api_token)
    
    print(f"\n✓ Successfully synced {run_dir} to Neptune (Run ID: {run_id})")


if __name__ == '__main__':
    main()
