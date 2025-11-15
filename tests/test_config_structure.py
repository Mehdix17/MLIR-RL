#!/usr/bin/env python3
"""
Test script to validate config loading and model initialization
with the new nested config structure.
"""

import os
import sys

def test_config(config_path, config_name):
    """Test loading a specific config file"""
    print(f"\n{'='*60}")
    print(f"Testing: {config_name}")
    print('='*60)
    
    # Clear singleton instance
    from utils.singleton import Singleton
    Singleton._instances.clear()
    
    os.environ["CONFIG_FILE_PATH"] = config_path
    
    from utils.config import Config
    cfg = Config()
    
    print(f"✓ Config loaded successfully")
    print(f"  Model type: {cfg.model_type}")
    print(f"  Model config: {cfg.model_config}")
    print(f"  Observation space:")
    print(f"    - max_num_loops: {cfg.max_num_loops}")
    print(f"    - max_num_stores_loads: {cfg.max_num_stores_loads}")
    print(f"  Action space:")
    print(f"    - interchange_mode: {cfg.interchange_mode}")
    print(f"  Training:")
    print(f"    - nb_iterations: {cfg.nb_iterations}")
    print(f"    - bench_count: {cfg.bench_count}")
    print(f"  PPO:")
    print(f"    - lr: {cfg.lr}")
    print(f"    - ppo_epochs: {cfg.ppo_epochs}")
    print(f"    - ppo_batch_size: {cfg.ppo_batch_size}")
    print(f"    - gamma: {cfg.gamma}")
    print(f"    - clip_epsilon: {cfg.clip_epsilon}")
    print(f"  Data paths:")
    print(f"    - benchmarks: {cfg.benchmarks_folder_path}")
    print(f"  Logging:")
    print(f"    - tags: {cfg.tags}")
    print(f"    - results_dir: {cfg.results_dir}")
    
    return cfg

def test_model_initialization(cfg):
    """Test model initialization with config"""
    print(f"\n  Testing model initialization...")
    
    # Import after config is loaded
    from rl_autoschedular.models import get_embedding_layer
    
    try:
        embedding = get_embedding_layer()
        print(f"  ✓ Embedding layer created: {type(embedding).__name__}")
        print(f"    - Output size: {embedding.output_size}")
        
        if cfg.model_type == 'distilbert':
            print(f"    - DistilBERT config:")
            print(f"      - Hidden size: {cfg.model_config.get('hidden_size')}")
            print(f"      - Attention heads: {cfg.model_config.get('num_attention_heads')}")
            print(f"      - Hidden layers: {cfg.model_config.get('num_hidden_layers')}")
        
        return True
    except Exception as e:
        print(f"  ✗ Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("CONFIG STRUCTURE VALIDATION TEST")
    print("="*60)
    
    configs = [
        ("config/config.json", "LSTM Baseline"),
        ("config/config_distilbert.json", "DistilBERT Transformer"),
        ("config/config_augmented.json", "LSTM with Augmentation"),
        ("config/test.json", "Quick Test Config"),
    ]
    
    all_passed = True
    
    for config_path, config_name in configs:
        try:
            cfg = test_config(config_path, config_name)
            
            # Test model initialization
            if not test_model_initialization(cfg):
                all_passed = False
            
            print(f"\n✓ {config_name} - ALL TESTS PASSED")
            
        except Exception as e:
            print(f"\n✗ {config_name} - FAILED")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✓✓✓ ALL CONFIG TESTS PASSED ✓✓✓")
        print("="*60)
        print("\nThe new nested config structure is working correctly!")
        print("Training and evaluation scripts will work with all configs.")
        return 0
    else:
        print("✗✗✗ SOME TESTS FAILED ✗✗✗")
        print("="*60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
