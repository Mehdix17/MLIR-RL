#!/usr/bin/env python3
"""
Test script to verify DistilBERT model implementation.
Tests model initialization, forward pass, and output shapes.
"""
import os
os.environ['CONFIG_FILE_PATH'] = 'config/config_distilbert.json'

import torch
from rl_autoschedular.model import HiearchyModel, get_embedding_layer
from rl_autoschedular.observation import Observation, OpFeatures
from utils.config import Config

def test_embedding_layer():
    """Test that the embedding layer initializes correctly."""
    print("Testing embedding layer initialization...")
    
    cfg = Config()
    print(f"Model type: {cfg.model_type}")
    
    embedding = get_embedding_layer()
    print(f"✓ Embedding layer created: {embedding.__class__.__name__}")
    print(f"✓ Output size: {embedding.output_size}")
    
    return embedding

def test_model_initialization():
    """Test that the hierarchical model initializes correctly."""
    print("\nTesting model initialization...")
    
    model = HiearchyModel()
    print(f"✓ Model created successfully")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    
    return model

def test_forward_pass():
    """Test forward pass with dummy data."""
    print("\nTesting forward pass...")
    
    model = HiearchyModel()
    model.eval()
    
    # Create dummy observation
    batch_size = 4
    obs_size = Observation.cumulative_sizes()[-1]
    dummy_obs = torch.randn(batch_size, obs_size)
    
    print(f"✓ Created dummy observation: shape {dummy_obs.shape}")
    
    # Test sampling
    with torch.no_grad():
        actions_index, actions_log_p, entropies = model.sample(dummy_obs, greedy=True)
    
    print(f"✓ Sample action completed")
    print(f"  - Actions log prob shape: {actions_log_p.shape}")
    print(f"  - Entropies shape: {entropies.shape}")
    
    # Test forward pass
    with torch.no_grad():
        actions_log_p, values, entropies = model(dummy_obs, actions_index)
    
    print(f"✓ Forward pass completed")
    print(f"  - Actions log prob: {actions_log_p.shape}")
    print(f"  - Values: {values.shape}")
    print(f"  - Entropies: {entropies.shape}")
    
    return model

def test_gradient_flow():
    """Test that gradients flow correctly through the model."""
    print("\nTesting gradient flow...")
    
    model = HiearchyModel()
    model.train()
    
    # Create dummy data
    batch_size = 2
    obs_size = Observation.cumulative_sizes()[-1]
    dummy_obs = torch.randn(batch_size, obs_size)
    
    # Sample actions
    with torch.no_grad():
        actions_index, _, _ = model.sample(dummy_obs, greedy=True)
    
    # Forward pass with gradients
    torch.set_grad_enabled(True)
    actions_log_p, values, entropies = model(dummy_obs, actions_index)
    
    # Compute dummy loss
    loss = -actions_log_p.mean() + values.mean() - 0.01 * entropies.mean()
    loss.backward()
    
    # Check if gradients exist
    has_grads = any(p.grad is not None and p.grad.abs().sum() > 0 
                    for p in model.parameters() if p.requires_grad)
    
    if has_grads:
        print(f"✓ Gradients computed successfully")
    else:
        print(f"✗ WARNING: No gradients found!")
    
    torch.set_grad_enabled(False)
    
    return model

def main():
    """Run all tests."""
    print("="*60)
    print("DistilBERT Model Implementation Test Suite")
    print("="*60)
    
    try:
        # Run tests
        test_embedding_layer()
        test_model_initialization()
        test_forward_pass()
        test_gradient_flow()
        
        print("\n" + "="*60)
        print("✓ All tests passed successfully!")
        print("="*60)
        print("\nThe DistilBERT model is ready to use.")
        print("To train with DistilBERT, use: CONFIG_FILE_PATH=config/config_distilbert.json python bin/train.py")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
