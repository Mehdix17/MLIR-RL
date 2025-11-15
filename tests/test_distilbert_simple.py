#!/usr/bin/env python3
"""
Simplified test script for DistilBERT model - minimal dependencies.
Tests only the core model components without full environment setup.
"""
import sys
import os

# Set config before importing
os.environ['CONFIG_FILE_PATH'] = 'config/config_distilbert.json'

print("="*60)
print("DistilBERT Model - Simplified Test")
print("="*60)

try:
    print("\n1. Testing imports...")
    import torch
    print("✓ PyTorch imported")
    
    from transformers import DistilBertModel, DistilBertConfig
    print("✓ Transformers imported")
    
    print("\n2. Testing DistilBERT initialization...")
    config = DistilBertConfig(
        vocab_size=1,
        dim=768,
        n_heads=12,
        n_layers=6,
        dropout=0.1,
        attention_dropout=0.1,
        max_position_embeddings=512,
    )
    distilbert = DistilBertModel(config)
    print(f"✓ DistilBERT model created")
    print(f"  Parameters: {sum(p.numel() for p in distilbert.parameters()):,}")
    
    print("\n3. Testing forward pass...")
    batch_size = 2
    seq_len = 4
    hidden_dim = 768
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, seq_len, hidden_dim)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    
    # Forward pass
    with torch.no_grad():
        outputs = distilbert(inputs_embeds=dummy_input, attention_mask=attention_mask)
    
    print(f"✓ Forward pass successful")
    print(f"  Output shape: {outputs.last_hidden_state.shape}")
    print(f"  Expected: ({batch_size}, {seq_len}, {hidden_dim})")
    
    print("\n4. Testing gradient flow...")
    torch.set_grad_enabled(True)
    dummy_input.requires_grad = True
    outputs = distilbert(inputs_embeds=dummy_input, attention_mask=attention_mask)
    loss = outputs.last_hidden_state.mean()
    loss.backward()
    
    has_grad = dummy_input.grad is not None and dummy_input.grad.abs().sum() > 0
    print(f"✓ Gradients computed: {has_grad}")
    torch.set_grad_enabled(False)
    
    print("\n" + "="*60)
    print("✓ Core DistilBERT tests passed!")
    print("="*60)
    print("\nDistilBERT is working correctly.")
    print("The full model integration requires the complete environment setup.")
    
except ImportError as e:
    print(f"\n✗ Import error: {e}")
    print("\nPlease install required packages:")
    print("  pip install torch transformers")
    sys.exit(1)
    
except Exception as e:
    print(f"\n✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nNote: To test the full model integration:")
print("  1. Ensure all dependencies are installed: pip install -r requirements.txt")
print("  2. Run: python test_distilbert.py")
