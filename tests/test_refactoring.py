#!/usr/bin/env python3
"""
Test script to verify refactored model structure works correctly.
"""
import os
os.environ['CONFIG_FILE_PATH'] = 'config/config.json'

print("="*60)
print("Model Refactoring Test")
print("="*60)

print("\n1. Testing model imports...")
try:
    from rl_autoschedular.models import (
        get_embedding_layer,
        list_available_models,
        BaseEmbedding,
        LSTMEmbedding,
        DistilBertEmbedding
    )
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    exit(1)

print("\n2. Testing model registry...")
available = list_available_models()
print(f"✓ Available models: {available}")
assert 'lstm' in available, "LSTM should be available"
assert 'distilbert' in available, "DistilBERT should be available"

print("\n3. Testing LSTM embedding creation...")
try:
    lstm_embedding = get_embedding_layer('lstm')
    print(f"✓ LSTM embedding created")
    print(f"  Output size: {lstm_embedding.output_size}")
    assert isinstance(lstm_embedding, BaseEmbedding), "Should be BaseEmbedding instance"
    assert isinstance(lstm_embedding, LSTMEmbedding), "Should be LSTMEmbedding instance"
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n4. Testing DistilBERT embedding creation...")
try:
    # Change config to distilbert
    from utils.config import Config
    config = Config()
    original_model_type = config.model_type
    
    # Test factory with explicit model type
    distilbert_embedding = get_embedding_layer('distilbert')
    print(f"✓ DistilBERT embedding created")
    print(f"  Output size: {distilbert_embedding.output_size}")
    assert isinstance(distilbert_embedding, BaseEmbedding), "Should be BaseEmbedding instance"
    assert isinstance(distilbert_embedding, DistilBertEmbedding), "Should be DistilBertEmbedding instance"
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n5. Testing main model import...")
try:
    from rl_autoschedular.model import HiearchyModel, PolicyModel, ValueModel
    print("✓ Main model classes imported successfully")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    exit(1)

print("\n6. Testing model instantiation...")
try:
    import torch
    model = HiearchyModel()
    print("✓ HiearchyModel instantiated")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n7. Testing backward compatibility...")
try:
    # Test that we can still import from rl_autoschedular.model
    from rl_autoschedular.model import ACTIVATION
    print("✓ Backward compatibility maintained")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    exit(1)

print("\n" + "="*60)
print("✅ All refactoring tests passed!")
print("="*60)
print("\nRefactoring successful! The modular structure is working correctly.")
print("\nNew structure:")
print("  rl_autoschedular/models/")
print("    ├── base.py                    # Base classes")
print("    ├── __init__.py                # Factory & registry")
print("    └── embeddings/")
print("        ├── lstm_embedding.py      # LSTM model")
print("        └── distilbert_embedding.py # DistilBERT model")
print("\nTo add a new model:")
print("  1. Create new file in models/embeddings/")
print("  2. Inherit from BaseEmbedding")
print("  3. Register in models/__init__.py EMBEDDING_REGISTRY")
