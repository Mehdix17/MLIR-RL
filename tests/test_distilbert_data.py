#!/usr/bin/env python3
"""
Test script for DistilBERT data preprocessing and tokenization.

This tests the MLIROperationTokenizer to ensure it correctly converts
operation features into token sequences for DistilBERT.
"""
import os
os.environ['CONFIG_FILE_PATH'] = 'config/config_distilbert.json'

import torch
import numpy as np
import sys

print("="*60)
print("DistilBERT Data Preprocessing Test")
print("="*60)

try:
    from rl_autoschedular.distilbert_tokenizer import MLIROperationTokenizer
    print("✓ Tokenizer imported successfully")
except ImportError as e:
    print(f"✗ Failed to import tokenizer: {e}")
    sys.exit(1)


def test_tokenizer_initialization():
    """Test tokenizer initialization with different parameters"""
    print("\n1. Testing tokenizer initialization...")
    
    tokenizer = MLIROperationTokenizer(vocab_size=10000, max_seq_length=128)
    
    assert tokenizer.vocab_size == 10000, "Vocab size mismatch"
    assert tokenizer.max_seq_length == 128, "Max sequence length mismatch"
    assert tokenizer.CLS_TOKEN_ID == 0, "CLS token ID should be 0"
    assert tokenizer.SEP_TOKEN_ID == 1, "SEP token ID should be 1"
    assert tokenizer.PAD_TOKEN_ID == 2, "PAD token ID should be 2"
    
    print("  ✓ Tokenizer initialized with correct parameters")
    print(f"    Vocab size: {tokenizer.vocab_size}")
    print(f"    Max sequence length: {tokenizer.max_seq_length}")
    print(f"    Special tokens: CLS={tokenizer.CLS_TOKEN_ID}, SEP={tokenizer.SEP_TOKEN_ID}, PAD={tokenizer.PAD_TOKEN_ID}")
    
    return tokenizer


def test_single_operation_tokenization(tokenizer):
    """Test tokenizing a single operation's features"""
    print("\n2. Testing single operation tokenization...")
    
    # Create dummy features (similar to actual operation features)
    # Typical OpFeatures size is around 50-100 features
    consumer_features = np.random.rand(50)
    
    tokens = tokenizer.tokenize_operation(consumer_features)
    
    assert len(tokens) == 50, f"Expected 50 tokens, got {len(tokens)}"
    assert all(isinstance(t, (int, np.integer)) for t in tokens), "All tokens should be integers"
    assert all(0 <= t < tokenizer.vocab_size for t in tokens), "All tokens should be within vocab size"
    
    print(f"  ✓ Operation tokenized: {len(tokens)} tokens")
    print(f"    First 10 tokens: {tokens[:10]}")
    print(f"    Token range: [{min(tokens)}, {max(tokens)}]")
    
    return tokens


def test_sequence_creation(tokenizer):
    """Test creating a full sequence with consumer and producer"""
    print("\n3. Testing sequence creation...")
    
    consumer_features = np.random.rand(50)
    producer_features = np.random.rand(50)
    
    input_ids, attention_mask = tokenizer.create_sequence(
        consumer_features, producer_features
    )
    
    assert input_ids.shape == (tokenizer.max_seq_length,), f"Input IDs shape mismatch: {input_ids.shape}"
    assert attention_mask.shape == (tokenizer.max_seq_length,), f"Attention mask shape mismatch: {attention_mask.shape}"
    
    # Check special tokens
    assert input_ids[0] == tokenizer.CLS_TOKEN_ID, "First token should be CLS"
    
    # Find where SEP token is
    sep_positions = (input_ids == tokenizer.SEP_TOKEN_ID).nonzero(as_tuple=True)[0]
    assert len(sep_positions) > 0, "Should have at least one SEP token"
    
    # Check attention mask
    non_padded_length = attention_mask.sum().item()
    assert non_padded_length > 2, "Should have more than just CLS and SEP tokens"
    
    print(f"  ✓ Sequence created successfully")
    print(f"    Input IDs shape: {input_ids.shape}")
    print(f"    Attention mask shape: {attention_mask.shape}")
    print(f"    Non-padded length: {non_padded_length}")
    print(f"    First 10 tokens: {input_ids[:10].tolist()}")
    print(f"    SEP token position(s): {sep_positions.tolist()}")
    
    return input_ids, attention_mask


def test_batch_creation(tokenizer):
    """Test batched sequence creation"""
    print("\n4. Testing batch sequence creation...")
    
    batch_size = 8
    consumer_batch = np.random.rand(batch_size, 50)
    producer_batch = np.random.rand(batch_size, 50)
    
    input_ids_batch, attention_mask_batch = tokenizer.batch_create_sequences(
        consumer_batch, producer_batch
    )
    
    assert input_ids_batch.shape == (batch_size, tokenizer.max_seq_length), \
        f"Input IDs batch shape mismatch: {input_ids_batch.shape}"
    assert attention_mask_batch.shape == (batch_size, tokenizer.max_seq_length), \
        f"Attention mask batch shape mismatch: {attention_mask_batch.shape}"
    
    # Check that all sequences have CLS token at the start
    assert (input_ids_batch[:, 0] == tokenizer.CLS_TOKEN_ID).all(), \
        "All sequences should start with CLS token"
    
    print(f"  ✓ Batch sequences created successfully")
    print(f"    Batch size: {batch_size}")
    print(f"    Input IDs shape: {input_ids_batch.shape}")
    print(f"    Attention mask shape: {attention_mask_batch.shape}")
    print(f"    All sequences start with CLS: {(input_ids_batch[:, 0] == tokenizer.CLS_TOKEN_ID).all()}")
    
    return input_ids_batch, attention_mask_batch


def test_feature_diversity(tokenizer):
    """Test that different features produce different tokens"""
    print("\n5. Testing feature diversity...")
    
    producer_features = np.random.rand(50)
    
    # Create features with different values
    consumer_zeros = np.zeros(50)
    consumer_ones = np.ones(50)
    consumer_half = np.full(50, 0.5)
    
    ids_zeros, _ = tokenizer.create_sequence(consumer_zeros, producer_features)
    ids_ones, _ = tokenizer.create_sequence(consumer_ones, producer_features)
    ids_half, _ = tokenizer.create_sequence(consumer_half, producer_features)
    
    # Count differences
    diff_zeros_ones = (ids_zeros != ids_ones).sum().item()
    diff_zeros_half = (ids_zeros != ids_half).sum().item()
    diff_ones_half = (ids_ones != ids_half).sum().item()
    
    assert diff_zeros_ones > 0, "Different features should produce different tokens"
    assert diff_zeros_half > 0, "Different features should produce different tokens"
    
    print(f"  ✓ Different features produce different tokens")
    print(f"    Differences (zeros vs ones): {diff_zeros_ones} tokens")
    print(f"    Differences (zeros vs half): {diff_zeros_half} tokens")
    print(f"    Differences (ones vs half): {diff_ones_half} tokens")


def test_edge_cases(tokenizer):
    """Test edge cases like NaN, inf, and negative values"""
    print("\n6. Testing edge cases...")
    
    # Test with NaN values
    features_with_nan = np.random.rand(50)
    features_with_nan[10] = np.nan
    features_with_nan[20] = np.inf
    features_with_nan[30] = -np.inf
    
    tokens = tokenizer.tokenize_operation(features_with_nan)
    
    # Check that NaN/inf values are handled (should become UNK token)
    assert tokens[10] == tokenizer.UNK_TOKEN_ID, "NaN should become UNK token"
    assert tokens[20] == tokenizer.UNK_TOKEN_ID, "Inf should become UNK token"
    assert tokens[30] == tokenizer.UNK_TOKEN_ID, "-Inf should become UNK token"
    
    print(f"  ✓ Edge cases handled correctly")
    print(f"    NaN → token {tokens[10]} (UNK)")
    print(f"    Inf → token {tokens[20]} (UNK)")
    print(f"    -Inf → token {tokens[30]} (UNK)")
    
    # Test with out-of-range values
    features_out_of_range = np.array([2.0, -1.0, 0.5, 10.0, -5.0])
    tokens_clipped = tokenizer.tokenize_operation(features_out_of_range)
    
    assert all(0 <= t < tokenizer.vocab_size for t in tokens_clipped), \
        "All tokens should be within vocab range even for out-of-range inputs"
    
    print(f"  ✓ Out-of-range values clipped correctly")
    print(f"    Input: {features_out_of_range}")
    print(f"    Tokens: {tokens_clipped}")


def test_device_compatibility():
    """Test that tokenizer outputs work with PyTorch"""
    print("\n7. Testing PyTorch device compatibility...")
    
    tokenizer = MLIROperationTokenizer()
    consumer_features = np.random.rand(50)
    producer_features = np.random.rand(50)
    
    input_ids, attention_mask = tokenizer.create_sequence(
        consumer_features, producer_features
    )
    
    # Test moving to different devices (if available)
    assert input_ids.dtype == torch.long, "Input IDs should be long tensors"
    assert attention_mask.dtype == torch.long, "Attention mask should be long tensors"
    
    # Test that tensors can be moved to GPU if available
    if torch.cuda.is_available():
        input_ids_gpu = input_ids.cuda()
        attention_mask_gpu = attention_mask.cuda()
        print(f"  ✓ Tensors moved to GPU successfully")
        print(f"    Input IDs device: {input_ids_gpu.device}")
        print(f"    Attention mask device: {attention_mask_gpu.device}")
    else:
        print(f"  ✓ Tensors are PyTorch compatible")
        print(f"    Input IDs dtype: {input_ids.dtype}")
        print(f"    Attention mask dtype: {attention_mask.dtype}")
        print(f"    (GPU not available for testing)")


def main():
    """Run all tests"""
    try:
        tokenizer = test_tokenizer_initialization()
        test_single_operation_tokenization(tokenizer)
        test_sequence_creation(tokenizer)
        test_batch_creation(tokenizer)
        test_feature_diversity(tokenizer)
        test_edge_cases(tokenizer)
        test_device_compatibility()
        
        print("\n" + "="*60)
        print("✅ All data preprocessing tests passed!")
        print("="*60)
        print("\nThe tokenizer is ready to use with DistilBERT.")
        print("Next steps:")
        print("  1. Update DistilBertEmbedding to use the tokenizer")
        print("  2. Test full model integration: python test_distilbert.py")
        print("  3. Train: CONFIG_FILE_PATH=config/config_distilbert.json python bin/train.py")
        
        return 0
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
