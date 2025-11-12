import numpy as np
import torch
from typing import Dict, List, Tuple

class MLIROperationTokenizer:
    """
    Tokenizes MLIR operation features into sequences for DistilBERT.
    
    Converts fixed-length feature vectors into:
    - [CLS] token
    - Consumer operation tokens
    - Producer operation tokens  
    - [SEP] token
    """
    
    def __init__(self, vocab_size: int = 10000, max_seq_length: int = 128):
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        
        # Special tokens
        self.CLS_TOKEN_ID = 0
        self.SEP_TOKEN_ID = 1
        self.PAD_TOKEN_ID = 2
        self.UNK_TOKEN_ID = 3
        
        # Feature buckets for quantization
        self.num_buckets = 50  # Discretize continuous features
        
    def tokenize_operation(self, op_features: np.ndarray) -> List[int]:
        """
        Convert operation features into token IDs.
        
        Args:
            op_features: Array of shape (num_features,)
            
        Returns:
            List of token IDs
        """
        tokens = []
        
        # Discretize each feature dimension into buckets
        for feat_idx, feat_value in enumerate(op_features):
            # Handle invalid values
            if np.isnan(feat_value) or np.isinf(feat_value):
                tokens.append(self.UNK_TOKEN_ID)
                continue
            
            # Normalize to [0, 1] then bucket
            normalized_value = np.clip(feat_value, 0, 1)
            bucket = int(normalized_value * (self.num_buckets - 1))
            bucket = np.clip(bucket, 0, self.num_buckets - 1)
            
            # Create unique token ID: special_tokens + feature_idx * buckets + bucket
            # Modulo to prevent exceeding vocab_size
            token_id = 4 + (feat_idx % 100) * self.num_buckets + bucket
            token_id = token_id % self.vocab_size
            
            tokens.append(token_id)
            
        return tokens
    
    def create_sequence(
        self, 
        consumer_features: np.ndarray,
        producer_features: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create a tokenized sequence: [CLS] consumer producer [SEP]
        
        Args:
            consumer_features: Shape (num_consumer_features,)
            producer_features: Shape (num_producer_features,)
            
        Returns:
            input_ids: Tensor of shape (max_seq_length,)
            attention_mask: Tensor of shape (max_seq_length,)
        """
        # Create token sequence
        tokens = [self.CLS_TOKEN_ID]
        
        # Add consumer operation tokens
        consumer_tokens = self.tokenize_operation(consumer_features)
        tokens.extend(consumer_tokens)
        
        # Add producer operation tokens
        producer_tokens = self.tokenize_operation(producer_features)
        tokens.extend(producer_tokens)
        
        # Add separator
        tokens.append(self.SEP_TOKEN_ID)
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(tokens)
        
        # Pad or truncate to max_seq_length
        if len(tokens) > self.max_seq_length:
            tokens = tokens[:self.max_seq_length]
            attention_mask = attention_mask[:self.max_seq_length]
        else:
            padding_length = self.max_seq_length - len(tokens)
            tokens.extend([self.PAD_TOKEN_ID] * padding_length)
            attention_mask.extend([0] * padding_length)
        
        return (
            torch.tensor(tokens, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long)
        )
    
    def batch_create_sequences(
        self,
        consumer_features_batch: np.ndarray,
        producer_features_batch: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create batched sequences.
        
        Args:
            consumer_features_batch: Shape (batch_size, num_consumer_features)
            producer_features_batch: Shape (batch_size, num_producer_features)
            
        Returns:
            input_ids: Tensor of shape (batch_size, max_seq_length)
            attention_mask: Tensor of shape (batch_size, max_seq_length)
        """
        batch_size = consumer_features_batch.shape[0]
        all_input_ids = []
        all_attention_masks = []
        
        for i in range(batch_size):
            input_ids, attention_mask = self.create_sequence(
                consumer_features_batch[i],
                producer_features_batch[i]
            )
            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)
        
        return (
            torch.stack(all_input_ids),
            torch.stack(all_attention_masks)
        )