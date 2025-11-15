"""
DistilBERT-based embedding for MLIR operation features.

Uses transformer architecture with self-attention to process operation
features as token sequences.
"""
import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig
from ..base import BaseEmbedding
from ...observation import OpFeatures, ActionHistory, ProducerOpFeatures, Observation
from ...distilbert_tokenizer import MLIROperationTokenizer
from utils.config import Config


class DistilBertEmbedding(BaseEmbedding):
    """
    DistilBERT-based embedding for operation features.
    
    Transforms fixed-length operation feature vectors into sequences
    and uses DistilBERT's self-attention mechanism to capture relationships
    between consumer and producer operations.
    
    Architecture:
        1. Tokenize features into discrete tokens
        2. Build sequence: [CLS] consumer_tokens producer_tokens [SEP]
        3. Process through DistilBERT (6 layers, 12 heads)
        4. Extract [CLS] token representation
        5. Concatenate with action history
    """
    
    def __init__(self):
        super(DistilBertEmbedding, self).__init__()

        # Load configuration from Config
        cfg = Config()
        model_cfg = cfg.model_config
        
        # DistilBERT configuration with defaults
        hidden_dim = model_cfg.get('hidden_size', 768)
        num_heads = model_cfg.get('num_attention_heads', 12)
        num_layers = model_cfg.get('num_hidden_layers', 6)
        max_seq_length = model_cfg.get('max_seq_length', 128)
        vocab_size = model_cfg.get('vocab_size', 10000)
        dropout = model_cfg.get('dropout', 0.1)
        
        # Initialize tokenizer
        self.tokenizer = MLIROperationTokenizer(
            vocab_size=vocab_size,
            max_seq_length=max_seq_length
        )
        
        distilbert_config = DistilBertConfig(
            vocab_size=vocab_size,
            dim=hidden_dim,
            n_heads=num_heads,
            n_layers=num_layers,
            dropout=dropout,
            attention_dropout=dropout,
            max_position_embeddings=max_seq_length,
        )
        
        # Initialize DistilBERT model
        self.distilbert = DistilBertModel(distilbert_config)
        
        # Output size includes DistilBERT embedding + action history
        self._output_size = hidden_dim + ActionHistory.size()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Process operation features through DistilBERT.
        
        The tokenizer converts continuous feature vectors into discrete token sequences:
        Sequence structure: [CLS] consumer_tokens producer_tokens [SEP]
        
        Args:
            obs: Observation tensor of shape (batch_size, obs_size)
            
        Returns:
            Embedded features of shape (batch_size, output_size)
        """
        batch_size = obs.size(0)
        
        # Extract operation features
        consumer_feats = Observation.get_part(obs, OpFeatures)
        producer_feats = Observation.get_part(obs, ProducerOpFeatures)
        
        # Convert to numpy for tokenization
        consumer_np = consumer_feats.cpu().detach().numpy()
        producer_np = producer_feats.cpu().detach().numpy()
        
        # Tokenize: [CLS] consumer_tokens producer_tokens [SEP]
        input_ids, attention_mask = self.tokenizer.batch_create_sequences(
            consumer_np, producer_np
        )
        
        # Move to same device as observation
        device = obs.device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # Pass through DistilBERT
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation (first token)
        cls_output = outputs.last_hidden_state[:, 0, :]  # (B, hidden_dim)
        
        # Concatenate with action history (same as LSTM approach)
        return torch.cat((cls_output, Observation.get_part(obs, ActionHistory)), dim=1)
