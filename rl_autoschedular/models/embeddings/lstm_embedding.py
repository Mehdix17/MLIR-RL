"""
LSTM-based embedding for MLIR operation features.

This is the original embedding architecture that uses LSTM to process
consumer and producer operation features sequentially.
"""
import torch
import torch.nn as nn
from ..base import BaseEmbedding
from ...observation import OpFeatures, ActionHistory, ProducerOpFeatures, Observation


class LSTMEmbedding(BaseEmbedding):
    """
    LSTM-based embedding for operation features.
    
    Architecture:
        1. Project consumer/producer features through feedforward network
        2. Process sequence [consumer, producer] through LSTM
        3. Concatenate LSTM output with action history
    """
    
    def __init__(self):
        super(LSTMEmbedding, self).__init__()

        embedding_size = 411
        self._output_size = embedding_size + ActionHistory.size()

        self.embedding = nn.Sequential(
            nn.Linear(OpFeatures.size(), 512),
            nn.ELU(),
            nn.Dropout(0.225),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Dropout(0.225),
        )

        self.lstm = nn.LSTM(512, embedding_size)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Process observation through LSTM.
        
        Args:
            obs: Observation tensor of shape (batch_size, obs_size)
            
        Returns:
            Embedded features of shape (batch_size, output_size)
        """
        consumer_feats = Observation.get_part(obs, OpFeatures)
        producer_feats = Observation.get_part(obs, ProducerOpFeatures)

        consumer_embeddings = self.embedding(consumer_feats).unsqueeze(0)
        producer_embeddings = self.embedding(producer_feats).unsqueeze(0)

        _, (final_hidden, _) = self.lstm(torch.cat((consumer_embeddings, producer_embeddings)))

        return torch.cat((final_hidden.squeeze(0), Observation.get_part(obs, ActionHistory)), 1)
