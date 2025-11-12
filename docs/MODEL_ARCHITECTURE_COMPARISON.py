"""
Model Architecture Comparison: LSTM vs DistilBERT

This file provides a side-by-side comparison of the architectures.
"""

# ============================================================================
# LSTM ARCHITECTURE (Original)
# ============================================================================

class LSTMEmbedding:
    """
    Input: OpFeatures (consumer + producer)
    
    Architecture:
        1. Feature Embedding (per operation):
           - Linear(OpFeatures.size() → 512)
           - ELU activation
           - Dropout(0.225)
           - Linear(512 → 512)
           - ELU activation
           - Dropout(0.225)
        
        2. Sequential Processing:
           - LSTM(input=512, hidden=411)
           - Process: [consumer_embedding, producer_embedding]
           - Extract final hidden state
        
        3. Concatenate:
           - final_hidden (411) + ActionHistory
           - Output size: 411 + ActionHistory.size()
    
    Parameters: ~2-3M
    Memory: 500MB-1GB
    Speed: 1x (baseline)
    """
    pass


# ============================================================================
# DISTILBERT ARCHITECTURE (New)
# ============================================================================

class DistilBertEmbedding:
    """
    Input: OpFeatures (consumer + producer)
    
    Architecture:
        1. Feature Projection (per operation):
           - Linear(OpFeatures.size() → 512)
           - LayerNorm(512)
           - GELU activation
           - Dropout(0.1)
           - Linear(512 → 768)
        
        2. Sequence Construction:
           - [CLS] consumer_embedding producer_embedding [SEP]
           - 4 tokens total
           - Add learnable special tokens
        
        3. Transformer Processing:
           - DistilBERT (6 layers, 12 heads, 768 dim)
           - Self-attention over all tokens
           - Each layer: MultiHeadAttention + FFN + LayerNorm
        
        4. Output Extraction:
           - Extract [CLS] token (first position)
           - Represents entire sequence
        
        5. Concatenate:
           - cls_output (768) + ActionHistory
           - Output size: 768 + ActionHistory.size()
    
    Parameters: ~66M (52M DistilBERT + 14M projection)
    Memory: 2-4GB
    Speed: 0.2-0.3x (3-5x slower than LSTM)
    """
    pass


# ============================================================================
# DETAILED LAYER-BY-LAYER COMPARISON
# ============================================================================

"""
STAGE 1: FEATURE EXTRACTION
----------------------------

LSTM:
  consumer_features → [Linear(512), ELU, Dropout, Linear(512), ELU, Dropout]
  producer_features → [Linear(512), ELU, Dropout, Linear(512), ELU, Dropout]
  
DistilBERT:
  consumer_features → [Linear(512), LayerNorm, GELU, Dropout, Linear(768)]
  producer_features → [Linear(512), LayerNorm, GELU, Dropout, Linear(768)]

Key Differences:
  - DistilBERT uses LayerNorm (transformer standard)
  - DistilBERT uses GELU (smoother than ELU/ReLU)
  - DistilBERT projects to 768 (transformer hidden size)
  - LSTM uses 512 (more compact)


STAGE 2: SEQUENCE MODELING
---------------------------

LSTM:
  Sequence: [consumer_emb, producer_emb]
  Processing: Sequential (must process in order)
  Hidden state: 411 dimensions
  Mechanism: Gating (forget, input, output gates)
  Complexity: O(n) for sequence length n
  
DistilBERT:
  Sequence: [CLS, consumer_emb, producer_emb, SEP]
  Processing: Parallel (all positions at once)
  Hidden state: 768 dimensions
  Mechanism: Self-attention (query, key, value)
  Complexity: O(n²) for sequence length n
  
Key Differences:
  - LSTM is sequential, DistilBERT is parallel
  - LSTM uses gating, DistilBERT uses attention
  - DistilBERT can attend to both ops simultaneously
  - LSTM processes ops in fixed order


STAGE 3: OUTPUT AGGREGATION
----------------------------

LSTM:
  Take final hidden state from LSTM
  Size: 411
  
DistilBERT:
  Take [CLS] token representation
  Size: 768
  
Both:
  Concatenate with ActionHistory
  Pass to policy/value heads


STAGE 4: POLICY/VALUE NETWORKS
-------------------------------

Both architectures use identical downstream networks:
  Backbone: [Linear(emb_size → 512), ReLU, Linear(512 → 512), ReLU, Linear(512 → 512), ReLU]
  Policy heads: Multiple heads for different action types
  Value network: [same backbone → Linear(512 → 1)]
"""


# ============================================================================
# MATHEMATICAL COMPARISON
# ============================================================================

"""
LSTM CELL COMPUTATION:
----------------------
At each timestep t:
  f_t = σ(W_f·[h_{t-1}, x_t] + b_f)     # Forget gate
  i_t = σ(W_i·[h_{t-1}, x_t] + b_i)     # Input gate
  c̃_t = tanh(W_c·[h_{t-1}, x_t] + b_c)  # Cell candidate
  c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t       # Cell state
  o_t = σ(W_o·[h_{t-1}, x_t] + b_o)     # Output gate
  h_t = o_t ⊙ tanh(c_t)                  # Hidden state

Where:
  - σ is sigmoid
  - ⊙ is element-wise multiplication
  - Sequential: must compute t before t+1


DISTILBERT ATTENTION:
---------------------
For each layer l and head h:
  Q = X·W_Q^{l,h}                        # Query
  K = X·W_K^{l,h}                        # Key
  V = X·W_V^{l,h}                        # Value
  
  Attention(Q,K,V) = softmax(QK^T/√d_k)V # Scaled dot-product attention
  
  MultiHead(X) = Concat(head_1,...,head_H)·W_O
  
Where:
  - All positions computed in parallel
  - Attention weights show which tokens are related
  - Each head can learn different relationships


PARAMETER COUNT:
----------------

LSTM:
  Embedding: 2 * (OpFeatures.size()*512 + 512*512) ≈ 1M
  LSTM: 4 * (512*411 + 411*411 + 411) ≈ 1.5M
  Total: ~2-3M parameters

DistilBERT:
  Projection: OpFeatures.size()*512 + 512*768 ≈ 0.5M
  DistilBERT:
    - Each attention layer: 768*768*3 (Q,K,V) + 768*768 (output) ≈ 2.4M
    - Each FFN: 768*3072 + 3072*768 ≈ 4.7M
    - 6 layers * (2.4M + 4.7M) ≈ 42.6M
    - Embeddings, LayerNorms: ~10M
  Total: ~53M in DistilBERT + ~13M in projections = ~66M parameters
"""


# ============================================================================
# WHEN TO USE WHICH MODEL
# ============================================================================

"""
USE LSTM WHEN:
--------------
✓ Fast iteration is important
✓ Limited computational resources
✓ Limited memory (GPU/CPU)
✓ Sequential nature of data is important
✓ Proven baseline performance is sufficient
✓ Quick experiments and debugging


USE DISTILBERT WHEN:
--------------------
✓ Better performance is more important than speed
✓ Sufficient computational resources available
✓ Can afford 3-5x longer training time
✓ Relationships between operations are complex
✓ Want to leverage transformer architecture benefits
✓ Can fine-tune pretrained models (future work)
✓ Need better long-range dependency modeling


HYBRID APPROACH:
----------------
1. Start with LSTM to establish baseline
2. Validate pipeline and hyperparameters
3. Switch to DistilBERT for final training
4. Compare results and choose best model
"""


# ============================================================================
# SAMPLE CONFIGURATIONS
# ============================================================================

lstm_config = {
    "model_type": "lstm",
    "lr": 0.001,
    "ppo_batch_size": 32,
    "value_batch_size": 32,
    "bench_count": 64,
    "ppo_epochs": 4,
    # Fast iterations, proven performance
}

distilbert_config = {
    "model_type": "distilbert",
    "lr": 0.0001,              # Lower learning rate
    "ppo_batch_size": 16,       # Smaller batches
    "value_batch_size": 16,     # Due to memory constraints
    "bench_count": 32,          # Fewer benchmarks per iteration
    "ppo_epochs": 4,            # Same number of epochs
    # Better representations, slower training
}


# ============================================================================
# IMPLEMENTATION NOTES
# ============================================================================

"""
FACTORY PATTERN:
----------------
Both models implement the same interface through get_embedding_layer():

def get_embedding_layer():
    model_type = Config().model_type
    if model_type == 'lstm':
        return LSTMEmbedding()
    elif model_type == 'distilbert':
        return DistilBertEmbedding()
    
This allows:
  - Easy model switching via config
  - Same training/evaluation code
  - Extensible to new architectures


INTERFACE CONTRACT:
-------------------
All embedding layers must provide:
  - __init__(): Initialize the model
  - forward(obs): Process observations
  - output_size: Property indicating embedding dimension
  
The rest of the pipeline (PolicyModel, ValueModel) works with any embedding
that follows this contract.


ADDING NEW MODELS:
------------------
To add a new architecture (e.g., BERT, ConvNext):
  1. Create new embedding class with same interface
  2. Add model type to Config.model_type
  3. Update get_embedding_layer() factory
  4. Create test config
  5. Document hyperparameters
"""
