"""
aggregation.py — Token aggregation strategy and feature extraction.
"""

from __future__ import annotations
import torch


TARGET_LAYERS = [12, 16, 22, 23]

def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert per-token hidden states into a single feature vector."""
    
    real_positions = attention_mask.nonzero(as_tuple=False).squeeze(-1)

    features = []
    
    for layer_idx in TARGET_LAYERS:
        layer_states = hidden_states[layer_idx]
        
        last_token_emb = layer_states[real_positions[-1]]
        features.append(last_token_emb)

        last_k_states = layer_states[real_positions[-int(len(real_positions) * 0.25):]]
            
        mean_last_k_emb = last_k_states.mean(dim=0)
        features.append(mean_last_k_emb)

    feature_vector = torch.cat(features, dim=0)

    return feature_vector


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    return torch.zeros(0, device=hidden_states.device)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    
    agg_features = aggregate(hidden_states, attention_mask)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        if geo_features.numel() > 0:
            return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
