# Prefix-DIFUSCO: prefix-conditioned denoising network for TSP (dense graphs).

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_components import (
    PrefixEncoder,
    PositionEmbeddingSine,
    DifuscoGNNEncoder
)

class ConditionalTSPSuffixDiffusionModel(nn.Module):
    def __init__(self, num_nodes, node_coord_dim,
                 node_embed_dim, pos_embed_num_feats,
                 gnn_n_layers, gnn_hidden_dim, 
                 gnn_aggregation, gnn_norm, gnn_learn_norm, gnn_gated,
                 prefix_node_embed_dim,
                 prefix_enc_hidden_dim, prefix_cond_dim,
                 time_embed_dim):
        super().__init__()
        self.num_nodes = num_nodes

        self.pos_embed = PositionEmbeddingSine(num_pos_feats=pos_embed_num_feats, normalize=True)
        actual_node_feature_dim = 2 * pos_embed_num_feats
        
        # ==================== NEW: Node Feature Projection Layer ====================
        # This layer will take the concatenated (pos_embed + state_feature) and project it
        # back to the expected dimension for the GNN and PrefixEncoder.
        # Input dim = pos_embed_dim (actual_node_feature_dim) + state_feature_dim (1)
        self.node_feature_proj = nn.Linear(actual_node_feature_dim + 1, actual_node_feature_dim)
        # ==========================================================================

        self.prefix_encoder = PrefixEncoder(
            node_feat_dim=actual_node_feature_dim,
            hidden_dim=prefix_enc_hidden_dim,
            output_dim=prefix_cond_dim
        )

        initial_gnn_edge_feature_dim = 2 # Noisy edge value + geometric distance
        self.gnn_encoder = DifuscoGNNEncoder(
            n_layers=gnn_n_layers,
            node_feature_dim=actual_node_feature_dim,
            edge_feature_dim=initial_gnn_edge_feature_dim,
            hidden_dim=gnn_hidden_dim,
            out_channels=1,
            aggregation=gnn_aggregation,
            norm=gnn_norm,
            learn_norm=gnn_learn_norm,
            gated=gnn_gated,
            time_embed_dim_ratio=0.25,
            prefix_cond_dim=prefix_cond_dim
        )

    def forward_bak(self, noisy_adj_matrix_xt, t_scalar, instance_locs, prefix_nodes_indices, prefix_lengths, node_prefix_state):
        """
        NEW: Accepts node_prefix_state
        """
        B, N, _coord_dim = instance_locs.shape
        
        # 1. Get positional embedding
        pos_features = self.pos_embed(instance_locs) # Shape: [B, N, D_pos]

        # ==================== NEW: Inject State Feature into Node Features ====================
        # Concatenate positional features with the new state feature
        combined_node_features = torch.cat([pos_features, node_prefix_state], dim=-1)
        # Project the combined features back to the standard node embedding dimension
        initial_node_features = self.node_feature_proj(combined_node_features) # Shape: [B, N, D_pos]
        # ====================================================================================

        # 2. Encode Prefix (uses the new, state-aware node features)
        prefix_cond_vector = None
        if torch.any(prefix_lengths > 0):
            batch_indices = torch.arange(B, device=instance_locs.device).unsqueeze(1).expand_as(prefix_nodes_indices)
            prefix_node_features_gathered = initial_node_features[batch_indices, prefix_nodes_indices.long()]
            prefix_cond_vector = self.prefix_encoder(prefix_node_features_gathered, prefix_lengths)

        # 3. Prepare Initial Edge Features
        noisy_adj_feature = noisy_adj_matrix_xt.unsqueeze(-1)
        dist_matrix = torch.cdist(instance_locs, instance_locs, p=2)
        dist_feature = F.normalize(dist_matrix, p=2, dim=-1).unsqueeze(-1)
        initial_edge_features = torch.cat([noisy_adj_feature, dist_feature], dim=-1)

        # 4. Pass through GNN Encoder
        adj_mask_for_gnn = torch.ones(B, N, N, device=instance_locs.device) - torch.eye(N, device=instance_locs.device).unsqueeze(0)
        
        edge_logits = self.gnn_encoder(
            initial_node_features=initial_node_features, # Use the new projected features
            initial_edge_features=initial_edge_features,
            timesteps_scalar=t_scalar,
            adj_matrix_mask=adj_mask_for_gnn, 
            prefix_cond_vector=prefix_cond_vector
        )
        
        return edge_logits



# In file: RLDF4CO_v6/diffusion_model.py

    def forward(self, noisy_adj_matrix_xt, t_scalar, instance_locs, prefix_nodes_indices, prefix_lengths, node_prefix_state):
        """
        NEW: Accepts node_prefix_state
        """
        B, N, _coord_dim = instance_locs.shape
        device = instance_locs.device # Get device info
    
        # 1. Get positional embedding
        pos_features = self.pos_embed(instance_locs) 
    
        combined_node_features = torch.cat([pos_features, node_prefix_state], dim=-1)
        initial_node_features = self.node_feature_proj(combined_node_features)
        
    
        # === [Key change]: more robust handling of prefix encoding ===
    
        # 1. Initialize an all-zero condition vector as the default (used for samples with k=0)
        #    Make sure its dimension matches the output dimension of prefix_encoder
        prefix_cond_dim = self.prefix_encoder.linear.out_features
        # === [Fix]: make sure dtypes match ===
        # Create prefix_cond_vector with the same dtype as the model's internal feature tensor.
        # Under autocast both will be float16; otherwise both will be float32.
        prefix_cond_vector = torch.zeros(B, prefix_cond_dim, device=device, dtype=initial_node_features.dtype)
        # ===============================================
    
        # 2. Find the samples in the batch whose prefix length > 0
        has_prefix_mask = prefix_lengths > 0
    
        # 3. If the batch contains prefixes that need to be processed
        if has_prefix_mask.any():
            # Only operate on samples that have a prefix
            prefix_nodes_to_process = prefix_nodes_indices[has_prefix_mask]
            prefix_lengths_to_process = prefix_lengths[has_prefix_mask]
            initial_features_to_process = initial_node_features[has_prefix_mask]
    
            # Extract the features of the prefix nodes for these samples from the raw node features
            batch_indices = torch.arange(prefix_nodes_to_process.shape[0], device=device).unsqueeze(1).expand_as(prefix_nodes_to_process)
            prefix_node_features_gathered = initial_features_to_process[batch_indices, prefix_nodes_to_process.long()]
    
            # 4. Call PrefixEncoder; all prefix_lengths it receives are now guaranteed to be > 0
            computed_prefix_cond = self.prefix_encoder(prefix_node_features_gathered, prefix_lengths_to_process)
    
            # 5. Scatter the computed condition vectors back into the full-batch condition tensor
            prefix_cond_vector[has_prefix_mask] = computed_prefix_cond
    
        # ===============================================
    
        # 3. Prepare Initial Edge Features
        noisy_adj_feature = noisy_adj_matrix_xt.unsqueeze(-1)
        dist_matrix = torch.cdist(instance_locs, instance_locs, p=2)
        dist_feature = F.normalize(dist_matrix, p=2, dim=-1).unsqueeze(-1)
        initial_edge_features = torch.cat([noisy_adj_feature, dist_feature], dim=-1)
    
        # 4. Pass through GNN Encoder
        adj_mask_for_gnn = torch.ones(B, N, N, device=instance_locs.device) - torch.eye(N, device=instance_locs.device).unsqueeze(0)
        
        edge_logits = self.gnn_encoder(
            initial_node_features=initial_node_features,
            initial_edge_features=initial_edge_features,
            timesteps_scalar=t_scalar,
            adj_matrix_mask=adj_mask_for_gnn, 
            prefix_cond_vector=prefix_cond_vector # Pass in the processed condition vector
        )
        
        return edge_logits