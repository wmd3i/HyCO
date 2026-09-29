# Prefix-DIFUSCO: prefix-conditioned denoising network for TSP on sparse (k-NN) graphs.

import torch
import torch.nn as nn
import torch.nn.functional as F

# <<< MODIFIED: Import the new sparse-aware GNN components
from model_components_sparse import (
    PrefixEncoder,
    PositionEmbeddingSine,
    DifuscoGNNEncoder
)

class ConditionalTSPSuffixDiffusionModel(nn.Module):
    def __init__(self, num_nodes, node_coord_dim,
                 pos_embed_num_feats, node_embed_dim,
                 gnn_n_layers, gnn_hidden_dim, 
                 gnn_aggregation, gnn_norm, gnn_learn_norm, gnn_gated,
                 prefix_node_embed_dim,
                 prefix_enc_hidden_dim, prefix_cond_dim,
                 time_embed_dim, sparse_factor): # <<< MODIFIED: Added sparse_factor
        super().__init__()
        self.num_nodes = num_nodes
        self.is_sparse = sparse_factor > 0 # <<< MODIFIED: Internal flag

        self.pos_embed = PositionEmbeddingSine(num_pos_feats=pos_embed_num_feats, normalize=True)
        actual_node_feature_dim = 2 * pos_embed_num_feats
        
        self.node_feature_proj = nn.Linear(actual_node_feature_dim + 1, actual_node_feature_dim)
        
        self.prefix_encoder = PrefixEncoder(
            node_feat_dim=actual_node_feature_dim,
            hidden_dim=prefix_enc_hidden_dim,
            output_dim=prefix_cond_dim
        )

        initial_gnn_edge_feature_dim = 2 # Noisy edge value + geometric distance
        
        # <<< MODIFIED: Instantiate the new sparse-aware GNN encoder
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
            time_embed_dim_ratio=0.25, # As per original Difusco
            prefix_cond_dim=prefix_cond_dim,
            is_sparse=self.is_sparse
        )

    def forward(self, noisy_data, t_scalar, batch_data):
        
        # Unpack the required data from the batch_data dict
        instance_locs = batch_data["instance_locs"]
        prefix_nodes_indices = batch_data["prefix_nodes"]
        prefix_lengths = batch_data["prefix_lengths"]
        node_prefix_state = batch_data["node_prefix_state"]
        edge_index = batch_data.get("edge_index")
        node_to_graph_batch = batch_data.get("node_to_graph_batch")
        
        # 1. Prepare the initial node features
        pos_features = self.pos_embed(instance_locs)
        combined_node_features = torch.cat([pos_features, node_prefix_state], dim=-1)
        initial_node_features = self.node_feature_proj(combined_node_features)

        # 2. Encode the prefix
        prefix_cond_vector = self._get_prefix_conditioning(
            initial_node_features, prefix_nodes_indices, prefix_lengths, node_to_graph_batch
        )

        # 3. Prepare the GNN inputs
        if self.is_sparse:
            # --- Sparse mode ---
            # <<< MODIFIED: directly use the normalized dist_feature provided by the data loader >>>
            dist_feature = batch_data["dist_feature"]
            
            # <<< REMOVED: no need to compute and normalize distances here >>>
            # src_nodes, dst_nodes = edge_index[0], edge_index[1]
            # dist_feature = torch.linalg.norm(instance_locs[src_nodes] - instance_locs[dst_nodes], dim=-1, keepdim=True)
            # dist_feature = F.normalize(dist_feature, p=2, dim=0) # <-- removed problematic normalization
            
            # <<< MODIFIED: noisy_data is already [E, 1], no unsqueeze needed >>>
            noisy_edge_feature = noisy_data
            
                        # <<< MODIFIED: add a dimension check for compatibility >>>
            # At inference, noisy_data is 1D [E] while dist_feature is 2D [E, 1].
            # During training, noisy_data is 2D [E, 1].
            # This check makes sure noisy_edge_feature is always 2D before concatenation.
            if noisy_edge_feature.dim() == 1:
                noisy_edge_feature = noisy_edge_feature.unsqueeze(-1)
            # <<< END MODIFICATION >>>
            
            initial_edge_features = torch.cat([noisy_edge_feature, dist_feature], dim=-1)
            adj_mask_for_gnn = None
            
        else:
            # --- Dense mode ---
            B, N, _ = instance_locs.shape
            noisy_adj_feature = noisy_data.unsqueeze(-1)
            dist_matrix = torch.cdist(instance_locs, instance_locs, p=2)

            # <<< MODIFIED: unify normalization in dense mode using per-graph min-max scaling >>>
            # Also apply min-max scaling (can be simplified to dividing by a constant such as sqrt(2))
            # For simplicity we use a simple scaling here; per-graph min-max is also possible
            max_dist = torch.sqrt(torch.tensor(2.0, device=dist_matrix.device)) # Assume coordinates are in [0,1]
            dist_feature = (dist_matrix / max_dist).unsqueeze(-1)
            
            initial_edge_features = torch.cat([noisy_adj_feature, dist_feature], dim=-1)
            adj_mask_for_gnn = torch.ones(B, N, N, device=instance_locs.device) - torch.eye(N, device=instance_locs.device).unsqueeze(0)

        # 4. Pass through GNN Encoder
        edge_logits = self.gnn_encoder(
            initial_node_features=initial_node_features,
            initial_edge_features=initial_edge_features,
            timesteps_scalar=t_scalar,
            adj_matrix_mask=adj_mask_for_gnn,
            prefix_cond_vector=prefix_cond_vector,
            edge_index=edge_index,
            node_to_graph_batch=node_to_graph_batch
        )
        
        return edge_logits

# Replace the _get_prefix_conditioning method in diffusion_model_sparse.py with this one.

# In diffusion_model_sparse.py

    def _get_prefix_conditioning(self, initial_node_features, prefix_nodes_indices, prefix_lengths, node_to_graph_batch):
        B = prefix_lengths.size(0)
        device = initial_node_features.device
        dtype = initial_node_features.dtype
        prefix_cond_dim = self.prefix_encoder.linear.out_features
        prefix_cond_vector = torch.zeros(B, prefix_cond_dim, device=device, dtype=dtype)

        has_prefix_mask = prefix_lengths > 0
        if not has_prefix_mask.any():
            return prefix_cond_vector

        # Keep only the samples that actually have a prefix
        prefix_nodes_to_process = prefix_nodes_indices[has_prefix_mask]
        prefix_lengths_to_process = prefix_lengths[has_prefix_mask]
        
        # --- START OF FIX ---
        
        if self.is_sparse:
            # --- Sparse mode (flattened node features) ---
            # Here initial_node_features has shape [Total_Nodes_in_Batch, D]
            
            # 1. Compute the node offset of each graph
            node_counts = torch.bincount(node_to_graph_batch)
            node_cumsum = torch.cat([torch.tensor([0], device=device), torch.cumsum(node_counts, dim=0)[:-1]])
            
            # 2. Only the offsets of graphs that have a prefix are needed
            offsets_for_prefix_graphs = node_cumsum[has_prefix_mask]
            
            # 3. Convert local prefix node indices into global flattened indices
            #    - Get the longest prefix length in the batch to avoid unnecessary computation
            max_len = prefix_lengths_to_process.max().item()
            #    - Truncate the prefix indices to the actual max length
            prefix_nodes_local = prefix_nodes_to_process[:, :max_len].long()
            #    - Add the correct offsets
            prefix_nodes_global = prefix_nodes_local + offsets_for_prefix_graphs.unsqueeze(1)

            # 4. Safely gather the prefix node features from the flattened feature tensor via advanced indexing
            prefix_node_features_gathered = initial_node_features[prefix_nodes_global]

        else: 
            # --- Dense mode (non-flattened node features) ---
            # Here initial_node_features has shape [B, N, D]
            
            # 1. Keep the features of graphs that have a prefix
            initial_features_to_process = initial_node_features[has_prefix_mask]
            
            # 2. Get the longest prefix length in the batch
            max_len = prefix_lengths_to_process.max().item()
            
            # 3. Truncate the prefix indices to the actual max length
            prefix_nodes_to_gather = prefix_nodes_to_process[:, :max_len].long()

            # 4. Gather features with standard 2D advanced indexing
            batch_indices = torch.arange(prefix_nodes_to_gather.shape[0], device=device).unsqueeze(1)
            prefix_node_features_gathered = initial_features_to_process[batch_indices, prefix_nodes_to_gather]
        
        # --- END OF FIX ---

        # Build a mask for the LSTM/Transformer that ignores the padded part
        max_len_for_mask = prefix_lengths_to_process.max()
        pad_mask = torch.arange(max_len_for_mask, device=device)[None, :] < prefix_lengths_to_process[:, None]
        prefix_node_features_gathered[~pad_mask] = 0.0

        # Compute the condition vector
        computed_prefix_cond = self.prefix_encoder(prefix_node_features_gathered, prefix_lengths_to_process)
        prefix_cond_vector[has_prefix_mask] = computed_prefix_cond
        
        return prefix_cond_vector
    
    def back_get_prefix_conditioning(self, initial_node_features, prefix_nodes_indices, prefix_lengths, node_to_graph_batch):
        B = prefix_lengths.size(0)
        device = initial_node_features.device
        dtype = initial_node_features.dtype
        prefix_cond_dim = self.prefix_encoder.linear.out_features
        prefix_cond_vector = torch.zeros(B, prefix_cond_dim, device=device, dtype=dtype)

        has_prefix_mask = prefix_lengths > 0
        if not has_prefix_mask.any():
            return prefix_cond_vector

        # Filter samples that have a prefix
        prefix_nodes_to_process = prefix_nodes_indices[has_prefix_mask]
        prefix_lengths_to_process = prefix_lengths[has_prefix_mask]
        
        # In sparse mode, node features are flat. We need to gather based on graph index.
        if self.is_sparse:
            # Build a mask for nodes belonging to graphs with prefixes
            graph_indices_with_prefix = torch.where(has_prefix_mask)[0]
            node_mask_for_prefix_graphs = torch.isin(node_to_graph_batch, graph_indices_with_prefix)
            
            # Map original node indices to their position in the flattened tensor
            node_counts = torch.bincount(node_to_graph_batch)
            node_cumsum = torch.cat([torch.tensor([0], device=device), torch.cumsum(node_counts, dim=0)[:-1]])
            
            # Adjust prefix node indices to the flattened layout
            adjusted_prefix_nodes = prefix_nodes_to_process + node_cumsum[has_prefix_mask].unsqueeze(1)
            
            prefix_node_features_gathered = initial_node_features[adjusted_prefix_nodes.long()]
        
        else: # Dense mode
            initial_features_to_process = initial_node_features[has_prefix_mask]
            batch_indices = torch.arange(prefix_nodes_to_process.shape[0], device=device).unsqueeze(1)
            prefix_node_features_gathered = initial_features_to_process[batch_indices, prefix_nodes_to_process.long()]
            
        # Mask out padding before passing to LSTM
        max_len = prefix_lengths_to_process.max()
        pad_mask = torch.arange(max_len, device=device)[None, :] < prefix_lengths_to_process[:, None]
        prefix_node_features_gathered[~pad_mask] = 0.0

        computed_prefix_cond = self.prefix_encoder(prefix_node_features_gathered, prefix_lengths_to_process)
        prefix_cond_vector[has_prefix_mask] = computed_prefix_cond
        
        return prefix_cond_vector