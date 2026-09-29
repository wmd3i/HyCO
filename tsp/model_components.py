# Building blocks of Prefix-DIFUSCO: GNN encoder, prefix encoder and embeddings (dense graphs).
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import functools # For GNNEncoder if using activation_checkpoint

# --- Existing Components ---
class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t): # t is a tensor of timesteps
        device = t.device
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t.float().unsqueeze(1) * embeddings.unsqueeze(0)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        if self.dim % 2 == 1: # For odd dimensions
            embeddings = torch.cat([embeddings, torch.zeros_like(embeddings[:, :1])], dim=-1)
        return embeddings
        
class PrefixEncoder(nn.Module):
    def __init__(self, node_feat_dim, hidden_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_size=node_feat_dim, hidden_size=hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, prefix_node_features, prefix_lengths):
        """
        prefix_node_features: (B, max_k, D_feat) - Padded tensor of features.
        prefix_lengths: (B) - Tensor of original lengths for each sequence.
        """
        # LSTM can be slow if it processes lots of padding. Packing avoids this.
        # Pack the padded sequence. ensure_sorted=False is important.
        packed_features = torch.nn.utils.rnn.pack_padded_sequence(
            prefix_node_features, 
            prefix_lengths.cpu(), # pack_padded_sequence expects lengths on CPU
            batch_first=True, 
            enforce_sorted=False
        )
        
        self.lstm.flatten_parameters()
        # The output of the LSTM is a PackedSequence; hn is the final hidden state of each sequence.
        _, (hn, _) = self.lstm(packed_features)
        
        return self.linear(hn.squeeze(0))
        
# class PrefixEncoder(nn.Module):
#     def __init__(self, node_feat_dim, hidden_dim, output_dim):
#         super().__init__()
#         self.lstm = nn.LSTM(input_size=node_feat_dim, hidden_size=hidden_dim, batch_first=True)
#         self.linear = nn.Linear(hidden_dim, output_dim)

#     def forward(self, prefix_node_features):
#         self.lstm.flatten_parameters()
#         _, (hn, _) = self.lstm(prefix_node_features)
#         return self.linear(hn.squeeze(0))

# --- DIFUSCO GNN Components (Adapted) ---
# Utility functions from reference_difusco/models/nn.py
def linear(*args, **kwargs):
    return nn.Linear(*args, **kwargs)

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

class GroupNorm32(nn.GroupNorm): # Used in normalization
    def forward(self, x):
        # Ensure input is float for GroupNorm, then cast back to original dtype
        original_dtype = x.dtype
        return super().forward(x.float()).type(original_dtype)


def normalization(channels): # Used in GNNEncoder and GNNLayer
    return GroupNorm32(32, channels) # 32 is a common number of groups


def timestep_embedding(timesteps, dim, max_period=10000): # For GNNEncoder internal time embedding
    """
    Matches the sinusoidal embedding used in reference_difusco.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

# PositionEmbeddingSine from reference_difusco/models/gnn_encoder.py
class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x_coords_b_n_2):
        x_embed = x_coords_b_n_2[:, :, 0] 
        y_embed = x_coords_b_n_2[:, :, 1] 

        if self.normalize:
            x_embed = x_embed * self.scale
            y_embed = y_embed * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x_coords_b_n_2.device)
        # Corrected dim_t calculation for temperature
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)


        pos_x = x_embed[:, :, None] / dim_t 
        pos_y = y_embed[:, :, None] / dim_t 
        
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2) 
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2) 
        
        pos = torch.cat((pos_y, pos_x), dim=2) 
        return pos


# DifuscoGNNLayer (remains the same as previous correct version)
class DifuscoGNNLayer(nn.Module):
    def __init__(self, hidden_dim, aggregation="sum", norm="layer", learn_norm=True, track_norm=False, gated=True):
        super(DifuscoGNNLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.aggregation = aggregation
        self.norm_type = norm 
        self.learn_norm = learn_norm
        self.track_norm = track_norm
        self.gated = gated
        if not self.gated: 
            print("Warning: Gating is recommended for DifuscoGNNLayer.")

        self.U = nn.Linear(hidden_dim, hidden_dim, bias=True) 
        self.V = nn.Linear(hidden_dim, hidden_dim, bias=True) 
        
        self.A = nn.Linear(hidden_dim, hidden_dim, bias=True) 
        self.B = nn.Linear(hidden_dim, hidden_dim, bias=True) 
        self.C = nn.Linear(hidden_dim, hidden_dim, bias=True) 

        norm_module_factory = {
            "layer": lambda: nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
            "batch": lambda: nn.BatchNorm1d(hidden_dim, affine=learn_norm, track_running_stats=track_norm)
        }
        self.norm_h = norm_module_factory.get(self.norm_type, lambda: None)()
        self.norm_e = norm_module_factory.get(self.norm_type, lambda: None)()


    def forward(self, h_nodes, h_edges, adj_matrix_mask=None):
        B, N, H = h_nodes.shape
        h_in_nodes = h_nodes
        h_in_edges = h_edges

        Uh = self.U(h_nodes) 
        
        Vh_expanded_j = self.V(h_nodes).unsqueeze(1).expand(B, N, N, H) 

        Ah_i = self.A(h_nodes).unsqueeze(2).expand(B, N, N, H) 
        Bh_j = self.B(h_nodes).unsqueeze(1).expand(B, N, N, H) 
        Ce = self.C(h_edges) 

        h_edges_updated = Ah_i + Bh_j + Ce 
        edge_gates = torch.sigmoid(h_edges_updated) if self.gated else torch.ones_like(h_edges_updated)
        
        gated_messages = edge_gates * Vh_expanded_j 
        
        if adj_matrix_mask is not None:
            gated_messages = gated_messages * adj_matrix_mask.unsqueeze(-1)

        if self.aggregation == "sum":
            aggregated_messages = torch.sum(gated_messages, dim=2) 
        elif self.aggregation == "mean":
            num_neighbors = torch.sum(adj_matrix_mask, dim=2, keepdim=True).clamp(min=1) if adj_matrix_mask is not None else N
            aggregated_messages = torch.sum(gated_messages, dim=2) / num_neighbors 
        elif self.aggregation == "max":
            aggregated_messages = torch.max(gated_messages, dim=2)[0] 
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        h_nodes_updated = Uh + aggregated_messages 

        if self.norm_h:
            if self.norm_type == "batch":
                 h_nodes_updated = self.norm_h(h_nodes_updated.view(B * N, H)).view(B, N, H)
            elif self.norm_type == "layer": # LayerNorm expects (..., Features)
                 h_nodes_updated = self.norm_h(h_nodes_updated)

        h_nodes_updated = F.relu(h_nodes_updated)

        if self.norm_e:
            if self.norm_type == "batch":
                h_edges_updated = self.norm_e(h_edges_updated.reshape(B * N * N, H)).view(B, N, N, H)
            elif self.norm_type == "layer": # LayerNorm expects (..., Features)
                h_edges_updated = self.norm_e(h_edges_updated)


        h_edges_activated = F.relu(h_edges_updated) # Apply ReLU after potential norm
        
        h_nodes = h_in_nodes + h_nodes_updated
        h_edges = h_in_edges + h_edges_activated # Use activated edges for residual

        return h_nodes, h_edges


class DifuscoGNNEncoder(nn.Module):
    def __init__(self, n_layers, node_feature_dim, edge_feature_dim, hidden_dim, out_channels=1,
                 aggregation="sum", norm="layer", learn_norm=True, track_norm=False, gated=True,
                 time_embed_dim_ratio=0.25, 
                 prefix_cond_dim=0):
        super(DifuscoGNNEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.node_feature_dim = node_feature_dim 
        self.edge_feature_dim = edge_feature_dim 

        actual_time_embed_dim = int(hidden_dim * time_embed_dim_ratio)

        self.node_embed = nn.Linear(node_feature_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_feature_dim, hidden_dim)
        
        self.time_embed_mlp = nn.Sequential( 
            linear(hidden_dim, actual_time_embed_dim), 
            nn.ReLU(),
            linear(actual_time_embed_dim, actual_time_embed_dim),
        )
        
        self.time_proj_for_edges = nn.Linear(actual_time_embed_dim, hidden_dim)
        
        self.prefix_cond_dim = prefix_cond_dim
        if self.prefix_cond_dim > 0:
            self.prefix_proj_for_edges = nn.Linear(prefix_cond_dim, hidden_dim)

        self.layers = nn.ModuleList([
            DifuscoGNNLayer(hidden_dim, aggregation, norm, learn_norm, track_norm, gated)
            for _ in range(n_layers)
        ])

        # Output head using Conv2d, similar to DIFUSCO reference
        self.output_head_seq = nn.Sequential(
            normalization(hidden_dim), # Expects (B, H, N, N)
            nn.ReLU(),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=True) # Outputs (B, out_channels, N, N)
        )

    def forward(self, initial_node_features, initial_edge_features,
                timesteps_scalar, adj_matrix_mask=None, prefix_cond_vector=None):
        B, N, _ = initial_node_features.shape

        h_nodes = self.node_embed(initial_node_features) 
        h_edges = self.edge_embed(initial_edge_features) 

        time_emb_sinusoidal = SinusoidalTimestepEmbedding(self.hidden_dim)(timesteps_scalar) 
        time_emb_processed = self.time_embed_mlp(time_emb_sinusoidal) 

        time_cond_for_edges = self.time_proj_for_edges(time_emb_processed) 
        time_cond_for_edges = time_cond_for_edges.unsqueeze(1).unsqueeze(1).expand(B, N, N, self.hidden_dim)
        h_edges = h_edges + time_cond_for_edges 

        if self.prefix_cond_dim > 0 and prefix_cond_vector is not None:
            prefix_cond_for_edges = self.prefix_proj_for_edges(prefix_cond_vector) 
            prefix_cond_for_edges = prefix_cond_for_edges.unsqueeze(1).unsqueeze(1).expand(B, N, N, self.hidden_dim)
            h_edges = h_edges + prefix_cond_for_edges
            
        for layer in self.layers:
            h_nodes, h_edges = layer(h_nodes, h_edges, adj_matrix_mask)

        # Prepare for output head: h_edges is (B, N, N, H)
        # Permute to (B, H, N, N) for Conv2d-based output head
        h_edges_permuted = h_edges.permute(0, 3, 1, 2).contiguous()
        
        edge_logits_conv = self.output_head_seq(h_edges_permuted) # Output: (B, out_channels, N, N)
        
        # Assuming out_channels = 1 for binary adjacency prediction (single logit per edge)
        edge_logits = edge_logits_conv.squeeze(1) # Output: (B, N, N)
        
        return edge_logits