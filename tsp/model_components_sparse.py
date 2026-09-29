# Building blocks of Prefix-DIFUSCO for sparse (k-NN) graphs, used for TSP-1000.
# <<< NEW: This file contains the sparse-aware GNN components.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# You will need torch_sparse for this to work.
from torch_sparse import SparseTensor
from torch_sparse import sum as sparse_sum, mean as sparse_mean, max as sparse_max

# --- Helper Modules (from original DIFUSCO) ---

class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        # x is (B, N, 2) for dense, or (Total_N, 2) for sparse
        if x.dim() == 3:
            y_embed, x_embed = x[:, :, 0], x[:, :, 1]
        else: # dim is 2
            y_embed, x_embed = x[:, 0], x[:, 1]
        
        if self.normalize:
            y_embed = y_embed * self.scale
            x_embed = x_embed * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2.0 * (torch.div(dim_t, 2, rounding_mode='trunc')) / self.num_pos_feats)

        pos_x = x_embed.unsqueeze(-1) / dim_t
        pos_y = y_embed.unsqueeze(-1) / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
        pos = torch.cat((pos_y, pos_x), dim=-1)
        return pos

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class PrefixEncoder(nn.Module):
    def __init__(self, node_feat_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=node_feat_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, prefix_features, lengths):
        packed_input = nn.utils.rnn.pack_padded_sequence(
            prefix_features, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed_input)
        # Use the hidden state of the last layer
        last_layer_hidden = hidden[-1]
        return self.linear(last_layer_hidden)

# --- Sparse GNN Layer ---

class GNNLayer(nn.Module):
    def __init__(self, hidden_dim, aggregation="sum", norm="layer", learn_norm=True, gated=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.aggregation = aggregation
        self.norm = norm
        self.learn_norm = learn_norm
        self.gated = gated

        self.U = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.V = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.A = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.B = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.C = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.norm_h = nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm) if norm == "layer" else None
        self.norm_e = nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm) if norm == "layer" else None

    def forward(self, h, e, graph_mask_or_adj, mode="residual", edge_index=None, is_sparse=False):
        h_in, e_in = h, e

        # Node and edge updates
        if not is_sparse:
            # Dense mode
            num_nodes = h.shape[1]
            Uh = self.U(h)
            Vh = self.V(h).unsqueeze(1).expand(-1, num_nodes, -1, -1)
            Ah, Bh = self.A(h), self.B(h)
            e = Ah.unsqueeze(1) + Bh.unsqueeze(2) + self.C(e)
        else:
            # Sparse mode
            Uh = self.U(h)
            # Vh operates on source nodes (j) for messages to target nodes (i)
            Vh = self.V(h[edge_index[1]])
            # A operates on target nodes (i), B on source nodes (j)
            e = self.A(h[edge_index[0]]) + self.B(h[edge_index[1]]) + self.C(e)

        gates = torch.sigmoid(e)
        
        # Aggregate messages
        if not is_sparse:
            h = Uh + torch.sum(gates * Vh, dim=2)
        else:
            # sparse aggregation
            aggr_msg = gates * Vh
            adj = SparseTensor(
                row=edge_index[0], col=edge_index[1],
                value=aggr_msg,
                sparse_sizes=(h.size(0), h.size(0))
            )
            if self.aggregation == "sum":
                h = Uh + adj.sum(dim=1)
            elif self.aggregation == "mean":
                h = Uh + adj.mean(dim=1)
            else: # max
                h = Uh + adj.max(dim=1)

        # Normalization and activation
        if self.norm_h: h = self.norm_h(h)
        if self.norm_e: e = self.norm_e(e)
        h, e = F.relu(h), F.relu(e)

        # Residual connection
        if mode == "residual":
            h, e = h_in + h, e_in + e

        return h, e

# --- Sparse GNN Encoder ---

class DifuscoGNNEncoder(nn.Module):
    def __init__(self, n_layers, node_feature_dim, edge_feature_dim, hidden_dim, out_channels,
                 aggregation, norm, learn_norm, gated, time_embed_dim_ratio, prefix_cond_dim, is_sparse):
        super().__init__()
        self.n_layers = n_layers
        self.is_sparse = is_sparse
        actual_time_embed_dim = int(hidden_dim * time_embed_dim_ratio)

        # Input projections
        self.node_embed = nn.Linear(node_feature_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_feature_dim, hidden_dim)
        
        # Time and prefix conditioning embeddings
        self.time_embed_in = nn.Sequential(
            nn.Linear(hidden_dim, actual_time_embed_dim), nn.ReLU(),
            nn.Linear(actual_time_embed_dim, actual_time_embed_dim)
        )
        if prefix_cond_dim > 0:
            self.prefix_embed_in = nn.Linear(prefix_cond_dim, actual_time_embed_dim)

        self.layers = nn.ModuleList([
            GNNLayer(hidden_dim, aggregation, norm, learn_norm, gated) for _ in range(n_layers)
        ])
        
        # Per-layer conditioning projections
        self.time_embed_layers = nn.ModuleList([
            nn.Sequential(nn.ReLU(), nn.Linear(actual_time_embed_dim, hidden_dim)) for _ in range(n_layers)
        ])
        if prefix_cond_dim > 0:
            self.prefix_embed_layers = nn.ModuleList([
                nn.Sequential(nn.ReLU(), nn.Linear(actual_time_embed_dim, hidden_dim)) for _ in range(n_layers)
            ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_channels)
        )

    def forward(self, initial_node_features, initial_edge_features, timesteps_scalar,
                adj_matrix_mask, prefix_cond_vector=None, edge_index=None, node_to_graph_batch=None):
        
        # 1. Initial Embeddings
        h = self.node_embed(initial_node_features)
        e = self.edge_embed(initial_edge_features)
        time_emb = self.time_embed_in(timestep_embedding(timesteps_scalar, self.node_embed.out_features))
        
        if prefix_cond_vector is not None:
            prefix_emb = self.prefix_embed_in(prefix_cond_vector)

        # 2. GNN Layers
        for i in range(self.n_layers):
            h_in, e_in = h, e
            
            h, e = self.layers[i](h, e, adj_matrix_mask, mode="direct", edge_index=edge_index, is_sparse=self.is_sparse)
            
            # Conditioning injection
            time_cond = self.time_embed_layers[i](time_emb)
            if prefix_cond_vector is not None:
                prefix_cond = self.prefix_embed_layers[i](prefix_emb)
                # Select correct conditioning vector for each node/edge in the batch
                if self.is_sparse:
                    time_cond = time_cond[node_to_graph_batch]
                    prefix_cond = prefix_cond[node_to_graph_batch]
                else: # Dense
                    time_cond = time_cond.unsqueeze(1)
                    prefix_cond = prefix_cond.unsqueeze(1)

            h = h + time_cond
            if prefix_cond_vector is not None:
                h = h + prefix_cond

            h, e = h_in + h, e_in + e

        # 3. Output Projection
        edge_logits = self.output_proj(e)
        return edge_logits.squeeze(-1)