# Dataset of TSP instances with optimal tours and randomly sampled prefixes (dense).

import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.utils.rnn as rnn_utils

def construct_target_adj_from_tour(tour_nodes, num_total_nodes, device):
    """
    Constructs a binary adjacency matrix from a single complete tour.
    """
    batch_size = 1 # This helper works on a single tour
    adj_matrix = torch.zeros((batch_size, num_total_nodes, num_total_nodes), dtype=torch.float32, device=device)
    
    # Add edges for the tour
    for j in range(len(tour_nodes) - 1):
        u, v = tour_nodes[j], tour_nodes[j+1]
        adj_matrix[0, u, v] = 1.0
        adj_matrix[0, v, u] = 1.0 # For undirected TSP representation
    
    # Close the loop
    if len(tour_nodes) > 1:
        u, v = tour_nodes[-1], tour_nodes[0]
        adj_matrix[0, u, v] = 1.0
        adj_matrix[0, v, u] = 1.0
            
    return adj_matrix.squeeze(0) # Return as (N,N)

class TSPConditionalSuffixDataset(Dataset):
    def __init__(self, npz_file_path, prefix_k_options, prefix_sampling_strategy='continuous_random_start'):
        """
        npz_file_path: Path to the NPZ file with solved TSP instances.
        prefix_k_options: A list or tuple of possible prefix lengths, e.g., [10, 20, 30, 40, 50].
        prefix_sampling_strategy: 'continuous_from_start', 'continuous_random_start', or 'scattered'.
        """
        data = np.load(npz_file_path)
        self.instances_locs = torch.tensor(data['locs'], dtype=torch.float32)
        self.num_samples, self.num_nodes, _ = self.instances_locs.shape
        
        if not isinstance(prefix_k_options, (list, tuple)) or not prefix_k_options:
            raise ValueError("prefix_k_options must be a non-empty list or tuple.")
        self.prefix_k_options = prefix_k_options
        
        if prefix_sampling_strategy not in ['continuous_from_start', 'continuous_random_start', 'scattered']:
            raise ValueError(f"Unknown prefix_sampling_strategy: {prefix_sampling_strategy}")
        self.prefix_sampling_strategy = prefix_sampling_strategy

        self.ground_truth_tours_indices = torch.arange(self.num_nodes, dtype=torch.long).unsqueeze(0).repeat(self.num_samples, 1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        instance_locs = self.instances_locs[idx]
        gt_tour_node_indices = self.ground_truth_tours_indices[idx]

        prefix_k = np.random.choice(self.prefix_k_options)
        
        if self.prefix_sampling_strategy == 'continuous_from_start':
            prefix_node_indices = gt_tour_node_indices[:prefix_k]
        elif self.prefix_sampling_strategy == 'continuous_random_start':
            start_node_idx = np.random.randint(0, self.num_nodes)
            rolled_tour = torch.roll(gt_tour_node_indices, shifts=-start_node_idx, dims=0)
            prefix_node_indices = rolled_tour[:prefix_k]
        elif self.prefix_sampling_strategy == 'scattered':
            sampled_indices = torch.randperm(self.num_nodes)[:prefix_k]
            prefix_node_indices = gt_tour_node_indices[sampled_indices]
        
        target_adj_matrix = construct_target_adj_from_tour(
            gt_tour_node_indices,
            self.num_nodes,
            device='cpu'
        )

        # ==================== NEW: Create Node State Feature ====================
        # Create a feature tensor indicating whether a node is in the prefix.
        # Shape: (N, 1)
        node_prefix_state = torch.zeros((self.num_nodes, 1), dtype=torch.float32)
        if prefix_k > 0:
            node_prefix_state[prefix_node_indices] = 1.0
        # ======================================================================

        return {
            "instance_locs": instance_locs,
            "prefix_nodes": prefix_node_indices,
            "target_adj_matrix": target_adj_matrix,
            "node_prefix_state": node_prefix_state  # <<< Add to returned sample
        }

def custom_collate_fn(batch):
    """
    Custom collate function to handle batches of samples with variable-length prefix_nodes.
    """
    instance_locs = torch.stack([item['instance_locs'] for item in batch], dim=0)
    target_adj_matrix = torch.stack([item['target_adj_matrix'] for item in batch], dim=0)
    
    # ==================== NEW: Collate Node State Feature ====================
    node_prefix_states = torch.stack([item['node_prefix_state'] for item in batch], dim=0)
    # ======================================================================

    prefix_nodes_list = [item['prefix_nodes'] for item in batch]
    prefix_lengths = torch.tensor([len(p) for p in prefix_nodes_list], dtype=torch.long)
    padded_prefixes = rnn_utils.pad_sequence(prefix_nodes_list, batch_first=True, padding_value=0)

    return {
        "instance_locs": instance_locs,
        "prefix_nodes": padded_prefixes,
        "prefix_lengths": prefix_lengths,
        "target_adj_matrix": target_adj_matrix,
        "node_prefix_state": node_prefix_states  # <<< Add to returned batch
    }