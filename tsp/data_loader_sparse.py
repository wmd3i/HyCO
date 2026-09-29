# Dataset of TSP instances with optimal tours and randomly sampled prefixes (sparse k-NN graphs).

import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.utils.rnn as rnn_utils
from sklearn.neighbors import KDTree # <<< MODIFIED: For k-NN graph construction

# <<< MODIFIED: This function is now only used in dense mode (for small problems)
def construct_target_adj_from_tour(tour_nodes, num_total_nodes, device):
    adj_matrix = torch.zeros((num_total_nodes, num_total_nodes), dtype=torch.float32, device=device)
    if len(tour_nodes) > 1:
        for j in range(len(tour_nodes) - 1):
            u, v = tour_nodes[j], tour_nodes[j+1]
            adj_matrix[u, v] = 1.0
            adj_matrix[v, u] = 1.0
        u, v = tour_nodes[-1], tour_nodes[0]
        adj_matrix[u, v] = 1.0
        adj_matrix[v, u] = 1.0
    return adj_matrix

class TSPConditionalSuffixDataset(Dataset):
    def __init__(self, npz_file_path, prefix_k_options, prefix_sampling_strategy='continuous_from_start', sparse_factor=-1):
        data = np.load(npz_file_path)
        self.instances_locs = torch.tensor(data['locs'], dtype=torch.float32)
        self.num_samples, self.num_nodes, _ = self.instances_locs.shape
        self.prefix_k_options = prefix_k_options
        self.prefix_sampling_strategy = prefix_sampling_strategy
        # <<< MODIFIED: Store sparse_factor to determine data format
        self.sparse_factor = sparse_factor
        
        # Assuming ground truth tours are provided or are identity for generated data
        # if 'locs' in data:
        #     self.ground_truth_tours_indices = torch.tensor(data['locs'], dtype=torch.long)
        # else:
        #     print("Warning: 'tours' not found in NPZ file. Using identity permutation as ground truth.")
        #     self.ground_truth_tours_indices = torch.arange(self.num_nodes, dtype=torch.long).unsqueeze(0).repeat(self.num_samples, 1)
        # +++ CORRECTED CODE +++
        # This correctly generates the tour indices as [0, 1, ..., N-1] for each sample.
        self.instances_locs = torch.tensor(data['locs'], dtype=torch.float32)
        self.num_samples, self.num_nodes, _ = self.instances_locs.shape
        self.prefix_k_options = prefix_k_options
        self.prefix_sampling_strategy = prefix_sampling_strategy
        self.sparse_factor = sparse_factor

        # <<< MODIFIED >>>: use torch.arange to build the correct tour indices
        base_tour_indices = torch.arange(self.num_nodes, dtype=torch.long)
        self.ground_truth_tours_indices = base_tour_indices.unsqueeze(0).repeat(self.num_samples, 1)

        # # Since locs are pre-sorted by tour, the ground truth tour is simply the sequence of indices.
        # base_tour_indices = torch.arange(self.num_nodes, dtype=torch.long)
        # self.ground_truth_tours_indices = base_tour_indices.unsqueeze(0).repeat(self.num_samples, 1)
        
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        instance_locs = self.instances_locs[idx]
        gt_tour_node_indices = self.ground_truth_tours_indices[idx]
        gt_tour_node_indices = gt_tour_node_indices.squeeze()
        prefix_k = np.random.choice(self.prefix_k_options)
        
        # --- Prefix Sampling (no changes here) ---
        if self.prefix_sampling_strategy == 'continuous_from_start':
            prefix_node_indices = gt_tour_node_indices[:prefix_k]
        else: # continuous_random_start
            start_node_idx = np.random.randint(0, self.num_nodes)
            rolled_tour = torch.roll(gt_tour_node_indices, shifts=-start_node_idx, dims=0)
            prefix_node_indices = rolled_tour[:prefix_k]
            
        # --- Node State Feature (no changes here) ---
        node_prefix_state = torch.zeros((self.num_nodes, 1), dtype=torch.float32)
        if prefix_k > 0:
            node_prefix_state[prefix_node_indices] = 1.0
        
        # --- <<< CORE LOGIC CHANGE: Sparse vs Dense Target Generation >>> ---
        if self.sparse_factor > 0:
            # --- Sparse Mode ---
            if self.num_nodes <= self.sparse_factor:
                raise ValueError("k-NN sparse_factor must be smaller than num_nodes.")
            
            # 1. Build k-NN graph
            kdt = KDTree(instance_locs.numpy(), metric='euclidean')
            _, knn_indices = kdt.query(instance_locs.numpy(), k=self.sparse_factor)
            
            source_nodes = torch.arange(self.num_nodes).view(-1, 1).repeat(1, self.sparse_factor).flatten()
            target_nodes = torch.from_numpy(knn_indices).flatten()
            
            # Make graph undirected and remove self-loops
            edge_index = torch.stack([source_nodes, target_nodes], dim=0)
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            edge_index = edge_index[:, edge_index[0] != edge_index[1]] # remove self-loops
            
            # Consolidate duplicate edges
            edge_index_sorted, _ = torch.sort(edge_index, dim=0)
            edge_index = torch.unique(edge_index_sorted, dim=1)
            
            # 2. Create sparse target edge attributes
            tour_edges = set()
            tour_nodes_np = gt_tour_node_indices.numpy()
            for i in range(self.num_nodes - 1):
                # u, v = tour_nodes_np[i], tour_nodes_np[i+1]
                # tour_edges.add(tuple(sorted((u, v))))

                u_node = tour_nodes_np[i]
                v_node = tour_nodes_np[(i + 1) % self.num_nodes]
                
                # --- FIX PART 2: Force conversion to Python scalars using .item() ---
                # This is the most critical change. It guarantees that u and v are plain numbers
                # that `sorted()` and `set.add()` can handle without ambiguity.
                edge_tuple = tuple(sorted((u_node.item(), v_node.item())))
                tour_edges.add(edge_tuple)


            # tour_edges.add(tuple(sorted((tour_nodes_np[-1], tour_nodes_np[0]))))
            
            target_edge_attrs = torch.zeros(edge_index.shape[1], 1, dtype=torch.float32)
            for i in range(edge_index.shape[1]):
                u, v = edge_index[0, i].item(), edge_index[1, i].item()
                if tuple(sorted((u, v))) in tour_edges:
                    target_edge_attrs[i] = 1.0

            # <<< MODIFIED: compute distances here and apply per-graph min-max scaling >>>
            # 1. Compute the true Euclidean length of every edge of the current graph
            src, dst = edge_index[0], edge_index[1]
            distances = torch.linalg.norm(instance_locs[src] - instance_locs[dst], dim=-1)

            # 2. Find the min and max distance of the current graph
            min_dist = distances.min()
            max_dist = distances.max()
            epsilon = 1e-8 # Avoid division by zero when max_dist == min_dist

            # 3. Apply min-max scaling to map distances to [0, 1]
            normalized_distances = (distances - min_dist) / (max_dist - min_dist + epsilon)
            
            # 4. Make sure it has shape [Num_Edges, 1] for downstream processing
            dist_feature = normalized_distances.unsqueeze(-1)
            # <<< END MODIFICATION >>>


            return {
                "instance_locs": instance_locs,
                "prefix_nodes": prefix_node_indices,
                "node_prefix_state": node_prefix_state,
                "edge_index": edge_index,
                "target_edge_attrs": target_edge_attrs,
                "dist_feature": dist_feature, # <<< MODIFIED: return the new normalized distance feature
                "num_nodes": self.num_nodes
            }
        else:
            # --- Dense Mode ---
            target_adj_matrix = construct_target_adj_from_tour(
                gt_tour_node_indices, self.num_nodes, device='cpu'
            )
            return {
                "instance_locs": instance_locs,
                "prefix_nodes": prefix_node_indices,
                "node_prefix_state": node_prefix_state,
                "target_adj_matrix": target_adj_matrix
            }


# <<< MODIFIED: The collate function is now much more complex to handle sparse batching
def custom_collate_fn(batch):
    # Check if the first item is sparse or dense
    is_sparse = 'edge_index' in batch[0]

    # Standard batching for common items
    instance_locs = torch.stack([item['instance_locs'] for item in batch], dim=0)
    node_prefix_states = torch.stack([item['node_prefix_state'] for item in batch], dim=0)
    prefix_nodes_list = [item['prefix_nodes'] for item in batch]
    prefix_lengths = torch.tensor([len(p) for p in prefix_nodes_list], dtype=torch.long)
    padded_prefixes = rnn_utils.pad_sequence(prefix_nodes_list, batch_first=True, padding_value=0) # Use a padding value not in node indices
    # <<< MODIFIED: add num_nodes to the batch dict >>>
    # This fixes a KeyError. We take it from the first sample in the batch.

    
    batched_data = {
        "instance_locs": instance_locs,
        "prefix_nodes": padded_prefixes,
        "prefix_lengths": prefix_lengths,
        "node_prefix_state": node_prefix_states,
        "is_sparse": is_sparse
    }
    if batch:
        batched_data["num_nodes"] = batch[0]["num_nodes"]
    # <<< END MODIFICATION >>>
    if is_sparse:
        # --- Sparse Batching (mimics torch_geometric.data.Batch) ---
        node_counts = [item['num_nodes'] for item in batch]
        node_cumsum = torch.tensor([0] + list(np.cumsum(node_counts)[:-1]), dtype=torch.long)
        
        # Create a batch vector mapping each node to its graph index
        # Example: [0,0,0, 1,1,1, 2,2,2] for 3 graphs of 3 nodes each
        node_to_graph_batch = torch.cat([
            torch.full((n,), i, dtype=torch.long) for i, n in enumerate(node_counts)
        ])
        
        # Combine edge_index and shift node indices
        edge_indices = []
        for i, item in enumerate(batch):
            edge_indices.append(item['edge_index'] + node_cumsum[i])
        
        batched_data["edge_index"] = torch.cat(edge_indices, dim=1)
        batched_data["target_edge_attrs"] = torch.cat([item['target_edge_attrs'] for item in batch], dim=0)
        batched_data["node_to_graph_batch"] = node_to_graph_batch
        # Overwrite instance_locs and node_prefix_states to be flat tensors
        # <<< MODIFIED: concatenate the normalized distance feature like the other sparse features >>>
        batched_data["dist_feature"] = torch.cat([item['dist_feature'] for item in batch], dim=0)
        # <<< END MODIFICATION >>>

        batched_data["instance_locs"] = torch.cat([item['instance_locs'] for item in batch], dim=0)
        batched_data["node_prefix_state"] = torch.cat([item['node_prefix_state'] for item in batch], dim=0)
    else:
        # --- Dense Batching ---
        batched_data["target_adj_matrix"] = torch.stack([item['target_adj_matrix'] for item in batch], dim=0)

    return batched_data