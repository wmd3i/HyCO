# OP dataset parser (used by gurobi_op.py).

import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.utils.rnn as rnn_utils
from sklearn.neighbors import KDTree # <<< MODIFIED: For k-NN graph construction

import re
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


# +++ NEW +++: Parser for the CVRP text file format
def parse_cvrp_line(line):
    """Parses a single line of the CVRP text file."""
    parts = line.strip().split()
    
    # Find indices of keywords
    depots_idx = parts.index('depots')
    points_idx = parts.index('points')
    demands_idx = parts.index('demands')
    capacity_idx = parts.index('capacity')
    output_idx = parts.index('output')

    # Extract depot coordinates
    depot_coords = [float(parts[depots_idx + 1]), float(parts[depots_idx + 2])]
    
    # Extract customer coordinates
    customer_coords_flat = [float(x) for x in parts[points_idx + 1:demands_idx]]
    customer_coords = np.array(customer_coords_flat).reshape(-1, 2).tolist()
    
    # Combine depot and customer coordinates (depot is node 0)
    all_coords = np.array([depot_coords] + customer_coords, dtype=np.float32)
    
    # Extract demands (depot demand is 0)
    demands = [0.0] + [float(x) for x in parts[demands_idx + 1:capacity_idx]]
    demands = np.array(demands, dtype=np.float32)
    
    # Extract capacity
    capacity = float(parts[capacity_idx + 1])
    
    # Extract solution tour
    tour = [int(x) for x in parts[output_idx + 1:]]
    
    # The raw tour from file may not have customers indexed from 1.
    # We assume the problem is standard: node 0 is depot, customers are 1...N
    # The locations array is already in this order. The tour refers to indices in this array.
    
    return all_coords, demands, capacity, tour


class CVRPConditionalSuffixDataset(Dataset): # <<< MODIFIED: Renamed class
    def __init__(self, txt_file_paths, prefix_k_options, prefix_sampling_strategy='continuous_from_start', sparse_factor=-1):
        self.instances = []
        if isinstance(txt_file_paths, str):
            txt_file_paths = [txt_file_paths]

        for file_path in txt_file_paths:
            with open(file_path, 'r') as f:
                for line in f:
                    if line.strip():
                        coords, demands, capacity, tour = parse_cvrp_line(line)
                        self.instances.append({
                            "locs": torch.tensor(coords, dtype=torch.float32),
                            "demands": torch.tensor(demands, dtype=torch.float32),
                            "capacity": torch.tensor(capacity, dtype=torch.float32),
                            "tour": torch.tensor(tour, dtype=torch.long)
                        })
        
        self.num_samples = len(self.instances)
        self.num_nodes = self.instances[0]["locs"].shape[0] if self.num_samples > 0 else 0
        self.prefix_k_options = prefix_k_options
        self.prefix_sampling_strategy = prefix_sampling_strategy
        self.sparse_factor = sparse_factor

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        instance = self.instances[idx]
        instance_locs = instance["locs"]
        demands = instance["demands"]
        capacity = instance["capacity"]
        gt_tour_node_indices = instance["tour"]

        prefix_k = np.random.choice(self.prefix_k_options)
        
        if self.prefix_sampling_strategy == 'continuous_from_start':
            prefix_node_indices = gt_tour_node_indices[:prefix_k]
        else: # continuous_random_start
            # Find a random starting point that isn't a depot (unless it's the first node)
            valid_starts = [i for i, node in enumerate(gt_tour_node_indices) if node != 0 or i == 0]
            start_node_idx = np.random.choice(valid_starts)
            rolled_tour = torch.roll(gt_tour_node_indices, shifts=-start_node_idx, dims=0)
            prefix_node_indices = rolled_tour[:prefix_k]

        # +++ NEW: CVRP Node Features +++
        # Feature 1: IsDepot (1 for depot, 0 for customer)
        is_depot_feature = torch.zeros((self.num_nodes, 1), dtype=torch.float32)
        is_depot_feature[0] = 1.0
        
        # Feature 2: Normalized Demand
        normalized_demand_feature = (demands / capacity).unsqueeze(-1)

        # Feature 3: IsInPrefix
        node_prefix_state = torch.zeros((self.num_nodes, 1), dtype=torch.float32)
        if prefix_k > 0:
            node_prefix_state[prefix_node_indices] = 1.0

        # Combine all node state features
        node_state_features = torch.cat([is_depot_feature, normalized_demand_feature, node_prefix_state], dim=-1)

        # <<< MODIFIED: Target generation for sparse CVRP >>>
        if self.sparse_factor > 0:
            if self.num_nodes <= self.sparse_factor:
                raise ValueError("k-NN sparse_factor must be smaller than num_nodes.")
            
            kdt = KDTree(instance_locs.numpy(), metric='euclidean')
            _, knn_indices = kdt.query(instance_locs.numpy(), k=self.sparse_factor)
            
            source_nodes = torch.arange(self.num_nodes).view(-1, 1).repeat(1, self.sparse_factor).flatten()
            target_nodes = torch.from_numpy(knn_indices).flatten()
            
            edge_index = torch.stack([source_nodes, target_nodes], dim=0)
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            edge_index = edge_index[:, edge_index[0] != edge_index[1]]
            edge_index_sorted, _ = torch.sort(edge_index, dim=0)
            edge_index = torch.unique(edge_index_sorted, dim=1)
            
            tour_edges = set()
            tour_nodes_np = gt_tour_node_indices.numpy()
            for i in range(len(tour_nodes_np) - 1):
                u, v = tour_nodes_np[i], tour_nodes_np[i+1]
                if u != 0 or v != 0: # Add edge unless it's 0-0
                    tour_edges.add(tuple(sorted((u.item(), v.item()))))

            target_edge_attrs = torch.zeros(edge_index.shape[1], 1, dtype=torch.float32)
            for i in range(edge_index.shape[1]):
                u, v = edge_index[0, i].item(), edge_index[1, i].item()
                if tuple(sorted((u, v))) in tour_edges:
                    target_edge_attrs[i] = 1.0

            src, dst = edge_index[0], edge_index[1]
            distances = torch.linalg.norm(instance_locs[src] - instance_locs[dst], dim=-1)
            min_dist, max_dist = distances.min(), distances.max()
            epsilon = 1e-8
            normalized_distances = (distances - min_dist) / (max_dist - min_dist + epsilon)
            dist_feature = normalized_distances.unsqueeze(-1)

            return {
                "instance_locs": instance_locs,
                "demands": demands,
                "capacity": capacity,
                "prefix_nodes": prefix_node_indices,
                "node_state_features": node_state_features, # <<< MODIFIED: New feature tensor
                "edge_index": edge_index,
                "target_edge_attrs": target_edge_attrs,
                "dist_feature": dist_feature,
                "num_nodes": self.num_nodes
            }
        else:
            # Dense mode (not fully implemented in this guide, focus is on sparse)
            raise NotImplementedError("Dense mode for CVRP is not implemented in this guide.")

def custom_collate_fn(batch):
    is_sparse = 'edge_index' in batch[0]
    if not is_sparse:
        raise NotImplementedError("Only sparse mode is supported for CVRP.")

    # <<< MODIFIED: Batch new CVRP fields >>>
    instance_locs = torch.stack([item['instance_locs'] for item in batch], dim=0)
    demands = torch.stack([item['demands'] for item in batch], dim=0)
    capacities = torch.stack([item['capacity'] for item in batch], dim=0)
    node_state_features = torch.stack([item['node_state_features'] for item in batch], dim=0)
    
    prefix_nodes_list = [item['prefix_nodes'] for item in batch]
    prefix_lengths = torch.tensor([len(p) for p in prefix_nodes_list], dtype=torch.long)
    padded_prefixes = rnn_utils.pad_sequence(prefix_nodes_list, batch_first=True, padding_value=0)
    
    batched_data = {
        "instance_locs_orig": instance_locs, # Keep original for reference
        "demands": demands,
        "capacities": capacities,
        "prefix_nodes": padded_prefixes,
        "prefix_lengths": prefix_lengths,
        "is_sparse": is_sparse
    }
    if batch:
        batched_data["num_nodes"] = batch[0]["num_nodes"]

    # --- Sparse Batching ---
    node_counts = [item['num_nodes'] for item in batch]
    node_cumsum = torch.tensor([0] + list(np.cumsum(node_counts)[:-1]), dtype=torch.long)
    
    node_to_graph_batch = torch.cat([
        torch.full((n,), i, dtype=torch.long) for i, n in enumerate(node_counts)
    ])
    
    edge_indices = [item['edge_index'] + node_cumsum[i] for i, item in enumerate(batch)]
    
    batched_data["edge_index"] = torch.cat(edge_indices, dim=1)
    batched_data["target_edge_attrs"] = torch.cat([item['target_edge_attrs'] for item in batch], dim=0)
    batched_data["node_to_graph_batch"] = node_to_graph_batch
    batched_data["dist_feature"] = torch.cat([item['dist_feature'] for item in batch], dim=0)
    
    # Flatten node-level features for sparse GNN
    batched_data["instance_locs"] = torch.cat([item['instance_locs'] for item in batch], dim=0)
    batched_data["node_state_features"] = torch.cat([item['node_state_features'] for item in batch], dim=0)
    
    return batched_data
    
    
# +++ NEW: Parser for the OP text file format +++
def parse_op_line(line):
    """Parses a single line of the OP text file."""
    parts = line.strip().split(';')
    data = {}
    for part in parts:
        if ':' in part:
            key, value = part.split(':', 1)
            data[key.strip()] = value.strip()

    # Extract depot coordinates
    depot_coords = [float(x) for x in data['depots'].split()]
    
    # Extract customer coordinates
    customer_coords_flat = [float(x) for x in data['points'].split()]
    customer_coords = np.array(customer_coords_flat).reshape(-1, 2).tolist()
    
    # Combine depot and customer coordinates (depot is node 0)
    all_coords = np.array([depot_coords] + customer_coords, dtype=np.float32)
    
    # Extract prizes (depot prize is 0)
    prizes = [0.0] + [float(x) for x in data['prizes'].split()]
    prizes = np.array(prizes, dtype=np.float32)
    
    # Extract max_length
    max_length = float(data['max_length'])
    
    # Extract solution tour
    tour = [int(x) for x in data['output'].split()]
    
    return all_coords, prizes, max_length, tour


# +++ NEW: Dataset Class for Orienteering Problem +++
class OPConditionalSuffixDataset(Dataset):
    def __init__(self, txt_file_paths, prefix_k_options, prefix_sampling_strategy='continuous_from_start', sparse_factor=-1):
        self.instances = []
        if isinstance(txt_file_paths, str):
            txt_file_paths = [txt_file_paths]

        for file_path in txt_file_paths:
            with open(file_path, 'r') as f:
                for line in f:
                    if line.strip():
                        # Use the new OP parser
                        coords, prizes, max_length, tour = parse_op_line(line)
                        self.instances.append({
                            "locs": torch.tensor(coords, dtype=torch.float32),
                            "prizes": torch.tensor(prizes, dtype=torch.float32),
                            "max_length": torch.tensor(max_length, dtype=torch.float32),
                            "tour": torch.tensor(tour, dtype=torch.long)
                        })
        
        self.num_samples = len(self.instances)
        self.num_nodes = self.instances[0]["locs"].shape[0] if self.num_samples > 0 else 0
        self.prefix_k_options = prefix_k_options
        self.prefix_sampling_strategy = prefix_sampling_strategy
        self.sparse_factor = sparse_factor

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        instance = self.instances[idx]
        instance_locs = instance["locs"]
        prizes = instance["prizes"] # <<< MODIFIED
        max_length = instance["max_length"] # <<< MODIFIED
        gt_tour_node_indices = instance["tour"]
        actual_tour_len = len(gt_tour_node_indices)

        # --- MODIFIED: Data-aware prefix_k selection ---
        # Filter the global k options to only include those valid for THIS tour's length.
        # We use k < actual_tour_len because a prefix of length tour_len is the whole tour.
        valid_k_options = [k for k in self.prefix_k_options if k < actual_tour_len]

        if not valid_k_options:
            # If the tour is too short for any of the curriculum's k-values
            # (e.g., tour_len=3, k_options=[23, ...]), default to the hardest meaningful task: 
            # generate the entire tour from just the depot.
            # A prefix of length 1 is just the starting depot.
            prefix_k = 0 
        else:
            # Otherwise, choose a random k from the valid options
            prefix_k = np.random.choice(valid_k_options)
        # --- END OF MODIFICATION ---
        
        # Prefix sampling logic can remain the same
        if self.prefix_sampling_strategy == 'continuous_from_start':
            prefix_node_indices = gt_tour_node_indices[:prefix_k]
        else: # continuous_random_start
            valid_starts = [i for i, node in enumerate(gt_tour_node_indices) if node != 0 or i == 0]
            start_node_idx = np.random.choice(valid_starts)
            rolled_tour = torch.roll(gt_tour_node_indices, shifts=-start_node_idx, dims=0)
            prefix_node_indices = rolled_tour[:prefix_k]

        # +++ NEW: OP Node Features +++
        # Feature 1: IsDepot (1 for depot, 0 for customer)
        is_depot_feature = torch.zeros((self.num_nodes, 1), dtype=torch.float32)
        is_depot_feature[0] = 1.0
        
        # Feature 2: Prize (already normalized in your data)
        prize_feature = prizes.unsqueeze(-1)

        # Feature 3: IsInPrefix
        node_prefix_state = torch.zeros((self.num_nodes, 1), dtype=torch.float32)
        if prefix_k > 0:
            node_prefix_state[prefix_node_indices] = 1.0

        # Combine all node state features
        node_state_features = torch.cat([is_depot_feature, prize_feature, node_prefix_state], dim=-1)

        # Graph construction and target generation logic remains the same
        if self.sparse_factor > 0:
            if self.num_nodes <= self.sparse_factor:
                raise ValueError("k-NN sparse_factor must be smaller than num_nodes.")
            
            kdt = KDTree(instance_locs.numpy(), metric='euclidean')
            _, knn_indices = kdt.query(instance_locs.numpy(), k=self.sparse_factor)
            
            source_nodes = torch.arange(self.num_nodes).view(-1, 1).repeat(1, self.sparse_factor).flatten()
            target_nodes = torch.from_numpy(knn_indices).flatten()
            
            edge_index = torch.stack([source_nodes, target_nodes], dim=0)
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            edge_index = edge_index[:, edge_index[0] != edge_index[1]]
            edge_index_sorted, _ = torch.sort(edge_index, dim=0)
            edge_index = torch.unique(edge_index_sorted, dim=1)
            
            # The logic to create tour_edges from the output tour is identical
            # --- CORRECTED LOGIC for creating target tour edges ---
            tour_edges = set()
            tour_nodes_np = gt_tour_node_indices.numpy()
            
            # A valid tour must have at least 2 nodes (e.g., 0 -> 0 for an empty tour)
            if len(tour_nodes_np) > 1:
                # Iterate through all segments of the tour: (node_1, node_2), (node_2, node_3), ...
                for i in range(len(tour_nodes_np) - 1):
                    u = tour_nodes_np[i].item()
                    v = tour_nodes_np[i+1].item()
                    
                    # Add the edge to the set, using a canonical representation (sorted tuple)
                    tour_edges.add(tuple(sorted((u, v))))

            target_edge_attrs = torch.zeros(edge_index.shape[1], 1, dtype=torch.float32)
            for i in range(edge_index.shape[1]):
                u, v = edge_index[0, i].item(), edge_index[1, i].item()
                if tuple(sorted((u, v))) in tour_edges:
                    target_edge_attrs[i] = 1.0
            # --- END OF CORRECTION ---

            src, dst = edge_index[0], edge_index[1]
            distances = torch.linalg.norm(instance_locs[src] - instance_locs[dst], dim=-1)
            min_dist, max_dist = distances.min(), distances.max()
            epsilon = 1e-8
            normalized_distances = (distances - min_dist) / (max_dist - min_dist + epsilon)
            dist_feature = normalized_distances.unsqueeze(-1)

            return {
                "instance_locs": instance_locs,
                "prizes": prizes, # <<< MODIFIED
                "max_length": max_length, # <<< MODIFIED
                "prefix_nodes": prefix_node_indices,
                "node_state_features": node_state_features, # <<< New feature tensor
                "edge_index": edge_index,
                "target_edge_attrs": target_edge_attrs,
                "dist_feature": dist_feature,
                "num_nodes": self.num_nodes
            }
        else:
            raise NotImplementedError("Dense mode for OP is not implemented.")

# +++ NEW: Collate function for OP batches +++
def op_custom_collate_fn(batch):
    is_sparse = 'edge_index' in batch[0]
    if not is_sparse:
        raise NotImplementedError("Only sparse mode is supported for OP.")

    # Batch new OP fields
    instance_locs = torch.stack([item['instance_locs'] for item in batch], dim=0)
    prizes = torch.stack([item['prizes'] for item in batch], dim=0)
    max_lengths = torch.stack([item['max_length'] for item in batch], dim=0)
    node_state_features = torch.stack([item['node_state_features'] for item in batch], dim=0)
    
    prefix_nodes_list = [item['prefix_nodes'] for item in batch]
    prefix_lengths = torch.tensor([len(p) for p in prefix_nodes_list], dtype=torch.long)
    padded_prefixes = rnn_utils.pad_sequence(prefix_nodes_list, batch_first=True, padding_value=0)
    
    batched_data = {
        "instance_locs_orig": instance_locs,
        "prizes": prizes, # <<< MODIFIED
        "max_lengths": max_lengths, # <<< MODIFIED
        "prefix_nodes": padded_prefixes,
        "prefix_lengths": prefix_lengths,
        "is_sparse": is_sparse
    }
    if batch:
        batched_data["num_nodes"] = batch[0]["num_nodes"]

    # Sparse Batching (this logic remains identical)
    node_counts = [item['num_nodes'] for item in batch]
    node_cumsum = torch.tensor([0] + list(np.cumsum(node_counts)[:-1]), dtype=torch.long)
    
    node_to_graph_batch = torch.cat([
        torch.full((n,), i, dtype=torch.long) for i, n in enumerate(node_counts)
    ])
    
    edge_indices = [item['edge_index'] + node_cumsum[i] for i, item in enumerate(batch)]
    
    batched_data["edge_index"] = torch.cat(edge_indices, dim=1)
    batched_data["target_edge_attrs"] = torch.cat([item['target_edge_attrs'] for item in batch], dim=0)
    batched_data["node_to_graph_batch"] = node_to_graph_batch
    batched_data["dist_feature"] = torch.cat([item['dist_feature'] for item in batch], dim=0)
    
    # Flatten node-level features for sparse GNN
    batched_data["instance_locs"] = torch.cat([item['instance_locs'] for item in batch], dim=0)
    batched_data["node_state_features"] = torch.cat([item['node_state_features'] for item in batch], dim=0)
    
    return batched_data
    
    
