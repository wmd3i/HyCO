# tsp_utils.py: TSP cost computation, batched 2-opt, tour decoding and visualization helpers.
import torch
import matplotlib.pyplot as plt # For visualization


def calculate_segment_cost_gpu(node_idx1_batch, node_idx2_batch, instance_locs_batch):
    """
    Helper function to calculate cost between two specific nodes for a batch.
    node_idx1_batch: (B) tensor of first node indices.
    node_idx2_batch: (B) tensor of second node indices.
    instance_locs_batch: (B, N, 2) tensor of city coordinates.
    """
    B = instance_locs_batch.shape[0]
    # Create indices for gathering: (B, 1, 2) for node coordinates
    idx1 = node_idx1_batch.unsqueeze(-1).unsqueeze(-1).expand(B, 1, 2)
    idx2 = node_idx2_batch.unsqueeze(-1).unsqueeze(-1).expand(B, 1, 2)

    loc1_batch = torch.gather(instance_locs_batch, 1, idx1).squeeze(1) # (B, 2)
    loc2_batch = torch.gather(instance_locs_batch, 1, idx2).squeeze(1) # (B, 2)
    return torch.sqrt(((loc1_batch - loc2_batch)**2).sum(dim=1)) # (B)


def calculate_tsp_cost_batch(instance_locs_batch, tour_indices_batch):
    """
    Calculates the total length of TSP tours for a batch.
    instance_locs_batch: (B, N, 2) tensor of city coordinates.
    tour_indices_batch: (B, N_tour) tensor of city indices in tour order.
                       N_tour can be <= N if tours are partial or not full.
    """
    B, N, _ = instance_locs_batch.shape
    N_tour = tour_indices_batch.shape[1]

    if N_tour == 0:
        return torch.zeros(B, device=instance_locs_batch.device)
    if N_tour < 2: # A tour with less than 2 nodes has zero cost
        return torch.zeros(B, device=instance_locs_batch.device)

    # Gather tour locations: (B, N_tour, 2)
    # tour_indices_batch needs to be (B, N_tour, 1) then expanded for gather
    tour_indices_expanded = tour_indices_batch.unsqueeze(-1).expand(B, N_tour, 2)
    tour_locs_batch = torch.gather(instance_locs_batch, 1, tour_indices_expanded)

    # Calculate segment lengths: (B, N_tour-1)
    segment_lengths = torch.sqrt(((tour_locs_batch[:, :-1] - tour_locs_batch[:, 1:])**2).sum(dim=2))

    # Calculate closing segment lengths: (B)
    closing_segment_diff_sq = ((tour_locs_batch[:, -1] - tour_locs_batch[:, 0])**2) # (B, 2)
    closing_segment_lengths = torch.sqrt(closing_segment_diff_sq.sum(dim=1)) # (B)

    total_costs = segment_lengths.sum(dim=1) + closing_segment_lengths
    return total_costs # (B)

# def apply_2opt_batch(initial_tours_batch, instance_locs_batch, max_iterations=300):
#     """
#     Applies the 2-opt local search algorithm to improve a batch of TSP tours on GPU.
#     Optimized to vectorize tour reversal.

#     initial_tours_batch: (B, N_nodes) tensor of node indices in the initial tour order.
#     instance_locs_batch: (B, N_nodes, 2) tensor of city coordinates.
#     max_iterations: Maximum number of iterations.

#     Returns: (B, N_nodes) tensor of node indices in the optimized tour order.
#     """
#     device = instance_locs_batch.device
#     B, num_nodes, _ = instance_locs_batch.shape

#     if num_nodes < 4: # 2-opt requires at least 4 nodes
#         return initial_tours_batch

#     current_tours_tensor = initial_tours_batch.clone().long()
#     best_tours_tensor = current_tours_tensor.clone()
    
#     best_costs = calculate_tsp_cost_batch(instance_locs_batch, best_tours_tensor)

#     for iter_count in range(max_iterations):
#         improved_in_pass = torch.zeros(B, dtype=torch.bool, device=device) # Tracks if any tour improved in this pass
        
#         for i in range(num_nodes - 2):
#             for j in range(i + 2, num_nodes):
#                 # Current edges for all tours in the batch
#                 node_i_indices = current_tours_tensor[:, i]
#                 node_i_plus_1_indices = current_tours_tensor[:, i+1]
#                 node_j_indices = current_tours_tensor[:, j]
#                 node_j_plus_1_indices = current_tours_tensor[:, (j + 1) % num_nodes]

#                 # Cost of current two edges for the batch (GPU calculation)
#                 cost_edge_i_ip1 = calculate_segment_cost_gpu(node_i_indices, node_i_plus_1_indices, instance_locs_batch)
#                 cost_edge_j_jp1 = calculate_segment_cost_gpu(node_j_indices, node_j_plus_1_indices, instance_locs_batch)
#                 current_edge_pair_costs = cost_edge_i_ip1 + cost_edge_j_jp1 # (B)

#                 # Cost of new two edges if swapped (GPU calculation)
#                 cost_edge_i_j = calculate_segment_cost_gpu(node_i_indices, node_j_indices, instance_locs_batch)
#                 cost_edge_ip1_jp1 = calculate_segment_cost_gpu(node_i_plus_1_indices, node_j_plus_1_indices, instance_locs_batch)
#                 new_edge_pair_costs = cost_edge_i_j + cost_edge_ip1_jp1 # (B)

#                 # Mask for tours where swapping these two edges is beneficial (edge cost heuristic)
#                 edge_improvement_mask = new_edge_pair_costs < current_edge_pair_costs # (B)
                
#                 if edge_improvement_mask.any():
#                     # Clone current tours to create a temporary version with potential swaps
#                     temp_swapped_tours = current_tours_tensor.clone()

#                     # Identify the actual indices in the batch that show edge improvement
#                     indices_for_swap = torch.where(edge_improvement_mask)[0]
                    
#                     if len(indices_for_swap) > 0:
#                         # Extract the subset of tours that need modification
#                         tours_to_modify_subset = current_tours_tensor[indices_for_swap]

#                         # Perform vectorized 2-opt swap (reverse segment) on the GPU
#                         # Segment is from index i+1 to j (inclusive)
#                         idx_segment_start = i + 1
#                         idx_segment_end = j 

#                         prefix = tours_to_modify_subset[:, :idx_segment_start]
#                         segment_to_flip = tours_to_modify_subset[:, idx_segment_start : idx_segment_end + 1]
#                         suffix = tours_to_modify_subset[:, idx_segment_end + 1 :]
                        
#                         flipped_segment = torch.flip(segment_to_flip, dims=[1])
                        
#                         # Concatenate parts to form the new tours for the subset
#                         modified_tours_subset = torch.cat([prefix, flipped_segment, suffix], dim=1)
                        
#                         # Place the modified subset back into the temporary full batch tensor
#                         temp_swapped_tours[indices_for_swap] = modified_tours_subset
                    
#                     # Recalculate full tour costs for tours in temp_swapped_tours
#                     # This confirms if the edge swap leads to an overall tour improvement.
#                     new_total_costs = calculate_tsp_cost_batch(instance_locs_batch, temp_swapped_tours)
                    
#                     # Mask for tours where the new total cost is better than the current best known cost
#                     total_cost_improvement_mask = new_total_costs < best_costs
                    
#                     # Final update mask: must satisfy both the edge heuristic AND overall cost reduction
#                     final_update_mask = edge_improvement_mask & total_cost_improvement_mask
                                        
#                     if final_update_mask.any():
#                         best_tours_tensor[final_update_mask] = temp_swapped_tours[final_update_mask]
#                         best_costs[final_update_mask] = new_total_costs[final_update_mask]
#                         improved_in_pass[final_update_mask] = True # Mark that these tours improved in this pass
                        
#                         # CRITICAL: Update current_tours_tensor for the next i,j iteration within this pass
#                         # with the tours that have shown improvement.
#                         current_tours_tensor[final_update_mask] = temp_swapped_tours[final_update_mask]

#         if not improved_in_pass.any(): # If no tour in the batch was improved in this entire pass over i,j
#             # print(f"  2-opt converged in {iter_count+1} iterations for this batch.")
#             break # Exit max_iterations loop early
            
#     # print(f"  2-opt finished after {iter_count+1} iterations for this batch.")
#     return best_tours_tensor


def apply_2opt_batch(initial_tours_batch, instance_locs_batch, max_iterations=1000):
    """
    Fully vectorized 2-opt implementation based on DIFUSCO logic, optimized for pure PyTorch GPU tensors.
    
    Args:
        initial_tours_batch: (B, N) long tensor, node indices.
        instance_locs_batch: (B, N, 2) float tensor, coordinates.
        max_iterations: max 2-opt steps.
    
    Returns:
        (B, N) long tensor, optimized tours.
    """
    with torch.inference_mode():
        cuda_tour = initial_tours_batch.clone()
        cuda_points = instance_locs_batch
        
        batch_size, num_nodes = cuda_tour.shape
        
        # Early check: instances with fewer than 4 nodes do not need 2-opt
        if num_nodes < 4:
            return cuda_tour

        iterator = 0
        completed_mask = torch.zeros(batch_size, dtype=torch.bool, device=cuda_tour.device)

        while iterator < max_iterations:
            # Exit early if all instances in the batch have converged
            if completed_mask.all():
                break

            # 1. Prepare data: use slicing and broadcasting to get the coordinates of all i and j at once
            # tour shape: (B, N)
            # rolled tour: shifted by one position, representing i+1 and j+1
            tour_rolled = torch.roll(cuda_tour, shifts=-1, dims=1)
            
            # Gather coordinates based on tour indices
            # coords shape: (B, N, 2) -> coordinates reordered by the current tour
            # Key step: reorder coordinates along the current tour so that index i is the i-th node on the tour
            ordered_locs = torch.gather(cuda_points, 1, cuda_tour.unsqueeze(-1).expand(-1, -1, 2))
            ordered_locs_next = torch.roll(ordered_locs, shifts=-1, dims=1)

            # 2. Build the distance matrices (vectorized)
            # We need the cost change of swapping edge(i, i+1) and edge(j, j+1)
            # Delta = d(i, j) + d(i+1, j+1) - d(i, i+1) - d(j, j+1)
            
            # Here i, j are position indices along the tour (0 to N-1)
            # Shapes:
            # points_i: (B, N, 1, 2)
            # points_j: (B, 1, N, 2)
            
            points_i = ordered_locs.unsqueeze(2)
            points_i_plus_1 = ordered_locs_next.unsqueeze(2)
            
            points_j = ordered_locs.unsqueeze(1)
            points_j_plus_1 = ordered_locs_next.unsqueeze(1)

            # Compute distances for all (i, j) combinations
            # A_ij corresponds to d(node_i, node_j)
            # dist_sq: (B, N, N)
            d_i_i_plus_1 = torch.sum((points_i - points_i_plus_1) ** 2, dim=-1).sqrt() # (B, N, 1)
            d_j_j_plus_1 = torch.sum((points_j - points_j_plus_1) ** 2, dim=-1).sqrt() # (B, 1, N)
            
            d_i_j = torch.sum((points_i - points_j) ** 2, dim=-1).sqrt() # (B, N, N)
            d_ip1_jp1 = torch.sum((points_i_plus_1 - points_j_plus_1) ** 2, dim=-1).sqrt() # (B, N, N)

            # Compute the change (B, N, N)
            change = d_i_j + d_ip1_jp1 - d_i_i_plus_1 - d_j_j_plus_1
            
            # 3. Masking
            # A 2-opt move requires i < j (i.e., j > i+1) and non-adjacent edges
            # torch.triu(change, diagonal=2) zeros out the lower triangle, the diagonal and the adjacent band (they should actually be +inf or ignored)
            # Only look at the upper triangle with offset=2
            valid_change = torch.triu(change, diagonal=2)

            # Set invalid regions (lower triangle and near-diagonal) to +inf so that min never selects them
            # Note: an original 0 in valid_change could be a valid value (if change happens to be 0), but triu also zeroes invalid regions
            # A more robust approach is to build an explicit mask
            mask = torch.ones_like(change, dtype=torch.bool).triu(diagonal=2)
            valid_change = torch.where(mask, change, torch.tensor(float('inf'), device=change.device))

            # 4. Find the move with the largest cost reduction in each batch element (min delta)
            min_change_values, flatten_argmin_index = torch.min(valid_change.reshape(batch_size, -1), dim=-1)
            
            # 5. Update the tours
            # Only update where min_change < -epsilon
            update_mask = (min_change_values < -1e-5) & (~completed_mask)
            
            if not update_mask.any():
                break # No instance in the batch can be improved any more

            # Update completed_mask: an instance that cannot be improved this round is finished
            # Note: this could be simplified: keep going as long as one instance can be improved and skip the others this round
            # For efficiency we record which instances have converged
            # Current logic: update wherever the change is negative, skip the rest this round, and exit when nothing improves
            
            # Get the indices i and j
            min_i = torch.div(flatten_argmin_index, num_nodes, rounding_mode='floor')
            min_j = torch.remainder(flatten_argmin_index, num_nodes)
            
            # Perform the flip
            # Flipping slices of different lengths is hard to fully vectorize in PyTorch, so we fall back to a loop here
            # The loop is over the batch size (e.g. 64), not the number of nodes (e.g. 100)
            # Compared with the original N^2 loop, this B loop is very fast
            
            indices = torch.nonzero(update_mask).squeeze(-1)
            for idx in indices:
                idx = idx.item()
                i_val = min_i[idx].item()
                j_val = min_j[idx].item()
                
                # 2-opt swap: reverse segment from i+1 to j
                # Note: in the DIFUSCO code the range is min_i[i]+1 : min_j[i]+1
                p1, p2 = i_val + 1, j_val + 1
                cuda_tour[idx, p1:p2] = torch.flip(cuda_tour[idx, p1:p2], dims=(0,))
            
            iterator += 1

        return cuda_tour



def apply_2opt_batch_bk(initial_tours_batch, instance_locs_batch, max_iterations=100):
    """
    Applies the 2-opt local search algorithm to improve a batch of TSP tours on GPU.

    initial_tours_batch: (B, N_nodes) tensor of node indices in the initial tour order.
    instance_locs_batch: (B, N_nodes, 2) tensor of city coordinates.
    max_iterations: Maximum number of iterations.

    Returns: (B, N_nodes) tensor of node indices in the optimized tour order.
    """
    device = instance_locs_batch.device
    B, num_nodes, _ = instance_locs_batch.shape

    if num_nodes < 4: # 2-opt requires at least 4 nodes
        return initial_tours_batch

    current_tours_tensor = initial_tours_batch.clone().long()
    best_tours_tensor = current_tours_tensor.clone()
    
    # Calculate initial costs for the batch
    best_costs = calculate_tsp_cost_batch(instance_locs_batch, best_tours_tensor)

    for iter_count in range(max_iterations):
        improved_batch = torch.zeros(B, dtype=torch.bool, device=device)
        
        for i in range(num_nodes - 2):
            for j in range(i + 2, num_nodes):
                # Current edges: (node_i, node_i_plus_1) and (node_j, node_j_plus_1)
                node_i_indices = current_tours_tensor[:, i]
                node_i_plus_1_indices = current_tours_tensor[:, i+1]
                node_j_indices = current_tours_tensor[:, j]
                node_j_plus_1_indices = current_tours_tensor[:, (j + 1) % num_nodes] # Handle loop

                # Cost of current two edges for the batch
                cost_edge_i_ip1 = calculate_segment_cost_gpu(node_i_indices, node_i_plus_1_indices, instance_locs_batch)
                cost_edge_j_jp1 = calculate_segment_cost_gpu(node_j_indices, node_j_plus_1_indices, instance_locs_batch)
                current_edge_costs = cost_edge_i_ip1 + cost_edge_j_jp1 # (B)

                # Cost of new two edges if swapped: (node_i, node_j) and (node_i_plus_1, node_j_plus_1)
                cost_edge_i_j = calculate_segment_cost_gpu(node_i_indices, node_j_indices, instance_locs_batch)
                cost_edge_ip1_jp1 = calculate_segment_cost_gpu(node_i_plus_1_indices, node_j_plus_1_indices, instance_locs_batch)
                new_edge_costs = cost_edge_i_j + cost_edge_ip1_jp1 # (B)

                # Identify improvements for the batch
                improvement_mask = new_edge_costs < current_edge_costs # (B)
                
                if improvement_mask.any():
                    # Create new tours for those that improve
                    temp_new_tours = current_tours_tensor.clone()
                    
                    # Perform 2-opt swap (reverse segment) only for tours that improve
                    # This is tricky to vectorize perfectly without advanced indexing or loops.
                    # For simplicity, we can iterate here or use more complex tensor ops.
                    # Let's try a loop for clarity first, then consider vectorization.
                    for k in range(B):
                        if improvement_mask[k]:
                            tour_to_modify = temp_new_tours[k].tolist() # Convert to list for easy slicing/reversal
                            segment_to_reverse = tour_to_modify[i+1 : j+1]
                            segment_to_reverse.reverse()
                            temp_new_tours[k] = torch.tensor(
                                tour_to_modify[:i+1] + segment_to_reverse + tour_to_modify[j+1:],
                                device=device, dtype=torch.long
                            )
                    
                    # Recalculate costs for the modified tours
                    current_new_tours_costs = calculate_tsp_cost_batch(instance_locs_batch, temp_new_tours)
                    
                    # Update tours and costs where improvement actually lowered total cost
                    # (Edge cost delta is a heuristic, full cost check is more robust)
                    update_mask = current_new_tours_costs < best_costs 
                    final_update_mask = improvement_mask & update_mask # Ensure both edge heuristic and total cost improve

                    if final_update_mask.any():
                        best_tours_tensor[final_update_mask] = temp_new_tours[final_update_mask]
                        best_costs[final_update_mask] = current_new_tours_costs[final_update_mask]
                        improved_batch[final_update_mask] = True
                        current_tours_tensor[final_update_mask] = temp_new_tours[final_update_mask]


        if not improved_batch.any(): # If no tour in the batch improved
            break
            
    # print(f"2-opt batch finished in {iter_count+1} iterations.")
    return best_tours_tensor


# def decode_adj_matrices_to_tours_batch(adj_matrices_probs, batch_prefix_nodes, num_nodes):
#     """
#     Decodes a batch of probabilistic adjacency matrices into TSP tours on GPU.
#     adj_matrices_probs: (B, N, N) tensor of edge probabilities (e.g., after sigmoid).
#     batch_prefix_nodes: (B, k) tensor of fixed prefix node indices. k can be 0.
#     num_nodes: Total number of nodes (N).

#     Returns: A (B, N) tensor of node indices representing the tours.
#              Returns (B, N) tensor of -1 for failed decodings.
#     """
#     B, N, _ = adj_matrices_probs.shape
#     k = batch_prefix_nodes.shape[1]
#     device = adj_matrices_probs.device

#     # Initialize tours: (B, N), filled with -1 (or a placeholder for not set)
#     tours = torch.full((B, N), -1, dtype=torch.long, device=device)
    
#     # Visited mask: (B, N)
#     visited_mask = torch.zeros((B, N), dtype=torch.bool, device=device)
    
#     # Set prefix nodes
#     if k > 0:
#         tours[:, :k] = batch_prefix_nodes
#         # Mark prefix nodes as visited
#         # Need to use scatter_ along dim 1 for visited_mask
#         # Create indices for scatter: (B, k)
#         # Create src for scatter: True values (B, k)
#         visited_mask.scatter_(1, batch_prefix_nodes, True)

#     # current_nodes are the last nodes in the (partially built) tours
#     # If k=0, we need a starting node policy. Let's default to 0.
#     current_nodes = torch.zeros(B, dtype=torch.long, device=device)
#     if k > 0 :
#         current_nodes = batch_prefix_nodes[:, -1] # Last node of prefix
#     else: # No prefix, start all tours at node 0
#         if N > 0:
#             tours[:, 0] = 0
#             visited_mask[:, 0] = True
#             current_nodes[:] = 0 # current_nodes is already zeros(B)
#         else: # num_nodes is 0
#             return tours # Returns (B,0) tensor of -1s, or handle as error

#     num_filled_per_tour = torch.full((B,), k if k > 0 else (1 if N > 0 else 0) , dtype=torch.long, device=device)
    
#     # Active mask for batches that are not yet complete
#     active_batches = torch.ones(B, dtype=torch.bool, device=device)
#     if N == 0 : active_batches[:] = False


#     for step in range(k if k > 0 else (1 if N > 0 else 0), N): # Iterate to fill up to N nodes
#         if not active_batches.any(): # All tours in batch are complete
#             break

#         # For active batches, find the next node
#         # Gather probabilities from current_nodes: (B_active, N)
#         # Need to index adj_matrices_probs for active batches and their current_nodes
        
#         active_indices = torch.where(active_batches)[0]
#         if len(active_indices) == 0: break

#         current_adj_probs = adj_matrices_probs[active_indices] # (B_active, N, N)
#         current_step_nodes = current_nodes[active_indices]     # (B_active)
        
#         # Get probabilities from current_node to all others (and vice-versa for symmetry)
#         # Probs from current: current_adj_probs[batch_idx, current_node_val, next_node_candidate]
#         # Probs to current:   current_adj_probs[batch_idx, next_node_candidate, current_node_val]
        
#         # Use advanced indexing to get probabilities for each active batch from its current node
#         # B_active = len(active_indices)
#         # batch_arange = torch.arange(B_active, device=device)
#         # probs_from = current_adj_probs[batch_arange, current_step_nodes, :] # (B_active, N)
#         # probs_to   = current_adj_probs[batch_arange, :, current_step_nodes] # (B_active, N)
#         # symmetrized_probs = (probs_from + probs_to) / 2.0

#         # Alternative way to gather, perhaps simpler for this specific case:
#         symmetrized_probs_list = []
#         for i_active, original_batch_idx in enumerate(active_indices):
#             node_val = current_nodes[original_batch_idx].item()
#             p_from = adj_matrices_probs[original_batch_idx, node_val, :]
#             p_to = adj_matrices_probs[original_batch_idx, :, node_val]
#             symmetrized_probs_list.append((p_from + p_to) / 2.0)
#         if not symmetrized_probs_list: break # Should not happen if active_indices is not empty
#         symmetrized_probs = torch.stack(symmetrized_probs_list) # (B_active, N)


#         # Apply visited mask (for active batches)
#         # Set probabilities of already visited nodes to a very low value
#         symmetrized_probs[visited_mask[active_indices]] = -float('inf')

#         # Select best next node for each active tour
#         best_next_probs, best_next_nodes = torch.max(symmetrized_probs, dim=1) # (B_active)
        
#         # Update tours, visited_mask, current_nodes, and num_filled for active batches
#         # Check for failures (no valid next node, prob is -inf)
#         valid_next_node_mask_active = best_next_probs > -float('inf')
        
#         # Update only if a valid next node was found
#         active_indices_with_valid_next = active_indices[valid_next_node_mask_active]
#         nodes_to_add = best_next_nodes[valid_next_node_mask_active]
        
#         if len(active_indices_with_valid_next) > 0:
#             # Add to tour: tours[batch_idx, step_idx] = node_to_add
#             # We need the correct step_idx for each tour in active_indices_with_valid_next
#             current_fill_counts = num_filled_per_tour[active_indices_with_valid_next]
#             tours[active_indices_with_valid_next, current_fill_counts] = nodes_to_add
            
#             # Update visited_mask
#             # visited_mask[batch_idx, node_to_add] = True
#             visited_mask.scatter_(1, nodes_to_add.unsqueeze(1), True) # More robust for non-contiguous indices

#             # Update current_nodes
#             current_nodes[active_indices_with_valid_next] = nodes_to_add
            
#             # Increment num_filled
#             num_filled_per_tour[active_indices_with_valid_next] += 1

#         # Update active_batches: a tour becomes inactive if it's full or failed
#         failed_construction_mask_active = ~valid_next_node_mask_active
#         active_batches[active_indices[failed_construction_mask_active]] = False # Mark failed as inactive
#         active_batches[active_indices_with_valid_next[num_filled_per_tour[active_indices_with_valid_next] == N]] = False # Mark completed as inactive

#     # Check for tours that didn't complete to N nodes (failures)
#     # They will have -1 in some positions.
#     # For simplicity, we return as is. Caller can check for -1.
#     return tours

import torch
from collections import defaultdict

def construct_tour_from_edges(edge_list, num_nodes, start_node=0):
    """
    Given a list of edges representing a valid tour, construct the node sequence.
    """
    if not edge_list or len(edge_list) < num_nodes -1:
        return []
    
    adj = defaultdict(list)
    for u, v in edge_list:
        adj[u].append(v)
        adj[v].append(u)
        
    # Find a starting node, preferably one from the prefix if available
    if start_node not in adj:
        # Fallback if start_node is isolated
        start_node = next(iter(adj)) if adj else 0

    tour = [start_node]
    prev_node = -1
    curr_node = start_node
    
    # Using a set for faster checking of visited nodes
    visited_nodes = {start_node}
    
    while len(tour) < num_nodes:
        neighbors = adj.get(curr_node, [])
        next_node_found = False
        for neighbor in neighbors:
            if neighbor != prev_node:
                next_node = neighbor
                next_node_found = True
                break
        
        if not next_node_found or next_node in visited_nodes:
             # This indicates a problem, like a sub-tour or dead end.
             return [] 
            
        tour.append(next_node)
        visited_nodes.add(next_node)
        prev_node = curr_node
        curr_node = next_node
        
    return tour


def decode_adj_matrices_to_tours_batch(adj_matrices_probs, instance_locs, batch_prefix_nodes):
    """
    FINAL & CORRECTED VERSION: Decodes heatmaps using the edge-based greedy strategy from DIFUSCO,
    while rigorously enforcing the prefix constraint required by the hybrid solver.
    This replaces the old node-based greedy decoder.
    """
    B, N, _ = adj_matrices_probs.shape
    device = adj_matrices_probs.device
    
    # Symmetrize the probability matrix and calculate edge scores
    adj_probs = (adj_matrices_probs + adj_matrices_probs.transpose(1, 2)) / 2.0
    dists = torch.cdist(instance_locs, instance_locs, p=2) + 1e-6
    edge_scores = adj_probs / dists
    
    # Flatten scores and sort all possible edges (upper triangle)
    indices = torch.triu_indices(N, N, offset=1, device=device)
    flat_scores = edge_scores[:, indices[0], indices[1]]
    sorted_scores, sorted_indices = torch.sort(flat_scores, dim=1, descending=True)
    
    # Get the edge coordinates (u, v) for all sorted edges
    sorted_edges_u = indices[0][sorted_indices]
    sorted_edges_v = indices[1][sorted_indices]

    final_tours = torch.full((B, N), -1, dtype=torch.long, device=device)

    # --- Batch-wise Greedy Construction ---
    for i in range(B):
        # Union-Find data structure for cycle detection
        parent = torch.arange(N, device=device)
        def find_set(v):
            if v == parent[v]: return v
            parent[v] = find_set(parent[v])
            return parent[v]
        def unite_sets(a, b):
            a, b = find_set(a), find_set(b)
            if a != b: parent[b] = a

        node_degrees = torch.zeros(N, dtype=torch.int, device=device)
        edges_in_tour = []
        is_prefix_edge = torch.zeros(N, N, dtype=torch.bool, device=device)
        
        # === 1. ENFORCE PREFIX CONSTRAINT ===
        prefix_nodes = batch_prefix_nodes[i]
        # Handle potential padding in prefix_nodes if it comes from a collate_fn
        prefix_len = (prefix_nodes != -1).sum().item() 
        prefix_nodes = prefix_nodes[:prefix_len]
        
        if prefix_len > 1:
            for j in range(prefix_len - 1):
                u, v = prefix_nodes[j].item(), prefix_nodes[j+1].item()
                # Ensure u < v for consistency with is_prefix_edge matrix
                if u > v: u, v = v, u
                
                edges_in_tour.append((u, v))
                node_degrees[u] += 1
                node_degrees[v] += 1
                unite_sets(u, v)
                is_prefix_edge[u, v] = True
        # ====================================

        # === 2. GREEDY EDGE INSERTION for remaining edges ===
        num_edges_to_add = N - len(edges_in_tour)
        edges_added_count = 0

        for u_tensor, v_tensor in zip(sorted_edges_u[i], sorted_edges_v[i]):
            u, v = u_tensor.item(), v_tensor.item()
            
            # Skip if it's already a prefix edge
            if is_prefix_edge[u, v]:
                continue
            
            # Check conditions: no degree > 2 and no cycles
            if node_degrees[u] < 2 and node_degrees[v] < 2 and find_set(u) != find_set(v):
                edges_in_tour.append((u, v))
                node_degrees[u] += 1
                node_degrees[v] += 1
                unite_sets(u, v)
                edges_added_count += 1
                if edges_added_count == num_edges_to_add:
                    break
        # =======================================================
        
        # === 3. FINALIZE AND CONSTRUCT TOUR ===
        # The logic from cython_merge to close the tour if N-1 edges are found is implicitly handled
        # by the greedy search. If N-1 edges are added, the last two degree-1 nodes form the last available valid edge.

        if len(edges_in_tour) == N:
            start_node = prefix_nodes[0].item() if prefix_len > 0 else 0
            tour_sequence = construct_tour_from_edges(edges_in_tour, N, start_node=start_node)
            if tour_sequence and len(tour_sequence) == N:
                final_tours[i] = torch.tensor(tour_sequence, device=device)
        # =====================================
                
    return final_tours



def visualize_tsp_tour(instance_locs, tour_indices, title="TSP Tour", ax=None, gt_tour_indices=None):
    """
    Visualizes a TSP tour. (Assumed to be mostly CPU-based for plotting)
    instance_locs: (N, 2) tensor of city coordinates (CPU).
    tour_indices: (N) tensor or list of city indices in tour order (CPU).
    gt_tour_indices: (Optional N) tensor or list of ground truth tour for comparison (CPU).
    """
    if ax is None:
        fig, ax = plt.subplots()

    if isinstance(instance_locs, torch.Tensor):
        instance_locs = instance_locs.cpu()
    if isinstance(tour_indices, torch.Tensor):
        tour_indices = tour_indices.cpu()
    if isinstance(gt_tour_indices, torch.Tensor):
        gt_tour_indices = gt_tour_indices.cpu()
        
    if isinstance(tour_indices, list):
        tour_indices = torch.tensor(tour_indices)
    
    # Filter out -1s from incomplete tours for visualization
    valid_tour_indices = tour_indices[tour_indices != -1]
    if len(valid_tour_indices) == 0:
        print(f"Warning: No valid tour to visualize for '{title}'")
        ax.scatter(instance_locs[:, 0], instance_locs[:, 1], color='blue', s=50, zorder=2, label="Cities (No tour)")
        ax.set_title(title + " (No Valid Tour)")
        return

    valid_tour_indices = valid_tour_indices.long()
    
    ax.scatter(instance_locs[:, 0], instance_locs[:, 1], color='blue', s=50, zorder=2, label="Cities")
    for i in range(instance_locs.size(0)):
        ax.text(instance_locs[i, 0], instance_locs[i, 1], str(i), fontsize=8, zorder=3)

    tour_locs = instance_locs[valid_tour_indices]
    for i in range(len(tour_locs)):
        start_node = tour_locs[i]
        end_node = tour_locs[(i + 1) % len(tour_locs)] 
        ax.plot([start_node[0], end_node[0]], [start_node[1], end_node[1]], 'r-', lw=1.5, zorder=1, label="Generated Tour" if i == 0 else None)
    if len(tour_locs) > 0:
        ax.scatter(tour_locs[0,0], tour_locs[0,1], color='red', s=100, marker='x', zorder=4, label="Start/End")

    if gt_tour_indices is not None:
        if isinstance(gt_tour_indices, list):
            gt_tour_indices = torch.tensor(gt_tour_indices)
        gt_tour_indices = gt_tour_indices.long()
        gt_tour_locs = instance_locs[gt_tour_indices]
        for i in range(len(gt_tour_locs)):
            start_node = gt_tour_locs[i]
            end_node = gt_tour_locs[(i + 1) % len(gt_tour_locs)]
            ax.plot([start_node[0], end_node[0]], [start_node[1], end_node[1]], 'g--', lw=1, zorder=0.5, label="Ground Truth Tour" if i == 0 else None)

    ax.set_title(title)
    ax.set_xlabel("X-coordinate")
    ax.set_ylabel("Y-coordinate")
    ax.legend()
    ax.axis('equal')
