# Standalone evaluation of the sparse Prefix-DIFUSCO (no RL prefix) on TSP-1000.

import torch
import numpy as np
import os
import time
import argparse
import matplotlib.pyplot as plt
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from collections import defaultdict

# <<< MODIFIED: import the _sparse versions of the modules >>>
from data_loader_sparse import TSPConditionalSuffixDataset, custom_collate_fn
from diffusion_model_sparse import ConditionalTSPSuffixDiffusionModel
from discrete_diffusion_sparse import AdjacencyMatrixDiffusion
from tsp_utils import calculate_tsp_cost_batch, visualize_tsp_tour, apply_2opt_batch

# ==============================================================================
# === Helper functions for decoding and evaluation ===
# ==============================================================================
def visualize_heatmap(adj_probs, instance_locs, title="Adjacency Probability Heatmap", ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    N = instance_locs.shape[0]
    for i in range(N):
        for j in range(i + 1, N):
            prob = adj_probs[i, j].item()
            if prob > 0.01:
                ax.plot(
                    [instance_locs[i, 0], instance_locs[j, 0]],
                    [instance_locs[i, 1], instance_locs[j, 1]],
                    color='red', linewidth=2, alpha=prob**0.5, zorder=1
                )
    ax.scatter(instance_locs[:, 0], instance_locs[:, 1], color='blue', s=50, zorder=2)
    ax.set_title(title)
    ax.set_aspect('equal', adjustable='box')
    
def construct_tour_from_edges(edge_list, num_nodes, start_node=0):
    if not edge_list or len(edge_list) < num_nodes: return []
    adj = defaultdict(list)
    for u, v in edge_list:
        adj[u].append(v)
        adj[v].append(u)
    if start_node not in adj:
        start_node = next(iter(adj)) if adj else 0
    tour = [start_node]
    visited_nodes = {start_node}
    prev_node, curr_node = -1, start_node
    while len(tour) < num_nodes:
        neighbors = adj.get(curr_node, [])
        next_node_found = False
        for neighbor in neighbors:
            if neighbor != prev_node:
                next_node, next_node_found = neighbor, True; break
        if not next_node_found or next_node in visited_nodes: return []
        tour.append(next_node); visited_nodes.add(next_node)
        prev_node, curr_node = curr_node, next_node
    return tour

def decode_dm_heatmap_edge_greedy_batch(adj_matrices_probs, instance_locs, batch_prefix_nodes):
    B, N, _ = adj_matrices_probs.shape
    device = adj_matrices_probs.device
    adj_probs = (adj_matrices_probs + adj_matrices_probs.transpose(1, 2)) / 2.0
    dists = torch.cdist(instance_locs, instance_locs, p=2) + 1e-9
    edge_scores = adj_probs / dists
    indices = torch.triu_indices(N, N, offset=1, device=device)
    flat_scores = edge_scores[:, indices[0], indices[1]]
    _, sorted_indices = torch.sort(flat_scores, dim=1, descending=True)
    sorted_edges_u, sorted_edges_v = indices[0][sorted_indices], indices[1][sorted_indices]
    final_tours = torch.full((B, N), -1, dtype=torch.long, device=device)

    for i in range(B):
        parent = torch.arange(N, device=device)
        def find_set(v):
            if v == parent[v]: return v
            parent[v] = find_set(parent[v]); return parent[v]
        def unite_sets(a, b):
            a, b = find_set(a), find_set(b)
            if a != b: parent[b] = a
        node_degrees, edges_in_tour = torch.zeros(N, dtype=torch.int, device=device), []
        prefix_nodes = batch_prefix_nodes[i]
        prefix_len = (prefix_nodes != 0).sum().item() # Padding is 0
        prefix_nodes = prefix_nodes[:prefix_len]
        if prefix_len > 1:
            for j in range(prefix_len - 1):
                u, v = prefix_nodes[j].item(), prefix_nodes[j+1].item()
                edges_in_tour.append((u, v)); node_degrees[u] += 1; node_degrees[v] += 1; unite_sets(u, v)
        
        for u_tensor, v_tensor in zip(sorted_edges_u[i], sorted_edges_v[i]):
            if len(edges_in_tour) >= N - 1: break
            u, v = u_tensor.item(), v_tensor.item()
            is_prefix = False
            if prefix_len > 1:
                for j in range(prefix_len - 1):
                    p_u, p_v = prefix_nodes[j].item(), prefix_nodes[j+1].item()
                    if (u == p_u and v == p_v) or (u == p_v and v == p_u): is_prefix = True; break
            if is_prefix: continue
            if node_degrees[u] < 2 and node_degrees[v] < 2 and find_set(u) != find_set(v):
                edges_in_tour.append((u, v)); node_degrees[u] += 1; node_degrees[v] += 1; unite_sets(u, v)
        
        if len(edges_in_tour) == N - 1:
            endpoints = (node_degrees == 1).nonzero(as_tuple=True)[0]
            if len(endpoints) == 2: edges_in_tour.append((endpoints[0].item(), endpoints[1].item()))
        
        if len(edges_in_tour) == N:
            start_node = prefix_nodes[0].item() if prefix_len > 0 else 0
            tour_sequence = construct_tour_from_edges(edges_in_tour, N, start_node=start_node)
            if tour_sequence and len(tour_sequence) == N:
                final_tours[i] = torch.tensor(tour_sequence, device=device)
    return final_tours

def calculate_tsp_cost_batch(instance_locs_batch, tour_indices_batch):
    if tour_indices_batch.shape[1] < 2: return torch.zeros(tour_indices_batch.shape[0], device=instance_locs_batch.device)
    tour_locs_batch = torch.gather(instance_locs_batch, 1, tour_indices_batch.unsqueeze(-1).expand(-1, -1, 2))
    segment_lengths = torch.linalg.vector_norm(tour_locs_batch[:, :-1] - tour_locs_batch[:, 1:], dim=2)
    closing_segment_lengths = torch.linalg.vector_norm(tour_locs_batch[:, -1] - tour_locs_batch[:, 0], dim=1)
    return segment_lengths.sum(dim=1) + closing_segment_lengths

# def apply_2opt_batch(initial_tours_batch, instance_locs_batch, max_iterations=100):
#     device = instance_locs_batch.device
#     B, num_nodes, _ = instance_locs_batch.shape
#     if num_nodes < 4: return initial_tours_batch
#     best_tours_tensor = initial_tours_batch.clone().long()
#     best_costs = calculate_tsp_cost_batch(instance_locs_batch, best_tours_tensor)
#     current_tours_tensor = best_tours_tensor.clone()
#     for iter_count in range(max_iterations):
#         improved_in_pass = torch.zeros(B, dtype=torch.bool, device=device)
#         for i in range(num_nodes - 2):
#             for j in range(i + 2, num_nodes):
#                 node_i, node_ip1 = current_tours_tensor[:, i], current_tours_tensor[:, i + 1]
#                 node_j, node_jp1 = current_tours_tensor[:, j], current_tours_tensor[:, (j + 1) % num_nodes]
                
#                 cost_current = torch.linalg.vector_norm(instance_locs_batch.gather(1, node_i.view(B,1,1).expand(B,1,2)).squeeze(1) - instance_locs_batch.gather(1, node_ip1.view(B,1,1).expand(B,1,2)).squeeze(1), dim=1) + \
#                                torch.linalg.vector_norm(instance_locs_batch.gather(1, node_j.view(B,1,1).expand(B,1,2)).squeeze(1) - instance_locs_batch.gather(1, node_jp1.view(B,1,1).expand(B,1,2)).squeeze(1), dim=1)
#                 cost_new = torch.linalg.vector_norm(instance_locs_batch.gather(1, node_i.view(B,1,1).expand(B,1,2)).squeeze(1) - instance_locs_batch.gather(1, node_j.view(B,1,1).expand(B,1,2)).squeeze(1), dim=1) + \
#                            torch.linalg.vector_norm(instance_locs_batch.gather(1, node_ip1.view(B,1,1).expand(B,1,2)).squeeze(1) - instance_locs_batch.gather(1, node_jp1.view(B,1,1).expand(B,1,2)).squeeze(1), dim=1)

#                 improvement_mask = cost_new < cost_current
#                 if improvement_mask.any():
#                     temp_tours = current_tours_tensor.clone()
#                     segment = temp_tours[improvement_mask, i+1:j+1]
#                     temp_tours[improvement_mask, i+1:j+1] = torch.flip(segment, [1])
#                     new_costs = calculate_tsp_cost_batch(instance_locs_batch, temp_tours)
#                     final_update_mask = (new_costs < best_costs) & improvement_mask
#                     if final_update_mask.any():
#                         best_costs[final_update_mask] = new_costs[final_update_mask]
#                         best_tours_tensor[final_update_mask] = temp_tours[final_update_mask]
#                         improved_in_pass[final_update_mask] = True
#                         current_tours_tensor[final_update_mask] = temp_tours[final_update_mask]
#         if not improved_in_pass.any(): break
#     return best_tours_tensor

# <<< MODIFIED: fix a logic error in the parallel-sampling helper >>>
def expand_sparse_batch(batch_data, num_parallel_samples, device):
    """Duplicates a sparse batch for parallel sampling."""
    if num_parallel_samples == 1:
        return batch_data
    
    B = batch_data["prefix_lengths"].size(0)
    N = batch_data["num_nodes"]
    
    expanded_batch = {}
    
    # 1. Simply repeat the non-graph tensors
    for key in ["prefix_lengths"]:
        expanded_batch[key] = batch_data[key].repeat_interleave(num_parallel_samples, dim=0)
    
    # 2. Expand the graph structure
    total_nodes_in_original_batch = batch_data["instance_locs"].shape[0]

    # <<< CORRECTED LOGIC FOR edge_index >>>
    # The new logic is more direct: for each of the k parallel copies,
    # we add a fixed, increasing offset to every node index in the original edge_index.
    # offset = copy index * total number of nodes in the original batch
    edge_index_copies = []
    for i in range(num_parallel_samples):
        offset = i * total_nodes_in_original_batch
        edge_index_copies.append(batch_data["edge_index"] + offset)
    expanded_batch["edge_index"] = torch.cat(edge_index_copies, dim=1)
    # <<< END CORRECTION >>>

    # Expand node_to_graph_batch and add the offset
    graph_offsets = torch.arange(0, num_parallel_samples, device=device) * B
    expanded_batch["node_to_graph_batch"] = torch.cat([batch_data["node_to_graph_batch"] + offset for offset in graph_offsets])
    
    # 3. Expand flattened node/edge attributes
    # .repeat() copies the whole tensor num_parallel_samples times
    for key in ["instance_locs", "node_prefix_state", "dist_feature", "target_edge_attrs"]:
         if key in batch_data:
              expanded_batch[key] = batch_data[key].repeat(num_parallel_samples, 1)

    # 4. Expand tensors that need special handling (per-sample replication)
    expanded_batch["prefix_nodes"] = batch_data["prefix_nodes"].repeat_interleave(num_parallel_samples, dim=0)

    # 5. Copy non-tensor data
    for key in ["is_sparse", "num_nodes"]:
        expanded_batch[key] = batch_data[key]
        
    return expanded_batch



@torch.no_grad()
def evaluate(cfg: DictConfig, model_checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # <<< MODIFIED: read sparse_factor from the config >>>
    sparse_factor = cfg.model.get("sparse_factor", -1)
    if sparse_factor <= 0:
        raise ValueError("This evaluation script is for sparse models. 'sparse_factor' must be > 0 in the config.")

    # --- Data Loading ---
    prefix_k_to_eval = cfg.data.prefix_k
    dataset = TSPConditionalSuffixDataset(
        npz_file_path=cfg.data.test_path,
        prefix_k_options=[prefix_k_to_eval], 
        prefix_sampling_strategy='continuous_from_start',
        sparse_factor=sparse_factor # <<< MODIFIED: pass sparse_factor
    )
    num_samples_to_evaluate = min(cfg.eval.num_samples_to_eval, len(dataset))
    eval_dataset = torch.utils.data.Subset(dataset, range(num_samples_to_evaluate))
    dataloader = DataLoader(
        eval_dataset, batch_size=cfg.eval.batch_size,
        shuffle=False, collate_fn=custom_collate_fn 
    )

    # --- Load Model ---
    model = ConditionalTSPSuffixDiffusionModel(
        num_nodes=cfg.model.num_nodes, node_coord_dim=cfg.model.node_coord_dim,
        pos_embed_num_feats=cfg.model.pos_embed_num_feats, node_embed_dim=cfg.model.node_embed_dim,
        prefix_node_embed_dim=cfg.model.node_embed_dim,
        prefix_enc_hidden_dim=cfg.model.prefix_enc_hidden_dim, prefix_cond_dim=cfg.model.prefix_cond_dim,
        gnn_n_layers=cfg.model.gnn_n_layers, gnn_hidden_dim=cfg.model.gnn_hidden_dim,
        gnn_aggregation=cfg.model.gnn_aggregation, gnn_norm=cfg.model.gnn_norm,
        gnn_learn_norm=cfg.model.gnn_learn_norm, gnn_gated=cfg.model.gnn_gated,
        time_embed_dim=cfg.model.time_embed_dim,
        sparse_factor=sparse_factor # <<< MODIFIED: pass sparse_factor
    ).to(device)
    model.load_state_dict(torch.load(model_checkpoint_path, map_location=device))
    model.eval()
    print(f"Loaded trained model from {model_checkpoint_path}")

    # --- Diffusion Handler ---
    diffusion_handler = AdjacencyMatrixDiffusion(
        num_nodes=cfg.model.num_nodes, num_timesteps=cfg.diffusion.num_timesteps,
        schedule_type=cfg.diffusion.schedule_type, device=device,
        sparse_factor=sparse_factor # <<< MODIFIED: pass sparse_factor
    )
    
    num_parallel_samples = cfg.eval.get("num_parallel_samples", 1)
    print(f"Running evaluation with {num_parallel_samples} parallel sample(s) per instance.")
    
    total_best_generated_cost_sum, total_gt_cost_sum, num_valid_instances_evaluated = 0.0, 0.0, 0
    start_time = time.time()
    
    for batch_idx, batch_data in enumerate(tqdm(dataloader, desc=f"Evaluating k={prefix_k_to_eval}")):
        # Move the original batch data to the device
        for k, v in batch_data.items():
            if torch.is_tensor(v):
                batch_data[k] = v.to(device)
        current_batch_size = batch_data["prefix_lengths"].shape[0]

        # <<< MODIFIED: use the new helper to expand the batch data >>>
        expanded_batch_data = expand_sparse_batch(batch_data, num_parallel_samples, device)
        
        # <<< MODIFIED: simplified inference call >>>
        _, generated_adj_matrices_probs, _,_,_ = diffusion_handler.p_sample_loop_ddim(
            denoiser_model=model, 
            batch_data=expanded_batch_data, 
            num_inference_steps=cfg.eval.num_inference_steps,
            schedule=cfg.eval.inference_schedule_type
        )
        
        # --- Decoding ---
        # Note: the sparse handler returns the reconstructed dense adjacency matrix, so the decoder is unchanged
        # However, the decoder must receive the correct (expanded) dense coordinates
        expanded_locs_dense = expanded_batch_data['instance_locs'].reshape(
            current_batch_size * num_parallel_samples, cfg.model.num_nodes, -1
        )
        decoded_tours_all_samples = decode_dm_heatmap_edge_greedy_batch(
            generated_adj_matrices_probs, expanded_locs_dense, expanded_batch_data["prefix_nodes"]
        )
# <<< MODIFIED: (optional) apply 2-opt >>>
        # Use .get() to safely read the config; defaults to False if unset
        if cfg.eval.get("apply_two_opt", False):
            valid_mask = (decoded_tours_all_samples != -1).all(dim=1)
            if valid_mask.any():
                # Inside the tqdm loop, use tqdm.write() instead of print() to avoid breaking the progress bar
                tqdm.write(f"  Applying 2-opt to {valid_mask.sum().item()} tours...")
                
                # Number of 2-opt iterations, default 400
                max_iter = cfg.eval.get("two_opt_max_iterations", 400)
                
                # Note: expanded_locs_dense here corresponds to expanded_locs in the TSP-500 script
                decoded_tours_all_samples[valid_mask] = apply_2opt_batch(
                    decoded_tours_all_samples[valid_mask], 
                    expanded_locs_dense[valid_mask],
                    max_iterations=max_iter)

        # --- Compute costs ---
        all_costs = torch.full((current_batch_size * num_parallel_samples,), float('inf'), device=device)
        valid_mask_all = (decoded_tours_all_samples != -1).all(dim=1)
        if valid_mask_all.any():
            all_costs[valid_mask_all] = calculate_tsp_cost_batch(
                expanded_locs_dense[valid_mask_all], decoded_tours_all_samples[valid_mask_all]
            )

        # Best-of-N logic
        costs_reshaped = all_costs.view(current_batch_size, num_parallel_samples)
        best_costs, _ = torch.min(costs_reshaped, dim=1)
        
        # Compute GT costs and accumulate results
        instance_has_valid_solution = ~torch.isinf(best_costs)
        if instance_has_valid_solution.any():
            valid_locs = batch_data["instance_locs"].reshape(current_batch_size, cfg.model.num_nodes, -1)[instance_has_valid_solution]
            gt_tours = torch.arange(cfg.model.num_nodes, device=device).unsqueeze(0).repeat(valid_locs.shape[0], 1)
            costs_gt = calculate_tsp_cost_batch(valid_locs, gt_tours)
            
            total_best_generated_cost_sum += best_costs[instance_has_valid_solution].sum().item()
            total_gt_cost_sum += costs_gt.sum().item()
            num_valid_instances_evaluated += instance_has_valid_solution.sum().item()

    total_time = time.time() - start_time
    total_samples_processed = len(eval_dataset)
    avg_sample_time = total_time / total_samples_processed if total_samples_processed > 0 else 0

    print("\n--------- Timing Summary ---------")
    print(f"Total evaluation time: {total_time:.3f}s for {total_samples_processed} instances.")
    print(f"Average time per instance (including all samples): {avg_sample_time:.4f}s")
    
    avg_generated_cost = total_best_generated_cost_sum / num_valid_instances_evaluated if num_valid_instances_evaluated > 0 else float('inf')
    avg_gt_cost = total_gt_cost_sum / num_valid_instances_evaluated if num_valid_instances_evaluated > 0 else float('inf')

    print("\n---------Diffusion Model Evaluation Summary ---------")
    print(f"Number of instances evaluated: {total_samples_processed}")
    print(f"Number of instances with at least one valid tour: {num_valid_instances_evaluated}")
    
    if num_valid_instances_evaluated > 0:
        optimality_gap = ((avg_generated_cost / avg_gt_cost) - 1) * 100 if avg_gt_cost > 0 else float('inf')
        print(f"Average Best-of-{num_parallel_samples} Generated Tour Cost: {avg_generated_cost:.4f}")
        print(f"Average Ground Truth Tour Cost: {avg_gt_cost:.4f}")
        print(f"Optimality Gap: {optimality_gap:.2f}%")
    else:
        print("No valid tours were successfully decoded.")    

if __name__ == "__main__":
    try:
        # <<< MODIFIED: make sure to load a config that contains sparse_factor >>>
        config_path = "configs/train_tsp1000.yaml" # i.e., the config used to train the sparse model
        cfg = OmegaConf.load(config_path)
        print(f"Loaded base configuration from: {config_path}")
    except FileNotFoundError:
        print(f"ERROR: Base config '{config_path}' not found.")
        exit()

    # Override evaluation-specific parameters
    eval_cfg_overrides = OmegaConf.create({
        'data': {
            'test_path':"./data/tspn1000_sol_valid3000_s7991_solver_concorde.npz", # Make sure the test path is correct
            'prefix_k': 0, # Test generation from scratch (k=0)
        },
        'eval': {
            'batch_size': 1, # Sparse models usually need a smaller batch size
            'num_samples_to_eval': 12,
            'num_parallel_samples': 1, # Number of parallel samples
            'num_inference_steps': 50,
            'inference_schedule_type': 'cosine',
            # <<< MODIFIED: add 2-opt config >>>
            'apply_two_opt': False, # Set to True to enable 2-opt
            'two_opt_max_iterations': 400 # Max number of 2-opt iterations
        }
    })
    final_cfg = OmegaConf.merge(cfg, eval_cfg_overrides)
    
    print("\n--- Final Evaluation Configuration ---")
    print(OmegaConf.to_yaml(final_cfg))

    # <<< MODIFIED: make sure the path points to the sparse model checkpoint >>>
    trained_model_checkpoint = "./checkpoints/prefix_difusco/tsp1000/stage4_k1_100_last_epoch_15.pth"
    if not os.path.exists(trained_model_checkpoint):
        raise FileNotFoundError(f"Model checkpoint not found at: {trained_model_checkpoint}")

    evaluate(final_cfg, trained_model_checkpoint)