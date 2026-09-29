# Standalone evaluation of Prefix-DIFUSCO (no RL prefix) on TSP-500.

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

# --- Make sure these imports point to your latest, corrected files ---
from data_loader import TSPConditionalSuffixDataset, custom_collate_fn
from diffusion_model import ConditionalTSPSuffixDiffusionModel
from discrete_diffusion import AdjacencyMatrixDiffusion
from tsp_utils import calculate_tsp_cost_batch, visualize_tsp_tour, apply_2opt_batch

# ==============================================================================
# === FINAL & CORRECT DECODING HELPERS (Based on DIFUSCO's logic) ===
# ==============================================================================
def visualize_heatmap(adj_probs, instance_locs, title="Adjacency Probability Heatmap", ax=None):
    """
    Visualizes the adjacency probability matrix as a heatmap on the node coordinates.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    N = instance_locs.shape[0]
    # Draw edges with opacity proportional to their probability
    for i in range(N):
        for j in range(i + 1, N):
            prob = adj_probs[i, j].item()
            if prob > 0.01: # Only draw edges with a minimum probability
                ax.plot(
                    [instance_locs[i, 0], instance_locs[j, 0]],
                    [instance_locs[i, 1], instance_locs[j, 1]],
                    color='red',
                    linewidth=2,
                    alpha=prob**0.5, # Use alpha to represent probability
                    zorder=1
                )
    
    # Draw nodes
    ax.scatter(instance_locs[:, 0], instance_locs[:, 1], color='blue', s=50, zorder=2)
    ax.set_title(title)
    ax.set_aspect('equal', adjustable='box')
    
def construct_tour_from_edges(edge_list, num_nodes, start_node=0):
    """
    Given a list of edges representing a valid tour, construct the node sequence.
    This version is made more robust.
    """
    if not edge_list or len(edge_list) < num_nodes:
        return []
    
    adj = defaultdict(list)
    for u, v in edge_list:
        adj[u].append(v)
        adj[v].append(u)
        
    # Check if start_node is valid, otherwise pick a new one
    if start_node not in adj:
        start_node = next(iter(adj)) if adj else 0

    tour = [start_node]
    visited_nodes = {start_node}
    prev_node = -1
    curr_node = start_node
    
    while len(tour) < num_nodes:
        neighbors = adj.get(curr_node, [])
        next_node_found = False
        for neighbor in neighbors:
            if neighbor != prev_node:
                next_node = neighbor
                next_node_found = True
                break
        
        # If we get stuck or form a premature cycle, the edge set was invalid.
        if not next_node_found or next_node in visited_nodes:
             return [] 
            
        tour.append(next_node)
        visited_nodes.add(next_node)
        prev_node = curr_node
        curr_node = next_node
        
    return tour


def decode_dm_heatmap_edge_greedy_batch(adj_matrices_probs, instance_locs, batch_prefix_nodes):
    """
    FINAL & CORRECTED VERSION: Decodes heatmaps using the exact two-stage logic from DIFUSCO:
    1. Greedily build a spanning path of N-1 edges.
    2. Deterministically add the final edge to close the tour.
    This version also correctly handles the prefix constraint.
    """
    B, N, _ = adj_matrices_probs.shape
    device = adj_matrices_probs.device
    
    adj_probs = (adj_matrices_probs + adj_matrices_probs.transpose(1, 2)) / 2.0
    dists = torch.cdist(instance_locs, instance_locs, p=2) + 1e-9
    edge_scores = adj_probs / dists
    
    indices = torch.triu_indices(N, N, offset=1, device=device)
    flat_scores = edge_scores[:, indices[0], indices[1]]
    _, sorted_indices = torch.sort(flat_scores, dim=1, descending=True)
    
    sorted_edges_u = indices[0][sorted_indices]
    sorted_edges_v = indices[1][sorted_indices]

    final_tours = torch.full((B, N), -1, dtype=torch.long, device=device)

    for i in range(B):
        # Union-Find data structure for cycle detection
        parent = torch.arange(N, device=device)
        def find_set(v):
            if v == parent[v]: return v
            parent[v] = find_set(parent[v]); return parent[v]
        def unite_sets(a, b):
            a, b = find_set(a), find_set(b)
            if a != b: parent[b] = a

        node_degrees = torch.zeros(N, dtype=torch.int, device=device)
        edges_in_tour = []
        
        # === 1. ENFORCE PREFIX CONSTRAINT ===
        prefix_nodes = batch_prefix_nodes[i]
        prefix_len = (prefix_nodes != -1).sum().item()
        prefix_nodes = prefix_nodes[:prefix_len]
        
        if prefix_len > 1:
            for j in range(prefix_len - 1):
                u, v = prefix_nodes[j].item(), prefix_nodes[j+1].item()
                if node_degrees[u] >= 2 or node_degrees[v] >= 2: continue # Should not happen with valid prefix
                edges_in_tour.append((u, v))
                node_degrees[u] += 1
                node_degrees[v] += 1
                unite_sets(u, v)
        
        # === 2. GREEDILY BUILD A SPANNING PATH (N-1 total edges) ===
        for u_tensor, v_tensor in zip(sorted_edges_u[i], sorted_edges_v[i]):
            if len(edges_in_tour) >= N - 1:
                break
            
            u, v = u_tensor.item(), v_tensor.item()
            
            # Check conditions: not a prefix edge, no degree > 2, and no cycles
            is_prefix = False
            if prefix_len > 1:
                for j in range(prefix_len - 1):
                    p_u, p_v = prefix_nodes[j].item(), prefix_nodes[j+1].item()
                    if (u == p_u and v == p_v) or (u == p_v and v == p_u):
                        is_prefix = True; break
            if is_prefix: continue

            if node_degrees[u] < 2 and node_degrees[v] < 2 and find_set(u) != find_set(v):
                edges_in_tour.append((u, v))
                node_degrees[u] += 1
                node_degrees[v] += 1
                unite_sets(u, v)

        # === 3. DETERMINISTICALLY CLOSE THE TOUR ===
        if len(edges_in_tour) == N - 1:
            # Find the two nodes with degree 1 (the endpoints of the path)
            endpoints = (node_degrees == 1).nonzero(as_tuple=True)[0]
            if len(endpoints) == 2:
                u, v = endpoints[0].item(), endpoints[1].item()
                edges_in_tour.append((u, v))
        
        # === 4. CONSTRUCT FINAL TOUR SEQUENCE ===
        if len(edges_in_tour) == N:
            start_node = prefix_nodes[0].item() if prefix_len > 0 else 0
            tour_sequence = construct_tour_from_edges(edges_in_tour, N, start_node=start_node)
            if tour_sequence and len(tour_sequence) == N:
                final_tours[i] = torch.tensor(tour_sequence, device=device)
                
    return final_tours

# ==============================================================================
# === Other Helper Functions and Main Evaluation Logic (Unchanged) ===
# ==============================================================================
def calculate_tsp_cost_batch(instance_locs_batch, tour_indices_batch):
    if tour_indices_batch.shape[1] < 2:
        return torch.zeros(tour_indices_batch.shape[0], device=instance_locs_batch.device)
    tour_locs_batch = torch.gather(instance_locs_batch, 1, tour_indices_batch.unsqueeze(-1).expand(-1, -1, 2))
    segment_lengths = torch.sqrt(((tour_locs_batch[:, :-1] - tour_locs_batch[:, 1:])**2).sum(dim=2))
    closing_segment_lengths = torch.sqrt((((tour_locs_batch[:, -1] - tour_locs_batch[:, 0])**2)).sum(dim=1))
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

def visualize_tsp_tour(instance_locs, tour_indices, title="TSP Tour", ax=None, gt_tour_indices=None):
    if ax is None: fig, ax = plt.subplots()
    locs_cpu, tour_cpu = instance_locs.cpu(), tour_indices.cpu()
    valid_tour_indices = tour_cpu[tour_cpu != -1]
    if len(valid_tour_indices) == 0:
        ax.set_title(title + " (No Valid Tour)"); ax.scatter(locs_cpu[:, 0], locs_cpu[:, 1], color='red'); return
    ax.scatter(locs_cpu[:, 0], locs_cpu[:, 1], color='blue', zorder=2)
    for i, txt in enumerate(range(locs_cpu.shape[0])): ax.annotate(txt, (locs_cpu[i, 0], locs_cpu[i, 1]))
    tour_locs = locs_cpu[valid_tour_indices.long()]
    tour_locs = torch.cat([tour_locs, tour_locs[0:1]], dim=0)
    ax.plot(tour_locs[:, 0], tour_locs[:, 1], 'r-', zorder=1)
    if gt_tour_indices is not None:
        gt_locs = locs_cpu[gt_tour_indices.cpu().long()]
        gt_locs = torch.cat([gt_locs, gt_locs[0:1]], dim=0)
        ax.plot(gt_locs[:, 0], gt_locs[:, 1], 'g--', zorder=0.5, label="Ground Truth")
    ax.set_title(title); ax.legend()


@torch.no_grad()
def evaluate(cfg: DictConfig, model_checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading ---
    prefix_k_to_eval = cfg.data.prefix_k
    dataset = TSPConditionalSuffixDataset(
        npz_file_path=cfg.data.test_path,
        prefix_k_options=[prefix_k_to_eval], 
        prefix_sampling_strategy='continuous_from_start'
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
        time_embed_dim=cfg.model.time_embed_dim
    ).to(device)
    model.load_state_dict(torch.load(model_checkpoint_path, map_location=device))
    model.eval()
    print(f"Loaded trained model from {model_checkpoint_path}")

    # --- Diffusion Handler ---
    diffusion_handler = AdjacencyMatrixDiffusion(
        num_nodes=cfg.model.num_nodes, num_timesteps=cfg.diffusion.num_timesteps,
        schedule_type=cfg.diffusion.schedule_type, device=device
    )
    
    # total_generated_cost_sum, total_gt_cost_sum, num_valid_tours_generated = 0.0, 0.0, 0
    # start_time = time.time()

    # === [Key change 1]: get the number of parallel samples ===
    num_parallel_samples = cfg.eval.get("num_parallel_samples",1)
    print(f"Running evaluation with {num_parallel_samples} parallel sample(s) per instance.")
    
    total_best_generated_cost_sum, total_gt_cost_sum, num_valid_instances_evaluated = 0.0, 0.0, 0
    start_time = time.time()

    
    # --- Evaluation Loop ---
    for batch_idx, batch_data in enumerate(tqdm(dataloader, desc=f"Evaluating k={prefix_k_to_eval}")):
        instance_locs_batch = batch_data["instance_locs"].to(device)
        prefix_nodes_batch = batch_data["prefix_nodes"].to(device)
        prefix_lengths_batch = batch_data["prefix_lengths"].to(device)
        node_prefix_state_batch = batch_data["node_prefix_state"].to(device)
        current_batch_size = instance_locs_batch.shape[0]

        # === [Key change 2]: prepare expanded tensors for parallel sampling ===
        # Replicate every instance num_parallel_samples times
        expanded_locs = instance_locs_batch.repeat_interleave(num_parallel_samples, dim=0)
        expanded_prefix_nodes = prefix_nodes_batch.repeat_interleave(num_parallel_samples, dim=0)
        expanded_prefix_lengths = prefix_lengths_batch.repeat_interleave(num_parallel_samples, dim=0)
        expanded_node_prefix_state = node_prefix_state_batch.repeat_interleave(num_parallel_samples, dim=0)
        
        # --- Run one large batched denoising pass ---
        _, generated_adj_matrices_probs,_ = diffusion_handler.p_sample_loop_ddim(
            denoiser_model=model, instance_locs=expanded_locs,
            prefix_nodes=expanded_prefix_nodes, prefix_lengths=expanded_prefix_lengths,
            node_prefix_state=expanded_node_prefix_state, num_inference_steps=cfg.eval.num_inference_steps,
            schedule=cfg.eval.inference_schedule_type
        )
        
        # --- Decode all generated tours ---
        decoded_tours_all_samples = decode_dm_heatmap_edge_greedy_batch(
            generated_adj_matrices_probs, expanded_locs, expanded_prefix_nodes
        )

        # (Optional) apply 2-opt
        if cfg.eval.apply_two_opt:
            valid_mask = (decoded_tours_all_samples != -1).all(dim=1)
            if valid_mask.any():
                print(f"  Applying 2-opt to {valid_mask.sum().item()} tours...")
                decoded_tours_all_samples[valid_mask] = apply_2opt_batch(
                    decoded_tours_all_samples[valid_mask], expanded_locs[valid_mask],
                    max_iterations=cfg.eval.two_opt_max_iterations
                )
        
        # --- Compute the cost of every generated tour ---
        all_costs = torch.full((current_batch_size * num_parallel_samples,), float('inf'), device=device)
        valid_mask_all = (decoded_tours_all_samples != -1).all(dim=1)
        if valid_mask_all.any():
            all_costs[valid_mask_all] = calculate_tsp_cost_batch(
                expanded_locs[valid_mask_all], decoded_tours_all_samples[valid_mask_all]
            )

        # === [Key change 3]: best-of-N logic ===
        # Reshape costs to (batch_size, num_parallel_samples)
        costs_reshaped = all_costs.view(current_batch_size, num_parallel_samples)
        
        # Find the best cost of each instance and its index
        best_costs, best_indices = torch.min(costs_reshaped, dim=1)
        
        # --- Compute GT costs and accumulate results ---
        # Only compute GT cost and gap for instances that have at least one valid solution
        instance_has_valid_solution = ~torch.isinf(best_costs)
        
        if instance_has_valid_solution.any():
            valid_locs = instance_locs_batch[instance_has_valid_solution]
            
            # Ground Truth Tour (ordered)
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
    # --- Use your tsp100_config.yaml or a dedicated eval config ---
    # Loading the main config and overriding a few values for evaluation
    try:
        config_path = "configs/train_tsp500.yaml"
        cfg = OmegaConf.load(config_path)
        print(f"Loaded base configuration from: {config_path}")
    except FileNotFoundError:
        print(f"ERROR: Base config '{config_path}' not found. Using a default.")
        cfg = OmegaConf.create({}) # Start with empty and merge defaults

    # Default eval config, can be overridden by loaded file
    default_eval_cfg = OmegaConf.create({
        'data': {
            # Set this to 0 to test generation from scratch
            'prefix_k': 0, 
        },
        'eval': {
            'batch_size': 1,
            'num_samples_to_eval': 256,
            'num_samples_to_visualize': 5,
            'num_inference_steps': 50, # More steps can improve quality
            'inference_schedule_type': 'polynomial',
            'apply_two_opt': False,
            'two_opt_max_iterations': 1000
        }
    })

    # Merge the default evaluation config with the loaded base config
    # This ensures model/diffusion params are from the training config,
    # but data/eval params are set for this script.
    final_cfg = OmegaConf.merge(cfg, default_eval_cfg)
    
    print("\n--- Final Evaluation Configuration ---")
    print(OmegaConf.to_yaml(final_cfg))

    # Path to the model checkpoint you want to evaluate

    trained_model_checkpoint =  "./checkpoints/prefix_difusco/tsp500/stage1_k0_20_epoch_20.pth" #./ckpt_tsp_difusco_style_new_prefix_new_new/stage5_k1_20_last_epoch_5.pth"
    if not os.path.exists(trained_model_checkpoint):
        raise FileNotFoundError(f"Model checkpoint not found at: {trained_model_checkpoint}")

    evaluate(final_cfg, trained_model_checkpoint)
