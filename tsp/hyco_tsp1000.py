# HyCO on TSP-1000 (RL backbone: POMO, sparse Prefix-DIFUSCO).
# Usage: python hyco_tsp1000.py --config configs/hyco_tsp1000_pomo.yaml
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import importlib
import argparse
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader
from tensordict import TensorDict
import inspect
from collections import defaultdict

# --- RL4CO Imports ---
from rl4co.envs import get_env
from rl4co.utils.ops import unbatchify

# <<< MODIFIED: Import sparse versions of the Diffusion Model components >>>
from diffusion_model_sparse import ConditionalTSPSuffixDiffusionModel
from discrete_diffusion_sparse import AdjacencyMatrixDiffusion

# --- Helper Function Imports ---
# (Helper functions for cost calculation and 2-opt are assumed to be in a separate file or included here)
from tsp_utils import calculate_tsp_cost_batch, visualize_tsp_tour, apply_2opt_batch
import random

def set_seed(seed):
    """Set all relevant random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Improves determinism at a small cost in speed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
class HybridSolver:
    """
    Implements a theoretically-driven hybrid solving approach for large-scale TSP (e.g., TSP1000)
    using a sparse graph-based Diffusion Model.
    """
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Solver using device: {self.device}")

        self.rl_policy = self._load_rl_policy()
        self.dm_model = self._load_dm_model()
        
        self.diffusion_handler = AdjacencyMatrixDiffusion(
            num_nodes=cfg.model.num_nodes,
            num_timesteps=cfg.diffusion.num_timesteps,
            schedule_type=cfg.diffusion.schedule_type,
            device=self.device,
            sparse_factor=cfg.model.sparse_factor # <<< MODIFIED: Pass sparse_factor
        )

    def _load_rl_policy(self):
        print(f"Loading RL model from: {self.cfg.rl_model.ckpt_path}")
        try:
            ckpt = torch.load(self.cfg.rl_model.ckpt_path, map_location='cpu',weights_only=False)
            hparams = ckpt.get('hyper_parameters', ckpt.get('hparams')) 
            if hparams is None:
                raise ValueError("Could not find hyperparameters in checkpoint.")

            rl_model_cls = getattr(importlib.import_module("rl4co.models.zoo"), self.cfg.rl_model.name)
            valid_args = inspect.signature(rl_model_cls.__init__).parameters
            
            cleaned_hparams = {arg_name: hparams[arg_name] for arg_name in valid_args if arg_name in hparams}
            cleaned_hparams.pop('env', None)
            
            env = get_env(self.cfg.rl_model.problem, generator_params={"num_loc": self.cfg.model.num_nodes})
            model = rl_model_cls(env=env, **cleaned_hparams)
            
            model.load_state_dict(ckpt['state_dict'], strict=False)
            policy = model.policy.to(self.device)
            policy.eval()
            return policy
        except Exception as e:
            print(f"Error loading RL model: {e}")
            print("This might be due to a version mismatch in the checkpoint file or rl4co library.")
            exit()
            
    def _load_dm_model(self):
        print(f"Loading Diffusion model from: {self.cfg.dm_model.ckpt_path}")
        model = ConditionalTSPSuffixDiffusionModel(
            num_nodes=self.cfg.model.num_nodes, node_coord_dim=self.cfg.model.node_coord_dim,
            pos_embed_num_feats=self.cfg.model.pos_embed_num_feats, node_embed_dim=self.cfg.model.node_embed_dim,
            prefix_node_embed_dim=self.cfg.model.node_embed_dim,
            prefix_enc_hidden_dim=self.cfg.model.prefix_enc_hidden_dim, prefix_cond_dim=self.cfg.model.prefix_cond_dim,
            gnn_n_layers=self.cfg.model.gnn_n_layers, gnn_hidden_dim=self.cfg.model.gnn_hidden_dim,
            gnn_aggregation=self.cfg.model.gnn_aggregation, gnn_norm=self.cfg.model.gnn_norm,
            gnn_learn_norm=self.cfg.model.gnn_learn_norm, gnn_gated=self.cfg.model.gnn_gated,
            time_embed_dim=self.cfg.model.time_embed_dim,
            sparse_factor=self.cfg.model.sparse_factor # <<< MODIFIED: Pass sparse_factor
        ).to(self.device)
        model.load_state_dict(torch.load(self.cfg.dm_model.ckpt_path, map_location=self.device))
        model.eval()
        return model

 
# <<< NEW: Helper function to prepare sparse batch data for the Diffusion Model >>>
# Replace the _prepare_sparse_dm_batch function in your main solver script with this one.

    def _prepare_sparse_dm_batch(self, instance_locs, prefix_nodes, prefix_lengths):
        """
        Constructs the k-NN graph and formats the data into the sparse dictionary format
        required by the sparse diffusion model.
        """
        B, N, _ = instance_locs.shape
        device = self.device
        k = self.cfg.model.sparse_factor

        dists = torch.cdist(instance_locs, instance_locs, p=2)
        dists.view(B * N, N)[:, torch.arange(N)] += 1e9 # Prevent self-loops
        
        _, top_k_indices = torch.topk(dists, k=k, dim=-1, largest=False) # Shape: [B, N, k]
        
        # --- START OF FIX: Simplified and more robust index creation ---
        # Create offsets for each graph in the batch: [0, N, 2N, ...]
        node_offsets = (torch.arange(B, device=device) * N).view(B, 1, 1)

        # Create local row indices (0 to N-1) for a single graph
        local_rows = torch.arange(N, device=device).view(1, N, 1)
        
        # Add offsets to local row and column indices to get global indices
        # Broadcasting handles expanding the offsets correctly
        row_b = (local_rows + node_offsets).expand(B, N, k).reshape(-1)
        col_b = (top_k_indices + node_offsets).reshape(-1)
        # --- END OF FIX ---

        # Make graph symmetric
        # Make graph symmetric
        edge_index = torch.stack([
            torch.cat([row_b, col_b]),
            torch.cat([col_b, row_b])
        ], dim=0)

        flat_locs = instance_locs.view(B * N, -1)
        node_to_graph_batch = torch.arange(B, device=device).repeat_interleave(N)
        
        # --- START OF THE DEFINITIVE FIX ---
        # 1. Compute the distance feature of every edge
        dist_feat_unsorted = torch.linalg.norm(
            flat_locs[edge_index[0]] - flat_locs[edge_index[1]], dim=-1
        ).unsqueeze(1)
        
        # 2. Get the graph ID of every edge (from its source node)
        #    This was the root cause of the ordering problem; we now use it to fix the ordering
        edge_graph_ids = node_to_graph_batch[edge_index[0]]
        
        # 3. Compute a permutation that sorts edges by graph ID
        #    torch.sort returns the sorted values and the original indices; we need the indices.
        _, sorted_permutation = torch.sort(edge_graph_ids)
        
        # 4. Apply this permutation to reorder edge_index and dist_feat
        edge_index = edge_index[:, sorted_permutation]
        dist_feat = dist_feat_unsorted[sorted_permutation]
        # --- END OF THE DEFINITIVE FIX ---
        
        # ... (computation of node_prefix_state unchanged) ...
        node_prefix_state = torch.zeros(B * N, 1, device=device)
        max_len = prefix_lengths.max().item()
        if max_len > 0:
            prefixes_for_scatter = prefix_nodes[:, :max_len].long()
            len_mask = torch.arange(max_len, device=device).unsqueeze(0) < prefix_lengths.unsqueeze(1)
            
            valid_nodes = prefixes_for_scatter[len_mask]
            batch_indices = torch.arange(B, device=device).unsqueeze(1).expand_as(prefixes_for_scatter)[len_mask]
            
            flat_indices = valid_nodes + batch_indices * N
            node_prefix_state[flat_indices] = 1.0

        return {
            "instance_locs": flat_locs,
            "prefix_nodes": prefix_nodes,
            "prefix_lengths": prefix_lengths,
            "edge_index": edge_index, # now sorted
            "dist_feature": dist_feat, # now sorted
            "node_to_graph_batch": node_to_graph_batch,
            "node_prefix_state": node_prefix_state,
            "num_nodes": N,
            "is_sparse": True
        }

    def _compute_dm_prior_scores(self, instance_locs, candidate_prefixes, prefix_lengths):
        """
        Computes a single-step denoising score for candidate prefixes using the sparse DM.
        """
        total_candidates = candidate_prefixes.shape[0]
        device = self.device
        if total_candidates == 0: return torch.empty(0, device=device)

        # Prepare sparse batch data
        dm_batch = self._prepare_sparse_dm_batch(instance_locs, candidate_prefixes, prefix_lengths)
        N = dm_batch['num_nodes']

        t_probe = torch.full((total_candidates,), self.cfg.solver.dm_probe_timestep, device=device, dtype=torch.long)
        
        prefix_adj_target = torch.zeros(total_candidates, N, N, device=device, dtype=torch.float)
        for i in range(total_candidates):
            if prefix_lengths[i] > 1:
                p_nodes = candidate_prefixes[i, :prefix_lengths[i]]
                prefix_adj_target[i, p_nodes[:-1], p_nodes[1:]] = 1.0
                prefix_adj_target[i, p_nodes[1:], p_nodes[:-1]] = 1.0
        
        batch_offsets = dm_batch['node_to_graph_batch'][dm_batch['edge_index'][0]]
        row_local = dm_batch['edge_index'][0] % N
        col_local = dm_batch['edge_index'][1] % N
        target_edge_attrs = prefix_adj_target[batch_offsets, row_local, col_local]
        
        edge_batch_indices = dm_batch['node_to_graph_batch'][dm_batch['edge_index'][0]]
        t_for_edges = t_probe[edge_batch_indices]
        
        x_t_noisy_attrs, _ = self.diffusion_handler.q_sample(target_edge_attrs.unsqueeze(1), t_for_edges)
        x_t_transformed = x_t_noisy_attrs.float() * 2.0 - 1.0
        
        predicted_x_0_logits_attrs = self.dm_model(
            noisy_data=x_t_transformed,
            t_scalar=t_for_edges.float(),
            batch_data=dm_batch
        )
        
        loss_mask = target_edge_attrs > 0
        if not loss_mask.any():
            return torch.full((total_candidates,), 999.0, device=device)
        
        # --- START OF FIX ---
        # The model output `predicted_x_0_logits_attrs` is already 1D, so we remove the unnecessary .squeeze(1)
        reconstruction_loss = F.binary_cross_entropy_with_logits(
            predicted_x_0_logits_attrs[loss_mask],
            target_edge_attrs[loss_mask],
            reduction='none'
        )
        # --- END OF FIX ---
        
        edge_to_graph_map = dm_batch['node_to_graph_batch'][dm_batch['edge_index'][0]]
        total_loss_per_candidate = torch.zeros(total_candidates, device=device)
        total_loss_per_candidate.scatter_add_(0, edge_to_graph_map[loss_mask], reconstruction_loss)
        
        num_edges_per_prefix = (prefix_lengths - 1).clamp(min=0)
        avg_loss_per_candidate = total_loss_per_candidate / (2 * num_edges_per_prefix.clamp(min=1).float())
        avg_loss_per_candidate[num_edges_per_prefix == 0] = 999.0
        
        return avg_loss_per_candidate
        

    
    @torch.no_grad()
    def solve_batch_hybrid_vs_proposals(self, td, env):
        print("\n--- Running hybrid solver with sparse DM for TSP1000 ---")
        B, N, _ = td['locs'].shape
        device = self.device

        # <<< MODIFIED: Use the more robust decoder from your sparse evaluation script >>>
        def construct_tour_from_edges(edge_list, num_nodes, start_node=0):
            if not edge_list or len(edge_list) < num_nodes: return []
            adj = defaultdict(list)
            for u, v in edge_list:
                adj[u].append(v); adj[v].append(u)
            if start_node not in adj:
                start_node = next(iter(adj)) if adj else 0
            tour, visited_nodes = [start_node], {start_node}
            prev_node, curr_node = -1, start_node
            while len(tour) < num_nodes:
                neighbors = adj.get(curr_node, [])
                next_node_found, next_node = False, -1
                for neighbor in neighbors:
                    if neighbor != prev_node:
                        next_node, next_node_found = neighbor, True; break
                if not next_node_found or next_node in visited_nodes: return []
                tour.append(next_node); visited_nodes.add(next_node)
                prev_node, curr_node = curr_node, next_node
            return tour

        def decode_dm_heatmap_edge_greedy_batch(adj_matrices_probs, instance_locs, batch_prefix_nodes):
            B_decode, N_decode, _ = adj_matrices_probs.shape
            device_decode = adj_matrices_probs.device
            adj_probs = (adj_matrices_probs + adj_matrices_probs.transpose(1, 2)) / 2.0
            dists = torch.cdist(instance_locs, instance_locs, p=2) + 1e-9
            edge_scores = adj_probs / dists
            indices = torch.triu_indices(N_decode, N_decode, offset=1, device=device_decode)
            flat_scores = edge_scores[:, indices[0], indices[1]]
            _, sorted_indices = torch.sort(flat_scores, dim=1, descending=True)
            sorted_edges_u, sorted_edges_v = indices[0][sorted_indices], indices[1][sorted_indices]
            final_tours = torch.full((B_decode, N_decode), -1, dtype=torch.long, device=device_decode)

            for i in range(B_decode):
                parent = torch.arange(N_decode, device=device_decode)
                def find_set(v):
                    if v == parent[v]: return v
                    parent[v] = find_set(parent[v]); return parent[v]
                def unite_sets(a, b):
                    a, b = find_set(a), find_set(b)
                    if a != b: parent[b] = a
                node_degrees, edges_in_tour = torch.zeros(N_decode, dtype=torch.int, device=device_decode), []
                # Handle fixed prefix edges first
                prefix_nodes = batch_prefix_nodes[i]
                prefix_len = (prefix_nodes != 0).sum().item() # Assuming padding is 0
                prefix_nodes = prefix_nodes[:prefix_len]
                if prefix_len > 1:
                    for j in range(prefix_len - 1):
                        u, v = prefix_nodes[j].item(), prefix_nodes[j+1].item()
                        edges_in_tour.append((u, v)); node_degrees[u] += 1; node_degrees[v] += 1; unite_sets(u, v)
                
                for u_tensor, v_tensor in zip(sorted_edges_u[i], sorted_edges_v[i]):
                    if len(edges_in_tour) >= N_decode - 1: break
                    u, v = u_tensor.item(), v_tensor.item()
                    # Check if edge is part of the prefix
                    is_prefix_edge = False
                    if prefix_len > 1:
                        for j in range(prefix_len - 1):
                            p_u, p_v = prefix_nodes[j].item(), prefix_nodes[j+1].item()
                            if (u == p_u and v == p_v) or (u == p_v and v == p_u): is_prefix_edge = True; break
                    if is_prefix_edge: continue

                    if node_degrees[u] < 2 and node_degrees[v] < 2 and find_set(u) != find_set(v):
                        edges_in_tour.append((u, v)); node_degrees[u] += 1; node_degrees[v] += 1; unite_sets(u, v)
                
                if len(edges_in_tour) == N_decode - 1:
                    endpoints = (node_degrees == 1).nonzero(as_tuple=True)[0]
                    if len(endpoints) == 2: edges_in_tour.append((endpoints[0].item(), endpoints[1].item()))
                
                if len(edges_in_tour) == N_decode:
                    start_node = prefix_nodes[0].item() if prefix_len > 0 else 0
                    tour_sequence = construct_tour_from_edges(edges_in_tour, N_decode, start_node=start_node)
                    if tour_sequence and len(tour_sequence) == N_decode:
                        final_tours[i] = torch.tensor(tour_sequence, device=device_decode)
            return final_tours, (final_tours != -1).all(dim=1)


        # --- Initialization ---
        td_step = env.reset(td.clone())
        hybrid_solutions = torch.zeros(B, N, dtype=torch.long, device=device)
        dm_triggered_flags = torch.zeros(B, dtype=torch.bool, device=device)
        dm_proposal_stats = [{"cost": torch.tensor(float('inf'), device=device)} for _ in range(B)]
        node_embeds, _ = self.rl_policy.encoder(td_step)
        cached_embeds = self.rl_policy.decoder._precompute_cache(node_embeds)
        trigger_counts = 0

        # --- Main Solving Loop ---
        while td_step['i'][0] < N:
            step_idx = td_step['i'].squeeze(-1)
            current_step_scalar = step_idx[0].item()

            logits, _ = self.rl_policy.decoder(td_step, cached_embeds)
            mask = td_step["action_mask"]
            probs = F.softmax(logits + mask.log(), dim=-1)
            
            rl_greedy_choice = probs.argmax(-1)
            best_next_nodes_for_hybrid_path = rl_greedy_choice.clone()

            active_mask = ~td_step["done"].squeeze(-1)
            if not active_mask.any(): break
            
            probe_mask = active_mask & ~dm_triggered_flags
            
            if probe_mask.any() and self.cfg.solver.use_theory_trigger:
                indices_to_probe = probe_mask.nonzero().squeeze(-1)
                num_to_probe = len(indices_to_probe)
                
                M = self.cfg.solver.probe_rl_top_m
                probs_rl_probe = probs[probe_mask]
                num_available_actions = int(mask[probe_mask].sum(dim=1).min().item())
                M = min(M, num_available_actions)

                if M > 0:
                    top_m_probs, top_m_indices = torch.topk(probs_rl_probe, k=M, dim=1)
                    entropy_rl = -torch.sum(top_m_probs * torch.log(top_m_probs + 1e-9), dim=-1)
                    is_high_entropy = entropy_rl > self.cfg.solver.entropy_threshold
                    
                    trigger_now_mask_relative = torch.zeros_like(is_high_entropy)

                    if current_step_scalar == 0 and False:
                        trigger_now_mask_relative = is_high_entropy
                    else:
                        # 1. Read the probe interval from the config, default 1 (probe at every step)
                        probe_interval = self.cfg.solver.get("probe_interval", 1)
                        
                        # 2. By default the trigger is based only on the (cheap) entropy
                        trigger_now_mask_relative = is_high_entropy

                        if current_step_scalar % probe_interval == 0:
                        
                            path_so_far = hybrid_solutions[probe_mask, :current_step_scalar]
                            
                            expanded_paths = path_so_far.repeat_interleave(M, dim=0)
                            candidate_nodes = top_m_indices.reshape(-1, 1)
                            # print(f"candidate_nodes is : {candidate_nodes}")
    
                            prefix_part = torch.cat([expanded_paths, candidate_nodes], dim=1)
                            padding = torch.zeros(prefix_part.shape[0], N - prefix_part.shape[1], dtype=torch.long, device=device)
                            candidate_prefixes = torch.cat([prefix_part, padding], dim=1)
                            prefix_lengths = torch.full((num_to_probe * M,), current_step_scalar + 1, device=device)
                            
                            dm_to_instance_idx = torch.arange(num_to_probe, device=device).repeat_interleave(M)
                            expanded_locs = td['locs'][indices_to_probe][dm_to_instance_idx]
                            
                            dm_scores = self._compute_dm_prior_scores(expanded_locs, candidate_prefixes, prefix_lengths).view(num_to_probe, M)
                            
                            log_p_dm = F.log_softmax(-dm_scores / self.cfg.solver.dm_prior_temp, dim=-1)
                            # kl_divergence = F.kl_div(log_p_dm, top_m_probs, reduction='none').sum(dim=-1)
    
                            # --- START OF CORRECTION for KL Calculation ---
                            # 1. We already have log_p_dm (which is log(Q))
                            # 2. We need log_p_rl, which is log(P)
                            log_p_rl = torch.log(top_m_probs + 1e-9) # Add epsilon for numerical stability
                            
                            # 3. Apply the direct formula: sum(P * (log(P) - log(Q)))
                            kl_divergence = (top_m_probs * (log_p_rl - log_p_dm)).sum(dim=-1)
                            # --- END OF CORRECTION ---
    
                            
                            is_high_divergence = kl_divergence > self.cfg.solver.kl_div_threshold
                            trigger_now_mask_relative = is_high_entropy | is_high_divergence
                            print(f"[DEBUG] step {current_step_scalar}, entropy={entropy_rl.mean():.3f}, KL={kl_divergence.mean():.3f}")
                        else:
                            # On skipped steps only print the entropy information
                            print(f"[DEBUG] step {current_step_scalar}, entropy={entropy_rl.mean():.3f} (Skipped)")

                    
                    if trigger_now_mask_relative.any():
                        absolute_trigger_indices = indices_to_probe[trigger_now_mask_relative]
                        print(f"--- Step {current_step_scalar}: DM triggered for {len(absolute_trigger_indices)} instances. ---")
                        #dm_triggered_flags[absolute_trigger_indices] = True
                        trigger_counts += 1
                        if trigger_counts <= 0:
                            dm_triggered_flags[absolute_trigger_indices] = False
                        else:
                            dm_triggered_flags[absolute_trigger_indices] = True
                        # --- Call Sparse Diffusion Model ---
                        using_diffusion_mask = absolute_trigger_indices
                        num_uncertain = using_diffusion_mask.numel()
                        
                        # Dynamically determine number of candidates per instance
                        probs_to_trigger = probs[using_diffusion_mask]
                        sorted_probs_trigger, sorted_indices_trigger = torch.sort(probs_to_trigger, dim=-1, descending=True)
                        cum_probs_trigger = torch.cumsum(sorted_probs_trigger, dim=-1)
                        cum_thresh = self.cfg.solver.dynamic_n_cumulative_threshold
                        dynamic_n_indices = torch.argmax((cum_probs_trigger >= cum_thresh).int(), dim=-1)
                        dynamic_n_candidates = (dynamic_n_indices + 1).clamp(max=self.cfg.solver.get("max_dynamic_n", N))

                        max_n_in_batch = int(dynamic_n_candidates.max().item())
                        print(f"candidate_nodes is : {max_n_in_batch}")

                        proposals = sorted_indices_trigger[:, :max_n_in_batch]
                        print(f"proposals is : {proposals}")
                        
                        path_so_far_triggered = hybrid_solutions[using_diffusion_mask, :current_step_scalar]
                        expanded_paths_triggered = path_so_far_triggered.repeat_interleave(dynamic_n_candidates, dim=0)
                        arange_mask = torch.arange(max_n_in_batch, device=device).unsqueeze(0)
                        selection_mask = arange_mask < dynamic_n_candidates.unsqueeze(1)
                        candidate_nodes_triggered = proposals[selection_mask]
                        
                        prefix_part = torch.cat([expanded_paths_triggered, candidate_nodes_triggered.unsqueeze(1)], dim=1)
                        padding = torch.zeros(prefix_part.shape[0], N - prefix_part.shape[1], dtype=torch.long, device=device)
                        final_prefixes = torch.cat([prefix_part, padding], dim=1)


        
                        
                        prefix_lengths_dm = torch.full((final_prefixes.shape[0],), current_step_scalar + 1, device=device)
                        dm_to_instance_idx_final = torch.arange(num_uncertain, device=device).repeat_interleave(dynamic_n_candidates)
                        expanded_locs_dm = td['locs'][using_diffusion_mask][dm_to_instance_idx_final]

                        
                        # 1. Prepare the sparse batch data
                        dm_batch_data = self._prepare_sparse_dm_batch(expanded_locs_dm, final_prefixes, prefix_lengths_dm)
                        
                        # 2. Call the modified sampling loop and receive the sparse outputs
                        _, _, _, final_edge_index, final_edge_logits = self.diffusion_handler.p_sample_loop_ddim(
                          denoiser_model=self.dm_model,
                          batch_data=dm_batch_data,
                          num_inference_steps=self.cfg.solver.dm_inference_steps,
                          schedule=self.cfg.eval.inference_schedule_type
                        )
                        


                        
                        # 3. [Key optimization] efficiently build a dense heatmap from the sparse outputs
                        batch_size_dm = final_prefixes.shape[0]
                        device = final_edge_logits.device
                        
                        # Create an empty dense adjacency matrix
                        adj_matrices_probs = torch.zeros(batch_size_dm, N, N, device=device)
                        
                        # Get the graph ID of every edge in the batch, used for scatter
                        # node_to_graph_batch[final_edge_index[0]] is an (E,) tensor holding the graph ID of every edge
                        batch_indices = dm_batch_data["node_to_graph_batch"][final_edge_index[0]]
                        
                        # Get the local indices of the edges (relative to each graph)
                        # % N gives the index inside each graph
                        rows_local = final_edge_index[0] % N
                        cols_local = final_edge_index[1] % N
                        
                        # Fill all edge probabilities into the dense matrix in one indexing operation
                        adj_matrices_probs[batch_indices, rows_local, cols_local] = torch.sigmoid(final_edge_logits)
                        
                        # Enforce symmetry since TSP is undirected
                        adj_matrices_probs = (adj_matrices_probs + adj_matrices_probs.transpose(1, 2)) / 2.0
                        
                        # 4. [Call the efficient decoder]
                        decoded_tours, decoding_ok = decode_dense_greedy_from_heatmaps(
                            adj_matrices_probs,
                            dm_batch_data["prefix_nodes"]
                        )
                        
                        # +++++++++++++++ End of optimized block +++++++++++++++

                        # --- END OF MODIFICATION ---
                        
                        # (The subsequent cost computation and best-of-N logic are unchanged)
                        costs = torch.full((final_prefixes.shape[0],), float('inf'), device=device)
                        if decoding_ok.any():
                          # Note: expanded_locs_dm must be reshaped into the flattened format expected by the GNN
                          flat_locs_for_cost = dm_batch_data["instance_locs"] 
                          costs[decoding_ok] = calculate_tsp_cost_batch(flat_locs_for_cost.view(-1, N, 2)[decoding_ok], decoded_tours[decoding_ok])
                        
                        costs_split = torch.split(costs, dynamic_n_candidates.cpu().tolist())
                        tours_split = torch.split(decoded_tours, dynamic_n_candidates.cpu().tolist())
                        
                        dm_chosen_nodes = torch.zeros(num_uncertain, dtype=torch.long, device=device)
                        for i in range(num_uncertain):
                          if len(costs_split[i]) == 0: continue
                          best_local_idx = torch.argmin(costs_split[i])
                          dm_chosen_nodes[i] = proposals[i, best_local_idx]
                          
                          best_dm_cost = costs_split[i][best_local_idx]
                          original_batch_idx = using_diffusion_mask[i].item()
                          if not torch.isinf(best_dm_cost) and best_dm_cost < dm_proposal_stats[original_batch_idx]["cost"]:
                              dm_proposal_stats[original_batch_idx] = {
                                  "cost": best_dm_cost,
                                  "tour": tours_split[i][best_local_idx],
                                  "source": "DM Proposal"
                              }
                        
                        best_next_nodes_for_hybrid_path[using_diffusion_mask] = dm_chosen_nodes

            hybrid_solutions[torch.arange(B), step_idx] = best_next_nodes_for_hybrid_path
            td_step.set("action", best_next_nodes_for_hybrid_path)
            td_step = env.step(td_step)["next"]

        # --- Final Selection ---
        final_hybrid_costs = calculate_tsp_cost_batch(td['locs'], hybrid_solutions)
        final_solutions = hybrid_solutions.clone()
        final_costs = final_hybrid_costs.clone()
        run_statistics = [{} for _ in range(B)]

        for i in range(B):
            proposal_cost = dm_proposal_stats[i]["cost"]
            if proposal_cost < final_hybrid_costs[i]:
                final_solutions[i] = dm_proposal_stats[i]["tour"]
                final_costs[i] = proposal_cost
            
            run_statistics[i] = {
                "best_cost_before_2opt": final_costs[i].item(),
                "best_tour": final_solutions[i],
                "source": dm_proposal_stats[i].get("source", "Hybrid Path")
            }

        print("--- Hybrid run finished. Final selection complete. ---")
        return final_solutions, run_statistics




def decode_dense_greedy_from_heatmaps(adj_matrices_probs, batch_prefix_nodes):
    """
    Efficient, fully vectorized greedy decoder (from the TSP-500 code).
    The input is a dense adjacency probability matrix.
    """
    B_decode, N_decode, _ = adj_matrices_probs.shape
    device = adj_matrices_probs.device
    final_tours = torch.full((B_decode, N_decode), -1, dtype=torch.long, device=device)
    visited_mask = torch.zeros((B_decode, N_decode), dtype=torch.bool, device=device)
    
    # Use the first node of the prefix as the start node
    current_nodes = batch_prefix_nodes[:, 0]
    final_tours[:, 0] = current_nodes
    visited_mask.scatter_(1, current_nodes.unsqueeze(1), True)

    for i in range(1, N_decode):
        # Copy the probabilities of the current step and mask visited nodes
        step_probs = adj_matrices_probs.clone()
        # [B, 1, N] -> [B, N, N]
        visited_expanded = visited_mask.unsqueeze(1).expand(-1, N_decode, -1)
        step_probs.masked_fill_(visited_expanded, -1e9)
        
        # From the current node, pick the highest-probability edge to an unvisited node
        # gather along dim=1 is the key: it indexes along rows (u)
        next_node_probs = step_probs.gather(1, current_nodes.view(-1, 1, 1).expand(-1, -1, N_decode)).squeeze(1)
        next_nodes = torch.argmax(next_node_probs, dim=1)
        
        # Update the path and mask
        final_tours[:, i] = next_nodes
        visited_mask.scatter_(1, next_nodes.unsqueeze(1), True)
        current_nodes = next_nodes
            
    decoding_ok_mask = (final_tours != -1).all(dim=1)
    return final_tours, decoding_ok_mask

# (construct_tour_from_edges is still needed)
def construct_tour_from_edges(edge_list, num_nodes, start_node=0):
    # ... unchanged ...
    if not edge_list or len(edge_list) < num_nodes: return []
    adj = defaultdict(list)
    for u, v in edge_list:
        adj[u].append(v); adj[v].append(u)
    if start_node not in adj:
        start_node = next(iter(adj)) if adj else 0
    tour, visited_nodes = [start_node], {start_node}
    prev_node, curr_node = -1, start_node
    while len(tour) < num_nodes:
        neighbors = adj.get(curr_node, [])
        next_node_found, next_node = False, -1
        for neighbor in neighbors:
            if neighbor != prev_node:
                next_node, next_node_found = neighbor, True; break
        if not next_node_found or next_node in visited_nodes: return []
        tour.append(next_node); visited_nodes.add(next_node)
        prev_node, curr_node = curr_node, next_node
    return tour



def decode_sparse_greedy_batch(
    sparse_edge_index, sparse_edge_logits,
    instance_locs, batch_prefix_nodes,
    num_nodes, batch_size, node_to_graph_batch
):
    """
    Greedy decoding directly on sparse edges, avoiding the N*N dense matrix.
    """
    device = sparse_edge_logits.device
    sparse_edge_probs = torch.sigmoid(sparse_edge_logits)
    
    # We need a list of size batch_size storing the edges and scores of each graph
    # because different graphs may have different numbers of edges
    final_tours = torch.full((batch_size, num_nodes), -1, dtype=torch.long, device=device)

    # Main loop over the graphs in the batch
    for i in range(batch_size):
        # 1. Select the edges of the current graph i from the large sparse tensor
        #    a. Build a mask marking which nodes belong to the current graph
        graph_node_mask = (node_to_graph_batch == i)
        #    b. Get the global node index range of the current graph
        node_offset = torch.where(graph_node_mask)[0].min()
        #    c. Build a mask marking which edges belong to the current graph (only the source node is checked)
        graph_edge_mask = (node_to_graph_batch[sparse_edge_index[0]] == i)
        
        # 2. Extract the edges, probabilities and node coordinates of the current graph
        current_edges = sparse_edge_index[:, graph_edge_mask]
        current_probs = sparse_edge_probs[graph_edge_mask]
        current_locs = instance_locs[graph_node_mask].view(num_nodes, -1)

        # 3. Convert global edge indices into local indices (0 to N-1)
        current_edges_local = current_edges - node_offset
        u_local, v_local = current_edges_local[0], current_edges_local[1]

        # 4. Compute edge lengths and final scores (prob / distance)
        edge_lengths = torch.linalg.norm(current_locs[u_local] - current_locs[v_local], dim=1)
        edge_scores = current_probs / (edge_lengths + 1e-9)

        # 5. Sort the sparse edges of the current graph
        _, sorted_indices = torch.sort(edge_scores, descending=True)
        sorted_edges_u = u_local[sorted_indices]
        sorted_edges_v = v_local[sorted_indices]

        # 6. Run exactly the same greedy algorithm (DSU) as the dense decoder
        parent = torch.arange(num_nodes, device=device)
        def find_set(v):
            if v == parent[v]: return v
            parent[v] = find_set(parent[v]); return parent[v]
        def unite_sets(a, b):
            a, b = find_set(a), find_set(b)
            if a != b: parent[b] = a
        
        node_degrees = torch.zeros(num_nodes, dtype=torch.int, device=device)
        edges_in_tour = []
        
        # Handle the fixed prefix edges
        prefix_nodes = batch_prefix_nodes[i]
        prefix_len = (prefix_nodes != 0).sum().item() # Assume 0 is the padding value
        prefix_nodes = prefix_nodes[:prefix_len]
        if prefix_len > 1:
            for j in range(prefix_len - 1):
                u, v = prefix_nodes[j].item(), prefix_nodes[j+1].item()
                edges_in_tour.append((u, v)); node_degrees[u] += 1; node_degrees[v] += 1; unite_sets(u, v)
        
        # Loop over the sorted sparse edges; far fewer iterations than N*N
        for u, v in zip(sorted_edges_u, sorted_edges_v):
            if len(edges_in_tour) >= num_nodes - 1: break
            u, v = u.item(), v.item()
            
            is_prefix_edge = False
            if prefix_len > 1:
                for j in range(prefix_len - 1):
                    p_u, p_v = prefix_nodes[j].item(), prefix_nodes[j+1].item()
                    if (u == p_u and v == p_v) or (u == p_v and v == p_u): is_prefix_edge = True; break
            if is_prefix_edge: continue

            if node_degrees[u] < 2 and node_degrees[v] < 2 and find_set(u) != find_set(v):
                edges_in_tour.append((u, v)); node_degrees[u] += 1; node_degrees[v] += 1; unite_sets(u, v)
        
        # 7. Build the final tour
        if len(edges_in_tour) == num_nodes - 1:
            endpoints = (node_degrees == 1).nonzero(as_tuple=True)[0]
            if len(endpoints) == 2: edges_in_tour.append((endpoints[0].item(), endpoints[1].item()))
        
        if len(edges_in_tour) == num_nodes:
            start_node = prefix_nodes[0].item() if prefix_len > 0 else 0
            tour_sequence = construct_tour_from_edges(edges_in_tour, num_nodes, start_node=start_node)
            if tour_sequence and len(tour_sequence) == num_nodes:
                final_tours[i] = torch.tensor(tour_sequence, device=device)

    # Return the decoded tours and a mask of successfully decoded instances
    return final_tours, (final_tours != -1).all(dim=1)



def run(cfg: DictConfig):
    solver = HybridSolver(cfg)
    device = solver.device
    env = get_env(cfg.rl_model.problem, generator_params={"num_loc": cfg.model.num_nodes})
    dataset = env.dataset(filename=cfg.data.test_path)
    eval_dataset = torch.utils.data.Subset(dataset, range(cfg.eval.num_samples_to_evaluate))
    dataloader = DataLoader(eval_dataset, batch_size=cfg.eval.batch_size, shuffle=False)

    all_stats = []
    all_gt_costs = []
    start_time = time.time()
    
    for batch in tqdm(dataloader, desc="Solving Batches"):
        td = TensorDict(batch, batch_size=batch['locs'].shape[0]).to(device)
        td['locs'] = td['locs'].float()
        
        solved_tours, batch_stats = solver.solve_batch_hybrid_vs_proposals(td, env)
        
        if cfg.solver.get("apply_two_opt", False):
            print("Applying 2-opt post-processing...")
            solved_tours = apply_2opt_batch(solved_tours, td['locs'],max_iterations=1000)
            
        final_costs = calculate_tsp_cost_batch(td['locs'], solved_tours)
        
        # Calculate Ground Truth (Ordered) Cost for gap calculation
        gt_tour_indices = torch.arange(cfg.model.num_nodes, device=device).unsqueeze(0).repeat(td.shape[0], 1)
        gt_costs = calculate_tsp_cost_batch(td['locs'], gt_tour_indices)
        all_gt_costs.append(gt_costs.cpu())
        
        for i, stat in enumerate(batch_stats):
            stat['final_cost_after_2opt'] = final_costs[i].item()
        all_stats.extend(batch_stats)
    
    total_time = time.time() - start_time

    
    # --- Instance-level statistics ---
    # 1. Extract the final cost of every instance from all_stats
    final_costs_all = [s['final_cost_after_2opt'] for s in all_stats]
    
    # 2. Make sure gt_costs and final_costs are aligned NumPy arrays
    gt_costs_tensor = torch.cat(all_gt_costs)
    
    # Convert to NumPy arrays
    final_costs_np = np.array(final_costs_all)
    gt_costs_np = gt_costs_tensor.cpu().numpy()

    # 3. Compute the gap of every instance
    # (gt_costs_np is assumed to be non-zero, which always holds for TSP)
    instance_gaps = ((final_costs_np / gt_costs_np) - 1) * 100
    
    # 4. Mean (Avg) and standard deviation (STD) of these gaps
    avg_gap_of_instances = np.mean(instance_gaps)
    std_gap_of_instances = np.std(instance_gaps) # <-- instance-level standard deviation
    
    # 5. Also compute the mean cost
    avg_final_cost = np.mean(final_costs_np)
    
    print("\n" + "=" * 60)
    print("--- Hybrid Solver Evaluation Summary (Single Run) ---")
    print(f"Total time: {total_time:.2f}s")
    print(f"Evaluated {len(all_stats)} instances.")
    # Print the instance-level statistics of this run
    print(f"Average Final Cost: {avg_final_cost:.4f}")
    print(f"Optimality Gap:     {avg_gap_of_instances:.2f}% ± {std_gap_of_instances:.2f}% (Mean ± Instance-level STD)")
    print("=" * 60)

    # Return for this run: 1. mean gap, 2. instance STD, 3. mean cost
    return avg_gap_of_instances, std_gap_of_instances, avg_final_cost


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid RL-DM Solver for TSP1000 with Sparse DM")
    parser.add_argument("--config", type=str, default="configs/hyco_tsp1000_pomo.yaml", help="Path to the YAML configuration file.")
    args = parser.parse_args()
    # --- Multi-seed configuration ---
    seeds = [42] # Seeds to run
    n_runs = len(seeds)
    
    # Collect statistics over N runs
    all_avg_gaps = []       # Stores N "mean gap" values
    all_instance_stds = []  # Stores N "instance STD" values
    all_avg_costs = []      # Stores N "mean cost" values
    
    print(f"--- STARTING MULTI-SEED EVALUATION FOR {n_runs} RUNS ---")
    print(f"Seeds to be used: {seeds}")

    for i, seed in enumerate(seeds):
        print("\n" + "#" * 80)
        print(f"--- RUN {i+1}/{n_runs} (Seed: {seed}) ---")
        print("#" * 80)
        
        # 1. Set the seed for this run
        set_seed(seed)
        
        # 2. Load the config
        cfg = OmegaConf.load(args.config)
        default_solver_cfg = OmegaConf.create({
            'solver': {
                'use_theory_trigger': True,
                'probe_rl_top_m': 15,
                'dm_probe_timestep': 500,
                'dm_prior_temp': 0.1,
                'entropy_threshold': 1.6,
                'kl_div_threshold': 8,
                'dm_inference_steps': 10,
                'dynamic_n_cumulative_threshold': 0.8,
                'apply_two_opt': True
                
            }
        })
        cfg = OmegaConf.merge(default_solver_cfg, cfg)

        if i == 0:
             print("--- Running Hybrid Solver with Final Configuration ---")
             print(OmegaConf.to_yaml(cfg))
             print("----------------------------------------------------")
        
        # 3. Run one evaluation
        # run() returns 3 values
        avg_gap, inst_std, avg_cost = run(cfg)
        
        # 4. Collect results
        all_avg_gaps.append(avg_gap)
        all_instance_stds.append(inst_std)
        all_avg_costs.append(avg_cost)
        
        print(f"--- RESULT (Seed: {seed}): Avg_Gap = {avg_gap:.2f}%, Inst_STD = {inst_std:.2f}%, Avg_Cost = {avg_cost:.4f} ---")

    # --- After the loop: compute the final two-level statistics ---
    
    # 1. Instance-level statistics (averaged over N runs)
    mean_avg_gap = np.mean(all_avg_gaps)
    mean_instance_std = np.mean(all_instance_stds) # Average of the N "instance STD" values
    
    # 2. Seed-level statistics (variation across the N runs)
    std_of_avg_gaps = np.std(all_avg_gaps) # <-- seed-level standard deviation
    
    # (cost statistics)
    mean_avg_cost = np.mean(all_avg_costs)
    std_of_avg_costs = np.std(all_avg_costs)


    print("\n" + "=" * 80)
    print("--- FINAL MULTI-SEED EVALUATION SUMMARY ---")
    print(f"Ran {n_runs} experiments with seeds: {seeds}")
    
    print("\n--- [FINAL STATISTICS (Instance-level Performance)] ---")
    print(f"  (Averaged across {n_runs} runs)")
    print(f"  Optimality Gap: {mean_avg_gap:.2f}% ± {mean_instance_std:.2f}%")
    print(f"  (This is: Mean_Gap ± Average_Instance_STD)")
    
    print("\n--- [FINAL STATISTICS (Experiment Stability)] ---")
    print(f"  (Calculated across {n_runs} runs)")
    print(f"  Optimality Gap: {mean_avg_gap:.2f}% ± {std_of_avg_gaps:.2f}%")
    print(f"  (This is: Mean_Gap ± Inter-Seed_STD)")
    print(f"  Average Cost:   {mean_avg_cost:.4f} ± {std_of_avg_costs:.4f}")
    print(f"  (This is: Mean_Cost ± Inter-Seed_STD)")

    print("\n--- Raw Data ---")
    print("  Individual Avg. Gap Results: " + ", ".join([f"{g:.2f}%" for g in all_avg_gaps]))
    print("  Individual Instance STD Results: " + ", ".join([f"{s:.2f}%" for s in all_instance_stds]))
    print("=" * 80)