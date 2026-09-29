# HyCO (LEHD) on TSP-1000 with trigger-step statistics.
# Usage (from tsp/lehd/): python trigger_stats_lehd_tsp1000.py --config configs/hyco_lehd_tsp1000.yaml

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
from dataclasses import dataclass
import sys

# --- RL4CO Imports ---
from rl4co.envs import get_env
from rl4co.utils.ops import unbatchify

# <<< Sparse diffusion model imports (sparse model for TSP-1000) >>>
import os
import sys
# Prefix-DIFUSCO modules live in the parent tsp/ directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diffusion_model_sparse import ConditionalTSPSuffixDiffusionModel
from discrete_diffusion_sparse import AdjacencyMatrixDiffusion

# --- Helper Function Imports ---
from tsp_utils import calculate_tsp_cost_batch, apply_2opt_batch
import random

# --- LEHD imports ---
try:
    from TSPModel import TSPModel
except ImportError:
    print("Warning: Could not import LEHD modules normally. Assuming files are in place.")

# --- LEHD data class ---
@dataclass
class Step_State:
    data: torch.Tensor

def set_seed(seed):
    """Set all relevant random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class HybridSolver:
    """
    Hybrid Solver for TSP1000: LEHD (RL) + Sparse Diffusion Model (Improver)
    """
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Solver using device: {self.device}")

        self.rl_policy = self._load_rl_policy()
        self.dm_model = self._load_dm_model()
        
        # Use the sparse diffusion handler
        self.diffusion_handler = AdjacencyMatrixDiffusion(
            num_nodes=cfg.model.num_nodes,
            num_timesteps=cfg.diffusion.num_timesteps,
            schedule_type=cfg.diffusion.schedule_type,
            device=self.device,
            sparse_factor=cfg.model.sparse_factor 
        )
        
        self.timing_stats = {
            "rl_forward_time": 0.0,
            "dm_probe_time": 0.0,
            "dm_sample_time": 0.0,
            "dm_decode_time": 0.0,
            "total_time": 0.0,
        }

    def _load_rl_policy(self):
        """
        Load the LEHD model
        """
        print(f"Loading LEHD RL model from: {self.cfg.rl_model.ckpt_path}")
        try:
            # These parameters must match the configuration used to train the LEHD model
            # If not specified, the default architecture is used, or it is read from cfg.rl_model.lehd_params
            model_params = {
                'mode': 'test',
                'embedding_dim': self.cfg.rl_model.lehd_params.get('embedding_dim', 128),
                'sqrt_embedding_dim': self.cfg.rl_model.lehd_params.get('sqrt_embedding_dim', 11.31),
                'decoder_layer_num': self.cfg.rl_model.lehd_params.get('decoder_layer_num', 6),
                'qkv_dim': self.cfg.rl_model.lehd_params.get('qkv_dim', 16),
                'head_num': self.cfg.rl_model.lehd_params.get('head_num', 8),
                'ff_hidden_dim': self.cfg.rl_model.lehd_params.get('ff_hidden_dim', 512),
            }
            
            model = TSPModel(**model_params).to(self.device)
            
            # Load weights
            # Note: weights_only=False is needed for compatibility with older PyTorch serialization
            ckpt = torch.load(self.cfg.rl_model.ckpt_path, map_location=self.device) 
            
            if 'model_state_dict' in ckpt:
                model.load_state_dict(ckpt['model_state_dict'])
            else:
                model.load_state_dict(ckpt, strict=False)
            
            model.eval()
            return model
        except Exception as e:
            print(f"Error loading LEHD model: {e}")
            import traceback
            traceback.print_exc()
            exit()
            
    def _load_dm_model(self):
        # Load the sparse diffusion model
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
            sparse_factor=self.cfg.model.sparse_factor 
        ).to(self.device)
        model.load_state_dict(torch.load(self.cfg.dm_model.ckpt_path, map_location=self.device))
        model.eval()
        return model

    # --- Sparse data preparation ---
    def _prepare_sparse_dm_batch(self, instance_locs, prefix_nodes, prefix_lengths):
        B, N, _ = instance_locs.shape
        device = self.device
        k = self.cfg.model.sparse_factor

        dists = torch.cdist(instance_locs, instance_locs, p=2)
        dists.view(B * N, N)[:, torch.arange(N)] += 1e9 
        
        _, top_k_indices = torch.topk(dists, k=k, dim=-1, largest=False) 
        
        node_offsets = (torch.arange(B, device=device) * N).view(B, 1, 1)
        local_rows = torch.arange(N, device=device).view(1, N, 1)
        row_b = (local_rows + node_offsets).expand(B, N, k).reshape(-1)
        col_b = (top_k_indices + node_offsets).reshape(-1)

        edge_index = torch.stack([torch.cat([row_b, col_b]), torch.cat([col_b, row_b])], dim=0)

        flat_locs = instance_locs.view(B * N, -1)
        node_to_graph_batch = torch.arange(B, device=device).repeat_interleave(N)
        
        dist_feat_unsorted = torch.linalg.norm(
            flat_locs[edge_index[0]] - flat_locs[edge_index[1]], dim=-1
        ).unsqueeze(1)
        
        edge_graph_ids = node_to_graph_batch[edge_index[0]]
        _, sorted_permutation = torch.sort(edge_graph_ids)
        
        edge_index = edge_index[:, sorted_permutation]
        dist_feat = dist_feat_unsorted[sorted_permutation]
        
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
            "edge_index": edge_index,
            "dist_feature": dist_feat,
            "node_to_graph_batch": node_to_graph_batch,
            "node_prefix_state": node_prefix_state,
            "num_nodes": N,
            "is_sparse": True
        }

    # --- Sparse score computation ---
    def _compute_dm_prior_scores(self, instance_locs, candidate_prefixes, prefix_lengths):
        """
        [Chunked, memory-safe version]
        Computes a single-step denoising score for candidate prefixes using the sparse DM.
        """
        total_candidates = candidate_prefixes.shape[0]
        device = self.device
        if total_candidates == 0: return torch.empty(0, device=device)

        # ---------------------------------------------------------
        # Key memory safeguard: define a chunk size; for large N only 5 candidates are probed at a time
        # ---------------------------------------------------------
        N = self.cfg.model.num_nodes
        chunk_size = 3 if N >= 3000 else 20 
        
        all_avg_losses = torch.zeros(total_candidates, device=device)

        # Run the DM forward pass chunk by chunk
        for start_idx in range(0, total_candidates, chunk_size):
            end_idx = min(start_idx + chunk_size, total_candidates)
            
            chunk_instance_locs = instance_locs[start_idx:end_idx]
            chunk_candidate_prefixes = candidate_prefixes[start_idx:end_idx]
            chunk_prefix_lengths = prefix_lengths[start_idx:end_idx]
            
            chunk_size_actual = end_idx - start_idx

            # Prepare sparse batch data for the current chunk
            dm_batch = self._prepare_sparse_dm_batch(chunk_instance_locs, chunk_candidate_prefixes, chunk_prefix_lengths)

            t_probe = torch.full((chunk_size_actual,), self.cfg.solver.dm_probe_timestep, device=device, dtype=torch.long)
            
            prefix_adj_target = torch.zeros(chunk_size_actual, N, N, device=device, dtype=torch.float)
            for i in range(chunk_size_actual):
                if chunk_prefix_lengths[i] > 1:
                    p_nodes = chunk_candidate_prefixes[i, :chunk_prefix_lengths[i]]
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
                all_avg_losses[start_idx:end_idx] = 999.0
                continue
            
            reconstruction_loss = F.binary_cross_entropy_with_logits(
                predicted_x_0_logits_attrs[loss_mask],
                target_edge_attrs[loss_mask],
                reduction='none'
            )
            
            edge_to_graph_map = dm_batch['node_to_graph_batch'][dm_batch['edge_index'][0]]
            total_loss_per_candidate = torch.zeros(chunk_size_actual, device=device)
            total_loss_per_candidate.scatter_add_(0, edge_to_graph_map[loss_mask], reconstruction_loss)
            
            num_edges_per_prefix = (chunk_prefix_lengths - 1).clamp(min=0)
            avg_loss_per_candidate = total_loss_per_candidate / (2 * num_edges_per_prefix.clamp(min=1).float())
            avg_loss_per_candidate[num_edges_per_prefix == 0] = 999.0
            
            all_avg_losses[start_idx:end_idx] = avg_loss_per_candidate
            
            # Manually free the large tensors of this chunk
            del dm_batch, prefix_adj_target, x_t_noisy_attrs, x_t_transformed, predicted_x_0_logits_attrs
            torch.cuda.empty_cache()
            
        return all_avg_losses
    
    def _compute_dm_prior_scores_back(self, instance_locs, candidate_prefixes, prefix_lengths):
        total_candidates = candidate_prefixes.shape[0]
        device = self.device
        if total_candidates == 0: return torch.empty(0, device=device)

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
        if not loss_mask.any(): return torch.full((total_candidates,), 999.0, device=device)
        
        reconstruction_loss = F.binary_cross_entropy_with_logits(
            predicted_x_0_logits_attrs[loss_mask],
            target_edge_attrs[loss_mask],
            reduction='none'
        )
        
        edge_to_graph_map = dm_batch['node_to_graph_batch'][dm_batch['edge_index'][0]]
        total_loss_per_candidate = torch.zeros(total_candidates, device=device)
        total_loss_per_candidate.scatter_add_(0, edge_to_graph_map[loss_mask], reconstruction_loss)
        
        num_edges_per_prefix = (prefix_lengths - 1).clamp(min=0)
        avg_loss_per_candidate = total_loss_per_candidate / (2 * num_edges_per_prefix.clamp(min=1).float())
        avg_loss_per_candidate[num_edges_per_prefix == 0] = 999.0
        return avg_loss_per_candidate
        
    @torch.no_grad()
    def solve_batch_hybrid_vs_proposals(self, td, env):
        print("\n--- Running [LEHD + Sparse DM] Hybrid for TSP1000 ---")
        B, N, _ = td['locs'].shape
        device = self.device

        # --- [Helper] dense heatmap decoding (for diffusion results) ---
        def decode_dense_greedy_from_heatmaps(adj_matrices_probs, batch_prefix_nodes):
            B_decode, N_decode, _ = adj_matrices_probs.shape
            device_d = adj_matrices_probs.device
            final_tours = torch.full((B_decode, N_decode), -1, dtype=torch.long, device=device_d)
            visited_mask = torch.zeros((B_decode, N_decode), dtype=torch.bool, device=device_d)
            
            # Use the first node of the prefix as the start node (usually 0)
            current_nodes = batch_prefix_nodes[:, 0]
            final_tours[:, 0] = current_nodes
            visited_mask.scatter_(1, current_nodes.unsqueeze(1), True)

            for i in range(1, N_decode):
                step_probs = adj_matrices_probs.clone()
                visited_expanded = visited_mask.unsqueeze(1).expand(-1, N_decode, -1)
                step_probs.masked_fill_(visited_expanded, -1e9)
                
                next_node_probs = step_probs.gather(1, current_nodes.view(-1, 1, 1).expand(-1, -1, N_decode)).squeeze(1)
                next_nodes = torch.argmax(next_node_probs, dim=1)
                
                final_tours[:, i] = next_nodes
                visited_mask.scatter_(1, next_nodes.unsqueeze(1), True)
                current_nodes = next_nodes
                    
            decoding_ok_mask = (final_tours != -1).all(dim=1)
            return final_tours, decoding_ok_mask

        # --- Initialization ---
        self.timing_stats = {k: 0.0 for k in self.timing_stats}
        start_total = time.time()
        
        td_step = env.reset(td.clone())
        hybrid_solutions = torch.zeros(B, N, dtype=torch.long, device=device)
        dm_triggered_flags = torch.zeros(B, dtype=torch.bool, device=device)
        dm_proposal_stats = [{"cost": torch.tensor(float('inf'), device=device)} for _ in range(B)]
        
        # Record the concrete trigger step of each sample; -1 means not triggered
        trigger_steps = torch.full((B,), -1, dtype=torch.long, device=device)

        # --- [LEHD change 1] pre-encoding ---
        # LEHD has an encoder-decoder structure; all nodes are encoded first
        lehd_state = Step_State(data=td['locs'])
        start_rl = time.time()
        encoded_nodes = self.rl_policy.encoder(lehd_state.data)
        self.timing_stats["rl_forward_time"] += (time.time() - start_rl)
        
        trigger_counts = 0
        pbar = tqdm(total=N, desc="Constructing Tour", leave=False)

        # --- Main loop ---
        while td_step['i'][0] < N:
            step_idx = td_step['i'].squeeze(-1)
            current_step_scalar = step_idx[0].item()
            pbar.update(1)

            # --- [LEHD change 2] decoding step ---
            start_rl = time.time()
            if current_step_scalar == 0:
                # Step 0: LEHD needs the start node to be handled manually; here we always start from node 0
                probs = torch.zeros(B, N, device=device)
                probs[:, 0] = 1.0
            else:
                # Step > 0: pass the slice of the already generated sequence
                selected_node_list = hybrid_solutions[:, :current_step_scalar]
                # The LEHD decoder directly returns softmax probabilities
                probs = self.rl_policy.decoder(encoded_nodes, selected_node_list)
            
            self.timing_stats["rl_forward_time"] += (time.time() - start_rl)
            
            rl_greedy_choice = probs.argmax(-1)
            best_next_nodes_for_hybrid_path = rl_greedy_choice.clone()

            active_mask = ~td_step["done"].squeeze(-1)
            if not active_mask.any(): break
            
            # --- Probe / trigger logic ---
            probe_mask = active_mask & ~dm_triggered_flags
            
            # [Optimization] circuit breaker: TSP-1000 episodes are long; stop probing if nothing triggered in the first 50 steps
            GIVE_UP_STEP = 50
            if current_step_scalar > GIVE_UP_STEP:
                probe_mask[:] = False
            
            # [Optimization] interval probing: probe every few steps
            PROBE_INTERVAL = 5
            is_interval_step = (current_step_scalar > 0) and (current_step_scalar % PROBE_INTERVAL == 0)

            if probe_mask.any() and self.cfg.solver.use_theory_trigger and is_interval_step:
                indices_to_probe = probe_mask.nonzero().squeeze(-1)
                
                # Get the RL probabilities and apply the mask (double safety)
                probs_rl_probe = probs[probe_mask]
                current_action_mask = td_step["action_mask"][probe_mask]
                probs_rl_probe = probs_rl_probe * current_action_mask
                probs_rl_probe = probs_rl_probe / (probs_rl_probe.sum(dim=-1, keepdim=True) + 1e-9)
                
                M = self.cfg.solver.probe_rl_top_m
                num_available = int(current_action_mask.sum(dim=1).min().item())
                M = min(M, num_available)

                if M > 0:
                    top_m_probs, top_m_indices = torch.topk(probs_rl_probe, k=M, dim=1)
                    entropy_rl = -torch.sum(top_m_probs * torch.log(top_m_probs + 1e-9), dim=-1)
                    
                    # [Optimization] entropy pre-filter
                    PROBE_ENTROPY_MIN_LIMIT = 0.1
                    print(f"entropy_rl is {entropy_rl}")
                    need_probe_local_mask = entropy_rl > PROBE_ENTROPY_MIN_LIMIT
                    trigger_now_mask_relative = torch.zeros_like(entropy_rl, dtype=torch.bool)

                    if need_probe_local_mask.any():
                        num_to_probe = len(indices_to_probe) # Simplification: build everything
                        path_so_far = hybrid_solutions[probe_mask, :current_step_scalar]
                        expanded_paths = path_so_far.repeat_interleave(M, dim=0)
                        candidate_nodes = top_m_indices.reshape(-1, 1)

                        prefix_part = torch.cat([expanded_paths, candidate_nodes], dim=1)
                        padding = torch.zeros(prefix_part.shape[0], N - prefix_part.shape[1], dtype=torch.long, device=device)
                        candidate_prefixes = torch.cat([prefix_part, padding], dim=1)
                        prefix_lengths = torch.full((num_to_probe * M,), current_step_scalar + 1, device=device)
                        
                        dm_to_instance_idx = torch.arange(num_to_probe, device=device).repeat_interleave(M)
                        expanded_locs = td['locs'][indices_to_probe][dm_to_instance_idx]
                        
                        start_dm_probe = time.time()
                        dm_scores = self._compute_dm_prior_scores(expanded_locs, candidate_prefixes, prefix_lengths).view(num_to_probe, M)
                        self.timing_stats["dm_probe_time"] += (time.time() - start_dm_probe)
                        
                        log_p_dm = F.log_softmax(-dm_scores / self.cfg.solver.dm_prior_temp, dim=-1)
                        log_p_rl = torch.log(top_m_probs + 1e-9)
                        # KL computation: P * (logP - logQ)
                        kl_divergence = (top_m_probs * (log_p_rl - log_p_dm)).sum(dim=-1)
                        print(f"kl_divergence is {kl_divergence},entropy is {entropy_rl}")
                        is_high_entropy = entropy_rl > self.cfg.solver.entropy_threshold
                        is_high_divergence = kl_divergence > self.cfg.solver.kl_div_threshold
                        trigger_now_mask_relative = is_high_entropy | is_high_divergence
                    
                    if trigger_now_mask_relative.any():
                        absolute_trigger_indices = indices_to_probe[trigger_now_mask_relative]
                        pbar.write(f"Step {current_step_scalar}: Trigger fired for {len(absolute_trigger_indices)} instances.")
                        
                        trigger_counts += 1
                        dm_triggered_flags[absolute_trigger_indices] = True

                        # Record the actual trigger step
                        trigger_steps[absolute_trigger_indices] = current_step_scalar
                        
                        # --- Prepare diffusion candidates ---
                        using_diffusion_mask = absolute_trigger_indices
                        num_uncertain = using_diffusion_mask.numel()
                        
                        probs_to_trigger = probs[using_diffusion_mask]
                        sorted_probs_trigger, sorted_indices_trigger = torch.sort(probs_to_trigger, dim=-1, descending=True)
                        cum_probs_trigger = torch.cumsum(sorted_probs_trigger, dim=-1)
                        cum_thresh = self.cfg.solver.dynamic_n_cumulative_threshold
                        dynamic_n_indices = torch.argmax((cum_probs_trigger >= cum_thresh).int(), dim=-1)
                        dynamic_n_candidates = dynamic_n_indices + 1
                        
                        # [Important] cap the number of candidates
                        MAX_CANDIDATES_LIMIT = 32 
                        dynamic_n_candidates = dynamic_n_candidates.clamp(max=MAX_CANDIDATES_LIMIT)
                        max_n_in_batch = int(dynamic_n_candidates.max().item())

                        proposals = sorted_indices_trigger[:, :max_n_in_batch]
                        
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

                        # 1. [Sparse] data preparation
                        dm_batch_data = self._prepare_sparse_dm_batch(expanded_locs_dm, final_prefixes, prefix_lengths_dm)
                        
                        # 2. [Sparse] diffusion sampling
                        start_dm_sample = time.time()
                        _, _, _, final_edge_index, final_edge_logits = self.diffusion_handler.p_sample_loop_ddim(
                          denoiser_model=self.dm_model,
                          batch_data=dm_batch_data,
                          num_inference_steps=self.cfg.solver.dm_inference_steps,
                          schedule=self.cfg.eval.inference_schedule_type
                        )
                        self.timing_stats["dm_sample_time"] += (time.time() - start_dm_sample)
                        
                        # 3. [Sparse to dense] decoding
                        start_dm_decode = time.time()
                        batch_size_dm = final_prefixes.shape[0]
                        adj_matrices_probs = torch.zeros(batch_size_dm, N, N, device=device)
                        batch_indices = dm_batch_data["node_to_graph_batch"][final_edge_index[0]]
                        rows_local = final_edge_index[0] % N
                        cols_local = final_edge_index[1] % N
                        adj_matrices_probs[batch_indices, rows_local, cols_local] = torch.sigmoid(final_edge_logits)
                        adj_matrices_probs = (adj_matrices_probs + adj_matrices_probs.transpose(1, 2)) / 2.0
                        
                        decoded_tours, decoding_ok = decode_dense_greedy_from_heatmaps(
                            adj_matrices_probs,
                            dm_batch_data["prefix_nodes"]
                        )
                        self.timing_stats["dm_decode_time"] += (time.time() - start_dm_decode)
                        
                        # 4. Cost computation and selection
                        costs = torch.full((final_prefixes.shape[0],), float('inf'), device=device)
                        if decoding_ok.any():
                          flat_locs_for_cost = dm_batch_data["instance_locs"] 
                          costs[decoding_ok] = calculate_tsp_cost_batch(flat_locs_for_cost.view(-1, N, 2)[decoding_ok], decoded_tours[decoding_ok])
                        
                        costs_split = torch.split(costs, dynamic_n_candidates.cpu().tolist())
                        tours_split = torch.split(decoded_tours, dynamic_n_candidates.cpu().tolist())
                        
                        dm_chosen_nodes = torch.zeros(num_uncertain, dtype=torch.long, device=device)
                        for i in range(num_uncertain):
                          if len(costs_split[i]) == 0: 
                              dm_chosen_nodes[i] = proposals[i, 0]
                              continue
                          
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
                        
                        # [Shadow mode] the DM only records better solutions and does not change the current path
                        # best_next_nodes_for_hybrid_path[using_diffusion_mask] = dm_chosen_nodes
                        pass 

            hybrid_solutions[torch.arange(B), step_idx] = best_next_nodes_for_hybrid_path
            td_step.set("action", best_next_nodes_for_hybrid_path)
            td_step = env.step(td_step)["next"]

        pbar.close()
        
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
            
            # Add a trigger_step field and pass it to the caller
            run_statistics[i] = {
                "best_cost_before_2opt": final_costs[i].item(),
                "best_tour": final_solutions[i],
                "source": dm_proposal_stats[i].get("source", "Hybrid Path"),
                "trigger_step": trigger_steps[i].item()
            }
        
        self.timing_stats["total_time"] = time.time() - start_total
        print(f"Total loop time: {self.timing_stats['total_time']:.3f}s")
        run_statistics[0]["timing_stats"] = self.timing_stats

        return final_solutions, run_statistics

def run(cfg: DictConfig):
    solver = HybridSolver(cfg)
    device = solver.device
    # Load data via rl4co get_env; unchanged, since LEHD can also consume TensorDict data
    env = get_env(cfg.rl_model.problem, generator_params={"num_loc": cfg.model.num_nodes})
    dataset = env.dataset(filename=cfg.data.test_path)
    
    # number of test samples
    num_samples_to_evaluate = 128 
    eval_dataset = torch.utils.data.Subset(dataset, range(num_samples_to_evaluate))
    dataloader = DataLoader(eval_dataset, batch_size=cfg.eval.batch_size, shuffle=False)

    all_stats = []
    all_gt_costs = []
    cumulative_timing = defaultdict(float)
    cumulative_timing["batches_processed"] = 0
    
    start_time = time.time()
    
    for batch in tqdm(dataloader, desc="Solving Batches"):
        td = TensorDict(batch, batch_size=batch['locs'].shape[0]).to(device)
        td['locs'] = td['locs'].float()
        
        solved_tours, batch_stats = solver.solve_batch_hybrid_vs_proposals(td, env)
        
        if "timing_stats" in batch_stats[0]:
            stats = batch_stats[0]["timing_stats"]
            for k, v in stats.items():
                if k != "total_time":
                    cumulative_timing[k] += v
            cumulative_timing["batches_processed"] += 1

        if cfg.solver.get("apply_two_opt", False):
            print("Applying 2-opt post-processing...")

            solved_tours = apply_2opt_batch(solved_tours, td['locs'], max_iterations=2000)
            
        final_costs = calculate_tsp_cost_batch(td['locs'], solved_tours)
        
        gt_tour_indices = torch.arange(cfg.model.num_nodes, device=device).unsqueeze(0).repeat(td.shape[0], 1)
        gt_costs = calculate_tsp_cost_batch(td['locs'], gt_tour_indices)
        all_gt_costs.append(gt_costs.cpu())
        
        for i, stat in enumerate(batch_stats):
            stat['final_cost_after_2opt'] = final_costs[i].item()
        all_stats.extend(batch_stats)
    
    total_time = time.time() - start_time

    final_costs_all = [s['final_cost_after_2opt'] for s in all_stats]
    gt_costs_tensor = torch.cat(all_gt_costs)
    final_costs_np = np.array(final_costs_all)
    gt_costs_np = gt_costs_tensor.cpu().numpy()

    instance_gaps = ((final_costs_np / gt_costs_np) - 1) * 100
    avg_gap_of_instances = np.mean(instance_gaps)
    std_gap_of_instances = np.std(instance_gaps)
    avg_final_cost = np.mean(final_costs_np)

    # --- Solution source statistics ---
    total_instances = len(all_stats)
    sources = [s['source'] for s in all_stats]
    num_dm = sources.count("DM Proposal")
    num_rl = sources.count("Hybrid Path")
    pct_dm = (num_dm / total_instances) * 100 if total_instances > 0 else 0
    pct_rl = (num_rl / total_instances) * 100 if total_instances > 0 else 0
    
    # Extract and average the trigger steps (ignoring -1)
    valid_trigger_steps = [s['trigger_step'] for s in all_stats if s.get('trigger_step', -1) != -1]
    if len(valid_trigger_steps) > 0:
        avg_trigger_step = np.mean(valid_trigger_steps)
    else:
        avg_trigger_step = -1
    
    print("\n" + "=" * 60)
    print("--- Hybrid Solver Evaluation Summary (Single Run) ---")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average Final Cost: {avg_final_cost:.4f}")
    print(f"Optimality Gap:     {avg_gap_of_instances:.2f}% ± {std_gap_of_instances:.2f}%")
    if avg_trigger_step != -1:
        print(f"Average Trigger Step: {avg_trigger_step:.2f}")
    else:
        print("Average Trigger Step: N/A (No triggers fired)")
    print("-" * 60)
    print(f"  - From Diffusion: {num_dm} ({pct_dm:.2f}%)")
    print(f"  - From RL/Hybrid: {num_rl} ({pct_rl:.2f}%)")
    print("=" * 60)
    
    if cumulative_timing["batches_processed"] > 0:
        num_batches = cumulative_timing["batches_processed"]
        print("\n--- PERFORMANCE OVERHEAD ANALYSIS (Avg per Batch) ---")
        print(f"Avg RL Forward Time: {cumulative_timing['rl_forward_time'] / num_batches:.4f}s")
        print(f"Avg DM Probe Time:   {cumulative_timing['dm_probe_time'] / num_batches:.4f}s")
        print(f"Avg DM Sample Time:  {cumulative_timing['dm_sample_time'] / num_batches:.4f}s")
        print("--------------------------------------------------")

    return avg_gap_of_instances, std_gap_of_instances, avg_final_cost, avg_trigger_step

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid LEHD + Sparse DM Solver for TSP1000")
    parser.add_argument("--config", type=str, default="configs/hyco_lehd_tsp1000.yaml", help="Path to the YAML configuration file.")
    args = parser.parse_args()
    
    seeds = [42]
    n_runs = len(seeds)
    all_avg_gaps = []
    all_instance_stds = []
    all_avg_costs = []
    all_avg_trigger_steps = []  # Record the average trigger step of every run
    
    print(f"--- STARTING MULTI-SEED EVALUATION FOR {n_runs} RUNS ---")

    for i, seed in enumerate(seeds):
        print(f"\n--- RUN {i+1}/{n_runs} (Seed: {seed}) ---")
        set_seed(seed)
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
                'apply_two_opt': False 
            }
        })
        cfg = OmegaConf.merge(default_solver_cfg, cfg)

        # Receive the new avg_trigger return value
        avg_gap, inst_std, avg_cost, avg_trigger = run(cfg)
        all_avg_gaps.append(avg_gap)
        all_instance_stds.append(inst_std)
        all_avg_costs.append(avg_cost)
        if avg_trigger != -1:
            all_avg_trigger_steps.append(avg_trigger)

    print("\n--- FINAL SUMMARY ---")
    print(f"Gap: {np.mean(all_avg_gaps):.2f}% ± {np.mean(all_instance_stds):.2f}%")
    # Print the final global average trigger step
    if len(all_avg_trigger_steps) > 0:
        print(f"Global Average Trigger Step: {np.mean(all_avg_trigger_steps):.2f}")
    else:
        print("Global Average Trigger Step: N/A")