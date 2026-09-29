# HyCO (LEHD) on TSP-500 with trigger-step statistics.
# Usage (from tsp/lehd/): python trigger_stats_lehd_tsp500.py --config configs/hyco_lehd_tsp500.yaml

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

# --- RL4CO imports (mainly for the environment and data loading) ---
from rl4co.envs import get_env
from rl4co.utils.ops import unbatchify

# --- Diffusion Model Imports ---
# These modules are expected in the current directory or on the Python path
import os
import sys
# Prefix-DIFUSCO modules live in the parent tsp/ directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diffusion_model import ConditionalTSPSuffixDiffusionModel
from discrete_diffusion import AdjacencyMatrixDiffusion

# --- Helper Function Imports ---
from tsp_utils import calculate_tsp_cost_batch, apply_2opt_batch
import random

# --- LEHD Imports ---
# Make sure the parent directory is on the path so that LEHD can be imported
try:
    from TSPModel import TSPModel
except ImportError:
    print("Warning: Could not import LEHD modules normally. Assuming files are in place or PYTHONPATH is set.")
    # If the LEHD files are in the same directory, TSPModel can be imported directly
    pass

# for compatibility with the LEHD input format
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
    Implements a theoretically-driven hybrid solving approach (LEHD + Diffusion).
    """
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Solver using device: {self.device}")

        # Load models
        self.rl_model = self._load_rl_policy() 
        self.dm_model = self._load_dm_model()
        
        self.diffusion_handler = AdjacencyMatrixDiffusion(
            num_nodes=cfg.model.num_nodes,
            num_timesteps=cfg.diffusion.num_timesteps,
            schedule_type=cfg.diffusion.schedule_type,
            device=self.device
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
            # Prepare the LEHD parameters
            model_params = {
                'mode': 'test',
                'embedding_dim': self.cfg.rl_model.lehd_params.embedding_dim,
                'sqrt_embedding_dim': self.cfg.rl_model.lehd_params.sqrt_embedding_dim,
                'decoder_layer_num': self.cfg.rl_model.lehd_params.decoder_layer_num,
                'qkv_dim': self.cfg.rl_model.lehd_params.qkv_dim,
                'head_num': self.cfg.rl_model.lehd_params.head_num,
                'ff_hidden_dim': self.cfg.rl_model.lehd_params.ff_hidden_dim,
            }
            
            # Initialize the model
            model = TSPModel(**model_params).to(self.device)
            
            # Load weights
            checkpoint = torch.load(self.cfg.rl_model.ckpt_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint) # Support different checkpoint formats
            
            model.eval()
            return model

        except Exception as e:
            print(f"Error loading LEHD model: {e}")
            import traceback
            traceback.print_exc()
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
            time_embed_dim=self.cfg.model.time_embed_dim
        ).to(self.device)
        model.load_state_dict(torch.load(self.cfg.dm_model.ckpt_path, map_location=self.device))
        model.eval()
        return model

    # ==========================================================================================
    #  DM score computation
    # ==========================================================================================
    def _compute_dm_prior_scores(self, instance_locs, candidate_prefixes, prefix_lengths):
        total_candidates, N = candidate_prefixes.shape[0], self.cfg.model.num_nodes
        device = self.device

        if total_candidates == 0:
            return torch.empty(0, device=device)

        t_probe = torch.full((total_candidates,), self.cfg.solver.dm_probe_timestep, device=device, dtype=torch.long)

        prefix_adj_target = torch.zeros(total_candidates, N, N, device=device, dtype=torch.float)
        
        # Build the adjacency matrix
        for i in range(total_candidates):
            if prefix_lengths[i] > 1:
                p_nodes = candidate_prefixes[i, :prefix_lengths[i]]
                prefix_adj_target[i, p_nodes[:-1], p_nodes[1:]] = 1.0
                prefix_adj_target[i, p_nodes[1:], p_nodes[:-1]] = 1.0

        x_t, _ = self.diffusion_handler.q_sample(prefix_adj_target, t_probe)
        x_t_transformed = x_t.float() * 2.0 - 1.0

        node_prefix_state_probe = torch.zeros(total_candidates, N, 1, device=device)
        max_len = prefix_lengths.max().item()
        if max_len > 0:
            prefixes_for_scatter = candidate_prefixes[:, :max_len].long().clone().unsqueeze(-1)
            src = torch.ones_like(prefixes_for_scatter, dtype=torch.float)
            len_mask = torch.arange(max_len, device=device).unsqueeze(0) < prefix_lengths.unsqueeze(1)
            src[~len_mask.unsqueeze(-1)] = 0
            node_prefix_state_probe.scatter_(dim=1, index=prefixes_for_scatter, src=src)

        predicted_x_0_logits = self.dm_model(
            x_t_transformed, t_probe.float(), instance_locs,
            candidate_prefixes, prefix_lengths, node_prefix_state_probe
        )

        loss_mask = prefix_adj_target > 0
        if not loss_mask.any():
            return torch.full((total_candidates,), 999.0, device=device)

        reconstruction_loss = F.binary_cross_entropy_with_logits(
            predicted_x_0_logits[loss_mask],
            prefix_adj_target[loss_mask],
            reduction='none'
        )
        
        num_edges_per_prefix = (prefix_lengths - 1).clamp(min=0)
        total_loss_per_candidate = torch.zeros(total_candidates, device=device)
        
        loss_indices_map = torch.where(loss_mask)
        # Simple sum
        for i in range(total_candidates):
            if (loss_indices_map[0] == i).any():
                total_loss_per_candidate[i] = reconstruction_loss[loss_indices_map[0] == i].sum()
        
        avg_loss_per_candidate = total_loss_per_candidate / (2 * num_edges_per_prefix.clamp(min=1).float())
        avg_loss_per_candidate[num_edges_per_prefix == 0] = 999.0
        
        return avg_loss_per_candidate
    
    # ==========================================================================================
    #  [Core function: hybrid solve with LEHD]
    # ==========================================================================================
    @torch.no_grad()
    def solve_batch_hybrid_vs_proposals(self, td, env):
        print("\n--- Running with LEHD + Diffusion Hybrid [FIXED] ---")
        B, N, _ = td['locs'].shape
        device = self.device
        
        # --- Helper: DM decoding ---
        def decode_dm_heatmap_simple_greedy_batch(adj_matrices_probs, batch_prefix_nodes):
            B_decode, N_decode, _ = adj_matrices_probs.shape
            final_tours = torch.full((B_decode, N_decode), -1, dtype=torch.long, device=device)
            visited_mask = torch.zeros((B_decode, N_decode), dtype=torch.bool, device=device)
            
            current_nodes = batch_prefix_nodes[:, 0]
            final_tours[:, 0] = current_nodes
            visited_mask.scatter_(1, current_nodes.unsqueeze(1), True)
        
            for i in range(1, N_decode):
                step_probs = adj_matrices_probs.clone()
                current_node_mask = visited_mask.unsqueeze(1).expand(-1, N_decode, -1)
                step_probs.masked_fill_(current_node_mask, -1e9)
                
                next_node_probs = step_probs.gather(1, current_nodes.view(-1, 1, 1).expand(-1, -1, N_decode)).squeeze(1)
                next_nodes = torch.argmax(next_node_probs, dim=1)
                
                final_tours[:, i] = next_nodes
                visited_mask.scatter_(1, next_nodes.unsqueeze(1), True)
                current_nodes = next_nodes
                
            decoding_ok_mask = (final_tours != -1).all(dim=1)
            return final_tours, decoding_ok_mask

        self.timing_stats = {k: 0.0 for k in self.timing_stats}
        start_total = time.time()

        # --- Initialization ---
        # Note: the rl4co env is still used to manage the state (action_mask, etc.),
        # while the policy decisions are made by LEHD.
        td_step = env.reset(td.clone())
        hybrid_solutions = torch.zeros(B, N, dtype=torch.long, device=device)
        dm_triggered_flags = torch.zeros(B, dtype=torch.bool, device=device)
        
        # Initialize the trigger step of each sample; -1 means not triggered
        trigger_steps = torch.full((B,), -1, dtype=torch.long, device=device)
        
        dm_proposal_stats = [{
            "cost": torch.tensor(float('inf'), device=device),
            "tour": torch.zeros(N, dtype=torch.long, device=device),
            "generation_step": -1, "prefix_node": -1, 
            "rl_greedy_node_at_step": -1, "candidates_for_the_step": None
        } for _ in range(B)]

        # --- [LEHD-specific] pre-encoding ---
        # LEHD has an encoder-decoder structure; coordinates are encoded first
        # Build a Step_State object to match the LEHD input
        lehd_state = Step_State(data=td['locs']) 
        
        # [Timer start] RL/LEHD encoder forward
        start_rl_forward = time.time()
        encoded_nodes = self.rl_model.encoder(lehd_state.data)
        self.timing_stats["rl_forward_time"] += (time.time() - start_rl_forward)
        # [Timer end]
        
        trigger_counts = 0
        pbar = tqdm(total=N, desc="Constructing Tour", leave=False)
        # --- Main loop ---
        # rl4co step_idx starts from 0
        while td_step['i'][0] < N:
            step_idx = td_step['i'].squeeze(-1)
            current_step_scalar = step_idx[0].item()
            
            # [Timer start] RL/LEHD decoder forward
            start_rl_forward = time.time()
            pbar.update(1)
            # Prepare the LEHD decoder input
            if current_step_scalar == 0:
                # Step 0: LEHD/TSP usually picks a random start node or node 0.
                probs = torch.zeros(B, N, device=device)
                probs[:, 0] = 1.0 # force node 0
            else:
                # Step > 0: pass the list of visited nodes
                selected_node_list = hybrid_solutions[:, :current_step_scalar]
                # Call the LEHD decoder
                probs = self.rl_model.decoder(encoded_nodes, selected_node_list)

            self.timing_stats["rl_forward_time"] += (time.time() - start_rl_forward)
            # [Timer end]

            rl_greedy_choice = probs.argmax(-1)
            best_next_nodes_for_hybrid_path = rl_greedy_choice.clone()

            # Active mask (unfinished instances)
            active_mask = ~td_step["done"].squeeze(-1)
            if not active_mask.any(): break
            
            # Probe mask: active and not triggered before
            probe_mask = active_mask & ~dm_triggered_flags
            # Circuit breaker
            GIVE_UP_STEP = 50 
            if current_step_scalar > GIVE_UP_STEP:
                probe_mask[:] = False # disable all

            # --- Theory-driven trigger ---
            if probe_mask.any() and self.cfg.solver.use_theory_trigger and current_step_scalar > 0:
 
                indices_to_probe = probe_mask.nonzero().squeeze(-1)
                num_to_probe = len(indices_to_probe)
                
                M = self.cfg.solver.probe_rl_top_m
                probs_rl_probe = probs[probe_mask]
                
                current_action_mask = td_step["action_mask"][probe_mask]
                probs_rl_probe = probs_rl_probe * current_action_mask
                probs_rl_probe = probs_rl_probe / (probs_rl_probe.sum(dim=-1, keepdim=True) + 1e-9)

                num_available_actions = int(current_action_mask.sum(dim=1).min().item())

                M = min(M, num_available_actions)

                if M > 0:
                    top_m_probs, top_m_indices = torch.topk(probs_rl_probe, k=M, dim=1)
                    
                    # 1. Policy entropy
                    entropy_rl = -torch.sum(top_m_probs * torch.log(top_m_probs + 1e-9), dim=-1)
                    print(f"[Debug] Step {current_step_scalar}: RL Entropy Stats - Mean: {entropy_rl.mean().item():.4f}, Max: {entropy_rl.max().item():.4f}, Min: {entropy_rl.min().item():.4f}")
                    is_high_entropy = entropy_rl > self.cfg.solver.entropy_threshold
                    
                    # 2. RL-DM disagreement (KL)
                    path_so_far = hybrid_solutions[probe_mask, :current_step_scalar]
                    expanded_paths = path_so_far.repeat_interleave(M, dim=0)
                    candidate_nodes = top_m_indices.reshape(-1, 1)

                    prefix_part = torch.cat([expanded_paths, candidate_nodes], dim=1)
                    padding = torch.zeros(prefix_part.shape[0], N - prefix_part.shape[1], dtype=torch.long, device=device)
                    candidate_prefixes = torch.cat([prefix_part, padding], dim=1)
                    prefix_lengths = torch.full((num_to_probe * M,), current_step_scalar + 1, device=device)

                    dm_to_instance_idx = torch.arange(num_to_probe, device=device).repeat_interleave(M)
                    expanded_locs = td['locs'][indices_to_probe][dm_to_instance_idx]

                    # [Timer start] DM probe
                    start_dm_probe = time.time()
                    dm_scores = self._compute_dm_prior_scores(
                        expanded_locs, candidate_prefixes, prefix_lengths
                    ).view(num_to_probe, M)
                    self.timing_stats["dm_probe_time"] += (time.time() - start_dm_probe)
                    # [Timer end] DM probe

                    # KL Divergence Calculation
                    log_p_dm = F.log_softmax(-dm_scores / self.cfg.solver.dm_prior_temp, dim=-1)
                    p_rl = top_m_probs # already probabilities summing to 1
                    
                    kl_divergence = F.kl_div(log_p_dm, p_rl, reduction='none').sum(dim=-1)
                    
                    is_high_divergence = kl_divergence > self.cfg.solver.kl_div_threshold
                    
                    # Combined trigger condition
                    trigger_now_mask_relative = is_high_entropy | is_high_divergence
                
                    if self.cfg.eval.get("verbose", False):
                        print(f"[DEBUG] step {current_step_scalar}, Entropy={entropy_rl.mean():.3f}, KL={kl_divergence.mean():.3f}")   

                    if trigger_now_mask_relative.any():
                        absolute_trigger_indices = indices_to_probe[trigger_now_mask_relative]
                        print(f"--- Step {current_step_scalar}: Trigger fired for {len(absolute_trigger_indices)} instances. ---")
                        
                        trigger_counts += 1
                        dm_triggered_flags[absolute_trigger_indices] = True 
                        
                        # Record the actual trigger step
                        trigger_steps[absolute_trigger_indices] = current_step_scalar
                        
                        # --- Run diffusion sampling ---
                        using_diffusion_mask = absolute_trigger_indices
                        num_uncertain = using_diffusion_mask.numel()
                        
                        # Select the top-N candidates from the RL probabilities (proposal selection)
                        probs_to_trigger = probs[using_diffusion_mask]
                        sorted_probs_trigger, sorted_indices_trigger = torch.sort(probs_to_trigger, dim=-1, descending=True)
                        cum_probs_trigger = torch.cumsum(sorted_probs_trigger, dim=-1)
                        cum_thresh = self.cfg.solver.dynamic_n_cumulative_threshold
                        dynamic_n_indices = torch.argmax((cum_probs_trigger >= cum_thresh).int(), dim=-1)
                        dynamic_n_candidates = dynamic_n_indices + 1

                        # Hard cap on the number of candidates
                        MAX_CANDIDATES_LIMIT = 32 
                        
                        max_n_raw = int(dynamic_n_candidates.max().item())
                        print(f"[Info] Step {current_step_scalar}: Max raw candidates needed: {max_n_raw}.")
                        if max_n_raw > MAX_CANDIDATES_LIMIT:
                            print(f"[Warning] Step {current_step_scalar}: RL is confused. "
                                  f"Raw candidates needed: {max_n_raw}. Capping to {MAX_CANDIDATES_LIMIT}.")
                        
                        # Apply the cap
                        dynamic_n_candidates = dynamic_n_candidates.clamp(max=MAX_CANDIDATES_LIMIT)

                        max_n_in_batch = int(dynamic_n_candidates.max().item())
                        
                        proposals = sorted_indices_trigger[:, :max_n_in_batch]
                        
                        # Prepare the diffusion inputs
                        path_so_far_triggered = hybrid_solutions[using_diffusion_mask, :current_step_scalar]
                        expanded_paths_triggered = path_so_far_triggered.repeat_interleave(dynamic_n_candidates, dim=0)
                        arange_mask = torch.arange(max_n_in_batch, device=device).unsqueeze(0)
                        selection_mask = arange_mask < dynamic_n_candidates.unsqueeze(1)
                        candidate_nodes_triggered = proposals[selection_mask]
                        
                        final_prefix_part = torch.cat([expanded_paths_triggered, candidate_nodes_triggered.unsqueeze(1)], dim=1)
                        final_padding = torch.zeros(final_prefix_part.shape[0], N - final_prefix_part.shape[1], dtype=torch.long, device=device)
                        final_prefixes = torch.cat([final_prefix_part, final_padding], dim=1)
                        
                        prefix_lengths_dm = torch.full((final_prefixes.shape[0],), current_step_scalar + 1, device=device)
                        dm_to_instance_idx_final = torch.arange(num_uncertain, device=device).repeat_interleave(dynamic_n_candidates)
                        expanded_locs_dm = td['locs'][using_diffusion_mask][dm_to_instance_idx_final]
                        
                        # Build node_prefix_state (masking)
                        node_prefix_state_dm = torch.zeros(final_prefixes.shape[0], N, 1, device=device)
                        max_len_dm = prefix_lengths_dm.max().item()
                        prefixes_for_scatter_dm = final_prefixes[:, :max_len_dm].long().clone().unsqueeze(-1)
                        src_dm = torch.ones_like(prefixes_for_scatter_dm, dtype=torch.float)
                        len_mask_dm = torch.arange(max_len_dm, device=device).unsqueeze(0) < prefix_lengths_dm.unsqueeze(1)
                        src_dm[~len_mask_dm.unsqueeze(-1)] = 0
                        node_prefix_state_dm.scatter_(dim=1, index=prefixes_for_scatter_dm, src=src_dm)
                        
                        # [Timer start] DM sample
                        start_dm_sample = time.time()
                        _, generated_adj_matrices_probs,_ = self.diffusion_handler.p_sample_loop_ddim(
                            denoiser_model=self.dm_model, instance_locs=expanded_locs_dm,
                            prefix_nodes=final_prefixes, prefix_lengths=prefix_lengths_dm,
                            node_prefix_state=node_prefix_state_dm,
                            num_inference_steps=self.cfg.solver.dm_inference_steps,
                            schedule=self.cfg.eval.inference_schedule_type
                        )
                        self.timing_stats["dm_sample_time"] += (time.time() - start_dm_sample)
                        # [Timer end] DM sample
                        
                        # [Timer start] DM decode
                        start_dm_decode = time.time()
                        decoded_tours, decoding_ok_mask = decode_dm_heatmap_simple_greedy_batch(generated_adj_matrices_probs, final_prefixes)
                        costs = torch.full((final_prefixes.shape[0],), float('inf'), device=device)
                        if decoding_ok_mask.any():
                             costs[decoding_ok_mask] = calculate_tsp_cost_batch(expanded_locs_dm[decoding_ok_mask], decoded_tours[decoding_ok_mask])
                        self.timing_stats["dm_decode_time"] += (time.time() - start_dm_decode)
                        # [Timer end] DM decode
                        
                        # --- Compare and select results ---
                        costs_split = torch.split(costs, dynamic_n_candidates.cpu().tolist())
                        tours_split = torch.split(decoded_tours, dynamic_n_candidates.cpu().tolist())
                        
                        dm_chosen_nodes = torch.zeros(num_uncertain, dtype=torch.long, device=device)
                        for i in range(num_uncertain):
                            if len(costs_split[i]) == 0: 
                                dm_chosen_nodes[i] = proposals[i, 0] # Fallback
                                continue
                            
                            best_local_idx = torch.argmin(costs_split[i])
                            # Record the node selected at this step (the first non-prefix node of the proposal)
                            dm_chosen_nodes[i] = proposals[i, best_local_idx]
                            
                            best_dm_cost = costs_split[i][best_local_idx]
                            original_batch_idx = using_diffusion_mask[i].item()
                            
                            # Update proposal stats (if the DM found a better solution)
                            if not torch.isinf(best_dm_cost) and best_dm_cost < dm_proposal_stats[original_batch_idx]["cost"]:
                                dm_proposal_stats[original_batch_idx].update({
                                    "cost": best_dm_cost, 
                                    "tour": tours_split[i][best_local_idx],
                                    "generation_step": current_step_scalar, 
                                    "prefix_node": proposals[i, best_local_idx].item(),
                                    "rl_greedy_node_at_step": rl_greedy_choice[original_batch_idx].item(),
                                    "candidates_for_the_step": proposals[i].cpu().numpy()
                                })
                        
                        # Update the action of the current step (the node of the most promising proposal according to the DM)
                        best_next_nodes_for_hybrid_path[using_diffusion_mask] = dm_chosen_nodes
            
            # --- Take one step ---
            hybrid_solutions[torch.arange(B), step_idx] = best_next_nodes_for_hybrid_path
            td_step.set("action", best_next_nodes_for_hybrid_path)
            td_step = env.step(td_step)["next"]

        # --- Final selection and statistics ---
        final_hybrid_costs = calculate_tsp_cost_batch(td['locs'], hybrid_solutions)
        final_solutions = torch.zeros(B, N, device=device, dtype=torch.long)
        run_statistics = [{} for _ in range(B)]

        for i in range(B):
            hybrid_cost = final_hybrid_costs[i]
            proposal_cost = dm_proposal_stats[i]["cost"]
            # Add a trigger_step field and pass it to the caller
            if proposal_cost < hybrid_cost:
                final_solutions[i] = dm_proposal_stats[i]["tour"]
                run_statistics[i] = {"best_cost": proposal_cost, "best_tour": dm_proposal_stats[i]["tour"], "source": "DM Proposal", "trigger_step": trigger_steps[i].item(), **dm_proposal_stats[i]}
            else:
                final_solutions[i] = hybrid_solutions[i]
                run_statistics[i] = {"best_cost": hybrid_cost, "best_tour": hybrid_solutions[i], "source": "Hybrid Path", "trigger_step": trigger_steps[i].item(), **dm_proposal_stats[i]}

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
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Solving Batches")):
        td = TensorDict(batch, batch_size=batch['locs'].shape[0]).to(device)
        td['locs'] = td['locs'].float()
        
        solved_tours, batch_stats = solver.solve_batch_hybrid_vs_proposals(td, env)
        
        if "timing_stats" in batch_stats[0]:
            stats = batch_stats[0]["timing_stats"]
            for k, v in stats.items():
                if k != "total_time":
                    cumulative_timing[k] += v
            cumulative_timing["batches_processed"] += 1
        
        all_stats.extend(batch_stats)

        if cfg.solver.get("apply_two_opt", False):
            print("Applying 2-opt post-processing...")
            solved_tours = apply_2opt_batch(solved_tours, td['locs'], max_iterations=2000)
            
        final_costs = calculate_tsp_cost_batch(td['locs'], solved_tours)
        
        # GT computation (assumes the test file contains optimal tours; otherwise for reference only)
        # Adjust here if the npz file does not contain optimal tours
        gt_tour_indices = torch.arange(cfg.model.num_nodes, device=device).unsqueeze(0).repeat(td.shape[0], 1)
        gt_costs = calculate_tsp_cost_batch(td['locs'], gt_tour_indices) # Dummy GT; should be read from the file
        all_gt_costs.append(gt_costs.cpu())        
        
        for i, stat in enumerate(batch_stats):
            stat['final_cost_after_2opt'] = final_costs[i].item()
    
    total_time = time.time() - start_time

    # Statistics
    final_costs_all = [s['final_cost_after_2opt'] for s in all_stats]
    gt_costs_tensor = torch.cat(all_gt_costs)
    final_costs_np = np.array(final_costs_all)
    gt_costs_np = gt_costs_tensor.cpu().numpy()

    # Gap computation
    instance_gaps = ((final_costs_np / gt_costs_np) - 1) * 100
    avg_gap_of_instances = np.mean(instance_gaps)
    std_gap_of_instances = np.std(instance_gaps)
    avg_final_cost = np.mean(final_costs_np)
    avg_GT_cost = np.mean(gt_costs_np)
    
    # ================= Solution source distribution and trigger steps =================
    total_instances = len(all_stats)
    # Extract the source field of all instances
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
    # ==========================================================
    
    print("\n" + "=" * 60)
    print("--- Hybrid Solver Evaluation Summary (Single Run) ---")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average Final Cost: {avg_final_cost:.4f}")
    print(f"Average GT Cost:    {avg_GT_cost:.4f}")
    print(f"Optimality Gap:     {avg_gap_of_instances:.2f}% ± {std_gap_of_instances:.2f}%")
    
    # Print the average trigger step
    if avg_trigger_step != -1:
        print(f"Average Trigger Step: {avg_trigger_step:.2f}")
    else:
        print("Average Trigger Step: N/A (No triggers fired)")

    print("-" * 60)
    print("--- Solution Source Statistics ---")
    print(f"Total Instances: {total_instances}")
    print(f"  - From Diffusion (Triggered & Better): {num_dm} ({pct_dm:.2f}%)")
    print(f"  - From RL/Hybrid (Base Path):          {num_rl} ({pct_rl:.2f}%)")
    print("=" * 60)

    if cumulative_timing["batches_processed"] > 0:
        num_batches = cumulative_timing["batches_processed"]
        print("\n--- PERFORMANCE OVERHEAD ANALYSIS (Avg per Batch) ---")
        print(f"Avg RL Forward Time: {cumulative_timing['rl_forward_time'] / num_batches:.4f}s")
        print(f"Avg DM Probe Time:   {cumulative_timing['dm_probe_time'] / num_batches:.4f}s")
        print(f"Avg DM Sample Time:  {cumulative_timing['dm_sample_time'] / num_batches:.4f}s")
        print("--------------------------------------------------")
        
    # Return the average trigger step
    return avg_gap_of_instances, std_gap_of_instances, avg_final_cost, avg_trigger_step

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid LEHD-DM Solver")
    parser.add_argument(
        "--config", type=str, default="configs/hyco_lehd_tsp500.yaml",
        help="Path to the unified YAML configuration file."
    )
    args = parser.parse_args()
    
    seeds = [42] 
    n_runs = len(seeds)
    all_avg_gaps = []
    all_instance_stds = []
    all_avg_costs = []
    all_avg_trigger_steps = []  # Record the global average trigger step
    
    for i, seed in enumerate(seeds):
        print(f"\n--- RUN {i+1}/{n_runs} (Seed: {seed}) ---")
        set_seed(seed)
        cfg = OmegaConf.load(args.config)

        # Receive the avg_trigger return value
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