# HyCO on TSP-100 (RL backbone: POMO).
# Usage: python hyco_tsp100.py --config configs/hyco_tsp100_pomo.yaml
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
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- RL4CO Imports ---
from rl4co.envs import get_env
# from rl4co.models.zoo.am.policy import AttentionModelPolicy # no longer needed
from rl4co.utils.ops import unbatchify

# --- Diffusion Model Imports ---
from diffusion_model import ConditionalTSPSuffixDiffusionModel
from discrete_diffusion import AdjacencyMatrixDiffusion

# --- Helper Function Imports ---
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
    Implements a theoretically-driven hybrid solving approach inspired by
    the concepts of semantic entropy and cognitive divergence.
    """
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Solver using device: {self.device}")

        self.rl_policy = self._load_rl_policy()
        self.dm_model = self._load_dm_model()
        self.kl_history = defaultdict(list)
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
        

    # Add this method to the HybridSolver class, or keep it at module scope
    def plot_kl_curve(self, save_path="kl_vs_step.png"):
        print("Plotting KL vs Step curve...")
        steps = sorted(self.kl_history.keys())
        
        if not steps:
            print("No KL data collected to plot.")
            return

        avg_kl = []
        std_kl = []
        
        # Compute the mean and std at every step
        for step in steps:
            values = np.array(self.kl_history[step])
            avg_kl.append(np.mean(values))
            std_kl.append(np.std(values))
        
        avg_kl = np.array(avg_kl)
        std_kl = np.array(std_kl)
        steps = np.array(steps)

        plt.figure(figsize=(10, 6))
        
        # Plot the main curve
        plt.plot(steps, avg_kl, label='Mean KL Divergence', color='blue', linewidth=2)
        
        # Plot the shaded band (Mean ± Std)
        plt.fill_between(steps, avg_kl - std_kl, avg_kl + std_kl, color='blue', alpha=0.2, label='Standard Deviation')
        
        plt.xlabel('Inference Step (Construction Step)')
        plt.ylabel('KL Divergence (RL vs DM)')
        plt.title('KL Divergence Trend over Construction Steps')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"KL curve saved to {save_path}")

    def _load_rl_policy(self):
        # ... (unchanged)
        print(f"Loading RL model from: {self.cfg.rl_model.ckpt_path}")
        try:
            ckpt = torch.load(self.cfg.rl_model.ckpt_path, map_location='cpu')
            hparams = ckpt.get('hyper_parameters', ckpt.get('hparams'))
            if hparams is None:
                raise ValueError("Could not find hyperparameters in checkpoint.")
            rl_model_cls = getattr(importlib.import_module("rl4co.models.zoo"), self.cfg.rl_model.name)
            valid_args = inspect.signature(rl_model_cls.__init__).parameters
            cleaned_hparams = {k: v for k, v in hparams.items() if k in valid_args}
            cleaned_hparams.pop('env', None)
            env = get_env(self.cfg.rl_model.problem, generator_params={"num_loc": self.cfg.model.num_nodes})
            model = rl_model_cls(env=env, **cleaned_hparams)
            model.load_state_dict(ckpt['state_dict'])
            policy = model.policy.to(self.device)
            policy.eval()
            return policy
        except Exception as e:
            print(f"Error loading RL model: {e}")
            exit()

    def _load_dm_model(self):
        # ... (unchanged)
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
    #  [Core function]
    # ==========================================================================================
    def _compute_dm_prior_scores(self, instance_locs, candidate_prefixes, prefix_lengths):
        """
        Computes a cheap, single-step denoising score ("energy") for a batch of candidate prefixes.
        A lower score (energy) indicates the DM finds the prefix more "plausible" or "self-consistent".
        """
        total_candidates, N = candidate_prefixes.shape[0], self.cfg.model.num_nodes
        device = self.device

        if total_candidates == 0:
            return torch.empty(0, device=device)

        t_probe = torch.full((total_candidates,), self.cfg.solver.dm_probe_timestep, device=device, dtype=torch.long)

        prefix_adj_target = torch.zeros(total_candidates, N, N, device=device, dtype=torch.float)
        valid_prefix_mask = prefix_lengths > 1
        
        if valid_prefix_mask.any():
            # Build the adjacency matrix in a safer way
            for i in range(total_candidates):
                if prefix_lengths[i] > 1:
                    p_nodes = candidate_prefixes[i, :prefix_lengths[i]]
                    prefix_adj_target[i, p_nodes[:-1], p_nodes[1:]] = 1.0
                    prefix_adj_target[i, p_nodes[1:], p_nodes[:-1]] = 1.0

        x_t, _ = self.diffusion_handler.q_sample(prefix_adj_target, t_probe)
        x_t_transformed = x_t.float() * 2.0 - 1.0

        # [Bug fix 2]: build node_prefix_state with a more robust scatter_ to avoid indexing errors
        node_prefix_state_probe = torch.zeros(total_candidates, N, 1, device=device)
        max_len = prefix_lengths.max().item()
        if max_len > 0:
            # .clone() avoids in-place operation issues
            prefixes_for_scatter = candidate_prefixes[:, :max_len].long().clone().unsqueeze(-1)
            # Prepare src with the same shape as index
            src = torch.ones_like(prefixes_for_scatter, dtype=torch.float)
            # Build a mask to ignore padding values (padding is assumed to be 0)
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
        loss_idx_for_scatter = loss_indices_map[0]
        # Sum efficiently with scatter_add_. Note: may need minor adjustments depending on the PyTorch version
        # A simpler and clearer approach is a loop
        for i in range(total_candidates):
            total_loss_per_candidate[i] = reconstruction_loss[loss_indices_map[0] == i].sum()
        
        avg_loss_per_candidate = total_loss_per_candidate / (2 * num_edges_per_prefix.clamp(min=1).float())
        avg_loss_per_candidate[num_edges_per_prefix == 0] = 999.0
        
        return avg_loss_per_candidate
    
    @torch.no_grad()
    def run_kl_variance_ablation(self, td, env, save_dir="./ablation_plots"):
        """
        Ablation: variance and distribution of the KL divergence
        under different DM probe timesteps.
        """
        
        
        os.makedirs(save_dir, exist_ok=True)
        print("\n--- Starting KL Variance Ablation Study for Rebuttal ---")
        
        B, N, _ = td['locs'].shape
        device = self.device
        td_step = env.reset(td.clone())
        
        # Timesteps to test
        test_timesteps = [50, 200, 500, 800, 950]
        kl_records = [] # Stores data for plotting
        
        node_embeds, _ = self.rl_policy.encoder(td_step)
        cached_embeds = self.rl_policy.decoder._precompute_cache(node_embeds)

        # To collect data quickly, only run the first 10 construction steps (where disagreement is most likely)
        max_steps_to_probe = 10 
        
        while td_step['i'][0] < max_steps_to_probe:
            step_idx = td_step['i'].squeeze(-1)
            current_step = step_idx[0].item()
            
            logits, _ = self.rl_policy.decoder(td_step, cached_embeds)
            mask = td_step["action_mask"]
            probs = F.softmax(logits + mask.log(), dim=-1)
            
            # Force probing for all instances without entropy filtering, to get the raw KL distribution
            active_mask = ~td_step["done"].squeeze(-1)
            indices_to_probe = active_mask.nonzero().squeeze(-1)
            num_to_probe = len(indices_to_probe)
            
            if num_to_probe == 0: break
                
            M = self.cfg.solver.probe_rl_top_m
            num_available = int(mask[active_mask].sum(dim=1).min().item())
            M = min(M, num_available)
            
            top_m_probs, top_m_indices = torch.topk(probs[active_mask], k=M, dim=1)
            
            # Fake path_so_far (we only care about the probe at this step; a greedy path is sufficient)
            # In a real run you would use the actual hybrid_solutions; simplified here to isolate the experiment
            greedy_path = probs.argmax(-1).unsqueeze(1).repeat(1, current_step) if current_step > 0 else torch.empty(B, 0, dtype=torch.long, device=device)
            path_so_far = greedy_path[active_mask]
            
            expanded_paths = path_so_far.repeat_interleave(M, dim=0)
            candidate_nodes = top_m_indices.reshape(-1, 1)

            prefix_part = torch.cat([expanded_paths, candidate_nodes], dim=1)
            padding = torch.zeros(prefix_part.shape[0], N - prefix_part.shape[1], dtype=torch.long, device=device)
            candidate_prefixes = torch.cat([prefix_part, padding], dim=1)
            prefix_lengths = torch.full((num_to_probe * M,), current_step + 1, device=device)

            dm_to_instance_idx = torch.arange(num_to_probe, device=device).repeat_interleave(M)
            expanded_locs = td['locs'][indices_to_probe][dm_to_instance_idx]
            
            # ========================================================
            # Core ablation: score the same RL proposal with different t
            # ========================================================
            for t_val in test_timesteps:
                # Temporarily override the timestep in the config
                original_t = self.cfg.solver.dm_probe_timestep
                self.cfg.solver.dm_probe_timestep = t_val
                
                dm_scores = self._compute_dm_prior_scores(expanded_locs, candidate_prefixes, prefix_lengths).view(num_to_probe, M)
                
                # Compute the KL divergence
                log_p_dm = F.log_softmax(-dm_scores / self.cfg.solver.dm_prior_temp, dim=-1)
                kl_divergence = F.kl_div(log_p_dm, top_m_probs, reduction='none').sum(dim=-1)
                
                # Record data
                for kl_val in kl_divergence.detach().cpu().numpy():
                    kl_records.append({
                        "Timestep": t_val,
                        "KL_Divergence": kl_val,
                        "Construction_Step": current_step
                    })
                
                # Restore the config
                self.cfg.solver.dm_probe_timestep = original_t

            # Step the RL environment (take one greedy step so the environment keeps going)
            td_step.set("action", probs.argmax(-1))
            td_step = env.step(td_step)["next"]

        # --- Plot (violin plot) ---
        df = pd.DataFrame(kl_records)
        plt.figure(figsize=(10, 6))
        # A violin plot best shows the sharpness and volatility of the distribution
        sns.violinplot(x="Timestep", y="KL_Divergence", data=df, inner="quartile", palette="muted")
        plt.title("Distribution of KL Divergence Trigger vs. DM Probe Timestep (t)", fontsize=14)
        plt.xlabel("Diffusion Probe Timestep (t)", fontsize=12)
        plt.ylabel("KL Divergence (RL vs DM)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        
        save_file = os.path.join(save_dir, "kl_variance_ablation.png")
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"--- Ablation Plot saved to {save_file} ---")
        return df
    
    # ==========================================================================================
    #  [Main solve function]
    # ==========================================================================================
    @torch.no_grad()
    def solve_batch_hybrid_vs_proposals(self, td, env):
        print("\n--- Running with SEAT-inspired, theory-driven trigger [FIXED] ---")
        B, N, _ = td['locs'].shape
        device = self.device
        
        # --- Helper functions ---
        def decode_dm_heatmap_simple_greedy_batch(adj_matrices_probs, batch_prefix_nodes):
            B_decode, N_decode, _ = adj_matrices_probs.shape
            final_tours = torch.full((B_decode, N_decode), -1, dtype=torch.long, device=device)
            visited_mask = torch.zeros((B_decode, N_decode), dtype=torch.bool, device=device)
            
            # Use the first node of the prefix as the start node
            # Note: batch_prefix_nodes must contain at least one node
            current_nodes = batch_prefix_nodes[:, 0]
            final_tours[:, 0] = current_nodes
            visited_mask.scatter_(1, current_nodes.unsqueeze(1), True)
        
            for i in range(1, N_decode):
                step_probs = adj_matrices_probs.clone()
                # Mask out already visited nodes
                current_node_mask = visited_mask.unsqueeze(1).expand(-1, N_decode, -1)
                step_probs.masked_fill_(current_node_mask, -1e9)
                
                next_node_probs = step_probs.gather(1, current_nodes.view(-1, 1, 1).expand(-1, -1, N_decode)).squeeze(1)
                next_nodes = torch.argmax(next_node_probs, dim=1)
                
                final_tours[:, i] = next_nodes
                visited_mask.scatter_(1, next_nodes.unsqueeze(1), True)
                current_nodes = next_nodes
                
            decoding_ok_mask = (final_tours != -1).all(dim=1)
            return final_tours, decoding_ok_mask

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
                if not next_node_found or next_node in visited_nodes:
                    return []
                tour.append(next_node)
                visited_nodes.add(next_node)
                prev_node = curr_node
                curr_node = next_node
            return tour
        

        self.timing_stats = {k: 0.0 for k in self.timing_stats}
        start_total = time.time()
        # --- Initialization ---
        td_step = env.reset(td.clone())
        hybrid_solutions = torch.zeros(B, N, dtype=torch.long, device=device)
        dm_triggered_flags = torch.zeros(B, dtype=torch.bool, device=device)
        
        dm_proposal_stats = [{
            "cost": torch.tensor(float('inf'), device=device),
            "tour": torch.zeros(N, dtype=torch.long, device=device),
            "generation_step": -1, "prefix_node": -1, "rl_greedy_node_at_step": -1, "candidates_for_the_step": None
        } for _ in range(B)]

        node_embeds, _ = self.rl_policy.encoder(td_step)
        cached_embeds = self.rl_policy.decoder._precompute_cache(node_embeds)

        # --- Main loop ---
        while td_step['i'][0] < N:
            step_idx = td_step['i'].squeeze(-1)
            current_step_scalar = step_idx[0].item()
# [Timer start] RL forward
            start_rl_forward = time.time()
            logits, _ = self.rl_policy.decoder(td_step, cached_embeds)
            mask = td_step["action_mask"]
            probs = F.softmax(logits + mask.log(), dim=-1)
            self.timing_stats["rl_forward_time"] += (time.time() - start_rl_forward)
            # [Timer end] RL forward

            rl_greedy_choice = probs.argmax(-1)
            best_next_nodes_for_hybrid_path = rl_greedy_choice.clone()

            active_mask = ~td_step["done"].squeeze(-1)
            if not active_mask.any(): break
            
            probe_mask = active_mask & ~dm_triggered_flags
            #print(f"[DEBUG] entering loop, i={current_step_scalar}")

         
            if probe_mask.any() and self.cfg.solver.use_theory_trigger:
 
                indices_to_probe = probe_mask.nonzero().squeeze(-1)
                num_to_probe = len(indices_to_probe)
                
                M = self.cfg.solver.probe_rl_top_m
                probs_rl_probe = probs[probe_mask]
                num_available_actions = int(mask[probe_mask].sum(dim=1).min().item())
                M = min(M, num_available_actions)

                if M > 0:
                    top_m_probs, top_m_indices = torch.topk(probs_rl_probe, k=M, dim=1)
                    
                    # Compute the policy entropy
                    entropy_rl = -torch.sum(top_m_probs * torch.log(top_m_probs + 1e-9), dim=-1)
                    is_high_entropy = entropy_rl > entropy_rl.mean().item()-0.1#self.cfg.solver.entropy_threshold
                    
                    # [Bug fix 2]: distinguish the trigger logic of step 0 from later steps
                    if current_step_scalar == 0:
                        # At step 0 the KL divergence is meaningless, so we only rely on the policy entropy
                        print(f"[DEBUG] step 0, Entropy={entropy_rl.mean().item():.3f}. TOP-M probs = {top_m_probs} ; KL divergence is skipped.")
                        trigger_now_mask_relative = is_high_entropy
                    else:
                        # At later steps we use both entropy and KL divergence
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

                        # [Bug fix 1]: use the numerically stable F.kl_div instead of a manual computation
                        # PyTorch's kl_div expects log-probabilities as input
                        log_p_dm = F.log_softmax(-dm_scores / self.cfg.solver.dm_prior_temp, dim=-1)
                        p_rl = top_m_probs
                        
                        # F.kl_div(q.log(), p) computes D_KL(p || q), where p and q are probabilities
                        # reduction='none' lets us sum per instance manually
                        kl_divergence = F.kl_div(log_p_dm, p_rl, reduction='none').sum(dim=-1)

                        # --- Collect KL data ---
                        # Convert the tensor to a numpy list and store it under the current step
                        # Only collect the KL of instances probed at this step
                        kl_values_np = kl_divergence.detach().cpu().numpy().tolist()
                        self.kl_history[current_step_scalar].extend(kl_values_np)
                        # -------------------------

                        print(f"[DEBUG] step {current_step_scalar}, Entropy={entropy_rl.mean().item():.3f}, KL={kl_divergence.mean().item():.3f}")
                        
                        is_high_divergence = kl_divergence > kl_divergence.mean().item()-0.3#-0.2 self.cfg.solver.kl_div_threshold
                        trigger_now_mask_relative = is_high_entropy | is_high_divergence
                
                        print(f"[DEBUG] step {current_step_scalar}, TOP-M probs = {top_m_probs} entropy={entropy_rl.mean():.3f}, KL={kl_divergence.mean():.3f}")   



                    
                    if trigger_now_mask_relative.any():
                        absolute_trigger_indices = indices_to_probe[trigger_now_mask_relative]
                        print(f"--- Step {current_step_scalar}: Theory-trigger fired for {len(absolute_trigger_indices)} instances. ---")
                        dm_triggered_flags[absolute_trigger_indices] = True
                        
                        # --- DM invocation ---
                        using_diffusion_mask = absolute_trigger_indices
                        num_uncertain = using_diffusion_mask.numel()
                        
                        probs_to_trigger = probs[using_diffusion_mask]
                        sorted_probs_trigger, sorted_indices_trigger = torch.sort(probs_to_trigger, dim=-1, descending=True)
                        cum_probs_trigger = torch.cumsum(sorted_probs_trigger, dim=-1)
                        cum_thresh = self.cfg.solver.dynamic_n_cumulative_threshold
                        dynamic_n_indices = torch.argmax((cum_probs_trigger >= cum_thresh).int(), dim=-1)
                        dynamic_n_candidates = dynamic_n_indices + 1
                        max_n_in_batch = int(dynamic_n_candidates.max().item())
                        
                        proposals = sorted_indices_trigger[:, :max_n_in_batch]
                        
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

                        # Call the decoding function
                        decoded_tours, decoding_ok_mask = decode_dm_heatmap_simple_greedy_batch(generated_adj_matrices_probs, final_prefixes)
                        costs = torch.full((final_prefixes.shape[0],), float('inf'), device=device)
                        if decoding_ok_mask.any():
                             costs[decoding_ok_mask] = calculate_tsp_cost_batch(expanded_locs_dm[decoding_ok_mask], decoded_tours[decoding_ok_mask])
                        self.timing_stats["dm_decode_time"] += (time.time() - start_dm_decode)
                        # [Timer end] DM decode

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
                                dm_proposal_stats[original_batch_idx].update({
                                    "cost": best_dm_cost, "tour": tours_split[i][best_local_idx],
                                    "generation_step": current_step_scalar, "prefix_node": proposals[i, best_local_idx].item(),
                                    "rl_greedy_node_at_step": rl_greedy_choice[original_batch_idx].item(),
                                    "candidates_for_the_step": proposals[i].cpu().numpy()
                                })
                        
                        best_next_nodes_for_hybrid_path[using_diffusion_mask] = dm_chosen_nodes
            
            hybrid_solutions[torch.arange(B), step_idx] = best_next_nodes_for_hybrid_path
            td_step.set("action", best_next_nodes_for_hybrid_path)
            td_step = env.step(td_step)["next"]

        # --- Final Selection and Statistics Logging ---
        final_hybrid_costs = calculate_tsp_cost_batch(td['locs'], hybrid_solutions)
        final_solutions = torch.zeros(B, N, device=device, dtype=torch.long)
        run_statistics = [{} for _ in range(B)]

        for i in range(B):
            hybrid_cost = final_hybrid_costs[i]
            proposal_cost = dm_proposal_stats[i]["cost"]
            if proposal_cost < hybrid_cost:
                final_solutions[i] = dm_proposal_stats[i]["tour"]
                run_statistics[i] = {"best_cost": proposal_cost, "best_tour": dm_proposal_stats[i]["tour"], "source": "DM Proposal", **dm_proposal_stats[i]}
            else:
                final_solutions[i] = hybrid_solutions[i]
                run_statistics[i] = {"best_cost": hybrid_cost, "best_tour": hybrid_solutions[i], "source": "Hybrid Path", **dm_proposal_stats[i]}


        self.timing_stats["total_time"] = time.time() - start_total
        print(f"Total loop time (includes final selection): {self.timing_stats['total_time']:.3f}s")
        print("--- Hybrid-vs-Proposals run finished. Final selection complete. ---")
        
        # Attach timing statistics to the first element of run_statistics so run() can access them
        run_statistics[0]["timing_stats"] = self.timing_stats
        return final_solutions, run_statistics

def run(cfg: DictConfig):
    solver = HybridSolver(cfg)
    device = solver.device
    env = get_env(cfg.rl_model.problem, generator_params={"num_loc": cfg.model.num_nodes})
    # Test data path
    dataset = env.dataset(filename=cfg.data.test_path)
    num_samples_to_evaluate = 1280
    eval_dataset = torch.utils.data.Subset(dataset, range(num_samples_to_evaluate))
    dataloader = DataLoader(eval_dataset, batch_size=cfg.eval.batch_size, shuffle=False)

    all_stats = []
    all_gt_costs = [] # List storing the GT cost of every batch
    # Accumulated timing statistics
    cumulative_timing = {
        "rl_forward_time": 0.0,
        "dm_probe_time": 0.0,
        "dm_sample_time": 0.0,
        "dm_decode_time": 0.0,
        "total_time": 0.0,
        "batches_processed": 0
    }
    
    start_time = time.time()
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Solving Batches")):
        td = TensorDict(batch, batch_size=batch['locs'].shape[0]).to(device)
        td['locs'] = td['locs'].float()
        
        solved_tours, batch_stats = solver.solve_batch_hybrid_vs_proposals(td, env)
        #all_stats.extend(batch_stats)

        # Collect timing statistics
        if "timing_stats" in batch_stats[0]:
            stats = batch_stats[0]["timing_stats"]
            for k, v in stats.items():
                if k != "total_time": # Ignore the per-batch total_time
                    cumulative_timing[k] += v
            cumulative_timing["batches_processed"] += 1
        
        all_stats.extend(batch_stats)

        if cfg.solver.get("apply_two_opt", True):
            print("Applying 2-opt post-processing...")
            solved_tours = apply_2opt_batch(solved_tours, td['locs'])
            
        final_costs = calculate_tsp_cost_batch(td['locs'], solved_tours)
        # Compute and store the ground-truth (ordered) cost
        gt_tour_indices = torch.arange(cfg.model.num_nodes, device=device).unsqueeze(0).repeat(td.shape[0], 1)
        gt_costs = calculate_tsp_cost_batch(td['locs'], gt_tour_indices)
        all_gt_costs.append(gt_costs.cpu())        
        # Update the final cost in batch_stats
        for i, stat in enumerate(batch_stats):
            stat['final_cost_after_2opt'] = final_costs[i].item()
    # --- After the loop, plot ---
    # The solver instance must be the same so that its history is preserved
    # In a multi-seed loop you may want to add a seed suffix to the file name
    solver.plot_kl_curve(save_path=f"./klvsstep/kl_vs_step_seed_{cfg.seed if 'seed' in cfg else 'default'}.png")
    # --------------------------------
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
# Print timing statistics
    if cumulative_timing["batches_processed"] > 0:
        num_batches = cumulative_timing["batches_processed"]
        avg_probe_time = cumulative_timing["dm_probe_time"] / num_batches
        avg_sample_time = cumulative_timing["dm_sample_time"] / num_batches
        avg_rl_time = cumulative_timing["rl_forward_time"] / num_batches
        avg_decode_time = cumulative_timing["dm_decode_time"] / num_batches

        print("\n--- PERFORMANCE OVERHEAD ANALYSIS (Avg per Batch) ---")
        print(f"Processed Batches: {num_batches}")
        print(f"Avg RL Forward Time: {avg_rl_time:.4f}s")
        print(f"Avg DM Probe Time:   {avg_probe_time:.4f}s (Overhead step)")
        print(f"Avg DM Sample Time:  {avg_sample_time:.4f}s (Expensive step)")
        print(f"Avg DM Decode Time:  {avg_decode_time:.4f}s")
        print(f"Ratio (Sample/Probe): {avg_sample_time / (avg_probe_time + 1e-9):.2f}x (Shows the saving potential)")
        print("--------------------------------------------------")
        
    print("\n" + "=" * 60)
    # Return for this run: 1. mean gap, 2. instance STD, 3. mean cost
    return avg_gap_of_instances, std_gap_of_instances, avg_final_cost

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid RL-DM Solver with Theory-Driven Trigger")
    parser.add_argument(
        "--config", type=str, default="configs/hyco_tsp100_pomo.yaml",
        help="Path to the unified YAML configuration file."
    )
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
                'probe_rl_top_m': 5,
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
    