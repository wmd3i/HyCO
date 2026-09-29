import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import importlib
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader
from tensordict import TensorDict
import inspect
from collections import defaultdict
import random

# --- RL4CO Imports ---
from rl4co.envs import get_env
from rl4co.utils.ops import unbatchify

# --- Diffusion Model Imports ---
from diffusion_model import ConditionalTSPSuffixDiffusionModel
from discrete_diffusion import AdjacencyMatrixDiffusion

# --- Helper Function Imports ---
from tsp_utils import calculate_tsp_cost_batch, apply_2opt_batch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==========================================
# [Visualization] Oracle Validation Plots
# ==========================================
def plot_oracle_validation(results_df, save_dir="."):
    """
    Plots:
    1. Step Correlation: Adaptive Step vs Best Fixed Step
    2. Cost Gap Distribution: Adaptive Cost - Best Fixed Cost
    
    
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    sns.set_theme(style="whitegrid")
    
    # Filter valid triggers
    df_triggered = results_df[results_df['adaptive_step'] != -1].copy()
    
    # 1. Step Correlation Scatter
    plt.figure(figsize=(8, 8))
    sns.scatterplot(
        data=df_triggered, 
        x='oracle_step', 
        y='adaptive_step', 
        s=100, alpha=0.6, color='crimson', edgecolor='w'
    )
    max_val = max(df_triggered['oracle_step'].max(), df_triggered['adaptive_step'].max()) if len(df_triggered) > 0 else 100
    plt.plot([0, max_val], [0, max_val], 'k--', label='Ideal Match (y=x)')
    plt.title("Adaptive Trigger vs. Optimal Fixed Step")
    plt.xlabel("Oracle Optimal Step (Grid Search)")
    plt.ylabel("Adaptive Trigger Step")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "tsp_oracle_step_correlation.png"), dpi=300)
    plt.close()
    
    # 2. Cost Gap Histogram
    plt.figure(figsize=(10, 6))
    # Gap > 0 means Adaptive is worse; Gap < 0 means Adaptive is better
    results_df['gap'] = results_df['adaptive_cost'] - results_df['oracle_cost']
    sns.histplot(results_df['gap'], kde=True, bins=15, color='teal')
    plt.axvline(0, color='red', linestyle='--', label='Zero Gap')
    plt.title("Performance Gap (Adaptive - Oracle)")
    plt.xlabel("Cost Gap (Lower is Better)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "tsp_oracle_cost_gap.png"), dpi=300)
    plt.close()
    
    print(f"Plots saved to {save_dir}")

class HybridSolver:
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
        self.timing_stats = {k: 0.0 for k in ["rl_forward_time", "dm_probe_time", "dm_sample_time", "dm_decode_time", "total_time"]}

    def _load_rl_policy(self):
        # ... (Same as original) ...
        try:
            ckpt = torch.load(self.cfg.rl_model.ckpt_path, map_location='cpu')
            hparams = ckpt.get('hyper_parameters', ckpt.get('hparams'))
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
            print(f"Error loading RL model: {e}"); exit()

    def _load_dm_model(self):
        # ... (Same as original) ...
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

    def _compute_dm_prior_scores(self, instance_locs, candidate_prefixes, prefix_lengths):
        # ... (Your original _compute_dm_prior_scores logic) ...
        # Copied from your provided code snippet
        total_candidates, N = candidate_prefixes.shape[0], self.cfg.model.num_nodes
        device = self.device
        if total_candidates == 0: return torch.empty(0, device=device)
        t_probe = torch.full((total_candidates,), self.cfg.solver.dm_probe_timestep, device=device, dtype=torch.long)
        prefix_adj_target = torch.zeros(total_candidates, N, N, device=device, dtype=torch.float)
        
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
            x_t_transformed, t_probe.float(), instance_locs, candidate_prefixes, prefix_lengths, node_prefix_state_probe
        )
        loss_mask = prefix_adj_target > 0
        if not loss_mask.any(): return torch.full((total_candidates,), 999.0, device=device)
        
        reconstruction_loss = F.binary_cross_entropy_with_logits(predicted_x_0_logits[loss_mask], prefix_adj_target[loss_mask], reduction='none')
        num_edges_per_prefix = (prefix_lengths - 1).clamp(min=0)
        total_loss_per_candidate = torch.zeros(total_candidates, device=device)
        loss_indices_map = torch.where(loss_mask)
        for i in range(total_candidates):
            total_loss_per_candidate[i] = reconstruction_loss[loss_indices_map[0] == i].sum()
        
        avg_loss = total_loss_per_candidate / (2 * num_edges_per_prefix.clamp(min=1).float())
        avg_loss[num_edges_per_prefix == 0] = 999.0
        return avg_loss

    @torch.no_grad()
    def solve_batch_hybrid_vs_proposals(self, td, env, override_fixed_step=None):
        """
        1. Contains the full adaptive trigger (entropy + KL).
        2. Uses the same execution logic for fixed and adaptive triggers (shared multi-candidate generation), for a fair comparison.
        """
        B, N, _ = td['locs'].shape
        device = self.device
                # --- Helper ---
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
        # 1. Determine the current mode
        if override_fixed_step is not None:
            current_fixed_step = override_fixed_step
        else:
            current_fixed_step = self.cfg.solver.get("fixed_trigger_step", -1)
        
        # ... (initialization) ...
        self.timing_stats = {k: 0.0 for k in self.timing_stats}
        start_total = time.time()
        td_step = env.reset(td.clone())
        hybrid_solutions = torch.zeros(B, N, dtype=torch.long, device=device)
        dm_triggered_flags = torch.zeros(B, dtype=torch.bool, device=device)
        trigger_step_recorded = torch.full((B,), -1, dtype=torch.long, device=device)
        
        dm_proposal_stats = [{
            "cost": torch.tensor(float('inf'), device=device),
            "tour": torch.zeros(N, dtype=torch.long, device=device),
            "generation_step": -1
        } for _ in range(B)]

        node_embeds, _ = self.rl_policy.encoder(td_step)
        cached_embeds = self.rl_policy.decoder._precompute_cache(node_embeds)

        while td_step['i'][0] < N:
            step_idx = td_step['i'].squeeze(-1)
            current_step_scalar = step_idx[0].item()

            # RL Forward
            start_rl = time.time()
            logits, _ = self.rl_policy.decoder(td_step, cached_embeds)
            mask = td_step["action_mask"]
            probs = F.softmax(logits + mask.log(), dim=-1)
            self.timing_stats["rl_forward_time"] += (time.time() - start_rl)

            rl_greedy_choice = probs.argmax(-1)
            best_next_nodes_for_hybrid_path = rl_greedy_choice.clone()

            active_mask = ~td_step["done"].squeeze(-1)
            if not active_mask.any(): break
            
            # ===============================================================
            # [Step A] trigger decision (entropy + KL vs. fixed)
            # ===============================================================
            
            # Select active instances that have not triggered yet
            candidates_mask = active_mask & ~dm_triggered_flags
            indices_to_trigger = torch.empty(0, device=device, dtype=torch.long)
            should_trigger_batch = False

            if candidates_mask.any():
                # --- Branch 1: fixed-step mode (oracle grid) ---
                if current_fixed_step >= 0:
                    if current_step_scalar == current_fixed_step:
                        indices_to_trigger = candidates_mask.nonzero().squeeze(-1)
                        if indices_to_trigger.numel() > 0:
                            should_trigger_batch = True
                
                # --- Branch 2: adaptive mode (entropy + KL) ---
                elif self.cfg.solver.use_theory_trigger:
                    indices_active = candidates_mask.nonzero().squeeze(-1)
                    num_active = indices_active.shape[0]
                    
                    # 1. Prepare the top-M probabilities
                    M = self.cfg.solver.probe_rl_top_m
                    probs_active = probs[indices_active]
                    mask_active = mask[indices_active]
                    actual_m = min(M, int(mask_active.sum(dim=1).min().item()))
                    
                    if actual_m > 0:
                        top_m_probs, top_m_indices = torch.topk(probs_active, k=actual_m, dim=1)
                        
                        # >>> Compute the entropy <<<
                        entropy = -torch.sum(top_m_probs * torch.log(top_m_probs + 1e-9), dim=-1)
                        tqdm.write(f"Step {current_step_scalar}: Entropy Stats - Mean: {entropy.mean().item():.4f}, Max: {entropy.max().item():.4f}")
                        is_high_entropy = entropy > self.cfg.solver.entropy_threshold
                        
                        # >>> Compute the KL divergence <<<
                        is_high_divergence = torch.zeros_like(is_high_entropy, dtype=torch.bool)
                        
                        # Skip the KL computation at step 0 (there is no prefix)
                        if current_step_scalar > 0 and self.cfg.solver.get('use_kl_trigger', True):
                            # Build the prefix input for the DM
                            path_so_far = hybrid_solutions[indices_active, :current_step_scalar]
                            expanded_paths = path_so_far.repeat_interleave(actual_m, dim=0)
                            candidate_nodes = top_m_indices.reshape(-1, 1)
                            
                            prefix_part = torch.cat([expanded_paths, candidate_nodes], dim=1)
                            padding = torch.zeros(prefix_part.shape[0], N - prefix_part.shape[1], dtype=torch.long, device=device)
                            candidate_prefixes = torch.cat([prefix_part, padding], dim=1)
                            
                            prefix_lengths = torch.full((num_active * actual_m,), current_step_scalar + 1, device=device)
                            dm_idx_map = torch.arange(num_active, device=device).repeat_interleave(actual_m)
                            expanded_locs = td['locs'][indices_active][dm_idx_map]
                            
                            # DM Probe
                            start_probe = time.time()
                            dm_scores = self._compute_dm_prior_scores(
                                expanded_locs, candidate_prefixes, prefix_lengths
                            ).view(num_active, actual_m)
                            self.timing_stats["dm_probe_time"] += (time.time() - start_probe)
                            
                            # KL computation
                            log_p_dm = F.log_softmax(-dm_scores / self.cfg.solver.dm_prior_temp, dim=-1)
                            # Renormalize the RL probabilities (only top-M are used)
                            p_rl_norm = top_m_probs / (top_m_probs.sum(dim=-1, keepdim=True) + 1e-9)
                            
                            kl_div = F.kl_div(log_p_dm, p_rl_norm, reduction='none').sum(dim=-1)
                            tqdm.write(f"Step {current_step_scalar}: KL Divergence Stats - Mean: {kl_div.mean().item():.4f}, Max: {kl_div.max().item():.4f}")
                            # Threshold check
                            is_high_divergence = kl_div > self.cfg.solver.kl_div_threshold

                        # >>> Combined trigger condition (OR logic) <<<
                        trigger_mask_local = is_high_entropy | is_high_divergence
                        
                        if trigger_mask_local.any():
                            # Map back to global indices
                            indices_to_trigger = indices_active[trigger_mask_local]
                            should_trigger_batch = True

            # ===============================================================
            # [Step B] shared trigger execution logic (fair comparison)
            # ===============================================================
            
            if should_trigger_batch:
                # 1. Mark
                dm_triggered_flags[indices_to_trigger] = True
                trigger_step_recorded[indices_to_trigger] = current_step_scalar
                
                using_diffusion_mask = indices_to_trigger
                num_uncertain = using_diffusion_mask.numel()
                
                # 2. Prepare candidates (based on the current RL distribution)
                probs_trig = probs[using_diffusion_mask]
                sorted_probs, sorted_indices = torch.sort(probs_trig, dim=-1, descending=True)
                
                # Dynamic N
                cum_probs = torch.cumsum(sorted_probs, dim=-1)
                cum_thresh = self.cfg.solver.dynamic_n_cumulative_threshold
                dyn_n = torch.argmax((cum_probs >= cum_thresh).int(), dim=-1) + 1
                max_n = min(int(dyn_n.max().item()), 32)
                dyn_n = dyn_n.clamp(max=max_n)
                
                proposals = sorted_indices[:, :max_n]
                
                # 3. Prepare the diffusion inputs
                path_so_far = hybrid_solutions[using_diffusion_mask, :current_step_scalar]
                expanded_paths = path_so_far.repeat_interleave(dyn_n, dim=0)
                
                arange_mask = torch.arange(max_n, device=device).unsqueeze(0)
                sel_mask = arange_mask < dyn_n.unsqueeze(1)
                cand_nodes = proposals[sel_mask]
                
                prefix_part = torch.cat([expanded_paths, cand_nodes.unsqueeze(1)], dim=1)
                padding = torch.zeros(prefix_part.shape[0], N - prefix_part.shape[1], dtype=torch.long, device=device)
                final_prefixes = torch.cat([prefix_part, padding], dim=1)
                
                p_lens = torch.full((final_prefixes.shape[0],), current_step_scalar + 1, device=device)
                dm_idx_map = torch.arange(num_uncertain, device=device).repeat_interleave(dyn_n)
                exp_locs = td['locs'][using_diffusion_mask][dm_idx_map]
                
                # Mask State
                node_state = torch.zeros(final_prefixes.shape[0], N, 1, device=device)
                max_l = p_lens.max().item()
                if max_l > 0:
                    p_scatter = final_prefixes[:, :max_l].long().unsqueeze(-1)
                    src = torch.ones_like(p_scatter, dtype=torch.float)
                    l_mask = torch.arange(max_l, device=device).unsqueeze(0) < p_lens.unsqueeze(1)
                    src[~l_mask.unsqueeze(-1)] = 0
                    node_state.scatter_(dim=1, index=p_scatter, src=src)
                
                # 4. Sample
                start_sample = time.time()
                _, gen_adj, _ = self.diffusion_handler.p_sample_loop_ddim(
                    denoiser_model=self.dm_model, instance_locs=exp_locs,
                    prefix_nodes=final_prefixes, prefix_lengths=p_lens,
                    node_prefix_state=node_state,
                    num_inference_steps=self.cfg.solver.dm_inference_steps,
                    schedule=self.cfg.eval.inference_schedule_type
                )
                self.timing_stats["dm_sample_time"] += (time.time() - start_sample)
                
                # 5. Decode
                start_decode = time.time()
                dec_tours, dec_ok = decode_dm_heatmap_simple_greedy_batch(gen_adj, final_prefixes)
                costs = torch.full((final_prefixes.shape[0],), float('inf'), device=device)
                if dec_ok.any():
                    costs[dec_ok] = calculate_tsp_cost_batch(exp_locs[dec_ok], dec_tours[dec_ok])
                self.timing_stats["dm_decode_time"] += (time.time() - start_decode)
                
                # 6. Select Best
                costs_split = torch.split(costs, dyn_n.cpu().tolist())
                tours_split = torch.split(dec_tours, dyn_n.cpu().tolist())
                
                dm_choice_next = torch.zeros(num_uncertain, dtype=torch.long, device=device)
                
                for i in range(num_uncertain):
                    if len(costs_split[i]) == 0: 
                        dm_choice_next[i] = proposals[i, 0]
                        continue
                    
                    best_idx = torch.argmin(costs_split[i])
                    # Next step of the hybrid path
                    dm_choice_next[i] = proposals[i, best_idx]
                    
                    # Update the global best
                    best_cost = costs_split[i][best_idx]
                    orig_idx = using_diffusion_mask[i].item()
                    if not torch.isinf(best_cost) and best_cost < dm_proposal_stats[orig_idx]["cost"]:
                        dm_proposal_stats[orig_idx].update({
                            "cost": best_cost, 
                            "tour": tours_split[i][best_idx],
                            "generation_step": current_step_scalar
                        })
                
                best_next_nodes_for_hybrid_path[using_diffusion_mask] = dm_choice_next

            # Step Update
            hybrid_solutions[torch.arange(B), step_idx] = best_next_nodes_for_hybrid_path
            td_step.set("action", best_next_nodes_for_hybrid_path)
            td_step = env.step(td_step)["next"]

        # Final Stats
        final_hybrid_costs = calculate_tsp_cost_batch(td['locs'], hybrid_solutions)
        final_solutions = torch.zeros(B, N, device=device, dtype=torch.long)
        batch_results = []

        for i in range(B):
            res = {}
            if dm_proposal_stats[i]["cost"] < final_hybrid_costs[i]:
                final_solutions[i] = dm_proposal_stats[i]["tour"]
                res = {"cost": dm_proposal_stats[i]["cost"].item(), "trigger_step": trigger_step_recorded[i].item(), "source": "DM"}
            else:
                final_solutions[i] = hybrid_solutions[i]
                res = {"cost": final_hybrid_costs[i].item(), "trigger_step": trigger_step_recorded[i].item(), "source": "Hybrid"}
            batch_results.append(res)

        self.timing_stats["total_time"] = time.time() - start_total
        return final_solutions, batch_results

# ==============================================================================
# === Oracle Validation Runner ===
# ==============================================================================
def run_oracle_validation(cfg: DictConfig):
    solver = HybridSolver(cfg)
    device = solver.device
    env = get_env(cfg.rl_model.problem, generator_params={"num_loc": cfg.model.num_nodes})
    dataset = env.dataset(filename=cfg.data.test_path)
    
    # 1. Setup Grid for Oracle Search
    # TSP-100: Check trigger at these fixed steps
    grid_steps = [0,1,2,3,4,5, 10, 20, 30, 40] 
    
    # Validation loop (Instance by Instance)
    # Using batch size 1 for clear instance-level comparison
    num_val_samples = 12
     # Limit for speed
    eval_dataset = torch.utils.data.Subset(dataset, range(num_val_samples))
    dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False)
    
    oracle_results = []
    
    print(f"\n>>> Starting Oracle Validation on {num_val_samples} instances <<<")
    
    for i, batch in enumerate(tqdm(dataloader, desc="Oracle Val")):
        td = TensorDict(batch, batch_size=batch['locs'].shape[0]).to(device)
        td['locs'] = td['locs'].float()
        
        # --- A. Run Adaptive Trigger (Yours) ---
        # override_fixed_step=-1 means use the original Adaptive logic
        _, adaptive_stats = solver.solve_batch_hybrid_vs_proposals(td, env, override_fixed_step=-1)
        adaptive_res = adaptive_stats[0] # Single item
        
        adaptive_cost = adaptive_res['cost']
        adaptive_step = adaptive_res['trigger_step']
        
        # --- B. Run Oracle Grid Search ---
        best_fixed_cost = float('inf')
        best_fixed_step = -1
        
        for step in grid_steps:
            _, fixed_stats = solver.solve_batch_hybrid_vs_proposals(td, env, override_fixed_step=step)
            fixed_res = fixed_stats[0]
            
            if fixed_res['cost'] < best_fixed_cost:
                best_fixed_cost = fixed_res['cost']
                best_fixed_step = step
        
        # --- C. Collect Data ---
        oracle_results.append({
            "instance_id": i,
            "adaptive_step": adaptive_step,
            "adaptive_cost": adaptive_cost,
            "best_fixed_step": best_fixed_step,
            "best_fixed_cost": best_fixed_cost,
            "oracle_cost": best_fixed_cost, # Alias for clarity
            "oracle_step": best_fixed_step  # Alias for clarity
        })
        
    # --- D. Analysis & Plotting ---
    df = pd.DataFrame(oracle_results)
    
    # Calculate Gap: (Adaptive - Oracle). 
    # Since Cost is minimized, Gap > 0 means Adaptive is worse. Gap < 0 means Adaptive beat the grid.
    avg_gap = (df['adaptive_cost'] - df['oracle_cost']).mean()
    triggered_count = len(df[df['adaptive_step'] != -1])
    
    print("\n" + "="*60)
    print("--- Oracle Validation Results ---")
    print(f"Total Instances: {len(df)}")
    print(f"Adaptive Triggered: {triggered_count}/{len(df)}")
    print(f"Avg Cost Gap (Adaptive - Oracle): {avg_gap:.4f}")
    if triggered_count > 0:
        # Distance only makes sense if adaptive actually triggered
        avg_dist = (df[df['adaptive_step'] != -1]['adaptive_step'] - df[df['adaptive_step'] != -1]['oracle_step']).abs().mean()
        print(f"Avg Step Distance (|Adaptive - Oracle|): {avg_dist:.2f}")
    print("="*60)
    
    df.to_csv("tsp_oracle_validation.csv", index=False)
    plot_oracle_validation(df, save_dir="oracle_plots_500")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/hyco_tsp500_pomo.yaml")
    args = parser.parse_args()
    
    cfg = OmegaConf.load(args.config)
    
    # Ensure config allows Adaptive logic by default
    default_solver_cfg = OmegaConf.create({
        'solver': {
            'use_theory_trigger': True,
            'fixed_trigger_step': -1 # Default Dynamic
        }
    })
    cfg = OmegaConf.merge(default_solver_cfg, cfg)

    run_oracle_validation(cfg)