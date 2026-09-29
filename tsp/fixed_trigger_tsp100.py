# Fixed-step trigger sweep on TSP-100 (adaptive vs. best fixed trigger).
# Usage: python fixed_trigger_tsp100.py --config configs/hyco_tsp100_pomo.yaml

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
import random

# --- RL4CO Imports ---
from rl4co.envs import get_env
from rl4co.utils.ops import unbatchify

# --- Diffusion Model Imports ---
from diffusion_model import ConditionalTSPSuffixDiffusionModel
from discrete_diffusion import AdjacencyMatrixDiffusion

# --- Helper Function Imports ---
from tsp_utils import calculate_tsp_cost_batch, visualize_tsp_tour, apply_2opt_batch

def set_seed(seed):
    """Sets all relevant random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==========================================
# [NEW] Plotting Function
# ==========================================
def plot_performance_curve(steps, gaps, costs, save_path="fixed_step_ablation.png"):
    """
    Plots the relationship between Fixed Trigger Step and Performance (Gap & Cost).
    
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Optimality Gap (Left Y-Axis)
    color = 'tab:red'
    ax1.set_xlabel('Fixed Trigger Step')
    ax1.set_ylabel('Optimality Gap (%)', color=color)
    ax1.plot(steps, gaps, marker='o', linestyle='-', color=color, linewidth=2, label='Gap')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Plot Average Cost (Right Y-Axis)
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Average Cost', color=color)  
    ax2.plot(steps, costs, marker='s', linestyle='--', color=color, linewidth=2, label='Cost')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Impact of Diffusion Trigger Step on Performance')
    fig.tight_layout()  
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Performance curve saved to {save_path}")


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
        self.timing_stats = {
            "rl_forward_time": 0.0,
            "dm_probe_time": 0.0,
            "dm_sample_time": 0.0,
            "dm_decode_time": 0.0,
            "total_time": 0.0,
        }

    def _load_rl_policy(self):
        print(f"Loading RL model from: {self.cfg.rl_model.ckpt_path}")
        try:
            ckpt = torch.load(self.cfg.rl_model.ckpt_path, map_location='cpu')
            hparams = ckpt.get('hyper_parameters', ckpt.get('hparams'))
            if hparams is None: raise ValueError("No hyperparameters in checkpoint.")
            
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

    # [Note] Included for completeness, but skipped in Fixed Step mode to save time
    def _compute_dm_prior_scores(self, instance_locs, candidate_prefixes, prefix_lengths):
        # ... (Same as your original code) ...
        # For brevity in this response, I'm omitting the body since it's not used in fixed step mode
        pass 

    @torch.no_grad()
    def solve_batch_hybrid_vs_proposals(self, td, env):
        # [MODIFICATION] Check for fixed step configuration
        fixed_step = self.cfg.solver.get("fixed_trigger_step", -1)
        if fixed_step > 0:
            print(f"\n--- Running with FIXED Trigger at Step {fixed_step} ---")
        else:
            print("\n--- Running with Theory-Driven Trigger (Standard) ---")

        B, N, _ = td['locs'].shape
        device = self.device
        
        # --- Helper: Decode DM Heatmap ---
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

        td_step = env.reset(td.clone())
        hybrid_solutions = torch.zeros(B, N, dtype=torch.long, device=device)
        # Ensure instances only trigger once
        dm_triggered_flags = torch.zeros(B, dtype=torch.bool, device=device) 
        
        dm_proposal_stats = [{
            "cost": torch.tensor(float('inf'), device=device),
            "tour": torch.zeros(N, dtype=torch.long, device=device),
            "generation_step": -1, "prefix_node": -1
        } for _ in range(B)]

        node_embeds, _ = self.rl_policy.encoder(td_step)
        cached_embeds = self.rl_policy.decoder._precompute_cache(node_embeds)

        while td_step['i'][0] < N:
            step_idx = td_step['i'].squeeze(-1)
            current_step_scalar = step_idx[0].item()

            # [Timer] RL Forward
            start_rl_forward = time.time()
            logits, _ = self.rl_policy.decoder(td_step, cached_embeds)
            mask = td_step["action_mask"]
            probs = F.softmax(logits + mask.log(), dim=-1)
            self.timing_stats["rl_forward_time"] += (time.time() - start_rl_forward)

            rl_greedy_choice = probs.argmax(-1)
            best_next_nodes_for_hybrid_path = rl_greedy_choice.clone()

            active_mask = ~td_step["done"].squeeze(-1)
            if not active_mask.any(): break
            
            # --- TRIGGER LOGIC START ---
            should_trigger = False
            indices_to_trigger = torch.empty(0, device=device, dtype=torch.long)
            
            # Filter candidates: Must be active AND not yet triggered
            potential_candidates_mask = active_mask & ~dm_triggered_flags

            if potential_candidates_mask.any():
                if fixed_step > 0:
                    # >>> MODE 1: Fixed Step Trigger <<<
                    # Force trigger if we are at the specific step
                    if current_step_scalar == fixed_step:
                        indices_to_trigger = potential_candidates_mask.nonzero().squeeze(-1)
                        should_trigger = True
                        # print(f"--- [Fixed] Triggering at step {current_step_scalar} for {len(indices_to_trigger)} instances ---")
                
                elif self.cfg.solver.use_theory_trigger:
                    # >>> MODE 2: Original Theory-Driven Trigger (Entropy/KL) <<<
                    # [Note] This block is skipped if fixed_step > 0 to ensure pure experimental results
                    # ... (Your original entropy/KL logic here) ...
                    # For this experiment, we assume this branch isn't taken.
                    pass
            
            # --- EXECUTE DIFFUSION IF TRIGGERED ---
            if should_trigger and len(indices_to_trigger) > 0:
                dm_triggered_flags[indices_to_trigger] = True 
                
                using_diffusion_mask = indices_to_trigger
                num_uncertain = using_diffusion_mask.numel()

                # Get Top-N Proposals from RL
                probs_to_trigger = probs[using_diffusion_mask]
                sorted_probs_trigger, sorted_indices_trigger = torch.sort(probs_to_trigger, dim=-1, descending=True)
                cum_probs_trigger = torch.cumsum(sorted_probs_trigger, dim=-1)
                cum_thresh = self.cfg.solver.dynamic_n_cumulative_threshold
                dynamic_n_indices = torch.argmax((cum_probs_trigger >= cum_thresh).int(), dim=-1)
                dynamic_n_candidates = dynamic_n_indices + 1
                
                # Cap max candidates to avoid OOM
                max_n_in_batch = int(dynamic_n_candidates.max().item())
                max_n_in_batch = min(max_n_in_batch, 32) 
                dynamic_n_candidates = dynamic_n_candidates.clamp(max=max_n_in_batch)
                
                proposals = sorted_indices_trigger[:, :max_n_in_batch]
                
                # Prepare DM Inputs
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
                
                # [Timer] DM Sample
                start_dm_sample = time.time()
                _, generated_adj_matrices_probs,_ = self.diffusion_handler.p_sample_loop_ddim(
                    denoiser_model=self.dm_model, instance_locs=expanded_locs_dm,
                    prefix_nodes=final_prefixes, prefix_lengths=prefix_lengths_dm,
                    node_prefix_state=node_prefix_state_dm,
                    num_inference_steps=self.cfg.solver.dm_inference_steps,
                    schedule=self.cfg.eval.inference_schedule_type
                )
                self.timing_stats["dm_sample_time"] += (time.time() - start_dm_sample)
                
                # [Timer] DM Decode
                start_dm_decode = time.time()
                decoded_tours, decoding_ok_mask = decode_dm_heatmap_simple_greedy_batch(generated_adj_matrices_probs, final_prefixes)
                costs = torch.full((final_prefixes.shape[0],), float('inf'), device=device)
                if decoding_ok_mask.any():
                        costs[decoding_ok_mask] = calculate_tsp_cost_batch(expanded_locs_dm[decoding_ok_mask], decoded_tours[decoding_ok_mask])
                self.timing_stats["dm_decode_time"] += (time.time() - start_dm_decode)
                
                # Selection Logic
                costs_split = torch.split(costs, dynamic_n_candidates.cpu().tolist())
                tours_split = torch.split(decoded_tours, dynamic_n_candidates.cpu().tolist())
                
                dm_chosen_nodes = torch.zeros(num_uncertain, dtype=torch.long, device=device)
                for i in range(num_uncertain):
                    if len(costs_split[i]) == 0: 
                        dm_chosen_nodes[i] = proposals[i, 0]
                        continue
                    
                    best_local_idx = torch.argmin(costs_split[i])
                    dm_chosen_nodes[i] = proposals[i, best_local_idx] # Next step according to DM's best path
                    
                    best_dm_cost = costs_split[i][best_local_idx]
                    original_batch_idx = using_diffusion_mask[i].item()
                    
                    # If DM found a better full tour, save it
                    if not torch.isinf(best_dm_cost) and best_dm_cost < dm_proposal_stats[original_batch_idx]["cost"]:
                        dm_proposal_stats[original_batch_idx].update({
                            "cost": best_dm_cost, "tour": tours_split[i][best_local_idx]
                        })
                
                # Execute the DM-chosen next step for the hybrid path
                best_next_nodes_for_hybrid_path[using_diffusion_mask] = dm_chosen_nodes

            # --- TRIGGER LOGIC END ---

            hybrid_solutions[torch.arange(B), step_idx] = best_next_nodes_for_hybrid_path
            td_step.set("action", best_next_nodes_for_hybrid_path)
            td_step = env.step(td_step)["next"]

        # Final Selection
        final_hybrid_costs = calculate_tsp_cost_batch(td['locs'], hybrid_solutions)
        final_solutions = torch.zeros(B, N, device=device, dtype=torch.long)
        run_statistics = [{} for _ in range(B)]

        for i in range(B):
            hybrid_cost = final_hybrid_costs[i]
            proposal_cost = dm_proposal_stats[i]["cost"]
            if proposal_cost < hybrid_cost:
                final_solutions[i] = dm_proposal_stats[i]["tour"]
                run_statistics[i] = {"source": "DM Proposal", **dm_proposal_stats[i]}
            else:
                final_solutions[i] = hybrid_solutions[i]
                run_statistics[i] = {"source": "Hybrid Path", **dm_proposal_stats[i]}

        self.timing_stats["total_time"] = time.time() - start_total
        run_statistics[0]["timing_stats"] = self.timing_stats
        return final_solutions, run_statistics

def run(cfg: DictConfig):
    solver = HybridSolver(cfg)
    device = solver.device
    env = get_env(cfg.rl_model.problem, generator_params={"num_loc": cfg.model.num_nodes})
    dataset = env.dataset(filename=cfg.data.test_path)
    
    # Evaluate a subset for speed
    num_samples_to_evaluate = 1280 
    eval_dataset = torch.utils.data.Subset(dataset, range(num_samples_to_evaluate))
    dataloader = DataLoader(eval_dataset, batch_size=cfg.eval.batch_size, shuffle=False)

    all_stats = []
    all_gt_costs = [] 
    
    step_label = cfg.solver.get('fixed_trigger_step', -1)
    
    for batch in tqdm(dataloader, desc=f"Eval (Step={step_label})"):
        td = TensorDict(batch, batch_size=batch['locs'].shape[0]).to(device)
        td['locs'] = td['locs'].float()
        
        solved_tours, batch_stats = solver.solve_batch_hybrid_vs_proposals(td, env)
        all_stats.extend(batch_stats)

        # Optional 2-opt
        if cfg.solver.get("apply_two_opt", False):
            solved_tours = apply_2opt_batch(solved_tours, td['locs'])
            
        final_costs = calculate_tsp_cost_batch(td['locs'], solved_tours)
        
        # Calculate Gap
        gt_tour_indices = torch.arange(cfg.model.num_nodes, device=device).unsqueeze(0).repeat(td.shape[0], 1)
        gt_costs = calculate_tsp_cost_batch(td['locs'], gt_tour_indices)
        all_gt_costs.append(gt_costs.cpu())        
        
        for i, stat in enumerate(batch_stats):
            stat['final_cost'] = final_costs[i].item()

    final_costs_np = np.array([s['final_cost'] for s in all_stats])
    gt_costs_np = torch.cat(all_gt_costs).cpu().numpy()
    instance_gaps = ((final_costs_np / gt_costs_np) - 1) * 100
    
    avg_gap = np.mean(instance_gaps)
    inst_std = np.std(instance_gaps)
    avg_cost = np.mean(final_costs_np)

    return avg_gap, inst_std, avg_cost

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/hyco_tsp100_pomo.yaml")
    args = parser.parse_args()
    
    # 1. Define the steps to test
    # e.g., Early, Middle, Late, Very Late
    test_trigger_steps = [1,2,3,4, 5, 10, 20, 40] 
    
    results_gap = []
    results_cost = []
    
    seed = 42
    set_seed(seed)
    
    print(f"--- STARTING FIXED STEP EXPERIMENT ---")
    print(f"Steps to test: {test_trigger_steps}")

    for step in test_trigger_steps:
        # 2. Dynamic Configuration Overwrite
        cfg = OmegaConf.load(args.config)
        override_cfg = OmegaConf.create({
            'solver': {
                'fixed_trigger_step': step, # Key Parameter
                'use_theory_trigger': False, # Disable theory trigger to ensure pure fixed step
                'dynamic_n_cumulative_threshold': 0.8,
                'apply_two_opt': False
            }
        })
        cfg = OmegaConf.merge(cfg, override_cfg)
        
        # 3. Run
        avg_gap, _, avg_cost = run(cfg)
        
        results_gap.append(avg_gap)
        results_cost.append(avg_cost)
        print(f"Step {step} Result: Gap={avg_gap:.2f}%, Cost={avg_cost:.4f}")

    # 4. Plot Results
    print("\n--- Experiment Finished. Plotting results... ---")
    plot_performance_curve(test_trigger_steps, results_gap, results_cost, save_path="fixed_step_ablation.png")
    
    print("All Results (Steps):", test_trigger_steps)
    print("All Results (Gaps): ", results_gap)