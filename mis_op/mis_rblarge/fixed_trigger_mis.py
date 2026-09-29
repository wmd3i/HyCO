# Fixed-step trigger sweep for MIS (adaptive vs. fixed trigger).
# Usage: python fixed_trigger_mis.py --config hyco_config.yaml --mode fixed_exp
import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
import time
import argparse
import scipy.sparse as sp
from tqdm.auto import tqdm
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader
from torch.distributions import Categorical

# --- Plotting ---
import matplotlib.pyplot as plt
import seaborn as sns

# --- Path setup ---
root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_folder not in sys.path:
    sys.path.append(root_folder)
base_dir = os.path.dirname(os.path.abspath(__file__))
co_expander_path = os.path.abspath(os.path.join(base_dir, '..'))
lwd_path = os.path.abspath(os.path.join(base_dir, 'learning_what_to_defer'))
if co_expander_path not in sys.path:
    sys.path.append(co_expander_path)
if lwd_path not in sys.path:
    sys.path.append(lwd_path)

import dgl

# --- RL (LwD) Imports ---
from learning_what_to_defer.data.graph_dataset import get_rb_small_dataset
from learning_what_to_defer.ppo.actor_critic import ActorCritic
from learning_what_to_defer.env import MaximumIndependentSetEnv 
from learning_what_to_defer.ppo.graph_net import PolicyGraphConvNet, ValueGraphConvNet

# --- CO-Expander Imports ---
from co_expander import (
    COExpanderCMModel, GNNEncoder, COExpanderEnv,
    COExpanderDecoder, COExpanderSparser
)
from ml4co_kit import MISGraphData

# ==============================================================================
# === Plotting function ===
# ==============================================================================
def plot_fixed_step_performance(steps, mis_sizes, save_path="fixed_step_performance_mis.png"):
    """
    Plot fixed trigger step vs. average MIS size
    """
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    plt.plot(steps, mis_sizes, marker='o', linestyle='-', linewidth=2, color='purple', label='Hybrid Solver MIS Size')
    
    # Annotate values
    for x, y in zip(steps, mis_sizes):
        plt.text(x, y, f'{y:.2f}', ha='center', va='bottom', fontsize=10)

    plt.title('Impact of Fixed Trigger Step on MIS Size', fontsize=16)
    plt.xlabel('Fixed Trigger Step (RL Steps)', fontsize=12)
    plt.ylabel('Average MIS Size', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"\n[Plot] Performance curve saved to: {save_path}")

def _get_lwd_dataloader(cfg: DictConfig, device: torch.device):
    test_data_path = "../data/lwd/rb_large/test" 
    # print(f"Loading LwD native dataset from: {test_data_path}")
    test_dataset = get_rb_small_dataset("test", test_data_path)
    def collate_fn(graphs):
        return dgl.batch(graphs)
    dataloader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0
    )
    # print(f"Found {len(test_dataset)} instances.")
    return dataloader

class HybridSolverMIS:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Solver using device: {self.device}")

        # 1. RL Policy
        self.rl_policy = self._load_lwd_rl_policy()
        max_nodes = 1200
        self.rl_env = MaximumIndependentSetEnv(
            max_epi_t=128, max_num_nodes=max_nodes,
            hamming_reward_coef=0.1, device=self.device
        )
        self.rl_eval_samples = self.cfg.rl_model.get('eval_num_samples', 1)

        # 2. CO-Expander Model
        self.co_expander_model, self.co_expander_cfg = self._load_co_expander_model()

    def _load_lwd_rl_policy(self):
        # ... (unchanged) ...
        # print(f"Loading LwD RL model from: {self.cfg.rl_model.ckpt_path}")
        n_layers = self.cfg.rl_model.n_layers
        hidden_dim = self.cfg.rl_model.hidden_dim
        max_nodes_from_training = 1200
        policy = ActorCritic(
            actor_class=PolicyGraphConvNet, critic_class=ValueGraphConvNet,
            max_num_nodes=max_nodes_from_training, hidden_dim=hidden_dim,
            num_layers=n_layers, device=self.device
        ).to(self.device)  
        checkpoint = torch.load(self.cfg.rl_model.ckpt_path, map_location=self.device)
        policy.load_state_dict(checkpoint['model_state_dict'])
        policy.eval()
        return policy

    def _load_co_expander_model(self):
        # ... (unchanged) ...
        # print(f"Loading CO-Expander model from: {self.cfg.co_expander_model.ckpt_path}")
        ce_train_cfg = OmegaConf.load(self.cfg.co_expander_model.config_path)
        ce_env = COExpanderEnv(
            task=ce_train_cfg.task, val_path=ce_train_cfg.data.test_path,
            val_data_size=ce_train_cfg.inference.num_test_samples,
            val_batch_size=ce_train_cfg.inference.batch_size,
            sparse_factor=ce_train_cfg.sparse_factor, device=self.device,
            prefix_k_options=ce_train_cfg.inference.prefix_k_options
        )
        if hasattr(ce_env, 'data_processor') and isinstance(ce_env.data_processor, COExpanderSparser):
            ce_env.data_processor.device = self.device
        encoder = GNNEncoder(**ce_train_cfg.encoder)
        decoder = COExpanderDecoder(**ce_train_cfg.decoder)
        model = COExpanderCMModel(
            env=ce_env, encoder=encoder, decoder=decoder,
            learning_rate=ce_train_cfg.train.learning_rate, **ce_train_cfg.model_params
        )
        checkpoint = torch.load(self.cfg.co_expander_model.ckpt_path, map_location=self.device)
        if 'state_dict' in checkpoint: model.load_state_dict(checkpoint['state_dict'])
        else: model.load_state_dict(checkpoint)
        model.eval(); model.to(self.device); model.env = ce_env 
        return model, ce_train_cfg

    @torch.no_grad()
    def _run_co_expander_probe(self, original_g, current_rl_x):
        
        # _run_co_expander_probe goes here
        N = original_g.number_of_nodes()
        if current_rl_x.dim() > 1: states = current_rl_x[:, 0]
        else: states = current_rl_x
        prefix_nodes_tensor = (states == 1.0).nonzero().squeeze(-1)
        ce_inputs = self._prepare_co_expander_input_tuple(original_g, prefix_nodes_tensor)
        (task, nodes_feature, x, edges_feature, e, edge_index, graph_list, 
         mask, ground_truth, nodes_num_list, edges_num_list, raw_data_list,
         prefix_nodes_batch_list, node_prefix_state) = ce_inputs
        x_input = (torch.randn(N, device=self.device) > 0).float()
        current_mask = torch.zeros(N, dtype=torch.bool, device=self.device)
        if prefix_nodes_tensor.numel() > 0:
            current_mask[prefix_nodes_tensor] = True
            x_input[prefix_nodes_tensor] = 1.0
        x_input_masked = torch.where(current_mask, x_input - 0.5, x_input - 0.5)
        probe_t_val = self.cfg.solver.get('probe_noise_t', 0)
        t = torch.tensor([probe_t_val], device=self.device).float()
        x_pred_logits, _ = self.co_expander_model.model.forward(
            task=task, focus_on_node=True, focus_on_edge=False,
            nodes_feature=nodes_feature, x=x_input_masked,
            edges_feature=edges_feature, e=e, mask=current_mask, 
            t=t, edge_index=edge_index, node_prefix_state=node_prefix_state,
            prefix_nodes=prefix_nodes_batch_list, nodes_num_list=nodes_num_list
        )
        return x_pred_logits

    def _compute_kl_divergence(self, rl_probs, ce_logits, active_indices):
        # ... (unchanged, omitted for brevity) ...
        ce_l_active = ce_logits[active_indices]
        ce_probs = torch.softmax(ce_l_active, dim=-1)
        p_ce_in = ce_probs[:, 1]
        p_rl_in = rl_probs[:, 1]
        eps = 1e-6
        p_ce_in = torch.clamp(p_ce_in, eps, 1-eps)
        p_rl_in = torch.clamp(p_rl_in, eps, 1-eps)
        kl_values = (p_ce_in * torch.log(p_ce_in / p_rl_in) + 
                     (1 - p_ce_in) * torch.log((1 - p_ce_in) / (1 - p_rl_in)))
        return kl_values.sum()

    # ==========================================================================
    # === [Key change] solve instance with fixed-step support ===
    # ==========================================================================
    @torch.no_grad()
    def solve_instance(self, g):
        original_g = g.to(self.device) 
        rl_state = self.rl_env.register(g, num_samples=self.rl_eval_samples) 

        rl_t = 0
        done = False
        ce_triggered_flag = False
        best_ce_mis_size = -1
        ce_solution_mask = np.array([])
        trigger_step_at = -1 
        trigger_reason = "None"

        # [NEW] get the fixed-step config
        fixed_step_trigger = getattr(self.cfg.solver, 'fixed_trigger_step', -1)

        best_rl_sol_tensor_so_far = torch.zeros((1, self.rl_eval_samples), device=self.device)
        best_rl_state_so_far = rl_state.clone()

        while not done:
            current_rl_state_clone = rl_state.clone()
            num_nodes, batch_size = rl_state.size(0), rl_state.size(1)

            # 1. RL Forward
            masks, idxs, subg, h = self.rl_policy.get_masks_idxs_subg_h(rl_state, g)
            node_mask, subg_mask, subg_node_mask = masks
            flatten_node_idxs, flatten_subg_idxs, flatten_subg_node_idxs = idxs

            logits = (
                self.rl_policy.actor_net(h, subg, mask = subg_node_mask)
                .view(-1, 3)
                .index_select(0, flatten_subg_node_idxs)
            )
            rl_dist = Categorical(logits = logits)
            sampled_actions = rl_dist.sample() 

            rl_action_flat = torch.zeros(num_nodes * batch_size, dtype=torch.long, device=self.device)   
            rl_action_flat[flatten_node_idxs] = sampled_actions
            rl_action = rl_action_flat.view(-1, batch_size)
            
            # ==================================================================
            # === Trigger Logic (Fixed vs Dynamic) ===
            # ==================================================================
            trigger_now = False

            # >>> Branching logic <<<
            if fixed_step_trigger >= 0:
                # [Mode 1: Fixed Step]
                # Note: in the MIS environment rl_t increases with the number of decisions.
                # Trigger if the current step equals the configured step and the trigger has not fired yet.
                if rl_t == fixed_step_trigger and not ce_triggered_flag:
                    trigger_now = True
                    trigger_reason = f"FixedStep({fixed_step_trigger})"
            else:
                # [Mode 2: Dynamic Trigger (Entropy/KL)]
                # Only compute KL/entropy in dynamic mode to save compute
                if self.cfg.solver.use_theory_trigger and not ce_triggered_flag:
                    entropy_val = torch.tensor(0.0)
                    kl_val = torch.tensor(0.0)
                    
                    # A. Entropy
                    entropies = rl_dist.entropy()
                    M = min(self.cfg.solver.probe_rl_top_m, len(entropies))
                    if M > 0:
                        top_m_entropies, _ = torch.topk(entropies, k=M)
                        entropy_val = torch.mean(top_m_entropies)
                        is_high_entropy = entropy_val > self.cfg.solver.entropy_threshold
                        
                        # B. KL Divergence
                        is_high_kl = False
                        if self.cfg.solver.get('use_kl_trigger', True):
                            ce_logits = self._run_co_expander_probe(original_g, self.rl_env.x)
                            rl_probs_all = torch.softmax(logits, dim=-1)
                            active_global_indices = flatten_node_idxs
                            total_kl = self._compute_kl_divergence(
                                rl_probs_all, ce_logits, active_global_indices
                            )
                            avg_kl = total_kl / (len(active_global_indices) + 1e-9)
                            kl_val = avg_kl
                            is_high_kl = avg_kl > self.cfg.solver.get('kl_threshold', 0.5)

                        # C. Combine
                        if is_high_entropy:
                            trigger_now = True
                            trigger_reason = f"Entropy({entropy_val:.3f})"
                        elif is_high_kl:
                            trigger_now = True
                            trigger_reason = f"KL({kl_val:.3f})"

            # ==================================================================
            
            if trigger_now:
                print(f"--- Step {rl_t}: CO-Expander Triggered by {trigger_reason} ---")
                ce_triggered_flag = True 
                trigger_step_at = rl_t
                
                # --- [K-lookahead logic, unchanged] ---
                backup_rl_env_x = self.rl_env.x.clone()
                backup_rl_env_t = self.rl_env.t
                K = self.cfg.solver.get('lookahead_k', 2)
                
                best_ce_lookahead_mis_size = -1
                best_ce_lookahead_mask = np.array([])
                best_action_to_take = None 

                probs = torch.softmax(logits, dim=-1)
                base_greedy_actions_flat = torch.argmax(logits, dim=-1)
                flat_probs = probs.view(-1)
                
                # Safe top-K (guards against too few candidates)
                actual_k = min(K, flat_probs.shape[0])
                top_k_probs, top_k_indices = torch.topk(flat_probs, k=actual_k)

                candidate_actions_to_test = []
                for i in range(actual_k):
                    flat_idx = top_k_indices[i]
                    node_idx_in_logits = (flat_idx // 3).item()
                    action_val = (flat_idx % 3).item()
                    candidate_action_flat = base_greedy_actions_flat.clone()
                    candidate_action_flat[node_idx_in_logits] = action_val
                    candidate_actions_to_test.append(candidate_action_flat)

                for i, candidate_action_flat in enumerate(candidate_actions_to_test):
                    rl_action_candidate_full_flat = torch.zeros(
                        num_nodes * batch_size, dtype=torch.long, device=self.device
                    )   
                    rl_action_candidate_full_flat[flatten_node_idxs] = candidate_action_flat
                    rl_action_candidate = rl_action_candidate_full_flat.view(-1, batch_size)

                    self.rl_env.step(rl_action_candidate) 
                    
                    future_prefix_states = self.rl_env.x[:, 0]
                    future_prefix_tensor = (future_prefix_states == 1.0).nonzero().squeeze(-1)

                    ce_input_tuple = self._prepare_co_expander_input_tuple(original_g, future_prefix_tensor)
                    vars_heatmap_batch = self.co_expander_model.inference_node_sparse_process(*ce_input_tuple)
                    solution_batch = self.co_expander_model.decoder.sparse_decode(
                        vars_heatmap_batch, *ce_input_tuple, return_cost=False
                    )
                    
                    ce_solution_nodes = solution_batch[0]
                    current_ce_mis_size = np.sum(ce_solution_nodes)

                    if current_ce_mis_size > best_ce_lookahead_mis_size:
                        best_ce_lookahead_mis_size = current_ce_mis_size
                        best_action_to_take = rl_action_candidate.clone()
                        best_ce_lookahead_mask = ce_solution_nodes.astype(int)

                    self.rl_env.x = backup_rl_env_x
                    self.rl_env.t = backup_rl_env_t
                
                rl_action = best_action_to_take if best_action_to_take is not None else rl_action
                best_ce_mis_size = best_ce_lookahead_mis_size
                ce_solution_mask = best_ce_lookahead_mask

            # RL Step
            rl_state, reward, done_bool, info = self.rl_env.step(rl_action)
            current_sol_tensor = info['sol']
            
            if current_sol_tensor.max().item() > best_rl_sol_tensor_so_far.max().item():
                best_rl_sol_tensor_so_far = current_sol_tensor.clone()
                best_rl_state_so_far = current_rl_state_clone
                
            done = torch.all(done_bool).item() 
            rl_t += 1

        # --- Final Selection ---
        best_rl_mis_size = best_rl_sol_tensor_so_far.max().item() 
        
        # Return results
        final_mis = best_rl_mis_size
        source = "RL"
        mask = None
        
        if ce_triggered_flag and best_ce_mis_size > best_rl_mis_size:
            final_mis = best_ce_mis_size
            source = "CE"
            mask = ce_solution_mask
        else:
            # If CO-Expander was not triggered, or RL is better
            source = "RL"
            final_mis = best_rl_mis_size
        
        return {
            "mis_size": final_mis, 
            "solution_mask": mask, 
            "source": source, 
            "trigger": trigger_reason,
            "trigger_step": trigger_step_at
        }

    def _prepare_co_expander_input_tuple(self, g, prefix_nodes_tensor):
        # ... (unchanged) ...
        N = g.number_of_nodes()
        src, dst = g.edges()
        edge_index = torch.stack([src, dst], dim=0).to(self.device)
        E = edge_index.shape[1]
        task = "MIS"
        nodes_feature = None
        x = torch.zeros(N, device=self.device)
        edges_feature = torch.ones(E, device=self.device)
        e = None
        adj_matrix_sp = sp.coo_matrix((np.ones(E), (src.cpu().numpy(), dst.cpu().numpy())), shape=(N, N))
        graph_list = [adj_matrix_sp]
        mask = torch.zeros(N, dtype=torch.bool, device=self.device)
        ground_truth = torch.zeros(N, dtype=torch.long, device=self.device)
        nodes_num_list = [N]
        edges_num_list = [E]
        raw_data = MISGraphData()
        raw_data_list = [raw_data]
        prefix_nodes_batch_list = [prefix_nodes_tensor]
        node_prefix_state = torch.zeros((N, 1), dtype=torch.float32, device=self.device)
        if prefix_nodes_tensor.numel() > 0:
            node_prefix_state[prefix_nodes_tensor] = 1.0
            
        return (
            task, nodes_feature, x, edges_feature, e, edge_index, graph_list,
            mask, ground_truth, nodes_num_list, edges_num_list, raw_data_list,
            prefix_nodes_batch_list, node_prefix_state
        )

# ==============================================================================
# === Runner (supports overriding the fixed step) ===
# ==============================================================================
def run_evaluation(cfg: DictConfig, fixed_step=-1):
    """
    Run the evaluation loop.
    If fixed_step != -1, it overrides the config setting.
    Returns the average MIS size.
    """
    # Dynamic override
    if fixed_step != -1:
        cfg.solver.fixed_trigger_step = fixed_step
        print(f"\n>>> Running Evaluation with FIXED TRIGGER STEP = {fixed_step} <<<")
    else:
        cfg.solver.fixed_trigger_step = -1
        print(f"\n>>> Running Evaluation with DYNAMIC TRIGGER (Entropy/KL) <<<")

    solver = HybridSolverMIS(cfg)
    device = solver.device
    val_dataloader = _get_lwd_dataloader(cfg, device)

    all_solutions = []
    
    # Limit the number of samples for quick experiments (e.g., 20)
    # Increase max_instances to run on the full test set
    max_instances = 500 
    
    pbar = tqdm(val_dataloader, desc=f"Solving (Step={fixed_step if fixed_step!=-1 else 'Dyn'})")
    for i, g in enumerate(pbar):
        if i >= max_instances: break
        
        g.set_n_initializer(dgl.init.zero_initializer)
        g = g.to(device)
        if g is None or g.number_of_nodes() == 0: continue
            
        solution_stats = solver.solve_instance(g)
        all_solutions.append(solution_stats)

    if len(all_solutions) > 0:
        avg_mis_size = np.mean([s['mis_size'] for s in all_solutions])
    else:
        avg_mis_size = 0.0
        
    print(f"Result: Avg MIS Size = {avg_mis_size:.4f}")
    return avg_mis_size

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid LwD-RL-COExpander Solver for MIS")
    parser.add_argument("--config", type=str, default="hyco_config.yaml")
    parser.add_argument("--mode", type=str, choices=["dynamic", "fixed_exp"], default="fixed_exp",
                        help="dynamic: run the original dynamic logic; fixed_exp: run the fixed-step experiment and plot")
    args = parser.parse_args()
    
    # Base config
    default_solver_cfg = OmegaConf.create({
        'solver': {
            'use_theory_trigger': True,
            'probe_rl_top_m': 15,
            'entropy_threshold': 0.6,
            'lookahead_k': 2,
            'use_kl_trigger': True,
            'kl_threshold': 0.5,
            'probe_noise_t': 50,
            'fixed_trigger_step': -1 # dynamic by default
        }
    })
    
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(default_solver_cfg, cfg)
    
    if args.mode == "dynamic":
        # Mode 1: original dynamic trigger
        run_evaluation(cfg, fixed_step=-1)
        
    elif args.mode == "fixed_exp":
        # Mode 2: fixed-step experiment
        # MIS graphs are large and episodes can be long, so we pick a few key steps
        # e.g., 0 (trigger immediately), 10, 20, 30, 40, 50
        test_steps = [0,1,2,3,4,5, 10, 20, 30, 40, 50, 60]
        results_mis = []
        
        print(f"Starting Fixed Step Experiment for steps: {test_steps}")
        
        for step in test_steps:
            avg_mis = run_evaluation(cfg, fixed_step=step)
            results_mis.append(avg_mis)
            
        # Plot 
        plot_fixed_step_performance(test_steps, results_mis)
        
        print("\nExperiment Completed.")
        print("Steps:", test_steps)
        print("MIS Sizes:", results_mis)