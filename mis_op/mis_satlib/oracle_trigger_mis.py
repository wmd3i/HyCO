# Oracle trigger validation for MIS: compares the adaptive trigger with an instance-wise grid search over fixed trigger steps.
# Usage: python oracle_trigger_mis.py --config hyco_config.yaml
import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
import time
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as sp 
from tqdm.auto import tqdm
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader
from torch.distributions import Categorical

# --- Path setup ---
root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_folder)
base_dir = os.path.dirname(os.path.abspath(__file__))
co_expander_path = os.path.abspath(os.path.join(base_dir, '..'))
lwd_path = os.path.abspath(os.path.join(base_dir, 'learning_what_to_defer'))
import dgl 
if co_expander_path not in sys.path:
    sys.path.append(co_expander_path)
if lwd_path not in sys.path:
    sys.path.append(lwd_path)

# --- RL (LwD) Imports ---
from learning_what_to_defer.data.graph_dataset import GraphDataset
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
# === Plotting function: based on node count ===
# ==============================================================================
def plot_oracle_validation(results_df, save_dir="."):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    sns.set_theme(style="whitegrid")
    
    # Filter out instances where the adaptive trigger never fired (-1)
    df_triggered = results_df[results_df['adaptive_node_count'] != -1].copy()
    
    # --- Figure 1: node count correlation ---
    plt.figure(figsize=(8, 8))
    sns.scatterplot(
        data=df_triggered, 
        x='best_fixed_node_count', 
        y='adaptive_node_count', 
        s=80, alpha=0.6, color='purple', edgecolor='w'
    )
    
    # Draw the diagonal
    max_val = max(df_triggered['best_fixed_node_count'].max(), df_triggered['adaptive_node_count'].max()) if len(df_triggered) > 0 else 100
    plt.plot([0, max_val], [0, max_val], 'r--', label='Ideal Match (State Alignment)')
    
    plt.title("Correlation: Trigger State (Node Count)", fontsize=14)
    plt.xlabel("Oracle Optimal Node Count (When to trigger)", fontsize=12)
    plt.ylabel("Adaptive Trigger Node Count", fontsize=12)
    plt.legend()
    plt.savefig(os.path.join(save_dir, "mis_val_node_count_correlation.png"))
    plt.close()
    
    # --- Figure 2: node count distance ---
    plt.figure(figsize=(10, 6))
    if len(df_triggered) > 0:
        df_triggered['node_dist'] = (df_triggered['best_fixed_node_count'] - df_triggered['adaptive_node_count']).abs()
        sns.histplot(df_triggered['node_dist'], kde=True, bins=15, color='orange')
        plt.title("State Distance Distribution (|Adaptive Nodes - Oracle Nodes|)", fontsize=14)
        plt.xlabel("Node Count Difference", fontsize=12)
        plt.savefig(os.path.join(save_dir, "mis_val_node_distance.png"))
    plt.close()

    print(f"[Plotting] Figures saved to {os.path.abspath(save_dir)}")

# ==============================================================================
# === Hybrid solver ===
# ==============================================================================
class HybridSolverMIS:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. RL Policy
        self.rl_policy = self._load_lwd_rl_policy()
        # 2. Env
        max_nodes = 1347
        self.rl_env = MaximumIndependentSetEnv(
            max_epi_t=128, max_num_nodes=max_nodes,
            hamming_reward_coef=0.01, device=self.device
        )
        self.rl_eval_samples = self.cfg.rl_model.get('eval_num_samples', 1)
        # 3. CO-Expander
        self.co_expander_model, self.co_expander_cfg = self._load_co_expander_model()

    def _load_lwd_rl_policy(self):
        # ... (unchanged) ...
        n_layers = self.cfg.rl_model.n_layers
        hidden_dim = self.cfg.rl_model.hidden_dim
        max_nodes_from_training = 1347
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
        try:
            adj_matrix_sp = sp.coo_matrix((np.ones(E), (src.cpu().numpy(), dst.cpu().numpy())), shape=(N, N))
        except:
            adj_matrix_sp = sp.coo_matrix((N, N))
        return (task, nodes_feature, x, edges_feature, e, edge_index, [adj_matrix_sp],
                torch.zeros(N, dtype=torch.bool, device=self.device), torch.zeros(N, dtype=torch.long, device=self.device),
                [N], [E], [MISGraphData()], [prefix_nodes_tensor], 
                torch.zeros((N, 1), dtype=torch.float32, device=self.device).index_fill_(0, prefix_nodes_tensor, 1.0) if prefix_nodes_tensor.numel() > 0 else torch.zeros((N, 1), dtype=torch.float32, device=self.device))

    @torch.no_grad()
    def solve_instance(self, g, override_fixed_step=None):
        """
        [Key fix]: the result contains trigger_node_count, not just trigger_step
        """
        original_g = g.to(self.device) 
        
        # Force batch_size = 1 for validation
        rl_state = self.rl_env.register(g, num_samples=1) 

        rl_t = 0
        done = False
        ce_triggered_flag = False
        best_ce_mis_size = -1
        ce_solution_mask = np.array([])
        
        trigger_step_at = -1 
        trigger_node_count_at = -1 # [New] record how many nodes had been selected when the trigger fired
        
        if override_fixed_step is not None:
            current_fixed_step = override_fixed_step
        else:
            current_fixed_step = getattr(self.cfg.solver, 'fixed_trigger_step', -1)

        best_rl_sol_tensor_so_far = torch.zeros((1, 1), device=self.device)
        best_rl_state_so_far = rl_state.clone()

        while not done:
            num_nodes, batch_size = rl_state.size(0), rl_state.size(1)

            masks, idxs, subg, h = self.rl_policy.get_masks_idxs_subg_h(rl_state, g)
            _, _, subg_node_mask = masks
            _, flatten_node_idxs, flatten_subg_node_idxs = idxs

            logits = (
                self.rl_policy.actor_net(h, subg, mask = subg_node_mask)
                .view(-1, 3).index_select(0, flatten_subg_node_idxs)
            )
            rl_dist = Categorical(logits = logits)
            sampled_actions = rl_dist.sample() 

            # Validation forces batch=1, so direct assignment is safe
            rl_action_flat = torch.zeros(num_nodes * batch_size, dtype=torch.long, device=self.device)   
            rl_action_flat[flatten_node_idxs] = sampled_actions
            rl_action = rl_action_flat.view(-1, batch_size)
            
            # --- Trigger Logic ---
            trigger_now = False
            
            # Single-trigger constraint
            if not ce_triggered_flag:
                if current_fixed_step >= 0:
                    if rl_t == current_fixed_step: trigger_now = True
                else:
                    if self.cfg.solver.use_theory_trigger:
                        entropies = rl_dist.entropy()
                        M = min(self.cfg.solver.probe_rl_top_m, len(entropies))
                        if M > 0:
                            top_m_entropies, _ = torch.topk(entropies, k=M)
                            if torch.mean(top_m_entropies) > self.cfg.solver.entropy_threshold:
                                trigger_now = True

            # --- Execution ---
            if trigger_now:
                ce_triggered_flag = True 
                trigger_step_at = rl_t
                
                # [Key] number of selected nodes at the current moment
                # rl_state: [N, 1, 1] -> [:, 0, 0]
                current_solution_states = self.rl_env.x[:, 0]
                current_selected_count = (current_solution_states == 1.0).sum().item()
                trigger_node_count_at = current_selected_count
                
                # K-Lookahead Logic
                backup_rl_env_x = self.rl_env.x.clone()
                backup_rl_env_t = self.rl_env.t
                
                K = 2 
                probs = torch.softmax(logits, dim=-1)
                flat_probs = probs.view(-1)
                actual_k = min(K, flat_probs.shape[0])
                
                best_action = None
                
                if actual_k > 0:
                    top_k_probs, top_k_indices = torch.topk(flat_probs, k=actual_k)
                    base_actions = torch.argmax(logits, dim=-1)
                    
                    candidate_actions = []
                    for i in range(actual_k):
                        flat_idx = top_k_indices[i]
                        node_idx = (flat_idx // 3).item()
                        act_val = (flat_idx % 3).item()
                        cand = base_actions.clone()
                        cand[node_idx] = act_val
                        candidate_actions.append(cand)
                        
                    for cand_act in candidate_actions:
                        full_act = torch.zeros(num_nodes * batch_size, dtype=torch.long, device=self.device)
                        full_act[flatten_node_idxs] = cand_act
                        full_act = full_act.view(-1, batch_size)
                        
                        self.rl_env.step(full_act)
                        
                        prefix = (self.rl_env.x[:, 0] == 1.0).nonzero().squeeze(-1)
                        ce_in = self._prepare_co_expander_input_tuple(original_g, prefix)
                        heatmap = self.co_expander_model.inference_node_sparse_process(*ce_in)
                        sol = self.co_expander_model.decoder.sparse_decode(heatmap, *ce_in, return_cost=False)[0]
                        
                        size = np.sum(sol)
                        if size > best_ce_mis_size:
                            best_ce_mis_size = size
                            best_action = full_act.clone()
                            ce_solution_mask = sol.astype(int)
                            
                        self.rl_env.x = backup_rl_env_x
                        self.rl_env.t = backup_rl_env_t
                
                if best_action is not None:
                    rl_action = best_action

            # RL Step
            rl_state, reward, done_bool, info = self.rl_env.step(rl_action)
            
            if info['sol'].max().item() > best_rl_sol_tensor_so_far.max().item():
                best_rl_sol_tensor_so_far = info['sol'].clone()
                best_rl_state_so_far = rl_state.clone()
                
            done = torch.all(done_bool).item() 
            rl_t += 1

        # Final Return
        best_rl = best_rl_sol_tensor_so_far.max().item()
        
        idx = best_rl_sol_tensor_so_far.argmax().item()
        if best_rl_state_so_far.ndim == 3:
            mask = (best_rl_state_so_far[:, idx, 0] == 1.0).cpu().numpy().astype(int)
        else:
            mask = (best_rl_state_so_far[:, 0] == 1.0).cpu().numpy().astype(int)

        final_mis = best_rl
        source = "RL"
        
        if ce_triggered_flag and best_ce_mis_size > best_rl:
            final_mis = best_ce_mis_size
            mask = ce_solution_mask
            source = "CE"
        
        return {
            "mis_size": final_mis, 
            "solution_mask": mask, 
            "source": source,
            "trigger_step": trigger_step_at,
            "trigger_node_count": trigger_node_count_at # [New] return the node count at trigger time
        }

# ==============================================================================
# === Oracle Validation Runner ===
# ==============================================================================
def run_oracle_validation(cfg: DictConfig):
    # [Key fix] force num_samples = 1
    cfg.rl_model.eval_num_samples = 1 
    
    solver = HybridSolverMIS(cfg)

    # Dataset loading is the same as before
    from learning_what_to_defer.data.graph_dataset import get_satlib_dataset
    dataset = get_satlib_dataset("test", "../data/lwd/satlib/test")
    
    # Grid Search Steps
    grid_search_steps = [0, 10, 20, 30, 40, 50, 60, 80]
    
    num_test_instances = 500

    results_data = []

    print(f"\n>>> Starting Oracle Validation on {num_test_instances} instances (State Alignment) <<<")
    
    for i in tqdm(range(num_test_instances), desc="Validating"):
        g = dataset[i]
        
        # 1. Adaptive
        res_adaptive = solver.solve_instance(g, override_fixed_step=-1)
        adaptive_step = res_adaptive['trigger_step']
        adaptive_node_count = res_adaptive['trigger_node_count'] # Get the node count
        adaptive_mis = res_adaptive['mis_size']
        
        # 2. Oracle Grid
        best_fixed_mis = -1.0
        best_fixed_step = -1
        best_fixed_node_count = -1
        
        for step in grid_search_steps:
            res_fixed = solver.solve_instance(g, override_fixed_step=step)
            
            # Note: if fixed_step is too large and never triggers, res_fixed['trigger_node_count'] is -1
            # Such results usually perform worse and will not be the best
            
            if res_fixed['mis_size'] > best_fixed_mis:
                best_fixed_mis = res_fixed['mis_size']
                best_fixed_step = step
                best_fixed_node_count = res_fixed['trigger_node_count']
                
        results_data.append({
            "instance_id": i,
            "adaptive_step": adaptive_step,
            "adaptive_node_count": adaptive_node_count, # Record the state
            "adaptive_mis": adaptive_mis,
            "best_fixed_step": best_fixed_step,
            "best_fixed_node_count": best_fixed_node_count, # Record the state
            "best_fixed_mis": best_fixed_mis
        })

    df = pd.DataFrame(results_data)
    avg_gap = (df['best_fixed_mis'] - df['adaptive_mis']).mean()
    triggered_count = len(df[df['adaptive_node_count'] != -1])
    
    print("\n" + "="*50)
    print("--- Oracle Validation Results (State Aligned) ---")
    print(f"Total: {len(df)}, Adaptive Triggered: {triggered_count}")
    print(f"Avg Performance Gap (Best Fixed - Adaptive): {avg_gap:.4f}")
    
    if triggered_count > 0:
        # Compute the node-count distance instead of the step distance
        df_trig = df[df['adaptive_node_count'] != -1]
        
        # Note: sometimes the best fixed setting never triggers (e.g., step too large); filter out best_fixed_node_count == -1
        df_valid = df_trig[df_trig['best_fixed_node_count'] != -1]
        
        if len(df_valid) > 0:
            avg_node_dist = (df_valid['best_fixed_node_count'] - df_valid['adaptive_node_count']).abs().mean()
            print(f"Avg STATE Distance (|Oracle Nodes - Adaptive Nodes|): {avg_node_dist:.2f}")
        else:
            print("No valid comparison for state distance (Fixed steps mostly failed to trigger).")
            
    print("="*50)
    
    df.to_csv("mis_oracle_validation_nodes.csv", index=False)
    plot_oracle_validation(df, save_dir="node_level_oracle_plots")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="hyco_config.yaml")
    args = parser.parse_args()
    
    cfg = OmegaConf.load(args.config)
    # Ensure default solver config
    default_solver = OmegaConf.create({
        'solver': {'use_theory_trigger': True, 'probe_rl_top_m': 15, 'entropy_threshold': 0.6, 'fixed_trigger_step': -1}
    })
    cfg = OmegaConf.merge(default_solver, cfg)
    
    run_oracle_validation(cfg)