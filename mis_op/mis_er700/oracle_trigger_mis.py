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
if root_folder not in sys.path:
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
# === [Plotting] validation plots based on node count ===
# ==============================================================================
def plot_oracle_validation(results_df, save_dir="."):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    sns.set_theme(style="whitegrid")
    
    # Filter out instances where the adaptive trigger never fired (-1)
    df_triggered = results_df[results_df['adaptive_node_count'] != -1].copy()
    
    # --- Figure 1: trigger-state correlation (node count) ---
    plt.figure(figsize=(8, 8))
    sns.scatterplot(
        data=df_triggered, 
        x='best_fixed_node_count', 
        y='adaptive_node_count', 
        s=80, alpha=0.6, color='purple', edgecolor='w'
    )
    
    # Draw the diagonal
    max_val = max(df_triggered['best_fixed_node_count'].max(), df_triggered['adaptive_node_count'].max()) if len(df_triggered) > 0 else 100
    plt.plot([0, max_val], [0, max_val], 'r--', label='Ideal Match')
    
    plt.title("Trigger Point Correlation (Node Count)", fontsize=14)
    plt.xlabel("Oracle Optimal Trigger Point (Node Count)", fontsize=12)
    plt.ylabel("Adaptive Trigger Point (Node Count)", fontsize=12)
    plt.legend()
    plt.savefig(os.path.join(save_dir, "mis_val_node_correlation.png"), dpi=300)
    plt.close()
    
    # --- Figure 2: state distance distribution ---
    plt.figure(figsize=(10, 6))
    if len(df_triggered) > 0:
        df_triggered['node_dist'] = (df_triggered['best_fixed_node_count'] - df_triggered['adaptive_node_count']).abs()
        sns.histplot(df_triggered['node_dist'], kde=True, bins=15, color='orange')
        plt.title("Trigger Distance Distribution (|Adaptive - Oracle|)", fontsize=14)
        plt.xlabel("Difference in Selected Node Count", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.savefig(os.path.join(save_dir, "mis_val_node_distance.png"), dpi=300)
    plt.close()

    # --- Figure 3: performance gap distribution ---
    plt.figure(figsize=(10, 6))
    # Gap = oracle - adaptive (larger MIS is better)
    results_df['gap'] = results_df['best_fixed_mis'] - results_df['adaptive_mis']
    sns.histplot(results_df['gap'], kde=True, bins=15, color='green')
    plt.axvline(0, color='red', linestyle='--', label='Zero Gap')
    plt.title("Performance Gap (Oracle - Adaptive)", fontsize=14)
    plt.xlabel("MIS Size Difference (Positive = Oracle Better)", fontsize=12)
    plt.savefig(os.path.join(save_dir, "mis_val_performance_gap.png"), dpi=300)
    plt.close()

    print(f"[Plotting] Figures saved to {os.path.abspath(save_dir)}")

# ==============================================================================
# === Hybrid solver ===
# ==============================================================================
class HybridSolverMIS:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Solver using device: {self.device}")

        # 1. Load the LwD RL policy
        self.rl_policy = self._load_lwd_rl_policy()
        
        # 2. Load the LwD environment (ER-700-800 setting)
        max_nodes_from_training = 800
        self.rl_env = MaximumIndependentSetEnv(
            max_epi_t=100, 
            max_num_nodes=max_nodes_from_training,
            hamming_reward_coef=0.1,
            device=self.device
        )
        
        # Number of samples used for evaluation (forced to 1 in oracle validation)
        self.rl_eval_samples = self.cfg.rl_model.get('eval_num_samples', 10) 
        print(f"LwD RL will use {self.rl_eval_samples} parallel samples for evaluation.")

        # 3. Load the CO-Expander model
        self.co_expander_model, self.co_expander_cfg = self._load_co_expander_model()

    def _load_lwd_rl_policy(self):
        print(f"Loading LwD RL model from: {self.cfg.rl_model.ckpt_path}")
        n_layers = self.cfg.rl_model.n_layers
        hidden_dim = self.cfg.rl_model.hidden_dim
        max_nodes_from_training = 800
        
        policy = ActorCritic(
            actor_class=PolicyGraphConvNet,
            critic_class=ValueGraphConvNet,
            max_num_nodes=max_nodes_from_training,
            hidden_dim=hidden_dim,
            num_layers=n_layers,
            device=self.device
        ).to(self.device)  

        checkpoint = torch.load(self.cfg.rl_model.ckpt_path, map_location=self.device)
        policy.load_state_dict(checkpoint['model_state_dict'])
        policy.eval()
        print("LwD RL Policy loaded successfully.")
        return policy

    def _load_co_expander_model(self):
        print(f"Loading CO-Expander model from: {self.cfg.co_expander_model.ckpt_path}")
        ce_train_cfg = OmegaConf.load(self.cfg.co_expander_model.config_path)
        ce_env = COExpanderEnv(
            task=ce_train_cfg.task,
            val_path=ce_train_cfg.data.test_path,
            val_data_size=ce_train_cfg.inference.num_test_samples,
            val_batch_size=ce_train_cfg.inference.batch_size,
            sparse_factor=ce_train_cfg.sparse_factor,
            device=self.device,
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
        print("CO-Expander model loaded successfully.")
        return model, ce_train_cfg

    def _prepare_co_expander_input_tuple(self, g, prefix_nodes_tensor):
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

    # ==========================================================================
    # === [Core] solve instance with override_fixed_step ===
    # ==========================================================================
    @torch.no_grad()
    def solve_instance(self, g, override_fixed_step=None):
        """
        Solve a single MIS instance.
        
        Args:
            g: DGL Graph
            override_fixed_step: 
                None/-1: use the adaptive trigger (entropy)
                >=0:     use the fixed-step trigger (trigger at step k)
        
        Returns:
            dict: contains mis_size, solution_mask, source, trigger_step, trigger_node_count
        """
        original_g = g.to(self.device) 
        
        # 1. Register the RL environment
        # Note: run_oracle_validation forces self.rl_eval_samples = 1
        rl_state = self.rl_env.register(g, num_samples=self.rl_eval_samples) 

        rl_t = 0
        done = False
        ce_triggered_flag = False
        best_ce_mis_size = -1
        ce_solution_mask = np.array([])
        
        # Record trigger information
        trigger_step_at = -1 
        trigger_node_count_at = -1 # [Key] record how many nodes had been selected when the trigger fired
        
        # Determine the trigger mode
        if override_fixed_step is not None and override_fixed_step >= 0:
            mode = "fixed"
            target_step = override_fixed_step
        else:
            mode = "adaptive"

        best_rl_sol_tensor_so_far = torch.zeros((1, self.rl_eval_samples), device=self.device)
        best_rl_state_so_far = rl_state.clone()

        while not done:
            num_nodes, batch_size = rl_state.size(0), rl_state.size(1)

            # RL Forward
            masks, idxs, subg, h = self.rl_policy.get_masks_idxs_subg_h(rl_state, g)
            _, _, subg_node_mask = masks
            flatten_node_idxs, flatten_subg_idxs, flatten_subg_node_idxs = idxs

            logits = (
                self.rl_policy.actor_net(h, subg, mask = subg_node_mask)
                .view(-1, 3).index_select(0, flatten_subg_node_idxs)
            )
            rl_dist = Categorical(logits = logits)
            sampled_actions = rl_dist.sample() 

            # Note: with eval_samples=1, flatten_node_idxs and sampled_actions have the same length, so direct assignment is safe
            rl_action_flat = torch.zeros(num_nodes * batch_size, dtype=torch.long, device=self.device)   
            rl_action_flat[flatten_node_idxs] = sampled_actions
            rl_action = rl_action_flat.view(-1, batch_size)
            
            # ========================================================
            # === Trigger decision ===
            # ========================================================
            trigger_now = False
            
            # [Constraint] only check if CO-Expander has not been triggered yet
            if not ce_triggered_flag:
                if mode == "fixed":
                    # [Mode 1: Fixed Step]
                    if rl_t == target_step:
                        trigger_now = True
                elif mode == "adaptive":
                    # [Mode 2: Adaptive (Entropy)]
                    if self.cfg.solver.use_theory_trigger:
                        entropies = rl_dist.entropy()
                        M = min(self.cfg.solver.probe_rl_top_m, len(entropies))
                        if M > 0:
                            top_m_entropies, _ = torch.topk(entropies, k=M)
                            avg_entropy = torch.mean(top_m_entropies)
                            if avg_entropy > self.cfg.solver.entropy_threshold:
                                trigger_now = True

            # ========================================================
            # === Trigger execution (K-lookahead) ===
            # ========================================================
            if trigger_now:
                ce_triggered_flag = True 
                trigger_step_at = rl_t
                
                # [Key] record the node count at trigger time
                # self.rl_env.x [N, B, 1] -> [:, 0, 0] takes the state of the first sample
                # State 1.0 means selected into the MIS
                current_solution_states = self.rl_env.x[:, 0] if self.rl_env.x.ndim == 2 else self.rl_env.x[:, 0, 0]
                current_selected_count = (current_solution_states == 1.0).sum().item()
                trigger_node_count_at = current_selected_count
                
                # --- Lookahead ---
                backup_rl_env_x = self.rl_env.x.clone()
                backup_rl_env_t = self.rl_env.t
                
                K = 2 # K-Lookahead
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
                        
                        # Extract the prefix
                        if self.rl_env.x.ndim == 3:
                            prefix_states = self.rl_env.x[:, 0, 0]
                        else:
                            prefix_states = self.rl_env.x[:, 0]
                            
                        prefix = (prefix_states == 1.0).nonzero().squeeze(-1)
                        
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

        # --- Extract the final solution ---
        best_rl = best_rl_sol_tensor_so_far.max().item()
        
        idx = best_rl_sol_tensor_so_far.argmax().item()
        if best_rl_state_so_far.ndim == 3:
            mask = (best_rl_state_so_far[:, idx, 0] == 1.0).cpu().numpy().astype(int)
        else:
            mask = (best_rl_state_so_far[:, 0] == 1.0).cpu().numpy().astype(int)

        final_mis = best_rl
        source = "RL"
        final_mask = mask

        if ce_triggered_flag and best_ce_mis_size > best_rl:
            final_mis = best_ce_mis_size
            final_mask = ce_solution_mask
            source = "CE"
        
        return {
            "mis_size": final_mis, 
            "solution_mask": final_mask, 
            "source": source,
            "trigger_step": trigger_step_at,
            "trigger_node_count": trigger_node_count_at # [New] return the node count
        }

# ==============================================================================
# === Oracle Validation Runner ===
# ==============================================================================
def run_oracle_validation(cfg: DictConfig):
    # [Key] force the number of evaluation samples to 1 to avoid shape mismatches during validation
    # This also keeps the comparison fair (single greedy vs. single greedy)
    cfg.rl_model.eval_num_samples = 1 
    
    solver = HybridSolverMIS(cfg)
    
    # Use the ER-700-800 dataset loader
    from learning_what_to_defer.data.graph_dataset import get_er_700_800_dataset
    dataset = get_er_700_800_dataset("test", "../data/lwd/er_700_800/test")
    
    # Grid-search steps (LwD episodes are usually long)
    grid_search_steps = [0, 10, 20, 30, 40, 50, 60, 80]
    
    num_test_instances = 500 
    results_data = []

    print(f"\n>>> Starting Oracle Validation on {num_test_instances} instances (Eval Samples=1) <<<")
    
    for i in tqdm(range(num_test_instances), desc="Validating"):
        g = dataset[i]
        g.set_n_initializer(dgl.init.zero_initializer)
        
        # 1. Run Adaptive Trigger (Heuristic)
        res_adaptive = solver.solve_instance(g, override_fixed_step=-1)
        
        # 2. Run Oracle Grid Search (Fixed Steps)
        best_fixed_mis = -1.0
        best_fixed_step = -1
        best_fixed_node_count = -1
        
        for step in grid_search_steps:
            res_fixed = solver.solve_instance(g, override_fixed_step=step)
            
            # Find the best fixed-step result
            if res_fixed['mis_size'] > best_fixed_mis:
                best_fixed_mis = res_fixed['mis_size']
                best_fixed_step = step
                best_fixed_node_count = res_fixed['trigger_node_count']
                
        results_data.append({
            "instance_id": i,
            "adaptive_step": res_adaptive['trigger_step'],
            "adaptive_node_count": res_adaptive['trigger_node_count'], # Record the node count
            "adaptive_mis": res_adaptive['mis_size'],
            "best_fixed_step": best_fixed_step,
            "best_fixed_node_count": best_fixed_node_count, # Record the node count
            "best_fixed_mis": best_fixed_mis
        })

    # --- Analysis ---
    df = pd.DataFrame(results_data)
    avg_gap = (df['best_fixed_mis'] - df['adaptive_mis']).mean()
    triggered_count = len(df[df['adaptive_node_count'] != -1])
    
    print("\n" + "="*50)
    print("--- Oracle Validation Results (Node Count Aligned) ---")
    print(f"Total: {len(df)}, Adaptive Triggered: {triggered_count}")
    print(f"Avg Performance Gap (Oracle - Adaptive): {avg_gap:.4f}")
    
    if triggered_count > 0:
        # Compute the node-count distance
        df_trig = df[df['adaptive_node_count'] != -1]
        # Filter out cases where the fixed trigger also never fired (best_fixed_node_count == -1)
        df_valid = df_trig[df_trig['best_fixed_node_count'] != -1]
        
        if len(df_valid) > 0:
            avg_node_dist = (df_valid['best_fixed_node_count'] - df_valid['adaptive_node_count']).abs().mean()
            print(f"Avg STATE Distance (|Oracle Nodes - Adaptive Nodes|): {avg_node_dist:.2f}")
    print("="*50)
    
    csv_path = "mis_oracle_validation_nodes.csv"
    df.to_csv(csv_path, index=False)
    plot_oracle_validation(df, save_dir="oracle_plots_node")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="hyco_config.yaml")
    args = parser.parse_args()
    
    cfg = OmegaConf.load(args.config)
    # Make sure the solver config exists
    if 'solver' not in cfg: cfg.solver = {}
    default_solver = OmegaConf.create({
        'solver': {'use_theory_trigger': True, 'probe_rl_top_m': 15, 'entropy_threshold': 0.6, 'fixed_trigger_step': -1}
    })
    cfg = OmegaConf.merge(default_solver, cfg)
    
    run_oracle_validation(cfg)