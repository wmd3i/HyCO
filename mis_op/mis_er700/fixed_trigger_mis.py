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

# --- Plotting libraries ---
import matplotlib.pyplot as plt
import seaborn as sns

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
# === Plotting function ===
# ==============================================================================
def plot_fixed_step_performance(steps, mis_sizes, save_path="fixed_step_performance_mis.png"):
    """
    Plot fixed trigger step vs. average MIS size
    """
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Plot the main curve
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
    """ (unchanged) """
    # print(f"Loading CO-Expander config to find test data path: {cfg.co_expander_model.config_path}")
    ce_train_cfg = OmegaConf.load(cfg.co_expander_model.config_path)
    test_data_path = "../data/lwd/er_700_800/test"
    # print(f"Loading LwD native dataset from: {test_data_path}")
    
    from learning_what_to_defer.data.graph_dataset import get_er_700_800_dataset
    test_dataset = get_er_700_800_dataset("test", test_data_path)
    
    def collate_fn(graphs):
        return dgl.batch(graphs)
        
    dataloader = DataLoader(
        test_dataset, batch_size = 1, shuffle = False, collate_fn = collate_fn, num_workers = 0
    )
    # print(f"Found {len(test_dataset)} instances in the LwD dataset.")
    return dataloader

class HybridSolverMIS:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Solver using device: {self.device}")

        # 1. Load the RL model
        self.rl_policy = self._load_lwd_rl_policy()
        max_nodes_from_training = 800
        self.rl_env = MaximumIndependentSetEnv(
            max_epi_t=100, max_num_nodes=max_nodes_from_training,
            hamming_reward_coef=0.1, device=self.device
        )
        self.rl_eval_samples = self.cfg.rl_model.get('eval_num_samples', 10)

        # 2. Load CO-Expander
        self.co_expander_model, self.co_expander_cfg = self._load_co_expander_model()

    def _load_lwd_rl_policy(self):
        # (unchanged)
        # print(f"Loading LwD RL model from: {self.cfg.rl_model.ckpt_path}")
        n_layers = self.cfg.rl_model.n_layers
        hidden_dim = self.cfg.rl_model.hidden_dim
        max_nodes_from_training = 800
        
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
        # (unchanged)
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
    def solve_instance(self, g):
        g.set_n_initializer(dgl.init.zero_initializer)
        original_g = g.to(self.device) 
        rl_state = self.rl_env.register(g, num_samples=self.rl_eval_samples) 

        rl_t = 0
        done = False
        ce_triggered_flag = False
        best_ce_mis_size = -1
        ce_solution_mask = np.array([]) 
        
        best_rl_sol_tensor_so_far = torch.zeros((1, self.rl_eval_samples), device=self.device)
        best_rl_state_so_far = rl_state.clone() 

        # === Get the fixed-step config ===
        # If fixed_trigger_step is missing from the config or equals -1, the original dynamic trigger is used
        fixed_step_trigger = getattr(self.cfg.solver, 'fixed_trigger_step', -1)

        while not done:
            current_rl_state_clone = rl_state.clone()
            num_nodes, batch_size = rl_state.size(0), rl_state.size(1) 

            masks, idxs, subg, h = self.rl_policy.get_masks_idxs_subg_h(rl_state, g)
            _, _, subg_node_mask = masks
            flatten_node_idxs, flatten_subg_idxs, flatten_subg_node_idxs = idxs

            logits = (
                self.rl_policy.actor_net(h, subg, mask = subg_node_mask)
                .view(-1, 3).index_select(0, flatten_subg_node_idxs)
            )
            rl_dist = Categorical(logits = logits)
            sampled_actions = rl_dist.sample() 
            
            rl_action_flat = torch.zeros(num_nodes * batch_size, dtype = torch.long, device = self.device)   
            rl_action_flat[flatten_node_idxs] = sampled_actions 
            rl_action = rl_action_flat.view(-1, batch_size)
            
            # =======================================================
            # === Trigger logic: fixed step vs. dynamic entropy ===
            # =======================================================
            trigger_now = False
            trigger_reason = ""

            if fixed_step_trigger >= 0:
                # --- Mode 1: fixed step ---
                # Trigger only at the specified step and only if not triggered before
                if rl_t == fixed_step_trigger and not ce_triggered_flag:
                    trigger_now = True
                    trigger_reason = f"Fixed Step {fixed_step_trigger}"
            else:
                # --- Mode 2: original dynamic trigger (entropy) ---
                if self.cfg.solver.use_theory_trigger and not ce_triggered_flag:
                    entropies = rl_dist.entropy()
                    M = min(self.cfg.solver.probe_rl_top_m, len(entropies))
                    if M > 0:
                        top_m_entropies, _ = torch.topk(entropies, k=M)
                        avg_entropy_top_m = torch.mean(top_m_entropies)
                        is_high_entropy = avg_entropy_top_m > self.cfg.solver.entropy_threshold
                        
                        if is_high_entropy:
                            trigger_now = True
                            trigger_reason = f"Entropy {avg_entropy_top_m:.3f}"

            # 4. Trigger CO-Expander
            if trigger_now:
                # print(f"--- Step {rl_t}: CO-Expander triggered by [{trigger_reason}] ---")
                ce_triggered_flag = True

                current_solution_states = rl_state[:, 0, 0] 
                prefix_nodes_tensor = (current_solution_states == 1.0).nonzero().squeeze(-1)

                ce_input_tuple = self._prepare_co_expander_input_tuple(original_g, prefix_nodes_tensor)
                vars_heatmap_batch = self.co_expander_model.inference_node_sparse_process(*ce_input_tuple)
                solution_batch = self.co_expander_model.decoder.sparse_decode(vars_heatmap_batch, *ce_input_tuple, return_cost=False)
                
                ce_solution_nodes = solution_batch[0]
                best_ce_mis_size = np.sum(ce_solution_nodes)
                ce_solution_mask = ce_solution_nodes.astype(int) 

            # 5. RL step
            rl_state, reward, done_bool, info = self.rl_env.step(rl_action)
            current_sol_tensor = info['sol']
            
            if current_sol_tensor.max().item() > best_rl_sol_tensor_so_far.max().item():
                best_rl_sol_tensor_so_far = current_sol_tensor.clone()
                best_rl_state_so_far = self.rl_env.x.clone()
                
            done = torch.all(done_bool).item() 
            rl_t += 1

        # --- Extract the final solution ---
        best_rl_mis_size = best_rl_sol_tensor_so_far.max().item() 
        best_sample_idx = best_rl_sol_tensor_so_far.argmax().item()

        final_mis_size = best_rl_mis_size
        source = "RL"
        final_mask = None

        if best_rl_mis_size > best_ce_mis_size:
            rl_solution_mask_floats = best_rl_state_so_far[:, best_sample_idx] 
            final_mask = (rl_solution_mask_floats == 1.0).cpu().numpy().astype(int)
            source = "RL"
            final_mis_size = best_rl_mis_size
        elif ce_triggered_flag:
            final_mask = ce_solution_mask
            source = "CE"
            final_mis_size = best_ce_mis_size
        else:
             rl_solution_mask_floats = best_rl_state_so_far[:, best_sample_idx]
             final_mask = (rl_solution_mask_floats == 1.0).cpu().numpy().astype(int)
             source = "RL"
             final_mis_size = best_rl_mis_size
             
        return {"mis_size": final_mis_size, "solution_mask": final_mask, "source": source}

    def _prepare_co_expander_input_tuple(self, g, prefix_nodes_tensor):
        """ (unchanged) """
        N = g.number_of_nodes()
        src, dst = g.edges()
        edge_index = torch.stack([src, dst], dim=0).to(self.device)
        E = edge_index.shape[1]
        task = "MIS"
        nodes_feature = None
        x = torch.zeros(N, device=self.device)
        edges_feature = torch.ones(E, device=self.device)
        e = None
        # Handle the possible case of no edges
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
            
        return (task, nodes_feature, x, edges_feature, e, edge_index, graph_list, mask, ground_truth, nodes_num_list, edges_num_list, raw_data_list, prefix_nodes_batch_list, node_prefix_state)

# ==============================================================================
# === Runner (supports a single run and returns the result) ===
# ==============================================================================
def run_evaluation(cfg: DictConfig):
    """
    Run one full evaluation loop and return the average MIS size
    """
    solver = HybridSolverMIS(cfg)
    device = solver.device
    val_dataloader = _get_lwd_dataloader(cfg, device)

    all_solutions = []
    
    # Limit the number of samples for faster experiments
    num_samples_to_test = 500 
    count = 0
    
    # Determine the current step label for display
    step_label = cfg.solver.get('fixed_trigger_step', -1)
    desc = f"Solving (FixedStep={step_label})" if step_label != -1 else "Solving (Dynamic)"

    for g in tqdm(val_dataloader, desc=desc):
        count += 1
        if count > num_samples_to_test: break
        
        g.set_n_initializer(dgl.init.zero_initializer)
        g = g.to(device)
        if g is None or g.number_of_nodes() == 0: continue
            
        solution_stats = solver.solve_instance(g)
        all_solutions.append(solution_stats)

    if len(all_solutions) > 0:
        avg_mis_size = np.mean([s['mis_size'] for s in all_solutions])
    else:
        avg_mis_size = 0.0
        
    return avg_mis_size

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid LwD-RL-COExpander Solver for MIS")
    parser.add_argument("--config", type=str, default="hyco_config.yaml")
    # Mode selection
    parser.add_argument("--mode", type=str, choices=["dynamic", "fixed_exp"], default="fixed_exp", 
                        help="dynamic: original dynamic trigger; fixed_exp: run the fixed-step experiment and plot")
    args = parser.parse_args()
    
    base_cfg = OmegaConf.load(args.config)
    
    if args.mode == "dynamic":
        # === Mode 1: original dynamic trigger ===
        print(">>> Running in DYNAMIC Trigger Mode <<<")
        # Make sure fixed_trigger_step is -1
        base_cfg.solver.fixed_trigger_step = -1
        avg_mis = run_evaluation(base_cfg)
        print(f"Final Dynamic Average MIS Size: {avg_mis:.4f}")

    elif args.mode == "fixed_exp":
        # === Mode 2: fixed-step experiment (ablation study) ===
        print(">>> Running FIXED STEP Experiment <<<")
        
        # Steps to test (0 means handing over to CO-Expander right away)
        # Note: the number of steps for MIS is typically between 0 and 100
        test_steps = [0,1,2,3,4,5, 10, 20, 30, 40, 50, 60] 
        results_mis = []
        
        for step in test_steps:
            print(f"\n--- Testing Fixed Trigger Step: {step} ---")
            
            # Modify the config dynamically
            current_cfg = base_cfg.copy()
            # Set the fixed step
            current_cfg.solver.fixed_trigger_step = step
            # Disable the theory trigger to avoid interference (the code already uses if/else, but this is safer)
            current_cfg.solver.use_theory_trigger = False 
            
            avg_mis = run_evaluation(current_cfg)
            results_mis.append(avg_mis)
            print(f"Step {step} Result: Avg MIS = {avg_mis:.4f}")
            
        # Plot 
        print("\nAll Results:")
        print("Steps:", test_steps)
        print("MIS Sizes:", results_mis)
        plot_fixed_step_performance(test_steps, results_mis)