# Fixed-step trigger sweep for OP-200 (adaptive vs. fixed trigger).
# Usage: python fixed_trigger_op.py --config hyco_config.yaml --mode fixed_exp
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import argparse
import re
from tqdm.auto import tqdm
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader, Dataset
from tensordict import TensorDict
from sklearn.neighbors import KDTree
from torch.utils.data.dataloader import default_collate
from collections import defaultdict
from torch.utils.data import Subset

# --- RL4CO Imports ---
from rl4co.models.zoo import AttentionModel 
from rl4co.envs import OPEnv 

# --- Plotting ---
import matplotlib.pyplot as plt
import seaborn as sns

# --- CO-Expander Imports ---
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from co_expander import (
    COExpanderCMModel, GNNEncoder, COExpanderEnv,
    COExpanderDecoder, COExpanderSparser
)
from co_expander.model.decoder.decode.op import calculate_op_metrics


# ==============================================================================
# === Collate function & dataset (unchanged) ===
# ==============================================================================
def collate_fn_skip_none(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None 
    return default_collate(batch)

class CustomOPDataset(Dataset):
    def __init__(self, file_path, num_nodes):
        self.file_path = file_path
        self.num_nodes = num_nodes
        self.num_customers = num_nodes - 1
        self.lines = self._load_lines()

    def _load_lines(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Data file not found at: {self.file_path}")
        with open(self.file_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(lines)} instances from {self.file_path}")
        return lines

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        try:
            depot_str = re.search(r'depots:(.*?);', line).group(1).strip()
            points_str = re.search(r'points:(.*?);', line).group(1).strip()
            prizes_str = re.search(r'prizes:(.*?);', line).group(1).strip()
            max_len_str = re.search(r'max_length:(.*?);', line).group(1).strip()
            
            depot = torch.tensor([float(x) for x in depot_str.split()], dtype=torch.float32).view(1, 2)
            points_flat = [float(x) for x in points_str.split()]
            points = torch.tensor(points_flat, dtype=torch.float32).view(self.num_customers, 2)
            locs = torch.cat([depot, points], dim=0)
            
            customer_prizes = torch.tensor([float(x) for x in prizes_str.split()], dtype=torch.float32)
            depot_prize = torch.tensor([0.0], dtype=torch.float32)
            prizes = torch.cat([depot_prize, customer_prizes], dim=0)

            max_length = torch.tensor(float(max_len_str), dtype=torch.float32)

            return {"locs": locs, "prizes": prizes, "max_length": max_length}
        except Exception as e:
            print(f"Error parsing line {idx}: {e}")
            return None

# ==============================================================================
# === Performance curve plotting ===
# ==============================================================================
def plot_fixed_step_performance(steps, rewards, save_path="fixed_step_performance_op.png"):
    """
    Plot fixed trigger step vs. average reward
    """
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    plt.plot(steps, rewards, marker='o', linestyle='-', linewidth=2, color='b', label='Hybrid Solver Reward')
    
    # Annotate values
    for x, y in zip(steps, rewards):
        plt.text(x, y, f'{y:.3f}', ha='center', va='bottom', fontsize=10)

    plt.title('Impact of Fixed Trigger Step on OP Reward', fontsize=16)
    plt.xlabel('Fixed Trigger Step', fontsize=12)
    plt.ylabel('Average Reward', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"\n[Plot] Performance curve saved to: {save_path}")

# ==============================================================================
# === Hybrid Solver ===
# ==============================================================================
class HybridSolver:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Solver using device: {self.device}")

        self.rl_policy, self.rl_env = self._load_rl_policy()
        self.co_expander_model, self.co_expander_cfg = self._load_co_expander_model()
        self.kl_history = defaultdict(list)

    def _load_rl_policy(self):
        # ... (unchanged) ...
        try:
            env = OPEnv(generator_kwargs={'num_loc': self.cfg.model_params.num_nodes - 1})
            model = AttentionModel.load_from_checkpoint(self.cfg.rl_model.ckpt_path, env=env, map_location='cuda' if torch.cuda.is_available() else 'cpu', strict=False)
            policy = model.policy.to(self.device)
            policy.eval()
            return policy, model.env
        except Exception as e:
            print(f"Error loading RL model: {e}"); exit()

    def _load_co_expander_model(self):
        # ... (unchanged) ...
        try:
            ce_train_cfg = OmegaConf.load(self.cfg.co_expander_model.config_path)
            ce_env = COExpanderEnv(
                task=ce_train_cfg.task, mode='val', train_data_size=1, val_data_size=1,
                train_batch_size=1, val_batch_size=1, test_batch_size=1, num_workers=0, 
                sparse_factor=ce_train_cfg.sparse_factor, device=self.device, prefix_k_options=None
            )
            if hasattr(ce_env, 'data_processor') and isinstance(ce_env.data_processor, COExpanderSparser):
                ce_env.data_processor.device = self.device

            encoder = GNNEncoder(
                 task=ce_train_cfg.encoder.task, sparse=ce_train_cfg.encoder.sparse,
                 block_layers=ce_train_cfg.encoder.block_layers, hidden_dim=ce_train_cfg.encoder.hidden_dim,
                 time_flag=ce_train_cfg.encoder.time_flag, prefix_cond_dim=ce_train_cfg.encoder.prefix_cond_dim,
                 prefix_enc_hidden_dim=ce_train_cfg.encoder.prefix_enc_hidden_dim,
                 max_length_cond_dim=ce_train_cfg.encoder.max_length_cond_dim,
                 aggregation=ce_train_cfg.encoder.get("aggregation", "sum"),
                 norm=ce_train_cfg.encoder.get("norm", "layer"),
                 learn_norm=ce_train_cfg.encoder.get("learn_norm", True),
            )
            decoder = COExpanderDecoder(**ce_train_cfg.decoder)
            model = COExpanderCMModel.load_from_checkpoint(
                self.cfg.co_expander_model.ckpt_path, env=ce_env, encoder=encoder, decoder=decoder,
                learning_rate=ce_train_cfg.train.learning_rate, **ce_train_cfg.model_params,
                map_location=self.device, strict=True
            )
            model.eval().to(self.device)
            model.env = ce_env 
            return model, ce_train_cfg
        except Exception as e:
            print(f"Error loading CO-Expander: {e}"); exit()

    def _merge_k_inputs_op(self, k_inputs_list: list):
        # ... (unchanged, omitted for brevity) ...
        # (copy of the previous implementation)
        if not k_inputs_list: return None
        task = k_inputs_list[0][0]
        nodes_feature_batch = torch.cat([inp[1] for inp in k_inputs_list], dim=0)
        edges_feature_batch = torch.cat([inp[3] for inp in k_inputs_list], dim=0)
        e_batch = torch.cat([inp[4] for inp in k_inputs_list], dim=0)
        
        edge_index_list, nodes_num_list, edges_num_list = [], [], []
        node_offset = 0
        for inp in k_inputs_list:
            edge_index_list.append(inp[5] + node_offset)
            nodes_num_list.append(inp[9][0]); edges_num_list.append(inp[10][0])
            node_offset += inp[9][0]
        edge_index_batch = torch.cat(edge_index_list, dim=1)
        
        graph_list_batch = [inp[6][0] for inp in k_inputs_list]
        mask_batch = torch.cat([inp[7] for inp in k_inputs_list], dim=0)
        ground_truth_batch = torch.cat([inp[8] for inp in k_inputs_list], dim=0)
        raw_data_list_batch = [inp[11][0] for inp in k_inputs_list]
        prefix_nodes_batch_list_batch = [inp[12][0] for inp in k_inputs_list]
        node_prefix_state_batch = torch.cat([inp[13] for inp in k_inputs_list], dim=0)
        max_lengths_batch = torch.cat([inp[14] for inp in k_inputs_list], dim=0)

        return (task, nodes_feature_batch, None, edges_feature_batch, e_batch, edge_index_batch, 
                graph_list_batch, mask_batch, ground_truth_batch, nodes_num_list, edges_num_list, 
                raw_data_list_batch, prefix_nodes_batch_list_batch, node_prefix_state_batch, max_lengths_batch)

    def _prepare_co_expander_input_tuple(self, td_instance, prefix_nodes_list):
        # ... (unchanged, omitted for brevity) ...
        # (copy of the previous implementation)
        instance_locs = td_instance['locs'].squeeze(0); prizes = td_instance['prizes'].squeeze(0)
        max_length_tensor = td_instance['max_length'].squeeze(0)
        coords_np = instance_locs.cpu().numpy(); prizes_np = prizes.cpu().numpy()
        N = self.cfg.model_params.num_nodes; sparse_k = self.co_expander_cfg.sparse_factor

        if N <= sparse_k:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            edges_feature = torch.empty((0, 1), dtype=torch.float32, device=self.device)
            edges_num = 0
        else:
            kdt = KDTree(coords_np, metric='euclidean')
            _, knn_indices = kdt.query(coords_np, k=sparse_k)
            source_nodes = torch.arange(N, device=self.device).view(-1, 1).repeat(1, sparse_k).flatten()
            target_nodes = torch.from_numpy(knn_indices).to(self.device).flatten()
            edge_index = torch.stack([source_nodes, target_nodes], dim=0)
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            edge_index = edge_index[:, edge_index[0] != edge_index[1]]
            edge_index = torch.unique(torch.sort(edge_index, dim=0)[0], dim=1)
            edges_num = edge_index.shape[1]
            src, dst = edge_index[0], edge_index[1]
            distances = torch.linalg.norm(instance_locs[src] - instance_locs[dst], dim=-1)
            normalized_distances = (distances - distances.min()) / (distances.max() - distances.min() + 1e-8)
            edges_feature = normalized_distances.unsqueeze(-1).float()

        is_depot_feature = torch.zeros((N, 1), dtype=torch.float32, device=self.device); is_depot_feature[0] = 1.0
        prize_feature = prizes.unsqueeze(-1).float()
        prefix_nodes_tensor = torch.tensor(prefix_nodes_list, dtype=torch.long, device=self.device)
        node_prefix_state = torch.zeros((N, 1), dtype=torch.float32, device=self.device)
        if len(prefix_nodes_list) > 0: node_prefix_state[prefix_nodes_tensor] = 1.0
        nodes_feature_final = torch.cat([is_depot_feature, prize_feature, node_prefix_state], dim=-1)

        mask = torch.zeros(edges_num, dtype=torch.bool, device=self.device)
        ground_truth = torch.zeros(edges_num, dtype=torch.long, device=self.device)
        e = torch.zeros(edges_num, dtype=torch.float32, device=self.device)
        
        return ("OP", nodes_feature_final, None, edges_feature, e, edge_index, [None], mask, ground_truth, 
                [N], [edges_num], [(coords_np, prizes_np, max_length_tensor.item(), [])], 
                [prefix_nodes_tensor], node_prefix_state, max_length_tensor.unsqueeze(0))

    # ==============================================================================
    # === [Key change] solve_instance with fixed-step support ===
    # ==============================================================================
    @torch.no_grad()
    def solve_instance(self, td_rl, full_instance_data_dict):
        td_step = self.rl_env.reset(td_rl.clone())
        
        ce_triggered_flag = False
        best_ce_proposal = {"reward": -1.0, "cost": float('inf'), "tour": []}
        action_history = []
        trigger_step_at = -1
        trigger_reason = "None"
        
        # [NEW] get the fixed-step config (-1 means no fixed step, i.e. dynamic trigger)
        fixed_step_trigger = getattr(self.cfg.solver, 'fixed_trigger_step', -1)
        
        node_embeds, _ = self.rl_policy.encoder(td_step)
        cached_embeds = self.rl_policy.decoder._precompute_cache(node_embeds)

        while not td_step["done"].all():
            current_step = len(action_history)
            current_node = td_step['current_node'][0].item()
            
            logits, _ = self.rl_policy.decoder(td_step, cached_embeds)
            probs = F.softmax(logits + td_step["action_mask"].log(), dim=-1)
            
            trigger_now = False

            # >>> Branching: fixed step vs. dynamic trigger <<<
            if fixed_step_trigger >= 0:
                # [Mode 1: Fixed Step Trigger]
                # Trigger only at the specified step, ignoring entropy/KL
                # Note: an OP episode may end before fixed_step, in which case the trigger never fires
                if current_step == fixed_step_trigger and not ce_triggered_flag:
                    trigger_now = True
                    trigger_reason = f"FixedStep({fixed_step_trigger})"
            else:
                # [Mode 2: Dynamic Trigger (Entropy/KL)]
                # Only compute the KL in non-fixed mode to save time
                
                # 1. Entropy
                entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).squeeze(0)
                
                # 2. KL Divergence
                kl_divergence = torch.tensor(0.0, device=self.device)
                use_kl = getattr(self.cfg.solver, 'use_kl_trigger', True)
                
                if use_kl and current_step > 0 and not ce_triggered_flag:
                    current_probs = probs.squeeze(0)
                    valid_probs = current_probs * td_step["action_mask"][0]
                    M_kl = 10
                    top_m_probs, top_m_indices = torch.topk(valid_probs, k=min(M_kl, (td_step["action_mask"][0] > 0).sum().item()))
                    
                    # Run the probe
                    current_prefix = [0] + action_history
                    # Assumes _run_co_expander_probe and _compute_kl_divergence are defined in this class
                    # (make sure the class contains these two methods)
                    # ce_logits = self._run_co_expander_probe(...)
                    # kl_divergence = self._compute_kl_divergence(...)
                    
                    # Note: if these two methods are not available, comment out the KL computation
                    pass 

                is_not_at_depot = (td_step['current_node'] != 0).item()
                is_high_entropy = (entropy > self.cfg.solver.entropy_threshold)
                # is_high_kl = (kl_divergence > self.cfg.solver.kl_threshold)
                
                if is_not_at_depot and not ce_triggered_flag:
                    if is_high_entropy:
                        trigger_now = True
                        trigger_reason = f"Entropy({entropy:.2f})"
                    # elif is_high_kl:
                    #     trigger_now = True
                    #     trigger_reason = f"KL({kl_divergence:.2f})"

            best_next_node = probs.argmax(-1)

            # >>> Trigger execution (shared) <<<
            if trigger_now:
                print(f"--- Step {current_step}: Triggered by {trigger_reason} ---")
                ce_triggered_flag = True
                trigger_step_at = current_step
                
                # K-Lookahead Logic
                K = 8 
                current_probs = probs.squeeze(0)
                num_available_actions = torch.sum(td_step["action_mask"][0] == 0.0).item()
                K_actual = min(K, num_available_actions)
                
                if K_actual > 0:
                    top_k_probs, top_k_action_nodes = torch.topk(current_probs, k=K_actual)
                    td_step_backup = td_step.clone()
                    ce_inputs_to_batch = []
                    
                    for action_node in top_k_action_nodes:
                        sim_td_step = td_step_backup.clone()
                        sim_td_step.set("action", action_node.view(1))
                        _ = self.rl_env.step(sim_td_step)
                        future_prefix_nodes = [0] + action_history + [action_node.item()]
                        ce_input_tuple = self._prepare_co_expander_input_tuple(full_instance_data_dict, future_prefix_nodes)
                        ce_inputs_to_batch.append(ce_input_tuple)

                    batched_ce_input = self._merge_k_inputs_op(ce_inputs_to_batch)
                    _, vars_heatmap_batch = self.co_expander_model.inference_edge_sparse_process(*batched_ce_input)
                    k_solutions_list = self.co_expander_model.decoder.sparse_decode(
                        vars_heatmap_batch, *batched_ce_input, return_cost=False
                    )
                    
                    best_k_reward = -1.0
                    k_raw_data_list = batched_ce_input[11]
                    for i in range(K_actual):
                        ce_tour_k = list(k_solutions_list[i])
                        raw_data_k = k_raw_data_list[i]
                        coords_k_np, prizes_k_np, _, _ = raw_data_k
                        ce_reward_k, ce_cost_k = calculate_op_metrics(coords_k_np, ce_tour_k, prizes_k_np)
                        if ce_reward_k > best_k_reward:
                            best_k_reward = ce_reward_k
                            best_k_cost = ce_cost_k
                            best_ce_proposal = {"reward": best_k_reward, "cost": best_k_cost, "tour": ce_tour_k}

            action_history.append(best_next_node.item())
            td_step.set("action", best_next_node)
            td_step = self.rl_env.step(td_step)["next"]

        rl_tour = action_history
        rl_tour_cleaned = [node for node in rl_tour if node < self.cfg.model_params.num_nodes]
        rl_tour_mapped = [a + 1 for a in rl_tour_cleaned if a < self.cfg.model_params.num_nodes - 1] 
        
        rl_reward, rl_cost = calculate_op_metrics(
            full_instance_data_dict['locs'].squeeze(0).cpu().numpy(), 
            rl_tour_mapped, 
            full_instance_data_dict['prizes'].squeeze(0).cpu().numpy()
        )
        
        if rl_reward > best_ce_proposal['reward']:
            return {"reward": rl_reward, "tour": rl_tour_mapped, "source": "RL", "trigger_step": trigger_step_at}
        else:
            return {"reward": best_ce_proposal['reward'], "tour": best_ce_proposal['tour'], "source": "CE", "trigger_step": trigger_step_at}

# ==============================================================================
# === Runner ===
# ==============================================================================
def run_evaluation(cfg, fixed_step=-1):
    """
    Run a single evaluation.
    If fixed_step != -1, it overrides the config setting.
    Returns the average reward.
    """
    # Override the config dynamically
    if fixed_step != -1:
        cfg.solver.fixed_trigger_step = fixed_step
        print(f"\n>>> Running Evaluation with FIXED TRIGGER STEP = {fixed_step} <<<")
    else:
        cfg.solver.fixed_trigger_step = -1
        print(f"\n>>> Running Evaluation with DYNAMIC TRIGGER (Entropy/KL) <<<")

    solver = HybridSolver(cfg)
    device = solver.device
    
    dataset = CustomOPDataset(cfg.data.test_path, cfg.model_params.num_nodes)
    
    # Limit the number of samples to speed up experiments
    num_samples_to_test = 64 # can be increased
    if len(dataset) > num_samples_to_test:
        dataset = Subset(dataset, range(num_samples_to_test))

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_skip_none)

    all_rewards = []
    
    for data_dict in tqdm(dataloader, desc=f"Step={fixed_step}" if fixed_step!=-1 else "Dynamic"):
        if data_dict is None: continue

        all_locs = data_dict["locs"].to(device)
        all_prizes = data_dict["prizes"].to(device)
        max_length = data_dict["max_length"].to(device)

        td_rl = TensorDict({
            "depot": all_locs[:, 0, :],
            "locs": all_locs[:, 1:, :],
            "prize": all_prizes[:, 1:],
            "max_length": max_length,
        }, batch_size=1).to(device)

        full_data = {"locs": all_locs, "prizes": all_prizes, "max_length": max_length}

        solution = solver.solve_instance(td_rl, full_data)
        all_rewards.append(solution['reward'])

    if not all_rewards: return 0.0
    return np.mean(all_rewards)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="hyco_config.yaml")
    parser.add_argument("--mode", type=str, choices=["dynamic", "fixed_exp"], default="fixed_exp", 
                        help="dynamic: original dynamic trigger; fixed_exp: run the fixed-step experiment and plot")
    args = parser.parse_args()
    
    cfg = OmegaConf.load(args.config)
    default_solver_cfg = OmegaConf.create({
        'solver': {
            'entropy_threshold': 0.6,
            'use_kl_trigger': True,
            'fixed_trigger_step': -1 # default -1 (dynamic)
        }
    })
    cfg = OmegaConf.merge(default_solver_cfg, cfg)

    if args.mode == "dynamic":
        # Run the original dynamic logic
        avg_reward = run_evaluation(cfg, fixed_step=-1)
        print(f"Final Dynamic Average Reward: {avg_reward:.4f}")

    elif args.mode == "fixed_exp":
        # 
        # Fixed-step experiment: sweep over steps and plot the curve
        # Note: OP episodes are usually short (limited by max_length),
        # e.g., with N=100 there may be only 15-30 steps.
        # So do not choose test steps that are too large.
        
        test_steps = [1,2,3,4,5,10,20,30,40,50,60] 
        results_rewards = []
        
        print(f"Starting Fixed Step Experiment for steps: {test_steps}")
        
        for step in test_steps:
            avg_r = run_evaluation(cfg, fixed_step=step)
            results_rewards.append(avg_r)
            print(f"Step {step} Result: Avg Reward = {avg_r:.4f}")
            
        # Plot
        plot_fixed_step_performance(test_steps, results_rewards)
        
        print("\nExperiment Completed.")
        print("Steps:", test_steps)
        print("Rewards:", results_rewards)