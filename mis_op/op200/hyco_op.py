# HyCO for the Orienteering Problem: AM (RL, rl4co) constructs a prefix, the prefix-conditioned CO-Expander completes it.
# Usage: python hyco_op.py --config hyco_config.yaml
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
from torch.utils.data.dataloader import default_collate # <-- new
from collections import defaultdict # <--- new
# --- RL4CO imports (for OP) ---
from rl4co.models.zoo import AttentionModel 
from rl4co.envs import OPEnv 
# --- [Added] ---
# For plotting and data analysis
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch.utils.data import Subset
# --- [End] ---
# --- CO-Expander imports (replaces the diffusion model) ---
import sys
# [CRITICAL] make sure these paths are correct relative to the working directory
# co_expander is assumed to be in the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from co_expander import (
    COExpanderCMModel, GNNEncoder, COExpanderEnv,
    COExpanderDecoder, COExpanderSparser
)
# Import OP metric computation from CO-Expander
from co_expander.model.decoder.decode.op import calculate_op_metrics


# ==============================================================================
# === Collate function ===
# ==============================================================================
def collate_fn_skip_none(batch):
    """
    Collate function that filters out None values from the batch.
    'batch' is a list of results from __getitem__.
    """
    batch = [item for item in batch if item is not None]
    if not batch:
        return None 
    return default_collate(batch)


# ==============================================================================
# === Data loader ===
# ==============================================================================
class CustomOPDataset(Dataset):
    """
    Parses OP datasets in the format
    "Instance: ...; depots: ...; points: ...; prizes: ...; max_length: ...".
    """
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
            # depot_str = re.search(r'depots:([\d.\s]+);', line).group(1).strip()
            # points_str = re.search(r'points:([\d.\s]+);', line).group(1).strip()
            # prizes_str = re.search(r'prizes:([\d.\s]+);', line).group(1).strip()
            # max_len_str = re.search(r'max_length:([\d.]+);', line).group(1).strip()
# --- [CRITICAL FIX] ---
            # Use a robust non-greedy pattern (.*?) to capture all characters,
            # which correctly handles scientific notation (e.g., "1.23e-05")
            depot_str = re.search(r'depots:(.*?);', line).group(1).strip()
            points_str = re.search(r'points:(.*?);', line).group(1).strip()
            prizes_str = re.search(r'prizes:(.*?);', line).group(1).strip()
            max_len_str = re.search(r'max_length:(.*?);', line).group(1).strip()
            # --- [END FIX] ---
            
            depot = torch.tensor([float(x) for x in depot_str.split()], dtype=torch.float32).view(1, 2)
            
            points_flat = [float(x) for x in points_str.split()]
            expected_points = self.num_customers * 2
            if len(points_flat) != expected_points:
                print(f"Warning: Instance {idx} (points) has {len(points_flat)//2} customers, expected {self.num_customers}. Skipping.")
                return None 
            
            points = torch.tensor(points_flat, dtype=torch.float32).view(self.num_customers, 2)
            locs = torch.cat([depot, points], dim=0) # Shape: [101, 2]
            
            customer_prizes = torch.tensor([float(x) for x in prizes_str.split()], dtype=torch.float32) # Shape: [100]
            if customer_prizes.shape[0] != self.num_customers:
                print(f"Warning: Instance {idx} (prizes) has {customer_prizes.shape[0]} prizes, expected {self.num_customers}. Skipping.")
                return None
            
            depot_prize = torch.tensor([0.0], dtype=torch.float32) # Shape: [1]
            prizes = torch.cat([depot_prize, customer_prizes], dim=0) # Shape: [101]

            max_length = torch.tensor(float(max_len_str), dtype=torch.float32)

            if locs.shape[0] != self.num_nodes or prizes.shape[0] != self.num_nodes:
                print(f"Warning: Instance {idx} final shape mismatch. Locs: {locs.shape[0]}, Prizes: {prizes.shape[0]}. Skipping.")
                return None 

            return {
                "locs": locs,
                "prizes": prizes,
                "max_length": max_length
            }

        except Exception as e:
            print(f"Error parsing line {idx}: {line}\nError: {e}")
            return None


def plot_distribution(steps, save_path, num_nodes):
    """
    Plot the distribution of trigger steps (histogram + KDE) with matplotlib and seaborn.
    
    Args:
        steps (list): all trigger steps (with -1 filtered out).
        save_path (str): output image path (e.g., "dist.png").
        num_nodes (int): total number of nodes, used for the x-axis range.
    """
    if not steps:
        print("No trigger data to plot.")
        return

    try:
        plt.figure(figsize=(12, 7))
        sns.set_theme(style="whitegrid")
        
        # Histogram and kernel density estimate (KDE)
        # stat="count" shows the number of instances
        # kde=True draws a smooth distribution curve
        ax = sns.histplot(steps, kde=True, bins=min(50, num_nodes), stat="count")
        
        ax.set_title(f'Distribution of CO-Expander Trigger Steps (Total Triggered = {len(steps)})', fontsize=16)
        ax.set_xlabel('Trigger Step (Length of RL Prefix)', fontsize=12)
        ax.set_ylabel('Number of Instances (Count)', fontsize=12)
        ax.set_xlim(0, num_nodes) # x-axis from 0 to the total number of nodes
        
        # Add mean and median
        mean_val = np.mean(steps)
        median_val = np.median(steps)
        
        plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean ({mean_val:.2f})')
        plt.axvline(median_val, color='green', linestyle=':', label=f'Median ({median_val:.0f})')
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path) # Save the figure
        plt.close() # Close the figure to avoid accumulating memory
        
        print(f"\nDistribution plot saved to: {save_path}")

    except ImportError:
        print("\nError: matplotlib or seaborn not found. Cannot generate plot.")
        print("Please install them: pip install matplotlib seaborn")
        # Fallback: if plotting fails, save the raw data as a .csv file
        csv_path = save_path.replace(".png", ".csv").replace(".pdf", ".csv")
        try:
            np.savetxt(csv_path, 
                       np.array(steps), 
                       delimiter=',', 
                       header='trigger_step',
                       comments='')
            print(f"Saved raw trigger step data to {csv_path} instead.")
        except Exception as e:
            print(f"Failed to save raw data to CSV: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred during plotting: {e}")

# ==============================================================================
# === Hybrid solver (RL + CO-Expander) ===
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
        print(f"Loading RL model for {self.cfg.rl_model.problem} from: {self.cfg.rl_model.ckpt_path}")
        try:
            num_loc = self.cfg.model_params.num_nodes - 1
            
            # --- [FIX for rl4co] ---
            env = OPEnv(
                generator_kwargs={'num_loc': self.cfg.model_params.num_nodes - 1}
            )
            # --- [END FIX] ---
            
            model = AttentionModel.load_from_checkpoint(
                self.cfg.rl_model.ckpt_path,
                env=env, map_location='cuda' if torch.cuda.is_available() else 'cpu',
                strict=False 
            )
            policy = model.policy.to(self.device)
            policy.eval()
            return policy, model.env
        except Exception as e:
            print(f"Error loading RL model: {e}"); exit()

    def _load_co_expander_model(self):
        """
        Load the CO-Expander model.
        """
        print(f"Loading CO-Expander model from: {self.cfg.co_expander_model.ckpt_path}")
        print(f"Using training config: {self.cfg.co_expander_model.config_path}")
        
        try:
            ce_train_cfg = OmegaConf.load(self.cfg.co_expander_model.config_path)
            
            # --- [FIX from previous turn] ---
            print("Instantiating minimal COExpanderEnv for model loading...")
            ce_env = COExpanderEnv(
                task=ce_train_cfg.task, mode='val', train_data_size=1, val_data_size=1,
                train_batch_size=1, val_batch_size=1, test_batch_size=1,
                num_workers=0, sparse_factor=ce_train_cfg.sparse_factor,
                device=self.device, train_folder=None, val_path=None,
                store_data=False, prefix_k_options=None
            )
            if hasattr(ce_env, 'data_processor') and isinstance(ce_env.data_processor, COExpanderSparser):
                ce_env.data_processor.device = self.device
            # --- [END FIX] ---

            encoder = GNNEncoder(
                 task=ce_train_cfg.encoder.task, sparse=ce_train_cfg.encoder.sparse,
                 block_layers=ce_train_cfg.encoder.block_layers, hidden_dim=ce_train_cfg.encoder.hidden_dim,
                 time_flag=ce_train_cfg.encoder.time_flag, prefix_cond_dim=ce_train_cfg.encoder.prefix_cond_dim,
                 prefix_enc_hidden_dim=ce_train_cfg.encoder.prefix_enc_hidden_dim,
                 max_length_cond_dim=ce_train_cfg.encoder.max_length_cond_dim,
                 aggregation=ce_train_cfg.encoder.get("aggregation", "sum"),
                 norm=ce_train_cfg.encoder.get("norm", "layer"),
                 learn_norm=ce_train_cfg.encoder.get("learn_norm", True),
                 track_norm=ce_train_cfg.encoder.get("track_norm", False),
                 mask_frozen=ce_train_cfg.encoder.get("mask_frozen", False)
            )
            decoder = COExpanderDecoder(**ce_train_cfg.decoder)
            
            model = COExpanderCMModel.load_from_checkpoint(
                self.cfg.co_expander_model.ckpt_path,
                env=ce_env, # <--- pass env
                encoder=encoder, decoder=decoder,
                learning_rate=ce_train_cfg.train.learning_rate,
                **ce_train_cfg.model_params,
                map_location=self.device,
                strict=True
            )
            
            model.eval()
            model.to(self.device)
            model.env = ce_env 
            print("CO-Expander model loaded successfully.")
            return model, ce_train_cfg

        except Exception as e:
            print(f"Error loading CO-Expander model: {e}")
            import traceback
            traceback.print_exc() 
            print("Please ensure co_expander code is in python path and config/ckpt paths are correct.")
            exit()

# ==============================================================================
    # === K-lookahead batching helper ===
    # ==============================================================================
    def _merge_k_inputs_op(self, k_inputs_list: list):
        """
        Merge K CO-Expander (OP) input tuples (15 elements each)
        into a single input tuple with batch_size=K.
        
        Mirrors the merge_process logic in co_expander/env/sparser.py
        """
        if not k_inputs_list:
            return None
        
        K = len(k_inputs_list)
        
        # 0. task (str)
        task = k_inputs_list[0][0] # keep "OP"
        
        # 1. nodes_feature (torch.cat) [N_total, 3]
        nodes_feature_list = [inp[1] for inp in k_inputs_list]
        nodes_feature_batch = torch.cat(nodes_feature_list, dim=0)
        
        # 2. x (None)
        x_batch = None
        
        # 3. edges_feature (torch.cat) [E_total, 1]
        edges_feature_list = [inp[3] for inp in k_inputs_list]
        edges_feature_batch = torch.cat(edges_feature_list, dim=0)
        
        # 4. e (torch.cat) [E_total]
        e_list = [inp[4] for inp in k_inputs_list]
        e_batch = torch.cat(e_list, dim=0)

        # 5. edge_index (torch.cat with offset) [2, E_total]
        # 9. nodes_num_list
        # 10. edges_num_list
        edge_index_list_to_cat = []
        nodes_num_list = [] 
        edges_num_list = [] 
        node_offset = 0
        
        for inp in k_inputs_list:
            edge_index_k = inp[5]   # [2, E_k]
            nodes_num_k = inp[9][0] # inp[9] is a list of length N_k
            edges_num_k = inp[10][0]# inp[10] is a list of length E_k
            
            edge_index_list_to_cat.append(edge_index_k + node_offset)
            
            nodes_num_list.append(nodes_num_k)
            edges_num_list.append(edges_num_k)
            node_offset += nodes_num_k
            
        edge_index_batch = torch.cat(edge_index_list_to_cat, dim=1)

        # 6. graph_list (list of None)
        graph_list_batch = [inp[6][0] for inp in k_inputs_list] # [None, None, ...]
        
        # 7. mask (torch.cat) [E_total]
        mask_list = [inp[7] for inp in k_inputs_list]
        mask_batch = torch.cat(mask_list, dim=0)
        
        # 8. ground_truth (torch.cat) [E_total]
        ground_truth_list = [inp[8] for inp in k_inputs_list]
        ground_truth_batch = torch.cat(ground_truth_list, dim=0)
        
        # 11. raw_data_list (list)
        raw_data_list_batch = [inp[11][0] for inp in k_inputs_list]
        
        # 12. prefix_nodes_batch_list (list of tensors)
        prefix_nodes_batch_list_batch = [inp[12][0] for inp in k_inputs_list]

        # 13. node_prefix_state (torch.cat) [N_total, 1]
        node_prefix_state_list = [inp[13] for inp in k_inputs_list]
        node_prefix_state_batch = torch.cat(node_prefix_state_list, dim=0)
        
        # 14. max_lengths (torch.cat) [K]
        max_lengths_list = [inp[14] for inp in k_inputs_list]
        max_lengths_batch = torch.cat(max_lengths_list, dim=0)

        # Return a 15-element tuple compatible with co_expander
        return (
            task, nodes_feature_batch, x_batch, edges_feature_batch, e_batch, 
            edge_index_batch, graph_list_batch, mask_batch, ground_truth_batch, 
            nodes_num_list, edges_num_list, raw_data_list_batch,
            prefix_nodes_batch_list_batch, node_prefix_state_batch, max_lengths_batch
        )


    @torch.no_grad()
    def _run_co_expander_probe(self, full_instance_data_dict, current_prefix, current_node, candidate_nodes):
        """
        Run one CO-Expander forward pass to get the global model's preference for the next step.
        Returns: the logits (unnormalized scores) of candidate_nodes.
        """
        # 1. Prepare the input for a single instance (batch size = 1)
        # Build the input from the current prefix
        ce_input_tuple = self._prepare_co_expander_input_tuple(
            full_instance_data_dict, 
            current_prefix
        )
        
        # 2. Unpack the required arguments (in the order returned by _prepare_co_expander_input_tuple)
        (task, nodes_feature, x, edges_feature, e, edge_index, graph_list, 
         mask, ground_truth, nodes_num_list, edges_num_list, raw_data_list,
         prefix_nodes, node_prefix_state, max_lengths) = ce_input_tuple

        # 3. Set the probe timestep t
        # We need neither full noise (t=1000) nor necessarily a clean input (t=0).
        # t=0 means the model predicts the final solution directly from the current context.
        # t=50 means slight noise, testing the model's ability to recover (energy).
        probe_t_val = getattr(self.cfg.solver, 'probe_noise_t', 0) 
        t = torch.tensor([probe_t_val], device=self.device).float()
        if max_lengths is not None:
            t = t.repeat(max_lengths.size(0))

        # 4. Run the forward pass (once, no loop)
        # Note: no mask update is needed here, only the raw model prediction
        # Pass the current e (1 on prefix edges, 0 elsewhere)
        # Key: the mask must contain the prefix edges so the model knows they are fixed
        
        # Build is_prefix_edge_mask (same logic as in train_edge_sparse_process)
        # For simplicity, assume the mask initialized by _prepare_co_expander_input_tuple is all zeros
        # Prefix edges must be marked True manually so that the GNN knows about them.
        # (For performance it is best if prepare already handles the mask; otherwise simplified here:
        #  the probe mainly looks at the predictions for *unknown* edges, so an all-zero mask also works,
        #  but to use the prefix information it is better to set e to 1 on prefix edges)
        
        # _prepare_co_expander_input_tuple already sets node_prefix_state to 1 for prefix nodes.
        # For the sparse GNN, the input e also matters.
        # Assume the GNN can infer it from node_prefix_state.
        
        # Call the model forward
        _, e_pred_logits = self.co_expander_model.model.forward(
            task=task, focus_on_node=False, focus_on_edge=True,
            nodes_feature=nodes_feature, x=x, edges_feature=edges_feature,
            e=e, # initial e (all zeros)
            mask=mask, # initial mask (all False)
            t=t, 
            edge_index=edge_index,
            node_prefix_state=node_prefix_state, 
            prefix_nodes=prefix_nodes, 
            nodes_num_list=nodes_num_list,
            max_lengths=max_lengths
        )
        
        # 5. Extract the logits of specific edges
        # e_pred_logits shape: [E, 2] (out_channels=2) or [E, 1]
        # Usually [E, 2]; take the logit at index 1 (edge present), or index 1 - index 0
        
        if e_pred_logits.dim() > 1 and e_pred_logits.shape[-1] == 2:
            edge_scores = e_pred_logits[:, 1] - e_pred_logits[:, 0] # Logit for class 1
        else:
            edge_scores = e_pred_logits.squeeze(-1)

        # 6. Map (current_node, candidate) to positions in edge_index
        # Find the indices of the edges in edge_index connecting current_node to the candidates
        # edge_index shape: [2, E]
        
        candidate_logits = []
        
        # Build a lookup table (a local mapping built once, for speed)
        # Since the graph is a sparse k-NN graph, some candidates may not be among the k-NN neighbors
        # If not, assign a very small logit (-inf)
        
        # Optimization: lookup via torch masks
        # Find all edges whose source or target is current_node
        src, dst = edge_index[0], edge_index[1]
        
        # Undirected graph (k-NN is usually symmetrized) or directed graph
        # Look up (current_node -> candidate)
        
        for cand in candidate_nodes:
            cand_idx = cand.item()
            # Look up edge: (current_node, cand_idx)
            # Note: edge_index is [2, E]
            # Mask: src == current_node AND dst == cand_idx
            mask_forward = (src == current_node) & (dst == cand_idx)
            
            if mask_forward.any():
                # Found
                idx = torch.nonzero(mask_forward, as_tuple=True)[0][0]
                candidate_logits.append(edge_scores[idx])
            else:
                # The edge is not in the k-NN graph, i.e. too far away; CO-Expander considers it very unlikely
                candidate_logits.append(torch.tensor(-100.0, device=self.device)) # Logit for ~0 prob
                
        return torch.stack(candidate_logits)

    def _compute_kl_divergence(self, rl_probs, ce_logits, temp=1.0):
        """
        Compute D_KL(P_CE || P_RL) or D_KL(P_RL || P_CE).
        Usually we compute the divergence of RL with respect to the "target" (CO-Expander).
        
        rl_probs: RL probabilities of the top-M candidates (normalized, sum to 1)
        ce_logits: CO-Expander logits of the top-M candidates (unnormalized)
        """
        # 1. Convert the CO-Expander logits into a probability distribution P_CE
        # Smooth with a temperature
        ce_log_probs = F.log_softmax(ce_logits / temp, dim=-1)
        ce_probs = torch.exp(ce_log_probs)
        
        # 2. Renormalize the RL probabilities (over the top-M only)
        # rl_probs already holds the top-M values, but they may not sum to 1 (taken from the full set)
        rl_probs_norm = rl_probs / (rl_probs.sum() + 1e-9)
        
        # 3. Compute the KL
        # KL(P || Q) = sum(P(x) * (log P(x) - log Q(x)))
        # Here we measure "inconsistency".
        # Common usage: how far RL (posterior) deviates from CO-Expander (prior) -> KL(RL || CE)
        # or CO-Expander (true distribution) vs. RL (approximation) -> KL(CE || RL)
        
        # The previous logic (DIFUSCO version) was:
        # log_p_dm = F.log_softmax(...)
        # p_rl = top_m_probs
        # kl = F.kl_div(log_p_dm, p_rl, reduction='none').sum()
        # PyTorch F.kl_div(input, target) computes KL(target || input) = sum(target * (log target - input))
        # so input should be log-probabilities (log P_model) and target probabilities (P_true)
        
        # Interpretation: if CO-Expander assigns a node high probability and RL low probability, the penalty is large.
        
        # We use KL(RL || CE) = sum(RL * (log RL - log CE))
        # i.e., if RL is very confident about a node that CO-Expander considers unlikely, the KL is large (conflict).
        
        kl_val = F.kl_div(
            ce_log_probs,       # input: log Q (CE)
            rl_probs_norm,      # target: P (RL)
            reduction='sum'
        )
        
        return kl_val


    def plot_kl_curve(self, save_path="kl_vs_step.png"):
        """
        Plot the KL divergence vs. inference step (mean ± std)
        """
        print("\nPlotting KL Divergence curve...")
        steps = sorted(self.kl_history.keys())
        
        if not steps:
            print("No KL data collected to plot (maybe KL trigger was disabled?).")
            return

        avg_kl = []
        std_kl = []
        counts = []
        
        valid_steps = []
        
        for step in steps:
            values = np.array(self.kl_history[step])
            # Filter out possible outliers or empty values
            if len(values) > 0:
                avg_kl.append(np.mean(values))
                std_kl.append(np.std(values))
                counts.append(len(values))
                valid_steps.append(step)
        
        if not valid_steps:
            print("No valid KL data found.")
            return

        avg_kl = np.array(avg_kl)
        std_kl = np.array(std_kl)
        valid_steps = np.array(valid_steps)

        plt.figure(figsize=(10, 6))
        
        # Plot the main curve (mean)
        plt.plot(valid_steps, avg_kl, label='Mean KL Divergence', color='blue', linewidth=2, marker='o', markersize=4)
        
        # Plot the shaded band (std)
        plt.fill_between(valid_steps, avg_kl - std_kl, avg_kl + std_kl, color='blue', alpha=0.2, label='Standard Deviation')
        
        plt.xlabel('Inference Step (Construction Step)', fontsize=12)
        plt.ylabel('KL Divergence (RL vs CO-Expander)', fontsize=12)
        plt.title(f'KL Divergence Trend over Steps (Aggregated from {max(counts)} instances)', fontsize=14)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        try:
            plt.savefig(save_path, dpi=300)
            print(f"✅ KL curve saved to: {save_path}")
        except Exception as e:
            print(f"Error saving plot: {e}")
        finally:
            plt.close()

    @torch.no_grad()
    def solve_instance(self, td_rl, full_instance_data_dict):
        td_step = self.rl_env.reset(td_rl.clone())
        
        ce_triggered_flag = False
        best_ce_proposal = {"reward": -1.0, "cost": float('inf'), "tour": []}
        action_history = []
        trigger_step_at = -1
        trigger_reason = "None" # Record the trigger reason
        
        node_embeds, _ = self.rl_policy.encoder(td_step)
        cached_embeds = self.rl_policy.decoder._precompute_cache(node_embeds)

        while not td_step["done"].all():
            current_step = len(action_history)
            current_node = td_step['current_node'][0].item() # Get the current node ID
            
            logits, _ = self.rl_policy.decoder(td_step, cached_embeds)
            probs = F.softmax(logits + td_step["action_mask"].log(), dim=-1)
            
            # 1. Compute the entropy
            entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).squeeze(0)
            
            # Get the RL top-M candidates for the KL computation
            M_kl = 10 # configurable
            current_probs = probs.squeeze(0)
            # Mask out infeasible actions
            valid_probs = current_probs * td_step["action_mask"][0]
            top_m_probs, top_m_indices = torch.topk(valid_probs, k=min(M_kl, (td_step["action_mask"][0] > 0).sum().item()))
            
            # 2. Compute the KL divergence (only if not the first step and not triggered yet)
            kl_divergence = torch.tensor(0.0, device=self.device)
            
            # Get config parameters
            use_kl = getattr(self.cfg.solver, 'use_kl_trigger', True) # enabled by default
            kl_thresh = getattr(self.cfg.solver, 'kl_threshold', 0.4) 
            entropy_thresh = self.cfg.solver.entropy_threshold
            
            if use_kl and current_step > 0 and not ce_triggered_flag:
                # Run the probe
                current_prefix = [0] + action_history # Depots + History
                
                ce_logits = self._run_co_expander_probe(
                    full_instance_data_dict, 
                    current_prefix, 
                    current_node, 
                    top_m_indices
                )
                
                kl_divergence = self._compute_kl_divergence(
                    top_m_probs, 
                    ce_logits, 
                    temp=getattr(self.cfg.solver, 'kl_temp', 1.0)
                )
                # --- Collect KL data ---
            # Only record when the KL was actually computed at this step, to avoid recording many 0.0 values
                print(f"Step {current_step}: KL Divergence = {kl_divergence.item():.4f}")
                self.kl_history[current_step].append(kl_divergence.item())
            # 3. Combined trigger condition

            is_not_at_depot = (td_step['current_node'] != 0).item()
            
            is_high_entropy = (entropy > 999)#entropy_thresh)
            is_high_kl = (kl_divergence > 999)#kl_thresh)
            
            trigger_now = False
            if is_not_at_depot and not ce_triggered_flag:
                if is_high_entropy:
                    trigger_now = True
                    trigger_reason = f"Entropy({entropy:.2f})"
                elif is_high_kl:
                    trigger_now = True
                    trigger_reason = f"KL({kl_divergence:.2f})"

            best_next_node = probs.argmax(-1)

            if trigger_now:
                print(f"--- Step {current_step}: Triggered by {trigger_reason} ---")
                ce_triggered_flag = True
                trigger_step_at = current_step
                
                # ... [K-lookahead logic] ...
                K = 8 
                # ... (Copy your existing K-Lookahead code here) ...
                
                # The K-lookahead logic below could be factored into a separate function
                # best_ce_proposal = self._run_k_lookahead(...)
                
                # --- BEGIN K-lookahead logic ---
                current_probs = probs.squeeze(0)
                num_available_actions = torch.sum(td_step["action_mask"][0] == 0.0).item()
                K_actual = min(K, num_available_actions)
                
                if K_actual > 0:
                    top_k_probs, top_k_action_nodes = torch.topk(current_probs, k=K_actual)
                    print(f"    Lookahead: Testing K={K_actual} nodes...")
                    td_step_backup = td_step.clone()
                    ce_inputs_to_batch = []
                    candidate_actions_list = []
                    for action_node in top_k_action_nodes:
                        sim_td_step = td_step_backup.clone()
                        sim_td_step.set("action", action_node.view(1))
                        # This step changes sim_td_step but does not affect the outside
                        _ = self.rl_env.step(sim_td_step) 
                        # Note: be careful, the RL4CO step may modify the internal state of the TensorDict
                        
                        future_prefix_nodes = [0] + action_history + [action_node.item()]
                        ce_input_tuple = self._prepare_co_expander_input_tuple(
                            full_instance_data_dict, future_prefix_nodes
                        )
                        ce_inputs_to_batch.append(ce_input_tuple)
                        candidate_actions_list.append(action_node)

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
                            
                    print(f"    Lookahead Result: Best CE Reward {best_k_reward:.4f}")
                # --- END K-Lookahead Logic ---

            # Execute the action
            
            # [unchanged]
            # Execute the finally selected action (from RL or from the CO-Expander lookahead)
            action_history.append(best_next_node.item())
            td_step.set("action", best_next_node)
            td_step = self.rl_env.step(td_step)["next"]

        rl_tour = action_history
        rl_tour_cleaned = [node for node in rl_tour if node < self.cfg.model_params.num_nodes]
        rl_tour_mapped = [a + 1 for a in rl_tour_cleaned if a < self.cfg.model_params.num_nodes - 1] 
        
        # (Fix from previous turn: add .numpy())
        rl_reward, rl_cost = calculate_op_metrics(
            full_instance_data_dict['locs'].squeeze(0).cpu().numpy(), 
            rl_tour_mapped, 
            full_instance_data_dict['prizes'].squeeze(0).cpu().numpy()
        )
        
        if rl_reward > best_ce_proposal['reward']:
            return {
                "reward": rl_reward, "cost": rl_cost, "tour": rl_tour_mapped, 
                "source": "RL", "trigger_step": trigger_step_at  # <-- add trigger_step
            }
        else:
            print(f"     Solution chosen from CO-Expander (R: {best_ce_proposal['reward']:.4f} vs RL R: {rl_reward:.4f})")
            return {
                "reward": best_ce_proposal['reward'], "cost": best_ce_proposal['cost'], 
                "tour": best_ce_proposal['tour'], "source": "CE", 
                "trigger_step": trigger_step_at  # <-- add trigger_step
            }
        
    # ==============================================================================
    # === [NEW] core helper ===
    # ==============================================================================
    def _prepare_co_expander_input_tuple(self, td_instance, prefix_nodes_list):
        """
        Manually build the 15-argument *tuple* required by the co_expander GNN and decoder.
        Mirrors op_batch_data_process and merge_process in co_expander/env/sparser.py
        for batch_size=1.
        """
        
        # 1. Extract instance data
        instance_locs = td_instance['locs'].squeeze(0) # [N, 2]
        prizes = td_instance['prizes'].squeeze(0)     # [N]
        max_length_tensor = td_instance['max_length'].squeeze(0) # [1]
        
        # Convert to numpy (required by k-NN and raw_data)
        coords_np = instance_locs.cpu().numpy()
        prizes_np = prizes.cpu().numpy()
        max_length_float = max_length_tensor.item()

        N = self.cfg.model_params.num_nodes
        sparse_k = self.co_expander_cfg.sparse_factor

        if N <= sparse_k:
            print(f"Warning: N ({N}) <= sparse_k ({sparse_k}). Skipping k-NN.")
            # Create an empty edge_index
            edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            edges_feature = torch.empty((0, 1), dtype=torch.float32, device=self.device)
            edges_num = 0
        else:
            # 2. Build the k-NN graph
            kdt = KDTree(coords_np, metric='euclidean')
            _, knn_indices = kdt.query(coords_np, k=sparse_k)
            source_nodes = torch.arange(N, device=self.device).view(-1, 1).repeat(1, sparse_k).flatten()
            target_nodes = torch.from_numpy(knn_indices).to(self.device).flatten()
            edge_index = torch.stack([source_nodes, target_nodes], dim=0)
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            edge_index = edge_index[:, edge_index[0] != edge_index[1]]
            edge_index_sorted, _ = torch.sort(edge_index, dim=0)
            edge_index = torch.unique(edge_index_sorted, dim=1)
            edges_num = edge_index.shape[1]

            # 3. Edge features (distances)
            src, dst = edge_index[0], edge_index[1]
            distances = torch.linalg.norm(instance_locs[src] - instance_locs[dst], dim=-1)
            normalized_distances = (distances - distances.min()) / (distances.max() - distances.min() + 1e-8)
            edges_feature = normalized_distances.unsqueeze(-1).float() # [E, 1]

        # 4. Node features (using the prefix provided by RL)
        is_depot_feature = torch.zeros((N, 1), dtype=torch.float32, device=self.device)
        is_depot_feature[0] = 1.0
        prize_feature = prizes.unsqueeze(-1).float() # [N, 1]

        prefix_nodes_tensor = torch.tensor(prefix_nodes_list, dtype=torch.long, device=self.device)
        node_prefix_state = torch.zeros((N, 1), dtype=torch.float32, device=self.device)
        if len(prefix_nodes_list) > 0:
            node_prefix_state[prefix_nodes_tensor] = 1.0
        
        # [arg 1] nodes_feature
        nodes_feature_final = torch.cat([is_depot_feature, prize_feature, node_prefix_state], dim=-1) # [N, 3]

        # 5. Other arguments (built manually for batch_size=1)
        
        # [arg 7] mask (edges)
        mask = torch.zeros(edges_num, dtype=torch.bool, device=self.device) 
        # [arg 8] ground_truth (edges)
        ground_truth = torch.zeros(edges_num, dtype=torch.long, device=self.device) # dummy GT
        # [arg 3] e (edge decision variables)
        e = torch.zeros(edges_num, dtype=torch.float32, device=self.device)
        # [arg 2] x (node decision variables)
        x = None # OP is an edge problem
        # [arg 5] graph (adjacency matrix)
        graph = None # not needed in sparse mode
        
        # [arg 0] task
        task = "OP"
        # [arg 6] graph_list (batch)
        graph_list = [graph] # [None]
        # [arg 9] nodes_num_list (batch)
        nodes_num_list = [N]
        # [arg 10] edges_num_list (batch)
        edges_num_list = [edges_num]
        # [arg 11] raw_data_list (batch)
        raw_data = (coords_np, prizes_np, max_length_float, []) # (coords, prizes, max_len, ref_tour)
        raw_data_list = [raw_data]
        # [arg 12] prefix_nodes (batch, list of tensors)
        prefix_nodes_batch_list = [prefix_nodes_tensor]
        # [arg 14] max_lengths (batch)
        max_lengths_batch_tensor = max_length_tensor.unsqueeze(0) # [1]
        
        # 6. Assemble the 15-argument tuple
        # The order must match exactly:
        # (task, nodes_feature, x, edges_feature, e, edge_index, graph_list, 
        #  mask, ground_truth, nodes_num_list, edges_num_list, raw_data_list,
        #  prefix_nodes, node_prefix_state, max_lengths)
        return (
            task,                       # 0
            nodes_feature_final,        # 1
            x,                          # 2
            edges_feature,              # 3
            e,                          # 4
            edge_index,                 # 5
            graph_list,                 # 6
            mask,                       # 7
            ground_truth,               # 8
            nodes_num_list,             # 9
            edges_num_list,             # 10
            raw_data_list,              # 11
            prefix_nodes_batch_list,    # 12
            node_prefix_state,          # 13
            max_lengths_batch_tensor    # 14
        )
    # ==============================================================================
    # === [END] core helper ===
    # ==============================================================================
def plot_kl_curve(self, save_path="kl_vs_step.png"):
        """
        Plot the KL divergence vs. inference step (mean ± std)
        """
        print("\nPlotting KL Divergence curve...")
        steps = sorted(self.kl_history.keys())
        
        if not steps:
            print("No KL data collected to plot (maybe KL trigger was disabled?).")
            return

        avg_kl = []
        std_kl = []
        counts = []
        
        valid_steps = []
        
        for step in steps:
            values = np.array(self.kl_history[step])
            # Filter out possible outliers or empty values
            if len(values) > 0:
                avg_kl.append(np.mean(values))
                std_kl.append(np.std(values))
                counts.append(len(values))
                valid_steps.append(step)
        
        if not valid_steps:
            print("No valid KL data found.")
            return

        avg_kl = np.array(avg_kl)
        std_kl = np.array(std_kl)
        valid_steps = np.array(valid_steps)

        plt.figure(figsize=(10, 6))
        
        # Plot the main curve (mean)
        plt.plot(valid_steps, avg_kl, label='Mean KL Divergence', color='blue', linewidth=2, marker='o', markersize=4)
        
        # Plot the shaded band (std)
        plt.fill_between(valid_steps, avg_kl - std_kl, avg_kl + std_kl, color='blue', alpha=0.2, label='Standard Deviation')
        
        plt.xlabel('Inference Step (Construction Step)', fontsize=12)
        plt.ylabel('KL Divergence (RL vs CO-Expander)', fontsize=12)
        plt.title(f'KL Divergence Trend over Steps (Aggregated from {max(counts)} instances)', fontsize=14)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        try:
            plt.savefig(save_path, dpi=300)
            print(f"✅ KL curve saved to: {save_path}")
        except Exception as e:
            print(f"Error saving plot: {e}")
        finally:
            plt.close()

def run(cfg: DictConfig):
    solver = HybridSolver(cfg)
    device = solver.device
    
    dataset = CustomOPDataset(
        file_path=cfg.data.test_path,
        num_nodes=cfg.model_params.num_nodes
    )
    # ==========================================
    # Limit the number of samples
    # ==========================================
    num_samples_to_test = 10  # <--- set the desired number here, e.g., 10, 50, 100
    
    if len(dataset) > num_samples_to_test:
        print(f"Limiting dataset from {len(dataset)} to {num_samples_to_test} samples.")
        dataset = Subset(dataset, range(num_samples_to_test))
    # ==========================================

    dataloader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        collate_fn=collate_fn_skip_none # (Fix from previous turn)
    )

    all_solutions = []
    start_time = time.time()
    
    for data_dict in tqdm(dataloader, desc="Solving OP Instances"):
        
        if data_dict is None: # (Fix from previous turn)
            print("Skipping bad batch (item failed to load).")
            continue

        all_locs = data_dict["locs"].to(device)     # [1, N, 2]
        all_prizes = data_dict["prizes"].to(device)   # [1, N]
        max_length = data_dict["max_length"].to(device) # [1]

        depot_locs = all_locs[:, 0, :]      # [1, 2]
        customer_locs = all_locs[:, 1:, :]  # [1, N-1, 2]
        customer_prizes = all_prizes[:, 1:] # [1, N-1]

        td_rl = TensorDict({
            "depot": depot_locs,
            "locs": customer_locs,
            "prize": customer_prizes,
            "max_length": max_length,
        }, batch_size=1)
        
        td_rl = td_rl.to(device) # (Fix from previous turn)

        # Full data used by CO-Expander
        full_instance_data_dict = {
            "locs": all_locs,
            "prizes": all_prizes,
            "max_length": max_length
        }

        solution = solver.solve_instance(td_rl, full_instance_data_dict)
        all_solutions.append(solution)
    solver.plot_kl_curve(save_path="kl_divergence_vs_step.png")
    total_time = time.time() - start_time
    if len(all_solutions) > 0:
        avg_reward = np.mean([s['reward'] for s in all_solutions])
    else:
        avg_reward = 0
        print("Warning: No solutions were generated.")
    
    print("\n" + "="*60)
    print("--- OP Hybrid (RL + CO-Expander) Solver Summary ---")
    print(f"Total time: {total_time:.2f}s for {len(all_solutions)} instances.")
    print(f"Average Final Reward: {avg_reward:.4f}")
    ce_count = sum(1 for s in all_solutions if s['source'] == 'CE')
    print(f"Solutions chosen from CO-Expander: {ce_count}/{len(all_solutions)}")
    print("="*60)

# ==============================================================================
    # === Statistics and plotting ===
    # ==============================================================================
    if not all_solutions:
        print("No solutions found, skipping statistics.")
        return

    # 1. Extract all trigger steps
    trigger_steps = [s['trigger_step'] for s in all_solutions]
    
    # 2. Keep only instances where the trigger actually fired
    # (only instances with step != -1)
    triggered_instance_steps = [step for step in trigger_steps if step != -1]
    
    num_total = len(trigger_steps)
    num_triggered = len(triggered_instance_steps)
    num_not_triggered = num_total - num_triggered

    print("\n--- Trigger Step Statistics ---")
    print(f"Total instances solved: {num_total}")
    print(f"Instances triggering CE: {num_triggered} ({num_triggered/num_total*100:.1f}%)")
    print(f"Instances NOT triggering CE: {num_not_triggered} ({num_not_triggered/num_total*100:.1f}%)")

    if num_triggered > 0:
        avg_trigger_step = np.mean(triggered_instance_steps)
        median_trigger_step = np.median(triggered_instance_steps)
        min_trigger_step = np.min(triggered_instance_steps)
        max_trigger_step = np.max(triggered_instance_steps)
        
        print(f"\nAvg. trigger step (for triggered instances): {avg_trigger_step:.2f}")
        print(f"Median trigger step: {median_trigger_step}")
        print(f"Trigger step range: [{min_trigger_step}, {max_trigger_step}]")

        # --- 3. Plotting ---
        plt.figure(figsize=(12, 7))
        
        # Create bins for the discrete step counts
        # e.g., if the max step is 20 we want bins 0, 1, 2, ..., 20
        bins = np.arange(int(min_trigger_step), int(max_trigger_step) + 2, 1) # +2 makes sure the last bin is included
        
        plt.hist(triggered_instance_steps, bins=bins, edgecolor='black', align='left')
        
        plt.title(f'Distribution of CO-Expander Trigger Steps (N={num_total})')
        plt.xlabel('Simulation Step Number (when entropy threshold met)')
        plt.ylabel('Number of Instances')
        
        # Set the x-axis ticks dynamically to avoid clutter
        tick_step = max(1, (max_trigger_step - min_trigger_step) // 20)
        plt.xticks(np.arange(int(min_trigger_step), int(max_trigger_step) + 1, tick_step))
        
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plot_filename = "trigger_step_distribution.png"
        plt.savefig(plot_filename)
        print(f"\n📈 Saved trigger step distribution plot to: {plot_filename}")
        # plt.show() # uncomment to display immediately while the script runs
    else:
        print("\nNo instances triggered CO-Expander, skipping plot.")
    # ==============================================================================
    # === [END] ===
    # ==============================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid RL-COExpander Solver for OP")
    parser.add_argument("--config", type=str, default="hyco_config.yaml")
    args = parser.parse_args()
    
    cfg = OmegaConf.load(args.config)
    default_solver_cfg = OmegaConf.create({
        'solver': {
            'entropy_threshold': 0.6, # original
            'use_kl_trigger': True,   # New
            'kl_threshold': 30,      # KL threshold
            'kl_temp': 2.0,           # temperature (higher = smoother distribution)
            'probe_noise_t': 0        # probe noise level
        }
    })
    cfg = OmegaConf.merge(default_solver_cfg, cfg)
    run(cfg)