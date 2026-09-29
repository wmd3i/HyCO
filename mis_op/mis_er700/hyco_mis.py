# HyCO for MIS: LwD (RL) constructs a partial solution, the prefix-conditioned CO-Expander completes it.
# Usage: python hyco_mis.py --config hyco_config.yaml
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

# --- Path setup ---
root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_folder not in sys.path:
    sys.path.append(root_folder)
base_dir = os.path.dirname(os.path.abspath(__file__))
co_expander_path = os.path.abspath(os.path.join(base_dir, '..'))
lwd_path = os.path.abspath(os.path.join(base_dir, '..', 'learning_what_to_defer'))
if co_expander_path not in sys.path:
    sys.path.append(co_expander_path)
if lwd_path not in sys.path:
    sys.path.append(lwd_path)

import dgl

# --- RL (LwD) Imports ---
from learning_what_to_defer.data.graph_dataset import get_rb_small_dataset, get_satlib_dataset
from learning_what_to_defer.ppo.actor_critic import ActorCritic
from learning_what_to_defer.env import MaximumIndependentSetEnv 
from learning_what_to_defer.ppo.graph_net import PolicyGraphConvNet, ValueGraphConvNet

# --- CO-Expander Imports ---
from co_expander import (
    COExpanderCMModel, GNNEncoder, COExpanderEnv,
    COExpanderDecoder, COExpanderSparser
)
from ml4co_kit import MISGraphData

# --- Plotting ---
import matplotlib.pyplot as plt
import seaborn as sns

def _get_lwd_dataloader(cfg: DictConfig, device: torch.device):
    # DataLoader construction
    test_data_path = cfg.data.test_path
    print(f"Loading LwD native dataset from: {test_data_path}")

    # ER / RB test sets are stored as .METIS files; SATLIB test graphs are stored as .gpickle files
    if cfg.data.get("dataset_format", "metis") == "gpickle":
        test_dataset = get_satlib_dataset("test", test_data_path)
    else:
        test_dataset = get_rb_small_dataset("test", test_data_path)
    
    def collate_fn(graphs):
        return dgl.batch(graphs)
        
    dataloader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0
    )
    print(f"Found {len(test_dataset)} instances.")
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
        print(f"Loading LwD RL model from: {self.cfg.rl_model.ckpt_path}")
        n_layers = self.cfg.rl_model.n_layers
        hidden_dim = self.cfg.rl_model.hidden_dim
        max_nodes_from_training = 1200
        
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
        """ (unchanged) """
        print(f"Loading CO-Expander model from: {self.cfg.co_expander_model.ckpt_path}")
        print(f"Using training config: {self.cfg.co_expander_model.config_path}")
        
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
            env=ce_env,
            encoder=encoder,
            decoder=decoder,
            learning_rate=ce_train_cfg.train.learning_rate,
            **ce_train_cfg.model_params
        )

        checkpoint = torch.load(self.cfg.co_expander_model.ckpt_path, map_location=self.device)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.eval()
        model.to(self.device)
        model.env = ce_env 
        print("CO-Expander model loaded successfully.")
        return model, ce_train_cfg

    # ==========================================================================
    # === [New 1] CO-Expander probe ===
    # ==========================================================================
    @torch.no_grad()
    def _run_co_expander_probe(self, original_g, current_rl_x):
        """
        Run one CO-Expander forward pass to get the model's "intuitive" probability that each node belongs to the MIS in the current state.
        
        Args:
            original_g: original DGL graph
            current_rl_x: current state tensor of the RL environment [N, 2].
                          current_rl_x[:, 0] == 1.0 means the node has been selected into the MIS (prefix).
        
        Returns:
            x_pred_logits: [N, 2] tensor with the [Out, In] logits of every node.
        """
        N = original_g.number_of_nodes()
        
        # 1. Extract the prefix (nodes already selected by RL)
        # In the RL state, dim 0 == 1.0 means selected
        if current_rl_x.dim() > 1:
            states = current_rl_x[:, 0]
        else:
            states = current_rl_x
        
        prefix_nodes_tensor = (states == 1.0).nonzero().squeeze(-1)
        
        # 2. Prepare the inputs (reuses _prepare_co_expander_input_tuple)
        # This builds node_prefix_state, which tells CO-Expander which nodes are fixed
        ce_inputs = self._prepare_co_expander_input_tuple(original_g, prefix_nodes_tensor)
        (task, nodes_feature, x, edges_feature, e, edge_index, graph_list, 
         mask, ground_truth, nodes_num_list, edges_num_list, raw_data_list,
         prefix_nodes_batch_list, node_prefix_state) = ce_inputs

        # 3. Build the input x (initial state of the node variables)
        # In DIFUSCO/CO-Expander inference, x is usually random noise.
        # For the probe, however, the prefix values must be fixed.
        # Assume the model was trained with inputs in [-0.5, 0.5] (corresponding to 0/1)
        
        # Initialize random noise (the model's guess for the unknown region)
        x_input = (torch.randn(N, device=self.device) > 0).float()
        
        # Build a mask that fixes only the prefix nodes
        current_mask = torch.zeros(N, dtype=torch.bool, device=self.device)
        if prefix_nodes_tensor.numel() > 0:
            current_mask[prefix_nodes_tensor] = True
            # Force the value of prefix nodes to 1.0 (in MIS)
            x_input[prefix_nodes_tensor] = 1.0
            
        # Normalize to [-0.5, 0.5] (the standard CO-Expander input format)
        x_input_masked = torch.where(current_mask, x_input - 0.5, x_input - 0.5)

        # 4. Set the probe timestep t
        # t=0 means direct prediction, t>0 means prediction from a noised input. t=50 works well for probing the energy.
        probe_t_val = self.cfg.solver.get('probe_noise_t', 0)
        t = torch.tensor([probe_t_val], device=self.device).float()

        # 5. Run the GNN encoder forward pass
        # Note: call the GNNEncoder directly and skip the diffusion loop
        x_pred_logits, _ = self.co_expander_model.model.forward(
            task=task, focus_on_node=True, focus_on_edge=False,
            nodes_feature=nodes_feature, 
            x=x_input_masked,
            edges_feature=edges_feature, e=e, mask=current_mask, 
            t=t, edge_index=edge_index,
            node_prefix_state=node_prefix_state,
            prefix_nodes=prefix_nodes_batch_list,
            nodes_num_list=nodes_num_list
        )
        
        return x_pred_logits # [N, 2]

    # ==========================================================================
    # === [New 2] KL divergence ===
    # ==========================================================================
    def _compute_kl_divergence(self, rl_probs, ce_logits, active_indices):
        """
        Binary KL divergence: KL(CE || RL)
        Measures how RL's probability of "select into MIS" differs from CO-Expander's.
        
        Args:
            rl_probs: [N_active, 3] RL action probabilities (0:?, 1:Select, 2:?)
            ce_logits: [N_total, 2] CO-Expander output logits
            active_indices: indices of the active nodes currently considered by RL [N_active]
        """
        # 1. Align the data
        ce_l_active = ce_logits[active_indices] # [N_active, 2]
        
        # 2. CO-Expander "select" probability (index 1 after softmax)
        ce_probs = torch.softmax(ce_l_active, dim=-1)
        p_ce_in = ce_probs[:, 1] # P(Node is IN)
        
        # 3. RL "select" probability
        # Assume RL action 1 means "select" (the usual LwD convention)
        # i.e., action 0=defer, 1=select, 2=deselect
        p_rl_in = rl_probs[:, 1]
        
        # 4. Numerical stability (avoid log(0))
        eps = 1e-6
        p_ce_in = torch.clamp(p_ce_in, eps, 1-eps)
        p_rl_in = torch.clamp(p_rl_in, eps, 1-eps)
        
        # 5. Bernoulli KL divergence
        # KL(P || Q) = p * log(p/q) + (1-p) * log((1-p)/(1-q))
        # Measures how far RL (posterior) deviates if CO-Expander is taken as the prior.
        # KL(CE || RL) measures how well RL matches CO-Expander.
        # Interpretation: if CO-Expander is confident a node is in (p_ce->1) but RL says out (p_rl->0), the KL is large.
        kl_values = (p_ce_in * torch.log(p_ce_in / p_rl_in) + 
                     (1 - p_ce_in) * torch.log((1 - p_ce_in) / (1 - p_rl_in)))
        
        # Return the mean or total KL
        return kl_values.sum()

    # ==========================================================================
    # === Solve instance (main logic) ===
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
            # === Trigger Logic (Entropy + KL) ===
            # ==================================================================
            trigger_now = False
            entropy_val = torch.tensor(0.0)
            kl_val = torch.tensor(0.0)
            
            # Only computed if not triggered yet and the theory trigger is enabled
            if self.cfg.solver.use_theory_trigger and not ce_triggered_flag:
                
                # A. Entropy Calculation
                entropies = rl_dist.entropy()
                M = min(self.cfg.solver.probe_rl_top_m, len(entropies))
                
                if M > 0:
                    # Mean entropy of the top-M nodes
                    top_m_entropies, top_m_indices_local = torch.topk(entropies, k=M)
                    avg_entropy = torch.mean(top_m_entropies)
                    entropy_val = avg_entropy
                    print(f"entropy_val is {entropy_val:.4f} at step {rl_t}")
                    is_high_entropy = avg_entropy > self.cfg.solver.entropy_threshold
                    
                    # B. KL divergence (if use_kl_trigger is enabled)
                    is_high_kl = False
                    if self.cfg.solver.get('use_kl_trigger', True):
                        # 1. Run the probe to get the CO-Expander distribution
                        # Note: the current rl_env.x is passed as the state
                        ce_logits = self._run_co_expander_probe(original_g, self.rl_env.x)
                        
                        # 2. RL probability distribution (softmax)
                        rl_probs_all = torch.softmax(logits, dim=-1)
                        
                        # 3. KL on the top-M (high-uncertainty) nodes only, or on the whole graph?
                        # For efficiency and focus, compute the KL on the active nodes of the current subgraph
                        # flatten_subg_node_idxs are the global indices of the nodes currently considered by RL
                        active_global_indices = flatten_node_idxs
                        
                        total_kl = self._compute_kl_divergence(
                            rl_probs_all, # [N_active, 3]
                            ce_logits,    # [N_total, 2]
                            active_global_indices
                        )
                        
                        # Normalize the KL (divide by the number of nodes so it is comparable to the threshold)
                        avg_kl = total_kl / (len(active_global_indices) + 1e-9)
                        kl_val = avg_kl
                        print(f"KL is {avg_kl:.4f} at step {rl_t}")
                        
                        kl_threshold = self.cfg.solver.get('kl_threshold', 0.5) # Default threshold
                        is_high_kl = avg_kl > kl_threshold

                    # C. Combined decision
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
                
                # ... [K-lookahead logic] ...
                backup_rl_env_x = self.rl_env.x.clone()
                backup_rl_env_t = self.rl_env.t
                K = self.cfg.solver.get('lookahead_k', 2) # Read K from the config
                
                best_ce_lookahead_mis_size = -1
                best_ce_lookahead_mask = np.array([])
                best_action_to_take = None 

                probs = torch.softmax(logits, dim=-1)
                base_greedy_actions_flat = torch.argmax(logits, dim=-1)
                flat_probs = probs.view(-1)
                top_k_probs, top_k_indices = torch.topk(flat_probs, k=K)

                # print(f"    Lookahead: Testing K={K} candidates...")
                candidate_actions_to_test = []
                for i in range(K):
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

                    self.rl_env.step(rl_action_candidate) # Simulate step
                    
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
                
                print(f"    Lookahead chosen best CE size: {best_ce_lookahead_mis_size:.1f}")
                rl_action = best_action_to_take
                best_ce_mis_size = best_ce_lookahead_mis_size
                ce_solution_mask = best_ce_lookahead_mask
                # ... [End of K-Lookahead] ...

            # RL Step
            rl_state, reward, done_bool, info = self.rl_env.step(rl_action)
            current_sol_tensor = info['sol']
            
            if current_sol_tensor.max().item() > best_rl_sol_tensor_so_far.max().item():
                best_rl_sol_tensor_so_far = current_sol_tensor.clone()
                best_rl_state_so_far = current_rl_state_clone
                
            done = torch.all(done_bool).item() 
            rl_t += 1

        # --- Final result processing ---
        best_rl_mis_size = best_rl_sol_tensor_so_far.max().item() 
        best_sample_idx = best_rl_sol_tensor_so_far.argmax().item()

        if best_rl_mis_size > best_ce_mis_size:
            state_slice = best_rl_state_so_far[:, best_sample_idx, 0] if best_rl_state_so_far.ndim == 3 else best_rl_state_so_far[:, 0]
            rl_solution_mask_numpy = (state_slice == 1.0).cpu().numpy().astype(int)
            # print(f"    Result: RL ({best_rl_mis_size}) > CE ({best_ce_mis_size})")
            return {"mis_size": best_rl_mis_size, "solution_mask": rl_solution_mask_numpy, "source": "RL", "trigger": trigger_reason}
        elif ce_triggered_flag:
            # print(f"    Result: CE ({best_ce_mis_size}) >= RL ({best_rl_mis_size})")
            return {"mis_size": best_ce_mis_size, "solution_mask": ce_solution_mask, "source": "CE", "trigger": trigger_reason}
        else:
            state_slice = best_rl_state_so_far[:, best_sample_idx] if best_rl_state_so_far.ndim==2 else best_rl_state_so_far[:, best_sample_idx, 0]
            rl_solution_mask_numpy = (state_slice == 1.0).cpu().numpy().astype(int)
            return {"mis_size": best_rl_mis_size, "solution_mask": rl_solution_mask_numpy, "source": "RL", "trigger": trigger_reason}

    def _prepare_co_expander_input_tuple(self, g, prefix_nodes_tensor):
        # Existing helper
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

# ... [run function unchanged] ...
# ==============================================================================
# === Runner ===
# ==============================================================================
def run(cfg: DictConfig):
    solver = HybridSolverMIS(cfg)
    device = solver.device

    # 1. Get the native LwD dataloader
    val_dataloader = _get_lwd_dataloader(cfg, device)
    num_batches = len(val_dataloader)

    all_solutions = []
    start_time = time.time()

    # 2. Iterate over the LwD batches
    pbar = tqdm(val_dataloader, desc="Solving MIS Instances (LwD Loader)")
    for i, g in enumerate(pbar):
        
        # --- Check ---
        if i >= 500:
            print(f"\nReached 'max_test_instances' limit 500. Stopping.")
            break # Exit the loop
        
        # 3. g is now a DGL graph loaded natively by LwD
        #    We must mimic the evaluation setup of the LwD training script
        g.set_n_initializer(dgl.init.zero_initializer)
        g = g.to(device)

        if g is None or g.number_of_nodes() == 0: 
            print("Skipping empty graph.")
            continue
            
        #print(f"\n[Debug] Solving graph with {g.number_of_nodes()} nodes (Loaded by LwD).")

        # 4. Solve using this "gold standard" g
        
        solution_stats = solver.solve_instance(g)
        all_solutions.append(solution_stats)

    total_time = time.time() - start_time
    
    if len(all_solutions) > 0:
        avg_mis_size = np.mean([s['mis_size'] for s in all_solutions])
    else:
        avg_mis_size = 0
        print("Warning: No solutions were generated.")
    
    print("\n" + "="*60)
    print("--- MIS Hybrid (LwD-RL + CO-Expander) Solver Summary ---")
    print(f"[Data Source: LwD Native DataLoader]")
    print(f"Total time: {total_time:.2f}s for {len(all_solutions)} instances.")
    print(f"Average Final MIS Size: {avg_mis_size:.4f}")
    ce_count = sum(1 for s in all_solutions if s['source'] == 'CE')
    print(f"Solutions chosen from CO-Expander: {ce_count}/{len(all_solutions)}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid LwD-RL-COExpander Solver for MIS")
    parser.add_argument("--config", type=str, default="hyco_config.yaml")
    args = parser.parse_args()
    
    # Merge default config
    default_solver_cfg = OmegaConf.create({
        'solver': {
            'use_theory_trigger': True,
            'probe_rl_top_m': 15,
            'entropy_threshold': 0.6,
            'lookahead_k': 2,
            'use_kl_trigger': True,   # Enable the KL trigger
            'kl_threshold': 0.2,      # KL threshold (adjust as needed)
            'probe_noise_t': 50       # Probe noise level
        }
    })
    
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(default_solver_cfg, cfg)
    
    run(cfg)