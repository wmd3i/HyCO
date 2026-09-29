# -*- coding: utf-8 -*-
"""
Generates OP data for training the prefix-conditioned CO-Expander.

Instead of Gurobi, it:
1. Randomly generates OP instances.
2. Solves every instance with a pretrained RL (AttentionModel) policy.
3. Checks feasibility (whether the max_length constraint is satisfied).
4. Writes the feasible instances and RL solutions in a single-line .txt format.
"""

# 1. Imports
import torch
import numpy as np
import time
import argparse
import os
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig
from tensordict import TensorDict

# --- RL4CO imports ---
from rl4co.models.zoo import AttentionModel
from rl4co.envs import OPEnv
# (the OP data loader is not needed here)
from data_loader_sparse import OPConditionalSuffixDataset
from omegaconf import DictConfig 
torch.serialization.add_safe_globals([DictConfig])


class RLSolver:
    def __init__(self, cfg: DictConfig, device):
        self.cfg = cfg
        self.device = device
        print(f"INFO: RL Solver using device: {self.device}")
        self.rl_model, self.rl_env = self._load_rl_policy()
        self.rl_model = self.rl_model.to(self.device)
        self.rl_model.eval()

    def _load_rl_policy(self):
        print(f"INFO: Loading RL model for OP from: {self.cfg.rl_model.ckpt_path}")
        try:
            num_customers = self.cfg.model.num_nodes - 1
            env = OPEnv(generator_kwargs={'num_loc': num_customers})
            
            model = AttentionModel.load_from_checkpoint(
                self.cfg.rl_model.ckpt_path,
                env=env, map_location='cuda' if torch.cuda.is_available() else 'cpu',
                strict=False
            )
            return model, model.env
        except Exception as e:
            print(f"ERROR: Failed to load RL model: {e}")
            print("Make sure rl_model.ckpt_path in the config is correct!")
            exit()

    @torch.no_grad()
    def solve_batch(self, td_batch):
        """
        Solve a batch of OP instances with the RL model.
        
        Args:
            td_batch (TensorDict): TensorDict holding a batch of *problem instances*
                                  (batch_size=N)
        
        Returns:
            paths (list[list]): list of N solution paths
        """
        
        # === Key fix ===
        # env.reset() must be called first to turn the "problem" TensorDict into a "state" TensorDict
        # The "state" TensorDict contains keys such as "done" and "action_mask", required by the model's internal loop.
        # .clone() is good practice; .to(self.device) makes sure the state is on the GPU.
        td_reset_state = self.rl_env.reset(td_batch.clone()).to(self.device)
        
        # Maximum number of steps
        max_steps = self.rl_env.generator.num_loc + 1
        
        # Pass the "state" TensorDict to the model
        out = self.rl_model(
            td_reset_state,        # <--- use td_reset_state instead of td_batch
            phase="test",
            decode_type="greedy",
            max_length=max_steps
        )
        
        # out['actions'] is a tensor of shape [batch_size, sequence_length]
        actions_tensor = out['actions']
        
        # Convert the actions tensor back into a list of paths
        paths = []
        batch_size = actions_tensor.shape[0]
        for i in range(batch_size):
            # .cpu().tolist() converts it to a Python list
            action_list = actions_tensor[i].cpu().tolist()
            
            # Path format: [0, customer_1, ..., customer_k, 0]
            # Extract all customer nodes (> 0)
            customer_stops = [a for a in action_list if a > 0]
            
            # Rebuild in the [0, ..., 0] format
            path = [0] + customer_stops + [0]
                
            paths.append(path)
            
        return paths


# ==============================================================================
# 3. Helper functions
# ==============================================================================
def calculate_path_metrics(path, prizes, time_matrix):
    if not path or len(path) < 2:
        return 0.0, 0.0
    
    # Use a set to compute the prize, just in case (the path should contain no duplicate customers)
    total_prize = sum(prizes[i] for i in set(path))
    total_length = sum(time_matrix[path[i], path[i+1]] for i in range(len(path)-1))
    return total_prize, total_length

# ==============================================================================
# 4. Instance generation and formatting
# ==============================================================================

def generate_op_instance(num_customers):
    """
    Generate a random OP instance (NumPy format).
    Coordinates and prizes are normalized to [0, 1].
    """
    num_nodes = num_customers + 1
    
    # Depot at index 0
    depot = np.random.rand(2)
    customers = np.random.rand(num_customers, 2)
    all_locs = np.vstack([depot, customers]).astype(np.float32)
    
    # Depot prize is 0
    customer_prizes = np.random.uniform(0.01, 1.0, size=num_customers).astype(np.float32)
    full_prizes = np.insert(customer_prizes, 0, 0.0) 
    
    # Randomize the max length around 4.0
    max_length = np.random.uniform(3.5, 5.2)
    
    return all_locs, full_prizes, max_length

def format_instance_to_line(instance_id, locs, prizes, max_length, solution_path):
    """
    Format a solved instance as a single-line string.
    """
    # Instance: ...
    instance_str = f"Instance:{instance_id}"
    
    # depots: x y
    depot_str = "depots:" + " ".join(map(str, locs[0]))
    
    # points: x1 y1 x2 y2 ...
    points_str = "points:" + " ".join(map(str, locs[1:].flatten()))
    
    # prizes: p1 p2 ... (customer prizes only)
    prizes_str = "prizes:" + " ".join(f"{p:.2f}" for p in prizes[1:])
    
    # max_length: L
    max_len_str = f"max_length:{max_length:.4f}"
    
    # output: 0 n1 n2 ... 0
    output_str = "output:" + " ".join(map(str, solution_path))
    
    return f"{instance_str};{depot_str};{points_str};{prizes_str};{max_len_str};{output_str}"





def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    num_customers = cfg.model.num_nodes - 1
    print(f"INFO: generating instances with N={num_customers} ({cfg.model.num_nodes} nodes).")

    # Initialize the RL solver
    rl_solver = RLSolver(cfg, device)
    
    np.random.seed(123)
    
    # Get the inference batch size from the config
    inference_batch_size = cfg.data_gen.inference_batch_size
    print(f"INFO: inference batch size: {inference_batch_size}")

    datasets_to_generate = {
        "train.txt": cfg.data_gen.num_train,
        "valid.txt": cfg.data_gen.num_valid,
        "test.txt": cfg.data_gen.num_test
    }
    
    os.makedirs(cfg.data_gen.output_dir, exist_ok=True)
    
    for filename, num_samples in datasets_to_generate.items():
        if num_samples == 0:
            continue
            
        output_path = os.path.join(cfg.data_gen.output_dir, filename)
        print(f"\nINFO: generating {num_samples} samples for {output_path}...")
        
        sample_count = 0
        with open(output_path, "w") as f, tqdm(total=num_samples) as pbar:
            while sample_count < num_samples:
                
                # Number of instances to generate in this batch
                # (the last batch may be partial)
                num_to_gen = min(inference_batch_size, num_samples - sample_count)
                if num_to_gen <= 0:
                    break # just in case

                # 1. Generate random instances in batch (NumPy)
                locs_list, prizes_list, max_len_list = [], [], []
                for _ in range(num_to_gen):
                    loc, prize, max_len = generate_op_instance(num_customers)
                    locs_list.append(loc)
                    prizes_list.append(prize)
                    max_len_list.append(max_len)
                
                # 2. Stack the NumPy list into batched tensors
                locs_np_batch = np.stack(locs_list)
                prizes_np_batch = np.stack(prizes_list)
                max_len_np_batch = np.array(max_len_list)

                all_locs_tensor = torch.tensor(locs_np_batch).to(device)
                all_prizes_tensor = torch.tensor(prizes_np_batch).to(device)
                max_len_tensor = torch.tensor(max_len_np_batch, dtype=torch.float32).to(device)

                # 3. Prepare the RL input (TensorDict)
                td_rl_input = TensorDict({
                    "depot": all_locs_tensor[:, 0, :],
                    "locs": all_locs_tensor[:, 1:, :],
                    "prize": all_prizes_tensor[:, 1:],
                    "max_length": max_len_tensor,
                }, batch_size=num_to_gen).to(device)
                
                # 4. Solve the batch with RL
                # rl_paths is a list of num_to_gen paths
                rl_paths = rl_solver.solve_batch(td_rl_input)
                
                # 5. Check feasibility sequentially and write
                # (this part is fast; the bottleneck is GPU inference)
                valid_count_in_batch = 0
                for i in range(num_to_gen):
                    rl_path = rl_paths[i]
                    locs_np = locs_np_batch[i]
                    prizes_np = prizes_np_batch[i]
                    max_len_np = max_len_np_batch[i]

                    time_matrix_np = np.linalg.norm(locs_np[:, np.newaxis, :] - locs_np[np.newaxis, :, :], axis=2)
                    rl_prize, rl_length = calculate_path_metrics(rl_path, prizes_np, time_matrix_np)
                    
                    if rl_length <= max_len_np + 1e-6 and rl_prize > 0:
                        instance_id = np.random.randint(100000, 999999)
                        line = format_instance_to_line(
                            instance_id,
                            locs_np,
                            prizes_np,
                            max_len_np,
                            rl_path
                        )
                        f.write(line + "\n")
                        valid_count_in_batch += 1
                
                # 6. Update counters and the progress bar
                sample_count += valid_count_in_batch
                pbar.update(valid_count_in_batch)
                # Show the batch efficiency (how many solutions are feasible) on the progress bar
                pbar.set_postfix({'batch_eff': f'{valid_count_in_batch}/{num_to_gen}'})

    print(f"\nINFO: dataset generation finished! Saved to {cfg.data_gen.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL-based Data Generator for OP Diffusion")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/op200.yaml",
        help="Path to the OmegaConf YAML configuration file (e.g., configs/op100.yaml)"
    )
    # Arguments controlling the number of generated samples
    parser.add_argument("--num_train", type=int, default=0, help="Number of training samples")
    parser.add_argument("--num_valid", type=int, default=0, help="Number of validation samples")
    parser.add_argument("--num_test", type=int, default=500, help="Number of test samples")
    parser.add_argument("--output_dir", type=str, default="../data/op/op200", help="Directory to save dataset files")
    
    # === New arguments ===
    parser.add_argument(
        "--inference_batch_size", 
        type=int, 
        default=1024,  # starting value; increase according to available GPU memory
        help="Batch size for parallel RL inference."
    )
    
    args = parser.parse_args()
    
    try:
        cfg = OmegaConf.load(args.config)
        
        # Merge command-line arguments into the config
        gen_cfg = OmegaConf.create({
            'data_gen': {
                'num_train': args.num_train,
                'num_valid': args.num_valid,
                'num_test': args.num_test,
                'output_dir': args.output_dir,
                'inference_batch_size': args.inference_batch_size # <--- added
            }
        })
        cfg = OmegaConf.merge(cfg, gen_cfg)
        
        if 'model' not in cfg or 'num_nodes' not in cfg.model:
            print(f"ERROR: Config file '{args.config}' must contain 'model.num_nodes' (e.g., 101 for OP100)")
            exit()
            
        print("--- Data Generation Configuration ---")
        print(OmegaConf.to_yaml(cfg))
        
        main(cfg)
        
    except FileNotFoundError:
        print(f"ERROR: Configuration file not found at '{args.config}'")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise e