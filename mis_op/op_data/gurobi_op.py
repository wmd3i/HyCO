# -*- coding: utf-8 -*-
"""
RL-Gurobi hybrid solver for the Orienteering Problem (OP), used as the reference solver.
A pretrained RL model first produces a high-quality initial solution.
This solution is passed to Gurobi as a MIP start (warm start),
and its objective is added as a lower-bound constraint, forcing Gurobi to search for better solutions
within a limited time budget.
"""

# 1. Imports
import torch
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time
import argparse
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader
from tensordict import TensorDict

# --- RL4CO imports ---
from rl4co.models.zoo import AttentionModel
from rl4co.envs import OPEnv
# OP data loader (data_loader_sparse.py)
from data_loader_sparse import OPConditionalSuffixDataset
# The following two lines are necessary for loading checkpoints saved with OmegaConf
from omegaconf import DictConfig 
torch.serialization.add_safe_globals([DictConfig])


# ==============================================================================
# 2. RL solver
# ==============================================================================
class RLSolver:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"INFO: RL Solver using device: {self.device}")
        self.rl_policy, self.rl_env = self._load_rl_policy()

    def _load_rl_policy(self):
        print(f"INFO: Loading RL model for OP from: {self.cfg.rl_model.ckpt_path}")
        try:
            env = OPEnv(generator_kwargs={'num_loc': self.cfg.model.num_nodes - 1})
            
            model = AttentionModel.load_from_checkpoint(
                self.cfg.rl_model.ckpt_path,
                env=env, map_location='cuda' if torch.cuda.is_available() else 'cpu',
                strict=False
            )
            policy = model.policy.to(self.device)
            policy.eval()
            return policy, model.env
        except Exception as e:
            print(f"ERROR: Failed to load RL model: {e}")
            exit()

    @torch.no_grad()
    def solve(self, td_instance):
        """
        Uses the RL model to solve a single OP instance.
        The rl4co OPEnv's action mask should inherently handle the max_length constraint during decoding.
        """
        td_step = self.rl_env.reset(td_instance.clone())
        
        # Encode the instance once
        node_embeds, _ = self.rl_policy.encoder(td_step)
        cached_embeds = self.rl_policy.decoder._precompute_cache(node_embeds)

        # Autoregressive decoding loop
        actions = []
        while not td_step["done"].all():
            logits, _ = self.rl_policy.decoder(td_step, cached_embeds)
            
            # Apply the action mask to prevent illegal moves
            masked_logits = logits + td_step["action_mask"].log()
            
            # Greedily select the best valid action
            action = masked_logits.argmax(-1)
            
            actions.append(action.item())
            td_step.set("action", action)
            td_step = self.rl_env.step(td_step)["next"]
            
        # Format the final path
        # The environment should handle the tour properly, we just collect non-depot actions
        path = [0] + [a for a in actions if a != 0] + [0]
        return path

# ==============================================================================
# 3. Gurobi solver (supports MIP start and a lower-bound constraint)
# ==============================================================================
def solve_op_milp_with_rl_start(data, rl_path=None, rl_objective_lower_bound=None, time_limit_sec=120):
    """
    Solves the OP MILP model, using the RL solution for both MIPStart 
    (for x and y variables only) and as a lower bound objective constraint.
    """
    num_nodes = data["num_nodes"]
    prizes = data["full_prizes"]
    time_matrix = data["time_matrix"]
    T_max = data["T_max"]
    depot = 0
    
    nodes = range(num_nodes)
    customer_nodes = [i for i in nodes if i != depot]
    
    m = gp.Model("RL_WarmStarted_OP")
    m.setParam('OutputFlag', 1) 
    m.setParam('TimeLimit', time_limit_sec)
    m.setParam('Threads', 1)
    m.setParam(GRB.Param.MIPFocus, 1) 

    # --- Define Variables ---
    x = m.addVars(nodes, nodes, vtype=GRB.BINARY, name="x")
    y = m.addVars(nodes, vtype=GRB.BINARY, name="y")
    u = m.addVars(nodes, vtype=GRB.CONTINUOUS, name="u")

    # --- Set MIP Start (if a valid path is provided) ---
    if rl_path and len(rl_path) > 2:
        print("INFO: Applying RL solution for x and y variables as MIP Start.")
        m.update() 
        
        # 1. Reset all Start attributes
        for i in nodes:
            y[i].Start = 0.0
            # We will NOT set u[i].Start. Let Gurobi handle it.
            for j in nodes:
                x[i, j].Start = 0.0
        
        # 2. Set Start attributes for y (visited nodes)
        visited_nodes = set(rl_path)
        for node_idx in visited_nodes:
            y[node_idx].Start = 1.0

        # 3. Set Start attributes for x (arcs) ONLY
        for k in range(len(rl_path) - 1):
            start_node = rl_path[k]
            end_node = rl_path[k+1]
            x[start_node, end_node].Start = 1.0
            
        # --- THE FIX: REMOVED THE CALCULATION AND SETTING OF u[i].Start ---
        # The previous code that calculated 'cumulative_length' and set u[...].Start is removed.

    # --- Objective Function ---
    objective_expr = gp.quicksum(prizes[i] * y[i] for i in customer_nodes)
    m.setObjective(objective_expr, GRB.MAXIMIZE)

    # --- Add Lower Bound Constraint ---
    if rl_objective_lower_bound is not None and rl_objective_lower_bound > 0:
        print(f"INFO: Applying RL objective {rl_objective_lower_bound:.4f} as a lower bound constraint.")
        m.addConstr(objective_expr >= rl_objective_lower_bound, name="objective_lower_bound")

    # --- Constraints (Unchanged) ---
    m.addConstrs((x[i, i] == 0 for i in nodes), name="no_self_loops")
    # ... (all other constraints remain the same) ...
    m.addConstr(gp.quicksum(x[depot, j] for j in customer_nodes) >= 1, name="leave_depot")
    m.addConstr(gp.quicksum(x[i, depot] for i in customer_nodes) >= 1, name="return_depot")
    m.addConstrs((gp.quicksum(x[j, i] for j in nodes) == gp.quicksum(x[i, k] for k in nodes) for i in customer_nodes), name="flow_conservation")
    m.addConstrs((y[i] == gp.quicksum(x[j, i] for j in nodes if j != i) for i in customer_nodes), name="link_y_x")
    m.addConstr(y[depot] == 1, name="visit_depot")
    m.addConstr(gp.quicksum(time_matrix[i, j] * x[i, j] for i in nodes for j in nodes) <= T_max, name="time_limit")
    
    m.addConstr(u[depot] == 0, name="u_depot_is_zero")
    for i in customer_nodes:
        m.addConstr(u[i] >= time_matrix[depot, i] * y[i], name=f"u_lower_{i}")
        m.addConstr(u[i] <= (T_max - time_matrix[i, depot]) * y[i], name=f"u_upper_{i}")
        for j in customer_nodes:
            if i != j:
                m.addConstr(u[i] + time_matrix[i, j] - T_max * (1 - x[i, j]) <= u[j], name=f"MTZ_{i}_{j}")
    
    # --- Optimize and Extract Results ---
    m.optimize()
    
    result = {'status': m.Status, 'solve_time': m.Runtime, 'objective': 0.0, 'path': []}
    if m.SolCount > 0:
        result['objective'] = m.ObjVal
        active_arcs = [(i, j) for i, j in x.keys() if x[i, j].X > 0.5]
        
        if not active_arcs: return result
        
        path = [depot]
        current_node = depot
        next_node_map = {i: j for i, j in active_arcs}
        
        while True:
            current_node = next_node_map.get(current_node)
            if current_node is None or current_node == depot:
                break
            path.append(current_node)
        path.append(depot)
        result['path'] = path
            
    return result

# ==============================================================================
# 4. Helper functions
# ==============================================================================
def calculate_path_metrics(path, prizes, time_matrix):
    if not path or len(path) < 2:
        return 0.0, 0.0
    
    # Use set for prizes to avoid double counting if a node appears multiple times (should not happen in a simple path)
    total_prize = sum(prizes[i] for i in set(path))
    total_length = sum(time_matrix[path[i], path[i+1]] for i in range(len(path)-1))
    return total_prize, total_length

def print_solution_summary(instance_id, rl_path, rl_prize, rl_length, gurobi_result, max_length_np, prizes_np, time_matrix_np):
    print("") 
    print(f"----------- Solution Summary for Instance {instance_id} -----------")
    
    # RL Solution Details
    feasibility_str = "FEASIBLE" if rl_length <= max_length_np else "INFEASIBLE"
    print(f"[RL Initial Solution] ({feasibility_str})")
    print(f"  - Path: {rl_path}")
    print(f"  - Objective: {rl_prize:.4f}")
    print(f"  - Length: {rl_length:.4f} / {max_length_np:.4f}")

    # Gurobi Solution Details
    gurobi_path = gurobi_result.get('path', [])
    gurobi_prize = gurobi_result.get('objective', 0.0)
    _, gurobi_length = calculate_path_metrics(gurobi_path, prizes_np, time_matrix_np)
    
    print(f"[Gurobi Final Solution]")
    if not gurobi_path:
        print("  - No solution found by Gurobi within the time limit.")
    else:
        print(f"  - Path: {gurobi_path}")
        print(f"  - Objective: {gurobi_prize:.4f}")
        print(f"  - Length: {gurobi_length:.4f} / {max_length_np:.4f}")
    
    # Improvement
    if rl_prize > 0 and gurobi_prize > rl_prize:
        improvement = (gurobi_prize - rl_prize) / rl_prize * 100
        print(f"-> Improvement: +{improvement:.2f}%")
    elif gurobi_prize > 0 and rl_prize <= 0:
        print(f"-> Gurobi found a valid solution where RL did not.")
    
    print(f"----------------------------------------------------")

# ==============================================================================
# 5. Main function
# ==============================================================================
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rl_solver = RLSolver(cfg)
    
    dataset = OPConditionalSuffixDataset(
        txt_file_paths=[cfg.data.test_path],
        prefix_k_options=[0],
        sparse_factor=cfg.model.sparse_factor
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    print(f"\nINFO: Found {len(dataset)} instances in the test file.")
    print(f"INFO: Gurobi time limit per instance: {cfg.solver.gurobi_time_limit} seconds.")
    print("-" * 60)

    results = []
    total_start_time = time.time()
    
    for i, data_dict in enumerate(tqdm(dataloader, desc="Solving Instances (RL + Gurobi)")):
        
        # --- a. Prepare RL input (TensorDict) ---
        all_locs = data_dict["instance_locs"].to(device)
        prizes = data_dict["prizes"].to(device)
        max_length = data_dict["max_length"].to(device)
        
        td_rl_input = TensorDict({
            "depot": all_locs[:, 0, :],
            "locs": all_locs[:, 1:, :],
            "prize": prizes[:, 1:],
            "max_length": max_length,
        }, batch_size=1).to(device)

        # --- b. Use RL to generate initial solution ---
        rl_time_start = time.time()
        rl_path = rl_solver.solve(td_rl_input)
        rl_time = time.time() - rl_time_start
        
        # --- c. Prepare Gurobi input (Numpy dictionary) ---
        locs_np = all_locs.squeeze(0).cpu().numpy()
        prizes_np = prizes.squeeze(0).cpu().numpy()
        max_length_np = max_length.item()
        num_nodes = locs_np.shape[0]
        time_matrix_np = np.linalg.norm(locs_np[:, np.newaxis, :] - locs_np[np.newaxis, :, :], axis=2)
        
        gurobi_data = {
            "num_nodes": num_nodes,
            "full_prizes": prizes_np,
            "time_matrix": time_matrix_np,
            "T_max": max_length_np
        }
        
        # --- d. Validate RL solution ---
        rl_prize, rl_length = calculate_path_metrics(rl_path, prizes_np, time_matrix_np)
        
        rl_path_for_gurobi = None
        rl_objective_for_gurobi = None
        
        if rl_length <= max_length_np:
            # If feasible, use both its path for MIPStart and its prize for the lower bound constraint
            rl_path_for_gurobi = rl_path
            rl_objective_for_gurobi = rl_prize
        else:
            # If infeasible, Gurobi will start from scratch
            rl_prize = -1 # Mark as invalid for statistics
            
        # --- e. Use Gurobi to optimize from RL solution ---
        gurobi_result = solve_op_milp_with_rl_start(
            data=gurobi_data,
            rl_path=rl_path_for_gurobi,
            rl_objective_lower_bound=rl_objective_for_gurobi, # <-- PASS THE LOWER BOUND
            time_limit_sec=cfg.solver.gurobi_time_limit
        )

        # Print summary for first 5 and last instance
        if i < 5 or i == len(dataset) - 1:
            print_solution_summary(
                instance_id=i,
                rl_path=rl_path,
                rl_prize=rl_prize if rl_path_for_gurobi else -1, # Show original prize even if infeasible
                rl_length=rl_length,
                gurobi_result=gurobi_result,
                max_length_np=max_length_np,
                prizes_np=prizes_np,
                time_matrix_np=time_matrix_np
            )

        results.append({
            'instance_id': i,
            'rl_objective': rl_prize,
            'rl_time': rl_time,
            'gurobi_objective': gurobi_result['objective'],
            'gurobi_time': gurobi_result['solve_time']
        })

    total_time = time.time() - total_start_time

    # --- 3. Result statistics and printing ---
    feasible_rl_results = [r for r in results if r['rl_objective'] > 0]
    avg_rl_obj = np.mean([r['rl_objective'] for r in feasible_rl_results]) if feasible_rl_results else 0
    avg_gurobi_obj = np.mean([r['gurobi_objective'] for r in results])
    avg_rl_time = np.mean([r['rl_time'] for r in results])
    avg_gurobi_time = np.mean([r['gurobi_time'] for r in results])
    improvement = (avg_gurobi_obj - avg_rl_obj) / avg_rl_obj * 100 if avg_rl_obj > 0 else float('inf')

    print("\n" + "="*60)
    print("--- RL-Gurobi Hybrid Solver Evaluation Summary ---")
    print("="*60)
    print(f"Total instances processed: {len(results)}")
    print(f"Number of feasible RL solutions: {len(feasible_rl_results)}/{len(results)}")
    print(f"Total processing time: {total_time:.2f}s")
    print("-" * 60)
    print(f"Average RL Solve Time: {avg_rl_time*1000:.2f} ms")
    print(f"Average Gurobi Solve Time: {avg_gurobi_time:.2f} s")
    print("-" * 60)
    print(f"Average RL Objective (for feasible solutions): {avg_rl_obj:.4f}")
    print(f"Average Gurobi Objective (with RL Start): {avg_gurobi_obj:.4f}")
    print(f"Average Improvement over RL by Gurobi: {improvement:.2f}%")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL-Gurobi Hybrid Solver for OP")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/op100.yaml",
        help="Path to the OmegaConf YAML configuration file."
    )
    args = parser.parse_args()
    
    try:
        cfg = OmegaConf.load(args.config)
        main(cfg)
    except FileNotFoundError:
        print(f"ERROR: Configuration file not found at '{args.config}'")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")