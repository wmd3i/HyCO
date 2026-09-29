# Standalone evaluation of the prefix-conditioned CO-Expander on MIS (no RL prefix), on the same test graphs as HyCO.
# Usage: python eval_prefix_coexpander.py
import os
import sys
import torch
import numpy as np
import scipy.sparse as sp
from omegaconf import OmegaConf
from tqdm.auto import tqdm
import dgl

# --- 1. Path setup ---
base_dir = os.path.dirname(os.path.abspath(__file__))
root_folder = os.path.dirname(base_dir) 
lwd_path = os.path.abspath(os.path.join(base_dir, 'learning_what_to_defer'))

if root_folder not in sys.path:
    sys.path.append(root_folder)
if lwd_path not in sys.path:
    sys.path.append(lwd_path)

# --- Imports ---
try:
    from ml4co_kit import MISGraphData
    from co_expander import (
        COExpanderEnv, GNNEncoder, COExpanderDecoder, COExpanderCMModel
    )
    # LwD data loader
    from learning_what_to_defer.data.graph_dataset import get_er_700_800_dataset
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)


def prepare_input_tuple(g, device):
    """
    Convert an LwD DGL graph into the input format required by CO-Expander (version without ground truth)
    """
    N = g.number_of_nodes()
    src, dst = g.edges()
    
    # Build edge_index
    edge_index = torch.stack([src, dst], dim=0).to(device)
    E = edge_index.shape[1]
    
    # Basic features
    task = "MIS"
    nodes_feature = None
    x = torch.zeros(N, device=device) 
    edges_feature = torch.ones(E, device=device)
    e = None
    
    # Adjacency matrix (sparse)
    adj_matrix_sp = sp.coo_matrix((np.ones(E), (src.cpu().numpy(), dst.cpu().numpy())), shape=(N, N))
    graph_list = [adj_matrix_sp]
    
    mask = torch.zeros(N, dtype=torch.bool, device=device)
    
    # --- Labels are not extracted; set to all zeros ---
    ground_truth = torch.zeros(N, dtype=torch.long, device=device)
        
    nodes_num_list = [N]
    edges_num_list = [E]
    
    raw_data = MISGraphData()
    raw_data_list = [raw_data]
    
    # No prefix
    prefix_nodes_tensor = torch.tensor([], dtype=torch.long, device=device)
    prefix_nodes_batch_list = [prefix_nodes_tensor]
    node_prefix_state = torch.zeros((N, 1), dtype=torch.float32, device=device)
    
    return (
        task, nodes_feature, x, edges_feature, e, edge_index, graph_list,
        mask, ground_truth, nodes_num_list, edges_num_list, raw_data_list,
        prefix_nodes_batch_list, node_prefix_state
    )


def run_inference_on_lwd_data():
    print("--- CO-Expander Pure Inference on LwD Dataset (No Labels) ---")

    # 1. Load Configuration
    config_path = 'infer_config.yaml'
    if not os.path.exists(config_path):
        print(f"Error: Config '{config_path}' not found.")
        sys.exit(1)
        
    cfg = OmegaConf.load(config_path)
    
    num_samples = cfg.inference.get('num_samples', 1) 
    print(f"Multi-Sampling Enabled: k={num_samples}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Initialize Dummy Environment
    dummy_env = COExpanderEnv(
        task=cfg.task,
        val_path=cfg.data.test_path, 
        val_data_size=500,
        device=device
    )

    # 3. Load LwD Dataset
    lwd_data_path = "../data/lwd/er_700_800/test"
    print(f"Loading LwD dataset from: {lwd_data_path}")
    
    try:
        dataset = get_er_700_800_dataset("test", lwd_data_path)
        print(f"Loaded {len(dataset)} graphs from LwD folder.")
    except Exception as e:
        print(f"Failed to load LwD dataset: {e}")
        sys.exit(1)

    # 4. Initialize Model
    encoder = GNNEncoder(**cfg.encoder)
    decoder = COExpanderDecoder(**cfg.decoder)

    model = COExpanderCMModel(
        env=dummy_env,
        encoder=encoder,
        decoder=decoder,
        learning_rate=cfg.train.learning_rate,
        **cfg.model_params
    )

    # 5. Load Weights
    model_path = cfg.inference.ckpt_path
    print(f"Loading checkpoint: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    # 6. Inference Loop
    total_pred_size = 0
    num_graphs_processed = 0
    max_samples = cfg.inference.num_test_samples
    
    pbar = tqdm(dataset, desc="Inference")
    for i, g in enumerate(pbar):
        print(i)
        if i >= max_samples:
            break
            
        g = g.to(device)
        input_tuple = prepare_input_tuple(g, device)
        
        # Multi-Sampling Logic
        best_sample_size = 0
        
        with torch.no_grad():
            for _ in range(num_samples):
                vars_output = model.inference_node_sparse_process(*input_tuple)
                solutions = model.decoder.sparse_decode(vars_output, *input_tuple, return_cost=False)
                
                current_sol = solutions[0] 
                current_size = np.sum(current_sol)
                
                if current_size > best_sample_size:
                    best_sample_size = current_size
        
        total_pred_size += best_sample_size
        num_graphs_processed += 1
        
        # Show the running average size
        pbar.set_postfix({"AvgSize": f"{total_pred_size / num_graphs_processed:.2f}"})

    # 7. Final Report
    print("\n" + "="*60)
    print("--- Pure CO-Expander Results (LwD Data) ---")
    if num_graphs_processed > 0:
        avg_pred = total_pred_size / num_graphs_processed
        
        print(f"Graphs Processed: {num_graphs_processed}")
        print(f"Sampling per graph (k): {num_samples}")
        print(f"-"*30)
        print(f"Average Predicted MIS Size:    {avg_pred:.4f}")
        print(f"(Note: No Ground Truth available for Gap calculation)")
    else:
        print("No graphs processed.")
    print("="*60)

if __name__ == "__main__":
    run_inference_on_lwd_data()