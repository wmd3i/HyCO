# Standalone evaluation of the prefix-conditioned CO-Expander on OP (greedy or best-of-k sampling).
# Usage: python eval_prefix_coexpander.py --k_samples 8

import os
import sys
import torch
import numpy as np
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm
import time
import argparse
import matplotlib.pyplot as plt

# --- Project-related imports ---
root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_folder)

from co_expander import (
    COExpanderCMModel, GNNEncoder, COExpanderEnv,
    COExpanderDecoder, COExpanderSparser
)
from co_expander.model.decoder.decode.op import calculate_op_metrics

# ==============================================================================
# === Visualization Function ===
# ==============================================================================
def visualize_op_solution(instance_locs, tour, prizes, title="OP Solution", reward=None, cost=None, ax=None):
    """Visualizes an OP solution."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    if isinstance(instance_locs, torch.Tensor):
        locs_cpu = instance_locs.cpu().numpy()
    else:
        locs_cpu = instance_locs

    if isinstance(prizes, torch.Tensor):
        prizes_cpu = prizes.cpu().numpy()
    else:
        prizes_cpu = prizes

    # Draw depot
    ax.scatter(locs_cpu[0, 0], locs_cpu[0, 1], c='black', s=150, label='Depot', zorder=5, marker='s')

    # Draw customers
    customer_locs = locs_cpu[1:]
    customer_prizes = prizes_cpu[1:]
    if customer_prizes.ndim > 1:
        customer_prizes = customer_prizes.squeeze()

    customer_sizes = 20 + np.maximum(customer_prizes, 0) * 100
    ax.scatter(customer_locs[:, 0], customer_locs[:, 1], c='lightblue', s=customer_sizes, label='Unvisited Customers', zorder=2, alpha=0.6)

    if tour:
        visited_indices = np.array(tour)
        if len(visited_indices) > 0:
            visited_locs = locs_cpu[visited_indices]
            visited_prizes = prizes_cpu[visited_indices]
            if visited_prizes.ndim > 1:
                visited_prizes = visited_prizes.squeeze()
            visited_sizes = 20 + np.maximum(visited_prizes, 0) * 100
            ax.scatter(visited_locs[:, 0], visited_locs[:, 1], c='red', s=visited_sizes, label='Visited Customers', zorder=3)

            tour_with_depot = [0] + tour + [0]
            tour_locs_plot = locs_cpu[tour_with_depot]
            ax.plot(tour_locs_plot[:, 0], tour_locs_plot[:, 1], color='maroon', marker='o', markersize=4, zorder=1, linestyle='-')

    plot_title = title
    if reward is not None and cost is not None:
        plot_title += f"\nReward: {reward:.4f} | Cost: {cost:.2f}"

    ax.set_title(plot_title)
    ax.legend(fontsize='small')
    ax.set_xlabel("X-coordinate")
    ax.set_ylabel("Y-coordinate")
    ax.set_aspect('equal', adjustable='box')


# ==============================================================================

def main(cfg: DictConfig, checkpoint_path: str):
    print("===== COExpander OP Evaluation =====")
    print(f"Using configuration derived from: {cfg.config_path}")
    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Running evaluation on '{cfg.eval.data_split}' data split.")
    
    # --- [1] number of samples per graph (K) ---
    k_samples = cfg.eval.k_samples
    print(f"Multi-Sampling Strategy: Best of {k_samples}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. Initialize Environment
    data_path = cfg.data.val_path if cfg.eval.data_split == 'val' else cfg.data.get('test_path', cfg.data.val_path)
    data_size = cfg.val_data_size if cfg.eval.data_split == 'val' else cfg.get('test_data_size', cfg.val_data_size)

    if not data_path or not os.path.exists(data_path):
        print(f"Error: Data path '{data_path}' for split '{cfg.eval.data_split}' not found.")
        sys.exit(1)

    env = COExpanderEnv(
        task=cfg.task,
        mode=cfg.eval.data_split,
        train_data_size=1,
        val_data_size=data_size,
        train_batch_size=1,
        val_batch_size=cfg.eval.batch_size,
        test_batch_size=cfg.eval.batch_size,
        num_workers=cfg.train.num_workers,
        sparse_factor=cfg.sparse_factor,
        device=device,
        train_folder=None,
        val_path=data_path,
        store_data=cfg.store_data,
        prefix_k_options=None
    )

    env.load_data()
    if env.val_data_cache is None or not env.val_data_cache.get("op_data"):
         print(f"Error: Failed to load data from {data_path}")
         sys.exit(1)
    actual_data_size = len(env.val_data_cache["op_data"])
    num_samples_to_evaluate = min(cfg.eval.num_samples_to_eval, actual_data_size)
    print(f"Loaded {actual_data_size} instances. Evaluating {num_samples_to_evaluate} instances.")

    # 3. Initialize Model Components
    encoder = GNNEncoder(
         task=cfg.encoder.task,
         sparse=cfg.encoder.sparse,
         block_layers=cfg.encoder.block_layers,
         hidden_dim=cfg.encoder.hidden_dim,
         time_flag=cfg.encoder.time_flag,
         prefix_cond_dim=cfg.encoder.prefix_cond_dim,
         prefix_enc_hidden_dim=cfg.encoder.prefix_enc_hidden_dim,
         max_length_cond_dim=cfg.encoder.max_length_cond_dim,
         aggregation=cfg.encoder.get("aggregation", "sum"),
         norm=cfg.encoder.get("norm", "layer"),
         learn_norm=cfg.encoder.get("learn_norm", True),
         track_norm=cfg.encoder.get("track_norm", False),
         mask_frozen=cfg.encoder.get("mask_frozen", False)
    )

    decoder = COExpanderDecoder(**cfg.decoder)

    try:
        model = COExpanderCMModel.load_from_checkpoint(
            checkpoint_path,
            env=env, encoder=encoder, decoder=decoder,
            learning_rate=cfg.train.learning_rate,
            cm_alpha=cfg.model_params.cm_alpha,
            cm_beta=cfg.model_params.cm_beta,
            prompt_prob=cfg.model_params.prompt_prob,
            delta_scale=cfg.model_params.delta_scale,
            inference_steps=cfg.model_params.inference_steps,
            determinate_steps=cfg.model_params.determinate_steps,
            beam_size=cfg.model_params.beam_size,
            energy_finetune=cfg.model_params.energy_finetune,
            map_location=device,
            strict=True
            )
        print(f"Checkpoint loaded successfully.")
    except Exception as e:
        print(f"Warning: Direct checkpoint load failed ({e}), trying state_dict...")
        model = COExpanderCMModel(
            env=env, encoder=encoder, decoder=decoder,
            learning_rate=cfg.train.learning_rate, **cfg.model_params
        )
        state_dict = torch.load(checkpoint_path, map_location=device)['state_dict']
        model.load_state_dict(state_dict, strict=True)

    model.eval()
    model.to(device)
    model.env.device = device
    if isinstance(model.env.data_processor, COExpanderSparser):
        model.env.data_processor.device = device

    # 6. Run Evaluation Loop
    all_solutions_gen = []
    all_prizes_gen = []
    all_lengths_gen = []
    all_prizes_gt = []
    all_lengths_gt = []

    num_batches = (num_samples_to_evaluate + cfg.eval.batch_size - 1) // cfg.eval.batch_size
    instance_count = 0
    start_time = time.time()
    num_visualized = 0

    print(f"\nStarting evaluation on {num_samples_to_evaluate} instances...")
    
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc=f"Eval (k={k_samples})"):
            if instance_count >= num_samples_to_evaluate:
                break

            # Generate batch data
            batch_data = env.generate_val_data(batch_idx)
            if batch_data is None: break

            raw_data_list = batch_data[11]
            current_batch_size = len(raw_data_list)
            
            # --- [2] initialize the best-solution tracker for the current batch ---
            # Stores the best result over the k samples for every instance in the batch
            # Structure: list of (best_tour_nodes, best_prize, best_length)
            batch_best_results = [None] * current_batch_size 

            # --- [3] multi-sampling loop ---
            for sample_idx in range(k_samples):
                # Run inference
                vars_heatmap = model.inference_edge_sparse_process(*batch_data)
                # Decode
                solutions_gen_batch = model.decoder.sparse_decode(vars_heatmap, *batch_data, return_cost=False)
                
                # Evaluate the current sample and update the best record
                for i in range(current_batch_size):
                    coords, instance_prizes, max_len, _ = raw_data_list[i]
                    
                    # Get the tour of the current sample
                    gen_tour_nodes = list(solutions_gen_batch[i])
                    
                    # Compute metrics (prize and length)
                    prize_curr, length_curr = calculate_op_metrics(coords, gen_tour_nodes, instance_prizes)
                    
                    # --- [4] best-of-K selection ---
                    # 1. First sample: store directly
                    # 2. Later samples: update if the prize is higher
                    # 3. Update if the prize is equal but the length is shorter
                    if batch_best_results[i] is None:
                        batch_best_results[i] = (gen_tour_nodes, prize_curr, length_curr)
                    else:
                        best_tour, best_prize, best_len = batch_best_results[i]
                        
                        # Simple lexicographic comparison: higher prize is better, shorter length is better
                        # Note: if the decoder can produce solutions that violate the max-length constraint,
                        # add a check here: if length_curr <= max_len ...
                        
                        if prize_curr > best_prize:
                            batch_best_results[i] = (gen_tour_nodes, prize_curr, length_curr)
                        elif prize_curr == best_prize and length_curr < best_len:
                             batch_best_results[i] = (gen_tour_nodes, prize_curr, length_curr)

            # --- [5] record the final best-of-K result and compare with GT ---
            for i in range(current_batch_size):
                current_instance_index = instance_count + i
                if current_instance_index >= num_samples_to_evaluate:
                    break 

                coords, instance_prizes, max_len, gt_tour_list = raw_data_list[i]
                
                # Take the best result just computed
                best_gen_tour, best_gen_prize, best_gen_len = batch_best_results[i]

                # Ground truth calculation
                gt_tour_nodes_customers = [node for node in gt_tour_list if node != 0]
                prize_gt, length_gt = calculate_op_metrics(coords, gt_tour_nodes_customers, instance_prizes)

                all_solutions_gen.append(best_gen_tour)
                all_prizes_gen.append(best_gen_prize)
                all_lengths_gen.append(best_gen_len)
                all_prizes_gt.append(prize_gt)
                all_lengths_gt.append(length_gt)

                # Visualization (Optional: Visualize the best result found)
                if num_visualized < cfg.eval.num_samples_to_visualize:
                    vis_dir = cfg.eval.visualization_dir
                    os.makedirs(vis_dir, exist_ok=True)
                    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

                    visualize_op_solution(coords, best_gen_tour, instance_prizes, f"Generated (Best of {k_samples})", best_gen_prize, best_gen_len, ax=axes[0])
                    visualize_op_solution(coords, gt_tour_nodes_customers, instance_prizes, "Ground Truth", prize_gt, length_gt, ax=axes[1])

                    fig.suptitle(f"Instance #{current_instance_index} (Max Len: {max_len:.2f})", fontsize=16)
                    save_path = os.path.join(vis_dir, f"op_vis_{current_instance_index}_k{k_samples}.png")
                    plt.savefig(save_path)
                    plt.close(fig)
                    num_visualized += 1

            instance_count += current_batch_size 

    # 7. Output Results
    total_time = time.time() - start_time
    num_evaluated = len(all_prizes_gen)

    print("\n" + "="*60)
    print(f"--- COExpander OP Evaluation (k={k_samples}) Summary ---")
    print(f"Evaluated {num_evaluated} instances in {total_time:.2f}s.")

    if num_evaluated > 0:
        avg_prize_gen = np.mean(all_prizes_gen)
        avg_length_gen = np.mean(all_lengths_gen)
        avg_prize_gt = np.mean(all_prizes_gt)
        avg_length_gt = np.mean(all_lengths_gt)

        print(f"Average Generated Prize : {avg_prize_gen:.4f}")
        print(f"Average Ground Truth Prize: {avg_prize_gt:.4f}")
        gap = ((avg_prize_gt - avg_prize_gen) / avg_prize_gt) * 100 if avg_prize_gt > 1e-6 else 0.0
        print(f"Optimality Gap (Prize)  : {gap:.2f}%")
        print("-" * 30)
        print(f"Average Generated Length: {avg_length_gen:.4f}")
        print(f"Average Ground Truth Length: {avg_length_gt:.4f}")
        print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate COExpander on OP with Multi-Sampling")
    parser.add_argument("--config", type=str, default='train_config.yaml', help="Config file path")
    parser.add_argument("--checkpoint", type=str, default="../checkpoints/coexpander/op50.ckpt", help="Path to the trained model .ckpt checkpoint file")
    parser.add_argument("--eval_split", type=str, default='test', choices=['val', 'test'], help="Split to evaluate")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_samples", type=int, default=500, help="Total instances to evaluate")
    
    # --- [New Argument] ---
    parser.add_argument("--k_samples", type=int, default=64, help="Number of samples per instance (k)")
    
    parser.add_argument("--num_visualize", type=int, default=5, help="Number of visualizations")
    parser.add_argument("--vis_dir", type=str, default='./eval_vis_op', help="Visualization dir")

    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config not found: {args.config}")
    cfg = OmegaConf.load(args.config)
    cfg.config_path = args.config

    eval_cfg = OmegaConf.create({
        'eval': {
            'data_split': args.eval_split,
            'batch_size': args.eval_batch_size,
            'num_samples_to_eval': args.num_samples,
            'num_samples_to_visualize': args.num_visualize,
            'visualization_dir': args.vis_dir,
            'k_samples': args.k_samples # Added to config
        }
    })
    final_cfg = OmegaConf.merge(cfg, eval_cfg)

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    main(final_cfg, args.checkpoint)