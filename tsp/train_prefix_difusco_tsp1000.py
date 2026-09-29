# Transfer (from TSP-100) + curriculum training of the sparse Prefix-DIFUSCO on TSP-1000.
# Usage: torchrun --nproc_per_node=2 train_prefix_difusco_tsp1000.py

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
import os
import time
from tqdm import tqdm

from data_loader_sparse import TSPConditionalSuffixDataset, custom_collate_fn
from diffusion_model_sparse import ConditionalTSPSuffixDiffusionModel
from discrete_diffusion_sparse import AdjacencyMatrixDiffusion
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast

# <<< MODIFIED: validate_model to handle the new batch structure
@torch.no_grad()
def validate_model(model, diffusion_handler, valid_dataloader, device):
    model.eval()
    total_valid_loss = 0
    num_batches = 0

    for batch_data in valid_dataloader:
        # Move all tensor data to device
        for k, v in batch_data.items():
            if torch.is_tensor(v):
                batch_data[k] = v.to(device)

        t = torch.randint(1, diffusion_handler.num_timesteps + 1, (batch_data["prefix_lengths"].size(0),), device=device).long()
        
        loss = diffusion_handler.training_loss(model, batch_data, t)
        total_valid_loss += loss.item()
        num_batches += 1
    
    if num_batches == 0: return float('inf')

    total_loss_tensor = torch.tensor([total_valid_loss, num_batches], dtype=torch.float64, device=device)
    dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
    global_total_loss, global_num_batches = total_loss_tensor[0].item(), total_loss_tensor[1].item()
    
    return global_total_loss / global_num_batches if global_num_batches > 0 else float('inf')


def ddp_setup():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank), local_rank


def load_weights_for_transfer_learning(new_model, pretrained_ckpt_path, device):
    """
    Intelligently loads weights from a pretrained model to a new model with a different size (e.g., N).
    It only loads weights for layers with matching names and shapes.

    Args:
        new_model: The newly initialized model (e.g., for TSP200).
        pretrained_ckpt_path: Path to the pretrained model checkpoint (e.g., from TSP100).
        device: The device to load the checkpoint onto.
    """
    # 1. Load the state_dict from the pretrained model checkpoint
    if not os.path.exists(pretrained_ckpt_path):
        print(f"Pretrained checkpoint not found at {pretrained_ckpt_path}. Starting from scratch.")
        return new_model

    print(f"Loading pretrained weights from {pretrained_ckpt_path} for transfer learning...")
    pretrained_dict = torch.load(pretrained_ckpt_path, map_location=device)
    
    # 2. Get the state_dict of the new model architecture
    new_model_dict = new_model.state_dict()
    
    # 3. Filter the pretrained_dict to include only layers that match in name and shape
    weights_to_load = {k: v for k, v in pretrained_dict.items() if k in new_model_dict and v.shape == new_model_dict[k].shape}
    
    # 4. Update the new model's state_dict with the transferable weights
    new_model_dict.update(weights_to_load)
    
    # 5. Load the updated state_dict into the new model
    # Use strict=False because some layers might be intentionally left out if their shapes differ.
    new_model.load_state_dict(new_model_dict, strict=False)
    
    # Optional: Print a report on what was transferred
    print("-" * 60)
    print("Transfer Learning Report:")
    print(f"Successfully transferred {len(weights_to_load)} tensors from the pretrained model.")
    
    loaded_keys = weights_to_load.keys()
    # Find keys in the new model that were not loaded from the pretrained one
    uninitialized_keys = [k for k in new_model_dict.keys() if k not in loaded_keys]
    
    if uninitialized_keys:
        print(f"Skipped or left uninitialized {len(uninitialized_keys)} tensors (this is expected for size-dependent layers).")
        # For debugging, you can print the skipped keys:
        # print("Uninitialized keys:", uninitialized_keys)
    else:
        print("All layers from the new model were successfully initialized from the pretrained model!")
    print("-" * 60)
    
    return new_model


def run_training_stage(cfg: DictConfig, stage_name: str, prefix_k_options: list, epochs_for_stage: int, device, local_rank, checkpoint_to_load: str = None):
    """
    Executes a single stage of the training curriculum.
    """
    
    if dist.get_rank() == 0:
        print(f"\n===== Starting Curriculum Stage: {stage_name} =====")
        print(f"===== Epochs: {epochs_for_stage}, Prefix K Range: {prefix_k_options[0]}-{prefix_k_options[-1]} =====")
        print(f"prefix in this stage is {prefix_k_options}")
        if checkpoint_to_load:
            print(f"===== Loading checkpoint from: {checkpoint_to_load} =====")


    time.sleep(2) # Pause for readability

    ckpt_dir = cfg.train.get("ckpt_dir", "./ckpt_difusco_style")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    prefix_sampling_strategy = cfg.data.get('prefix_sampling_strategy', 'continuous_from_start')

    #global_batch_size = cfg.train.batch_size
    #per_gpu_batch_size = global_batch_size // dist.get_world_size()  # per-process share

        # <<< MODIFIED: Pass sparse_factor to Dataset
    sparse_factor = cfg.model.get("sparse_factor", -1)
    
    train_dataset = TSPConditionalSuffixDataset(
        npz_file_path=cfg.data.train_path,
        prefix_k_options=prefix_k_options,
        prefix_sampling_strategy=cfg.data.prefix_sampling_strategy,
        sparse_factor=sparse_factor
    )
    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg.train.batch_size, shuffle=False,
        sampler=train_sampler, num_workers=cfg.train.get("num_workers", 4),
        collate_fn=custom_collate_fn
    )
    
    valid_dataloader = None
    if cfg.data.get("valid_path"):
        valid_dataset = TSPConditionalSuffixDataset(
            npz_file_path=cfg.data.valid_path,
            prefix_k_options=prefix_k_options,
            prefix_sampling_strategy=cfg.data.prefix_sampling_strategy,
            sparse_factor=sparse_factor
        )
        valid_sampler = DistributedSampler(valid_dataset, shuffle=False)
        valid_dataloader = DataLoader(
            valid_dataset, batch_size=cfg.train.batch_size, sampler=valid_sampler,
            shuffle=False, num_workers=cfg.train.get("num_workers", 4),
            collate_fn=custom_collate_fn
        )

    # <<< MODIFIED: Pass sparse_factor to Model
    model = ConditionalTSPSuffixDiffusionModel(
        num_nodes=cfg.model.num_nodes, node_coord_dim=cfg.model.node_coord_dim,
        pos_embed_num_feats=cfg.model.pos_embed_num_feats, node_embed_dim=cfg.model.node_embed_dim,
        prefix_node_embed_dim=cfg.model.node_embed_dim,
        prefix_enc_hidden_dim=cfg.model.prefix_enc_hidden_dim, prefix_cond_dim=cfg.model.prefix_cond_dim,
        gnn_n_layers=cfg.model.gnn_n_layers, gnn_hidden_dim=cfg.model.gnn_hidden_dim,
        gnn_aggregation=cfg.model.gnn_aggregation, gnn_norm=cfg.model.gnn_norm,
        gnn_learn_norm=cfg.model.gnn_learn_norm, gnn_gated=cfg.model.gnn_gated,
        time_embed_dim=cfg.model.time_embed_dim,
        sparse_factor=sparse_factor # Pass the factor
    ).to(device)

    # 2. Load the single-GPU training checkpoint (its keys have no 'module.' prefix either)
    #    Since neither has the 'module.' prefix, the keys match exactly
    if checkpoint_to_load:
        model_checkpoint_path = checkpoint_to_load
        if os.path.exists(model_checkpoint_path):
            try:
            # The function will print a detailed report, so we don't need extra prints here.
                model = load_weights_for_transfer_learning(model, checkpoint_to_load, device)
            except Exception as e:
                if dist.get_rank() == 0:
                    print(f"Successfully loaded single-GPU checkpoint into base model from {model_checkpoint_path}")
            except Exception as e:
                if dist.get_rank() == 0:
                    print(f"Could not load checkpoint: {e}. Starting from scratch.")
        else:
            if dist.get_rank() == 0:
                 print(f"Checkpoint file not found at {model_checkpoint_path}. Starting from scratch.")
    
    # (Optional but recommended) use a barrier to make sure all processes have finished loading
    dist.barrier()  # Make sure all processes have finished loading

    
    # 3. Finally, wrap the model with loaded weights in DDP
    #    DDP automatically adds the 'module.' prefix to all keys for gradient synchronization
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)  
    

    # <<< MODIFIED: Pass sparse_factor to Diffusion Handler
    diffusion_handler = AdjacencyMatrixDiffusion(
        num_nodes=cfg.model.num_nodes, num_timesteps=cfg.diffusion.num_timesteps,
        schedule_type=cfg.diffusion.schedule_type, device=device,
        sparse_factor=sparse_factor
    )

    optimizer = optim.Adam(model.parameters(), lr=cfg.train.learning_rate)
    scaler = GradScaler()# 20250626
    
    best_valid_loss = float('inf')
    epochs_no_improve = 0
    early_stopping_patience = cfg.train.get("early_stopping_patience", 10)
    min_delta = cfg.train.get("early_stopping_min_delta", 0.00001)
    
    # Main training loop for the stage
    for epoch in range(epochs_for_stage):
        train_sampler.set_epoch(epoch)
        model.train()
        total_train_loss = 0
        num_train_batches = 0
        is_main_process = (dist.get_rank() == 0)
        data_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs_for_stage}", disable=not is_main_process)

            
        for batch_idx, batch_data in enumerate(data_iterator):
            optimizer.zero_grad(set_to_none=True)
            
            # Move all tensors in the batch dict to the device
            for k, v in batch_data.items():
                if isinstance(v, torch.Tensor):
                    batch_data[k] = v.to(device, non_blocking=True)
            
            # The batch size is the number of graphs, which is the length of prefix_lengths
            current_batch_size = len(batch_data["prefix_lengths"])
            t = torch.randint(1, diffusion_handler.num_timesteps + 1, (current_batch_size,), device=device).long()

            with autocast():
                # CORRECTED: Pass the whole batch_data dictionary, which matches the
                # definition of training_loss in the diffusion handler.
                loss = diffusion_handler.training_loss(model, batch_data, t)
            
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            
            total_train_loss += loss.item()
            num_train_batches += 1

            if is_main_process and batch_idx % cfg.train.log_interval == 0 and batch_idx > 0:
                print(f"Stage '{stage_name}', Epoch {epoch+1}/{epochs_for_stage}, Batch {batch_idx}/{len(train_dataloader)}, Avg Train Loss: {(total_train_loss/num_train_batches):.5f}")
        
        print(f"Stage '{stage_name}', Epoch {epoch+1} completed. Average Training Loss: {(total_train_loss/num_train_batches):.5f}")

        if valid_dataloader:
            current_valid_loss = validate_model(model, diffusion_handler, valid_dataloader, device)
            print(f"Stage '{stage_name}', Epoch {epoch+1}: Validation Loss: {current_valid_loss:.5f}")
            if is_main_process:

                if current_valid_loss < best_valid_loss - min_delta:
                    best_valid_loss = current_valid_loss
                    epochs_no_improve = 0
                    best_model_path_stage = os.path.join(ckpt_dir, f"best_model_{stage_name}.pth")
                    torch.save(model.module.state_dict(), best_model_path_stage)
                    print(f"Validation loss improved. Saved best model for this stage to {best_model_path_stage}")
                else:
                    epochs_no_improve += 1
    
                if epochs_no_improve >= early_stopping_patience:
                    print(f"Early stopping triggered for stage '{stage_name}'.")
                    break
        if is_main_process:
            if (epoch + 1) % cfg.train.save_interval == 0:
                periodic_save_path = os.path.join(ckpt_dir, f"{stage_name}_epoch_{epoch+1}.pth")
                torch.save(model.module.state_dict(), periodic_save_path)
                
                print(f"Saved model checkpoint (periodic) at epoch {epoch+1} to {periodic_save_path}")
    print(f"Finished stage '{stage_name}'. Best validation loss for this stage: {best_valid_loss:.5f}")
    return os.path.join(ckpt_dir, f"best_model_{stage_name}.pth")

def ddp_cleanup():
    dist.destroy_process_group()

    
def train_with_curriculum(cfg: DictConfig):
    """
    Main function to orchestrate the curriculum learning process.
    """    
    device, local_rank = ddp_setup()
    print(f"[Rank {dist.get_rank()}] DDP setup complete. Using device: {device}")

    try:
        # Stage 1: Easy task - long prefixes

        stage1_k_options = list(range(1, 500))
        stage1_epochs = 5
        tsp100_best_ckpt = "./checkpoints/prefix_difusco/tsp100/Stage5_1_20_best_model_checkpoint.pth"

        stage1_best_ckpt = run_training_stage(
            cfg=cfg,
            stage_name="stage1_k0_500",
            prefix_k_options=stage1_k_options,
            epochs_for_stage=stage1_epochs,
            device=device,              # <<< pass device
            local_rank=local_rank,      # <<< pass local_rank
            checkpoint_to_load=tsp100_best_ckpt
        )
        #tage 2: Medium task - short prefixes
        stage2_k_options = list(range(1, 200))
        stage2_epochs = 5
        stage2_best_ckpt = run_training_stage(
            cfg=cfg,
            stage_name="stage2_k1_200",
            prefix_k_options=stage2_k_options,
            epochs_for_stage=stage2_epochs,
            device=device,              # <<< pass device
            local_rank=local_rank,      # <<< pass local_rank
            checkpoint_to_load=stage1_best_ckpt
        )

        # Stage 3: Full task - all prefixes
        stage3_k_options = list(range(1, cfg.model.num_nodes))
        stage3_epochs = 10
        final_best_ckpt = run_training_stage(
            cfg=cfg,
            stage_name="stage3_k1_1000_final",
            prefix_k_options=stage3_k_options,
            epochs_for_stage=stage3_epochs,
            device=device,              # <<< pass device
            local_rank=local_rank,      # <<< pass local_rank
            checkpoint_to_load=stage2_best_ckpt
        )

        # Stage 4: Front task - [1-30]
        stage4_k_options = list(range(1, 100))
        stage4_epochs = 5
        final_last_best_ckpt = run_training_stage(
            cfg=cfg,
            stage_name="stage4_k1_100_last",
            prefix_k_options=stage4_k_options,
            epochs_for_stage=stage4_epochs,
            device=device,              # <<< pass device
            local_rank=local_rank,      # <<< pass local_rank
            checkpoint_to_load=final_best_ckpt
        )

        if dist.get_rank() == 0:
            print("\nCurriculum training finished!")
            final_generic_path = os.path.join(os.path.dirname(stage1_best_ckpt), "Final_0_20_best_model_checkpoint.pth")
            if os.path.exists(stage1_best_ckpt):
                os.rename(stage1_best_ckpt, final_generic_path)
                print(f"Renamed final model to: {final_generic_path}")


    finally:

        if dist.is_initialized():
            rank = dist.get_rank()
            ddp_cleanup()
            print(f"[Rank {rank}] DDP resources cleaned up.")
        else:
            # This branch would execute if setup failed in the first place
            print("DDP was not initialized, no cleanup needed.")

if __name__ == "__main__":
    config_path = "configs/train_tsp1000.yaml" 
    try:
        config = OmegaConf.load(config_path)
        print("Loaded configuration from:", config_path)
    except FileNotFoundError:
        print(f"ERROR: Configuration file '{config_path}' not found.")
        exit()
    except Exception as e:
        print(f"Error loading configuration: {e}")
        exit()
        
    train_with_curriculum(config)
