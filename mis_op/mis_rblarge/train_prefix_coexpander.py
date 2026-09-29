# Curriculum training of the prefix-conditioned CO-Expander for MIS.
# Usage: python train_prefix_coexpander.py train_config.yaml

import os
import sys
import torch
from omegaconf import OmegaConf
from pytorch_lightning.strategies import DDPStrategy
# --- Project-related imports ---
root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_folder)

from ml4co_kit import Trainer
from co_expander import COExpanderCMModel, GNNEncoder, COExpanderEnv, COExpanderDecoder

def main():
    """Main entry point for training."""
    # 1. Load Configuration
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'train_config.yaml'
    if not os.path.exists(config_path):
        print(f"Error: Configuration file '{config_path}' not found.")
        sys.exit(1)
        
    cfg = OmegaConf.load(config_path)
    
    # Ensure checkpoint directory exists
    os.makedirs(cfg.train.ckpt_dir, exist_ok=True)

    print("===== COExpander MIS Prefix-Conditioned Training =====")
    print(f"Configuration loaded from: {config_path}")

    # 2. Initialize Model and Environment
    # Note: We pass an empty list for prefix_k_options initially.
    # It will be updated in the curriculum loop.
    env = COExpanderEnv(
        task=cfg.task,
        mode=cfg.mode,
        train_data_size=cfg.train_data_size,
        val_data_size=cfg.val_data_size,
        train_batch_size=cfg.train.batch_size,
        val_batch_size=cfg.val.batch_size,
        num_workers=cfg.train.num_workers,
        sparse_factor=cfg.sparse_factor,
        device="cuda", # The Trainer will handle device placement
        train_folder=cfg.data.train_path,
        val_path=cfg.data.val_path,
        store_data=cfg.store_data,
        prefix_k_options=[] 
    )

    encoder = GNNEncoder(**cfg.encoder)
    decoder = COExpanderDecoder(**cfg.decoder)

    model = COExpanderCMModel(
        env=env,
        encoder=encoder,
        decoder=decoder,
        **cfg.model_params
    )

    # 3. Curriculum Learning Loop
    last_checkpoint_path = cfg.train.resume_ckpt
    best_model_of_last_stage = None

    for stage_name in cfg.train.curriculum:
        stage_cfg = cfg.train.curriculum[stage_name]
        
        print(f"\n----- Starting Curriculum Stage: {stage_cfg.name} -----")
        print(f"Epochs for this stage: {stage_cfg.epochs}")
        print(f"Prefix k-options: {list(stage_cfg.prefix_k_options)}")
        
        # Update the model's environment with the new prefix options for this stage
        model.env.prefix_k_options = list(stage_cfg.prefix_k_options)

        # Load weights if available from the previous stage
# Load weights if available from the previous stage
        if last_checkpoint_path and os.path.exists(last_checkpoint_path):
            print(f"Loading weights from: {last_checkpoint_path}")
            
            # 1. Load the full PyTorch Lightning checkpoint
            checkpoint = torch.load(last_checkpoint_path, map_location="cpu")
            
            # 2. Extract the model state_dict from the checkpoint
            # PL stores it under the 'state_dict' key by default
            if 'state_dict' in checkpoint:
                model_state_dict = checkpoint['state_dict']
            else:
                # Fallback in case it is a raw state_dict
                model_state_dict = checkpoint
            
            # 3. Load the extracted state_dict into the model
            model.load_state_dict(model_state_dict)
        
        # Initialize the Trainer for the current stage
        # The Trainer from ml4co_kit automatically handles DDP when devices > 1
# --- MODIFY THIS PART ---
        trainer = Trainer(
            model=model,
            devices=cfg.train.devices, # Pass the list of GPU IDs
            max_epochs=stage_cfg.epochs,
            # Add the strategy argument here:
            strategy=DDPStrategy(find_unused_parameters=True),
            # Let the Trainer handle accelerators
        )
        # --- MODIFICATION END ---

        # Start training for the current stage
        trainer.model_train()

        # After training, find the best checkpoint to use for the next stage
        # This logic assumes ml4co_kit's Trainer saves checkpoints and exposes the path
        # A robust way is to find the latest created checkpoint in the log directory
        try:
            # This is a common pattern in PyTorch Lightning callbacks
            best_model_path = trainer.checkpoint_callback.best_model_path
            if best_model_path and os.path.exists(best_model_path):
                 last_checkpoint_path = best_model_path
                 best_model_of_last_stage = last_checkpoint_path
                 print(f"Stage '{stage_cfg.name}' finished. Best model for next stage is: {last_checkpoint_path}")
            else:
                 raise AttributeError # Fallback to last model
        except (AttributeError, FileNotFoundError):
             # Fallback: Save the last model state manually if best_model_path isn't available
             fallback_path = os.path.join(cfg.train.ckpt_dir, f"last_epoch_{stage_cfg.name}.pth")
             trainer.save_checkpoint(fallback_path)
             last_checkpoint_path = fallback_path
             best_model_of_last_stage = fallback_path
             print(f"Stage '{stage_cfg.name}' finished. Saved last epoch model to: {fallback_path}")
    
    # Finalize by renaming the last best model
    if best_model_of_last_stage:
        final_path = os.path.join(cfg.train.ckpt_dir, "final_best_model.pth")
        os.rename(best_model_of_last_stage, final_path)
        print(f"\n✅ Curriculum training finished! Final best model saved as: {final_path}")

if __name__ == "__main__":
    # torchrun will handle setting up the environment variables
    # We don't need manual DDP setup anymore
    main()

