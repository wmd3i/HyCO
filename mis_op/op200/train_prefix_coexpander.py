# Curriculum training of the prefix-conditioned CO-Expander for OP.
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
from co_expander import (
    COExpanderCMModel, GNNEncoder, COExpanderEnv,
    COExpanderDecoder, COExpanderSparser # <-- ADD COExpanderSparser HERE
)
def main():
    """Main entry point for OP prefix-conditioned training."""
    # 1. Load Configuration
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'train_config.yaml' # Default config name
    if not os.path.exists(config_path):
        print(f"Error: Configuration file '{config_path}' not found.")
        sys.exit(1)

    cfg = OmegaConf.load(config_path)

    # Ensure checkpoint directory exists
    os.makedirs(cfg.train.ckpt_dir, exist_ok=True)

    print("===== COExpander OP Prefix-Conditioned Training =====")
    print(f"Configuration loaded from: {config_path}")

    # 2. Initialize Model and Environment
    env = COExpanderEnv(
        task=cfg.task, # Should be "OP"
        mode=cfg.mode,
        train_data_size=cfg.train_data_size, # Might need adjustment based on OP data
        val_data_size=cfg.val_data_size,
        train_batch_size=cfg.train.batch_size,
        val_batch_size=cfg.val.batch_size,
        num_workers=cfg.train.num_workers,
        sparse_factor=cfg.sparse_factor,
        device="cuda",
        train_folder=cfg.data.train_path,
        val_path=cfg.data.val_path,
        store_data=cfg.store_data, # OP data might be large, consider False
        prefix_k_options=[] # Initial empty list
    )

    # Make sure encoder config includes OP specific params if needed
    encoder = GNNEncoder(
         task=cfg.encoder.task,
         sparse=cfg.encoder.sparse,
         block_layers=cfg.encoder.block_layers,
         hidden_dim=cfg.encoder.hidden_dim,
         time_flag=cfg.encoder.time_flag,
         prefix_cond_dim=cfg.encoder.prefix_cond_dim,
         prefix_enc_hidden_dim=cfg.encoder.prefix_enc_hidden_dim,
         max_length_cond_dim=cfg.encoder.max_length_cond_dim # Add max_length dim
         # Add other GNNEncoder params like aggregation, norm etc. if specified in config
    )

    decoder = COExpanderDecoder(**cfg.decoder)

    model = COExpanderCMModel(
        env=env,
        encoder=encoder,
        decoder=decoder,
        learning_rate=cfg.train.learning_rate, # Get LR from train config
        # weight_decay=cfg.train.weight_decay, # Add if needed
        **cfg.model_params
    )

    # 3. Curriculum Learning Loop (Similar to MIS script)
    last_checkpoint_path = cfg.train.get("resume_ckpt", None) # Use .get for optional resume
    best_model_of_last_stage = None

    if not hasattr(cfg.train, 'curriculum') or not cfg.train.curriculum:
         print("Error: 'curriculum' section not found or empty in training config.")
         sys.exit(1)


    for stage_name in cfg.train.curriculum:
        stage_cfg = cfg.train.curriculum[stage_name]

        print(f"\n----- Starting Curriculum Stage: {stage_cfg.name} -----")
        print(f"Epochs for this stage: {stage_cfg.epochs}")
        print(f"Prefix k-options: {list(stage_cfg.prefix_k_options)}")

        # Update prefix options for the current stage
        model.env.prefix_k_options = list(stage_cfg.prefix_k_options)
        # Also update the sparser instance within the env
        if isinstance(model.env.data_processor, COExpanderSparser):
             model.env.data_processor.prefix_k_options = list(stage_cfg.prefix_k_options)


        # Load weights from previous stage or resume_ckpt
        if last_checkpoint_path and os.path.exists(last_checkpoint_path):
             print(f"Loading weights from: {last_checkpoint_path}")
             try:
                 checkpoint = torch.load(last_checkpoint_path, map_location="cpu")
                 state_dict = checkpoint.get('state_dict', checkpoint) # Handle both formats
                 # --- Load with strict=False initially for flexibility ---
                 load_result = model.load_state_dict(state_dict, strict=False)
                 if load_result.missing_keys or load_result.unexpected_keys:
                      print("Warning: Mismatched keys during state_dict loading.")
                      print("Missing keys:", load_result.missing_keys)
                      print("Unexpected keys:", load_result.unexpected_keys)
                 else:
                      print("State dict loaded successfully.")
             except Exception as e:
                 print(f"Error loading checkpoint: {e}. Starting stage from scratch or previous state.")
                 # Decide whether to continue or exit based on severity

        # Initialize Trainer for the current stage

        trainer = Trainer(
            model=model,
            devices=cfg.train.devices, # Pass the list of GPU IDs
            max_epochs=stage_cfg.epochs,
            # Add the strategy argument here:
            strategy=DDPStrategy(find_unused_parameters=True),
            # Let the Trainer handle accelerators
        )
        # Start training
        trainer.model_train()

        # Find best checkpoint for next stage (using PL's callback attribute)
        try:
             # Check if checkpoint_callback exists and has best_model_path
             if hasattr(trainer, 'checkpoint_callback') and hasattr(trainer.checkpoint_callback, 'best_model_path'):
                 best_model_path = trainer.checkpoint_callback.best_model_path
                 if best_model_path and os.path.exists(best_model_path):
                      last_checkpoint_path = best_model_path
                      best_model_of_last_stage = last_checkpoint_path # Keep track of the absolute best
                      print(f"Stage '{stage_cfg.name}' finished. Best model path: {last_checkpoint_path}")
                 else:
                     print(f"Warning: best_model_path not found or invalid after stage '{stage_cfg.name}'. Using last saved model.")
                     # Fallback logic might be needed here if checkpoints aren't saved automatically
                     # Saving last manually:
                     fallback_path = os.path.join(cfg.train.ckpt_dir, f"{stage_cfg.name}_last.ckpt")
                     trainer.save_checkpoint(fallback_path)
                     last_checkpoint_path = fallback_path
                     best_model_of_last_stage = fallback_path
             else:
                  print("Warning: Checkpoint callback or best_model_path not found. Saving last model.")
                  fallback_path = os.path.join(cfg.train.ckpt_dir, f"{stage_cfg.name}_last.ckpt")
                  trainer.save_checkpoint(fallback_path)
                  last_checkpoint_path = fallback_path
                  best_model_of_last_stage = fallback_path

        except Exception as e:
             print(f"Error getting best checkpoint path: {e}. Saving last model.")
             fallback_path = os.path.join(cfg.train.ckpt_dir, f"{stage_cfg.name}_last.ckpt")
             trainer.save_checkpoint(fallback_path)
             last_checkpoint_path = fallback_path
             best_model_of_last_stage = fallback_path

    # Rename the final best model
    if best_model_of_last_stage and os.path.exists(best_model_of_last_stage):
        final_path = os.path.join(cfg.train.ckpt_dir, "final_best_model_op.ckpt") # Suffix with task
        try:
             os.rename(best_model_of_last_stage, final_path)
             print(f"\n✅ Curriculum training finished! Final best model saved as: {final_path}")
        except OSError as e:
             print(f"Error renaming final model: {e}. Best model remains at: {best_model_of_last_stage}")
    else:
        print("\nCurriculum training finished, but no best model path was found or saved.")


if __name__ == "__main__":
    main()