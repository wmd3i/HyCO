# Discrete (Bernoulli) diffusion over adjacency matrices with prefix conditioning, DDPM/DDIM sampling (dense).
import torch
import torch.nn.functional as F
import numpy as np
import math
from tqdm.auto import tqdm # Use tqdm for a progress bar


# Inference schedule (adapted from reference_difusco/utils/diffusion_schedulers.py)
class InferenceSchedule:
  def __init__(self, inference_schedule="cosine", T=1000, inference_T=1000):
    self.inference_schedule = inference_schedule
    self.T = T # Total diffusion steps during training
    self.inference_T = inference_T # Number of steps for inference

  def __call__(self, i_inference_step):
    # i_inference_step is from 0 to inference_T-1
    assert 0 <= i_inference_step < self.inference_T

    if self.inference_schedule == "linear":
      # Timesteps go from T-1 down to 0
      t1 = self.T - int(((float(i_inference_step) / self.inference_T) * self.T))
      t1 = np.clip(t1, 1, self.T) # Current (noisier) timestep t
      
      t2 = self.T - int(((float(i_inference_step + 1) / self.inference_T) * self.T))
      t2 = np.clip(t2, 0, self.T -1) # Target (less noisy) timestep t-delta_t
      return t1, t2 # t, t_prev
      
    elif self.inference_schedule == "cosine":
      # Implements a cosine schedule for selecting timesteps during inference [cite: 381]
      # This schedule tends to take more steps in the low-noise regime (later in the process) [cite: 381]
      s = 0.008 # Offset to prevent beta_t from being too small at t=0
      
      # Current fraction of inference steps completed
      frac_current = (float(i_inference_step) / self.inference_T)
      # Next fraction of inference steps
      frac_next = (float(i_inference_step + 1) / self.inference_T)

      # Map fractions to actual timesteps T using cosine curve
      # Cosine curve goes from T (or T-1) down to 0
      def get_t_from_frac(frac):
          return int(self.T * (math.cos(frac * math.pi / 2 + s) / math.cos(s)))

      # This creates a sequence of timesteps that are more densely packed towards the end (low noise)
      time_points = np.linspace(0, self.T, self.inference_T + 1)
      time_points_cosine = self.T * (1 - np.cos(np.pi/2 * (time_points / self.T)))**2 # this is not quite right for sequence selection
      
      # Let's use the provided schedule from reference_difusco/utils/diffusion_schedulers.py (InferenceSchedule)
      # The t1, t2 are actual diffusion timesteps (1-indexed for t1, 0-indexed for t2)
      t1 = self.T - int(math.sin((float(i_inference_step) / self.inference_T) * np.pi / 2) * self.T)
      t1 = np.clip(t1, 1, self.T)

      t2 = self.T - int(math.sin((float(i_inference_step + 1) / self.inference_T) * np.pi / 2) * self.T)
      t2 = np.clip(t2, 0, self.T - 1)
      return t1, t2
    else:
      raise ValueError(f"Unknown inference schedule: {self.inference_schedule}")


class AdjacencyMatrixDiffusion:
    def __init__(self, num_nodes, num_timesteps, schedule_type='cosine', device='cpu'):
        self.num_nodes = num_nodes
        self.num_timesteps = num_timesteps
        self.device = device

        if schedule_type == 'cosine':
            # Cosine schedule for betas, as in D3PM and used by DIFUSCO for discrete diffusion.
            # This is beta_t for q(x_t | x_{t-1}) = Cat(x_t; p = x_{t-1} Q_t)
            # Q_t = [[1-beta_t, beta_t], [beta_t, 1-beta_t]]
            # Let's follow the beta schedule from reference_difusco/utils/diffusion_schedulers.py -> CategoricalDiffusion
            # Which itself follows DIFUSCO paper (Sec 4.1) beta_1=1e-4, beta_T=0.02 for linear
            # Or for cosine schedule, it derives betas from a cosine alpha_bar schedule
            s = 0.008
            t = torch.arange(num_timesteps + 1, dtype=torch.float64, device=device)
            alpha_bar = torch.cos((t / num_timesteps + s) / (1 + s) * math.pi * 0.5) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
            self.betas = torch.clip(betas, 0, 0.999).float() # (T)
        elif schedule_type == 'linear':
            # Linear schedule for beta_t [cite: 420]
            self.betas = torch.linspace(1e-4, 0.02, num_timesteps, device=device, dtype=torch.float32) # (T)

        else:
            raise ValueError(f"Unsupported schedule_type: {schedule_type}")

        self.alphas = (1.0 - self.betas) # (T)
        
        # Transition matrices Q_t = [[alpha_t, beta_t], [beta_t, alpha_t]]
        # Q_t[0,0] = alpha_t, Q_t[0,1] = beta_t (prob of 0->0, prob of 0->1)
        # Q_t[1,0] = beta_t, Q_t[1,1] = alpha_t (prob of 1->0, prob of 1->1)
        # This is (1-beta_t)*I + beta_t*ONES_MATRIX/2 if K=2 in D3PM general form
        # For binary {0,1}, Q_t_ij = P(X_t = j | X_{t-1} = i)
        # Q_t = [[1-beta_t, beta_t], [beta_t, 1-beta_t]]
        
        # Store Q_t and Q_bar_t (cumulative product)
        # Q_t is (T, 2, 2)
        self.Q_t = torch.zeros(num_timesteps, 2, 2, device=device, dtype=torch.float32)
        for t_idx in range(num_timesteps):
            beta_t = self.betas[t_idx]
            alpha_t = self.alphas[t_idx]
            self.Q_t[t_idx] = torch.tensor([[alpha_t, beta_t], [beta_t, alpha_t]], device=device)

        # Q_bar_t = Q_1 * Q_2 * ... * Q_t
        # Q_bar_t is (T+1, 2, 2) where Q_bar_t[0] is identity
        self.Q_bar_t = torch.zeros(num_timesteps + 1, 2, 2, device=device, dtype=torch.float32)
        self.Q_bar_t[0] = torch.eye(2, device=device)
        for t_idx in range(num_timesteps):
            self.Q_bar_t[t_idx+1] = torch.matmul(self.Q_bar_t[t_idx], self.Q_t[t_idx])
        
        # For q_sample, we need probability of x_t given x_0.
        # P(x_t=1|x_0=0) = Q_bar_t[0,1]
        # P(x_t=0|x_0=0) = Q_bar_t[0,0]
        # P(x_t=1|x_0=1) = Q_bar_t[1,1]
        # P(x_t=0|x_0=1) = Q_bar_t[1,0]

    def q_sample(self, x_0_adj_matrix, t_steps):
        """
        Noises a binary (0/1) adjacency matrix x_0 to x_t.
        x_0_adj_matrix: (B, N, N) binary ground truth {0,1}.
        t_steps: (B) tensor of timesteps (1 to T).
        Returns x_t (binary {0,1}) and the probabilities used for sampling x_t.
        """
        B, N, _ = x_0_adj_matrix.shape
        
        Q_bar_t_selected = self.Q_bar_t[t_steps] # Shape: (B, 2, 2)

        x_0_flat = x_0_adj_matrix.reshape(B, -1).float() # Shape: (B, N*N)
        
        # P(x_t=1 | x_0=1) for each batch item. Shape: (B, 1) for broadcasting
        prob_xt_eq_1_given_x0_eq_1 = Q_bar_t_selected[:, 1, 1].unsqueeze(1)
        # P(x_t=1 | x_0=0) for each batch item. Shape: (B, 1) for broadcasting
        prob_xt_eq_1_given_x0_eq_0 = Q_bar_t_selected[:, 0, 1].unsqueeze(1)

        # Calculate P(x_t=1 | x_0) for each element
        # x_0_flat * P(x_t=1|x_0=1) + (1-x_0_flat) * P(x_t=1|x_0=0)
        # (B, N*N) * (B, 1) -> broadcasts to (B, N*N)
        prob_xt_is_one = x_0_flat * prob_xt_eq_1_given_x0_eq_1 + \
                         (1 - x_0_flat) * prob_xt_eq_1_given_x0_eq_0
        
        prob_xt_is_one = prob_xt_is_one.reshape(B, N, N) # Shape: (B, N, N)

        # Sample x_t from Bernoulli(prob_xt_is_one)
        x_t = torch.bernoulli(prob_xt_is_one).float() # (B, N, N) - binary
        return x_t, prob_xt_is_one

    def training_loss(self, denoiser_model, x_0_adj_matrix, t_steps, instance_locs, prefix_nodes, prefix_lengths, node_prefix_state):
        """
        Masked loss.
        The loss is only computed on the "suffix", i.e., not on edges between two prefix nodes.
        This forces the model to spend all of its capacity on learning how to best complete the given prefix.
        """
        B, N, _ = instance_locs.shape
        device = instance_locs.device

        # 1. Sample x_t from q(x_t | x_0) as usual
        x_t_adj_matrix, noise_adj_matrix = self.q_sample(x_0_adj_matrix, t_steps)
        
        # === [Key change] 20250626 ===
        # Following DIFUSCO, map the input xt from {0, 1} to [-1, 1] and add a small amount of noise.
        # This stabilizes training and improves gradient flow.
        x_t_transformed = x_t_adj_matrix.float() * 2.0 - 1.0
        x_t_transformed = x_t_transformed * (1.0 + 0.05 * torch.rand_like(x_t_transformed))
        # ====================

        # 2. Let the model predict the logits of x_0 as usual
        predicted_x_0_logits = denoiser_model(
            x_t_transformed, t_steps.float(), instance_locs, #x_t_adj_matrix.float()
            prefix_nodes, prefix_lengths, node_prefix_state
        )

        # 3. === Build and apply the loss mask ===
        # Build a mask marking edges on which no loss should be computed (edges with both endpoints inside the prefix)
        prefix_mask = (node_prefix_state == 1.0).squeeze(-1)  # Shape: (B, N)
        prefix_row_mask = prefix_mask.unsqueeze(2).expand(B, N, N)
        prefix_col_mask = prefix_mask.unsqueeze(1).expand(B, N, N)
        
        # internal_prefix_mask is True where both endpoints of the edge lie inside the prefix
        internal_prefix_mask = prefix_row_mask & prefix_col_mask

        # Only compute the loss on positions that are not internal prefix edges
        loss_mask = ~internal_prefix_mask
        
        # Also skip the diagonal (self-loops)
        identity_mask = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0).expand(B, N, N)
        loss_mask &= ~identity_mask

        # Flatten targets and predictions, keeping only elements that contribute to the loss
        target = x_0_adj_matrix[loss_mask]
        prediction = predicted_x_0_logits[loss_mask]

        # Compute the loss only on unmasked elements
        loss = F.binary_cross_entropy_with_logits(prediction, target.float())
        # ========================================

        return loss

    def get_selection_schedule(self, num_inference_steps, schedule_type):
        """
        Timestep selection logic.
        Returns a list of tuples [(t_current, t_prev), ...] for reverse diffusion.
        """
        timesteps = []
        if schedule_type == 'linear':
            # Linear spacing
            t_points = np.linspace(self.num_timesteps, 0, num_inference_steps + 1).astype(int)
        elif schedule_type == 'cosine':
            # Cosine spacing
            # Decrease linearly from 1.0 to 0.0
            s_points = np.linspace(1.0, 0.0, num_inference_steps + 1)
            # Apply the cosine transform
            t_points = (0.5 * self.num_timesteps * (1 + np.cos(s_points * np.pi))).astype(int)
        elif schedule_type == 'polynomial':
            power = 3  # Tunable hyperparameter (20250625)
            t_points = np.linspace(self.num_timesteps** (1/power), 0, num_inference_steps + 1) ** power
            t_points = t_points.astype(int)

        else:
            raise ValueError(f"Unknown inference schedule: {schedule_type}")
        
        # Make sure t_points are unique to avoid repeated steps
        unique_t_points = np.unique(t_points)
        if len(unique_t_points) < 2: # If all points are identical, fall back to linear
             t_points = np.linspace(self.num_timesteps, 0, num_inference_steps + 1).astype(int)
        else:
             t_points = unique_t_points
        
        # Sort in descending order
        t_points = np.sort(t_points)[::-1]
        
        for i in range(len(t_points) - 1):
            t_current = max(1, t_points[i])   # t_current is at least 1
            t_prev = t_points[i+1]            # t_prev may be 0
            timesteps.append((t_current, t_prev))
            
        return timesteps

    
    @torch.no_grad()
    def p_sample_loop_ddim(self, denoiser_model, instance_locs, prefix_nodes, prefix_lengths, node_prefix_state, num_inference_steps=5, schedule='cosine', visualize_instance_idx=-1):
        """
        NEW: Implements a DDIM-like deterministic sampling loop for discrete diffusion.
        This is significantly faster as it bypasses the complex posterior calculation.
        MODIFIED: Now supports capturing intermediate frames for visualization.
        """
        B, N, _ = instance_locs.shape
        device = self.device
        
        # Start with a random graph
        img = torch.randint(0, 2, (B, N, N), device=device).float()
        img = (img + img.transpose(1, 2)).clamp(0, 1)

        timesteps_pairs = self.get_selection_schedule(num_inference_steps, schedule)

        # --- List for storing visualization frames ---
        viz_frames = []
        num_steps = len(timesteps_pairs)
        # Indices of the frames to capture (early, middle, late)
        # Capture the model's prediction of x_0, which best represents the "denoised" graph
        if visualize_instance_idx != -1 and num_steps > 2:
            viz_indices = {0,1,2,3,4,5,7,9,10}
        else:
            viz_indices = set()
        # --- End of addition ---

        final_logits = None # Stores the logits of the last step

        for i, (t_current, t_prev) in enumerate(tqdm(timesteps_pairs, desc='DDIM Denoising...', leave=False)):
            t_current_tensor = torch.full((B,), t_current, device=device, dtype=torch.long)
            
            # 1. Get the model's prediction for x_0
            img_transformed = img.float() * 2.0 - 1.0 # 1117
            if t_current > 0: # t_current is at least 1 inside the loop
                img_transformed = img_transformed * (1.0 + 0.05 * torch.rand_like(img_transformed))
            # --- [End of addition] ---

            pred_x0_logits = denoiser_model(
                img_transformed, t_current_tensor.float(), instance_locs,
                prefix_nodes, prefix_lengths, node_prefix_state
            )
            pred_x0_probs = torch.sigmoid(pred_x0_logits)
            
            if i == num_steps - 1: # Save the logits of the last step
                final_logits = pred_x0_logits.clone()

            # --- Optional but recommended: Guidance/Sharpening ---
            guidance_strength = 1.7
            pred_x0_probs_guided = torch.pow(pred_x0_probs, guidance_strength)
            pred_x0_probs_guided = pred_x0_probs_guided / (pred_x0_probs_guided + torch.pow(1 - pred_x0_probs, guidance_strength))
            
            # --- Clamp prefix edges to 1 ---
            prefix_adj_mask = torch.zeros_like(pred_x0_probs, dtype=torch.bool)
            for b in range(B):
                k = prefix_lengths[b].item()
                if k > 1:
                    p_nodes = prefix_nodes[b, :k]
                    prefix_adj_mask[b, p_nodes[:-1], p_nodes[1:]] = True
                    prefix_adj_mask[b, p_nodes[1:], p_nodes[:-1]] = True
            pred_x0_probs_clamped = torch.where(prefix_adj_mask, 1.0, pred_x0_probs_guided)

            # --- Capture visualization frames ---
            if i in viz_indices:
                frame = pred_x0_probs_clamped[visualize_instance_idx].cpu().clone()
                viz_frames.append(frame)
            # --- End of addition ---

            if t_prev == 0:
                img = torch.bernoulli(pred_x0_probs_clamped)
                continue

            # 2. THE CORE OF DISCRETE DDIM
            Q_bar_t_prev = self.Q_bar_t[t_prev]
            
            prob_xtm1_eq_1_given_x0_eq_1 = Q_bar_t_prev[1, 1]
            prob_xtm1_eq_1_given_x0_eq_0 = Q_bar_t_prev[0, 1]

            probs_xt_minus_1_is_1 = (
                prob_xtm1_eq_1_given_x0_eq_1 * pred_x0_probs_clamped +
                prob_xtm1_eq_1_given_x0_eq_0 * (1 - pred_x0_probs_clamped)
            )
            
            # 3. Sample the next state x_{t-1}
            img = torch.bernoulli(probs_xt_minus_1_is_1)

        final_adj = (img + img.transpose(1, 2)).clamp(0, 1)
        
        # Return the captured frames
        return final_adj, torch.sigmoid(final_logits), viz_frames


    
    @torch.no_grad()
    def p_sample_loop(self, denoiser_model, instance_locs, prefix_nodes, prefix_lengths, node_prefix_state, num_inference_steps=50, schedule='cosine'):
        B, N, _ = instance_locs.shape
        device = self.device
        
        img = torch.randint(0, 2, (B, N, N), device=device).float()
        img = (img + img.transpose(1, 2)).clamp(0, 1)

        # === Use the get_selection_schedule method ===
        timesteps_pairs = self.get_selection_schedule(num_inference_steps, schedule)

        # ==================== 20250625: resampling parameters ====================#

        for t_current, t_prev in tqdm(timesteps_pairs, desc='Denoising...'):
            t_current_tensor = torch.full((B,), t_current, device=device, dtype=torch.long)
            
            pred_x0_logits = denoiser_model(
                img.float(), t_current_tensor.float(), instance_locs,
                prefix_nodes, prefix_lengths, node_prefix_state
            )
            pred_x0_probs = torch.sigmoid(pred_x0_logits)
            
            #20250625 === Apply guidance / sharpening ===#
            # Simple power sharpening; guidance_strength > 1.0 makes the model more confident
            # This penalizes low-probability edges and boosts high-probability ones, making decisions sharper
            
            guidance_strength = 3.8 # Tunable hyperparameter (3.8)
            pred_x0_probs_guided = torch.pow(pred_x0_probs, guidance_strength)
            # Renormalize to keep valid probabilities (not strictly required for Bernoulli sampling, but good practice)
            pred_x0_probs_guided = pred_x0_probs_guided / (pred_x0_probs_guided + torch.pow(1 - pred_x0_probs, guidance_strength))
            #20250625 === Apply guidance / sharpening ===#

            prefix_adj_mask = torch.zeros_like(pred_x0_probs, dtype=torch.bool)
            for b in range(B):
                k = prefix_lengths[b].item()
                if k > 1:
                    p_nodes = prefix_nodes[b, :k]
                    prefix_adj_mask[b, p_nodes[:-1], p_nodes[1:]] = True
                    prefix_adj_mask[b, p_nodes[1:], p_nodes[:-1]] = True
            
            #pred_x0_probs_clamped = torch.where(prefix_adj_mask, 1.0, pred_x0_probs)
            pred_x0_probs_clamped = torch.where(prefix_adj_mask, 1.0, pred_x0_probs_guided)

            if t_prev == 0:
                img = torch.bernoulli(pred_x0_probs_clamped)
            else:
                probs_xt_minus_1_is_1 = self._get_posterior_probs_x_t_minus_1(
                    img, pred_x0_probs_clamped, t_current, t_prev
                )
                img = torch.bernoulli(probs_xt_minus_1_is_1)
        
        final_adj = (img + img.transpose(1, 2)).clamp(0, 1)
        
        return final_adj, torch.sigmoid(pred_x0_logits)


    def _get_posterior_probs_x_t_minus_1(self, x_t, pred_x0_probs, t_current, t_prev):
        """
        Calculate p(x_{t-1} | x_t, predicted_x_0_probs).
        x_t: (B, N, N) current noisy state (binary {0,1}).
        pred_x0_probs: (B, N, N) model's prediction of P(x_0=1|x_t).
        t_current: Scalar current timestep (1 to T).
        t_prev: Scalar previous timestep (0 to T-1).
        Returns probabilities for x_{t-1} being 1. (B, N, N)
        Based on DIFUSCO paper Eq. 5 & 6:
        q(x_{t-1}|x_t, x_0) propto (x_t Q_t^T) element_wise_prod (x_0 Q_bar_{t-1})
        Here Q_t is for step t_current. Q_bar_{t-1} is for step t_prev.
        """
        B, N, _ = x_t.shape
        
        # Ensure pred_x0_probs is {0,1} for one-hot if strictly following paper.
        # Or, use probabilities directly in the sum if p_theta(x_0|x_t) is a distribution.
        # DIFUSCO Eq. 6: sum over x_0_tilde of q(x_{t-1}|x_t,x_0_tilde)p_theta(x_0_tilde|x_t)
        # Since our p_theta(x_0_tilde|x_t) is effectively P(x_0_tilde=1|x_t) and P(x_0_tilde=0|x_t)
        # for each edge independently.
        # Let pred_x0_is_0_probs = 1.0 - pred_x0_probs
        # Let pred_x0_is_1_probs = pred_x0_probs

        # x_0_tilde_one_hot for x_0=0: [1,0]; for x_0=1: [0,1]
        # Term for x_0=0: q(x_{t-1}|x_t, x_0=0) * P(model_predicts_x_0=0 | x_t)
        # Term for x_0=1: q(x_{t-1}|x_t, x_0=1) * P(model_predicts_x_0=1 | x_t)
        
        Q_t_current = self.Q_t[t_current-1] # (2,2)
        Q_bar_t_prev = self.Q_bar_t[t_prev]   # (2,2)

        # Flatten x_t and pred_x0_probs
        x_t_flat = x_t.reshape(B, -1).long() # (B, N*N)
        pred_x0_probs_flat = pred_x0_probs.reshape(B, -1) # (B, N*N)

        # Create one-hot versions for calculations
        # x_t_one_hot (B, N*N, 2)
        x_t_one_hot = F.one_hot(x_t_flat, num_classes=2).float()
        
        # term1 = x_t Q_t^T (element-wise)
        # term1 will be (B, N*N, 2), for x_{t-1}=0 and x_{t-1}=1
        # Q_t_current.T is (2,2)
        # x_t_one_hot is (B, N*N, 2). Want (x_t_one_hot @ Q_t_current.T)
        # This represents P(x_t | x_{t-1})
        # No, q(x_t|x_{t-1}) = Cat(x_t; p = x_{t-1}_one_hot @ Q_t)
        # The formula uses (x_t @ Q_t^T), which is P(x_{t-1} | x_t) if Q was P(x_t|x_{t-1})
        # The derivation of Eq.5 in DIFUSCO paper:
        # q(x_{t-1}|x_t,x_0) = Cat(x_{t-1}; p = (x_t_one_hot @ Q_t.T) * (x_0_one_hot @ Q_bar_{t-1}))
        # Note: * is element-wise product (Hadamard).
        # Denominator is sum over x_{t-1} states (0 and 1).
        
        # Numerator term for x_{t-1}=0 and x_{t-1}=1, marginalized over pred_x0
        # Numerator(x_{t-1}=0) = P(x_{t-1}=0|x_t,x_0=0)P(x_0=0|x_t) + P(x_{t-1}=0|x_t,x_0=1)P(x_0=1|x_t)
        # Numerator(x_{t-1}=1) = P(x_{t-1}=1|x_t,x_0=0)P(x_0=0|x_t) + P(x_{t-1}=1|x_t,x_0=1)P(x_0=1|x_t)
        
        # Let's compute f(xt, x0, xt-1) = (xt @ Qt.T)[xt-1] * (x0 @ Qbar_{t-1})[xt-1]
        # This means:
        # f_val0 = (xt @ Qt.T)[0] * (x0 @ Qbar_{t-1})[0]  (for xt-1 = 0)
        # f_val1 = (xt @ Qt.T)[1] * (x0 @ Qbar_{t-1})[1]  (for xt-1 = 1)

        # Coeffs from (x_t_one_hot @ Q_t_current.T) -> (B, N*N, 2)
        # These are P(x_{t-1}=j | x_t=i) if Q is forward P(x_t|x_{t-1})
        # Let's call source_dist = x_t_one_hot @ Q_t_current.T # P(x_{t-1} | x_t) under uniform prior for x_{t-1}
        # This is P(X_t | X_{t-1}=k) * P(X_{t-1}=k) / P(X_t)
        # The elements of Q_t are P(x_t=new_val | x_{t-1}=old_val)
        # (x_t_one_hot @ Q_t_current.T) gives for each x_t value, a distribution over x_{t-1}
        # Entry [k,j] of this is sum_i x_t_one_hot[k,i] * Q_t_current.T[i,j]
        # = sum_i P(x_t=i) * P(x_{t-1}=j | x_t=i) -- if x_t_one_hot was probs
        # = Q_t_current.T[x_t_flat[k], j]
        # So, if x_t_flat[k] = 0, this is Q_t_current.T[0,:] = [Q_00, Q_10] (prob for x_{t-1}=0, x_{t-1}=1)
        # if x_t_flat[k] = 1, this is Q_t_current.T[1,:] = [Q_01, Q_11]
        # This is effectively selecting a row from Q_t_current.T based on value of x_t.
        # Q_t_current.T[x_t_flat] will be (B, N*N, 2), representing [P(x_{t-1}=0|x_t), P(x_{t-1}=1|x_t)]
        term_A = Q_t_current.T[x_t_flat] # (B, N*N, 2)

        # Coeffs from (x_0_one_hot @ Q_bar_t_prev)
        # This is P(x_{t-1} | x_0) if Q_bar was P(x_{t-1}|x_0)
        # Q_bar_t_prev[x_0_val, x_{t-1}_val]
        # This should be P(x_{t-1}=j | x_0=i)
        # For x_0 = 0: Q_bar_t_prev[0,:] = [P(x_{t-1}=0|x_0=0), P(x_{t-1}=1|x_0=0)]
        # For x_0 = 1: Q_bar_t_prev[1,:] = [P(x_{t-1}=0|x_0=1), P(x_{t-1}=1|x_0=1)]
        
        # Contribution from pred_x0 being 0
        # term_B_if_x0_is_0 is (2) = [P(x_{t-1}=0|x_0=0), P(x_{t-1}=1|x_0=0)]
        term_B_if_x0_is_0 = Q_bar_t_prev[0, :] # (2)
        # Contribution from pred_x0 being 1
        term_B_if_x0_is_1 = Q_bar_t_prev[1, :] # (2)

        # Prob for x_{t-1} states (0 and 1)
        # prob_xt_minus_1_is_0 = ( (term_A[:,:,0] * term_B_if_x0_is_0[0]) * (1-pred_x0_probs_flat) + \
        #                           (term_A[:,:,0] * term_B_if_x0_is_1[0]) * pred_x0_probs_flat )
        # prob_xt_minus_1_is_1 = ( (term_A[:,:,1] * term_B_if_x0_is_0[1]) * (1-pred_x0_probs_flat) + \
        #                           (term_A[:,:,1] * term_B_if_x0_is_1[1]) * pred_x0_probs_flat )
        
        # Simplified: The term_A is P(x_t | x_{t-1}=k) if Q is P(x_t | x_{t-1}). No, this is term_A_k = Q_t[k, x_t]
        # Let's use the form from reference_difusco/pl_tsp_model.py (categorical_posterior which is not fully written out there)
        # but their Eq. (5) from paper is $Cat(x_{t-1}; p \propto (Q_t^T x_t^T) \odot (\bar{Q}_{t-1} x_0^T) )$
        # This implies element-wise product of two vectors, one for each state of x_{t-1}.
        # x_t^T and x_0^T are one-hot vectors.
        # So, for x_{t-1}=k, the unnormalized prob is (Q_t^T)_{k, x_t_val} * (\bar{Q}_{t-1})_{k, x_0_val}
        # which is Q_t[x_t_val, k] * \bar{Q}_{t-1}[x_0_val, k]
        
        # Unnormalized log_prob for x_{t-1}=0, given x_t and a specific x_0_val
        log_P_xtm1_0_given_xt_x0val0 = torch.log(self.Q_t[t_current-1, x_t_flat, 0] + 1e-12) + \
                                        torch.log(self.Q_bar_t[t_prev, 0, 0] + 1e-12)
        log_P_xtm1_0_given_xt_x0val1 = torch.log(self.Q_t[t_current-1, x_t_flat, 0] + 1e-12) + \
                                        torch.log(self.Q_bar_t[t_prev, 1, 0] + 1e-12)
        
        # Unnormalized log_prob for x_{t-1}=1, given x_t and a specific x_0_val
        log_P_xtm1_1_given_xt_x0val0 = torch.log(self.Q_t[t_current-1, x_t_flat, 1] + 1e-12) + \
                                        torch.log(self.Q_bar_t[t_prev, 0, 1] + 1e-12)
        log_P_xtm1_1_given_xt_x0val1 = torch.log(self.Q_t[t_current-1, x_t_flat, 1] + 1e-12) + \
                                        torch.log(self.Q_bar_t[t_prev, 1, 1] + 1e-12)

        # Combine with pred_x0_probs
        # log P(x_{t-1}=0 | x_t) = log [ exp(log_P_xtm1_0_given_xt_x0val0) * (1-pred_x0_probs_flat) + 
        #                               exp(log_P_xtm1_0_given_xt_x0val1) * pred_x0_probs_flat ]
        # This is logsumexp
        
        log_pred_x0_probs_0 = torch.log(1.0 - pred_x0_probs_flat + 1e-12)
        log_pred_x0_probs_1 = torch.log(pred_x0_probs_flat + 1e-12)

        log_posterior_xtm1_0 = torch.logsumexp(torch.stack([
            log_P_xtm1_0_given_xt_x0val0 + log_pred_x0_probs_0,
            log_P_xtm1_0_given_xt_x0val1 + log_pred_x0_probs_1
        ], dim=0), dim=0)
        
        log_posterior_xtm1_1 = torch.logsumexp(torch.stack([
            log_P_xtm1_1_given_xt_x0val0 + log_pred_x0_probs_0,
            log_P_xtm1_1_given_xt_x0val1 + log_pred_x0_probs_1
        ], dim=0), dim=0)

        # Stack and softmax to get normalized probabilities for x_{t-1}
        log_probs_xt_minus_1 = torch.stack([log_posterior_xtm1_0, log_posterior_xtm1_1], dim=-1) # (B, N*N, 2)
        probs_xt_minus_1 = F.softmax(log_probs_xt_minus_1, dim=-1) # (B, N*N, 2)
        
        # Return probability for x_{t-1} to be 1
        return probs_xt_minus_1[:, :, 1].reshape(B, N, N)

