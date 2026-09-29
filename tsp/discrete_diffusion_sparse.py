# Discrete (Bernoulli) diffusion with prefix conditioning on sparse (k-NN) graphs.
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
    def __init__(self, num_nodes, num_timesteps, schedule_type='cosine', device='cpu', sparse_factor=-1):
        self.num_nodes = num_nodes
        self.num_timesteps = num_timesteps
        self.device = device
        self.is_sparse = sparse_factor > 0

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

    def q_sample(self, x_0_data, t_steps):
        Q_bar_t_selected = self.Q_bar_t[t_steps]
        prob_xt_eq_1_given_x0_eq_1 = Q_bar_t_selected[:, 1, 1].unsqueeze(1)
        prob_xt_eq_1_given_x0_eq_0 = Q_bar_t_selected[:, 0, 1].unsqueeze(1)
        x_0_flat = x_0_data.float().flatten(1)
        prob_xt_is_one = x_0_flat * prob_xt_eq_1_given_x0_eq_1 + \
                         (1 - x_0_flat) * prob_xt_eq_1_given_x0_eq_0
        prob_xt_is_one = prob_xt_is_one.reshape_as(x_0_data)
        x_t = torch.bernoulli(prob_xt_is_one).float()
        return x_t, prob_xt_is_one

    def training_loss(self, denoiser_model, batch_data, t_steps):
        device = t_steps.device
        
        if self.is_sparse:
            x_0 = batch_data["target_edge_attrs"]
            t_steps_expanded = t_steps[batch_data["node_to_graph_batch"][batch_data["edge_index"][0]]]
            x_t, _ = self.q_sample(x_0, t_steps_expanded)
            noisy_data_in = x_t.float() # No {-1, 1} mapping needed if model expects {0, 1}
        else:
            x_0 = batch_data["target_adj_matrix"]
            x_t, _ = self.q_sample(x_0, t_steps)
            noisy_data_in = x_t.float()

        # <<< MODIFIED: simplified model call, pass batch_data directly >>>
        # The model now receives the full batch_data dict and fetches everything it needs from it,
        # including the newly added dist_feature.
        predicted_x_0_logits = denoiser_model(
            noisy_data=noisy_data_in,
            t_scalar=t_steps.float(),
            batch_data=batch_data
        )
        # <<< END MODIFICATION >>>

        if self.is_sparse:
            node_prefix_state_flat = batch_data["node_prefix_state"].squeeze(-1)
            edge_index = batch_data["edge_index"]
            src_in_prefix = node_prefix_state_flat[edge_index[0]] == 1.0
            dst_in_prefix = node_prefix_state_flat[edge_index[1]] == 1.0
            internal_prefix_edge_mask = src_in_prefix & dst_in_prefix
            loss_mask = ~internal_prefix_edge_mask
            target = x_0.squeeze(-1)[loss_mask]
            prediction = predicted_x_0_logits[loss_mask]
        else:
            B, N, _ = batch_data["instance_locs"].shape
            prefix_mask = (batch_data["node_prefix_state"] == 1.0).squeeze(-1)
            prefix_row_mask = prefix_mask.unsqueeze(2).expand(B, N, N)
            prefix_col_mask = prefix_mask.unsqueeze(1).expand(B, N, N)
            internal_prefix_mask = prefix_row_mask & prefix_col_mask
            identity_mask = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0).expand(B, N, N)
            loss_mask = ~internal_prefix_mask & ~identity_mask
            target = x_0[loss_mask]
            prediction = predicted_x_0_logits[loss_mask]

        return F.binary_cross_entropy_with_logits(prediction, target.float())

        
    # --- get_selection_schedule (no changes needed) ---
    def get_selection_schedule(self, num_inference_steps, schedule_type):
        timesteps = []
        if schedule_type == 'linear':
            t_points = np.linspace(self.num_timesteps, 0, num_inference_steps + 1).astype(int)
        elif schedule_type == 'cosine':
            s_points = np.linspace(1.0, 0.0, num_inference_steps + 1)
            t_points = (0.5 * self.num_timesteps * (1 + np.cos(s_points * np.pi))).astype(int)
        elif schedule_type == 'polynomial':
            power = 3
            t_points = np.linspace(self.num_timesteps**(1/power), 0, num_inference_steps + 1)**power
            t_points = t_points.astype(int)
        else:
            raise ValueError(f"Unknown inference schedule: {schedule_type}")
        
        unique_t_points = np.unique(t_points)
        t_points = np.sort(unique_t_points)[::-1]
        
        for i in range(len(t_points) - 1):
            t_current = max(1, t_points[i])
            t_prev = t_points[i+1]
            timesteps.append((t_current, t_prev))
        return timesteps
        
    # <<< NEW: Helper function to build a map for fast edge lookups
    def _get_edge_map(self, edge_index):
        """Creates a map from a canonical edge tuple (min_node, max_node) to its index in the edge tensor."""
        return {tuple(sorted(edge.tolist())): i for i, edge in enumerate(edge_index.T)}
        
    # <<< NEW: Helper function to reconstruct batched dense matrices from sparse edge data
    # def _reconstruct_adj_matrices(self, final_edge_states, batch_data):
    #     B = batch_data["prefix_lengths"].size(0)
    #     N = self.num_nodes
    #     device = final_edge_states.device
        
    #     final_adj = torch.zeros(B, N, N, device=device, dtype=torch.float32)
        
    #     node_counts = torch.bincount(batch_data["node_to_graph_batch"])
    #     edge_counts = torch.bincount(batch_data["node_to_graph_batch"][batch_data["edge_index"][0]])
        
    #     node_offsets = torch.cat([torch.tensor([0], device=device), torch.cumsum(node_counts, dim=0)[:-1]])
    #     edge_offsets = torch.cat([torch.tensor([0], device=device), torch.cumsum(edge_counts, dim=0)[:-1]])

    #     for b in range(B):
    #         # Extract edges and their states for the current graph
    #         edge_offset_start = edge_offsets[b]
    #         edge_offset_end = edge_offsets[b] + edge_counts[b]
            
    #         graph_edge_indices = batch_data["edge_index"][:, edge_offset_start:edge_offset_end]
    #         graph_edge_states = final_edge_states[edge_offset_start:edge_offset_end]
            
    #         # Convert global node indices back to local
    #         local_edge_indices = graph_edge_indices - node_offsets[b]
            
    #         # Populate the dense adjacency matrix
    #         src, dst = local_edge_indices[0], local_edge_indices[1]
    #         final_adj[b, src, dst] = graph_edge_states
    #         final_adj[b, dst, src] = graph_edge_states # Ensure symmetry
            
    #     return final_adj.clamp(0, 1)

# RLDF4CO/discrete_diffusion_sparse.py

    def _reconstruct_adj_matrices(self, final_edge_states, batch_data):
        B = batch_data["prefix_lengths"].size(0)
        N = self.num_nodes
        device = final_edge_states.device
        
        final_adj = torch.zeros(B, N, N, device=device, dtype=torch.float32)
        
        # --- START OF FIX ---
        # Use minlength=B so that the bincount output has length at least B
        node_counts = torch.bincount(batch_data["node_to_graph_batch"], minlength=B)
        edge_counts = torch.bincount(batch_data["node_to_graph_batch"][batch_data["edge_index"][0]], minlength=B)
        
        # bincount only guarantees a minimum length; if graph indices in the batch are not contiguous and the max index is < B-1,
        # the length may still be smaller than B. Add an explicit check for safety.
        if len(node_counts) < B:
            node_counts = F.pad(node_counts, (0, B - len(node_counts)))
        if len(edge_counts) < B:
            edge_counts = F.pad(edge_counts, (0, B - len(edge_counts)))
        # --- END OF FIX ---

        node_offsets = torch.cat([torch.tensor([0], device=device), torch.cumsum(node_counts, dim=0)[:-1]])
        edge_offsets = torch.cat([torch.tensor([0], device=device), torch.cumsum(edge_counts, dim=0)[:-1]])

        for b in range(B):
            # Extract the edges of the current graph and their states
            edge_offset_start = edge_offsets[b]
            edge_offset_end = edge_offsets[b] + edge_counts[b]
            
            graph_edge_indices = batch_data["edge_index"][:, edge_offset_start:edge_offset_end]
            graph_edge_states = final_edge_states[edge_offset_start:edge_offset_end]
            
            # Convert global node indices back to local indices
            local_edge_indices = graph_edge_indices - node_offsets[b]
            
            # Fill the dense adjacency matrix
            src, dst = local_edge_indices[0], local_edge_indices[1]
            final_adj[b, src, dst] = graph_edge_states
            final_adj[b, dst, src] = graph_edge_states # Enforce symmetry
            
        return final_adj.clamp(0, 1)

    
    @torch.no_grad()
    def p_sample_loop_ddim(self, denoiser_model, batch_data, num_inference_steps=50, schedule='cosine', guidance_strength=1.7, visualize_instance_idx=-1):
        # <<< MODIFIED: Fully adapted for sparse and dense inference
        denoiser_model.eval()
        device = self.device
        B = batch_data["prefix_lengths"].size(0)
        timesteps_pairs = self.get_selection_schedule(num_inference_steps, schedule)

        # 1. Initialize random noise 'img' based on mode
        if self.is_sparse:
            num_edges = batch_data["edge_index"].shape[1]
            img = torch.randint(0, 2, (num_edges,), device=device).float()
            # Create edge map once for efficient prefix clamping
            edge_map = self._get_edge_map(batch_data["edge_index"])
        else: # Dense
            N = self.num_nodes
            img = torch.randint(0, 2, (B, N, N), device=device).float()
            img = (img + img.transpose(1, 2)).clamp(0, 1)

        final_logits = None
        viz_frames = []
        num_steps = len(timesteps_pairs)

        for i, (t_current, t_prev) in enumerate(tqdm(timesteps_pairs, desc='DDIM Denoising...', leave=False)):
            t_current_tensor = torch.full((B,), t_current, device=device, dtype=torch.long)
            
            # 2. Get model prediction
            img_transformed = img.float() * 2.0 - 1.0


            
            # pred_x0_logits = denoiser_model(
            #     noisy_data=img_transformed, t_scalar=t_current_tensor.float(), **batch_data
            # )
            
            # <<< MODIFIED: This is the line to fix >>>
            # The original call passed keyword arguments that the new model forward() doesn't accept.
            # The new call correctly passes the whole batch_data dictionary.
            pred_x0_logits = denoiser_model(
                noisy_data=img_transformed, 
                t_scalar=t_current_tensor.float(), 
                batch_data=batch_data
            )
            # <<< END MODIFICATION >>>

            pred_x0_probs = torch.sigmoid(pred_x0_logits)
            
            if i == num_steps - 1:
                final_logits = pred_x0_logits.clone()

            # 3. Apply guidance/sharpening
            pred_x0_probs_guided = torch.pow(pred_x0_probs, guidance_strength)
            pred_x0_probs_guided = pred_x0_probs_guided / (pred_x0_probs_guided + torch.pow(1 - pred_x0_probs, guidance_strength) + 1e-9)
            
            # 4. Clamp prefix edges to 1.0
            prefix_mask = torch.zeros_like(pred_x0_probs, dtype=torch.bool)
            if self.is_sparse:
                node_counts = torch.bincount(batch_data["node_to_graph_batch"])
                node_cumsum = torch.cat([torch.tensor([0], device=device), torch.cumsum(node_counts, dim=0)[:-1]])
            
            for b in range(B):
                k = batch_data["prefix_lengths"][b].item()
                if k > 1:
                    p_nodes = batch_data["prefix_nodes"][b, :k]
                    if self.is_sparse:
                        global_p_nodes = p_nodes + node_cumsum[b]
                        for j in range(k - 1):
                            u, v = global_p_nodes[j].item(), global_p_nodes[j+1].item()
                            edge_key = tuple(sorted((u, v)))
                            if edge_key in edge_map:
                                prefix_mask[edge_map[edge_key]] = True
                    else: # Dense
                        prefix_mask[b, p_nodes[:-1], p_nodes[1:]] = True
                        prefix_mask[b, p_nodes[1:], p_nodes[:-1]] = True
            
            pred_x0_probs_clamped = torch.where(prefix_mask, 1.0, pred_x0_probs_guided)

            if t_prev == 0:
                img = torch.bernoulli(pred_x0_probs_clamped)
                continue
            
            # 5. DDIM update step
            Q_bar_t_prev = self.Q_bar_t[t_prev]
            prob_xtm1_eq_1_given_x0_eq_1 = Q_bar_t_prev[1, 1]
            prob_xtm1_eq_1_given_x0_eq_0 = Q_bar_t_prev[0, 1]
            
            probs_xt_minus_1_is_1 = (
                prob_xtm1_eq_1_given_x0_eq_1 * pred_x0_probs_clamped +
                prob_xtm1_eq_1_given_x0_eq_0 * (1 - pred_x0_probs_clamped)
            )
            img = torch.bernoulli(probs_xt_minus_1_is_1)
        
        # 6. Reconstruct final adjacency matrix
        if self.is_sparse:
            final_adj = self._reconstruct_adj_matrices(img, batch_data)
            final_probs_adj = self._reconstruct_adj_matrices(torch.sigmoid(final_logits), batch_data)
            
            # Also return the final sparse logits and the corresponding edge_index for efficient sparse decoding
            sparse_final_logits = final_logits 
            sparse_edge_index = batch_data["edge_index"]

        else: # Dense
            final_adj = (img + img.transpose(1, 2)).clamp(0, 1)
            final_probs_adj = torch.sigmoid(final_logits)
            # No sparse outputs in dense mode
            sparse_final_logits = None
            sparse_edge_index = None

        # The return values additionally include sparse_edge_index and sparse_final_logits
        return final_adj, final_probs_adj, viz_frames, sparse_edge_index, sparse_final_logits
        

        # return final_adj, final_probs_adj, viz_frames

    @torch.no_grad()
    def p_sample_loop(self, denoiser_model, batch_data, num_inference_steps=50, schedule='cosine', guidance_strength=3.8):
        # <<< MODIFIED: Fully adapted for sparse and dense inference
        denoiser_model.eval()
        device = self.device
        B = batch_data["prefix_lengths"].size(0)
        timesteps_pairs = self.get_selection_schedule(num_inference_steps, schedule)

        # 1. Initialize random noise 'img' based on mode
        if self.is_sparse:
            num_edges = batch_data["edge_index"].shape[1]
            img = torch.randint(0, 2, (num_edges,), device=device).float()
            edge_map = self._get_edge_map(batch_data["edge_index"])
        else: # Dense
            N = self.num_nodes
            img = torch.randint(0, 2, (B, N, N), device=device).float()
            img = (img + img.transpose(1, 2)).clamp(0, 1)

        final_logits = None

        for t_current, t_prev in tqdm(timesteps_pairs, desc='DDPM Denoising...', leave=False):
            t_current_tensor = torch.full((B,), t_current, device=device, dtype=torch.long)
            
            # 2. Get model prediction
            # For DDPM, the input is {0,1} not {-1,1}
            pred_x0_logits = denoiser_model(
                noisy_data=img.float(), t_scalar=t_current_tensor.float(), **batch_data
            )
            pred_x0_probs = torch.sigmoid(pred_x0_logits)
            final_logits = pred_x0_logits
            
            # 3. Apply guidance
            pred_x0_probs_guided = torch.pow(pred_x0_probs, guidance_strength)
            pred_x0_probs_guided = pred_x0_probs_guided / (pred_x0_probs_guided + torch.pow(1 - pred_x0_probs, guidance_strength) + 1e-9)
            
            # 4. Clamp prefix edges
            prefix_mask = torch.zeros_like(pred_x0_probs, dtype=torch.bool)
            if self.is_sparse:
                node_counts = torch.bincount(batch_data["node_to_graph_batch"])
                node_cumsum = torch.cat([torch.tensor([0], device=device), torch.cumsum(node_counts, dim=0)[:-1]])

            for b in range(B):
                k = batch_data["prefix_lengths"][b].item()
                if k > 1:
                    p_nodes = batch_data["prefix_nodes"][b, :k]
                    if self.is_sparse:
                        global_p_nodes = p_nodes + node_cumsum[b]
                        for j in range(k - 1):
                            u, v = global_p_nodes[j].item(), global_p_nodes[j+1].item()
                            edge_key = tuple(sorted((u, v)))
                            if edge_key in edge_map:
                                prefix_mask[edge_map[edge_key]] = True
                    else: # Dense
                        prefix_mask[b, p_nodes[:-1], p_nodes[1:]] = True
                        prefix_mask[b, p_nodes[1:], p_nodes[:-1]] = True
            
            pred_x0_probs_clamped = torch.where(prefix_mask, 1.0, pred_x0_probs_guided)

            # 5. DDPM update step
            if t_prev == 0:
                img = torch.bernoulli(pred_x0_probs_clamped)
            else:
                probs_xt_minus_1_is_1 = self._get_posterior_probs_x_t_minus_1(
                    img, pred_x0_probs_clamped, t_current, t_prev
                )
                img = torch.bernoulli(probs_xt_minus_1_is_1)
        
        # 6. Reconstruct final adjacency matrix
        if self.is_sparse:
            final_adj = self._reconstruct_adj_matrices(img, batch_data)
            final_probs_adj = self._reconstruct_adj_matrices(torch.sigmoid(final_logits), batch_data)
        else: # Dense
            final_adj = (img + img.transpose(1, 2)).clamp(0, 1)
            final_probs_adj = torch.sigmoid(final_logits)

        return final_adj, final_probs_adj

    def _get_posterior_probs_x_t_minus_1(self, x_t, pred_x0_probs, t_current, t_prev):
        # <<< MODIFIED: Works with both dense and sparse (flattened) inputs
        # x_t and pred_x0_probs can be (B, N*N) or (Total_E,)
        x_t_flat = x_t.long().flatten()
        pred_x0_probs_flat = pred_x0_probs.flatten()

        log_pred_x0_probs_0 = torch.log(1.0 - pred_x0_probs_flat + 1e-12)
        log_pred_x0_probs_1 = torch.log(pred_x0_probs_flat + 1e-12)

        # Unnormalized log_prob for x_{t-1}=0
        log_P_xtm1_0_given_xt_x0val0 = torch.log(self.Q_t[t_current-1, x_t_flat, 0] + 1e-12) + torch.log(self.Q_bar_t[t_prev, 0, 0] + 1e-12)
        log_P_xtm1_0_given_xt_x0val1 = torch.log(self.Q_t[t_current-1, x_t_flat, 0] + 1e-12) + torch.log(self.Q_bar_t[t_prev, 1, 0] + 1e-12)
        log_posterior_xtm1_0 = torch.logsumexp(torch.stack([
            log_P_xtm1_0_given_xt_x0val0 + log_pred_x0_probs_0,
            log_P_xtm1_0_given_xt_x0val1 + log_pred_x0_probs_1
        ], dim=0), dim=0)

        # Unnormalized log_prob for x_{t-1}=1
        log_P_xtm1_1_given_xt_x0val0 = torch.log(self.Q_t[t_current-1, x_t_flat, 1] + 1e-12) + torch.log(self.Q_bar_t[t_prev, 0, 1] + 1e-12)
        log_P_xtm1_1_given_xt_x0val1 = torch.log(self.Q_t[t_current-1, x_t_flat, 1] + 1e-12) + torch.log(self.Q_bar_t[t_prev, 1, 1] + 1e-12)
        log_posterior_xtm1_1 = torch.logsumexp(torch.stack([
            log_P_xtm1_1_given_xt_x0val0 + log_pred_x0_probs_0,
            log_P_xtm1_1_given_xt_x0val1 + log_pred_x0_probs_1
        ], dim=0), dim=0)

        # Stack, softmax, and reshape
        log_probs_xt_minus_1 = torch.stack([log_posterior_xtm1_0, log_posterior_xtm1_1], dim=-1)
        probs_xt_minus_1 = F.softmax(log_probs_xt_minus_1, dim=-1)
        
        # Return probability of being 1, in the original shape of x_t
        return probs_xt_minus_1[:, 1].reshape_as(x_t)