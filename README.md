# HyCO: A Hybrid Neural Solver for Combinatorial Optimization

Official implementation of **HyCO** (NeurIPS 2026).

Yuheng Li, Di Yang, Haipeng Chen, Yanhai Xiong — College of William & Mary

HyCO is a plug-and-play hybrid inference algorithm for neural combinatorial optimization.
A sequential RL solver constructs a solution prefix; a lightweight adaptive trigger watches two
trajectory-level signals and, once either fires, hands the remaining decisions over to a
prefix-conditioned diffusion model (DM) exactly once:

- **Policy entropy** `H(π_θ(·|s_k))`: the RL policy is uncertain about the next action.
- **RL–DM disagreement** `D_KL(π_θ ‖ q_DM)`: the RL policy prefers actions the DM considers globally inconsistent. `q_DM` is obtained from a single-step denoising energy probe over the RL top-M candidates.

At the trigger step HyCO runs **dual-track generation**: (i) the DM completes the prefix for the
top RL candidates (cumulative probability threshold), and (ii) the RL policy continues from the
DM-corrected next action. The best solution of both tracks is returned.

| Problem | RL backbone | Conditional DM |
| --- | --- | --- |
| TSP-50/100/500/1000 | AM, POMO (rl4co), LEHD | Prefix-DIFUSCO (this repo) |
| MIS (RB-large, ER-700-800, SATLIB) | LwD | prefix-conditioned CO-Expander |
| OP-50/100/200 | AM (rl4co) | prefix-conditioned CO-Expander |

## Repository structure

```
HyCO/
├── tsp/                         # TSP experiments (run all TSP commands from tsp/)
│   ├── model_components*.py     # Prefix-DIFUSCO GNN / prefix encoder (dense and sparse)
│   ├── diffusion_model*.py      # Prefix-conditioned denoising network
│   ├── discrete_diffusion*.py   # Bernoulli diffusion, DDIM sampler, masked loss
│   ├── data_loader*.py          # TSP datasets with sampled prefixes
│   ├── tsp_utils.py             # tour cost, batched 2-opt, decoding, plotting
│   ├── gen_data.py              # instance generation + Concorde / LKH / OR-Tools labels
│   ├── train_prefix_difusco_tsp{50,100,500,1000}.py
│   ├── eval_prefix_difusco_tsp{50,100,500,1000}.py   # standalone Prefix-DIFUSCO baseline
│   ├── hyco_tsp{50,100,500,1000}.py                  # HyCO with POMO
│   ├── hyco_tsp_am.py                                # HyCO with AM
│   ├── ablation_trigger_tsp100.py                    # trigger ablations (App. C)
│   ├── fixed_trigger_tsp{100,500}.py                 # fixed-step trigger sweeps (App. B.4)
│   ├── oracle_trigger_tsp{100,500}.py                # oracle trigger validation (Sec. 6.3)
│   ├── trigger_stats_tsp{500,1000}.py                # trigger-step statistics (App. B.5)
│   ├── configs/                                      # train_*.yaml and hyco_*.yaml
│   └── lehd/                                         # HyCO with LEHD (run from tsp/lehd/)
├── mis_op/                      # MIS and OP experiments (built on CO-Expander)
│   ├── setup_coexpander.sh      # clones COExpander and applies coexpander_hyco.patch
│   ├── coexpander_hyco.patch    # prefix conditioning + OP support for CO-Expander
│   ├── learning_what_to_defer/  # LwD (RL backbone for MIS), inference code
│   ├── mis_er700/ mis_rblarge/ mis_satlib/   # one folder per MIS benchmark (run from inside)
│   ├── op50/ op100/ op200/                   # one folder per OP size (run from inside)
│   ├── mis_data/download_mis_data.py         # ML4CO-Bench-101 MIS data
│   └── op_data/                              # OP data generation and Gurobi reference solver
├── requirements.txt
└── environment.yml              # full conda environment used for the paper
```

## Installation

Tested with Python 3.10, PyTorch 2.4 and CUDA 11.8 on NVIDIA A40 GPUs.

```bash
conda create -n hyco python=3.10 -y
conda activate hyco
# install torch / dgl / PyG wheels for your CUDA version (see comments in requirements.txt), then
pip install -r requirements.txt
```

Alternatively, recreate the exact environment with `conda env create -f environment.yml`.

For MIS and OP, fetch and patch CO-Expander once:

```bash
cd mis_op
bash setup_coexpander.sh      # creates mis_op/co_expander/
```

Optional dependencies: `pyconcorde`, `lkh`, `ortools`, `tsplib95` (TSP data labeling) and `gurobipy` (OP reference solutions).

## Checkpoints and test data

Pretrained checkpoints and the test sets are distributed as release assets:

| File | Content |
| --- | --- |
| `HyCO_checkpoints.tar.gz` | Prefix-DIFUSCO (TSP-50/100/500/1000), AM/POMO (rl4co) for TSP, LEHD, LwD (MIS), prefix-conditioned CO-Expander (MIS/OP), AM for OP |
| `HyCO_test_data.tar.gz` | TSP test sets (Concorde labels), MIS test graphs (LwD format + ML4CO-Bench labels), OP test sets |

Both archives mirror the repository layout; extract them in the repository root:

```bash
tar -xzf HyCO_checkpoints.tar.gz
tar -xzf HyCO_test_data.tar.gz
```

This creates

```
tsp/checkpoints/prefix_difusco/tsp{50,100,500,1000}/   tsp/checkpoints/rl/   tsp/lehd/checkpoints/
tsp/data/
mis_op/checkpoints/{lwd,coexpander,rl_op}/
mis_op/data/{lwd,ml4co_mis,op}/
```

All config files already point to these locations.

## Reproducing the results

### TSP (Tables 1, 2, 5, 6)

```bash
cd tsp
# HyCO with POMO
python hyco_tsp50.py   --config configs/hyco_tsp50_pomo.yaml
python hyco_tsp100.py  --config configs/hyco_tsp100_pomo.yaml
python hyco_tsp500.py  --config configs/hyco_tsp500_pomo.yaml
python hyco_tsp1000.py --config configs/hyco_tsp1000_pomo.yaml
# HyCO with AM
python hyco_tsp_am.py  --config configs/hyco_tsp50_am.yaml
python hyco_tsp_am.py  --config configs/hyco_tsp100_am.yaml
# HyCO with LEHD
cd lehd
python hyco_lehd_tsp500.py  --config configs/hyco_lehd_tsp500.yaml
python hyco_lehd_tsp1000.py --config configs/hyco_lehd_tsp1000.yaml
```

Set `apply_two_opt: True` in the config for the `+2OPT` rows.
Standalone Prefix-DIFUSCO baselines: `python eval_prefix_difusco_tsp{50,100,500,1000}.py`
(greedy by default; set `num_parallel_samples: 16` in the eval config inside the script for the `(S)` rows).

Trigger analyses:

```bash
python ablation_trigger_tsp100.py --config configs/hyco_tsp100_ablation.yaml   # Tables 10-13
python fixed_trigger_tsp100.py    --config configs/hyco_tsp100_pomo.yaml       # Figure 2
python fixed_trigger_tsp500.py    --config configs/hyco_tsp500_pomo.yaml
python oracle_trigger_tsp100.py   --config configs/hyco_tsp100_pomo.yaml       # Table 4
python oracle_trigger_tsp500.py   --config configs/hyco_tsp500_pomo.yaml
python trigger_stats_tsp500.py    --config configs/hyco_tsp500_pomo.yaml       # Table 9
python trigger_stats_tsp1000.py   --config configs/hyco_tsp1000_pomo.yaml
```

Trigger hyperparameters live in the `solver` section of each config:
`entropy_threshold` (H_th), `kl_div_threshold` (D_th), `probe_rl_top_m` (M),
`dm_probe_timestep`, `dm_prior_temp` (τ_temp) and `dynamic_n_cumulative_threshold` (P_th).

### MIS (Tables 3, 7)

```bash
cd mis_op/mis_rblarge      # or mis_er700 / mis_satlib
python hyco_mis.py --config hyco_config.yaml               # HyCO (LwD + CO-Expander)
python eval_prefix_coexpander.py                           # prefix-CO-Expander baseline
python fixed_trigger_mis.py --config hyco_config.yaml --mode fixed_exp
python oracle_trigger_mis.py --config hyco_config.yaml     # mis_er700 and mis_satlib
```

### OP (Table 8)

```bash
cd mis_op/op200            # or op50 / op100
python hyco_op.py --config hyco_config.yaml                # HyCO (AM + CO-Expander)
python eval_prefix_coexpander.py --k_samples 8             # prefix-CO-Expander baseline (s=8)
python fixed_trigger_op.py --config hyco_config.yaml --mode fixed_exp   # op200 only
```

Gurobi reference (Gurobi-300 / Gurobi-30): `cd mis_op/op_data && python gurobi_op.py --config configs/op200.yaml`.

## Training

### Prefix-DIFUSCO (TSP)

1. Generate instances and Concorde labels (`tsp/gen_data.py`), e.g.

   ```bash
   cd tsp
   python gen_data.py --mode generate --num_nodes 100 --num_samples 1000000 --seed 1997
   python gen_data.py --mode solve --solver concorde --num_nodes 100 --num_samples 1000000 --seed 1997
   ```

   The labeled instances are written to `tsp/data/`; point `train_path` / `valid_path` in `configs/train_*.yaml` to them.
2. Train with the curriculum described in Appendix D.2 (2 GPUs):

   ```bash
   torchrun --nproc_per_node=2 train_prefix_difusco_tsp100.py
   torchrun --nproc_per_node=2 train_prefix_difusco_tsp500.py    # initialized from the TSP-100 model
   torchrun --nproc_per_node=2 train_prefix_difusco_tsp1000.py   # sparse model, initialized from TSP-100
   ```

   Checkpoints are written to `tsp/runs/`.

The RL backbones (AM, POMO) were trained with [RL4CO](https://github.com/ai4co/rl4co) using the hyperparameters in Tables 15-16;
LEHD uses the official checkpoint from [LEHD](https://github.com/CIAM-Group/NCO_code).

### Prefix-conditioned CO-Expander (MIS / OP)

```bash
cd mis_op/mis_data && python download_mis_data.py      # ML4CO-Bench-101 MIS data -> mis_op/data/ml4co_mis
cd ../mis_rblarge && python train_prefix_coexpander.py train_config.yaml
```

OP training data are generated with a pretrained AM policy:

```bash
cd mis_op/op_data
python generate_op_dataset_rl.py --config configs/op100.yaml --num_train 280000 --num_valid 500 --num_test 500 --output_dir ../data/op/op100
cd ../op100 && python train_prefix_coexpander.py train_config.yaml
```

## Notes

- Scripts use relative paths; run each script from the directory it lives in.
- The HyCO scripts report the optimality gap against the reference solutions stored in the test files
  (Concorde for TSP, KaMIS for MIS, Gurobi for OP).
- Results can differ slightly across GPUs and library versions.

## Citation

```bibtex
@inproceedings{li2026hyco,
  title     = {HyCO: A Hybrid Neural Solver for Combinatorial Optimization},
  author    = {Li, Yuheng and Yang, Di and Chen, Haipeng and Xiong, Yanhai},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## Acknowledgements

This code builds on [DIFUSCO](https://github.com/Edward-Sun/DIFUSCO), [RL4CO](https://github.com/ai4co/rl4co),
[LEHD](https://github.com/CIAM-Group/NCO_code), [CO-Expander](https://github.com/Thinklab-SJTU/COExpander),
[Learning What to Defer](https://github.com/sungsoo-ahn/learning_what_to_defer) and [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit).
We thank the authors for releasing their code. `tsp/lehd/TSPModel.py` and `tsp/lehd/TSPEnv.py` are adapted from LEHD (MIT license, see `tsp/lehd/LICENSE_LEHD`).

## License

MIT (see `LICENSE`).
