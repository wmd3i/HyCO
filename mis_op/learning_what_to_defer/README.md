# learning_what_to_defer (LwD)

RL backbone for the MIS experiments. This is the code of
*Learning What to Defer for Maximum Independent Sets* (Ahn et al., ICML 2020,
https://github.com/sungsoo-ahn/learning_what_to_defer), with small changes for HyCO
(e.g., additional dataset loaders in `data/graph_dataset.py` for RB-large graphs and SATLIB `.gpickle` / `.bin` graphs).

Only the inference code needed by HyCO is included. To train LwD policies, use the original repository.
Pretrained policies used in the paper are provided in the checkpoint bundle (`checkpoints/lwd/`).
