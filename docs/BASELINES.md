# Baseline sources and configurations

This is the only baseline configuration file in the release. Third-party source code is not redistributed. Clone the linked official implementation, use the official Train, Dev, and Test partitions, and run the five seeds in `configs/seeds.json`. Dev Bot-F1 is the sole selector and Test is not used for checkpoint, hyperparameter, threshold, or model selection.

## Official sources

| Method | Official source |
|---|---|
| SGBot | https://github.com/LuoUndergradXJTU/TwiBot-22/tree/master/src/SGBot |
| RoBERTa | https://github.com/LuoUndergradXJTU/TwiBot-22/tree/master/src/RoBERTa |
| LMBot-LM | https://doi.org/10.1145/3616855.3635843; obtain the author code release linked by the paper |
| LMBot-GNN | https://doi.org/10.1145/3616855.3635843; obtain the author code release linked by the paper |
| BotRGCN | https://github.com/BunsenFeng/BotRGCN |
| RGT | https://github.com/BunsenFeng/BotHeterogeneity |
| BotMoE | https://github.com/lyh6560new/BotMoE |
| BotDGT | https://github.com/Peien429/BotDGT |
| MPS-Bot | https://doi.org/10.1145/3774904.3792485; obtain the author code release linked by the paper |

For an author release without a public repository URL, record the source archive checksum and acquisition source before reporting a reproduced result. Do not silently substitute an unofficial implementation.

## TwiBot-20

| Method | Configuration used in the paper reproduction |
|---|---|
| SGBot | `n_estimators=100`, `n_jobs=1`, screen-name bigram model fitted on Train only, collection date `2020-12-31` |
| RoBERTa | input 768, hidden 128, dropout 0.3, Adam, learning rate `1e-5`, weight decay `1e-6`, batch 64, maximum 100 epochs |
| LMBot-LM | `LM_pretrain_epochs=4.5`, `alpha=0.5`, `max_iterations=10`, LM batch 32, GNN disabled |
| LMBot-GNN | `LM_pretrain_epochs=4.5`, `alpha=0.5`, `max_iterations=10`, LM batch 32, GNN enabled |
| BotRGCN | description 768, tweets 768, numeric 5, categorical 3, embedding 128, dropout 0.3, AdamW, learning rate `1e-3`, weight decay `5e-3`, 2 relations, maximum 100 epochs |
| RGT | embedding/output 128, dropout 0.5, transformer heads 2, semantic heads 2, AdamW, learning rate `1e-3`, weight decay `3e-5`, cosine schedule `T_max=16`, 2 relations, maximum 50 epochs |
| BotMoE | `AllInOne1_rgcn_rgt_gcn`, alignment 128, graph/text/metadata experts 3/2/3, top-k 1, dropout 0.3, batch 1024, neighbors `[256,256]`, Adam, learning rate `1e-5`, weight decay `1e-6`, maximum 400 epochs |
| BotDGT | yearly interval, full window, batch 64, train neighbors `[2560,2560]`, full-neighbor evaluation, hidden 128, structural/temporal heads 4/4, dropout 0/0.5/0.3, AdamW, learning rates `1e-4` and `1e-5`, weight decay `1e-2`, maximum 20 epochs |
| MPS-Bot | documented TwiBot-20 compatibility port, learning rate `1e-3`, maximum 200 epochs; retain the author architecture and loss |

## TwiBot-22

| Method | Configuration used in the paper reproduction |
|---|---|
| SGBot | `n_estimators=100`, `n_jobs=1`, screen-name bigram model fitted on Train only |
| RoBERTa | input 768, hidden 128, dropout 0.5, Adam, learning rate `1e-3`, weight decay `3e-5`, batch 64, maximum 20 epochs |
| LMBot-LM | Not reported because the TwiBot-22 port did not complete within the allocated runtime |
| LMBot-GNN | Not reported because the TwiBot-22 port did not complete within the allocated runtime |
| BotRGCN | embedding 32, dropout 0.1, AdamW, learning rate `1e-2`, weight decay `5e-2`, train/eval batch 128/256, train/eval neighbors `[256,256]`, 2 relations, maximum 20 epochs |
| RGT | embedding/output 128, dropout 0.3, heads 2/2, AdamW, learning rate `5e-4`, weight decay `1e-5`, train/eval batch 32/64, train/eval neighbors `[256,256,256,256]`, maximum 12 epochs |
| BotMoE | `AllInOne1_rgcn_rgt_gcn`, alignment 128, experts 3/2/3, top-k 1, dropout 0.3, batch 64, neighbors `[256,256]`, Adam, learning rate `1e-5`, weight decay `1e-6`, maximum 12 epochs |
| BotDGT | yearly interval, full window, batch 16, train/eval neighbors `[256,256]`, hidden 128, heads 4/4, dropout 0/0.5/0.3, AdamW, learning rates `1e-4` and `1e-5`, weight decay `1e-2`, maximum 4 epochs |
| MPS-Bot | author TwiBot-22 setting, learning rate `1e-3`, dropout 0.5, maximum 200 epochs |

## Allowed compatibility changes

Compatibility changes are limited to official partition loading, deterministic seed injection, device and output paths, memory-safe data loading, per-epoch observation logging, and Dev-only checkpoint selection. Do not change graph direction, relation types, message-passing depth, neighborhood sizes, modal inputs, loss terms, expert count, hidden size, or model capacity. Unsupported dataset-method pairs must be reported as unavailable rather than assigned synthetic values.
