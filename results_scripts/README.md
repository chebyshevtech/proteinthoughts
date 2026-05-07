# Protein Thoughts Result Reproduction Code

This directory contains standalone Python scripts for regenerating the tables and figures used in the Protein Thoughts submission. Each script includes the data loading, model setup, evaluation, and plotting steps needed for its corresponding result.

The scripts are organized by paper result. GPU execution is recommended for the ESM-2 embedding, PPIProjectedNet training, Qwen LoRA fine-tuning, and H-ESFM flow-matching sections.

## Contents

| Result | Script | Main outputs |
|---|---|---|
| Table 1: decomposed evidence | `table1_decomposed_evidence.py` | `antibody_antigen_decomposed_evidence.csv`, `enzyme_inhibitor_decomposed_evidence.csv` |
| Antibody-antigen score plots | `figure_antibody_score_distributions.py` | `ant_boxplot_scores.png`, `ant_tensionmap.png` |
| Table 2: Micro-F1 benchmark | `table2_microf1_comparison.py` | printed benchmark summary, `data/four_trials_summary.png` |
| Figure 4: virtual screening | `figure4_virtual_screening.py` | `screen_10_trials.png`, `selected_screen_annotated.png` |
| Table 3: SHS148k Qwen-guided discovery | `table3_discovery_results.py` | `shs148k_10trial_fullpool_summary.csv`, `shs148k_10trial_fullpool_baseline.csv`, `shs148k_10trial_fullpool_qwen.csv` |
| Table 4: H-ESFM rank benchmark | `table4_hesfm_true_distance_rank.py` | `cellB32_true_distance_rank_all_candidates.csv`, `cellB32_true_distance_rank_summary.csv` |
| Figure 5: final distance after flow | `figure5_final_distance_distribution.py` | `test_distance_hist.png`, `test_improvement_hist.png`, `test_rank_summary.png`, `test_flow_scores.csv`, `test_metrics.csv`, `test_rankings.csv` |

## Running

Run a result script from the repository root:

```bash
python results_scripts/table2_microf1_comparison.py
```

Some scripts download benchmark data or model weights on first run. Re-running uses cached files where the implementation supports caching.

## Dependencies

The scripts install or import the dependencies used by each result section. A typical environment should include:

```bash
pip install torch torchvision torchaudio numpy pandas scikit-learn matplotlib seaborn tqdm requests fair-esm transformers accelerate peft bitsandbytes
```

Qwen fine-tuning and ESM-2 embedding are substantially faster on CUDA-capable GPUs.

## Reproducibility Notes

- Random seeds, model definitions, feature construction, metric computation, and output file names are preserved from the result-generating implementation.
- Interactive display-only statements were adapted where necessary so tables are written to CSV and figures are saved from script execution.
- The largest runs, especially Table 2 and Table 3, may take hours depending on hardware and whether embeddings/model weights are already cached.
