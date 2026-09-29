# Models

Model weights are not stored in this repository. The three fine-tuned models are published on the Hugging Face Hub;
only their training curves are committed here.

| Directory | Produced by | Committed here | On the Hub |
|---|---|---|---|
| `Qwen2.5-1.5B-Instruct/` | downloaded from `Qwen/Qwen2.5-1.5B-Instruct` | nothing | (upstream) |
| `Qwen1_5_SFT_GLM/` | `scripts/train/sft_on_glm53_traces.py` | `loss_history.json` | https://huggingface.co/Tahatest123456/Qwen2.5-1.5B-GSM8K-SFT |
| `Qwen1_5_SFT_GLM_GRPO/` | `scripts/train/grpo_on_the_bootstrapped_GLM_student.py` | `training_curves.json` | https://huggingface.co/Tahatest123456/Qwen2.5-1.5B-GSM8K-SFT-GRPO |
| `Qwen2.5_zero/` | `scripts/train/grpo_zero_from_base_model.py` | `training_curves.json` | https://huggingface.co/Tahatest123456/Qwen2.5-1.5B-GSM8K-Zero |

To reproduce a benchmark, download the model into its directory above and run the matching
`scripts/inference/run_inference_*_on_val.py`.
