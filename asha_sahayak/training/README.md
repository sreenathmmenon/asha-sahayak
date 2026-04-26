# ASHA Sahayak — Training

## Notebooks

| Notebook | Description | Colab |
|---|---|---|
| [`asha_grpo_training.ipynb`](asha_grpo_training.ipynb) | Clean, re-runnable | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sreenathmmenon/asha-sahayak/blob/main/training/asha_grpo_training.ipynb) |
| [`...run2_200steps.ipynb`](asha_grpo_training_with_outputs_run2_200steps.ipynb) | Run 2 full outputs — final 0.75, peak 0.947 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sreenathmmenon/asha-sahayak/blob/main/training/asha_grpo_training_with_outputs_run2_200steps.ipynb) |
| [`...run3_400steps.ipynb`](asha_grpo_training_with_outputs_run3_400steps.ipynb) | Run 3 full outputs — 400 steps, peak 0.947 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sreenathmmenon/asha-sahayak/blob/main/training/asha_grpo_training_with_outputs_run3_400steps.ipynb) |

## What This Trains

Trains Qwen3-0.6B using GRPO on the ASHA Sahayak clinical triage environment. The model learns to:

1. Ask the right clinical questions in the right order
2. Recognize IMNCI danger signs from ASHA worker responses
3. Make correct referral decisions (REFER_IMMEDIATELY / REFER_WITHIN_24H / TREAT_AT_HOME / MONITOR)

## Requirements

- Google Colab with L4 or T4 GPU
- `HF_TOKEN` secret (for checkpoint push to HuggingFace)
- `ENV_BASE_URL` pointing to the ASHA Sahayak HF Space

## Actual Results

| Run | Steps | Baseline | Final | Peak |
|---|---|---|---|---|
| Run 1 — regex parsing | 200 | ~0.47 | ~0.52 | ~0.75 |
| Run 2 — JSON fix | 200 | 0.31 | **0.75** | **0.947** |
| Run 3 — extended | 400 | 0.14 | 0.66 | **0.947** |

**Best:** Run 2 — +142% improvement over baseline. Same peak (0.947) confirmed in Run 3.

## Training Config

- Model: Qwen/Qwen3-0.6B (600M params, fits on T4/L4)
- Algorithm: GRPO via TRL + Unsloth
- Num generations: 4
- Gradient accumulation: 8
- Learning rate: 5e-6
- Checkpoint: [sreenathmmenon/asha-sahayak-grpo](https://huggingface.co/sreenathmmenon/asha-sahayak-grpo)
