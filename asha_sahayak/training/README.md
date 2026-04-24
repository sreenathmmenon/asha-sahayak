# ASHA Sahayak — Training

## Quick Start

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sreenathmmenon/asha-sahayak/blob/main/training/asha_grpo_training.ipynb)

## What this trains

The notebook trains Qwen3-0.6B using GRPO (Group Relative Policy Optimization) 
on the ASHA Sahayak clinical triage environment. The model learns to:

1. Ask the right clinical questions in the right order
2. Recognize danger signs from ASHA worker responses
3. Make correct referral decisions per the IMNCI protocol

## Requirements

- Google Colab with T4 GPU (free tier works)
- `HF_TOKEN` environment variable (for model push, optional)
- `ENV_BASE_URL` pointing to the ASHA Sahayak HF Space

## Expected Results

| Task Level | Random Baseline | After GRPO (200 steps) |
|---|---|---|
| Easy | ~0.25 | ~0.55+ |
| Medium | ~0.20 | ~0.45+ |
| Hard | ~0.15 | ~0.35+ |

## Training Config

- Model: Qwen/Qwen3-0.6B (600M params, fits on T4)
- Steps: 200
- Num generations: 4
- Gradient accumulation: 16
- Learning rate: 5e-6
