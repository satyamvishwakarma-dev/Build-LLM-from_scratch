# Building a Large Language Model (From Scratch)

This repository contains my personal implementation and codebase for building, pre-training, and instruction fine-tuning a GPT-style Large Language Model from the ground up using **PyTorch**. 

Following the architecture detailed in Sebastian Raschka's *Build a Large Language Model (From Scratch)*, this project explores the internal mechanics of modern LLMs without relying on high-level abstraction frameworks like Hugging Face or LangChain.

---

## 🛠️ Architecture & Technical Scope

* **Tokenizer:** Implemented Byte-Pair Encoding (BPE) using `tiktoken` to tokenize input text sequences.
* **Core Transformer Engine:** Constructed Multi-Head Causal Self-Attention mechanisms with positional encodings, LayerNormalization.
* **Pre-Training Pipeline:** Designed custom dataset loaders, sliding window target preparation, cross-entropy loss computation, and autoregressive generation loops.
* **Fine-Tuning Strategies:** 
  * Classification fine-tuning for downstream task adaptation (e.g., spam detection).
  * Instruction fine-tuning using formatted prompt datasets to transform the base model into an instruction-following assistant.
* **Weights & Optimization:** Implemented loading mechanisms for OpenAI's pre-trained GPT-2 parameters (124M and 355M) and adapted training routines for memory-constrained local GPU/Colab execution.

---

## 📁 Repository Overview

| Module | Focus Area | Key Components |
| :--- | :--- | :--- |
| **`ch02`** | Data Pipelines | Vocabulary construction, tokenization, data sampling, embedding layers. |
| **`ch03`** | Attention Mechanism | Single-head, causal multi-head attention, attention masks. |
| **`ch04`** | GPT Architecture | Full transformer block assembly, feed-forward layers, residual connections. |
| **`ch05`** | Pre-training & Decoding | Autoregressive sampling (Top-k, temperature), weight initialization, loss tracking. |
| **`ch06`** | Classification Fine-Tuning | Target head adaptation, fine-tuning loss optimization, classification evaluation. |
| **`ch07`** | Instruction Fine-Tuning | Prompt template formatting, dataset splitting, evaluation, and response generation. |

---

## 📊 Key Results & Key Takeaways

* **Loss Trajectory:** Successfully verified smooth loss convergence during instruction fine-tuning, reducing training loss down to $\approx 0.30$.
* **Inference Mechanics:** Experimented with Top-k sampling and temperature scaling to balance deterministic precision and creative response generation.
* **Hardware Efficiency:** Optimized PyTorch tensor operations and batching strategy for training on single-GPU environments.

---

## 📚 Acknowledgments & References

* Author: **Sebastian Raschka** — *Build a Large Language Model (From Scratch)* (Manning Publications).
* Official Repository: [`rasbt/LLMs-from-scratch`](https://github.com/rasbt/LLMs-from-scratch)
