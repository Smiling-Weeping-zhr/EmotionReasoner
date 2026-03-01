# EmotionReasoner: Emotion-Explanation-Oriented Reinforcement Learning for Explainable Multimodal Emotion Recognition

Multimodal emotion recognition (MER) aims to identify emotional cues from multimodal inputs such as text and vision. With the rise of multimodal large language models (MLLMs), research has shifted toward explainable multimodal emotion recognition (EMER), which seeks not only accurate emotion prediction but also transparent reasoning. However, existing reinforcement learning (RL)-based methods often generate verbose and unfocused explanations, and current evaluation protocols remain coarse-grained.

To address these issues, we propose an RL-based optimization framework that integrates an explanation-aware reward and an emotion-polarity-oriented loss weighting strategy. The reward jointly considers structural validity, prediction correctness, and explanation quality, while the weighting mechanism emphasizes emotionally salient tokens to encourage emotion-centric reasoning. In addition, we introduce three complementary evaluation metrics—MEER-E, LACE-E, and UTAI-E—to provide a more fine-grained assessment of explanation fidelity, logical coherence, and practical utility. Extensive experiments demonstrate that the proposed framework improves both emotion recognition performance and explanation quality across multiple datasets.

---

## 📌 Overview

EmotionReasoner introduces a reasoning-enhanced learning framework to:

- We propose EmotionReasoner, an emotion-polarity-oriented RL framework that leverages explanation-aware rewards and emotion-polarity–guided token weighting to incentivize both emotion recognition and explanation. 
- Augmenting existing EMER evaluation frameworks, three complementary metrics—UTAI-E, LACE-E, and MEER-E—are introduced to rigorously evaluate finer-grained explanation quality on practical utility, logical clarity, and fidelity to multimodal cues. 
- Extensive experiments conducted on the MER-UniBench, which encompasses nine representative datasets, demonstrate that the proposed method achieves state-of-the-art average performance in MER, while additional evaluations further confirm its superiority in EMER. 

The framework integrates reinforcement learning (GRPO-based optimization), structured reward design, and model-driven self-evaluation to enhance reasoning quality.

---

## 📦 Dataset

### MERCaption+ Dataset

We conduct experiments on the **MERCaption+** dataset:

👉 https://huggingface.co/datasets/MERChallenge/MER2025

Please download the dataset from HuggingFace and place it under your designated data directory.

---

## 🧠 Base Model

EmotionReasoner is built upon the following pretrained backbone:

| Model | Description |
|-------|------------|
| [Qwen2.5-Omni-7B](https://www.modelscope.cn/models/qwen/Qwen2.5-7B) | Multimodal Large Language Model Backbone |

Please download the base checkpoint before training.

---

## 🚀 Released Checkpoints

### EmotionReasoner Models

| Model | Description |
|-------|------------|
| [EmotionReasoner](https://www.modelscope.cn/models/SmilingWeeping/EmotionReasoner) | RL-trained on **MERCaption+** |
| [EmotionReasoner-SelfEval](https://www.modelscope.cn/models/SmilingWeeping/EmotionReasoner) | RL + Self-Evaluation trained on **MERCaption+** |

---

## 🏋️ Training

This section describes how to train **EmotionReasoner** and **EmotionReasoner-SelfEval** from **Qwen2.5-Omni-7B**.

```bash
# Train EmotionReasoner
./bash/train_EmotionReasoner_SelfEval.sh
# Train EmotionReasoner-SelfEval
./bash/train_EmotionReasoner.sh
```