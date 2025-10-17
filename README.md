# CLAWS:Creativity detection for LLM-generated solutions using Attention Window of Sections
> Keuntae Kim*, Eunhye Jeong*, Sehyeon Lee, Seohee Yoon, Yong Suk Choi†
> 
> \*: Equal contribution, †: Corresponding author

This repository is the official implementation of [CLAWS:Creativity detection for LLM-generated solutions using Attention Window of Sections](https://neurips.cc/virtual/2025/poster/115116) (NeurIPS 2025).

## Overview
![image](./images/overview.png)
**Step 1:** LLM generates a solution. \
**Step 2:** Features are extracted during generation. \
**Step 3:** LLM Evaluator assigns labels (Hallucinated / Creative / Typical). \
**Step 4:** Detection methods are evaluated by comparing predictions with the labels.
### CLAWS
![image](./images/CLAWS.png)
All tokens are segmented into five sections: Guideline, Problem, Solutions, Instruction, and Response. 
The average attention weight for each section $${\mathrm{AVGA_{\mathcal{u}}}}$$ is computed, normalized to obtain section-wise attention ratios $$\mathrm{CLAWS_\mathcal{u}}$$, and used as features for detecting Hallucinated, Creative, or Typical solutions.

## ⚙️ Requirements
To install requirements:
```
pip install -r requirements.txt
```

## 💻 Running CLAWS
### Reference Set Construction
```
python src/generation.py --model_name "model_name" --n_generation 20 --dataset_name REF
python src/evaluate.py --generation_model_name "model_name" --dataset_name REF
```
### TEST & Extended TEST set Construction
- dataset_name: TEST / AMC / AIME / AHSME
```
python src/generation.py --model_name "model_name" --n_generation 3 --dataset_name TEST
python src/evaluate.py --generation_model_name "model_name" --dataset_name TEST
```

## 📚 Citation
```

```
