# CLAWS:Creativity detection for LLM-generated solutions using Attention Window of Sections
> Keuntae Kim*, Eunhye Jeong*, Sehyeon Lee, Seohee Yoon, Yong Suk Choi†
> 
> \*: Equal contribution, †: Corresponding author

This repository is the official implementation of [CLAWS:Creativity detection for LLM-generated solutions using Attention Window of Sections](https://neurips.cc/virtual/2025/poster/115116) (NeurIPS 2025).


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
