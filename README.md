## Jailbreaking LLMs with Arabic Transliteration and Arabizi

This repository contains codes for our paper "[Jailbreaking LLMs with Arabic Transliteration and Arabizi](https://arxiv.org/pdf/2406.18725)" accepted in EMNLP 2024. Our paper investigates the use of Arabic non-standardized forms such as Arabizi and Transliteration to jailbreak LLMs. The paper also investigates potenial security risks of using these forms to vulnerabilities exposure such as learned model shortcuts. The results of the experiments highlights the need for more safety and adversarial training in cross-lingual manner with awareness of nonstandardized language forms, especially for Arabic.

## Environment Setup
1. Requirements:   <br/>
Python  <br/>
PyTorch  <br/>
openai  <br/>
anthropic <br/>

2. Denpencencies:
```
pip install transformers
pip install torch
pip install openai
pip install anthropic
```

## Data preparation
1. Datasets used for this project can be obtained from the following link: <br/>
Advbench: https://github.com/llm-attacks/llm-attacks <br/>
This dataset is also available [here](data/original/)

2. Use file [translate_convert_arabic.ipynb](translate_convert_arabic.ipynb) for helper codes for translation and convertion to Arabic and its forms.
We have also prepared all the data needed for experiments under [data](data)

## Experiments
```llm-test-ar.py``` contains all necessary codes to prompt Anthropic and OpenAI models. The file is commented and self-explained. <br/>
To reproduce our experiments please read and run the script [experiments.sh](experiments.sh). Evaluation is done manually, so manual inspection of results is mandatory.

### If you like this work please cite it.

## Citation

```
@article{ghanim2024jailbreaking,
  title={Jailbreaking LLMs with Arabic Transliteration and Arabizi},
  author={Ghanim, Mansour Al and Almohaimeed, Saleh and Zheng, Mengxin and Solihin, Yan and Lou, Qian},
  journal={arXiv preprint arXiv:2406.18725},
  year={2024}
}
```
