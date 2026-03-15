# ELLMob
This is the official implementation for the paper "[ICLR'26] ELLMob: Event-Driven Human Mobility Generation with Self-Aligned LLM Framework".

# ArXiv 
[![arXiv](https://img.shields.io/badge/arXiv-2401.12345-b31b1b.svg)](https://arxiv.org/abs/2603.07946)

[OpenReview Link](https://openreview.net/forum?id=MPYsaBgZIT)

# Structure of ELLMob
![image](structure.png)

# Dataset
You can use the [raw data](https://github.com/deepkashiwa20/ELLMob/tree/main/data) for your own research, and you can also use the [preprocessed files](https://github.com/deepkashiwa20/ELLMob/blob/main/data/preprocess.py) to continue research on event-driven human mobility.

```
python preprocess.py
```

# Event
In traj_generator.py to switch event type:

event_context = "Put event context here."

# API

Put your API configuration at gpt_structure.py.

client = AzureOpenAI(azure_endpoint="", api_key="", api_version="")

# Inference

```
python run.py
```

# evaluation

```
python evaluation.py
```

## Citation
If you find our datasets or codes useful, please kindly cite our paper and give a star.

```bibtex
@inproceedings{wang2026ellmob,
  title     = {ELLMob: Event-Driven Human Mobility Generation with Self-Aligned LLM Framework},
  author    = {Yusong Wang, Chuang Yang, Jiawei Wang, Xiaohang Xu, Jiayi Xu, Dongyuan Li, Chuan Xiao, Renhe Jiang},
  booktitle = {The Fourteenth International Conference on Learning Representations},
  year      = {2026}
}
```

```bibtex
@article{wang2026ellmob,
  title={ELLMob: Event-Driven Human Mobility Generation with Self-Aligned LLM Framework},
  author={Yusong Wang, Chuang Yang, Jiawei Wang, Xiaohang Xu, Jiayi Xu, Dongyuan Li, Chuan Xiao, Renhe Jiang},
  journal={arXiv preprint arXiv:2603.07946},
  year={2026}
}
```
