# ELLMob
This is the official implementation for the paper "[ICLR'26] ELLMob: Event-Driven Human Mobility Generation with Self-Aligned LLM Framework".

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
If you find this repository useful for your research, please consider citing our paper and giving the repository a star.
