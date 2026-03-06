# ELLMob
This is the official implementation for the paper "[ICLR'26] ELLMob: Event-Driven Human Mobility Generation with Self-Aligned LLM Framework".

# Structure of ELLMob
![image](structure.png)

# Dataset
You can use the raw data for your own research, and you can also use the preprocessed files to continue research on event-driven human mobility.

```
python preprocess.py
```

# Event
In traj_generator.py to switch event type:

event_context = "Put event context here."

# API

Put your API configuration here at gpt_structure.py.

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
If this repository is useful for you, please cite as:
```

```
