def import_required_modules():
    import torch
    from torch.utils.data import Dataset, DataLoader, random_split
    from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast
    from transformers import TextDataset, DataCollatorForLanguageModeling
    from transformers import Trainer, TrainingArguments
    import matplotlib.pyplot as plt
