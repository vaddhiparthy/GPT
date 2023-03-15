from transformers import GPT2TokenizerFast, GPT2Config, GPT2LMHeadModel

def load_gpt2_model():
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    config = GPT2Config.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", config=config)
    return tokenizer, model
