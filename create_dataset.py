from transformers import TextDataset, DataCollatorForLanguageModeling

def create_text_dataset(tokenizer, file_path):
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=128,
    )
    return dataset

def create_data_collator(tokenizer):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    return data_collator
