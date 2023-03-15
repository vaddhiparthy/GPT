from transformers import Trainer, TrainingArguments

def configure_trainer_instance(model, tokenizer, train_dataset, test_dataset):
    training_args = TrainingArguments(
        output_dir="./output",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
        logging_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=create_data_collator(tokenizer),
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    return trainer
