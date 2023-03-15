from install_libraries import install_required_libraries
from import_modules import import_required_modules
from load_gpt2 import load_gpt2_model
from create_dataset import create_text_dataset, create_data_collator
from train_test_split import split_train_test
from configure_trainer import configure_trainer_instance
from train_model import train_gpt2_model
from evaluate_model import evaluate_gpt2_model
from visualize_performance import plot_training_metrics

# Install required libraries
install_required_libraries()

# Import required modules
import_required_modules()

# Load GPT-2 model and tokenizer
tokenizer, model = load_gpt2_model()

# Create dataset and data_collator
dataset = create_text_dataset(tokenizer, "path_to_your_text_corpus.txt")
data_collator = create_data_collator(tokenizer)

# Train-test split
train_dataset, test_dataset = split_train_test(dataset)

# Configure Trainer instance
trainer = configure_trainer_instance(model, tokenizer, train_dataset, test_dataset)

# Train the model
train_gpt2_model(trainer)

# Evaluate the model
eval_results = evaluate_gpt2_model(trainer)
print(eval_results)

# Visualize the model performance
plot_training_metrics(trainer.state.log_history, "loss")
