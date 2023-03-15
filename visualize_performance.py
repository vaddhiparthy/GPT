import matplotlib.pyplot as plt

def plot_training_metrics(trainer, metric_name):
    plt.plot(trainer.state.log_history[metric_name], label=metric_name)
    plt.xlabel("Training Steps")
    plt.ylabel(metric_name.capitalize())
    plt.title(f"{metric_name.capitalize()} vs Training Steps")
    plt.legend()
    plt.show()
