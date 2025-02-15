import os
import pickle
import yaml
import matplotlib.pyplot as plt
import numpy as np

def load_log_data(log_file_path):
    with open(log_file_path, 'rb') as log_file:
        log_dict = pickle.load(log_file)
    return log_dict

def plot_training_log(log_dict, label, color, n_plot=200):
    epochs = log_dict["Epoch"]
    train_loss = log_dict["Train Loss"]
    test_loss = log_dict["Test Loss"]

    plt.plot(np.array(epochs)[:n_plot], train_loss[:n_plot], linestyle=':', label=f'{label} Train loss', color=color)
    plt.plot(np.array(epochs)[:n_plot], test_loss[:n_plot], label=f'{label} Validation loss', color=color)

def load_config_files(config_file_paths):
    config_files = {}
    for config_file_path in config_file_paths:
        with open(config_file_path, 'r') as f:
            conf = yaml.load(f, Loader=yaml.FullLoader)
            config_files[os.path.basename(config_file_path)] = conf
    return config_files

def main():
    n_plot = 100

    config_file_paths = [
        'config_files/foundParam_conv_4663_e2_155_small.yaml',
        'config_files/foundParam_conv_4663_e3_155_small.yaml',
        'config_files/foundParam_conv_4663_e4_155_small.yaml',
        'config_files/foundParam_conv_4663_e5_155_small.yaml'
    ]

    # Labels for the plots_for_report
    plot_labels = ["$e^{-2}$", "$e^{-3}$", "$e^{-4}$", "$e^{-5}$"]

    config_files = load_config_files(config_file_paths)
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']  # Different colors for each config file

    plt.figure(figsize=(10, 8))

    for idx, (config_name, conf) in enumerate(config_files.items()):
        log_file_path = os.path.join(conf["modelDir"], 'training_log.pkl')

        if os.path.exists(log_file_path):
            log_dict = load_log_data(log_file_path)

            # Use the corresponding label from the plot_labels list
            plot_training_log(log_dict, label=plot_labels[idx], color=colors[idx % len(colors)], n_plot=n_plot)

    plt.xlabel('Epoch [-]', fontsize=16)
    plt.ylabel('Loss [-]', fontsize=16)
    plt.title('Training and validation losses', fontsize=18)
    plt.legend(fontsize=12)
    plt.ylim(bottom=0)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"/home/justinbrusche/modeldirs_FVM/log_plot_LR.png")
    plt.show()

if __name__ == "__main__":
    import sys

    sys.path.append("/home/justinbrusche/scripts_FVM_2/trainTestModel")
    main()
