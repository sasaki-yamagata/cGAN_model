
def get_config(debug=False):
    config = {
        "n_epoch": 12,
        "batch_size": 128,
        "noise_size": 100,
        "lr": 0.001,
        "n_samples": 60
    }
    return config

config = get_config()