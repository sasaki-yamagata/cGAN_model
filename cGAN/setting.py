
def get_config(debug=False):
    config = {
        "n_epoch": 50,
        "batch_size": 32,
        "noise_size": 100,
        "lr": 0.001
    }
    return config

config = get_config()