import os

from cGAN.data.datasets import CreateData, GanDataset
from cGAN.models.cgan_model import Discriminator, Generator
from cGAN.common.method import fit
from cGAN.setting import config
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
from sklearn.model_selection import train_test_split



ABS_PATH = os.path.dirname(os.path.abspath(__file__))
def main(file_path):
    data = CreateData(file_path)

    g_input_num = data.x_data.shape[1]
    g_output_num = data.y_data.shape[1]
    d_input_num = g_output_num

    x_train, x_test, y_train, y_test = train_test_split(data.x_data, data.y_data, test_size=0.25, random_state=1)
    del data

    dataset_train = GanDataset(x_train, y_train)
    dataset_test = GanDataset(x_test, y_test)
    datasets = {"train": dataset_train, "test": dataset_test}
    del x_train, y_train, x_test, y_test, dataset_train, dataset_test

    dataloaders = {
        phase: DataLoader(
            datasets[phase],
            batch_size=config["batch_size"],
        ) for phase in ["train", "test"]
    }
    criterion = nn.BCELoss()
    d_model = Discriminator(d_input_num)
    g_model = Generator(g_input_num, g_output_num)
    d_optimizer = optim.Adam(d_model.parameters(), lr=config["lr"])
    g_optimizer = optim.Adam(g_model.parameters(), lr=config["lr"])
    
    for epoch in range(config["n_epoch"]):
        
        for real_x, y in dataloaders["train"]:
            d_model()


    


if __name__ == "__main__":
    main(file_path=f"{ABS_PATH}/datasets/Dataset_I.csv")
