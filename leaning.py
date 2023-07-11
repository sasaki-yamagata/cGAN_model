import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
from sklearn.model_selection import train_test_split
from cGAN.data.datasets import CreateData, GanDataset
from cGAN.models.cgan_model import Discriminator, Generator
from cGAN.common.method import generate_real_samples, generate_fake_samples
from cGAN.setting import config



ABS_PATH = os.path.dirname(os.path.abspath(__file__))
def main(file_path):
    data = CreateData(file_path)

    g_input_num = data.x_data.shape[1] + config["noise_size"]
    g_output_num = data.y_data.shape[1] 
    d_input_num = data.x_data.shape[1] + data.y_data.shape[1]

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
    criterion = nn.MSELoss()
    d_model = Discriminator(d_input_num)
    g_model = Generator(g_input_num, g_output_num)
    d_optimizer = optim.Adam(d_model.parameters(), lr=config["lr"])
    g_optimizer = optim.Adam(g_model.parameters(), lr=config["lr"])
    
    for epoch in range(config["n_epoch"]):
        history = {"epoch_num" : [],
               "train_loss" : [],
               "test_loss" : []}
        d_loss_accum = 0
        g_loss_accum = 0
        for phase in ["train", "test"]:
            for x_data, y_data in dataloaders[phase]:
                d_optimizer.zero_grad()
                g_optimizer.zero_grad()
                # Discriminatorの学習
                # torch.autograd.set_detect_anomaly(True)
                # with torch.set_grad_enabled()
                real_inputs, real_labels = generate_real_samples(x_data, y_data)
                real_outputs = d_model(real_inputs)
                d_loss = criterion(real_labels, real_outputs)
                
                d_loss.backward()
                d_optimizer.step()
                d_loss_accum += d_loss * config["batch_size"]

                # Generatorの学習
                fake_inputs, fake_labels = generate_fake_samples(g_model, x_data, config["noise_size"])
                fake_outputs = d_model(fake_inputs)
                g_loss = criterion(fake_labels, fake_outputs)
                
                g_loss.backward()
                g_optimizer.step()
                g_loss_accum += g_loss * config["batch_size"]
                # print(torch.sum(torch.isnan(d_model.state_dict())))
                # print(torch.sum(torch.isnan(g_model.state_dict())))

            n_data = dataloaders["train"].dataset.x_data.shape[0]
            d_loss_avg = d_loss_accum.item() / n_data
            g_loss_avg = g_loss_accum.item() / n_data
            print(f"epoch: {epoch}, d_loss: {d_loss_avg:.4f}, g_loss: {g_loss_avg:.4f}")



        





    


if __name__ == "__main__":
    main(file_path=f"{ABS_PATH}/datasets/Dataset_I.csv")
