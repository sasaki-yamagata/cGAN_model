import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
from sklearn.model_selection import train_test_split
from cGAN.data.datasets import CreateData, GanDataset
from cGAN.models.cgan_model import Discriminator, Generator
from cGAN.common.method import generate_real_samples, generate_fake_samples, generate_noize
from cGAN.setting import config



ABS_PATH = os.path.dirname(os.path.abspath(__file__))
def main(file_path):
    data = CreateData(file_path)
    n_data = data.x_data.shape[0]
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

    criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.MSELoss()
    d_model = Discriminator(d_input_num)
    g_model = Generator(g_input_num, g_output_num)
    d_optimizer = optim.Adam(d_model.parameters(), lr=config["lr"])
    g_optimizer = optim.Adam(g_model.parameters(), lr=config["lr"])
    
    history = {"d_train_loss" : [],
                   "g_train_loss" : [],
                   "d_test_loss" : [],
                   "g_test_loss" : []}
    for epoch in range(config["n_epoch"]):
        d_loss_accum = 0
        g_loss_accum = 0
        for phase in ["train", "test"]:

            if phase == "train":
                d_model.train()
                g_model.train()
            else:
                d_model.eval()
                g_model.eval()

            for x_data, y_data in dataloaders[phase]:
                with torch.autograd.set_detect_anomaly(False):
                    with torch.set_grad_enabled(phase == "train"):

                        # ---------------------
                        #  Discriminatorの学習
                        # ---------------------
                        d_optimizer.zero_grad()
                        real_inputs, real_labels = generate_real_samples(x_data, y_data)
                        real_outputs = d_model(real_inputs)
                        d_loss_real = criterion(real_outputs, real_labels)

                        fake_inputs, fake_labels = generate_fake_samples(g_model, x_data, config["noise_size"])
                        fake_outputs = d_model(fake_inputs.detach())
                        d_loss_fake = criterion(fake_outputs, fake_labels)

                        d_loss = d_loss_real + d_loss_fake

                        # trainのときのみ重み更新
                        if phase == "train":
                            d_loss.backward()
                            d_optimizer.step()
                        d_loss_accum += d_loss * config["batch_size"]

                        # ---------------------
                        #  Generatorの学習
                        # ---------------------
                        g_optimizer.zero_grad()
                        fake_inputs, _ = generate_fake_samples(g_model, x_data, config["noise_size"])
                        fake_labels = torch.ones(fake_inputs.shape[0], 1)
                        fake_outputs = d_model(fake_inputs)
                        g_loss = criterion(fake_outputs, fake_labels)

                        # trainのときのみ重み更新
                        if phase == "train":
                            g_loss.backward()
                            g_optimizer.step()
                        g_loss_accum += g_loss * config["batch_size"]

            n_data = dataloaders[phase].dataset.x_data.shape[0]
            d_loss_avg = d_loss_accum.item() / n_data
            g_loss_avg = g_loss_accum.item() / n_data
            print(f"epoch: {epoch}, phase: {phase}, d_loss: {d_loss_avg:.4f}, g_loss: {g_loss_avg:.4f}")

            if phase == "train":
                history["d_train_loss"].append(d_loss_avg)
                history["g_train_loss"].append(g_loss_avg)
            else:
                history["d_test_loss"].append(d_loss_avg)
                history["g_test_loss"].append(g_loss_avg)


    # pred
    del dataloaders
    sampleloaders = {
        phase: DataLoader(
            datasets[phase],
            batch_size=config["n_samples"],
            shuffle=True
        ) for phase in ["train", "test"]
    }
    with torch.no_grad():
        for phase in ["train", "test"]:
            x_data, y_data = next(iter(sampleloaders[phase]))
            noise = generate_noize(x_data.shape[0], config["noise_size"])
            inputs = torch.cat((x_data, noise), dim=1)
            pred = g_model(inputs)
            col = 12
            row = 5
            fig, ax = plt.subplots(col, row, figsize=(18, 35))            
            count = 0
            for i in range(col):
                for j in range(row):
                    ax[i, j].plot(pred[count, :])
                    ax[i, j].plot(y_data[count, :])
                    ax[i, j].set_xlabel("Wavelength (μm)")
                    ax[i, j].set_ylabel("Absorbance")
                    count += 1
            fig.tight_layout()
            plt.savefig(f"result_{phase}.png")
        
    # save
    pd.DataFrame(history).to_csv("history.csv", index=False)
    torch.save(g_model.state_dict(), "generator.pth")





    


if __name__ == "__main__":
    main(file_path=f"{ABS_PATH}/datasets/Dataset_I.csv")
