import torch
import numpy as np
# def fit(discriminator, generator, optimizer, criterion, n_epoch, dataloaders):
#     for epoch in range(n_epoch):
#         for phase in ["train", "test"]:
#             if phase == "train":
#                 discriminator.train()
#                 generator.train()
#             else:
#                 discriminator.eval()
#                 generator.eval()

#         for x_data, y_data in dataloaders[phase]:

# def d_train(model, optimizer, x, criterion, is_real=True):

#     # 1エポックあたりの累積損失(平均化前)
#     loss_accum = 0

#     for x_data, y in :     

#         optimizer.zero_grad()
        
#         # foward 
#         # 自動微分をtrainのときのみ行う
#         outputs = model(x_data, label)

#         loss = criterion(outputs, y)


#         loss.backward()
        
#         optimizer.step()

#     # lossは平均計算が行われているので平均前の損失に戻して加算
#     loss_accum += loss * batch_size
#     del loss
    
def generate_real_samples(x_data, y_data):
    real_inputs = torch.cat((x_data, y_data), dim=1)
    real_labels = torch.ones(real_inputs.shape[0], 1)
    return real_inputs, real_labels

def generate_fake_samples(generator, x_data, noize_size):
    noise = generate_noize(x_data.shape[0], noize_size)
    x_noise = torch.cat((x_data, noise), dim=1)
    fake_y_data = generator(x_noise)
    fake_inputs = torch.cat((x_data, fake_y_data), dim=1)
    fake_labels = torch.zeros(fake_inputs.shape[0], 1)
    return fake_inputs, fake_labels

def generate_noize(n_data, noise_size):
    noise = torch.randn(n_data, noise_size).float()
    return noise




    # @staticmethod
    # def _add_noize(fps_data, noize_size):
    #     noize_shape = (fps_data.shape[0], noize_size)
    #     z = np.random.randn(noize_shape[0] * noize_shape[1])
    #     z =  z.reshape(noize_shape)
    #     x_data = np.concatenate([fps_data, z], axis=1)
    #     x_data = torch.from_numpy(x_data)
    #     return x_data