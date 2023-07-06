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

def d_train(model, optimizer, x, criterion, is_real=True):

    # 1エポックあたりの累積損失(平均化前)
    loss_accum = 0

    for x_data, y in :     

        optimizer.zero_grad()
        
        # foward 
        # 自動微分をtrainのときのみ行う
        outputs = model(x_data, label)

        loss = criterion(outputs, y)


        loss.backward()
        
        optimizer.step()

    # lossは平均計算が行われているので平均前の損失に戻して加算
    loss_accum += loss * batch_size
    del loss
    
        