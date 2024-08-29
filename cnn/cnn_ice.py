import os
from datetime import datetime
import numpy as np
import torch
from matplotlib import pyplot as plt
from skimage.transform import resize
from torch import nn, optim
from torch.utils.data import DataLoader
from torchcnnbuilder.models import ForecasterBase
from torchcnnbuilder.preprocess.time_series import multi_output_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Calculating on device: {device}')
def get_timespatial_series(sea_name, start_date, stop_date):
    """
    Function for loading spatiotemporal data for sea
    """
    datamodule_path = 'C:/Users/Julia/Documents/ICE_DATA_MODULE/OSISAF'
    files_path = f'{datamodule_path}/{sea_name}'
    timespatial_series = []
    dates_series = []
    for file in os.listdir(files_path):
        date = datetime.strptime(file, f'osi_%Y%m%d.npy')
        if start_date <= date.strftime('%Y%m%d') < stop_date:
            array = np.load(f'{files_path}/{file}')
            timespatial_series.append(array)
            dates_series.append(date)
    timespatial_series = np.array(timespatial_series)
    return timespatial_series, dates_series

sea_name = 'kara'
start_date = '19790101'
end_date = '20200101'
sea_data, dates = get_timespatial_series(sea_name, start_date, end_date)
sea_data = sea_data[::7]
dates = dates[::7]

batch = 10

general_folder = f'{sea_name}_cnn_2020({batch})'
weights_folder = f'{sea_name}_cnn_2020({batch})/weights'
images_folder = f'{sea_name}_cnn_2020({batch})/images'

if not os.path.exists(general_folder):
    os.mkdir(general_folder)
if not os.path.exists(weights_folder):
    os.mkdir(weights_folder)
if not os.path.exists(images_folder):
    os.mkdir(images_folder)

validation_sea_data, _ = get_timespatial_series(sea_name, end_date, '20240101')
validation_sea_data = validation_sea_data[::7]

pre_history_size = 104
forecast_size = 52

# init train dataset
sea_data = resize(sea_data, (sea_data.shape[0], 64, 64), anti_aliasing=False)
dataset = multi_output_tensor(data=sea_data,
                              forecast_len=forecast_size,
                              pre_history_len=pre_history_size)

# init validation dataset
validation_sea_data = resize(validation_sea_data, (validation_sea_data.shape[0], 64, 64), anti_aliasing=False)
validation_dataset = multi_output_tensor(data=validation_sea_data,
                              forecast_len=forecast_size,
                              pre_history_len=pre_history_size)
print('Creation train dataloader')
dataloader = DataLoader(dataset, batch_size=batch, shuffle=False)
print('Creation validation dataloader')
validation_dataloader = DataLoader(validation_dataset, batch_size=batch, shuffle=False)

model = ForecasterBase(input_size=(64, 64),
                       n_layers=5,
                       in_time_points=104,
                       out_time_points=52)
print(model)
model.to(device)

criterion = nn.L1Loss()
loss_history = []
val_loss_history = []
best_val = float('inf')
best_model = None
max_epochs = 500
optimizer = optim.Adam(model.parameters(), lr=0.001)
model_name = f'{weights_folder}/cnn_{sea_name}_{start_date}-{end_date}.pt'

for epoch in range(max_epochs):
    model.train()
    loss = 0
    for train_features, train_target in dataloader:
        train_features = train_features.to(device)
        train_target = train_target.to(device)

        optimizer.zero_grad()

        outputs = model(train_features)
        train_loss = criterion(outputs, train_target)
        train_loss.backward()
        optimizer.step()
        loss += train_loss.item()

    loss = loss / len(dataloader)
    loss_history.append(loss)

    val_loss = 0
    for val_features, val_target in validation_dataloader:
        val_features = val_features.to(device)
        val_target = val_target.to(device)

        val_outputs = model(val_features)
        val_batch_loss = criterion(val_outputs, val_target)

        val_loss += val_batch_loss.item()

    val_loss = val_loss / len(validation_dataloader)
    val_loss_history.append(val_loss)

    if loss < best_val:
        best_model = model
        best_val = loss
        print('Upd model')
    print(f"-- epoch : {epoch + 1}/{max_epochs}, loss = {round(loss, 5)}, validation loss = {round(val_loss, 5)}")
    model.eval()

    if epoch % 10 == 0:
        torch.save(model.state_dict(), model_name)

        plt.plot(range(len(loss_history)), loss_history, label='Train')
        plt.plot(range(len(val_loss_history)), val_loss_history, label='Validation')
        plt.grid()
        plt.xlabel('Epoch')
        plt.ylabel(f'L1Loss')
        plt.legend()
        plt.title('Convergence CNN')
        plt.savefig(f'{model_name}.png')
        plt.show()

        val_outputs = val_outputs.detach().cpu().numpy()[0]
        real = val_target.detach().cpu().numpy()[0]

        fig, (axs) = plt.subplots(2, 52, figsize=(40, 3))
        for i in range(52):
            axs[1, i].imshow(val_outputs[i], cmap='Blues_r', vmax=1, vmin=0)
            axs[1, i].set_title(F't={i}')
            axs[0, i].imshow(real[i], cmap='Blues_r', vmax=1, vmin=0)
            axs[0, i].set_title(F't={i}')
            axs[0, i].set_xticks([])
            axs[1, i].set_xticks([])
            axs[0, i].set_yticks([])
            axs[1, i].set_yticks([])
        plt.suptitle(f'Epoch={epoch}, loss={round(val_loss, 3)}')
        plt.tight_layout()
        plt.savefig(f'{images_folder}/val_images_{epoch}.png')
        plt.show()

torch.save(model.state_dict(), model_name)

plt.plot(range(len(loss_history)), loss_history, label='Train')
plt.plot(range(len(val_loss_history)), val_loss_history, label='Validation')
plt.grid()
plt.xlabel('Epoch')
plt.ylabel(f'L1Loss')
plt.legend()
plt.title('Convergence CNN')
plt.savefig(f'{model_name}.png')
plt.show()


