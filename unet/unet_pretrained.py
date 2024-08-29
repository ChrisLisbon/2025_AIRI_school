import segmentation_models_pytorch as smp
import torch
from matplotlib import pyplot as plt
from skimage.transform import resize
from torch import nn, optim
from torch.utils.data import DataLoader
from torchcnnbuilder.preprocess.time_series import multi_output_tensor
from get_sinthetic_data import get_anime_timeseries, get_cycled_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Calculating on device: {device}')


prehistory_size = 20
forecast_length = 10


cycles = 5
timeseries = get_anime_timeseries()
timeseries = get_cycled_data(timeseries, cycles)[:, :, :, 0]
timeseries = resize(timeseries, (timeseries.shape[0], 64, 64))

train_dataset = multi_output_tensor(data=timeseries,
                                    forecast_len=forecast_length,
                                    pre_history_len=prehistory_size)
dataloader = DataLoader(train_dataset, batch_size=10, shuffle=False)

model = smp.Unet('resnet34', in_channels=prehistory_size, encoder_weights='imagenet')
head_conv = nn.Sequential(nn.Conv2d(16, forecast_length, kernel_size=(3, 3), padding=(1, 1)),
                     nn.ReLU(),
                     )
model.segmentation_head = head_conv
model.to(device)

criterion = nn.L1Loss()
loss_history = []
best_val = float('inf')
best_model = None
max_epochs = 10000
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(max_epochs):
    model.train()
    loss = 0
    for train_features, test_features in dataloader:
        train_features = train_features.to(device)
        test_features = test_features.to(device)

        optimizer.zero_grad()

        outputs = model(train_features)
        train_loss = criterion(outputs, test_features)
        train_loss.backward()
        optimizer.step()
        loss += train_loss.item()

    loss = loss / len(dataloader)
    #scheduler.step(loss)
    loss_history.append(loss)

    if loss < best_val:
        best_model = model
        best_val = loss
        print('Upd model')
    #print(f"-- epoch : {epoch + 1}/{max_epochs}, {loss=}, lr={scheduler.get_last_lr()}")
    print(f"-- epoch : {epoch + 1}/{max_epochs}, {loss=}")

    model.eval()

model_name = f'models/unet_imagenet_anime.pt'
torch.save(model.state_dict(), model_name)

plt.plot(list(range(len(loss_history))), loss_history)
plt.grid()
plt.ylim(0, 1)
plt.xlabel('Epoch')
plt.ylabel(f'L1Loss')
plt.savefig(f'{model_name}.png')
plt.show()
