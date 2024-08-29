import time

import numpy as np

import torch
import matplotlib.pyplot as plt
from torch import optim, nn
from torch.utils.data import TensorDataset, DataLoader
from torchcnnbuilder.preprocess.time_series import multi_output_tensor

from get_sinthetic_data import get_anime_timeseries, get_cycled_data
from linear.linear_architecture import LinearNN, LinearEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Calculating on device: {device}')

cycles = 5
timeseries = get_anime_timeseries()
timeseries = get_cycled_data(timeseries, cycles)[:, :, :, 0]

'''plt.plot(timeseries[:, 15, 15])
plt.show()'''

timeseries_mask = np.sum(timeseries, axis=0)
timeseries_mask[timeseries_mask < 0.95 * cycles * 10] = np.nan
#timeseries[timeseries > 0.95*50] = np.nan
plt.imshow(timeseries_mask)
plt.colorbar()
plt.scatter(15, 15)
plt.show()

real_tss_mask = np.argwhere(np.isnan(timeseries_mask))
data = timeseries[:, real_tss_mask[:, 0], real_tss_mask[:, 1]]

train_dataset = multi_output_tensor(data=data,
                                    forecast_len=10,
                                    pre_history_len=20)

# складывам временные ряды один за одним
features = torch.reshape(train_dataset.tensors[0], (
train_dataset.tensors[0].shape[0], train_dataset.tensors[0].shape[1] * train_dataset.tensors[0].shape[2]))
target = torch.reshape(train_dataset.tensors[1], (
train_dataset.tensors[1].shape[0], train_dataset.tensors[1].shape[1] * train_dataset.tensors[1].shape[2]))
dataset = TensorDataset(features, target)

dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

model = LinearEncoder(features.shape[1], target.shape[1]).to(device)
print(model)

optimizer = optim.Adam(model.parameters(), lr=0.001)
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=10)
criterion = nn.L1Loss()
loss_history = []
best_val = float('inf')
best_model = None
max_epochs = 1000
start = time.time()

for epoch in range(max_epochs):
    model.train()
    loss = 0
    for train_features, test_features in dataloader:
        train_features = train_features[:, None].to(device)
        test_features = test_features[:, None].to(device)

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

end = time.time() - start
model_name = f'models/synthetic_linear_att.pt'
torch.save(model.state_dict(), model_name)

plt.plot(list(range(len(loss_history))), loss_history)
plt.grid()
plt.ylim(0, 1)
plt.title(f'Runtime={end}')
plt.xlabel('Epoch')
plt.ylabel(f'L1Loss')
plt.savefig(f'{model_name}.png')
plt.show()
