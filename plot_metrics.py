from datetime import datetime

import pandas as pd
from matplotlib import pyplot as plt

cnn_df = pd.read_csv('cnn/results/kara_metrics(20200101-20230101).csv')
#cnn_df['dates'] = pd.to_datetime(cnn_df['dates'])
#cnn_df = cnn_df.resample('Y', on='dates').mean()

cnn_att_df = pd.read_csv('cnn_attention/results/kara_metrics(20200101-20230101).csv')
#cnn_att_df['dates'] = pd.to_datetime(cnn_att_df['dates'])
#cnn_att_df = cnn_att_df.resample('Y', on='dates').mean()

unet_df = pd.read_csv('unet/results/kara_metrics(20200101-20230101).csv')
#unet_df['dates'] = pd.to_datetime(unet_df['dates'])
#unet_df = unet_df.resample('Y', on='dates').mean()

dates = pd.to_datetime(cnn_df['dates'])

mae_cnn = cnn_df['l1']
ssim_cnn = cnn_df['ssim']
acc_cnn = cnn_df['accuracy']

mae_att_cnn = cnn_att_df['l1']
ssim_att_cnn = cnn_att_df['ssim']
acc_att_cnn = cnn_att_df['accuracy']

mae_unet = unet_df['l1']
ssim_unet = unet_df['ssim']
acc_unet = unet_df['accuracy']

year_lines = [datetime(2021, 1, 1),
              datetime(2022, 1, 1),
              datetime(2023, 1, 1)]

plt.rcParams['figure.figsize'] = (12, 3)
plt.plot(dates, mae_cnn, label='CNN')
plt.plot(dates, mae_att_cnn, label='CNN+attention')
plt.plot(dates, mae_unet, label='U-Net')
for line in year_lines:
    plt.axvline(line, c='black', linewidth=1, linestyle='--')
plt.title('MAE')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

plt.plot(dates, ssim_cnn, label='CNN')
plt.plot(dates, ssim_att_cnn, label='CNN+attention')
plt.plot(dates, ssim_unet, label='U-Net')
for line in year_lines:
    plt.axvline(line, c='black', linewidth=1, linestyle='--')
plt.title('SSIM')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

plt.plot(dates, acc_cnn, label='CNN')
plt.plot(dates, acc_att_cnn, label='CNN+attention')
plt.plot(dates, acc_unet, label='U-Net')
for line in year_lines:
    plt.axvline(line, c='black', linewidth=1, linestyle='--')
plt.title('Accuracy (threshold=0.2)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()