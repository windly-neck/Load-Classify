#%%
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import time

class DayDataLoader:
    def __init__(self, max_missing=10, mad_thresh=4, method='interpolate'):
        self.max_missing = max_missing
        self.mad_thresh = mad_thresh
        self.method = method
        self.too_many_nan_count = 0
        self.too_many_zero_count = 0
        self.duplicate_count = 0

    def global_outlier_process(self, df):
        # 对全表做MAD异常值检测
        for col in ['kVARh_D', 'kWh_D']:
            series = df[col]
            median = series.median()
            mad = np.median(np.abs(series - median))
            if mad == 0 or np.isnan(mad):
                continue
            modified_z = 0.6745 * (series - median) / mad
            mask = modified_z.abs() > self.mad_thresh
            df.loc[mask, col] = np.nan
        return df

    def remove_invalid_samples(self, samples, zero_thresh=64):
        if len(samples) == 0:
            return samples
        valid_mask = []
        self.too_many_zero_count = 0
        for s in samples:
            zero_count = (s == 0).sum(axis=0)
            if (zero_count > zero_thresh).any():
                valid_mask.append(False)
                self.too_many_zero_count += 1
            else:
                valid_mask.append(True)
        return samples[np.array(valid_mask)]

    def remove_duplicate_samples(self, samples, corr_thresh=0.99):
        if len(samples) == 0:
            return samples
        keep = []
        seen = []
        self.duplicate_count = 0
        for s in samples:
            is_dup = False
            for t in seen:
                corr1 = np.corrcoef(s[:,0], t[:,0])[0,1]
                corr2 = np.corrcoef(s[:,1], t[:,1])[0,1]
                if corr1 > corr_thresh and corr2 > corr_thresh:
                    is_dup = True
                    break
            if not is_dup:
                keep.append(True)
                seen.append(s)
            else:
                keep.append(False)
                self.duplicate_count += 1
        return samples[np.array(keep, dtype=bool)]

    # def smooth_samples(self, samples, window=3):
    #     from scipy.ndimage import uniform_filter1d
    #     return uniform_filter1d(samples, size=window, axis=1)

    def handle_missing_one_day(self, df_day):
        feats = df_day[['kVARh_D', 'kWh_D']].copy()
        missing_count = feats.isnull().sum().sum()
        if missing_count > self.max_missing:
            self.too_many_nan_count += 1
            return None
        if self.method == 'interpolate':
            feats = feats.interpolate(limit_direction='both')
        elif self.method == 'mean':
            feats = feats.fillna(feats.mean())
        else:
            raise ValueError("Unknown method")
        if feats.isnull().any().any():
            return None
        # 对每一列做归一化
        for col in feats.columns:
            min_v = feats[col].min()
            max_v = feats[col].max()
            if max_v > min_v:
                feats[col] = (feats[col] - min_v) / (max_v - min_v)
            else:
                feats[col] = 0.0
        return feats.to_numpy()

    def load_and_process(self, file_path):
        self.too_many_nan_count = 0
        self.too_many_zero_count = 0
        self.duplicate_count = 0

        df = pd.read_excel(file_path, usecols=['IV_Date', 'IV_Time', 'kVARh_D', 'kWh_D'], na_values=["missing"])
        df = self.global_outlier_process(df)
        samples = []
        for date, group in df.groupby('IV_Date'):
            group = group.sort_values('IV_Time')
            feats = self.handle_missing_one_day(group)
            if feats is not None and feats.shape == (96, 2):
                samples.append(feats)
        if not samples:
            print(f"File: {file_path} has no valid samples after processing.")
            return np.empty((0, 96, 2))
        samples = np.stack(samples)
        samples = self.remove_invalid_samples(samples)
        # samples = self.smooth_samples(samples)  # 先不考虑滤波
        samples = self.remove_duplicate_samples(samples)
        print(f"File: {file_path}, Samples: {len(samples)}, Too many NaN: {self.too_many_nan_count}, "
              f"Too many zeros: {self.too_many_zero_count}, Duplicates removed: {self.duplicate_count}")
        return samples

