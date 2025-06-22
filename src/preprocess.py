#%%
import pandas as pd
import numpy as np


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

    def remove_invalid_samples(self, samples, zero_thresh=64*7):
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
        # 不做归一化，直接返回原始数据
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

    def get_weekly_samples_from_file(self, file_path):
        """
        读取文件，返回以周为单位的样本，shape=(样本数, 672, 2)
        每个周样本的起点间隔七天（不重叠）。
        并统计各类无效样本数量。
        """
        self.too_many_nan_count = 0
        self.too_many_zero_count = 0
        self.duplicate_count = 0
        invalid_week_count = 0
        total_week_count = 0
        removed_zero_week = 0
        removed_dup_week = 0

        df = pd.read_excel(file_path, usecols=['IV_Date', 'IV_Time', 'kVARh_D', 'kWh_D'], na_values=["missing"])
        df = self.global_outlier_process(df)
        # 按天分组
        day_groups = list(df.groupby('IV_Date'))
        day_feats = []
        for date, group in day_groups:
            arr = self.handle_missing_one_day(group)
            day_feats.append(arr)
        # 按7天为一组，步长为7，生成周样本
        week_samples = []
        for i in range(0, len(day_feats) - 6, 7):
            total_week_count += 1
            week = day_feats[i:i+7]
            if any([d is None for d in week]):
                invalid_week_count += 1
                continue  # 有无效天，跳过
            week_arr = np.concatenate(week, axis=0)  # (672, 2)
            # 对每周整体归一化
            min_v = week_arr.min(axis=0)
            max_v = week_arr.max(axis=0)
            normed = (week_arr - min_v) / (max_v - min_v + 1e-8)
            week_samples.append(normed)
        if len(week_samples) == 0:
            print(f"File: {file_path} has no valid weekly samples after processing.")
            print(f"Total weeks: {total_week_count}, Invalid weeks: {invalid_week_count}, "
                  f"Too many NaN days: {self.too_many_nan_count}, Too many zeros: {self.too_many_zero_count}")
            return np.empty((0, 672, 2))
        week_samples = np.stack(week_samples)
        # 去除全零周样本
        before_zero = len(week_samples)
        week_samples = self.remove_invalid_samples(week_samples, zero_thresh=64*7)  # 7天
        removed_zero_week = before_zero - len(week_samples)
        # 去除重复周样本
        before_dup = len(week_samples)
        week_samples = self.remove_duplicate_samples(week_samples, corr_thresh=0.99)
        removed_dup_week = before_dup - len(week_samples)
        print(f"File: {file_path}, Weekly samples: {len(week_samples)}, Total weeks: {total_week_count}, "
              f"Invalid weeks: {invalid_week_count}, Too many NaN days: {self.too_many_nan_count}, "
              f"Too many zeros: {self.too_many_zero_count}, Removed zero weeks: {removed_zero_week}, "
              f"Removed duplicate weeks: {removed_dup_week}")
        return week_samples
