# -*- coding: utf-8 -*-
import pandas as pd
import os
import os.path as op
import numpy as np
import glob

from PIL import Image
from torch.utils.data import Dataset
from Data import dgp_config as dcf
from Data import equity_data as eqd
from Misc import utilities as ut


class EquityDataset(Dataset):
    def __init__(
        self,
        window_size,
        predict_window,
        freq,
        year,
        country="USA",
        has_volume_bar=True,
        has_ma=True,
        chart_type="bar",
        annual_stocks_num="all",
        tstat_threshold=0,
        stockid_filter=None,
        remove_tail=False,
        ohlc_len=None,
        regression_label=None,
        delayed_ret=0,
    ):
        self.ws = window_size
        self.pw = predict_window
        self.freq = freq
        assert self.freq in ["daily", "week", "month", "quarter", "year"]
        self.year = year
        self.ohlc_len = ohlc_len if ohlc_len is not None else window_size
        assert self.ohlc_len in [5, 20, 60]
        self.data_freq = self.freq if self.ohlc_len == self.ws else "month"
        self.country = country
        self.has_vb = has_volume_bar
        self.has_ma = has_ma
        self.chart_type = chart_type
        assert self.chart_type in ["bar", "pixel", "centered_pixel"]
        self.regression_label = regression_label
        assert self.regression_label in [None, "raw_ret", "vol_adjust_ret"]

        # Base path for dataset (update to match your data location)
        self.save_dir = op.join(dcf.STOCKS_SAVEPATH, f"stocks_{country}")
        
        # Try to load images and labels
        self.images, self.label_dict = self.load_images_and_labels_by_country(
            self.country
        )

        self.demean = self._get_insample_mean_std()

        assert delayed_ret in [0, 1, 2, 3, 4, 5]
        if self.country == "USA":
            self.ret_val_name = f"Ret_{dcf.FREQ_DICT[self.pw]}" + (
                "" if delayed_ret == 0 else f"_{delayed_ret}delay"
            )
        else:
            self.ret_val_name = f"next_{dcf.FREQ_DICT[self.pw]}_ret_{delayed_ret}delay"
        self.label = self.get_label_value()

        self.filter_data(
            annual_stocks_num, stockid_filter, tstat_threshold, remove_tail
        )

    def filter_data(
        self, annual_stocks_num, stockid_filter, tstat_threshold, remove_tail
    ):
        """过滤数据的方法"""
        try:
            df = pd.DataFrame(
                {
                    "StockID": self.label_dict["StockID"],
                    "MarketCap": abs(self.label_dict["MarketCap"]),
                    "Date": pd.to_datetime([str(t) for t in self.label_dict["Date"]]),
                }
            )
            
            # 确保所有数组长度一致
            n_samples = len(self.images)
            print(f"Initial number of samples: {n_samples}")
            
            if len(df) != n_samples:
                print(f"Warning: DataFrame length ({len(df)}) does not match number of images ({n_samples})")
                min_len = min(len(df), n_samples)
                df = df.iloc[:min_len]
                self.images = self.images[:min_len]
                for k in self.label_dict.keys():
                    self.label_dict[k] = self.label_dict[k][:min_len]
                n_samples = min_len
                print(f"Adjusted to {n_samples} samples")

            if annual_stocks_num != "all":
                num_stockid = len(np.unique(df.StockID))
                new_df = df
                period_end_dates = eqd.get_period_end_dates(self.freq)
                for i in range(15):
                    date = period_end_dates[
                        (period_end_dates.year == self.year) & (period_end_dates.month == 6)
                    ][
                        -i
                    ]
                    print(date)
                    new_df = df[df.Date == date]
                    if len(np.unique(new_df.StockID)) > num_stockid / 2:
                        break
                if stockid_filter is not None:
                    new_df = new_df[new_df.StockID.isin(stockid_filter)]
                new_df = new_df.sort_values(by=["MarketCap"], ascending=False)
                if len(new_df) > int(annual_stocks_num):
                    stockids = new_df.iloc[: int(annual_stocks_num)]["StockID"]
                else:
                    stockids = new_df.StockID
                print(
                    f"Year {self.year}: select top {annual_stocks_num} stocks ({len(stockids)}/{num_stockid}) stocks for training"
                )
            else:
                stockids = (
                    stockid_filter if stockid_filter is not None else np.unique(df.StockID)
                )
            stockid_idx = pd.Series(df.StockID).isin(stockids)

            idx = pd.Series(True, index=range(n_samples))  # 初始化全True的索引
            
            if tstat_threshold != 0:
                tstats = np.divide(
                    self.label_dict[self.ret_val_name], np.sqrt(self.label_dict["EWMA_vol"])
                )
                tstats = np.abs(tstats)
                t_th = np.nanpercentile(tstats[idx], tstat_threshold)
                tstat_idx = tstats > t_th
                print(
                    f"Before filtering bottom {tstat_threshold}% tstats, sample size:{np.sum(idx)}"
                )
                idx = idx & tstat_idx
                print(
                    f"After filtering bottom {tstat_threshold}% tstats, sample size:{np.sum(idx)}"
                )

            if remove_tail:
                print(
                    f"I{self.ws}R{self.pw}: removing tail for year {self.year} ({np.sum(idx)} samples)"
                )
                last_day = "12/24" if self.pw == 5 else "12/1" if self.pw == 20 else "10/1"
                last_day = pd.Timestamp("{}/{}".format(last_day, self.year))
                idx = idx & (
                    pd.to_datetime([str(t) for t in self.label_dict["Date"]]) < last_day
                )

            if self.freq != self.data_freq and self.ohlc_len != self.ws:
                assert self.freq in ["quarter", "year"] and self.data_freq == "month"
                print(f"Selecting data of freq {self.freq}")
                dates = pd.DatetimeIndex(self.label_dict["Date"])
                date_idx = (
                    dates.month.isin([3, 6, 9, 12])
                    if self.freq == "quarter"
                    else dates.month == 12
                )
                idx = idx & date_idx

            # 应用过滤器
            self.label = self.label[idx]
            print(f"Year {self.year}: filtered samples size: {len(self.label)}")
            
            for k in self.label_dict.keys():
                self.label_dict[k] = self.label_dict[k][idx]
            self.images = self.images[idx]
            
            # 转换数据类型
            self.label_dict["StockID"] = self.label_dict["StockID"].astype(str)
            self.label_dict["Date"] = self.label_dict["Date"].astype(str)
            
            # 最终验证
            assert len(self.label) == len(self.images)
            for k in self.label_dict.keys():
                assert len(self.images) == len(self.label_dict[k])
            
        except Exception as e:
            print(f"Error in filter_data: {e}")
            raise

    def get_label_value(self):
        print(f"Using {self.ret_val_name} as label")
        ret = self.label_dict[self.ret_val_name]

        print(
            f"Using {self.regression_label} regression label (None represents classification label)"
        )
        if self.regression_label == "raw_ret":
            label = np.nan_to_num(ret, nan=-99)
        elif self.regression_label == "vol_adjust_ret":
            label = np.nan_to_num(ret / np.sqrt(self.label_dict["EWMA_vol"]), nan=-99)
        else:
            label = np.where(ret > 0, 1, 0)
            label = np.nan_to_num(label, nan=-99)

        return label

    def _get_insample_mean_std(self):
        ohlc_len_srt = f"_{self.ohlc_len}ohlc" if self.ohlc_len != self.ws else ""
        chart_str = f"_{self.chart_type}" if self.chart_type != "bar" else ""
        fname = f"mean_std_{self.ws}d{self.data_freq}_vb{self.has_vb}_ma{self.has_ma}_{self.year}{ohlc_len_srt}{chart_str}.npz"
        mean_std_path = op.join(self.save_dir, "dataset_all", fname)
        
        # If mean/std file exists, use it
        if op.exists(mean_std_path):
            print(f"Loading mean and std from {mean_std_path}")
            x = np.load(mean_std_path, allow_pickle=True)
            demean = [x["mean"], x["std"]]
            return demean

        # Otherwise calculate from the loaded images
        print(f"Calculating mean and std")
        # Use a sample of images to save memory
        sample_size = min(50000, len(self.images))
        mean, std = (
            self.images[:sample_size].mean() / 255.0,
            self.images[:sample_size].std() / 255.0,
        )
        
        # Save the calculated values
        os.makedirs(op.dirname(mean_std_path), exist_ok=True)
        np.savez(mean_std_path, mean=mean, std=std)
        return [mean, std]

    def find_dataset_files(self, country=None):
        """Find dataset files using glob pattern based on window size and year"""
        if country is None:
            country = self.country
            
        # Look in dataset_daily_full directory
        daily_full_dir = op.join(self.save_dir, "dataset_daily_full")
        
        # Pattern for your actual data files with new format
        pattern = f"*_w{self.ohlc_len}_y{self.year}_*d_daily_has_vb_[{self.ohlc_len}]_ma_{self.year}_images.dat"
        image_files = glob.glob(op.join(daily_full_dir, pattern))
        
        if not image_files:
            # If we can't find files with that pattern, try a more general one
            pattern = f"*_w{self.ohlc_len}_*{self.year}*_images.dat"
            image_files = glob.glob(op.join(daily_full_dir, pattern))
            
        if not image_files:
            # If still no files, raise an error
            raise FileNotFoundError(
                f"Dataset files not found for window size {self.ohlc_len} and year {self.year} in {daily_full_dir}"
            )
        
        # Find the corresponding label file
        img_file = image_files[0]
        label_file = img_file.replace('_images.dat', '_labels.feather')
        
        if not op.exists(label_file):
            raise FileNotFoundError(f"Label file not found: {label_file}")
            
        print(f"Found dataset files: {img_file} and {label_file}")
        return img_file, label_file

    def load_annual_data_by_country(self, country):
        """Load image and label data for a specific country and year"""
        try:
            # 找到对应的文件
            img_save_path, label_path = self.find_dataset_files(country)
            
            print(f"Loading images from {img_save_path}")
            images = self.load_image_np_data(img_save_path, self.ohlc_len)
            
            # 创建重建图像的目录
            rebuild_dir = op.join(self.save_dir, "images_rebuilt_from_dataset")
            os.makedirs(rebuild_dir, exist_ok=True)
            
            # 保存样本图像用于可视化
            self.rebuild_image(
                images[0][0],
                image_name=f"sample_{self.ohlc_len}d_{self.year}",
                par_save_dir=self.save_dir,
            )
            
            # 加载标签数据
            print(f"Loading labels from {label_path}")
            label_df = pd.read_feather(label_path)
            label_df["StockID"] = label_df["StockID"].astype(str)
            
            # 确保图像数量和标签数量匹配
            if len(images) != len(label_df):
                print(f"Warning: Number of images ({len(images)}) does not match number of labels ({len(label_df)})")
                # 使用较小的长度
                min_len = min(len(images), len(label_df))
                images = images[:min_len]
                label_df = label_df.iloc[:min_len]
            
            label_dict = {c: np.array(label_df[c]) for c in label_df.columns}
            
            print(f"Loaded {len(images)} samples")
            return images, label_dict
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    @staticmethod
    def rebuild_image(image, image_name, par_save_dir, image_mode="L"):
        """Save a sample image for visualization"""
        img = Image.fromarray(image, image_mode)
        save_dir = op.join(par_save_dir, "images_rebuilt_from_dataset")
        os.makedirs(save_dir, exist_ok=True)
        img.save(op.join(save_dir, "{}.png".format(image_name)))

    def load_image_np_data(self, img_save_path, ohlc_len):
        """Load image data from binary file"""
        # 首先获取文件大小
        file_size = os.path.getsize(img_save_path)
        
        # 计算实际的样本数量
        # 每个样本是一个图像，大小为 IMAGE_HEIGHT * IMAGE_WIDTH
        sample_size = dcf.IMAGE_HEIGHT[ohlc_len] * dcf.IMAGE_WIDTH[ohlc_len]
        n_samples = file_size // sample_size
        
        # 使用实际的样本数量来读取数据
        images = np.memmap(img_save_path, dtype=np.uint8, mode="r")
        images = images.reshape(
            (n_samples, 1, dcf.IMAGE_HEIGHT[ohlc_len], dcf.IMAGE_WIDTH[ohlc_len])
        )
        return images

    def load_images_and_labels_by_country(self, country):
        """Load images and labels for a country"""
        images, label_dict = self.load_annual_data_by_country(country)
        return images, label_dict

    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.label)

    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        # Normalize the image
        image = (self.images[idx] / 255.0 - self.demean[0]) / self.demean[1]
        
        # Create the sample dictionary
        sample = {
            "image": image,
            "label": self.label[idx],
            "ret_val": self.label_dict[self.ret_val_name][idx],
            "ending_date": self.label_dict["Date"][idx],
            "StockID": self.label_dict["StockID"][idx],
            "MarketCap": self.label_dict["MarketCap"][idx],
        }
        return sample


def load_ts1d_np_data(ts1d_save_path, ohlc_len):
    images = np.memmap(ts1d_save_path, dtype=np.uint8, mode="r")
    images = images.reshape(
        (-1, 6, dcf.IMAGE_HEIGHT[ohlc_len], dcf.IMAGE_WIDTH[ohlc_len])
    )
    return images


class TS1DDataset(Dataset):
    def __init__(
        self,
        window_size,
        predict_window,
        freq,
        year,
        country="USA",
        remove_tail=False,
        ohlc_len=None,
        ts_scale="image_scale",
        regression_label=None,
    ):
        self.ws = window_size
        self.pw = predict_window
        self.freq = freq
        self.year = year
        self.ohlc_len = ohlc_len if ohlc_len is not None else window_size
        self.data_freq = self.freq if self.ohlc_len == self.ws else "month"
        self.country = country
        self.remove_tail = remove_tail
        self.ts_scale = ts_scale
        assert self.ts_scale in ["image_scale", "ret_scale", "vol_scale"]
        self.regression_label = regression_label
        assert self.regression_label in [None, "raw_ret", "vol_adjust_ret"]
        self.ret_val_name = f"Retx_{dcf.FREQ_DICT[self.pw]}"
        self.images, self.label_dict = self.load_ts1d_data()
        self.label = self.get_label_value()
        self.demean = self._get_1d_mean_std()
        self.filter_data(self.remove_tail)

    def load_ts1d_data(self):
        dataset_name = self.__get_stock_dataset_name()
        filename = op.join(
            dcf.STOCKS_SAVEPATH,
            "stocks_USA_ts/dataset_all/",
            "{}_data_new.npz".format(dataset_name),
        )
        data = np.load(filename, mmap_mode="r", encoding="latin1", allow_pickle=True)
        label_dict = data["data_dict"].item()
        images = label_dict["predictor"].copy()
        assert images[0].shape == (6, self.ohlc_len)
        del label_dict["predictor"]
        label_dict["StockID"] = label_dict["StockID"].astype(str)
        return images, label_dict

    def get_label_value(self):
        print(f"Using {self.ret_val_name} as label")
        ret = self.label_dict[self.ret_val_name]

        print(
            f"Using {self.regression_label} regression label (None represents classification label)"
        )
        if self.regression_label == "raw_ret":
            label = np.nan_to_num(ret, nan=-99)
        elif self.regression_label == "vol_adjust_ret":
            label = np.nan_to_num(ret / np.sqrt(self.label_dict["EWMA_vol"]), nan=-99)
        else:
            label = np.where(ret > 0, 1, 0)
            label = np.nan_to_num(label, nan=-99)
        return label

    def filter_data(self, remove_tail):
        idx = pd.Series(self.label != -99) & pd.Series(
            self.label_dict["EWMA_vol"] != 0.0
        )

        if remove_tail:
            print(
                f"I{self.ws}R{self.pw}: removing tail for year {self.year} ({np.sum(idx)} samples)"
            )
            last_day = "12/24" if self.pw == 5 else "12/1" if self.pw == 20 else "10/1"
            last_day = pd.Timestamp("{}/{}".format(last_day, self.year))
            idx = idx & (
                pd.to_datetime([str(t) for t in self.label_dict["Date"]]) < last_day
            )

        if self.freq != self.data_freq and self.ohlc_len != self.ws:
            assert self.freq in ["quarter", "year"] and self.data_freq == "month"
            print(f"Selecting data of freq {self.freq}")
            dates = pd.DatetimeIndex(self.label_dict["Date"])
            date_idx = (
                dates.month.isin([3, 6, 9, 12])
                if self.freq == "quarter"
                else dates.month == 12
            )
            idx = idx & date_idx

        self.label = self.label[idx]
        print(f"Year {self.year}: samples size: {len(self.label)}")
        for k in self.label_dict.keys():
            self.label_dict[k] = self.label_dict[k][idx]
        self.images = self.images[idx]
        self.label_dict["StockID"] = self.label_dict["StockID"].astype(str)
        self.label_dict["Date"] = self.label_dict["Date"].astype(str)

        assert len(self.label) == len(self.images)
        for k in self.label_dict.keys():
            assert len(self.images) == len(self.label_dict[k])

    def __get_stock_dataset_name(self):
        str_list = [
            f"{self.ws}d",
            self.data_freq,
            "has_vb",
            f"[{self.ws}]_ma",
            str(self.year),
        ]
        if self.ohlc_len != self.ws:
            str_list.append(f"{self.ohlc_len}ohlc")
        str_list.append("ts")
        dataset_name = "_".join(str_list)
        return dataset_name

    def _get_1d_mean_std(self):
        ohlc_len_srt = f"_{self.ohlc_len}ohlc" if self.ohlc_len != self.ws else ""
        raw_surfix = (
            ""
            if self.ts_scale == "image_scale"
            else "_raw_price"
            if self.ts_scale == "ret_scale"
            else "_vol_scale"
        )
        fname = f"mean_std_ts1d_{self.ws}d{self.data_freq}_vbTrue_maTrue_{self.year}{ohlc_len_srt}{raw_surfix}.npz"
        mean_std_path = op.join(
            ut.get_dir(
                op.join(dcf.STOCKS_SAVEPATH, f"stocks_{self.country}_ts", "dataset_all")
            ),
            fname,
        )

        if op.exists(mean_std_path):
            print(f"Loading mean and std from {mean_std_path}")
            x = np.load(mean_std_path, allow_pickle=True)
            demean = [x["mean"], x["std"]]
            return demean

        if self.ts_scale == "image_scale":
            for i in range(self.images.shape[0]):
                self.images[i] = self._minmax_scale_ts1d(self.images[i])
        elif self.ts_scale == "vol_scale":
            for i in range(self.images.shape[0]):
                self.images[i] = self._vol_scale_ts1d(self.images[i]) / np.sqrt(
                    self.label_dict["EWMA_vol"][i]
                )

        print(f"Calculating mean and std for {fname}")
        mean, std = np.nanmean(self.images, axis=(0, 2)), np.nanstd(
            self.images, axis=(0, 2)
        )
        np.savez(mean_std_path, mean=mean, std=std)
        return [mean, std]

    def _minmax_scale_ts1d(self, image):
        assert image.shape == (6, self.ohlc_len)
        ohlcma = image[:5]
        image[:5] = (ohlcma - np.nanmin(ohlcma)) / (
            np.nanmax(ohlcma) - np.nanmin(ohlcma)
        )
        image[5] = (image[5] - np.nanmin(image[5])) / (
            np.nanmax(image[5]) - np.nanmin(image[5])
        )
        return image

    def _vol_scale_ts1d(self, image):
        img = image.copy()
        img[:, 0] = 0
        for i in range(1, 5):
            img[:, i] = image[:, i] / image[0, i - 1] - 1
        return img

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.ts_scale == "image_scale":
            image = self._minmax_scale_ts1d(image)
        elif self.ts_scale == "vol_scale":
            image = self._vol_scale_ts1d(image) / np.sqrt(
                self.label_dict["EWMA_vol"][idx]
            )

        image = (image - self.demean[0].reshape(6, 1)) / self.demean[1].reshape(6, 1)
        image = np.nan_to_num(image, nan=0, posinf=0, neginf=0)

        sample = {
            "image": image,
            "label": self.label[idx],
            "ret_val": self.label_dict[self.ret_val_name][idx],
            "ending_date": self.label_dict["Date"][idx],
            "StockID": self.label_dict["StockID"][idx],
            "MarketCap": self.label_dict["MarketCap"][idx],
        }
        return sample


def main():
    pass


if __name__ == "__main__":
    main()
