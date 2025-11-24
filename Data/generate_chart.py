# -*- coding: utf-8 -*-
from __future__ import print_function, division
import os
import os.path as op
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2

# 导入自定义图表库 - 修正导入路径
try:
    from Data.chart_library_cv import DrawOHLC, DrawChartError
except ImportError:
    try:
        from chart_library_cv import DrawOHLC, DrawChartError
    except ImportError:
        raise ImportError("无法导入chart_library_cv模块，请检查模块路径")

# 常量配置
WORK_DIR = "/hpc2hdd/home/jliu043/CNN_Replicate"
CACHE_DIR = os.path.join(WORK_DIR, "CACHE_DIR")
STOCKS_SAVEPATH = os.path.join(WORK_DIR, "data/stocks_dataset")
PROCESSED_DATA_DIR = os.path.join(WORK_DIR, "data/processed_data")


class ChartGenerationError(Exception):
    """图表生成过程中的异常"""
    pass


def get_dir(dir_path):
    """确保目录存在，如果不存在则创建它"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    return dir_path


class GenerateStockData(object):
    def __init__(
        self,
        year,
        window_size,
        batch_size=100,  # 添加默认值，避免调用问题
        freq="month",
        chart_freq=1,
        ma_lags=None,
        volume_bar=False,
        need_adjust_price=True,
        allow_tqdm=True,
        chart_type="bar",
        sample_rate=100,  # 添加采样率参数
        parallel_cores=4,  # 添加并行处理核心数
        temp_dir=None,    # 添加临时目录参数
    ):
        """初始化股票数据生成器
        
        Args:
            year: 处理的年份
            window_size: 窗口大小，必须为5、20或60
            batch_size: 批处理大小
            freq: 数据频率，"week"、"month"或"quarter"
            chart_freq: 图表频率
            ma_lags: 移动平均线的滞后期
            volume_bar: 是否包含交易量条形图
            need_adjust_price: 是否需要调整价格
            allow_tqdm: 是否允许显示进度条
            chart_type: 图表类型，"bar"、"pixel"或"centered_pixel"
            sample_rate: 采样率(1-100)，控制处理的数据比例
            parallel_cores: 并行处理的CPU核心数
            temp_dir: 临时文件目录
        """
        self.country = "USA"  # 固定为USA，不做国别区分
        self.year = year
        self.window_size = window_size
        self.batch_size = batch_size
        self.freq = freq
        self.sample_rate = min(max(1, sample_rate), 100)  # 确保在1-100之间
        self.parallel_cores = parallel_cores
        self.temp_dir = temp_dir
        
        assert self.freq in ["daily", "week", "month", "quarter"], f"不支持的频率: {self.freq}"
        self.chart_freq = chart_freq
        assert window_size % chart_freq == 0, "窗口大小必须是图表频率的整数倍"
        self.chart_len = int(window_size / chart_freq)
        assert self.chart_len in [5, 20, 60], f"不支持的图表长度: {self.chart_len}"
        
        self.ma_lags = ma_lags
        self.volume_bar = volume_bar
        self.need_adjust_price = need_adjust_price
        self.allow_tqdm = allow_tqdm
        
        assert chart_type in ["bar", "pixel", "centered_pixel"], f"不支持的图表类型: {chart_type}"
        self.chart_type = chart_type

        self.ret_len_list = [5, 20, 60]  # 只关注这三个时间窗口
        self.bar_width = 3
        self.image_width = {
            5: self.bar_width * 5,
            20: self.bar_width * 20,
            60: self.bar_width * 60,
        }
        self.image_height = {5: 32, 20: 64, 60: 96}

        self.width, self.height = (
            self.image_width[int(self.chart_len)],
            self.image_height[int(self.chart_len)],
        )

        self.df = None
        self.stock_id_list = None

        # 设置文件路径和名称
        self.save_dir = get_dir(
            op.join(STOCKS_SAVEPATH, f"stocks_{self.country}/dataset_all")
        )
        self.image_save_dir = get_dir(op.join(STOCKS_SAVEPATH, "sample_images"))
        vb_str = "has_vb" if self.volume_bar else "no_vb"
        ohlc_len_str = "" if self.chart_freq == 1 else f"_{self.chart_len}ohlc"
        chart_type_str = "" if self.chart_type == "bar" else f"{self.chart_type}_"
        self.file_name = f"{chart_type_str}{self.window_size}d_{self.freq}_{vb_str}_{str(self.ma_lags)}_ma_{self.year}{ohlc_len_str}"
        self.log_file_name = op.join(self.save_dir, f"{self.file_name}.txt")
        self.labels_filename = op.join(
            self.save_dir, f"{self.file_name}_labels.feather"
        )
        self.images_filename = op.join(self.save_dir, f"{self.file_name}_images.dat")
        
        # 输出初始化信息
        print(f"初始化 GenerateStockData: 年份={self.year}, 窗口={self.window_size}, 频率={self.freq}")
        print(f"图表类型: {self.chart_type}, 成交量: {self.volume_bar}, 移动平均线: {self.ma_lags}")
        print(f"数据将保存到: {self.save_dir}")

    @staticmethod
    def adjust_price(df):
        """调整价格，使第一天收盘价为1"""
        if len(df) == 0:
            raise ChartGenerationError("adjust_price: Empty Dataframe")
        if len(df.Date.unique()) != len(df):
            raise ChartGenerationError("adjust_price: Dates not unique")
        df = df.reset_index(drop=True)

        fd_close = abs(df.at[0, "Close"])
        if fd_close == 0.0 or pd.isna(fd_close):
            raise ChartGenerationError("adjust_price: First day close is nan or zero")

        pre_close = fd_close
        res_df = df.copy()

        res_df.at[0, "Close"] = 1.0
        res_df.at[0, "Open"] = abs(res_df.at[0, "Open"]) / pre_close
        res_df.at[0, "High"] = abs(res_df.at[0, "High"]) / pre_close
        res_df.at[0, "Low"] = abs(res_df.at[0, "Low"]) / pre_close

        pre_close = 1
        for i in range(1, len(res_df)):
            today_closep = abs(res_df.at[i, "Close"])
            today_openp = abs(res_df.at[i, "Open"])
            today_highp = abs(res_df.at[i, "High"])
            today_lowp = abs(res_df.at[i, "Low"])
            today_ret = np.float64(res_df.at[i, "Ret"])

            res_df.at[i, "Close"] = (1 + today_ret) * pre_close
            res_df.at[i, "Open"] = res_df.at[i, "Close"] / today_closep * today_openp
            res_df.at[i, "High"] = res_df.at[i, "Close"] / today_closep * today_highp
            res_df.at[i, "Low"] = res_df.at[i, "Close"] / today_closep * today_lowp
            res_df.at[i, "Ret"] = today_ret

            if not pd.isna(res_df.at[i, "Close"]):
                pre_close = res_df.at[i, "Close"]

        return res_df

    def load_adjusted_daily_prices(self, stock_df, date):
        """加载并调整每日价格"""
        if date not in set(stock_df.Date):
            return 0
        date_index = stock_df[stock_df.Date == date].index[0]
        ma_offset = 0 if self.ma_lags is None else np.max(self.ma_lags)
        data = stock_df.loc[
            (date_index - (self.window_size - 1) - ma_offset) : date_index
        ]
        if len(data) < self.window_size:
            return 1
        if len(data) < (self.window_size + ma_offset):
            ma_lags = []
            data = stock_df.loc[(date_index - (self.window_size - 1)) : date_index]
        else:
            ma_lags = self.ma_lags
        if self.chart_freq != 1:
            data = self.convert_daily_df_to_chart_freq_df(data)
        if self.need_adjust_price:
            if data["Close"].iloc[0] == 0.0 or pd.isna(data["Close"].iloc[0]):
                return 2
            data = self.adjust_price(data)
        else:
            data = data.copy()
        start_date_index = data.index[-1] - self.chart_len + 1
        if data["Close"].loc[start_date_index] == 0 or np.isnan(
            data["Close"].loc[start_date_index]
        ):
            return 3
        data[["Open", "High", "Low", "Close"]] *= (
            1.0 / data["Close"].loc[start_date_index]
        )
        if self.ma_lags is not None:
            ma_name_list = ["ma" + str(lag) for lag in ma_lags]
            for i, ma_name in enumerate(ma_name_list):
                chart_num = int(ma_lags[i] / self.chart_freq)
                data[ma_name] = (
                    data["Close"].rolling(chart_num, win_type=None).sum() / chart_num
                )

        data["Prev_Close"] = data["Close"].shift(1)

        df = data.loc[start_date_index:]
        if (
            len(df) != self.chart_len
            or np.around(df.iloc[0]["Close"], decimals=3) != 1.000
        ):
            return 4
        df = df.reset_index(drop=True)
        return df

    def convert_daily_df_to_chart_freq_df(self, daily_df):
        """将每日数据转换为图表频率数据"""
        if not len(daily_df) % self.chart_freq == 0:
            raise ChartGenerationError("df not divided by chart freq")
        ohlc_len = int(len(daily_df) / self.chart_freq)
        df = pd.DataFrame(index=range(ohlc_len), columns=daily_df.columns)
        for i in range(ohlc_len):
            subdata = daily_df.iloc[
                int(i * self.chart_freq) : int((i + 1) * self.chart_freq)
            ]
            df.loc[i] = subdata.iloc[-1]
            df.loc[i, "Open"] = subdata.iloc[0]["Open"]
            df.loc[i, "High"] = subdata["High"].max()
            df.loc[i, "Low"] = subdata["Low"].min()
            df.loc[i, "Vol"] = subdata["Vol"].sum()
            df.loc[i, "Ret"] = np.prod(1 + np.array(subdata["Ret"])) - 1
        return df

    def _generate_daily_features(self, stock_df, date):
        """生成每日特征"""
        res = self.load_adjusted_daily_prices(stock_df, date)
        if isinstance(res, int):
            return res
        else:
            df = res
        ma_lags = [int(ma_col[2:]) for ma_col in [c for c in df.columns if "ma" in c]]
        ohlc_obj = DrawOHLC(
            df,
            has_volume_bar=self.volume_bar,
            ma_lags=ma_lags,
            chart_type=self.chart_type,
        )
        image_data = ohlc_obj.draw_image()

        if image_data is None:
            return 5

        # 将OpenCV图像转换为与原代码兼容的对象
        pil_compatible_image = ohlc_obj.pil_compatible(image_data)

        last_day = stock_df[stock_df.Date == date].iloc[0]
        feature_dict = {feature: last_day[feature] for feature in stock_df.columns}
        ret_list = ["Ret"] + [f"Ret_{i}d" for i in self.ret_len_list]
        for ret in ret_list:
            if ret in feature_dict:
                feature_dict[f"{ret}_label"] = (
                    1 if feature_dict[ret] > 0 else 0 if feature_dict[ret] <= 0 else 2
                )
                vol = feature_dict.get("EWMA_vol", 0)
                feature_dict[f"{ret}_tstat"] = (
                    0 if (vol == 0 or pd.isna(vol)) else feature_dict[ret] / vol
                )
        feature_dict["image"] = pil_compatible_image
        feature_dict["window_size"] = self.window_size
        feature_dict["Date"] = date
        return feature_dict

    def _get_feature_and_dtype_list(self):
        """获取特征列表和数据类型"""
        # 定义全局变量
        global float32_features, uint8_features, int8_features, object_features
        
        float32_features = (
            [
                "EWMA_vol",
                "Ret",
                "Ret_tstat",
                "MarketCap",
            ]
            + [f"Ret_{i}d" for i in self.ret_len_list]
            + [f"Ret_{i}d_tstat" for i in self.ret_len_list]
        )
        
        # 检查频率相关的列
        for freq in ["daily", "week", "month", "quarter", "year"]:
            col_name = f"Ret_{freq}"
            float32_features.append(col_name)
        
        int8_features = ["Ret_label"] + [f"Ret_{i}d_label" for i in self.ret_len_list]
        uint8_features = ["image", "window_size"]
        object_features = ["StockID"]
        datetime_features = ["Date"]
        feature_list = (
            float32_features
            + int8_features
            + uint8_features
            + object_features
            + datetime_features
        )
        
        # 创建dtype字典
        float32_dict = {feature: np.float32 for feature in float32_features}
        int8_dict = {feature: np.int8 for feature in int8_features}
        uint8_dict = {feature: np.uint8 for feature in uint8_features}
        object_dict = {feature: object for feature in object_features}
        datetime_dict = {feature: "datetime64[ns]" for feature in datetime_features}
        dtype_dict = {
            **float32_dict,
            **int8_dict,
            **uint8_dict,
            **object_dict,
            **datetime_dict,
        }
        return dtype_dict, feature_list

    def get_processed_data_by_year(self, year):
        """获取指定年份的处理后数据"""
        processed_us_data_path = op.join(PROCESSED_DATA_DIR, "us_ret.feather")
        if not op.exists(processed_us_data_path):
            raise FileNotFoundError(f"找不到文件: {processed_us_data_path}")
            
        print(f"Loading processed data from {processed_us_data_path}")
        df = pd.read_feather(processed_us_data_path)
        df.set_index(["Date", "StockID"], inplace=True)
        df.sort_index(inplace=True)
        
        # 提取指定年份及前两年的数据
        df = df[
            df.index.get_level_values("Date").year.isin([year, year - 1, year - 2])
        ].copy()
        
        return df

    def get_period_end_dates(self, period):
        """获取特定周期的期末日期"""
        assert period in ["daily", "week", "month", "quarter", "year"], f"不支持的周期: {period}"
        
        # 从处理后的数据中提取对应频率的非空收益率日期作为期末日期
        df = self.get_processed_data_by_year(self.year)
        freq_column = f"Ret_{period}"
        
        if freq_column in df.columns:
            # 筛选出有收益率数据的日期
            period_dates = df[~pd.isna(df[freq_column])].index.get_level_values("Date").unique()
            # 筛选指定年份
            period_dates = period_dates[period_dates.year == self.year]
            
            # 应用采样率
            if self.sample_rate < 100:
                sample_size = max(1, int(len(period_dates) * self.sample_rate / 100))
                period_dates = np.random.choice(period_dates, size=sample_size, replace=False)
                period_dates = np.sort(period_dates)
                
            return period_dates
        else:
            # 如果没有对应的收益率列，创建合理的期末日期
            if period == "daily":
                dates = pd.date_range(start=f'{self.year}-01-01', end=f'{self.year}-12-31', freq='B')
            elif period == "week":
                dates = pd.date_range(start=f'{self.year}-01-01', end=f'{self.year}-12-31', freq='W-FRI')
            elif period == "month":
                dates = pd.date_range(start=f'{self.year}-01-01', end=f'{self.year}-12-31', freq='M')
            elif period == "quarter":
                dates = pd.date_range(start=f'{self.year}-01-01', end=f'{self.year}-12-31', freq='Q')
            else:  # year
                dates = pd.date_range(start=f'{self.year}-12-31', end=f'{self.year}-12-31', freq='Y')
                
            # 应用采样率
            if self.sample_rate < 100:
                sample_size = max(1, int(len(dates) * self.sample_rate / 100))
                dates = np.random.choice(dates, size=sample_size, replace=False)
                dates = np.sort(dates)
                
            return dates

    def save_annual_data(self):
        """生成并保存年度图表数据"""
        # 如果文件已存在，则跳过处理
        if (
            op.isfile(self.log_file_name)
            and op.isfile(self.labels_filename)
            and op.isfile(self.images_filename)
        ):
            print(f"Found pregenerated file {self.file_name}")
            return
        
        print(f"Generating {self.file_name}")
        self.df = self.get_processed_data_by_year(self.year)
        self.stock_id_list = np.unique(self.df.index.get_level_values("StockID"))
        
        # 如果设置了采样率，随机选择股票
        if self.sample_rate < 100:
            sample_size = max(1, int(len(self.stock_id_list) * self.sample_rate / 100))
            self.stock_id_list = np.random.choice(self.stock_id_list, size=sample_size, replace=False)
            print(f"应用{self.sample_rate}%采样率，处理{sample_size}只股票")
        
        dtype_dict, feature_list = self._get_feature_and_dtype_list()
        data_miss = np.zeros(6)
        data_dict = {
            feature: np.empty(len(self.stock_id_list) * 60, dtype=dtype_dict[feature])
            for feature in feature_list
        }
        # 修复：使用0代替np.nan来填充图像数组
        data_dict["image"] = np.empty(
            [len(self.stock_id_list) * 60, self.width * self.height],
            dtype=dtype_dict["image"],
        )
        data_dict["image"].fill(0)  # 使用0代替np.nan

        sample_num = 0
        iterator = (
            tqdm(self.stock_id_list)
            if (self.allow_tqdm and "tqdm" in sys.modules)
            else self.stock_id_list
        )
        
        try:
            for i, stock_id in enumerate(iterator):
                stock_df = self.df.xs(stock_id, level=1).copy()
                stock_df = stock_df.reset_index()
                dates = self.get_period_end_dates(self.freq)
                for j, date in enumerate(dates):
                    try:
                        image_label_data = self._generate_daily_features(stock_df, date)

                        if type(image_label_data) is dict:
                            # 保存样例图像，用于验证
                            if (i < 2) and (j == 0):
                                image_label_data["image"].save(
                                    op.join(
                                        self.image_save_dir,
                                        f"{self.file_name}_{stock_id}_{date.strftime('%Y%m%d')}.png",
                                    )
                                )

                            image_label_data["StockID"] = stock_id
                            im_arr = np.frombuffer(
                                image_label_data["image"].tobytes(), dtype=np.uint8
                            )
                            assert im_arr.size == self.width * self.height, f"图像尺寸不匹配: 期望{self.width * self.height}, 实际{im_arr.size}"
                            data_dict["image"][sample_num, :] = im_arr[:]
                            for feature in [x for x in feature_list if x != "image"]:
                                if feature in image_label_data:
                                    data_dict[feature][sample_num] = image_label_data[feature]
                                else:
                                    # 对于缺失的特征，填充默认值
                                    if feature in float32_features:
                                        data_dict[feature][sample_num] = np.nan
                                    elif feature in int8_features:
                                        data_dict[feature][sample_num] = -99
                                    elif feature in uint8_features:
                                        data_dict[feature][sample_num] = 0
                                    elif feature in object_features:
                                        data_dict[feature][sample_num] = ""
                            sample_num += 1
                        elif type(image_label_data) is int:
                            data_miss[image_label_data] += 1
                        else:
                            raise ValueError
                    except ChartGenerationError:
                        print(f"DGP Error on {stock_id} {date}")
                        continue
                    except Exception as e:
                        print(f"Unexpected error on {stock_id} {date}: {e}")
                        continue
                    
            # 截取有效数据
            for feature in feature_list:
                data_dict[feature] = data_dict[feature][:sample_num]

            # 保存图像数据
            print(f"Saving {sample_num} image data records to {self.images_filename}")
            fp_x = np.memmap(
                self.images_filename,
                dtype=np.uint8,
                mode="w+",
                shape=data_dict["image"].shape,
            )
            fp_x[:] = data_dict["image"][:]
            del fp_x
            
            # 保存标签数据
            print(f"Saving label data to {self.labels_filename}")
            data_dict = {x: data_dict[x] for x in data_dict.keys() if x != "image"}
            pd.DataFrame(data_dict).to_feather(self.labels_filename)
            
            # 保存日志
            log_file = open(self.log_file_name, "w+")
            log_file.write(
                "total_dates:%d total_missing:%d type0:%d type1:%d type2:%d type3:%d type4:%d type5:%d"
                % (
                    sample_num,
                    sum(data_miss),
                    data_miss[0],
                    data_miss[1],
                    data_miss[2],
                    data_miss[3],
                    data_miss[4],
                    data_miss[5],
                )
            )
            log_file.close()
            print(f"Generation completed for {self.file_name}")
            
        except Exception as e:
            print(f"Error in save_annual_data: {e}")
            import traceback
            traceback.print_exc()

# 其他方法保持不变...

def main():
    """命令行入口，支持直接调用"""
    import argparse
    
    parser = argparse.ArgumentParser(description="生成股票图表数据")
    parser.add_argument("--window_size", type=int, default=20, choices=[5, 20, 60],
                       help="图表窗口大小 (5, 20, 或 60)")
    parser.add_argument("--year", type=int, default=2000,
                       help="数据年份")
    parser.add_argument("--freq", type=str, default="month", 
                       choices=["daily", "week", "month", "quarter"],
                       help="数据频率")
    parser.add_argument("--chart_type", type=str, default="bar", 
                       choices=["bar", "pixel", "centered_pixel"],
                       help="图表类型")
    parser.add_argument("--volume_bar", action="store_true", default=False,
                       help="是否包含成交量条形图")
    parser.add_argument("--ts1d", action="store_true", default=False,
                       help="是否生成TS1D数据")
    parser.add_argument("--batch_size", type=int, default=100,
                       help="批处理大小")
    parser.add_argument("--sample_rate", type=int, default=100,
                       help="采样率(1-100)")
                       
    args = parser.parse_args()
    
    # 创建生成器
    generator = GenerateStockData(
        year=args.year,
        window_size=args.window_size,
        batch_size=args.batch_size,
        freq=args.freq,
        chart_freq=1,
        ma_lags=[args.window_size],
        volume_bar=args.volume_bar,
        chart_type=args.chart_type,
        sample_rate=args.sample_rate
    )
    
    # 生成图表数据
    generator.save_annual_data()
    
    # 如果需要，同时生成时间序列数据
    if args.ts1d:
        generator.save_annual_ts_data()
    
    print("处理完成")


if __name__ == "__main__":
    main()