# Data/equity_data.py
import os
import pandas as pd
import numpy as np
from Data import dgp_config as dcf

def get_period_ret(freq, country="USA"):
    """获取不同频率的收益率数据"""
    print(f"Loading {freq} return data for {country}")
    
    # 加载预处理的数据
    us_ret_path = os.path.join(dcf.PROCESSED_DATA_DIR, "us_ret.feather")
    df = pd.read_feather(us_ret_path)
    df.set_index(["Date", "StockID"], inplace=True)
    
    # 选择收益率列和市值列
    ret_cols = [f"Ret_{freq}"]
    
    # 0天延迟的下一期收益率
    df[f"next_{freq}_ret_0delay"] = df[f"Ret_{freq}"]
    
    # 对于不同的延迟天数(1-5天)，创建对应的收益率列
    for delay in range(1, 6):
        df[f"next_{freq}_ret_{delay}delay"] = df[f"Ret_{freq}_{delay}d"] if f"Ret_{freq}_{delay}d" in df.columns else np.nan
    
    # 确保结果中包含市值
    if "MarketCap" not in df.columns:
        df["MarketCap"] = 1.0  # 默认市值，如果数据中没有
    
    # 返回的列包括市值和不同延迟的收益率
    return_cols = ["MarketCap"] + [f"next_{freq}_ret_{d}delay" for d in range(6)]
    return df[return_cols]

def get_spy_freq_rets(freq):
    """获取SPY指数在特定频率下的收益率"""
    # 简化版本，创建一个假的SPY数据
    dates = pd.date_range(start="2001-01-01", end="2019-12-31", freq="B")
    np.random.seed(42)  # 为了可重复性
    rets = np.random.normal(0.001, 0.02, size=len(dates))  # 假设的收益率
    
    spy_df = pd.DataFrame({
        "Date": dates,
        "nxt_freq_ewret": rets,
        "nxt_freq_vwret": rets
    })
    spy_df.set_index("Date", inplace=True)
    
    # 根据频率过滤
    if freq == "week":
        spy_df = spy_df.resample("W").last()
    elif freq == "month":
        spy_df = spy_df.resample("M").last()
    elif freq == "quarter":
        spy_df = spy_df.resample("Q").last()
    
    return spy_df