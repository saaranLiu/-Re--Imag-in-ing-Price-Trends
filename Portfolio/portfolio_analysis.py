import os
import sys
import pandas as pd
import numpy as np
import glob
import re
import traceback
import matplotlib.pyplot as plt
sys.path.insert(0, "/hpc2hdd/home/jliu043/CNN_Replicate")

from Portfolio.portfolio import PortfolioManager
from Misc import config as cf
from Misc import utilities as ut

def find_prediction_file(input_window, predict_window, year):
    """查找预测结果文件"""
    base_dir = "/hpc2hdd/home/jliu043/CNN_Replicate/results"
    model_dir = f"CNN_I{input_window}R{predict_window}"
    config_dir = "epoch5_ensem1_train1993-2000"
    pred_file = f"pred_year{year}.csv"
    
    file_path = os.path.join(
        base_dir,
        model_dir,
        config_dir,
        "predictions",
        pred_file
    )
    
    if not os.path.exists(file_path):
        print(f"预测文件不存在: {file_path}")
        return None
    
    print(f"找到预测文件: {file_path}")
    return file_path

def process_prediction_data(file_path):
    """
    处理预测数据文件
    
    Args:
        file_path (str): 预测文件路径
    
    Returns:
        pd.DataFrame: 处理后的数据框
    """
    print(f"处理文件: {file_path}")
    
    # 读取数据
    df = pd.read_csv(file_path)
    print(f"原始数据大小: {df.shape}")
    
    # 删除不需要的列
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    
    # 转换日期格式
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 处理缺失值
    print(f"\n缺失值统计:")
    print(df.isnull().sum())
    
    # 只保留有ret_val的记录（非NaN）
    df = df[df['ret_val'].notna()].copy()
    print(f"\n处理后数据大小: {df.shape}")
    
    # 确保数据类型正确
    df['StockID'] = df['StockID'].astype(str)
    df['MarketCap'] = df['MarketCap'].abs()
    
    # 设置索引
    df = df.set_index(['Date', 'StockID'])
    
    return df

def run_portfolio_analysis(input_window, predict_window, year):
    """
    运行投资组合分析
    
    Args:
        input_window (int): 输入窗口大小
        predict_window (int): 预测窗口大小
        year (int): 分析年份
    
    Returns:
        bool: 分析是否成功
    """
    print(f"\n分析 I{input_window}/R{predict_window} 模型 {year}年数据...")
    
    # 查找预测文件
    pred_file = find_prediction_file(input_window, predict_window, year)
    
    if pred_file is None:
        return False
    
    try:
        # 处理预测数据
        df = process_prediction_data(pred_file)
        
        if df.empty:
            print("警告: 处理后的数据为空")
            return False
        
        # 创建portfolio输出目录
        portfolio_dir = ut.get_dir(os.path.join(
            cf.PORTFOLIO_DIR, 
            f"CNN_I{input_window}R{predict_window}",
            f"epoch5_ensem1_train1993-2000",
            f"year{year}"
        ))
        
        print("\n创建投资组合管理器...")
        portfolio_manager = PortfolioManager(
            signal_df=df,
            freq="week",  # 使用周频数据
            portfolio_dir=portfolio_dir,
            country="USA",
            start_year=year,
            end_year=year
        )
        
        # 生成投资组合
        print("\n生成投资组合...")
        portfolio_manager.generate_portfolio(cut=10, delay=0)
        
        print(f"\nPortfolio分析完成，结果保存在: {portfolio_dir}")
        return True
        
    except Exception as e:
        print(f"分析过程出错: ")
        traceback.print_exc()
        return False

def combine_yearly_results(input_window, predict_window, start_year, end_year):
    """
    合并多年的分析结果
    
    Args:
        input_window (int): 输入窗口大小
        predict_window (int): 预测窗口大小
        start_year (int): 起始年份
        end_year (int): 结束年份
    """
    print(f"\n合并 I{input_window}/R{predict_window} {start_year}-{end_year}年结果...")
    
    # 基础目录
    base_dir = os.path.join(
        cf.PORTFOLIO_DIR,
        f"CNN_I{input_window}R{predict_window}",
        f"epoch5_ensem1_train1993-2000"
    )
    
    # 创建合并结果目录
    combined_dir = ut.get_dir(os.path.join(base_dir, "combined_results"))
    
    # 合并等权重和市值加权结果
    for weight_type in ["ew", "vw"]:
        try:
            # 收集所有年份的数据
            all_returns = []
            all_summaries = []
            
            for year in range(start_year, end_year + 1):
                year_dir = os.path.join(base_dir, f"year{year}")
                
                # 读取收益率数据
                returns_file = os.path.join(year_dir, "pf_data", f"pf_data_{weight_type}.csv")
                if os.path.exists(returns_file):
                    returns = pd.read_csv(returns_file, index_col=0, parse_dates=True)
                    all_returns.append(returns)
                
                # 读取摘要数据
                summary_file = os.path.join(year_dir, f"{weight_type}.csv")
                if os.path.exists(summary_file):
                    summary = pd.read_csv(summary_file, index_col=0)
                    all_summaries.append(summary)
            
            if all_returns:
                # 合并收益率数据
                combined_returns = pd.concat(all_returns)
                combined_returns.to_csv(
                    os.path.join(combined_dir, f"combined_{weight_type}_returns.csv")
                )
            
            if all_summaries:
                # 合并摘要数据
                combined_summary = pd.concat(all_summaries, axis=1)
                combined_summary.to_csv(
                    os.path.join(combined_dir, f"combined_{weight_type}_summary.csv")
                )
            
            print(f"{weight_type.upper()} 结果合并完成")
            
        except Exception as e:
            print(f"合并 {weight_type} 结果时出错: {str(e)}")
            traceback.print_exc()

def find_model_results(input_window, predict_window, year=2001):
    base_dir = "/hpc2hdd/home/jliu043/CNN_Replicate/results"
    pattern = f"D{input_window}*/*/ensem_res/ensem1_res_{year}_week.csv"
    
    # 使用 glob 查找匹配的文件
    result_files = glob.glob(os.path.join(base_dir, pattern))
    
    if not result_files:
        print(f"未找到 I{input_window}/R{predict_window} 的测试结果文件")
        return None
        
    # 按照文件路径中的 epoch 数排序，选择最新的结果
    result_files.sort(key=lambda x: int(re.search(r'-e(\d+)-', x).group(1)), reverse=True)
    return result_files[0]

def analyze_full_period_portfolio(input_window, predict_window, start_year, end_year):
    """对整个时期(2001-2019)进行完整的portfolio分析"""
    print(f"\n进行 I{input_window}/R{predict_window} {start_year}-{end_year} 整体分析...")
    
    # 收集所有年份的预测数据
    all_predictions = []
    for year in range(start_year, end_year + 1):
        pred_file = find_prediction_file(input_window, predict_window, year)
        if pred_file is None:
            continue
            
        df = process_prediction_data(pred_file)
        if not df.empty:
            all_predictions.append(df)
    
    if not all_predictions:
        print("没有找到有效的预测数据")
        return False
    
    # 合并所有年份的数据
    combined_df = pd.concat(all_predictions)
    
    # 创建整体分析的输出目录
    portfolio_dir = ut.get_dir(os.path.join(
        cf.PORTFOLIO_DIR, 
        f"CNN_I{input_window}R{predict_window}",
        f"epoch5_ensem1_train1993-2000",
        "full_period_2001_2019"
    ))
    
    print("\n创建投资组合管理器...")
    portfolio_manager = PortfolioManager(
        signal_df=combined_df,
        freq="week",  # 使用周频数据
        portfolio_dir=portfolio_dir,
        country="USA",
        start_year=start_year,
        end_year=end_year
    )
    
    # 生成投资组合
    print("\n生成整体时期投资组合...")
    portfolio_manager.generate_portfolio(cut=10, delay=0)
    
    # 计算和保存长期表现指标
    calculate_long_term_metrics(portfolio_dir)
    
    print(f"\n整体时期分析完成，结果保存在: {portfolio_dir}")
    return True

def calculate_long_term_metrics(portfolio_dir):
    """计算长期投资表现指标"""
    metrics = {}
    
    for weight_type in ['ew', 'vw']:
        # 读取收益率数据
        returns_file = os.path.join(portfolio_dir, "pf_data", f"pf_data_{weight_type}.csv")
        if not os.path.exists(returns_file):
            continue
            
        returns = pd.read_csv(returns_file, index_col=0, parse_dates=True)
        ls_returns = returns['LS']  # 多空组合收益
        
        # 计算年化指标
        annual_return = ls_returns.mean() * 252
        annual_vol = ls_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_vol if annual_vol != 0 else 0
        
        # 计算累积收益
        cum_returns = (1 + ls_returns).cumprod()
        total_return = cum_returns.iloc[-1] - 1
        
        # 计算最大回撤
        rolling_max = cum_returns.expanding().max()
        drawdowns = cum_returns / rolling_max - 1
        max_drawdown = drawdowns.min()
        
        # 计算胜率
        win_rate = (ls_returns > 0).mean()
        
        metrics[weight_type] = {
            'Total Return': total_return,
            'Annual Return': annual_return,
            'Annual Volatility': annual_vol,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown,
            'Win Rate': win_rate
        }
        
        # 保存详细指标
        metrics_df = pd.DataFrame.from_dict(metrics[weight_type], orient='index', columns=['Value'])
        metrics_df.to_csv(os.path.join(portfolio_dir, f"full_period_{weight_type}_metrics.csv"))
        
        # 打印结果
        print(f"\n{weight_type.upper()} 策略长期表现 (2001-2019):")
        for metric, value in metrics[weight_type].items():
            print(f"{metric}: {value:.4f}")
        
        # 绘制累积收益图
        plt.figure(figsize=(12, 6))
        cum_returns.plot()
        plt.title(f'{weight_type.upper()} Strategy Cumulative Returns (2001-2019)')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.grid(True)
        plt.savefig(os.path.join(portfolio_dir, f"full_period_{weight_type}_cumulative_returns.png"))
        plt.close()

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="投资组合分析")
    
    parser.add_argument("--years", type=str, default="2001-2019",
                      help="分析年份范围 (格式: START-END)")
    parser.add_argument("--input-windows", type=str, required=True,
                      help="输入窗口大小列表，用逗号分隔 (例如: 5,20,60)")
    parser.add_argument("--predict-windows", type=str, required=True,
                      help="预测窗口大小列表，用逗号分隔 (例如: 5)")
    
    args = parser.parse_args()
    
    # 解析年份范围
    start_year, end_year = map(int, args.years.split("-"))
    
    # 解析窗口参数
    input_windows = [int(x) for x in args.input_windows.split(",")]
    predict_windows = [int(x) for x in args.predict_windows.split(",")]
    
    print("开始Portfolio分析...")
    print("=" * 80)
    
    # 分析每个模型
    for input_window in input_windows:
        for predict_window in predict_windows:
            print(f"\n{'-' * 40}")
            
            # 1. 按年分析
            for year in range(start_year, end_year + 1):
                success = run_portfolio_analysis(
                    input_window=input_window,
                    predict_window=predict_window,
                    year=year
                )
                if not success:
                    print(f"I{input_window}/R{predict_window} {year}年分析失败")
            
            # 2. 合并年度结果
            combine_yearly_results(
                input_window=input_window,
                predict_window=predict_window,
                start_year=start_year,
                end_year=end_year
            )
            
            # 3. 进行整体时期分析
            analyze_full_period_portfolio(
                input_window=input_window,
                predict_window=predict_window,
                start_year=start_year,
                end_year=end_year
            )
    
    print("\n分析完成!")

if __name__ == "__main__":
    main()