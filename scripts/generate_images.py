#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成CNN模型训练用的图像数据
"""
import os
import sys
import time
import argparse
from datetime import datetime

# 添加项目根目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)

from Data.generate_chart import GenerateStockData

def generate_images(args):
    """生成图像数据"""
    start_time = time.time()
    
    # 记录参数
    print("="*80)
    print(f"Data generation started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Parameters:")
    print(f"  Window Size: {args.window_size}")
    print(f"  Year Range: {args.start_year} to {args.end_year}")
    print(f"  Frequency: {args.freq}")
    print(f"  Chart Type: {args.chart_type}")
    print(f"  Volume Bar: {args.volume_bar}")
    print(f"  Generate TS1D: {args.ts1d}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Parallel Cores: {args.parallel_cores}")
    print(f"  Temp Directory: {args.temp_dir}")
    print(f"  Sample Rate: {args.sample_rate}%")
    print("="*80)
    
    # 如果指定了临时目录，设置环境变量
    if args.temp_dir and os.path.exists(args.temp_dir):
        os.environ['TEMP_DIR'] = args.temp_dir
        print(f"Using temporary directory: {args.temp_dir}")
    
    # 遍历年份和窗口大小
    for year in range(args.start_year, args.end_year + 1):
        print(f"\nProcessing year {year}...")
        
        # 创建数据生成器，添加新参数
        generator = GenerateStockData(
            year=year,
            window_size=args.window_size,
            freq=args.freq,
            chart_freq=args.chart_freq,
            ma_lags=[args.window_size],  # 使用窗口大小作为移动平均线长度
            volume_bar=args.volume_bar,
            chart_type=args.chart_type,
            allow_tqdm=True,
            batch_size=args.batch_size,  # 添加批处理大小参数
        )
        
        # 设置生成器的并行核心数和临时目录（如果支持）
        if hasattr(generator, 'parallel_cores'):
            generator.parallel_cores = args.parallel_cores
        
        if args.temp_dir and hasattr(generator, 'temp_dir'):
            generator.temp_dir = args.temp_dir
            
        # 设置采样率（如果支持）
        if hasattr(generator, 'sample_rate'):
            generator.sample_rate = args.sample_rate
        
        # 生成CNN2D数据
        print(f"Generating CNN2D data for year {year}, window size {args.window_size}...")
        generator.save_annual_data()
        
        # 生成CNN1D数据
        if args.ts1d:
            print(f"Generating CNN1D data for year {year}, window size {args.window_size}...")
            generator.save_annual_ts_data()
    
    # 输出总耗时
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print("\n" + "="*80)
    print(f"Data generation completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate stock chart images for CNN training")
    parser.add_argument("--window_size", type=int, default=20, choices=[5, 20, 60],
                        help="Window size for chart generation (5, 20, or 60)")
    parser.add_argument("--start_year", type=int, default=1993,
                        help="Start year for data generation")
    parser.add_argument("--end_year", type=int, default=2019,
                        help="End year for data generation")
    parser.add_argument("--freq", type=str, default="daily", 
                        choices=["daily", "week", "month", "quarter"],
                        help="Frequency of data sampling")
    parser.add_argument("--chart_type", type=str, default="bar", 
                        choices=["bar", "pixel", "centered_pixel"],
                        help="Type of chart to generate")
    parser.add_argument("--volume_bar", action="store_true", default=True,
                        help="Include volume bar in chart")
    parser.add_argument("--chart_freq", type=int, default=1,
                        help="Chart frequency")
    parser.add_argument("--ts1d", action="store_true", default=False,
                        help="Generate TS1D data for CNN1D models")
    # 新增参数
    parser.add_argument("--batch_size", type=int, default=100, 
                        help="Number of stocks to process in each batch")
    parser.add_argument("--parallel_cores", type=int, default=4,
                        help="Number of CPU cores to use for parallel processing")
    parser.add_argument("--temp_dir", type=str, default="",
                        help="Temporary directory for storing intermediate files")
    parser.add_argument("--sample_rate", type=int, default=100,
                        help="Percentage of available dates to sample (1-100)")
    
    args = parser.parse_args()
    generate_images(args)
