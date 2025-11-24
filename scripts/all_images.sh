#!/bin/bash
#SBATCH --job-name=full_data
#SBATCH --partition=i64m512u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --output=/hpc2hdd/home/jliu043/CNN_Replicate/logs/full_data_w%2_y%3_%A.out
#SBATCH --error=/hpc2hdd/home/jliu043/CNN_Replicate/logs/full_data_w%2_y%3_%A.err

# 直接使用参数指定窗口大小和年份
WINDOW_SIZE=$1
YEAR=$2

if [ -z "$WINDOW_SIZE" ] || [ -z "$YEAR" ]; then
    echo "错误: 必须提供窗口大小和年份参数"
    echo "用法: sbatch submit_job.sh 窗口大小 年份"
    echo "例如: sbatch submit_job.sh 5 2000"
    exit 1
fi

# 创建临时目录，使用本地存储提高I/O性能
TEMP_DIR=/tmp/${USER}/full_data_w${WINDOW_SIZE}_y${YEAR}_${SLURM_JOB_ID}
mkdir -p ${TEMP_DIR}
echo "使用临时目录: ${TEMP_DIR}"

# 创建所有必要的目录，确保它们存在
mkdir -p /hpc2hdd/home/jliu043/CNN_Replicate/logs
mkdir -p /hpc2hdd/home/jliu043/CNN_Replicate/data/stocks_dataset/sample_images/full_data
mkdir -p /hpc2hdd/home/jliu043/CNN_Replicate/data/stocks_dataset/stocks_USA/dataset_daily_full

# 设置环境变量以优化性能
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_MAX_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONHASHSEED=0

# 加载环境
source ~/.bashrc
conda activate cnn_env

# 设置工作目录
cd /hpc2hdd/home/jliu043/CNN_Replicate

# 打印任务信息
echo "========================================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "处理窗口大小: ${WINDOW_SIZE}, 年份: ${YEAR}"
echo "运行节点: $(hostname)"
echo "CPU核心数: ${SLURM_CPUS_PER_TASK}, 内存: ${SLURM_MEM_PER_NODE}"
echo "开始时间: $(date)"
echo "临时目录: ${TEMP_DIR}"
echo "========================================================="

# 创建一个临时Python脚本，处理全部数据
cat > ${TEMP_DIR}/daily_all.py << 'EOF'
import sys
import os
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import gc
import time
from tqdm import tqdm
import resource
import math

# 添加项目路径到Python路径
sys.path.insert(0, "/hpc2hdd/home/jliu043/CNN_Replicate")

# 从Data目录导入必要的类
try:
    from Data.generate_chart import GenerateStockData, get_dir
except ImportError as e:
    print(f"Error importing GenerateStockData: {e}")
    sys.exit(1)

def memory_usage():
    """获取当前内存使用情况，使用resource模块代替psutil"""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_maxrss / 1024  # 以MB为单位

def log_memory(message):
    """记录内存使用日志"""
    mem = memory_usage()
    print(f"{message} - 内存使用: {mem:.2f} MB")

def optimize_data_generation(window_size, year, job_id, temp_dir):
    """优化的数据生成函数，支持超大规模数据处理
    
    使用分块处理方法生成图表数据，减少内存占用
    """
    # 指定输出目录
    WORK_DIR = "/hpc2hdd/home/jliu043/CNN_Replicate"
    OUTPUT_DIR = os.path.join(WORK_DIR, "data/stocks_dataset/stocks_USA/dataset_daily_full")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 创建输出文件名
    output_prefix = f"full_data_{job_id}_w{window_size}_y{year}"
    
    log_memory("开始内存优化数据生成")
    start_time = time.time()

    # 1. 创建GenerateStockData实例 - 注意这里sample_rate=100表示处理全部数据
    generator = GenerateStockData(
        year=year,
        window_size=window_size,
        batch_size=100,
        freq="daily",   # 使用daily频率
        chart_freq=1,
        ma_lags=[window_size],
        volume_bar=True,
        chart_type="bar",
        sample_rate=100  # 处理所有股票
    )
    
    # 2. 修改输出路径和文件名
    generator.save_dir = OUTPUT_DIR
    generator.file_name = f"{output_prefix}_{generator.file_name}"
    generator.log_file_name = os.path.join(generator.save_dir, f"{generator.file_name}.txt")
    generator.labels_filename = os.path.join(generator.save_dir, f"{generator.file_name}_labels.feather")
    generator.images_filename = os.path.join(generator.save_dir, f"{generator.file_name}_images.dat")
    generator.image_save_dir = os.path.join(WORK_DIR, "data/stocks_dataset/sample_images/full_data")
    
    # 3. 修改save_annual_data方法进行分块处理
    original_save_method = generator.save_annual_data
    
    def optimized_save_annual_data(self):
        """优化版本的save_annual_data，使用分块处理和内存映射来处理超大规模数据"""
        # 检查文件是否已存在
        if (os.path.isfile(self.log_file_name) and 
            os.path.isfile(self.labels_filename) and 
            os.path.isfile(self.images_filename)):
            print(f"文件已存在: {self.file_name}")
            return
        
        print(f"生成数据: {self.file_name}")
        self.df = self.get_processed_data_by_year(self.year)
        
        # 获取股票ID列表
        self.stock_id_list = np.unique(self.df.index.get_level_values("StockID"))
        print(f"数据集中总股票数: {len(self.stock_id_list)}")
        
        # 获取每只股票的交易日数量估计
        dates = self.get_period_end_dates(self.freq)
        trading_days_per_year = len(dates)
        print(f"年份{self.year}估计交易日数: {trading_days_per_year}")
        
        # 预估样本量 - 考虑更大的上限 (300万/年/窗口)
        max_samples = 3_000_000
        print(f"使用样本上限: {max_samples}")
        
        # 获取数据类型和特征列表
        dtype_dict, feature_list = self._get_feature_and_dtype_list()
        
        # 为标签数据准备临时存储文件
        temp_label_files = []
        
        # 计算分块大小 - 每块处理部分股票
        chunk_size = 200  # 每块处理的股票数
        chunk_count = math.ceil(len(self.stock_id_list) / chunk_size)
        
        # 创建内存映射，预先分配足够空间
        fp_x = np.memmap(
            self.images_filename,
            dtype=np.uint8,
            mode="w+",
            shape=(max_samples, self.width * self.height)
        )
        
        # 初始化样本计数器
        total_samples = 0
        data_miss = np.zeros(6)
        
        # 分块处理股票
        for chunk_idx in range(chunk_count):
            print(f"处理分块 {chunk_idx+1}/{chunk_count}")
            
            # 确定当前块的股票
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, len(self.stock_id_list))
            chunk_stock_ids = self.stock_id_list[start_idx:end_idx]
            
            # 为当前块创建标签数据存储
            chunk_label_data = {feature: [] for feature in feature_list if feature != "image"}
            chunk_samples = 0
            
            # 处理当前块的股票
            for i, stock_id in enumerate(chunk_stock_ids):
                if total_samples >= max_samples:
                    print(f"警告: 已达到样本上限 {max_samples}")
                    break
                
                try:
                    # 获取当前股票的数据
                    stock_df = self.df.xs(stock_id, level=1).copy()
                    stock_df = stock_df.reset_index()
                    
                    # 处理每个交易日
                    for j, date in enumerate(dates):
                        if total_samples >= max_samples:
                            break
                            
                        try:
                            image_label_data = self._generate_daily_features(stock_df, date)
                            
                            if type(image_label_data) is dict:
                                # 保存样例图像
                                if chunk_idx == 0 and i < 2 and j == 0:
                                    sample_path = os.path.join(
                                        self.image_save_dir,
                                        f"{self.file_name}_{stock_id}_{date.strftime('%Y%m%d')}.png",
                                    )
                                    os.makedirs(os.path.dirname(sample_path), exist_ok=True)
                                    image_label_data["image"].save(sample_path)
                                
                                # 处理图像数据
                                image_label_data["StockID"] = stock_id
                                im_arr = np.frombuffer(
                                    image_label_data["image"].tobytes(), dtype=np.uint8
                                )
                                
                                # 验证图像尺寸
                                if im_arr.size != self.width * self.height:
                                    print(f"图像尺寸不匹配: 期望{self.width * self.height}, 实际{im_arr.size}")
                                    continue
                                    
                                # 存储图像数据
                                fp_x[total_samples, :] = im_arr[:]
                                
                                # 存储标签数据
                                for feature in [x for x in feature_list if x != "image"]:
                                    if feature in image_label_data:
                                        chunk_label_data[feature].append(image_label_data[feature])
                                    else:
                                        # 对缺失特征填充默认值
                                        if feature in [f for f in dtype_dict if dtype_dict[f] == np.float32]:
                                            chunk_label_data[feature].append(np.nan)
                                        elif feature in [f for f in dtype_dict if dtype_dict[f] == np.int8]:
                                            chunk_label_data[feature].append(-99)
                                        elif feature in [f for f in dtype_dict if dtype_dict[f] == np.uint8]:
                                            chunk_label_data[feature].append(0)
                                        elif feature in [f for f in dtype_dict if dtype_dict[f] == object]:
                                            chunk_label_data[feature].append("")
                                        else:
                                            chunk_label_data[feature].append(None)
                                
                                total_samples += 1
                                chunk_samples += 1
                                
                            elif type(image_label_data) is int:
                                data_miss[image_label_data] += 1
                                
                        except Exception as e:
                            print(f"处理{stock_id} {date}时出错: {e}")
                            continue
                    
                    # 每处理20只股票进行一次垃圾回收
                    if i % 20 == 19:
                        gc.collect()
                        
                except Exception as e:
                    print(f"处理股票{stock_id}时发生错误: {e}")
                    continue
            
            # 保存当前块的标签数据到临时文件
            if chunk_samples > 0:
                temp_label_file = os.path.join(temp_dir, f"chunk_{chunk_idx}_labels.feather")
                chunk_df = pd.DataFrame(chunk_label_data)
                chunk_df.to_feather(temp_label_file)
                temp_label_files.append((temp_label_file, chunk_samples))
                
                print(f"分块{chunk_idx+1}完成: 处理了{len(chunk_stock_ids)}只股票，生成{chunk_samples}个样本")
                
                # 释放内存
                del chunk_label_data
                del chunk_df
                gc.collect()
            
            # 如果已达到样本上限，停止处理
            if total_samples >= max_samples:
                break
        
        # 调整内存映射文件大小为实际样本数
        print(f"总共生成{total_samples}个样本")
        fp_x.flush()
        
        if total_samples < max_samples:
            print(f"调整内存映射文件大小为{total_samples}个样本")
            new_fp_x = np.memmap(
                self.images_filename,
                dtype=np.uint8,
                mode="r+",
                shape=(total_samples, self.width * self.height)
            )
            new_fp_x[:] = fp_x[:total_samples]
            del fp_x
            fp_x = new_fp_x
            
        # 合并所有临时标签文件
        print(f"合并{len(temp_label_files)}个临时标签文件")
        all_dfs = []
        for file_path, _ in temp_label_files:
            df = pd.read_feather(file_path)
            all_dfs.append(df)
            # 读取后立即删除临时文件以释放磁盘空间
            os.remove(file_path)
            
        final_df = pd.concat(all_dfs, ignore_index=True)
        
        # 保存最终标签数据
        print(f"保存标签数据到{self.labels_filename}")
        final_df.to_feather(self.labels_filename)
        
        # 清理资源
        del all_dfs
        del final_df
        del fp_x
        gc.collect()
        
        # 记录错误统计
        with open(self.log_file_name, "w") as f:
            f.write(f"Total samples: {total_samples}\n")
            f.write(f"Missing data statistics: {data_miss}\n")
        
        print(f"数据生成完成，总样本数: {total_samples}")
        return total_samples
    
    # 替换方法
    setattr(GenerateStockData, 'save_annual_data', optimized_save_annual_data)
    
    # 4. 执行数据生成
    log_memory("生成数据前")
    start = time.time()
    generator.save_annual_data()
    end = time.time()
    log_memory("生成数据后")
    
    print(f"处理完成，用时: {end - start:.2f}秒")
    return generator.labels_filename, generator.images_filename

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成全采样日频数据的优化脚本")
    parser.add_argument("--window_size", type=int, required=True, help="窗口大小 (5, 20或60)")
    parser.add_argument("--year", type=int, required=True, help="处理的年份")
    parser.add_argument("--job_id", type=str, required=True, help="作业ID")
    parser.add_argument("--temp_dir", type=str, required=True, help="临时目录")
    
    args = parser.parse_args()
    
    # 验证参数
    if args.window_size not in [5, 20, 60]:
        print(f"错误: 窗口大小必须是5, 20或60中的一个，收到的值是{args.window_size}")
        sys.exit(1)
        
    print(f"开始处理: 窗口大小={args.window_size}, 年份={args.year}")
    
    try:
        # 生成数据
        labels_file, images_file = optimize_data_generation(
            window_size=args.window_size,
            year=args.year,
            job_id=args.job_id,
            temp_dir=args.temp_dir
        )
        
        print(f"处理完成")
        print(f"标签文件: {labels_file}")
        print(f"图像文件: {images_file}")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        sys.exit(1)
EOF

# 运行Python脚本
echo "运行数据生成脚本..."
python ${TEMP_DIR}/daily_all.py \
    --window_size ${WINDOW_SIZE} \
    --year ${YEAR} \
    --job_id ${SLURM_JOB_ID} \
    --temp_dir ${TEMP_DIR}

# 输出结束信息
echo "========================================================="
echo "任务完成时间: $(date)"
echo "处理完成: 窗口大小=${WINDOW_SIZE}, 年份=${YEAR}"
echo "========================================================="

# 清理临时目录
rm -rf ${TEMP_DIR}