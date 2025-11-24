#!/bin/bash
#SBATCH --job-name=portfolio_R5
#SBATCH --partition=i64m512u       # CPU队列
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G                 # 增加内存以处理完整时期的数据
#SBATCH --time=48:00:00           # 增加运行时间
#SBATCH --array=0-2               # 3个任务，对应I5/I20/I60 - R5
#SBATCH --output=/hpc2hdd/home/jliu043/CNN_Replicate/logs/portfolio_R5_%A_%a.out
#SBATCH --error=/hpc2hdd/home/jliu043/CNN_Replicate/logs/portfolio_R5_%A_%a.err

# 创建日志目录
mkdir -p /hpc2hdd/home/jliu043/CNN_Replicate/logs

# 设置环境变量
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_MAX_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# 加载环境
source ~/.bashrc
conda activate cnn_env

# 设置工作目录
cd /hpc2hdd/home/jliu043/CNN_Replicate

# 根据任务ID设置输入窗口
case $SLURM_ARRAY_TASK_ID in
    0) INPUT=5 ;;
    1) INPUT=20 ;;
    2) INPUT=60 ;;
esac

# 固定预测窗口为5
PREDICT=20

# 打印任务信息
echo "========================================================="
echo "Job Array ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "模型配置: I${INPUT}/R${PREDICT}"
echo "运行节点: $(hostname)"
echo "开始时间: $(date)"
echo "========================================================="

# 运行portfolio分析
python Portfolio/portfolio_analysis.py \
    --years 2001-2019 \
    --input-windows $INPUT \
    --predict-windows $PREDICT

# 输出结束信息
echo "========================================================="
echo "任务完成时间: $(date)"
echo "I${INPUT}/R${PREDICT} Portfolio分析完成"
echo "========================================================="