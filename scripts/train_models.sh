#!/bin/bash
#SBATCH --job-name=cnn_R5_e30
#SBATCH --partition=i64m1tga800u      # GPU队列
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1                  # 1个GPU
#SBATCH --time=72:00:00              # 72小时运行时限
#SBATCH --array=0-2                  # 3个任务，对应I5/I20/I60 - R5
#SBATCH --output=/hpc2hdd/home/jliu043/CNN_Replicate/logs/cnn_R5_e30_%A_%a.out
#SBATCH --error=/hpc2hdd/home/jliu043/CNN_Replicate/logs/cnn_R5_e30_%A_%a.err

# 创建日志目录
mkdir -p /hpc2hdd/home/jliu043/CNN_Replicate/logs

# 设置环境变量
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_MAX_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONHASHSEED=0

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
PREDICT=5

# 打印任务信息
echo "========================================================="
echo "Job Array ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "模型配置: I${INPUT}/R${PREDICT}"
echo "运行节点: $(hostname)"
echo "开始时间: $(date)"
echo "GPU信息:"
nvidia-smi
echo "========================================================="

# 运行训练和测试
python scripts/train_models.py \
    --input $INPUT \
    --predict $PREDICT \
    --train-start-year 1993 \
    --train-end-year 2000 \
    --test-start-year 2001 \
    --test-end-year 2019 \
    --max-epochs 30 \
    --early-stop 5 \
    --ensemble-size 1

# 输出结束信息
echo "========================================================="
echo "任务完成时间: $(date)"
echo "I${INPUT}/R${PREDICT} 模型训练和测试完成"
echo "========================================================="