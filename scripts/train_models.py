# scripts/train_models.py
import os
import sys
import torch
import pandas as pd
import numpy as np
from torch import nn
import torch.optim as optim
from torch.backends import cudnn
from torch.utils.data import DataLoader, ConcatDataset, random_split
import time
import copy
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, "/hpc2hdd/home/jliu043/CNN_Replicate")

from Model.cnn_model import Model, CNNModel, CNN1DModel
from Misc import config as cf
from Data import dgp_config as dcf
from Misc import utilities as ut
from Data.chart_dataset import EquityDataset
from Portfolio.portfolio import PortfolioManager

class CNNExperiment:
    def __init__(
        self,
        input_window,  # 输入窗口大小 (5, 20, 60)
        predict_window,  # 预测窗口大小 (5, 20, 60)
        device_number=0,
        ensemble_size=5,
        learning_rate=1e-4,
        dropout=0.5,
        batch_norm=True,
        has_volume_bar=True,
        has_ma=True,
        train_years=list(range(1993, 2001)),  # 训练年份:1993-2000
        test_years=list(range(2001, 2020)),   # 测试年份:2001-2019
        chart_type="bar",
        train_size_ratio=0.7,                 # 训练集比例:70%
        max_epochs=30,
        early_stop=True,
    ):
        self.input_window = input_window
        self.predict_window = predict_window
        self.ensemble_size = ensemble_size
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.device_number = device_number
        self.device = torch.device(f"cuda:{device_number}" if torch.cuda.is_available() else "cpu")
        self.batch_norm = batch_norm
        self.has_volume_bar = has_volume_bar
        self.has_ma = has_ma
        self.train_years = train_years
        self.test_years = test_years
        self.chart_type = chart_type
        self.train_size_ratio = train_size_ratio
        self.max_epochs = max_epochs
        self.early_stop = early_stop
        
        # 设置模型相关参数
        self.layer_number = cf.BENCHMARK_MODEL_LAYERNUM_DICT[input_window]
        self.inplanes = cf.TRUE_DATA_CNN_INPLANES
        
        # 获取CNN设置
        self.filter_size_list, self.stride_list, self.dilation_list, self.max_pooling_list = cf.EMP_CNN_BL_SETTING[input_window]
        
        # 确定训练频率
        self.train_freq = dcf.FREQ_DICT[predict_window]
        
        # 创建模型
        self.model_obj = self._create_model()
        
        # 创建目录
        self.model_name = self.model_obj.name
        self.exp_name = self._get_exp_name()
        self.model_dir = ut.get_dir(os.path.join(cf.EXP_DIR, self.model_name, self.exp_name))
        self.portfolio_dir = ut.get_dir(os.path.join(cf.PORTFOLIO_DIR, f"USA_{self.model_name}_{self.exp_name}_ensem{self.ensemble_size}"))
        self.ensemble_res_dir = ut.get_dir(os.path.join(self.model_dir, "ensem_res"))
        
        print(f"Model will be saved to: {self.model_dir}")
        print(f"Portfolio results will be saved to: {self.portfolio_dir}")
    
    def _create_model(self):
        """创建模型对象"""
        model = Model(
            self.input_window,
            layer_number=self.layer_number,
            inplanes=self.inplanes,
            drop_prob=self.dropout,
            filter_size_list=self.filter_size_list,
            stride_list=self.stride_list,
            dilation_list=self.dilation_list,
            max_pooling_list=self.max_pooling_list,
            batch_norm=self.batch_norm,
            ts1d_model=False,  # 使用CNN2D
        )
        return model
    
    def _get_exp_name(self):
        """生成实验名称"""
        exp_setting_list = [
            f"{self.input_window}d{self.predict_window}p-lr{self.learning_rate:.0E}-dp{self.dropout:.2f}",
            f"ma{self.has_ma}-vb{self.has_volume_bar}-{self.train_freq}lyTrained",
            "noDelayedReturn"  # 不使用延迟收益率
        ]
        if not self.batch_norm:
            exp_setting_list.append("noBN")
        if self.chart_type != "bar":
            exp_setting_list.append(self.chart_type)
        
        return "-".join(exp_setting_list)
    
    def get_model_checkpoint_path(self, model_num):
        """获取模型检查点路径"""
        return os.path.join(self.model_dir, f"checkpoint{model_num}.pth.tar")
    
    def get_train_validate_dataloaders(self):
        """
        准备训练和验证数据加载器
        按照论文要求: 使用1993-2000年数据，随机分配70%用于训练，30%用于验证
        """
        datasets = [EquityDataset(
            window_size=self.input_window,
            predict_window=self.predict_window,
            freq=self.train_freq,
            year=year,
            country="USA",
            has_volume_bar=self.has_volume_bar,
            has_ma=self.has_ma,
            chart_type=self.chart_type,
            remove_tail=(year == self.test_years[0] - 1),
        ) for year in self.train_years]
        
        combined_dataset = ConcatDataset(datasets)
        train_size = int(len(combined_dataset) * self.train_size_ratio)
        validate_size = len(combined_dataset) - train_size
        
        print(f"总样本数: {len(combined_dataset)}")
        print(f"训练集: {train_size}样本 ({self.train_size_ratio*100:.0f}%)")
        print(f"验证集: {validate_size}样本 ({(1-self.train_size_ratio)*100:.0f}%)")
        
        # 使用随机分割，确保样本随机性
        train_dataset, validate_dataset = random_split(
            combined_dataset,
            [train_size, validate_size],
            generator=torch.Generator().manual_seed(42)  # 设置随机种子以确保可重复性
        )
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=cf.BATCH_SIZE,
            shuffle=True,
            num_workers=cf.NUM_WORKERS,
        )
        
        validate_dataloader = DataLoader(
            validate_dataset,
            batch_size=cf.BATCH_SIZE,
            num_workers=cf.NUM_WORKERS,
        )
        
        return {"train": train_dataloader, "validate": validate_dataloader}
    
    def _update_metrics(self, loss, labels, preds, metrics):
        """更新度量指标"""
        metrics["running_loss"] += loss.item() * len(labels)
        metrics["running_correct"] += (preds == labels).sum().item()
        metrics["TP"] += (preds * labels).sum().item()
        metrics["TN"] += ((preds - 1) * (labels - 1)).sum().item()
        metrics["FP"] += (preds * (labels - 1)).sum().abs().item()
        metrics["FN"] += ((preds - 1) * labels).sum().abs().item()
    
    def _generate_epoch_stats(self, epoch, lr, num_samples, metrics):
        """生成每个epoch的统计信息"""
        TP, TN, FP, FN = float(metrics["TP"]), float(metrics["TN"]), float(metrics["FP"]), float(metrics["FN"])
        stats = {"epoch": epoch, "lr": f"{lr:.2E}"}
        stats["diff"] = 1.0 * ((TP + FP) - (TN + FN)) / num_samples
        stats["loss"] = metrics["running_loss"] / num_samples
        stats["accy"] = 1.0 * metrics["running_correct"] / num_samples
        
        # 计算MCC (Matthews Correlation Coefficient)
        # 将所有值转换为float类型进行计算
        denom = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        stats["MCC"] = np.nan if denom == 0 else 1.0 * (TP * TN - FP * FN) / denom
        
        return stats
    
    def train_single_model(self, model_num):
        """训练单个模型"""
        model_path = self.generate_model_path()
        checkpoint_path = os.path.join(model_path, f"checkpoint{model_num}.pth.tar")
        
        # 检查是否存在具有相同配置的旧模型
        if os.path.exists(checkpoint_path):
            # 检查模型配置
            checkpoint = torch.load(checkpoint_path)
            old_epochs = checkpoint.get('max_epochs', 0)
            old_early_stop = checkpoint.get('early_stop_patience', 0)
            
            if old_epochs == self.max_epochs and old_early_stop == self.early_stop:
                print(f"模型已存在于 {checkpoint_path}，且具有相同的训练配置")
                return True
            else:
                print(f"发现旧模型，但训练配置不同 (epochs: {old_epochs}, early_stop: {old_early_stop})")
                print(f"使用新配置 (epochs: {self.max_epochs}, early_stop: {self.early_stop}) 重新训练")
                os.remove(checkpoint_path)  # 删除旧模型
        
        print(f"训练模型 {model_num+1}/{self.ensemble_size} (I{self.input_window}/R{self.predict_window})")
        
        dataloaders_dict = self.get_train_validate_dataloaders()
        
        # 初始化模型
        model = self.model_obj.init_model(device=self.device)
        optimizer = optim.Adam(
            model.parameters(), 
            lr=self.learning_rate,
            weight_decay=0
        )
        
        # 打印模型信息
        print(f"训练设备: {self.device}")
        print(model)
        
        # 启用cuDNN基准测试以加速
        cudnn.benchmark = True
        
        # 初始化变量
        best_validate_metrics = {"loss": 10.0, "accy": 0.0, "MCC": 0.0, "epoch": 0}
        best_model = copy.deepcopy(model.state_dict())
        
        # 记录开始时间
        since = time.time()
        
        # 开始训练循环
        for epoch in range(self.max_epochs):
            for phase in ["train", "validate"]:
                if phase == "train":
                    model.train()
                else:
                    model.eval()
                
                running_metrics = {
                    "running_loss": 0.0,
                    "running_correct": 0.0,
                    "TP": 0,
                    "TN": 0,
                    "FP": 0,
                    "FN": 0,
                }
                
                # 使用tqdm显示进度条
                data_iterator = tqdm(
                    dataloaders_dict[phase],
                    desc=f"Epoch {epoch}/{self.max_epochs-1} ({phase})",
                    leave=False
                )
                
                for batch in data_iterator:
                    inputs = batch["image"].to(self.device, dtype=torch.float)
                    labels = batch["label"].to(self.device, dtype=torch.long)
                    
                    # 零梯度
                    optimizer.zero_grad()
                    
                    # 前向传播
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        loss = nn.CrossEntropyLoss()(outputs, labels)
                        _, preds = torch.max(outputs, 1)
                        
                        # 如果是训练阶段，反向传播和优化
                        if phase == "train":
                            loss.backward()
                            optimizer.step()
                    
                    # 更新指标
                    self._update_metrics(loss, labels, preds, running_metrics)
                    
                    # 释放内存
                    del inputs, labels, outputs, preds
                
                # 计算当前epoch的统计信息
                num_samples = len(dataloaders_dict[phase].dataset)
                epoch_stats = self._generate_epoch_stats(epoch, self.learning_rate, num_samples, running_metrics)
                
                # 打印统计信息
                print(f"Epoch {epoch}/{self.max_epochs-1} ({phase}): Loss: {epoch_stats['loss']:.4f}, Acc: {epoch_stats['accy']:.4f}, MCC: {epoch_stats['MCC']:.4f}")
                
                # 如果是验证阶段并且当前模型更好，则保存
                if phase == "validate" and epoch_stats["loss"] < best_validate_metrics["loss"]:
                    for metric in ["loss", "accy", "MCC", "epoch"]:
                        best_validate_metrics[metric] = epoch_stats[metric]
                    best_model = copy.deepcopy(model.state_dict())
                    print(f"Epoch {epoch}: 发现更好的模型，验证损失: {epoch_stats['loss']:.4f}")
            
            # 早停策略：如果连续2个epoch没有改善，则停止训练
            if self.early_stop and (epoch - best_validate_metrics["epoch"]) >= 2:
                print(f"Early stopping at epoch {epoch}, best epoch was {best_validate_metrics['epoch']}")
                break
        
        # 计算训练时间
        time_elapsed = time.time() - since
        print(f"训练完成，用时 {time_elapsed//60:.0f}分 {time_elapsed%60:.0f}秒")
        print(f"最佳验证损失: {best_validate_metrics['loss']:.4f} (Epoch {best_validate_metrics['epoch']})")
        
        # 加载最佳模型
        model.load_state_dict(best_model)
        
        # 保存模型
        best_validate_metrics["model_state_dict"] = model.state_dict()
        torch.save(best_validate_metrics, checkpoint_path)
        print(f"模型已保存到 {checkpoint_path}")
        
        # 释放内存
        del model, dataloaders_dict
        torch.cuda.empty_cache()
        
        return True
    
    def train_ensemble_models(self):
        """训练集成模型（5个独立训练的CNN模型）"""
        print(f"开始训练 I{self.input_window}/R{self.predict_window} 的 {self.ensemble_size} 个模型...")
        success = True
        
        for model_num in range(self.ensemble_size):
            result = self.train_single_model(model_num)
            success = success and result
        
        if success:
            print(f"I{self.input_window}/R{self.predict_window} 模型集成训练完成")
        else:
            print(f"I{self.input_window}/R{self.predict_window} 部分模型训练失败")
        
        return success
    
    def _get_dataloader_for_year(self, year):
        """获取特定年份的数据加载器（用于测试）"""
        dataset = EquityDataset(
            window_size=self.input_window,
            predict_window=self.predict_window,
            freq=self.train_freq,
            year=year,
            country="USA",
            has_volume_bar=self.has_volume_bar,
            has_ma=self.has_ma,
            chart_type=self.chart_type,
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=cf.BATCH_SIZE,
            shuffle=False,
            num_workers=cf.NUM_WORKERS
        )
        
        return dataloader
    
    def get_model_output_path(self):
        """获取模型输出路径"""
        # 基础目录
        base_dir = "/hpc2hdd/home/jliu043/CNN_Replicate/results"
        
        # 模型名称格式：CNN_I{input}R{predict}
        model_name = f"CNN_I{self.input_window}R{self.predict_window}"
        
        # 模型配置信息
        config_str = (
            f"epoch{self.max_epochs}_"
            f"ensem{self.ensemble_size}_"
            f"train{self.train_years[0]}-{self.train_years[-1]}"
        )
        
        # 完整路径
        model_dir = os.path.join(base_dir, model_name, config_str)
        
        return model_dir
    
    def generate_predictions_for_year(self, year):
        """生成预测结果"""
        # 获取模型输出目录
        output_dir = os.path.join(self.get_model_output_path(), "predictions")
        os.makedirs(output_dir, exist_ok=True)
        
        # 预测文件命名：pred_year{year}.csv
        output_file = os.path.join(output_dir, f"pred_year{year}.csv")
        
        # 检查输出文件是否已存在
        if os.path.exists(output_file):
            print(f"预测结果已存在: {output_file}")
            return output_file
        
        # 加载测试数据
        dataloader = self._get_dataloader_for_year(year)
        
        # 加载所有模型
        models = []
        for model_num in range(self.ensemble_size):
            model_path = self.get_model_checkpoint_path(model_num)
            if not os.path.exists(model_path):
                print(f"错误: 模型 {model_path} 不存在")
                return None
            
            model = self.model_obj.init_model(device=self.device)
            model.load_state_dict(torch.load(model_path, map_location=self.device)["model_state_dict"])
            model.eval()
            models.append(model)
        
        # 生成预测
        print(f"使用 {len(models)} 个模型进行集成预测...")
        results = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"预测 {year} 年数据"):
                inputs = batch["image"].to(self.device, dtype=torch.float)
                
                # 集成预测
                total_probs = torch.zeros((inputs.size(0), 2), device=self.device)
                for model in models:
                    outputs = model(inputs)
                    probs = torch.softmax(outputs, dim=1)
                    total_probs += probs
                
                # 计算平均概率
                avg_probs = total_probs / len(models)
                
                # 收集结果
                for i in range(inputs.size(0)):
                    results.append({
                        "StockID": batch["StockID"][i],
                        "Date": batch["ending_date"][i],
                        "up_prob": avg_probs[i, 1].item(),  # 上涨概率
                        "ret_val": batch["ret_val"][i].item(),
                        "MarketCap": batch["MarketCap"][i].item() if "MarketCap" in batch else 0,
                    })
        
        # 保存结果
        df = pd.DataFrame(results)
        df.to_csv(output_file)
        print(f"预测结果已保存到 {output_file}")
        
        # 释放内存
        del models
        torch.cuda.empty_cache()
        
        return output_file
    
    def generate_all_predictions(self):
        """为所有测试年份生成预测"""
        print(f"为 I{self.input_window}/R{self.predict_window} 生成所有测试年份的预测...")
        
        all_files = []
        for year in self.test_years:
            file = self.generate_predictions_for_year(year)
            if file:
                all_files.append(file)
        
        return all_files
    
    def run_portfolio_analysis(self):
        """运行投资组合分析"""
        print(f"为 I{self.input_window}/R{self.predict_window} 运行投资组合分析...")
        
        # 首先确保所有预测已生成
        prediction_files = self.generate_all_predictions()
        if not prediction_files:
            print("无法进行投资组合分析：缺少预测结果")
            return False
        
        # 加载所有预测结果
        dfs = []
        for file in prediction_files:
            try:
                df = pd.read_csv(file, parse_dates=["Date"], index_col=0)
                if not df.empty:
                    dfs.append(df)
                    print(f"成功加载预测文件: {file}, 数据量: {len(df)}")
                else:
                    print(f"警告: 预测文件为空: {file}")
            except Exception as e:
                print(f"加载预测文件失败: {file}")
                print(f"错误: {str(e)}")
        
        if not dfs:
            print("没有有效的预测数据，无法进行投资组合分析")
            return False
        
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df.set_index(["Date", "StockID"], inplace=True)
        
        print(f"合并后的预测数据大小: {combined_df.shape}")
        print(f"日期范围: {combined_df.index.get_level_values('Date').min()} 到 {combined_df.index.get_level_values('Date').max()}")
        
        # 创建投资组合管理器
        portfolio_manager = PortfolioManager(
            combined_df,
            freq=self.train_freq,
            portfolio_dir=self.portfolio_dir,
            country="USA",
            start_year=min(self.test_years),
            end_year=max(self.test_years)
        )
        
        # 生成投资组合
        portfolio_manager.generate_portfolio(cut=10, delay=0)
        
        print(f"I{self.input_window}/R{self.predict_window} 投资组合分析完成")
        return True

    def generate_model_path(self):
        """生成模型保存路径"""
        # 基础模型架构标识
        model_arch = f"D{self.input_window}L{self.layer_number}F{self.inplanes}S{self.stride_list}..."
        
        # 训练参数标识
        train_params = (f"{self.input_window}d{self.predict_window}p"
                       f"-e{self.max_epochs}"  # 添加 epoch 信息
                       f"-es{self.early_stop}"  # 添加 early stopping 信息
                       f"-lr{self.learning_rate}"
                       f"-dp{self.dropout}"
                       f"-ma{self.has_ma}"
                       f"-vb{self.has_volume_bar}"
                       f"-{self.train_freq}lyTrained"
                       f"-noDelayedReturn")
        
        return os.path.join(
            self.model_dir,
            model_arch,
            train_params
        )

    def generate_portfolio_path(self):
        """生成 portfolio 结果保存路径"""
        model_arch = f"D{self.input_window}L{self.layer_number}F{self.inplanes}S{self.stride_list}..."
        
        portfolio_name = (f"USA_{model_arch}"
                         f"_{self.input_window}d{self.predict_window}p"
                         f"-e{self.max_epochs}"  # 添加 epoch 信息
                         f"-es{self.early_stop}"  # 添加 early stopping 信息
                         f"-lr{self.learning_rate}"
                         f"-dp{self.dropout}"
                         f"-ma{self.has_ma}"
                         f"-vb{self.has_volume_bar}"
                         f"-{self.train_freq}lyTrained"
                         f"-noDelayedReturn"
                         f"_ensem{self.ensemble_size}")
        
        return os.path.join(
            self.portfolio_dir,
            portfolio_name
        )


def train_models():
    """训练所有9个模型组合"""
    # 定义输入和预测窗口组合
    input_windows = [5, 20, 60]
    predict_windows = [5, 20, 60]
    
    # 为每个组合创建实验
    experiments = []
    for input_window in input_windows:
        for predict_window in predict_windows:
            # 对于较大窗口使用较小学习率
            lr = 1e-5 if input_window == 60 or predict_window == 60 else 1e-4
            
            exp = CNNExperiment(
                input_window=input_window,
                predict_window=predict_window,
                learning_rate=lr,
                device_number=0  # 使用第一个GPU
            )
            experiments.append(exp)
    
    # 训练所有模型
    for exp in experiments:
        print(f"\n{'='*80}")
        print(f"训练 I{exp.input_window}/R{exp.predict_window} 模型")
        print(f"{'='*80}\n")
        
        exp.train_ensemble_models()
    
    # 进行投资组合分析
    for exp in experiments:
        print(f"\n{'='*80}")
        print(f"为 I{exp.input_window}/R{exp.predict_window} 进行投资组合分析")
        print(f"{'='*80}\n")
        
        exp.run_portfolio_analysis()


def test_small_sample():
    """在小样本上测试模型"""
    # 使用小样本测试I5/R5模型
    exp = CNNExperiment(
        input_window=5,
        predict_window=5,
        ensemble_size=1,  # 只训练2个模型以节省时间
        learning_rate=1e-4,
        train_years=[2000],  # 仅使用2000年数据训练
        test_years=[2001],   # 仅使用2001年数据测试
        device_number=0
    )
    
    print("\n测试小样本训练...\n")
    success = exp.train_ensemble_models()
    
    if success:
        print("\n测试小样本预测...\n")
        prediction_file = exp.generate_predictions_for_year(2001)
        
        if prediction_file:
            print("\n测试小样本投资组合分析...\n")
            # 修改 Portfolio 分析的年份范围
            exp.portfolio_dir = ut.get_dir(os.path.join(cf.PORTFOLIO_DIR, "test_portfolio"))
            exp.run_portfolio_analysis()
    
    return success


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CNN股票预测模型训练与测试")
    
    # 模型配置参数
    parser.add_argument("--input", type=int, choices=[5, 20, 60], required=True,
                      help="Input window size")
    parser.add_argument("--predict", type=int, choices=[5, 20, 60], required=True,
                      help="Prediction window size")
    parser.add_argument("--max-epochs", type=int, default=30,
                      help="Maximum number of epochs")
    parser.add_argument("--early-stop", type=int, default=5,
                      help="Early stopping patience")
    parser.add_argument("--ensemble-size", type=int, default=1,
                      help="Number of models in ensemble")
    
    # 训练参数
    parser.add_argument("--train-start-year", type=int, default=1993,
                      help="训练起始年份")
    parser.add_argument("--train-end-year", type=int, default=2000,
                      help="训练结束年份")
    parser.add_argument("--test-start-year", type=int, default=2001,
                      help="测试起始年份")
    parser.add_argument("--test-end-year", type=int, default=2019,
                      help="测试结束年份")
    
    args = parser.parse_args()
    
    # 创建实验实例
    exp = CNNExperiment(
        input_window=args.input,
        predict_window=args.predict,
        ensemble_size=args.ensemble_size,
        train_years=list(range(args.train_start_year, args.train_end_year + 1)),
        test_years=list(range(args.test_start_year, args.test_end_year + 1)),
        max_epochs=args.max_epochs
    )
    
    # 训练模型
    print(f"\n开始训练 I{args.input}/R{args.predict} 模型...")
    success = exp.train_ensemble_models()
    
    if success:
        # 生成所有年份的预测
        print(f"\n生成 {args.test_start_year}-{args.test_end_year} 预测结果...")
        for year in range(args.test_start_year, args.test_end_year + 1):
            exp.generate_predictions_for_year(year)