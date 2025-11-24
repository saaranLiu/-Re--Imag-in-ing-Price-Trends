# Model/cnn_model.py
import torch
import torch.nn as nn
import numpy as np

from Misc import config as cf

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)

def init_weights(m):
    """初始化模型权重"""
    if type(m) in [nn.Conv2d, nn.Conv1d]:
        nn.init.xavier_uniform_(m.weight)
    elif type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class Model(object):
    def __init__(
        self,
        ws,
        layer_number=None,
        inplanes=cf.TRUE_DATA_CNN_INPLANES,
        drop_prob=0.50,
        filter_size=None,
        stride=None,
        dilation=None,
        max_pooling=None,
        filter_size_list=None,
        stride_list=None,
        dilation_list=None,
        max_pooling_list=None,
        batch_norm=True,
        xavier=True,
        lrelu=True,
        ts1d_model=False,
        bn_loc="bn_bf_relu",
        conv_layer_chanls=None,
        regression_label=None,
    ):
        self.ws = ws
        
        # 根据窗口大小设置默认层数
        if layer_number is None:
            self.layer_number = cf.BENCHMARK_MODEL_LAYERNUM_DICT[ws]
        else:
            self.layer_number = layer_number
            
        self.inplanes = inplanes
        self.drop_prob = drop_prob
        self.filter_size_list = (
            [filter_size] * self.layer_number
            if filter_size_list is None
            else filter_size_list
        )
        self.stride_list = (
            [stride] * self.layer_number if stride_list is None else stride_list
        )
        self.max_pooling_list = (
            [max_pooling] * self.layer_number
            if max_pooling_list is None
            else max_pooling_list
        )
        self.dilation_list = (
            [dilation] * self.layer_number if dilation_list is None else dilation_list
        )
        self.batch_norm = batch_norm
        self.xavier = xavier
        self.lrelu = lrelu
        self.ts1d_model = ts1d_model
        self.bn_loc = bn_loc
        self.conv_layer_chanls = conv_layer_chanls
        self.regression_label = regression_label

        # 计算padding值
        self.padding_list = (
            [int(fs / 2) for fs in self.filter_size_list]
            if self.ts1d_model
            else [(int(fs[0] / 2), int(fs[1] / 2)) for fs in self.filter_size_list]
        )
        
        # 生成模型名称
        self.name = self._get_full_model_name()
        
        # 设置输入大小
        self.input_size = self._get_input_size()

    def _get_input_size(self):
        """获取输入大小"""
        if self.ts1d_model:
            input_size_dict = {5: (6, 5), 20: (6, 20), 60: (6, 60)}
        else:
            input_size_dict = {5: (32, 15), 20: (64, 60), 60: (96, 180)}
        return input_size_dict[self.ws]

    def init_model(self, device=None, state_dict=None):
        """初始化模型"""
        if self.ts1d_model:
            model = CNN1DModel(
                self.layer_number,
                self.input_size,
                inplanes=self.inplanes,
                drop_prob=self.drop_prob,
                filter_size_list=self.filter_size_list,
                stride_list=self.stride_list,
                padding_list=self.padding_list,
                dilation_list=self.dilation_list,
                max_pooling_list=self.max_pooling_list,
                regression_label=self.regression_label,
            )
        else:
            model = CNNModel(
                self.layer_number,
                self.input_size,
                inplanes=self.inplanes,
                drop_prob=self.drop_prob,
                filter_size_list=self.filter_size_list,
                stride_list=self.stride_list,
                padding_list=self.padding_list,
                dilation_list=self.dilation_list,
                max_pooling_list=self.max_pooling_list,
                batch_norm=self.batch_norm,
                xavier=self.xavier,
                lrelu=self.lrelu,
                bn_loc=self.bn_loc,
                conv_layer_chanls=self.conv_layer_chanls,
                regression_label=self.regression_label,
            )

        # 加载预训练权重
        if state_dict is not None:
            model.load_state_dict(state_dict)

        # 将模型移至指定设备
        if device is not None:
            model.to(device)

        return model

    def init_model_with_model_state_dict(self, model_state_dict, device=None):
        """使用模型状态字典初始化模型"""
        model = self.init_model(device=device)
        model.load_state_dict(model_state_dict)
        return model
    
    def _get_full_model_name(self):
        """生成完整的模型名称"""
        # 为1D CNN模型生成名称
        if self.ts1d_model:
            fs_st_str = ""
            for i in range(self.layer_number):
                fs, st, mp, dl = (
                    self.filter_size_list[i],
                    self.stride_list[i],
                    self.max_pooling_list[i],
                    self.dilation_list[i],
                )
                fs_st_str += f"F{fs}S{st}D{dl}MP{mp}"
                
            arch_name = f"TSD{self.ws}L{self.layer_number}{fs_st_str}"
            
            if self.conv_layer_chanls is None:
                arch_name += f"C{self.inplanes}"
                
            return arch_name
        
        # 为2D CNN模型生成名称
        fs_st_str = ""
        for i in range(self.layer_number):
            fs, st, dl, mp = (
                self.filter_size_list[i],
                self.stride_list[i],
                self.dilation_list[i],
                self.max_pooling_list[i],
            )
            fs_st_str += f"F{fs[0]}{fs[1]}S{st[0]}{st[1]}D{dl[0]}{dl[1]}MP{mp[0]}{mp[1]}"
            
        arch_name = f"D{self.ws}L{self.layer_number}{fs_st_str}"
        
        if self.conv_layer_chanls is None:
            arch_name += f"C{self.inplanes}"
            
        # 添加其他设置
        name_list = [arch_name]
        
        if self.drop_prob != 0.5:
            name_list.append(f"DROPOUT{self.drop_prob:.2f}")
        if not self.batch_norm:
            name_list.append("NoBN")
        if not self.xavier:
            name_list.append("NoXavier")
        if not self.lrelu:
            name_list.append("ReLU")
        if self.bn_loc != "bn_bf_relu":
            name_list.append(self.bn_loc)
        if self.regression_label is not None:
            name_list.append("reg_" + self.regression_label)
            
        return "-".join(name_list)


class CNNModel(nn.Module):
    """2D CNN模型"""
    def __init__(
        self,
        layer_number,
        input_size,
        inplanes=cf.TRUE_DATA_CNN_INPLANES,
        drop_prob=0.50,
        filter_size_list=[(3, 3)],
        stride_list=[(1, 1)],
        padding_list=[(1, 1)],
        dilation_list=[(1, 1)],
        max_pooling_list=[(2, 2)],
        batch_norm=True,
        xavier=True,
        lrelu=True,
        conv_layer_chanls=None,
        bn_loc="bn_bf_relu",
        regression_label=None,
    ):
        super(CNNModel, self).__init__()
        self.layer_number = layer_number
        self.input_size = input_size
        self.conv_layer_chanls = conv_layer_chanls
        
        # 初始化卷积层
        self.conv_layers = self._init_conv_layers(
            layer_number,
            inplanes,
            drop_prob,
            filter_size_list,
            stride_list,
            padding_list,
            dilation_list,
            max_pooling_list,
            batch_norm,
            lrelu,
            bn_loc,
        )
        
        # 获取卷积层flatten后的大小
        fc_size = self._get_conv_layers_flatten_size()
        
        # 初始化全连接层
        if regression_label is not None:
            self.fc = nn.Linear(fc_size, 1)  # 回归任务
        else:
            self.fc = nn.Linear(fc_size, 2)  # 分类任务
            
        # 初始化权重
        if xavier:
            self.conv_layers.apply(init_weights)
            self.fc.apply(init_weights)

    @staticmethod
    def conv_layer(
        in_chanl, 
        out_chanl, 
        lrelu=True,
        double_conv=False,
        batch_norm=True,
        bn_loc="bn_bf_relu",
        filter_size=(3, 3),
        stride=(1, 1),
        padding=1,
        dilation=1,
        max_pooling=(2, 2),
    ):
        """创建卷积层"""
        assert bn_loc in ["bn_bf_relu", "bn_af_relu", "bn_af_mp"]

        # 不使用批归一化
        if not batch_norm:
            conv = [
                nn.Conv2d(
                    in_chanl,
                    out_chanl,
                    filter_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                ),
                nn.LeakyReLU() if lrelu else nn.ReLU(),
            ]
        # 使用批归一化
        else:
            if bn_loc == "bn_bf_relu":
                # 批归一化在激活函数之前
                conv = [
                    nn.Conv2d(
                        in_chanl,
                        out_chanl,
                        filter_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                    ),
                    nn.BatchNorm2d(out_chanl),
                    nn.LeakyReLU() if lrelu else nn.ReLU(),
                ]
            elif bn_loc == "bn_af_relu":
                # 批归一化在激活函数之后
                conv = [
                    nn.Conv2d(
                        in_chanl,
                        out_chanl,
                        filter_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                    ),
                    nn.LeakyReLU() if lrelu else nn.ReLU(),
                    nn.BatchNorm2d(out_chanl),
                ]
            else:
                # 批归一化在最大池化之后
                conv = [
                    nn.Conv2d(
                        in_chanl,
                        out_chanl,
                        filter_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                    ),
                    nn.LeakyReLU() if lrelu else nn.ReLU(),
                ]

        # 是否使用双倍卷积
        layers = conv * 2 if double_conv else conv

        # 添加最大池化层
        if max_pooling != (1, 1):
            layers.append(nn.MaxPool2d(max_pooling, ceil_mode=True))

        # 在最大池化后添加批归一化
        if batch_norm and bn_loc == "bn_af_mp":
            layers.append(nn.BatchNorm2d(out_chanl))

        return nn.Sequential(*layers)

    def _init_conv_layers(
        self,
        layer_number,
        inplanes,
        drop_prob,
        filter_size_list,
        stride_list,
        padding_list,
        dilation_list,
        max_pooling_list,
        batch_norm,
        lrelu,
        bn_loc,
    ):
        """初始化卷积层"""
        # 设置每层的通道数
        if self.conv_layer_chanls is None:
            conv_layer_chanls = [inplanes * (2**i) for i in range(layer_number)]
        else:
            conv_layer_chanls = self.conv_layer_chanls
            
        layers = []
        prev_chanl = 1  # 输入通道数为1（灰度图像）
        
        # 为每一层创建卷积层
        for i, conv_chanl in enumerate(conv_layer_chanls):
            layers.append(
                self.conv_layer(
                    prev_chanl,
                    conv_chanl,
                    filter_size=filter_size_list[i],
                    stride=stride_list[i],
                    padding=padding_list[i],
                    dilation=dilation_list[i],
                    max_pooling=max_pooling_list[i],
                    batch_norm=batch_norm,
                    lrelu=lrelu,
                    bn_loc=bn_loc,
                )
            )
            prev_chanl = conv_chanl
            
        # 添加Flatten层和Dropout层
        layers.append(Flatten())
        layers.append(nn.Dropout(p=drop_prob))
        
        return nn.Sequential(*layers)

    def _get_conv_layers_flatten_size(self):
        """获取卷积层flatten后的大小"""
        dummy_input = torch.rand((1, 1, self.input_size[0], self.input_size[1]))
        x = self.conv_layers(dummy_input)
        return x.shape[1]

    def forward(self, x):
        """前向传播"""
        x = self.conv_layers(x)
        x = self.fc(x)
        return x


class CNN1DModel(nn.Module):
    """1D CNN模型"""
    def __init__(
        self,
        layer_number,
        input_size,
        inplanes=cf.TRUE_DATA_CNN_INPLANES,
        drop_prob=0.5,
        filter_size_list=[3],
        stride_list=[1],
        padding_list=[1],
        dilation_list=[1],
        max_pooling_list=[2],
        regression_label=None,
    ):
        super(CNN1DModel, self).__init__()
        self.layer_number = layer_number
        self.input_size = input_size

        # 初始化卷积层
        self.conv_layers = self._init_ts1d_conv_layers(
            layer_number,
            inplanes,
            drop_prob,
            filter_size_list,
            stride_list,
            padding_list,
            dilation_list,
            max_pooling_list,
        )
        
        # 获取卷积层flatten后的大小
        fc_size = self._get_ts1d_conv_layers_flatten_size()
        
        # 初始化全连接层
        if regression_label is not None:
            self.fc = nn.Linear(fc_size, 1)  # 回归任务
        else:
            self.fc = nn.Linear(fc_size, 2)  # 分类任务
            
        # 初始化权重
        self.conv_layers.apply(init_weights)
        self.fc.apply(init_weights)

    @staticmethod
    def conv_layer_1d(
        in_chanl,
        out_chanl,
        filter_size=3,
        stride=1,
        padding=1,
        dilation=1,
        max_pooling=2,
    ):
        """创建1D卷积层"""
        layers = [
            nn.Conv1d(
                in_chanl,
                out_chanl,
                filter_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            ),
            nn.BatchNorm1d(out_chanl),
            nn.LeakyReLU(),
            nn.MaxPool1d(max_pooling, ceil_mode=True),
        ]
        return nn.Sequential(*layers)

    def _init_ts1d_conv_layers(
        self,
        layer_number,
        inplanes,
        drop_prob,
        filter_size_list,
        stride_list,
        padding_list,
        dilation_list,
        max_pooling_list,
    ):
        """初始化1D卷积层"""
        conv_layer_chanls = [inplanes * (2**i) for i in range(layer_number)]
        layers = []
        prev_chanl = 6  # 输入通道数为6（OHLCV+MA）
        
        # 为每一层创建卷积层
        for i, conv_chanl in enumerate(conv_layer_chanls):
            layers.append(
                self.conv_layer_1d(
                    prev_chanl,
                    conv_chanl,
                    filter_size=filter_size_list[i],
                    stride=stride_list[i],
                    padding=padding_list[i],
                    dilation=dilation_list[i],
                    max_pooling=max_pooling_list[i],
                )
            )
            prev_chanl = conv_chanl
            
        # 添加Flatten层和Dropout层
        layers.append(Flatten())
        layers.append(nn.Dropout(p=drop_prob))
        
        return nn.Sequential(*layers)

    def _get_ts1d_conv_layers_flatten_size(self):
        """获取1D卷积层flatten后的大小"""
        dummy_input = torch.rand((1, self.input_size[0], self.input_size[1]))
        x = self.conv_layers(dummy_input)
        return x.shape[1]

    def forward(self, x):
        """前向传播"""
        x = self.conv_layers(x)
        x = self.fc(x)
        return x