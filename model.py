import torch
from torch import nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        # Initialize weights with normal distribution
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ################################ 1. TCN_GMP #############################



class TCN_GMP(nn.Module):
    def __init__(self, input_dim, num_classes, num_channels=[64, 128], kernel_size=3, dropout=0.1):
        super(TCN_GMP, self).__init__()
        self.tcn = TemporalConvNet(input_dim, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.fc1 = nn.Linear(num_channels[-1], num_channels[-1])
        self.fc2 = nn.Linear(num_channels[-1], num_classes)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(num_channels[-1])

    def forward(self, x, lengths=None):
        x = x.permute(0, 2, 1)  # (batch_size, input_dim, seq_length)

        # Pass through the TCN layers
        out = self.tcn(x)

        # Pooling over time dimension
        out = torch.mean(out, dim=-1)  # Mean pooling across the time dimension

        # Apply dropout and normalization
        out = self.layer_norm(out)
        out = self.dropout(out)

        # Pass through fully connected layers
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)

        return out

# ############################### 2. TCN_GA ##############################
class TCNWithGlobalAttention(nn.Module):
    def __init__(self, input_dim, num_classes, num_channels=[64, 128], kernel_size=3, dropout=0.1):
        super(TCNWithGlobalAttention, self).__init__()
        self.tcn = TemporalConvNet(input_dim, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.fc1 = nn.Linear(num_channels[-1], num_channels[-1])
        self.fc2 = nn.Linear(num_channels[-1], num_classes)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(num_channels[-1])

    def forward(self, x, lengths=None):
        x = x.permute(0, 2, 1)  # (batch_size, input_dim, seq_length)
        out = self.tcn(x)  # TCN层输出，(batch_size, num_channels, seq_length)

        # 全局注意力机制
        attn_weights = torch.mean(out, dim=1)  # (batch_size, seq_length)
        attn_weights = torch.softmax(attn_weights, dim=-1)  # (batch_size, seq_length)

        # 调整形状以进行矩阵乘法
        attn_weights = attn_weights.unsqueeze(1)  # (batch_size, 1, seq_length)

        # 加权输出
        out = torch.bmm(attn_weights, out.transpose(1, 2))  # (batch_size, 1, num_channels)
        out = out.squeeze(1)  # (batch_size, num_channels)

        # Apply dropout and normalization
        out = self.layer_norm(out)
        out = self.dropout(out)

        # Pass through fully connected layers
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)

        return out

#################################### 3.TCN_ChannelAttention ################################################
class TCNWithChannelAttention(nn.Module):
    def __init__(self, input_dim, num_classes, num_channels=[64, 128], kernel_size=3, dropout=0.1, reduction_ratio=16):
        super(TCNWithChannelAttention, self).__init__()
        self.tcn = TemporalConvNet(input_dim, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.fc1 = nn.Linear(num_channels[-1], num_channels[-1])
        self.fc2 = nn.Linear(num_channels[-1], num_classes)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(num_channels[-1])

        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc_se = nn.Sequential(
            nn.Linear(num_channels[-1], num_channels[-1] // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels[-1] // reduction_ratio, num_channels[-1], bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths=None):
        x = x.permute(0, 2, 1)  # (batch_size, input_dim, seq_length)
        out = self.tcn(x)  # TCN层输出，(batch_size, num_channels, seq_length)

        # 通道注意力机制
        avg_out = self.fc_se(self.avg_pool(out).squeeze(-1))
        max_out = self.fc_se(self.max_pool(out).squeeze(-1))
        channel_attention = self.sigmoid(avg_out + max_out).unsqueeze(-1)  # (batch_size, num_channels, 1)

        # 加权输出
        out = out * channel_attention.expand_as(out)  # (batch_size, num_channels, seq_length)

        # 全局池化或者取最后一个时间步作为分类器输入
        out = out.mean(dim=-1)  # 或者 out[:, :, -1]

        # Apply dropout and normalization
        out = self.layer_norm(out)
        out = self.dropout(out)

        # Pass through fully connected layers
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)

        return out


###################################### 4.TCN_CSMA ############################################
class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)

    def forward(self, query, key, value):
        # query, key, and value should have shape (seq_len, batch_size, embed_dim)
        out, _ = self.multihead_attn(query=query, key=key, value=value)
        return out

class TCNWithCrossAttention(nn.Module):
    def __init__(self, input_dim, num_classes, num_channels=[64, 128], kernel_size=3, dropout=0.1, reduction_ratio=16):
        super(TCNWithCrossAttention, self).__init__()
        self.tcn = TemporalConvNet(input_dim, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.fc1 = nn.Linear(num_channels[-1], num_channels[-1])
        self.fc2 = nn.Linear(num_channels[-1], num_classes)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(num_channels[-1])

        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc_se = nn.Sequential(
            nn.Linear(num_channels[-1], num_channels[-1] // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels[-1] // reduction_ratio, num_channels[-1], bias=False),
        )
        self.sigmoid = nn.Sigmoid()

        # Cross-Attention
        self.cross_attention = CrossAttention(embed_dim=num_channels[-1])

    def forward(self, x, lengths=None):
        x = x.permute(0, 2, 1)  # (batch_size, input_dim, seq_length)
        out = self.tcn(x)  # (batch_size, num_channels, seq_length)

        # Time Attention
        time_attn_weights = torch.mean(out, dim=1)  # (batch_size, seq_length)
        time_attn_weights = torch.softmax(time_attn_weights, dim=-1).unsqueeze(1)  # (batch_size, 1, seq_length)
        time_weighted_out = torch.bmm(time_attn_weights, out.transpose(1, 2)).squeeze(1)  # (batch_size, num_channels)

        # Channel Attention
        avg_out = self.fc_se(self.avg_pool(out).squeeze(-1))
        max_out = self.fc_se(self.max_pool(out).squeeze(-1))
        channel_attention = self.sigmoid(avg_out + max_out).unsqueeze(-1)  # (batch_size, num_channels, 1)
        channel_weighted_out = out * channel_attention.expand_as(out)  # (batch_size, num_channels, seq_length)
        channel_weighted_out = channel_weighted_out.mean(dim=-1)  # (batch_size, num_channels)

        # Prepare for Cross-Attention: Transpose to (seq_len, batch_size, embed_dim)
        time_weighted_out = time_weighted_out.unsqueeze(0)  # (1, batch_size, num_channels)
        channel_weighted_out = channel_weighted_out.unsqueeze(0)  # (1, batch_size, num_channels)

        # Cross-Attention
        cross_attended_out = self.cross_attention(query=time_weighted_out, key=channel_weighted_out, value=channel_weighted_out)
        cross_attended_out = cross_attended_out.squeeze(0)  # (batch_size, num_channels)

        # Apply dropout and normalization
        out1 = self.layer_norm(cross_attended_out)
        out1 = self.dropout(out1)

        # Pass through fully connected layers
        out1 = F.relu(self.fc1(out1))
        out1 = self.dropout(out1)
        out1 = self.fc2(out1)

        return out1


#################################### 5.TCN-GA-EfficientNet ######################
class EfficientNet_FeatureExtractor(nn.Module):
    def __init__(self,hidden_dim=128, dropout=0.1):
        super(EfficientNet_FeatureExtractor, self).__init__()
        # 加载预训练的EfficientNet-B0模型
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')

        # 替换分类头
        in_features = self.efficientnet._fc.in_features
        self.efficientnet._fc = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, images):
        out = self.efficientnet(images)
        return out

class TCNWithGlobalAttention_EfficientNet(nn.Module):
    def __init__(self,
        input_dim,
        num_classes,
        num_channels=[64, 128],
        kernel_size=3,
        dropout=0.1,
        hidden_dim=128,  # EfficientNet 特征维度
        pretrained=True  # 是否使用预训练权重
    ):
        super(TCNWithGlobalAttention_EfficientNet, self).__init__()

        # 序列数据分支 (TCN)
        self.tcn = TemporalConvNet(
            input_dim,
            num_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )
        self.fc1 = nn.Linear(num_channels[-1], num_channels[-1])
        self.fc2 = nn.Linear(num_channels[-1] + hidden_dim, num_classes)  # 拼接后的维度变化
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(num_channels[-1])

        # 图像数据分支 (EfficientNet)
        self.efficientnet_extractor = EfficientNet_FeatureExtractor(
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        # 可选：冻结 EfficientNet 参数
        for param in self.efficientnet_extractor.efficientnet.parameters():
            param.requires_grad = False

    def forward(self, x, lengths=None, images=None):
        # ----------------------------
        # 处理序列数据
        # ----------------------------
        x_seq = x.permute(0, 2, 1)  # (batch_size, input_dim, seq_length)
        out = self.tcn(x_seq)  # (batch_size, num_channels[-1], seq_length)

        # 全局注意力机制
        attn_weights = torch.mean(out, dim=1)  # (batch_size, seq_length)
        attn_weights = torch.softmax(attn_weights, dim=-1)  # (batch_size, seq_length)
        attn_weights = attn_weights.unsqueeze(1)  # (batch_size, 1, seq_length)

        # 加权输出
        out = torch.bmm(attn_weights, out.transpose(1, 2))  # (batch_size, 1, num_channels[-1])
        out = out.squeeze(1)  # (batch_size, num_channels[-1])

        # 全连接层
        out = self.layer_norm(out)
        out = self.dropout(out)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)

        # 处理图像数据
        if images is not None:
            img_features = self.efficientnet_extractor(images)  # (batch_size, hidden_dim)

        # 拼接序列特征和图像特征
        combined_features = torch.cat((out, img_features), dim=1)  # (batch_size, num_channels[-1] + hidden_dim)

        # 最终分类
        out = self.fc2(combined_features)  # (batch_size, num_classes)

        return out

################################## 6.   #########################################
class CrossModalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(CrossModalAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.ffn_dropout = nn.Dropout(dropout)
        self.ffn_layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, query, key, value, key_padding_mask=None):
        # query: (seq_len, batch, embed_dim)
        # key, value: (seq_len, batch, embed_dim)
        attn_output, attn_weights = self.multihead_attn(query, key, value, key_padding_mask=key_padding_mask)
        attn_output = self.dropout(attn_output)
        attn_output = self.layer_norm(attn_output + query)  # Residual connection

        # Feed Forward Network
        ffn_output = self.ffn(attn_output)
        ffn_output = self.ffn_dropout(ffn_output)
        ffn_output = self.ffn_layer_norm(ffn_output + attn_output)  # Residual connection

        return ffn_output, attn_weights

class TCNWithGlobalAttention_EfficientNet_Select(nn.Module):
    def __init__(self,
        input_dim,
        num_classes,
        num_channels=[64, 128],
        kernel_size=3,
        dropout=0.1,
        hidden_dim=128,  # EfficientNet 特征维度
        pretrained=True  # 是否使用预训练权重
    ):
        super(TCNWithGlobalAttention_EfficientNet_Select, self).__init__()

        # 序列数据分支 (TCN)
        self.tcn = TemporalConvNet(
            input_dim,
            num_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )
        self.fc1 = nn.Linear(num_channels[-1], num_channels[-1])
        self.fc2 = nn.Linear(num_channels[-1] + hidden_dim, num_classes)  # 拼接后的维度变化

        self.fc2_single = nn.Linear(num_channels[-1], num_classes)  # 只用TCN时的输出

        self.fc2_image = nn.Linear(hidden_dim, num_classes)  # 只用图像时的输出

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(num_channels[-1])

        # 图像数据分支 (EfficientNet)
        self.efficientnet_extractor = EfficientNet_FeatureExtractor(
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        # 可选：冻结 EfficientNet 参数
        for param in self.efficientnet_extractor.efficientnet.parameters():
            param.requires_grad = False

        # 交叉模态注意力
        self.cross_modal_attention = CrossModalAttention(embed_dim=128, num_heads=8, dropout=dropout)

    def forward(self, x=None, lengths=None, images=None, use_images=True, use_ais=True):
        # ----------------------------
        # 处理序列数据 (AIS)
        # ----------------------------
        if use_ais and x is not None:
            x_seq = x.permute(0, 2, 1)  # (batch_size, input_dim, seq_length)
            out = self.tcn(x_seq)  # (batch_size, num_channels[-1], seq_length)

            # 全局注意力机制
            attn_weights = torch.mean(out, dim=1)  # (batch_size, seq_length)
            attn_weights = torch.softmax(attn_weights, dim=-1)  # (batch_size, seq_length)
            attn_weights = attn_weights.unsqueeze(1)  # (batch_size, 1, seq_length)

            # 加权输出
            out = torch.bmm(attn_weights, out.transpose(1, 2))  # (batch_size, 1, num_channels[-1])
            out = out.squeeze(1)  # (batch_size, num_channels[-1])

            # 全连接层
            out = self.layer_norm(out)
            out = self.dropout(out)
            out = F.relu(self.fc1(out))
            out = self.dropout(out)
        else:
            out = None

        # ----------------------------
        # 处理图像数据
        # ----------------------------
        if use_images and images is not None:
            img_features = self.efficientnet_extractor(images)  # (batch_size, hidden_dim)
        else:
            img_features = None

        # ----------------------------
        # 根据分支选择最终输出
        # ----------------------------
        if use_ais and out is not None and use_images and img_features is not None:
            # 拼接序列特征和图像特征
            combined_features = torch.cat((out, img_features), dim=1)  # (batch_size, num_channels[-1] + hidden_dim)
            out = self.fc2(combined_features)  # 最终分类

            # # 交叉注意力机制
            # query = out.unsqueeze(0)  # (1, batch_size, embed_dim)
            # key = img_features.unsqueeze(0)  # (1, batch_size, embed_dim)
            # value = img_features.unsqueeze(0)  # (1, batch_size, embed_dim)
            #
            # # 应用交叉模态注意力
            # fused_features, attn_weights = self.cross_modal_attention(query, key, value)  # (1, batch_size, embed_dim)
            #
            # # 去除时间步长维度
            # fused_features = fused_features.squeeze(0)  # (batch_size, embed_dim)
            # out = self.fc2_single(fused_features)
        elif use_ais and out is not None:
            # 只使用AIS时序数据
            out = self.fc2_single(out)  # (batch_size, num_classes)
        elif use_images and img_features is not None:
            # 只使用图像数据
            out = self.fc2_image(img_features)  # (batch_size, num_classes)

        return out
