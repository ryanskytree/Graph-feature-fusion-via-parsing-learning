

import time
import os.path as osp
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
import torch_geometric.transforms as T

# 设置参数
dataset_name = 'Cora'
hidden_channels = 64
lr = 0.01
epochs = 100
seed = 1000

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 设置随机种子（可复现）
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# 加载数据并归一化特征
path = osp.join('/tmp', 'Planetoid')
dataset = Planetoid(path, dataset_name, transform=T.NormalizeFeatures())
data = dataset[0]

# 自定义 1:1:8 数据划分（随机）
def random_split_1_1_8(data, seed=42):
    num_nodes = data.num_nodes
    indices = torch.randperm(num_nodes, generator=torch.Generator().manual_seed(seed))
    
    num_train = int(0.1 * num_nodes)
    num_val = int(0.1 * num_nodes)
    num_test = num_nodes - num_train - num_val

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[indices[:num_train]] = True
    val_mask[indices[num_train:num_train+num_val]] = True
    test_mask[indices[num_train+num_val:]] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data

# 应用新的划分
data = random_split_1_1_8(data, seed=seed).to(device)

# 定义 GCN 模型
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x

model = GCN(
    in_channels=dataset.num_features,
    hidden_channels=hidden_channels,
    out_channels=dataset.num_classes,
).to(device)

# 优化器（只对第一层施加weight_decay）
optimizer = torch.optim.Adam([
    dict(params=model.conv1.parameters(), weight_decay=5e-4),
    dict(params=model.conv2.parameters(), weight_decay=0)
], lr=lr)

# 训练函数
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)

# 测试函数
@torch.no_grad()
def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        acc = (pred[mask] == data.y[mask]).sum().item() / int(mask.sum())
        accs.append(acc)
    return accs

# 训练与测试流程
best_val_acc = test_acc = 0
times = []
for epoch in range(1, epochs + 1):
    start = time.time()
    loss = train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    if epoch % 20 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
              f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
    times.append(time.time() - start)

print(f'Median time per epoch: {torch.tensor(times).median():.4f}s')
# 获取训练好模型的 conv1 中间输出（可用于 frozen 特征提取）
from torch_geometric.utils import to_dense_adj, dense_to_sparse
def diffusion_embedding_on_feat(x_feat, edge_index, method='ppr', alpha=0.2, t=5.0):
    A = to_dense_adj(edge_index, max_num_nodes=x_feat.size(0))[0]  # [N, N]
    A = A + torch.eye(x_feat.size(0), device=x_feat.device)  # 加 self-loop
    D = A.sum(dim=1)

    if method == 'ppr':
        D_inv = torch.diag(1.0 / D.clamp(min=1e-8))
        P = alpha * torch.inverse(torch.eye(A.size(0), device=x_feat.device) - (1 - alpha) * A @ D_inv)
        return P @ x_feat

    elif method == 'heat':
        D_sqrt_inv = torch.diag(1.0 / D.sqrt().clamp(min=1e-8))
        L = torch.eye(A.size(0), device=x_feat.device) - D_sqrt_inv @ A @ D_sqrt_inv
        H = torch.matrix_exp(-t * L)  # Heat kernel: exp(-tL)
        return H @ x_feat

    else:
        raise ValueError("method must be 'ppr' or 'heat'")



# 获取训练好模型的 conv1 中间输出（可用于 frozen 特征提取）
with torch.no_grad():
    x_input = F.dropout(data.x, p=0.5, training=False)  # 模拟训练时 dropout
    conv1_output = model.conv1(x_input, data.edge_index, data.edge_attr).relu()


print(f"conv1 输出特征维度: {conv1_output.shape}")

# 新的一层 GCNConv 映射到 128维
gcn_next = GCNConv(conv1_output.size(1), 128).to(device)

# 初始化 + 前向传播（不训练时用 .eval() 和 no_grad）
gcn_next.eval()
with torch.no_grad():
    base_feat = gcn_next(conv1_output, data.edge_index)
print(f"base_feat 输出特征维度: {base_feat.shape}")


x_feat = base_feat  # [N, d]

# PPR 嵌入
ppr_feat = diffusion_embedding_on_feat(x_feat, data.edge_index, method='ppr', alpha=0.8)

# Heat 嵌入
heat_feat = diffusion_embedding_on_feat(x_feat, data.edge_index, method='heat', t=1.0)

print("PPR 特征维度:", ppr_feat.shape)
print("Heat 特征维度:", heat_feat.shape)
base_feat = ppr_feat
