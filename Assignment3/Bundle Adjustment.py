import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pytorch3d.transforms import euler_angles_to_matrix
import os

# ==========================================
# 1. 定义 Bundle Adjustment 模型
# ==========================================
class BundleAdjustmentModel(nn.Module):
    def __init__(self, n_cameras=50, n_points=20000, img_size=(1024, 1024)):
        super().__init__()
        W, H = img_size
        
        # 相机内参：焦距 f (优化参数)，光轴中心 cx, cy (固定)
        self.focal_length = nn.Parameter(torch.tensor([1000.0]))#f = H / (2 * \tan(fov/2)),1000是初始值，这样计算得来
        self.cx = W / 2.0 #cx, cy 初始化为图像中心
        self.cy = H / 2.0

        # 相机外参：R (Euler角参数化), T (平移)
        # 初始化：旋转为0，平移为 [0, 0, -2.5]
        self.camera_rotations = nn.Parameter(torch.zeros(n_cameras, 3)) 
        self.camera_translations = nn.Parameter(torch.tile(torch.tensor([0.0, 0.0, -2.5]), (n_cameras, 1)))

        # 3D点云：在原点附近随机初始化
        self.points_3d = nn.Parameter(torch.randn(n_points, 3) * 0.1)

    def forward(self):
        # 1. Euler角转旋转矩阵 R: (N_cam, 3, 3)
        R = euler_angles_to_matrix(self.camera_rotations, convention="XYZ")
        T = self.camera_translations # (N_cam, 3)

        # 2. 世界坐标系 -> 相机坐标系: P_cam = R @ P_world + T
        # 利用广播机制并行处理所有相机和点
        # pts_world: (1, N_pts, 3) -> transpose -> (1, 3, N_pts)
        pts_world = self.points_3d.unsqueeze(0).transpose(1, 2)
        
        # pts_cam: (N_cam, 3, N_pts)
        pts_cam = torch.matmul(R, pts_world) + T.unsqueeze(2)#对应内部的线形变换
        
        # 转回 (N_cam, N_pts, 3) 方便索引 X, Y, Z
        pts_cam = pts_cam.transpose(1, 2)
        
        Xc = pts_cam[:, :, 0]
        Yc = pts_cam[:, :, 1]
        Zc = pts_cam[:, :, 2]

        # 3. 将 3D 坐标映射到 2D 像素平面的透视变换
        # u = -f * Xc/Zc + cx, v = f * Yc/Zc + cy
        eps = 1e-8
        u = -self.focal_length * Xc / (Zc - eps) + self.cx
        v =  self.focal_length * Yc / (Zc - eps) + self.cy

        return torch.stack([u, v], dim=-1) # (N_cam, N_pts, 2)

# ==========================================
# 2. 数据加载函数
# ==========================================
def load_data(data_path, n_cameras=50):
    data = np.load(data_path)
    all_obs = []
    for i in range(n_cameras):
        key = f"view_{i:03d}"
        all_obs.append(data[key])
    
    all_obs_tensor = torch.from_numpy(np.stack(all_obs)).float()
    target_2d = all_obs_tensor[:, :, :2]
    mask = all_obs_tensor[:, :, 2] > 0.5 # 第三列是可见性
    return target_2d, mask

# ==========================================
# 3. 保存带颜色的 OBJ 文件
# ==========================================
def save_colored_obj(points3d, colors_path, filename):
    points = points3d.detach().cpu().numpy()
    colors = np.load(colors_path)
    if colors.max() > 1.0:
        colors = colors / 255.0
    
    with open(filename, 'w') as f:
        for i in range(len(points)):
            x, y, z = points[i]
            r, g, b = colors[i]
            f.write(f"v {x:.6f} {y:.6f} {z:.6f} {r:.6f} {g:.6f} {b:.6f}\n")
    print(f"OBJ saved to {filename}")

# ==========================================
# 4. 主训练循环
# ==========================================
def train():
    # 设备检测
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载数据
    target_2d, mask = load_data("data/points2d.npz")
    target_2d, mask = target_2d.to(device), mask.to(device)

    # 初始化模型
    model = BundleAdjustmentModel(img_size=(1024, 1024)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)#通过改变这些变量来缩小差距
    
    loss_history = []

    print("Starting optimization...")
    for step in range(1001):
        optimizer.zero_grad()
        
        pred_2d = model()
        
        # 只计算可见点的重投影误差 (MSE)
        diff = pred_2d[mask] - target_2d[mask]
        loss = torch.mean(diff**2)
        
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if step % 100 == 0:
            print(f"Step {step:4d} | Loss: {loss.item():.4f} | Focal: {model.focal_length.item():.2f}")

    # 绘制 Loss 曲线
    plt.plot(loss_history)
    plt.yscale('log')
    plt.title("Bundle Adjustment Loss")
    plt.xlabel("Iteration")
    plt.ylabel("MSE Loss")
    plt.savefig("loss_curve.png")
    plt.show()

    # 导出结果
    save_colored_obj(model.points_3d, "data/points3d_colors.npy", "reconstructed_points.obj")

if __name__ == "__main__":
    train()