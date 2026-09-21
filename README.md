# DiffPhysGraspNav

基于可微物理仿真（Genesis 1.3.3）的抓取位姿采样与抓取成功判定工具链。

项目分两个阶段：

- **Stage 1 — 抓取基座位姿解析计算与概率建模**：由物体位姿、接近方向与期望抓取距离解析构造基座位姿 `T_base`，在右扰动切空间建模扰动分布（平移/姿态小高斯 + Roll 均匀分布），批量采样并做接近路径–AABB 预检；数值核与可微路径可选由 quadrants（Genesis 可微内核）在 CPU/CUDA 上加速。
- **Stage 2 — 可微碰撞抓取检测器**（`grasp_detector` 包）：推理模式给出"双边接触 + 有效力 + 提升确认"布尔判定；训练模式提供基于力控输入与几何穿透代理的可微 loss，经 `scene.backward` 将梯度回传至指力参数。

## 目录结构

```
DiffPhysGraspNav/
├── assets/
│   └── mini_gripper.urdf        # 极简夹爪（fixed 基座 + lift 平移关节 + 双平移指）
├── doc/
│   ├── stage1.md                # Stage 1 开发文档
│   └── stage2.md                # Stage 2 开发文档（v1.2）
├── grasp_detector/
│   ├── __init__.py
│   ├── detector.py              # 推理模式：GraspDetector / detect_parallel
│   ├── diff_loss.py             # 训练模式：DiffGraspLoss + 力参数辅助函数
│   ├── utils.py                 # 张量工具（to_bool / to_float / force_magnitude）
│   └── examples/
│       └── train_example.py     # 两阶段训练完整示例（位姿引导 + 力优化 + 推理判定）
└── scripts/
    ├── stage1_demo.py           # Stage 1 验证演示（24 项检查 + 位姿分布可视化）
    └── stage2_grasp_detector_check.py   # Stage 2 验证脚本（30 项检查）
```

## 环境配置

```bash
mamba activate diffphy
```

依赖：Python ≥ 3.12、genesis 1.3.3、torch、numpy、matplotlib、quadrants（Stage 1 加速，可选）。
可微训练建议 64 位精度初始化：`gs.init(backend=..., precision='64')`。

## 快速开始

### Stage 1：位姿分布采样演示

```bash
python scripts/stage1_demo.py                       # 默认 20000 样本，quadrants auto 后端
python scripts/stage1_demo.py --arch cuda           # 指定 CUDA
python scripts/stage1_demo.py --no-qd               # 纯 numpy 参考模式
python scripts/stage1_demo.py --n-samples 5000 --seed 1
```

运行 24 项检查（采样精度、SO(3) 有效性、Tape 梯度 vs 中心差分等），并输出
`scripts/stage1_pose_viz.png`（TCP 点云按 |roll| 着色 + 接近方向箭头 + 指尖点云 + 物体代理盒）。

### Stage 2：抓取检测器验证（30 项检查全过）

```bash
python scripts/stage2_grasp_detector_check.py --backend cpu --precision 64
```

- Part A（推理）：七用例覆盖远离 / 单侧接触 / 轻触 / 牢固闭合 / 未提拉 / 提拉保持 / 滑落，含 5 帧 ring buffer 去抖；
- Part B（训练）：力控 rollout + 穿透代理 loss，验证梯度非零、结构正确（lift 分量可忽略、指分量方向正确）、有限差分交叉验证（实测比值 1.00）、梯度下降单调降 loss。

### Stage 2：两阶段训练示例

```bash
python grasp_detector/examples/train_example.py --backend cpu
```

流程：PD 位姿引导到预抓取 → `scene.reset(pre_state)` 注册初始态 → 力控梯度优化
（DiffGraspLoss，lr=500）→ 提拉后 `detect(confirm_lift=True)` 推理判定，期望输出 `RESULT: SUCCESS`。

### 作为库使用

推理模式（布尔判定）：

```python
import genesis as gs
from grasp_detector import GraspDetector

detector = GraspDetector(
    scene=scene, robot_entity=robot, object_entity=cube,
    left_finger_name="left_finger_link",
    right_finger_name="right_finger_link",
    support_entity=plane,        # 提升确认阶段验证物体已离开支撑面
    force_threshold=1.5,         # 建议 mass × g × 1.5
)
success, info = detector.detect(confirm_lift=True, return_debug_info=True)
```

训练模式（可微 loss，场景须 `SimOptions(requires_grad=True)`）：

```python
from grasp_detector import DiffGraspLoss, make_force_parameter, ramp_force

diff = DiffGraspLoss(scene, robot, cube,
                     "left_finger_link", "right_finger_link",
                     contact_distance=0.025,      # D_TOUCH：恰触投影距离
                     target_penetration=0.001)    # PEN*：挤压饱和上限

f_base = gs.tensor([0.0, 2.0, 2.0])               # [lift, left, right] (N)
delta_f = make_force_parameter([0.0, 0.0, 0.0])   # sceneless gs.Tensor 叶子

for it in range(iters):
    delta_f.zero_grad()
    scene.reset()
    for t in range(horizon):
        robot.control_dofs_force(ramp_force(f_base, delta_f, t, ramp_steps))
        scene.step()
    loss = diff.compute_loss()    # 基于 Scene.get_state() 的 links_pos 穿透代理
    scene.backward(loss)          # 梯度入口，勿用 loss.backward()
    with torch.no_grad():
        delta_f -= lr * delta_f.grad
```

## 关键设计要点

**推理三要素**（GraspDetector）：双边接触 ∧ 双侧接触力超阈值 ∧ 提升确认（短提拉后
物体离开支撑面且接触保持）。闭合阶段禁止以"物体未接触支撑面"为判据（桌面抓取中该
信号恒为 False）。

**可微训练通路**（Genesis 1.3.3 实测约束，详见 doc/stage2.md §4.2）：

| 环节 | 事实 |
|---|---|
| 可微状态查询 | 仅 `Scene.get_state()`（links_pos 等）；`get_contacts()` 的力为 detached |
| 可微控制入口 | 仅 `control_dofs_force`；PD 位置目标回传梯度显式抛异常 |
| 力张量类型 | 必须 sceneless `gs.Tensor`（`make_force_parameter` 创建） |
| 梯度入口 | `scene.backward(loss)`；每轮迭代以 `scene.reset()` 开头 |
| Loss | links_pos 几何穿透代理：`relu(PEN* − δ) + w_s·|δ_L − δ_R|` |
| 两阶段方案 | PD 位姿引导（不可微）→ `reset(pre_state)` → 力控优化（可微） |

## 已知事项

- **Genesis 1.3.3 site-packages 补丁**：zerocopy 路径将 int32 排序索引直接喂给
  `torch.gather` 导致 `get_contacts()` 崩溃（"Expected dtype int64 for index"）。
  本地安装已打一行补丁（索引升型 int64）。新环境如遇此错误，按
  doc/stage2.md §11 处理（打同样补丁，或设 `GS_ENABLE_ZEROCOPY=0` 绕行）。
- Genesis 1.3.3 接触 adjoint 为近似线性化：解析梯度与有限差分以"符号一致 + 量级
  一致"为验收判据（默认配置下实测比值 1.00）。

## 文档

- [doc/stage1.md](doc/stage1.md) — Stage 1 位姿计算与扰动建模设计文档
- [doc/stage2.md](doc/stage2.md) — Stage 2 可微抓取检测器开发文档（v1.2，含 API 参考、梯度通路表、参数速查、排查表）
