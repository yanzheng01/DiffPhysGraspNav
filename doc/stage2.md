可微碰撞抓取检测器开发文档
项目名称	基于几何碰撞的可微抓取检测器
仿真平台	Genesis 1.3.3
核心技术	碰撞检测（推理）+ 力控输入与穿透代理损失（可微训练）
文档版本	v1.2
更新日期	2026-09-21
目录
1. 项目概述
2. 系统架构
3. 环境依赖
4. 核心原理
5. 模块设计
6. 快速开始
7. API 参考
8. 可微实现细节
9. 并行环境适配
10. 测试与验证
11. 常见问题排查
12. 已知限制与路线图
1. 项目概述
1.1 项目背景
在机械臂抓取任务的仿真训练中，需要一种自动判定抓取是否成功的机制。传统方法依赖人工观测或布尔型碰撞检测，存在两个问题：
• 不可微：布尔型碰撞检测返回 0/1 信号，梯度为 0，无法用于基于梯度的优化；
• 鲁棒性差：单一接触信号易受瞬时抖动、轻微擦碰干扰。
1.2 设计目标
目标	说明
准确性	正确判定“双边接触 + 有效力 + 提升确认”的抓取状态
可微性	提供 loss 张量，经 scene.backward(loss) 将梯度回传至可优化控制参数
鲁棒性	通过力阈值、提升确认、时间窗滤波抑制误判
可扩展	支持并行环境（scene.build(n_envs=N)，N ≥ 1）
易集成	独立类封装，与主控制循环解耦
1.3 判定准则（三要素）
一次成功的抓取须同时满足：
抓取成功 = 双边接触 (左指 ∧ 右指与物体碰撞)
         ∧ 有效接触力 (双侧接触力均超过阈值)
         ∧ 提升确认 (短提拉后物体离开支撑面，且双边接触与力阈值保持)
判定在两个时机执行：
• 闭合阶段：夹爪闭合稳定后，判定前两条（双边接触 ∧ 力阈值）；
• 提升阶段：控制夹爪沿接近轴短提拉（5–10 cm）后再次判定，三要素须全部成立，此时物体应已离开支撑面（物体–支撑面接触力为零）。
⚠️ 不可在闭合阶段以“物体未接触支撑面”为判据：桌面抓取中，物体在被夹起、离开支撑面之前始终与其接触，该信号恒为 False，判定将永不通过。“离开支撑面”只能作为提升动作之后的结果验证，不能作为闭合阶段的判据。
2. 系统架构
┌─────────────────────────────────────────────────┐
│                  主控制循环                      │
│   (策略网络 / 优化器 / RL Agent)                 │
└───────────────┬─────────────────────────────────┘
                │ 调用
                ▼
┌─────────────────────────────────────────────────┐
│            GraspDetector / DiffGraspLoss         │
│                                                 │
│  ┌─────────────┐  ┌──────────────┐  ┌────────┐ │
│  │ 双边碰撞检测 │  │ 接触力验证    │  │ 提升确认│ │
│  └──────┬──────┘  └──────┬───────┘  └───┬────┘ │
└─────────┼────────────────┼──────────────┼───────┘
          │                │              │
          ▼                ▼              ▼
┌─────────────────────────────────────────────────┐
│              Genesis 1.3.3 Scene                 │
│  推理：RigidEntity.get_contacts(with_entity=...) │
│        RigidEntity.get_links_net_contact_force() │
│  训练：Scene.get_state()（唯一可微状态查询）      │
│        RigidEntity.control_dofs_force()（可微    │
│        控制入口，需 sceneless gs.Tensor）         │
│  Scene.step()                                    │
└─────────────────────────────────────────────────┘
两种工作模式：
• 推理模式（GraspDetector）：返回布尔值，用于评估、数据集生成、RL 环境的 reward/termination 信号；
• 训练模式（DiffGraspLoss）：返回可微 loss 张量，用于基于梯度的抓取优化。
3. 环境依赖
依赖	版本要求	说明
genesis	1.3.3	物理仿真平台
torch	≥ 2.8.0	自动微分（genesis 1.3.3 官方基线；低于此版本可运行但触发 "'torch<2.8.0' is not supported" 警告，建议对齐）
Python	≥ 3.9	—
pip install genesis==1.3.3 torch
数值精度建议：diffsim 训练建议以 64 位初始化——gs.init(backend=gs.cuda, precision='64')（缺省 precision='32'）。接触力对穿透深度的梯度量级小，f32 下易受舍入噪声干扰。
机器人模型要求：
• URDF/MJCF 中包含可识别的左右指尖 Link；
• 碰撞几何（collision geometry）已正确配置；
• 建议在初始化时打印 [l.name for l in robot.links] 确认指尖 Link 名称（1.3.3 无 get_link_names()；get_link(name) 未命中时会在异常信息中列出全部可用 Link 名）。
4. 核心原理
4.1 碰撞检测原理
物理引擎中，碰撞检测分为两阶段：
1. Broad Phase（粗检测）：AABB 包围盒快速排除不可能碰撞的物体对；
2. Narrow Phase（精检测）：逐对计算接触点、法向量、穿透深度。
Genesis 内部维护接触求解器，每步 scene.step() 后通过实体级 API 查询接触信息（1.3.3 的接触查询挂在 RigidEntity 上，无 Scene 级 check_collision / get_contact_force）：
• RigidEntity.get_contacts(with_entity=...)：返回调用实体与指定实体间的接触明细，键含 link_a / link_b（两侧 link 全局索引，与 link.idx 匹配）、geom_a / geom_b、penetration、position（世界系）、normal、force_a / force_b（作用在两侧 geom 上的接触力）、valid_mask（仅并行环境存在）。单环境形状 (n_contacts, ...)，并行环境 (n_envs, n_contacts, ...)。布尔“是否接触”信号由接触掩码导出（并行环境需 & valid_mask）；
• RigidEntity.get_links_net_contact_force(envs_idx=...)：返回实体各 link 的外部接触合力 (n_links, 3) / (n_envs, n_links, 3)，适合快速读取指尖合力。
4.2 可微性原理（Genesis 1.3.3 实测约束）
⚠️ v1.2 修订：以下三条均经源码核对与运行时实测确认，与早期版本文档的假设不同：
1. 唯一可微的状态查询是 Scene.get_state()：其返回的 RigidSolverState 中 qpos / dofs_vel / dofs_acc / links_pos / links_quat 携带 autograd 计算图（requires_grad=True）。而 get_contacts() 的 force_a/force_b、get_links_net_contact_force()、get_dofs_position()、get_pos() 以及 ContactForce 传感器全部返回 detached 张量（无梯度图）——建立在实测接触力上的 loss 无梯度链。
2. PD 位置目标不可微：control_dofs_position 的目标在 tape 回放时仅为前向正确性重放，无输入梯度通路（显式抛出 "Gradients with respect to PD control targets are not supported yet. Use 'control_dofs_force' for differentiable control inputs."）。唯一可微控制入口是 control_dofs_force。
3. 力张量必须是 sceneless 的 gs.Tensor：taped 力张量需带 _backward_from_qd 方法（定义于 genesis.grad.tensor.Tensor），普通 torch.Tensor 的运算结果会在 scene.backward 的 tape 回放中抛 AttributeError；带 scene 属性的张量则会从 Tensor.backward 重入 scene._backward() 并抛 "Multiple backward calls not allowed"。正确做法：经 gs.tensor(...) / gs.zeros(..., requires_grad=True) 创建（scene=None），其上的算术运算经 __torch_function__ 自动保持 gs.Tensor 类型。
因此可微训练的实际通路为：

控制力 f (sceneless gs.Tensor)
   │ control_dofs_force(f)  ── @tracked，写入 tape
   ▼
Scene.step() × horizon     ── 前向仿真，记录梯度 tape
   ▼
Scene.get_state() ──► links_pos（可微）
   ▼
几何穿透代理 δ(p_L, p_R, p_O)  ── 连续可微
   ▼
Loss(δ)
   ▼
scene.backward(loss)  ── 快照 → torch.autograd.backward → 仿真反向（tape 回放，
   │                     含 control_dofs_force 的输入梯度回填）→ 恢复快照
   ▼
delta_f.grad ──► 梯度优化
由于实测接触力不可微查询，Loss 采用几何穿透代理（§4.3）：穿透深度由可微的 links_pos 直接计算，物理上等价于"接触力 ≈ 接触刚度 × 穿透深度"的弹性接触模型，保留了"接触越深 ↔ 抓得越牢"的连续信号，同时拥有完整梯度链。
4.3 Loss 设计（穿透代理）
以指尖 link 原点 p_L、p_R 与物体中心 p_O（均取自可微 links_pos）定义闭合轴与穿透深度：
u = (p_L − p_R) / ‖p_L − p_R‖（闭合轴单位向量）
d_L = (p_L − p_O)·u，d_R = (p_O − p_R)·u（指–物投影距离）
δ_L = relu(D_TOUCH − d_L)，δ_R = relu(D_TOUCH − d_R)（穿透深度代理，米）
Total_Loss = Loss_squeeze            (负向激励：穿透低于 PEN* 时越小 loss 越小，达到 PEN* 后饱和)
           + w_s · Loss_symmetry     (对称约束：双侧穿透平衡)
项	公式	作用
Loss_squeeze	mean(relu(PEN* − δ_L) + relu(PEN* − δ_R))	鼓励闭合并达到目标穿透 PEN*（饱和，防无限挤压）
Loss_symmetry	mean(|δ_L − δ_R|)	防止单侧受力导致物体滑出
D_TOUCH 为"恰触时指尖原点–物体中心的投影距离"（mini 夹爪 + 0.04 m 立方体：立方体半宽 0.02 + 指内侧偏置 0.005 = 0.025 m）。实测梯度质量：解析梯度与中心差分交叉验证比值 1.00（CPU/precision=64）；Genesis 1.3.3 的接触 adjoint 本身是近似线性化，不同 rollout 配置下可能有量级偏差，建议以符号 + 量级一致性作为验收判据。
4.4 梯度有效性说明
⚠️ 重要：当指尖与物体完全无接触时，穿透代理 δ = 0 且 relu 梯度为 0（零穿透区域梯度消失）。因此本检测器适用于：
• 精细化抓取阶段（夹爪已接近/轻触物体）；
• 需要配合不可微的位姿引导阶段（PD 位置控制接近，见 §6.2 两阶段方案）完成从远处接近物体的粗定位。
5. 模块设计
5.1 模块清单
grasp_detector/
├── __init__.py
├── detector.py          # 推理模式：布尔判定
├── diff_loss.py         # 训练模式：可微 loss
├── utils.py             # 张量工具函数
└── examples/
    └── train_example.py # 完整训练示例
5.2 工具函数（utils.py）
import torch
def to_bool(x):
    """将 Genesis 返回值统一转为 Python bool（单环境）"""
    return bool(x.item() if hasattr(x, 'item') else x)
def to_float(x):
    """将 Genesis 返回值统一转为 Python float（单环境）"""
    return float(x.item() if hasattr(x, 'item') else x)
def force_magnitude(force):
    """
    计算接触力模长。
    支持末维为 3 的任意形状：(3,)、(n_contacts, 3)、(n_envs, n_contacts, 3)，
    对应返回标量、(n_contacts,)、(n_envs, n_contacts)。
    """
    if hasattr(force, 'shape') and force.shape[-1] == 3:
        return torch.norm(force, dim=-1)
    return force
6. 快速开始
6.1 推理模式（布尔判定）
import genesis as gs
from grasp_detector.detector import GraspDetector
gs.init(backend=gs.cuda)
# --- 搭建场景 ---
scene = gs.Scene(
    show_viewer=True,
    sim_options=gs.options.SimOptions(dt=0.01),
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(2.0, 0.0, 1.5),
        camera_lookat=(0.0, 0.0, 0.5),
    ),
)
plane = scene.add_entity(gs.morphs.Plane())
robot = scene.add_entity(
    gs.morphs.URDF(file="path/to/robot_with_gripper.urdf")
)
cube = scene.add_entity(
    gs.morphs.Box(pos=(0.5, 0.0, 0.05), size=(0.04, 0.04, 0.04))
)
cube.set_mass(0.1)  # 1.3.3：morphs.Box 无 mass 形参，质量经实体级 set_mass 设置
scene.build()
# --- 确认 Link 名称（首次运行务必执行）---
print([l.name for l in robot.links])
# --- 实例化检测器 ---
detector = GraspDetector(
    scene=scene,
    robot_entity=robot,
    object_entity=cube,
    left_finger_name="left_finger_link",   # ← 替换为实际名称
    right_finger_name="right_finger_link", # ← 替换为实际名称
    support_entity=plane,  # 支撑面（提升确认阶段验证物体已离开）
    force_threshold=1.5,   # N；本例 mass=0.1 → mass × g × 1.5 ≈ 1.47 N
)
# --- 主循环 ---
for i in range(1000):
    robot.control_dofs_position(target_q)  # 你的控制逻辑（PD 目标；含 600–700 步间的提拉）
    scene.step()
    if i == 600:  # 闭合稳定后：阶段一判定（仅双边接触 ∧ 力阈值）
        success, info = detector.detect(confirm_lift=False,
                                        return_debug_info=True)
    if i > 700:  # 提拉完成后：阶段二判定（三要素全判，含提升确认）
        success, info = detector.detect(return_debug_info=True)
        if success:
            print(f"Step {i}: 抓取成功 "
                  f"(L={info['force_left']:.2f}N, R={info['force_right']:.2f}N)")
6.2 训练模式（可微优化，力控设计）
训练模式的四个前提（缺一不可）：
• 场景以可微模式构建：SimOptions(requires_grad=True)（§6.1 推理场景缺省为 False，不可直接复用；Scene.requires_grad 为只读 property，无法事后切换）；
• 可优化参数经 control_dofs_force 接入，且力张量为 sceneless gs.Tensor（§4.2 第 3 条；包内 make_force_parameter() 负责创建，ramp_force() 负责保持类型的渐变叠加）；
• loss 建立在 Scene.get_state() 的可微字段（links_pos）上——即 DiffGraspLoss 的穿透代理，勿直接用 get_contacts() 的力（detached，无梯度）；
• 梯度入口为 scene.backward(loss)，而非 loss.backward()（后者只走 torch 计算图，不驱动仿真反向）。
两阶段方案（§4.4）：先用 PD 位置控制（不可微，开环）把手指带到预抓取轻触位姿，经 scene.reset(pre_state) 将该状态注册为新的 rollout 初始态，再进入力控优化循环。
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.01, requires_grad=True),  # diffsim 开关
    # ... 其余同 6.1（robot / cube 等实体与 scene.build()）
)
from grasp_detector import DiffGraspLoss, make_force_parameter, ramp_force
diff_detector = DiffGraspLoss(scene, robot, cube,
                              "left_finger_link", "right_finger_link",
                              contact_distance=0.025,   # D_TOUCH (m)
                              target_penetration=0.001) # PEN* (m)
# 基础力 [lift, left, right] (N) 与可优化增量（sceneless gs.Tensor 叶子）
f_base = gs.tensor([0.0, 2.0, 2.0])
delta_f = make_force_parameter([0.0, 0.0, 0.0])
horizon, ramp_steps, lr = 15, 8, 500.0
for i in range(100):
    delta_f.zero_grad()
    scene.reset()                                  # 回到注册初始态，重放前向
    for t in range(horizon):                       # 前向 rollout（力控 + 渐变）
        robot.control_dofs_force(ramp_force(f_base, delta_f, t, ramp_steps))
        scene.step()
    loss = diff_detector.compute_loss()            # 基于 links_pos 的穿透代理
    # scene.backward：快照当前态 → torch.autograd.backward → 仿真反向
    #（tape 回放，回填 control_dofs_force 输入梯度）→ 恢复快照
    scene.backward(loss)
    with torch.no_grad():                          # 手写梯度下降（亦可换 Adam）
        delta_f -= lr * delta_f.grad
    print(f"Step {i}, Loss: {loss.item():.6f}")
完整可运行示例（含两阶段与最终推理判定）见 grasp_detector/examples/train_example.py。
7. API 参考
7.1 GraspDetector（推理模式）
GraspDetector(
    scene,                 # Genesis 场景对象
    robot_entity,          # 机器人实体
    object_entity,         # 目标物体实体
    left_finger_name,      # 左指尖 Link 名称
    right_finger_name,     # 右指尖 Link 名称
    support_entity=None,   # 支撑面实体（提升确认阶段验证物体已离开）
    force_threshold=0.5,   # 接触力阈值
)
方法：
方法	返回	说明
detect()	bool	抓取是否成功（三要素全判；应在提升动作后调用）
detect(confirm_lift=False)	bool	仅判定双边接触 ∧ 力阈值（闭合阶段）
detect(return_debug_info=True)	(bool, dict)	附带调试信息
debug_info 字典字段：
字段	类型	说明
touching_left / touching_right	bool	左/右指是否接触物体
force_left / force_right	float	左/右接触力模长
force_threshold_passed	bool	力阈值是否通过
on_support	bool	物体是否仍在支撑面上（提升确认阶段应为 False）
7.2 DiffGraspLoss（训练模式）
DiffGraspLoss(
    scene,                  # Genesis 场景对象（须 requires_grad=True）
    robot_entity,           # 机器人实体
    object_entity,          # 目标物体实体（取其首个 link 为物体代理）
    left_finger_name,       # 左指尖 Link 名称
    right_finger_name,      # 右指尖 Link 名称
    contact_distance=0.025, # D_TOUCH (m)：恰触时指原点–物心投影距离
    target_penetration=0.001, # PEN* (m)：挤压饱和上限，双侧达到后 loss 不再下降
    symmetry_weight=0.5,    # w_s：对称项权重
)
方法：
方法	返回	说明
compute_loss()	torch.Tensor	标量 loss，经 scene.backward(loss) 回传
finger_distances()	(d_L, d_R)	可微指–物投影距离（B,），调试/可视化用
配套模块级辅助函数（diff_loss.py）：
函数	说明
make_force_parameter(values)	创建 sceneless gs.Tensor 叶子（可优化力增量）
ramp_force(base, delta, step, ramp_steps)	线性渐变 base + a·delta，保持 gs.Tensor 类型
内部 Loss 组成（公式见 §4.3）：
项	默认权重	公式
loss_squeeze	1.0	mean(relu(PEN* − δ_L) + relu(PEN* − δ_R))
loss_symmetry	w_s = 0.5	mean(|δ_L − δ_R|)
8. 可微实现细节
8.1 核心实现（diff_loss.py，节选）
import torch
import genesis as gs

def make_force_parameter(values):
    """Sceneless gs.Tensor 叶子：control_dofs_force 的可微输入。"""
    return gs.tensor(list(values), requires_grad=True)

def ramp_force(base, delta, step, ramp_steps):
    """线性力渐变 base + a*delta；经 __torch_function__ 保持 gs.Tensor。"""
    a = min(1.0, (step + 1) / max(int(ramp_steps), 1))
    return base + a * delta

class DiffGraspLoss:
    def __init__(self, scene, robot_entity, object_entity,
                 left_finger_name, right_finger_name, *,
                 contact_distance=0.025, target_penetration=0.001,
                 symmetry_weight=0.5):
        self.scene = scene
        self.l_link = robot_entity.get_link(left_finger_name)
        self.r_link = robot_entity.get_link(right_finger_name)
        self.o_link = object_entity.links[0]
        self.contact_distance = contact_distance      # D_TOUCH (m)
        self.target_penetration = target_penetration  # PEN* (m)
        self.symmetry_weight = symmetry_weight        # w_sym

    def _rigid_state(self):
        """当前场景的可微 RigidSolverState（唯一携带梯度的状态查询）。"""
        for st in self.scene.get_state().solvers_state:
            if st is not None and hasattr(st, "qpos"):
                return st
        raise RuntimeError("no rigid solver state in scene.get_state()")

    def finger_distances(self):
        """可微指–物投影距离 (d_L, d_R)：由 links_pos 计算闭合轴投影。"""
        links_pos = self._rigid_state().links_pos  # (B, n_links, 3)
        pL = links_pos[:, self.l_link.idx, :]
        pR = links_pos[:, self.r_link.idx, :]
        pO = links_pos[:, self.o_link.idx, :]
        u = (pL - pR) / (pL - pR).norm(dim=-1, keepdim=True).clamp(min=1e-6)
        dL = ((pL - pO) * u).sum(-1)
        dR = ((pO - pR) * u).sum(-1)
        return dL, dR

    def compute_loss(self):
        """返回可微标量 loss，梯度经 scene.backward(loss) 回传至控制力。"""
        dL, dR = self.finger_distances()
        penL = torch.relu(self.contact_distance - dL)   # 穿透代理 δ (m)
        penR = torch.relu(self.contact_distance - dR)
        # 挤压项：双侧达到 PEN* 后梯度为零，防止无限挤压/损伤物体
        loss_squeeze = torch.mean(
            torch.relu(self.target_penetration - penL)
            + torch.relu(self.target_penetration - penR))
        # 对称项：平衡双侧穿透，防止单侧滑出
        loss_symmetry = torch.mean(torch.abs(penL - penR))
        return loss_squeeze + self.symmetry_weight * loss_symmetry
8.2 梯度通路注意事项
操作	梯度行为	建议
SimOptions(requires_grad=True)	开启 diffsim：scene.step() 逐步记录梯度 tape	训练场景必须在构造时开启；Scene.requires_grad 为只读 property，无法事后切换
robot.control_dofs_force(f)	@tracked：力张量写入 tape，scene.backward 时回填输入梯度	训练模式的唯一可微参数接入点；f 及其运算结果须为 sceneless gs.Tensor（§4.2 第 3 条）
robot.control_dofs_position/velocity()	@tracked 但无输入梯度通路：tape 回放时仅为前向正确性重放，回传梯度显式抛异常	仅用于推理模式或不可微的位姿引导阶段（两阶段方案的第一阶段）
robot.set_dofs_position()	直写状态、不经 tape，不建立梯度链	仅用于初始化/复位，勿用于训练控制
loss.backward()	仅走 torch 计算图，不驱动仿真反向	梯度入口必须是 scene.backward(loss)（快照 → torch.autograd.backward → 仿真反向 → 恢复快照）
Scene.get_state() 的 qpos/links_pos/links_quat 等	携带 autograd 计算图（requires_grad=True）	训练模式 loss 的数据源（DiffGraspLoss 的穿透代理即建立其上）
get_contacts() 的 force_a/force_b、get_links_net_contact_force()、ContactForce 传感器	detached 张量：数值正确但无梯度图	仅用于推理模式判定与调试打印，勿作为训练 loss 来源
get_contacts() 的布尔化结果（valid_mask & link 匹配）	0/1 信号，不可微	仅用于推理模式判定
scene.reset()	重新武装 forward/backward（backward 后必须先 reset 才能继续前向）；reset(state) 可注册新初始态	每轮训练迭代以 scene.reset() 开头；两阶段方案用 reset(pre_state) 衔接
8.3 优化建议
1. 力阈值动态化：轻物体场景阈值设为 mass × g × 1.5，避免误判；
2. 两阶段优化：
• 阶段一（不可微）：PD 位置控制驱动手指到预抓取轻触位姿，scene.reset(pre_state) 注册为 rollout 初始态；
• 阶段二（可微）：力控 + 穿透代理 Loss（本检测器，捏紧物体）；
3. 学习率：力参数的梯度量级小（实测 ~1e-4 量级），建议 lr=500（纯梯度下降）起步，观察 delta_f 每步增量 0.05–0.1 N 为宜；lr 过大（>2000）会在目标穿透附近震荡；
4. 渐变施力：rollout 内用 ramp_force 线性渐变（ramp_steps≈horizon/2），避免阶跃力激发接触冲击；
5. 梯度验收：以有限差分交叉验证符号一致 + 量级一致（比值 0.2–5）为判据；Genesis 1.3.3 接触 adjoint 为近似线性化，不要求精确匹配（本检测器默认配置下实测比值 1.00）。
9. 并行环境适配
Genesis 支持多环境并行仿真。并行模式下 get_contacts 各项形状为 (n_envs, n_contacts, ...)，无效接触槽位需用 valid_mask 屏蔽；判定逻辑需全程张量运算。
def detect_parallel(robot, obj, support, l_link, r_link,
                    force_threshold=1.0, confirm_lift=True):
    """并行环境批量判定，返回成功环境索引"""
    contacts = robot.get_contacts(with_entity=obj)
    valid = contacts["valid_mask"]                    # (n_envs, n_contacts)
    # 指尖接触掩码：robot 侧可能出现在接触对的 a 或 b 侧，两侧均按全局 link 索引匹配
    la = (contacts["link_a"] == l_link.idx) & valid
    lb = (contacts["link_b"] == l_link.idx) & valid
    ra = (contacts["link_a"] == r_link.idx) & valid
    rb = (contacts["link_b"] == r_link.idx) & valid
    touch_l = (la | lb).any(dim=-1)                   # (n_envs,)
    touch_r = (ra | rb).any(dim=-1)                   # (n_envs,)
    fa, fb = contacts["force_a"], contacts["force_b"] # (n_envs, n_contacts, 3)
    # force_a 作用在 a 侧 geom、force_b 作用在 b 侧 geom，按 link 匹配分别累加
    force_l = (fa * la.unsqueeze(-1) + fb * lb.unsqueeze(-1)).sum(dim=1).norm(dim=-1)  # (n_envs,)
    force_r = (fa * ra.unsqueeze(-1) + fb * rb.unsqueeze(-1)).sum(dim=1).norm(dim=-1)  # (n_envs,)
    is_success = (touch_l & touch_r
                  & (force_l > force_threshold)
                  & (force_r > force_threshold))
    if confirm_lift:
        # 提升确认：物体须已离开支撑面（本模式应在短提拉后调用）
        on_support = obj.get_contacts(with_entity=support)["valid_mask"].any(dim=-1)  # (n_envs,)
        is_success = is_success & (~on_support)
    return torch.where(is_success)[0]  # 成功环境的索引张量
规则：
场景	检测方式
单环境（scene.build() 缺省 n_envs=0）	get_contacts 已按有效性过滤、返回中无 valid_mask 键；to_bool() 转标量后可用 if
并行环境（scene.build(n_envs=N)，N ≥ 1）	全程张量运算，先以 valid_mask 屏蔽无效接触槽位，禁止 .item() 逐环境循环（注意 n_envs=1 也走并行路径，返回带 batch 维）
训练模式（DiffGraspLoss）天然支持并行：links_pos 形状为 (B, n_links, 3)，穿透代理对 batch 维取 mean 归约，无需 valid_mask；control_dofs_force 的力张量按 (B, n_dofs) 广播即可。
10. 测试与验证
10.1 单元测试用例
用例	预期结果
夹爪完全张开，远离物体	False，双侧未接触
仅左指接触物体	False，单边接触
双侧轻触（力 < 阈值）	False，力度不足
闭合阶段双侧牢固接触（detect(confirm_lift=False)）	True
双侧接触但物体仍在支撑面（未提拉，confirm_lift=True）	False，未通过提升确认
提拉 10 cm 后接触与力保持、物体离开支撑面	True 持续保持
提拉后物体滑落（单侧失去接触）	False
10.2 可微性验证
完整验证脚本：scripts/stage2_grasp_detector_check.py（Part B，实测 11 项全过）。
# 在一轮力控前向 rollout 之后执行（循环骨架见 6.2）
loss = diff_detector.compute_loss()
scene.backward(loss)
assert delta_f.grad is not None
assert delta_f.grad.abs().sum().item() > 0, "梯度为零，检查接触状态与控制通路"
# 梯度结构：lift 分量可忽略（穿透与提升无关），指分量显著且方向正确
# 有限差分交叉验证：符号一致 + 量级一致（比值 0.2–5；实测 1.00）
# 优化收敛：lr=500 梯度下降 10 步 loss 单调下降，双侧 delta_f 对称收敛
# 端到端：examples/train_example.py 两阶段训练后 detect(confirm_lift=True) 为 True
10.3 鲁棒性建议
• 时间窗滤波：连续 5 帧判定成功才输出 True，抑制瞬时抖动。实现需维护滚动历史（ring buffer）：单环境用 collections.deque(maxlen=5) 记录近 5 帧 bool、以 all(buffer) 归约；并行环境用 (n_envs, 5) bool 张量缓冲、每步移位写入并以 .all(dim=-1) 归约；
• 延迟检测：控制信号发出后等待 10–20 个仿真步再判定。
11. 常见问题排查
现象	可能原因	解决方案
始终返回 False	Link 名称错误	print([l.name for l in robot.links]) 核对
物体穿过指尖	碰撞几何配置过小	检查 URDF 中 collision geometry 与 visual geometry 是否一致
力始终为 0（推理）	未接触 / Link 名称错误	确认夹爪已闭合；print([l.name for l in robot.links]) 核对
误判“抓取成功”	物体搁在桌面被指尖轻碰	提拉后以 confirm_lift=True 二次判定（物体须已离开支撑面）
两指合拢时自碰撞干扰	指指碰撞被计入	get_contacts(with_entity=object) 只含 robot–object 接触对，指指接触天然不计入；若自行遍历全场景接触，需按 link_a/link_b 过滤
并行环境报错	使用了 .item()	改为第 9 节的张量判定逻辑
训练梯度为 0	物体与指尖零接触 / loss 建立在 detached 数据上	先用位姿引导接近（§6.2 两阶段）；确认 loss 数据源为 Scene.get_state() 的 links_pos 而非 get_contacts() 的力
backward 抛 "PD control targets are not supported yet"	经 control_dofs_position/velocity 传入待优化参数	改用 control_dofs_force + make_force_parameter 创建的 sceneless gs.Tensor（§4.2）
backward 抛 AttributeError: '_backward_from_qd'	力张量为普通 torch.Tensor	经 gs.tensor(...) 创建（或用 make_force_parameter），运算保持 gs.Tensor 类型
backward 抛 "Multiple backward calls not allowed"	力张量带 scene 属性，或一轮迭代内多次 scene.backward	确保力张量 sceneless（§4.2 第 3 条）；每轮迭代只调一次 scene.backward，迭代开头 scene.reset()
backward 后 step 抛 "Forward simulation not allowed"	未 reset 就继续前向	scene.backward 后必须先 scene.reset()（或 reset(state)）才能 scene.step()
zerocopy 路径 gather 报 "Expected dtype int64 for index"	1.3.3 collider 将 int32 排序索引直接喂给 torch.gather	本地安装已打补丁（索引升型 int64）；无补丁环境可设 GS_ENABLE_ZEROCOPY=0 绕行
12. 已知限制与路线图
12.1 当前限制
1. 零接触梯度消失：无接触时穿透代理梯度为 0，需两阶段方案（§6.2）先经位姿引导接近；
2. 纯几何/力信号不含摩擦闭环验证：极滑物体可能在检测通过后仍滑落；
3. 复杂/铰接物体：get_contacts(with_entity=...) 已自动覆盖物体全部 link，但各 link 的接触当前同等对待；若需限定抓取部位（如仅抓把手），须自行按 link_a/link_b 过滤物体侧 link；训练模式物体代理取首个 link 的 links_pos，对质心偏移大的物体需改用加权中心；
4. Genesis 1.3.3 接触 adjoint 为近似线性化：解析梯度与有限差分可能有量级偏差（本检测器默认配置下实测比值 1.00，但配置变化时不保证），梯度验收以符号 + 量级一致为准；
5. 力控训练不优化提升动作：lift 通道在训练 rollout 中保持基础力（靠下限位停在抓取高度），提升阶段由推理侧 PD 位置控制完成（见 examples/train_example.py）。
12.2 路线图
版本	特性
v1.3	接触点法向量检测（get_contacts 已返回 normal / penetration，无需额外数据即可实现）
v1.3	最少接触点数量判据（对 get_contacts 明细按 link 掩码计数即可，防“点接触”）
v1.4	force_threshold 随物体质量自适应
v1.5	接触力 + 触觉/力矩多模态融合
附录 A：完整代码索引
• GraspDetector（推理模式）：detector.py 实现随代码仓库提供，本文档收录其接口契约（§7.1）与判定准则（§1.3）
• DiffGraspLoss（训练模式）：见本文第 8.1 节；完整实现见 grasp_detector/diff_loss.py
• 并行判定函数：见本文第 9 节
• 工具函数：见本文第 5.2 节
• 验证脚本：scripts/stage2_grasp_detector_check.py（§10.1 七用例 + §10.2 梯度验证，实测 30 项全过）
• 训练示例：grasp_detector/examples/train_example.py（§6.2 两阶段完整流程）
• 机器人模型：assets/mini_gripper.urdf（fixed 基座 + lift 平移关节 + 双平移指）
附录 B：参数推荐速查
参数	推荐值	说明
force_threshold	mass × g × 1.5	力阈值（推理），随物体质量调整
contact_distance (D_TOUCH)	立方体半宽 + 指内侧偏置	穿透代理零点；mini 夹爪 + 0.04 m 立方体为 0.025 m
target_penetration (PEN*)	0.001 m（名义穿透的 2–3 倍）	loss_squeeze 饱和上限，防止无限挤压
提拉幅度	5–10 cm	提升确认的提拉距离
延迟检测步数	10–20 步	闭合后等待物理稳定
时间窗长度	5 帧	抑制瞬时抖动
w_s	0.5	对称项权重，可网格搜索
horizon / ramp_steps	15 / 8	训练 rollout 步数 / 力渐变步数
基础指力 f_base	2 N/指（mini 夹爪）	名义穿透 ~0.4 mm；lift 通道 0 N 靠下限位停在抓取高度
学习率 lr	500（纯梯度下降）	力参数梯度 ~1e-4 量级；>2000 会震荡
本文档基于 Genesis 1.3.3 API 编写。如 Genesis 版本升级导致 API 变化（如 get_contacts 返回结构调整），请以官方文档为准并相应更新 detector.py / diff_loss.py。

修订记录：
• v1.2（2026-09-21）：按 diffsim 运行时实测修订训练模式——确认唯一可微状态查询为 Scene.get_state()（get_contacts 的力 / get_links_net_contact_force / get_dofs_position 均为 detached）；确认 PD 位置目标不可微（显式异常），唯一可微控制入口为 control_dofs_force；明确力张量须为 sceneless gs.Tensor（_backward_from_qd / 防重入 scene._backward）；Loss 由实测接触力改为 links_pos 几何穿透代理（D_TOUCH / PEN* / w_s 参数化）；新增 make_force_parameter / ramp_force 辅助；两阶段方案改为"PD 位姿引导 + reset(pre_state) + 力控优化"；验证脚本 Part B 11 项全过（FD 交叉验证比值 1.00）；补充 backward 系列报错排查与附录 B 训练参数。
• v1.1（2026-09-21）：按 Genesis 1.3.3 实测 API 全面修订——接触查询改为实体级 get_contacts / get_links_net_contact_force（无 Scene 级 check_collision / get_contact_force）；微分入口改为 SimOptions(requires_grad=True) + control_dofs_*（@tracked）+ scene.backward(loss)；第三判定要素改为提升确认（两时机判定）；loss_force 增加 F* 饱和；修正示例（Box 质量经 set_mass、持续控制用 control_dofs_position、单/并行环境按 n_envs 划分）；公式补 abs 与代码对齐。
• v1.0（2026-09-21）：初稿。