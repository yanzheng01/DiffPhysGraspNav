# Plan B′ 开发测试方案

**依据文档**：[plan_b.md](plan_b.md) v1.3.6（判定降级运动学碰撞 / 全可微单一代价链路 / Genesis 退出训练与推理环）
**版本**：v1.0（2026-09-28）
**运行环境**：`mamba activate diffphy`（torch + GPU）。训练与推理**零 Genesis 依赖**；动力学抽检脚本单独复用 Genesis（§5.2 IT-SPOT-1）。

---

## 1. 目标与范围

**目标**：实现并验证 Plan B′ 全管道——基元参数编码 + 条件扩散多假设生成 + 可微解析运动学层 + 可微碰撞代价（八项）+ T1/T2/T3 训练 + 阈值化推理与失败回传。

**范围内（v0）**：

- 物体：基元参数扫描（盒/圆柱/球）$\sim 10^4$，每物体 $\sim 10$ 个 $\hat{a}$（plan_b §6）
- 网络：条件扩散头（$x_0$-prediction、零初始化双头）+ 基元参数嵌入（plan_b §3）
- 代价：§4.2 八项 + §4.3 扫掠走廊；基元**闭式 SDF**（无 kernel / oracle 路径）
- 训练：T1（C0/C1 课程 + DSM 伪数据）→ T2（全约束 + 短链一致性）→ T3（开度联合 + 失败回流）
- 推理：K=8 采样 → 阈值判定 → Top-K 交付（plan_b §7.1 字段）
- 验收：plan_b §10 八项指标 + 消融实验（plan_b §11）

**范围外（预留不实施）**：点云 encoder（mesh/ABC 进入训练集时启用）、teacher 蒸馏、仿真在环训练、mesh SDF 场预计算与 quadrants oracle 管线。

---

## 2. 里程碑与依赖

```
M0 几何核心 ──▶ M1 代价与判定 ──▶ M3 数据与 T1 ──▶ M4 T2/T3 ──▶ M5 推理与验收
                    ▲                                   ▲
      M2 网络头（依赖 M0，可与 M1 并行开发）───────────────┘
```

| 里程碑 | 交付模块 | 通过门 |
|---|---|---|
| M0 | `se3` / `sdf` / `nominal` / `kinematics` / `config` | G0 |
| M1 | `cost`（八项 + 扫掠 + 阈值判定 + 失败分类） | G1（与 M2 并行） |
| M2 | `encoder` / `diffusion` | G2 |
| M3 | `data` / `train`(T1) / DSM 伪标签 | G3 |
| M4 | `train`(T2/T3) / 失败回流缓冲 | G4 |
| M5 | `infer` / `eval` / `spotcheck` / 消融 | G5 |

每里程碑完成定义：模块代码 + 对应测试 PASS + 全量 `planb_check.py` 回归无退化 + git 提交点。

---

## 3. 代码结构

```
planb/
├── __init__.py
├── config.py        # §12 参数表单一来源（dataclass + 相容性校验）
├── se3.py           # exp map（小角 Taylor 分支）、右扰动合成、s_ξ 归一化
├── sdf.py           # 基元闭式 SDF + 桌面场景 SDF（{W} 系）
├── nominal.py       # T_nom(â)、Xₑ 两级策略、w_obj(T_TCP) 截面宽度
├── kinematics.py    # 可微 FK：T_task + q_open → 指垫/非接触面采样点
├── cost.py          # §4.2 八项代价 + §4.3 扫掠走廊 + 阈值判定 + 失败分类
├── encoder.py       # 基元参数嵌入 + EEF 固定嵌入 → c ∈ R^256
├── diffusion.py     # 条件扩散头（x₀-pred、零初始化、DDIM、DSM、短链）
├── data.py          # 基元扫描数据集 + held-out 划分 + DSM 伪标签
├── train.py         # T1/T2/T3 循环 + 失败回流缓冲（DAGGER）
├── infer.py         # Top-K 生成（§7.1 字段）+ 失败回传（stage0 §8 协议）
└── eval.py          # §10 指标 + AD/FD + 多假设有效率
scripts/
├── planb_check.py       # 单元+集成测试总入口（--group m0|m1|m2|it|all，PASS/FAIL 汇总）
├── planb_train.py       # 训练入口 --stage t1|t2|t3
├── planb_eval.py        # 验收评估入口（ACC 指标表输出）
└── planb_spotcheck.py   # stage2 动力学抽检（唯一 Genesis 依赖，离线监控）
```

**架构隔离约束**：`planb/` 包内**禁止** import genesis / grasp_detector；唯一桥接点在 `scripts/planb_spotcheck.py`（由 IT-ISO-1 强制）。EEF 几何参数 $G_{EEF}$（指厚/指长/开度行程/掌部/指垫宽 $w_{pad}$）从 `assets/mini_gripper.urdf` 提取为静态配置。

---

## 4. 模块规格与关键接口

### 4.1 `config.py` —— 参数单一来源

```python
@dataclass
class PlanBConfig:
    K: int = 8                      # hypothesis count
    ddim_steps: int = 10
    n_p: int = 16                   # pad samples per finger
    q_c: float = 0.5                # contact-patch quantile
    delta_lo: float = 0.0           # [m]
    delta_hi: float = 0.002
    delta_clear: float = 0.005
    mu: float = 0.5                 # friction prior
    theta_max: float = radians(15)
    n_l: int = 8                    # sweep configs
    l_retreat: float = 0.05         # TBD, calibrate in M1
    h_max: float = 0.005
    m_seg: float = 0.2
    s_p: float = 0.01; s_r: float = 0.1     # xi normalization scales
    N_sc: int = 10                  # short-chain period
    lam: dict = dict(touch=1.0, two_s=1.0, sym=0.5, cor=0.5,
                     w=0.3, hgt=0.3, com=0.3, anti=0.5)
    lam_guide: float = 0.5; lam_geo: float = 0.3
    def validate(self, r_min: float) -> None:
        # zero-floor compatibility (plan_b §12):
        # r_min - sqrt(r_min^2 - (q_c * w_pad / 2)^2) <= delta_hi
```

### 4.2 `se3.py` —— SE(3) 工具（plan_b §3.1）

```python
def se3_exp(xi: Tensor[..., 6]) -> Tensor[..., 4, 4]      # Taylor branch near |w| -> 0
def apply_right_perturb(T: Tensor, xi: Tensor) -> Tensor  # T @ exp(xi^), stage1 §4.1 convention
def normalize_xi(xi, s_xi) / denormalize_xi(xi_n, s_xi)   # diffusion noise space
```

约定：$\boldsymbol{\xi} = (\boldsymbol{v}, \boldsymbol{\omega})$ 平移在前；全部计算 {W} 系；Roll 采样均匀 $[-\pi, \pi)$。

### 4.3 `sdf.py` —— 闭式距离场（plan_b §4.1）

```python
class Primitive:   # type ∈ {box, cyl, sphere}; params + pose T_wo
    def sdf(self, pts_w: Tensor[..., 3]) -> Tensor[...]   # signed, autograd on pts_w
def scene_sdf(pts_w, table: TablePlane) -> Tensor         # D_S for L_cor / judgment
```

盒分象限 / 圆柱 $\lVert p-c \rVert - r$ 形式，同 DiffPhyRobot `_clearance_to_obstacles`；v0 无 mesh 路径。

### 4.4 `nominal.py` —— 解析锚点（plan_b §3.1 / §5.3）

```python
def nominal_pose(obj, a_hat: Tensor[3]) -> Tensor[4, 4]   # T_nom(a_hat)
def closing_axis(obj, a_hat, contact_pair=None) -> Tensor[3]  # X_e, two-level strategy
def section_width(T_tcp, obj, x_e) -> Tensor              # w_obj, differentiable, recompute per pose
```

### 4.5 `kinematics.py` —— 可微 FK（plan_b §4.1）

```python
def finger_points(T_task, q_open, g_eef: EefGeom) -> FingerPoints
# FingerPoints: pad_L (n_p,3), pad_R (n_p,3), back (n_b,3)
# differentiable w.r.t. (T_task, q_open); pure torch, no learned components
```

### 4.6 `cost.py` —— 代价与判定（plan_b §4.2–§4.3、§5.3）

```python
def grasp_cost(fp: FingerPoints, obj, scene, q_open, cfg) -> CostOutput
# CostOutput: 8 scalars + total (L_touch/L_2s/L_sym/L_hgt/L_com/L_anti/L_cor/L_w)
# contact patch: top-n_c nearest per finger (differentiable on selected values)
def swept_corridor_cost(T_task, q_open, a_hat, obj, scene, cfg) -> Tensor
def judge_hypothesis(contact_state: dict, cfg) -> tuple[bool, str | None]
# success = (D_min^L, D_min^R ∈ [δ_lo, δ_hi]) ∧ (θ_opp ≤ θ_max) ∧ (corridor ≥ δ_clear)
# failure ∈ {no_contact, hard_collision, corridor_violation, not_antipodal}
```

### 4.7 `encoder.py` —— 条件编码（plan_b §3.2，v1.3.4 决策）

```python
class PrimitiveEncoder(nn.Module):     # params + learnable type token (8d/class)
    def forward(self, params, type_id) -> Tensor[256]   # log/Fourier scale preprocessing
def make_condition(obj_enc, eef_emb, a_hat_enc) -> Tensor[256]  # c consumed by diffusion head
```

条件接口与编码器解耦：扩散头只消费 `c ∈ R^256`（点云塔未来零改动接入）。

### 4.8 `diffusion.py` —— 生成头（plan_b §3.1 / §5）

```python
class GraspDiffusion(nn.Module):
    # x0-prediction; final layers of BOTH heads zero-initialized
    def forward(self, c, t, x_t) -> (x0_hat[..., 6], dq_hat[..., 1])
    @torch.no_grad()
    def ddim_sample(self, c, K, steps=10) -> (xi[K, 6], dq[K, 1])   # denormalized
    def dsm_loss(self, xi_plus, dq_plus, c) -> Tensor
    def short_chain_loss(self, c, geom_cost_fn, steps=2) -> Tensor   # differentiable DDIM chain
```

### 4.9 `data.py` / `train.py` / `infer.py` / `eval.py`

```python
# data.py
class PrimitiveGraspDataset: ...                # ~1e4 objects × ~10 a_hat, {O}-canonicalized
def split_objects(ds, ratio) -> (train, heldout) # object-level split, no param leakage
def make_pseudo_labels(model, batch, cost_fn, K, thresh) -> (xi_plus, dq_plus)  # best-of-K filter

# train.py
class ReplayBuffer: ...                          # (cond, hypothesis, reason, diagnostics)
def train_t1(model, ds, cfg) / train_t2(...) / train_t3(...)   # stage configs

# infer.py
@dataclass
class Hypothesis:                                # plan_b §7.1 fields
    T_task_k; q_open_k; cost_k; contact_state_k; xi_k; T_TCP_k
def generate_topk(model, cond, K=8) -> list[Hypothesis]         # cost-ascending
def classify_failure(results) -> str             # candidate-level vs direction-level (stage0 §8)

# eval.py
def adfd_check(cost_fn, n=200, dtype=float64) -> float          # cosine similarity
def topk_pass_rate(...) / top1_pass_rate(...) / effective_modes(hyps, d_t=0.02, d_r=radians(15))
def cold_start_baseline(...) / latency_benchmark(...) / generalization_gap(...)
```

---

## 5. 测试方案

测试入口统一为 `scripts/planb_check.py`（沿用 stage2 check 风格：`N/N PASS` 汇总、**英文输出**）。梯度类检查统一 float64 中心差分；训练默认 float32。所有 fixture 固定种子确定性生成。

### 5.1 单元测试（M0–M2）

**M0：几何核心**

| ID | 测试 | 方法 | 通过标准 | 依据 |
|---|---|---|---|---|
| UT-SE3-1 | exp map 正确性 | 随机 1000 组 ξ 对照 `scipy.linalg.expm`；校验 R∈SO(3)（det=+1） | max err < 1e-8 | §3.1 |
| UT-SE3-2 | 小角 Taylor 分支 | $\lVert\omega\rVert \in \{10^{-10}, 10^{-8}, 10^{-6}, 10^{-4}\}$ 无 NaN | 相对误差 < 1e-10 | §3.1 注记 |
| UT-SE3-3 | ±π 边界 | 各旋转分量取 $[-\pi, \pi)$ 邻域（含 $\pi - \epsilon$） | 有限且正确 | §3.1 零测奇异 |
| UT-SE3-4 | 右扰动合成 | 已知 T_nom、ξ 手算对照 `T_nom·exp(ξ^)`（与 stage1 §4.1 逐项一致） | err < 1e-10 | §3.1 |
| UT-SE3-5 | s_ξ 归一化往返 | normalize → denormalize；归一化后六分量 std 同量级 | 往返 err < 1e-12 | §3.1 |
| UT-SE3-6 | exp map 梯度 | AD vs 中心差分，200 组随机 ξ | cosine ≥ 0.999 | §10 |
| UT-SDF-1 | 盒 SDF | 内/面/外 $10^4$ 点 vs 暴力最近表面点 | max err < 1e-6 | §4.1 |
| UT-SDF-2 | 圆柱/球 SDF | 同上（含端面、轴线上点） | max err < 1e-6 | §4.1 |
| UT-SDF-3 | 场景 SDF | 桌面平面闭式 + 多物体 min 聚合 | 精确 | §4.2 $D_S$ |
| UT-SDF-4 | SDF 梯度 | 对查询点 AD/FD（float64） | cosine ≥ 0.999 | §4.1 |
| UT-NOM-1 | T_nom 对称性 | $\hat{a}$ 与 $-\hat{a}$ 的 T_nom 关于物体中心镜像；名义深度沿 â | err < 1e-8 | §3.1 |
| UT-NOM-2 | Xₑ 两级策略 | 有接触点对 → 投影优先；无 → 参考向量回退 | 两分支正确触发 | 工程约定 |
| UT-NOM-3 | w_obj 截面宽度 | 长方体沿 Xₑ 平移 TCP：宽度曲线与解析 extent 一致；AD/FD | err < 1e-8；cosine ≥ 0.999 | §5.3 |
| UT-FK-1 | 指垫采样点 | 对照测试内独立数值 FK（同 URDF 参数、独立实现） | err < 1e-8 | §4.1 |
| UT-FK-2 | 非接触面点 | 指背/指侧/掌部点集生成 | err < 1e-8 | §4.2 |
| UT-FK-3 | FK 梯度 | ∂p/∂(ξ, q_open) AD/FD | cosine ≥ 0.999 | §4.1 |

**M1：代价与判定**

| ID | 测试 | 方法 | 通过标准 | 依据 |
|---|---|---|---|---|
| UT-COST-1 | L_touch 接触斑 | 构造已知距离场：$q_c{=}0.5, n_p{=}16 \to n_c{=}8$；仅被选点携带梯度（未选点 grad=0）；窗口内零罚 | 全部成立 | §4.2 分位数口径 |
| UT-COST-2 | L_2s/L_sym/L_hgt/L_w | 构造单侧无接触 / 双侧接触 / 高度差 / 开度越界场景逐项数值对照 | 数值精确 | §4.2 |
| UT-COST-3 | L_com 线段判据 | 2D 构造解析对照 $s$、$d_\perp$、$\ell$ clamp；$s \in [0.2, 0.8]$ 零边距罚；质心 detached、梯度仅流经接触点 | 数值精确 + AD 验证 | §4.2 v1.3.6 |
| UT-COST-4 | L_anti | $\hat\theta_{opp}$ 由可微连线 + detached 法向计算；法向梯度 = 0、接触点梯度 ≠ 0 | 通过 | §4.2 |
| UT-COST-5 | L_cor | $\min(D_O, D_S)$ 聚合；全部非接触点 > δ_clear 零罚 | 通过 | §4.2 |
| UT-COST-6 | 扫掠走廊 | $n_\ell{=}8$ 中间构型含端点、沿 $-\hat{a}$ 回退 $\ell_{retreat}$、路径最大值口径 | 通过 | §4.3 |
| UT-COST-7 | 判定 + 失败分类 | 四类失败场景各构造一例：`no_contact` / `hard_collision` / `corridor_violation` / `not_antipodal` | 分类正确 | §5.3 |
| UT-COST-8 | 蕴含性（统计） | 随机 $10^5$ 个 $(\xi, \Delta q)$：总训练代价 < ε 的样本**必须全部**通过推理三条件判定 | 违例 = 0 | §4.2 → §10 |
| UT-COST-9 | 全代价 AD/FD | 对 $(\xi \in \mathbb{R}^6, \Delta q)$ 联合，float64 | cosine ≥ 0.99 | §10 |
| UT-COST-10 | 零地板相容校验 | 数据集最小曲率半径 $r_{min}$：$r_{min} - \sqrt{r_{min}^2 - (q_c w_{pad}/2)^2} \le \delta_{hi}$ | 配置校验通过 | §12 |

**M2：网络头**

| ID | 测试 | 方法 | 通过标准 | 依据 |
|---|---|---|---|---|
| UT-ENC-1 | 嵌入确定性与尺度区分 | 同输入同输出；参数跨 3 个数量级（1–100 mm）特征范数同量级（log/Fourier 生效） | 通过 | §3.2 |
| UT-ENC-2 | 条件组装 | `c = f(obj_enc, eef_emb, â_enc) ∈ R^256`，类型 token 注入 | shape + 确定性 | §3.2 |
| UT-DIF-1 | 零初始化 | 未训练网络：$\hat{x}_0 = 0$、$\Delta\hat{q} = 0$ → 采样输出逐元素 == T_nom(â) + 闭式开度 | err < 1e-6 | §3.1 |
| UT-DIF-2 | DDIM 确定性 | 同噪声同条件 10 步采样两次（DDPM 对照为随机） | 逐元素 err < 1e-6 | §12 |
| UT-DIF-3 | DSM 损失 | 干净 $\xi^+$ 上损失有限、可反传（grad 非零） | 通过 | §5.1 |
| UT-DIF-4 | 短链可微 | 2–3 步 DDIM 短链上几何代价 backward 无错 | 通过 | §5.2 |

### 5.2 集成测试（M3–M5）

| ID | 测试 | 方法 | 通过标准 | 依据 |
|---|---|---|---|---|
| IT-DATA-1 | 数据集与划分 | $10^4$ 基元 × ~10 â；半空间均匀 + 偏置统计；物体级 held-out 无参数泄漏 | 通过 | §6 |
| IT-ISO-1 | Genesis 隔离 | 无 genesis 环境下 `import planb` 成功；`planb/` 源码 grep 无 genesis/grasp_detector 引用 | 通过 | §2/§8 |
| IT-T1-1 | T1 冒烟 | 200 物体小子集训练：C0 引导 $\overline{\lvert D_O \rvert}$ 相对初始下降 ≥ 50%；无 NaN | 通过 | §5.1（R4） |
| IT-T1-2 | 伪标签周期 | 采样 K → 代价过滤 → $\xi^+$ 进入 $\mathcal{L}_{DSM}$；过滤通过率 ∈ (0, 1) | 通过 | §5.1（R7） |
| IT-T2-1 | T2 全约束 | 八项损失 + 扫掠进入计算图；每 $N_{sc}{=}10$ 步短链损失触发（计数验证） | 通过 | §5.2 |
| IT-T3-1 | 开度联合 | 训练后 $\Delta\hat{q}$ 分布脱离零初始化（激活非零） | 通过 | §5.3 |
| IT-T3-2 | 失败回流闭环 | 判定失败 → 四元组写入 → 读出 → 对比项损失 → 反传，往返一致 | 通过 | §5.3 |
| IT-INFER-1 | Top-K 管道 | K=8 输出字段齐全（`T_task_k/q_open_k/cost_k/contact_state_k/xi_k/T_TCP_k`）且按代价升序 | 通过 | §7.1 |
| IT-INFER-2 | 失败回传 | 全败样本按 stage0 §8 分类候选性/方向性并请求更换 â | 通过 | §7.2 |
| IT-SPOT-1 | 动力学抽检 | held-out 运动学通过样本 → stage2（Genesis）前向提升 → 滑落率记录 | 产出趋势日志（监控，非门） | §5.3/§10 |

### 5.3 验收测试（plan_b §10 映射）

| ID | 指标 | 测量规程 | 目标 |
|---|---|---|---|
| ACC-1 | Top-K 通过率 | 全数据集逐条件采样 K=8，统计 ≥1 通过的条件占比 | 训练集 ≥ 90%；held-out ≥ 80% |
| ACC-2 | Top-1 通过率 | 代价升序首假设通过率 | ≥ ACC-1 的 70% |
| ACC-3 | 推理延迟 | GPU、单条件 K=8（前向 + 解析层 + 代价），warmup 10 + 中位 100 次 | ≤ 50 ms |
| ACC-4 | 泛化衰减 | train − held-out 通过率差 | ≤ 10 pp |
| ACC-5 | 冷启动基线 | 零初始化网络（= T_nom + 闭式开度）通过率 | > 0 |
| ACC-6 | 多假设有效率 | 贪心聚类（$\Delta t > 2$ cm 或 $\Delta\theta > 15°$ 为可分辨，阈值待标定）计模式数 | ≥ 2 |
| ACC-7 | 动力学抽检 | IT-SPOT-1 滑落率跨 epoch 趋势 | 下降（监控线，非验收门） |
| ACC-8 | 代价–梯度保真 | AD/FD 交叉验证（float64，对 $(\xi, \Delta q)$） | cosine ≥ 0.99 |

### 5.4 消融实验（plan_b §11 映射，M5 执行）

| ID | 消融 | 观测 | 依据 |
|---|---|---|---|
| ABL-1 | 移除 $\mathcal{L}_{anti}$ | "训练零损失但推理拒判"率上升（边缘/同面假盆） | §4.2 |
| ABL-2 | $q_c = 1.0$ vs $0.5$ | 平垫贴曲面物体的 $\mathcal{L}_{touch}$ 地板复现/消除 | §4.2 |
| ABL-3 | C0 signed vs $\lvert\cdot\rvert$ | signed 形式穿透深度统计（P2 历史复现） | §5.1 |
| ABL-4 | 去短链一致性 | 10 步采样代价分布相对单步训练分布的偏移 | §5.2 |
| ABL-5 | 单峰回归 vs 扩散 | 有效模式数（多模态平均陷阱） | §11 |
| ABL-6 | 随机 vs 零初始化 | 冷启动收敛曲线 | §3.1 |
| ABL-7 | $m_{seg} \in \{0, 0.2, 0.5\}$ | 通过率与稳定性的敏感性 | §4.2 v1.3.6 |
| ABL-8 | $K \in \{4, 8, 16\}$ | Top-K 通过率–延迟权衡 | §12 |

---

## 6. 里程碑验收门

| Gate | 条件（全部满足方可进入下一里程碑） |
|---|---|
| G0 | UT-SE3-1..6 / UT-SDF-1..4 / UT-NOM-1..3 / UT-FK-1..3 全 PASS |
| G1 | UT-COST-1..10 全 PASS；蕴含性统计测试（UT-COST-8）违例 = 0 |
| G2 | UT-ENC-1..2 / UT-DIF-1..4 全 PASS（含零初始化数值等式 UT-DIF-1） |
| G3 | IT-T1-1 / IT-T1-2 通过；小子集 C0 收敛达标 |
| G4 | IT-T2-1 / IT-T3-1 / IT-T3-2 通过；T1→T2→T3 全量连续训练无 NaN |
| G5 | ACC-1..6、ACC-8 达标；ACC-7 产出趋势记录；ABL-1..8 完成并归档 |

---

## 7. 工程约定

- 全部计算与输出统一在规划参考系 {W}；物体几何规范到 {O} 后变换回 {W}
- 切空间向量 (平移, 旋转) 顺序；与 Sophus/Pinocchio 等库交互时换序
- Roll 分量均匀 $[-\pi, \pi)$；`exp` map 必须含小角 Taylor 分支
- 训练默认 float32；一切梯度检查（AD/FD）float64 中心差分
- 固定种子（torch/numpy）；测试 fixture 确定性生成，保证回归可比
- **所有脚本打印输出使用英文**（项目硬性要求）
- `planb/` 禁止 import genesis / grasp_detector（IT-ISO-1）
- 每里程碑一个 git 提交点；提交前全量 `planb_check.py` 回归
- 失败回流四元组 `(cond, hypothesis, reason, diagnostics)` 中 reason 枚举固定为四类（§5.3），上游协议遵循 stage0 §8

---

## 8. 风险映射（plan_b §9 → 测试项）

| 风险 | 覆盖测试 |
|---|---|
| R1 动力学 gap 无验证器 | IT-SPOT-1 / ACC-7（监控）+ UT-COST-3/4（L_com/L_anti 几何代理正确性） |
| R2 代价局部极小 | IT-T1-1（C0 引导）+ ACC-6（多假设） |
| R3 多模态平均 | ABL-5 |
| R4 冷启动零梯度 | UT-DIF-1 + ACC-5 |
| R6 开度–位姿耦合非凸 | IT-T3-1（闭式初始化 + 联合微调） |
| R7 伪数据确认偏置/模式坍缩 | IT-T1-2 + ACC-6 |

---

## 9. 版本历史

| 版本 | 日期 | 变更 |
|---|---|---|
| v1.0 | 2026-09-28 | 初稿：依据 plan_b.md v1.3.6 生成；M0–M5 里程碑、模块规格、42 项单元/集成测试、8 项验收、8 项消融 |
