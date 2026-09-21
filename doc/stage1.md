以下是将第一阶段最终方案整理为一份结构清晰、可直接用于工程开发参考的 Markdown 文档。
***
# 抓取位姿解析初步计算与概率建模方案 (第一阶段)
## 1. 方案概述
本方案对应抓取规划的“由粗到精”策略中的**第一阶段**。目标是在给定目标物体几何、末端执行器（EEF）构型和近进方向的条件下，通过解析几何方法快速计算出一个初步的目标位姿基准，并将其概率化为一个位姿分布，以消除解析计算中的奇异性，并为第二阶段的局部数值采样精调提供数学指导。
## 2. 系统定义与输入条件
### 2.1 坐标系定义
*   **规划参考系 $\{W\}$**：世界坐标系或机器人基座系（由系统配置决定）。第一阶段的所有中间计算与最终输出均统一在该系下表达。
*   **物体坐标系 $\{O\}$**：固连于目标物体的参考系。
*   **EEF 工具坐标系 $\{E\}$**：固连于末端执行器夹爪中心的参考系（TCP）。
    *   规定夹爪的接近轴为 $Z_e$ 轴，开合轴为 $X_e$ 轴，法向轴为 $Y_e$ 轴。
### 2.2 输入参数
*   **$T^W_O$**：物体坐标系 $\{O\}$ 在规划参考系 $\{W\}$ 下的位姿（来源于感知/定位模块）。
*   **$\vec{p}_c$**：目标物体局部主接触区域中心点在 $\{O\}$ 下的坐标。
*   **$d$**：EEF 的 TCP 偏移量（TCP 原点沿接近轴到夹持中心的距离）。
*   **$\hat{a}$**：EEF 近进方向（在 $\{O\}$ 下的单位向量）。
*   **$(\vec{p}_1, \vec{p}_2)$**（可选）：两指预期接触点在 $\{O\}$ 下的坐标（来源于接触预测模块，约定 $\vec{p}_1 \to \vec{p}_2$ 为开合轴正方向）。提供时可确定开合轴几何先验，显著降低 Roll 维不确定性（见 3.2 与 4.2）。

**坐标系统一**：输入的 $\vec{p}_c$、$\hat{a}$（及可选的 $\vec{p}_1, \vec{p}_2$）需先变换至规划参考系 $\{W\}$ 再进行后续计算：
$$ \vec{p}_c^{W} = R^W_O\, \vec{p}_c + \vec{t}^W_O, \qquad \hat{a}^{W} = R^W_O\, \hat{a}, \qquad \vec{p}_i^{W} = R^W_O\, \vec{p}_i + \vec{t}^W_O $$
## 3. 确定性解析计算模型 (基准位姿构建)
本步骤旨在计算一个确定的基准位姿矩阵 $T_{base}$（即 EEF 在 $\{W\}$ 下的位姿 $T^W_E$）。
### 3.1 位置解析计算
平移向量 $P_{base}$（$\{W\}$ 下）由接触中心点沿接近方向反方向偏移 TCP 距离 $d$ 得到：
$$ P_{base} = \vec{p}_c^{W} - d \cdot \hat{a}^{W} $$
### 3.2 姿态解析计算 (含抗奇异处理)
确定 EEF 的旋转矩阵 $R_{base} = [\vec{X}_e, \vec{Y}_e, \vec{Z}_e]$：
1.  **确定接近轴 $\vec{Z}_e$**：
    $$ \vec{Z}_e = \hat{a}^{W} $$
2.  **确定开合轴 $\vec{X}_e$（几何先验优先，参考向量回退）**：
    *   **优先——接触点对几何先验**：若提供接触点对 $(\vec{p}_1^W, \vec{p}_2^W)$，开合轴取接触点连线方向向垂直于 $\vec{Z}_e$ 的平面投影（约定 $\vec{X}_e$ 由 $\vec{p}_1$ 指向 $\vec{p}_2$）：
        $$ \Delta\vec{p} = \vec{p}_2^W - \vec{p}_1^W, \qquad \vec{X}_e = \frac{\Delta\vec{p} - (\Delta\vec{p} \cdot \vec{Z}_e)\vec{Z}_e}{\left\| \Delta\vec{p} - (\Delta\vec{p} \cdot \vec{Z}_e)\vec{Z}_e \right\|} $$
        **退化检测**：若投影范数 $\left\| \Delta\vec{p} - (\Delta\vec{p} \cdot \vec{Z}_e)\vec{Z}_e \right\| < \epsilon_p$（连线与接近轴接近平行，几何先验失效；$\epsilon_p$ 与物体尺度相关，如 $1\text{mm}$），回退至参考向量法。
        **一致性检查**（轻量）：若 $\left\| \vec{p}_c^W - (\vec{p}_1^W + \vec{p}_2^W)/2 \right\|$ 明显超出感知误差量级，提示上游 $\vec{p}_c$ 与接触点对数据不一致。
    *   **回退——参考向量法 (抗奇异策略)**：
        默认选取规划参考系 $\{W\}$ 的 Z 轴方向 $\vec{r}_{default} = [0, 0, 1]^T$（与变换后的 $\hat{a}^W$ 同处 $\{W\}$，避免物体系/世界系混用）。
        **奇异检测**：若 $|\vec{r}_{default} \cdot \vec{Z}_e| > 1 - \epsilon$（两向量接近平行，$\epsilon$ 取 $10^{-6}$），则切换备用参考轴：$\vec{r} = [1, 0, 0]^T$。否则 $\vec{r} = \vec{r}_{default}$。
        $$ \vec{X}_e = \frac{\vec{r} - (\vec{r} \cdot \vec{Z}_e)\vec{Z}_e}{\left\| \vec{r} - (\vec{r} \cdot \vec{Z}_e)\vec{Z}_e \right\|} $$
        **跳变说明**：参考向量切换仅改变绕 $\vec{Z}_e$ 的旋转基准（$\vec{X}_e, \vec{Y}_e$ 跳变约 90°），不影响 $\vec{Z}_e$ 与平移基准；该跳变由第 4 节 Roll 维的均匀分布建模吸收，第二阶段不得依赖 Roll 基准值的连续性。几何先验模式下 $\vec{X}_e$ 由数据连续决定，不存在此跳变问题。
3.  **构建法向轴**：
    $$ \vec{Y}_e = \vec{Z}_e \times \vec{X}_e $$
### 3.3 组装基准位姿矩阵
$$ T_{base} = \begin{bmatrix} \vec{X}_e & \vec{Y}_e & \vec{Z}_e & P_{base} \\ 0 & 0 & 0 & 1 \end{bmatrix} $$
### 3.4 基准位姿有效性预检
解析计算完成后，对 $T_{base}$ 做轻量有效性预检（按系统提供的能力分级执行）：
*   **接近路径碰撞检查**（推荐）：检查线段 $P_{base} \to \vec{p}_c^{W}$（TCP 后退 $d$ 的路径）与环境障碍（尤其支撑面）的碰撞。典型风险场景：$\vec{p}_c$ 靠近支撑面且 $\hat{a}^W$ 接近水平时，后退路径可能扫过桌面。
*   **工作空间检查**（可选，需机器人模型）：检查 $P_{base}$ 是否位于机器人可达工作空间内。
*   **失败处置**：预检失败时**不输出无效基准**，而是向上游返回失败原因（碰撞/不可达）及诊断量（如 $P_{base}$ 坐标、接近路径几何），请求更换近进方向或接触区域；不得静默输出后依赖第二阶段采样纠正——采样无法修复系统性不可达/必碰撞的基准。
## 4. 概率化改造与位姿分布建模
由于第一阶段的解析计算中，绕接近轴 $\vec{Z}_e$ 的滚动角缺乏物理约束，强行确定的 $T_{base}$ 在该维度上具有极大的不可靠性。因此，将模型改造为**李群 $SE(3)$ 切空间上的因子化混合分布**（5 维小方差高斯 × 1 维 Roll 分布），输出位姿的概率模型供第二阶段采样。Roll 维分布按开合轴来源分级：无几何先验时为均匀分布，有接触点对先验时退化为小方差高斯（见 4.2）。
### 4.1 李代数参数化
目标位姿 $T_{target}$ 定义为基准位姿 $T_{base}$ 附加一个 6 维扰动 $\delta\boldsymbol{\xi} \in \mathbb{R}^6$（李代数 $se(3)$ 元素，**右扰动 / body 系**）：
$$ T_{target} = T_{base} \cdot \exp(\delta\boldsymbol{\xi}^\wedge) $$
其中 $\delta\boldsymbol{\xi} = [\Delta x, \Delta y, \Delta z, \Delta\alpha, \Delta\beta, \Delta\gamma]^T$，各分量与 EEF body 系三轴的对应关系**必须**按下表理解：

| 分量 | 对应轴 | 物理含义 | 分布 |
|---|---|---|---|
| $\Delta x, \Delta y, \Delta z$ | 沿 $\vec{X}_e, \vec{Y}_e, \vec{Z}_e$ | 平移扰动 | 小方差高斯 |
| $\Delta\alpha$ | 绕 $\vec{X}_e$（开合轴） | 开合面内倾斜 | 小方差高斯 |
| $\Delta\beta$ | 绕 $\vec{Y}_e$（法向轴） | 法向面内倾斜 | 小方差高斯 |
| $\Delta\gamma$ | 绕 $\vec{Z}_e$（**接近轴**） | **Roll（回退模式下无约束）** | **回退：均匀 / 几何先验：小方差高斯** |

> **实现约定警告**：本文档采用 $(\vec{v}, \vec{\omega})$ 顺序（平移在前 3 位）。Sophus、Pinocchio 等常用库的 $se(3)$ 切向量为 $(\vec{\omega}, \vec{v})$ 顺序（旋转在前），对接时必须重排分量，否则协方差各维错位、均匀维错装。
### 4.2 扰动分布设计
各维度置信度由物理约束决定，采用**因子化混合分布**（5 维高斯 × 1 维 Roll 分布）：

**高斯部分**：$\delta\boldsymbol{\xi}_g = [\Delta x, \Delta y, \Delta z, \Delta\alpha, \Delta\beta]^T \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma}_g)$，其中 $\boldsymbol{\Sigma}_g = \text{diag}(\sigma_{px}^2, \sigma_{py}^2, \sigma_{pz}^2, \sigma_{\alpha}^2, \sigma_{\beta}^2)$：
*   **平移分量 ($\sigma_{px}, \sigma_{py}, \sigma_{pz}$)**：来源于几何感知误差与标定误差。设置较小（如 $2\text{mm}$，单位约定为 m）。
*   **绕开合轴/法向轴旋转 ($\Delta\alpha, \Delta\beta$)**：绕 $\vec{X}_e$（开合轴）与 $\vec{Y}_e$（法向轴）的旋转，受近进方向执行误差影响。设置较小（如 $0.02\text{ rad}$，约 1.1 度）。

**Roll 部分（$\Delta\gamma$，绕接近轴 $\vec{Z}_e$）——按开合轴来源分级**：
*   **回退模式（无接触点对）**：$\Delta\gamma \sim \mathcal{U}(\mathcal{R}_\gamma)$，默认 $\mathcal{R}_\gamma = [-\pi, \pi)$。开合轴由参考向量任意决定，Roll 是解析法无法确定的自由度，建模为均匀分布——周期量在无先验时的最大熵分布。**不采用大方差高斯**的原因：若 $\sigma_{roll} = \pi$，则 $|\Delta\gamma| > \pi$ 的样本约占 32%，经 $\exp$ 映射回绕折叠后严重扭曲分布。
*   **几何先验模式（提供接触点对）**：开合轴已由接触点连线确定（见 3.2），Roll 仅余接触点定位误差引入的小幅不确定，退化为小方差分布 $\Delta\gamma \sim \mathcal{N}(0, \sigma_\gamma^2)$（如 $0.05 \sim 0.1\text{ rad}$，由接触点定位误差传播估计；$\sigma_\gamma \ll \pi$，无回绕问题）。此模式可将第二阶段 Roll 维采样预算降低约一个数量级。
*   **夹爪 180° 对称性**：若夹爪关于绕 $\vec{Z}_e$ 旋转 180° 对称（理想平行夹爪），回退模式可收缩为 $\mathcal{R}_\gamma = [0, \pi)$（即在商空间 $SO(3)/C_2$ 上采样），节省一半采样预算。注意指面纹理、指尖形状、走线等会破坏该对称性，需按实际夹爪显式确认。

**平移-旋转耦合说明**：$\exp$ 映射的平移部分为 $V(\vec{\omega})\,\vec{v}$，其中 $V = \int_0^1 \exp(s\,\vec{\omega}^\wedge)\,ds$ 是旋转的平均，故恒有 $\|V\| \le 1$。回退模式下大 $\Delta\gamma$ 样本（可达 $\pm\pi$）会使平移扰动在 $\vec{X}_e\text{-}\vec{Y}_e$ 平面内被旋转约 $\|\vec{\omega}\|/2$ 并**收缩** $\mathrm{sinc}(\|\vec{\omega}\|/2)$ 倍（$\|\vec{\omega}\| \le \pi$ 时收缩因子 $\in [2/\pi, 1]$；$\vec{Z}_e$ 方向分量不变），平移误差**不会被放大**。均匀 Roll 下横向实际标准差 $\approx 0.88\sigma_p$（已由 demo 数值验证：0.874），且方向与 $\vec{v}$ 的采样方向去相关——proposal 的横向覆盖略小于 $\sigma_p$，由重要性权重纠正即可；几何先验模式下 $\|\vec{\omega}\|$ 很小，此效应可忽略。
### 4.3 输出密度（切空间能量函数形式）
记 $\boldsymbol{\xi}(T) = \log(T_{base}^{-1} \cdot T) = [\xi_x, \xi_y, \xi_z, \xi_\alpha, \xi_\beta, \xi_\gamma]^T$（$\log$ 取主值分支 $\|\vec{\omega}\| \in [0, \pi)$）。密度的形状在切空间坐标下因子化为：
$$ p(\boldsymbol{\xi}) \propto \exp \left( -\frac{1}{2} \left\| [\xi_x, \xi_y, \xi_z, \xi_\alpha, \xi_\beta]^T \right\|_{\boldsymbol{\Sigma}_g^{-1}}^2 \right) \cdot \mathbb{1}[\xi_\gamma \in \mathcal{R}_\gamma] $$
（上式为回退模式；几何先验模式下将 $\xi_\gamma$ 并入马氏距离项，即 $\boldsymbol{\Sigma}_g$ 扩展为 6 维对角阵 $\text{diag}(\sigma_{px}^2, \sigma_{py}^2, \sigma_{pz}^2, \sigma_{\alpha}^2, \sigma_{\beta}^2, \sigma_\gamma^2)$，指示因子去掉）

**数学一致性说明**：
1.  上式是**切空间坐标下的未归一化密度（能量函数）**，用于第二阶段的样本排序与筛选，**不是** $SE(3)$ 流形上的归一化密度——$\delta\boldsymbol{\xi}$ 空间的分布经 $\exp$ 映射诱导到流形时需乘以体积 Jacobian $J(\delta\boldsymbol{\xi})$，大旋转扰动下该因子不可忽略。
2.  **重要性采样的正确用法**：proposal 直接取 4.2 的切空间分布（高斯 × Roll 分布），其密度在 $\delta\boldsymbol{\xi}$ 空间解析可算；计算重要性权重时 proposal 密度取切空间值，**不要**经上式在流形上反求密度。
3.  $\log$ 映射在 $\|\vec{\omega}\| = \pi$（cut locus）处不唯一，实现时需固定主值分支。
## 5. 第一阶段输出与第二阶段对接接口
第一阶段最终不输出单一矩阵，而是输出**参数组 $(T_{base}, \boldsymbol{\Sigma}_g, \mathcal{D}_\gamma)$**（$T_{base}$ 为 $\{W\}$ 下的 $T^W_E$；$\mathcal{D}_\gamma$ 为 Roll 维分布及参数——几何先验模式 $\mathcal{N}(0, \sigma_\gamma^2)$ 或回退模式 $\mathcal{U}(\mathcal{R}_\gamma)$），供第二阶段的数值采样精调进行**重要性采样**：
*   **采样空间定义**：第二阶段在 $T_{base}$ 的**右扰动切空间**（body 系，$(\vec{v}, \vec{\omega})$ 分量顺序）内生成样本，经 $T = T_{base} \cdot \exp(\delta\boldsymbol{\xi}^\wedge)$ 映射到位姿空间。
*   **采样策略指导**：
    *   平移与 $\Delta\alpha, \Delta\beta$（绕开合轴/法向轴）：按 $\boldsymbol{\Sigma}_g$ 小范围高斯采样。
    *   $\Delta\gamma$（绕接近轴 Roll）：按 $\mathcal{D}_\gamma$ 采样——几何先验模式按 $\mathcal{N}(0, \sigma_\gamma^2)$ 小范围采样；回退模式在 $\mathcal{R}_\gamma$ 内**均匀采样**（默认 $[-\pi, \pi)$；夹爪 180° 对称时收缩为 $[0, \pi)$）。
    *   proposal 密度在 $\delta\boldsymbol{\xi}$ 空间解析计算（高斯 × Roll 分布之积），勿在流形上反求。
    *   回退模式下横向实际平移散布 $\approx 0.88\sigma_p$、方向去相关（见 4.2 平移-旋转耦合说明）；几何先验模式 $\approx \sigma_p$。
*   **约束声明**：回退模式下 $T_{base}$ 中的 Roll 基准（$\vec{X}_e, \vec{Y}_e$ 朝向）仅作为切空间坐标的定义参考，受抗奇异参考向量切换影响可能跳变约 90°；第二阶段**不得依赖其连续性**，也不得以其为初值做基于梯度的局部优化。几何先验模式不受此限，但仍建议以扰动分布而非单点基准驱动采样。
*   **无缝衔接**：Roll 维的分布建模从数学上吸收了 Gram-Schmidt 正交化引入的奇异跳变（回退模式），使后续采样可遍历完整的 6-DOF 空间。
