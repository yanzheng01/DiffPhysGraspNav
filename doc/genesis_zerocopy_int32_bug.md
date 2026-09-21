# Bug 记录：Genesis 1.3.3 zerocopy 路径 torch.gather int32 索引崩溃

## 概述

Genesis 1.3.3 的 `collider.get_contacts()` zerocopy 快速路径将 int32 的接触排序
索引直接传给 `torch.gather`，违反其 int64 索引契约，导致接触查询在默认配置下
必然崩溃。已在本地 site-packages 打一行补丁修复（索引升型 int64）。

## 环境信息

| 项 | 值 |
|---|---|
| genesis | 1.3.3 |
| Python | 3.12（mamba/conda 环境 `diffphy`） |
| torch | < 2.8.0（genesis 1.3.3 官方基线以下，触发兼容警告但可运行） |
| 补丁文件 | `genesis/engine/solvers/rigid/collider/collider.py`（site-packages 内） |
| 补丁位置 | `Collider.get_contacts()` 的 zerocopy 分支，L1017–L1021 |

## 现象

任何触发 zerocopy 路径的 `RigidEntity.get_contacts()` 调用抛出：

```
RuntimeError: gather(): Expected dtype int64 for index
```

复现条件：
- `GS_ENABLE_ZEROCOPY` 未设为 `0`（zerocopy 缺省**开启**，`gs.use_zerocopy=True`）；
- 场景已产生接触数据（`self._contact_data` 非 None）；
- 接触数据未走 `zerocopy_aligned` 对齐路径（一般对齐路径）。

受影响的功能：
- `grasp_detector.GraspDetector` 推理判定（核心数据源即 `get_contacts()`）；
- `RigidEntity.get_links_net_contact_force()`、ContactForce / Contact 传感器
  （内部同样调用 `collider.get_contacts()`）；
- 任何用户代码直接调用 `get_contacts()`。

## 根因分析

两边类型契约不匹配：

- **torch 侧**：`torch.gather` 硬性要求 `index` 参数为 int64（LongTensor），
  这是 PyTorch 稳定 API 契约；
- **Genesis 侧**：`contact_sort_idx` 是 quadrants 字段，声明为 `gs.qd_int`
  = **int32**。zerocopy 路径经 `qd_to_torch(..., copy=False)` 把它转为零拷贝
  torch 视图后直接切片作为 gather 索引，未做升型。

zerocopy 是较新引入的性能优化路径（零拷贝共享内存直读 quadrants 字段），
与 torch.gather 的类型契约没有对齐，且默认开启，属于 Genesis 侧 bug，
与本项目代码无关。

## 修复补丁

文件：
`<site-packages>/genesis/engine/solvers/rigid/collider/collider.py`
（本机：`/home/yanzheng/miniforge3/envs/diffphy/lib/python3.12/site-packages/genesis/engine/solvers/rigid/collider/collider.py`）

原代码（L1017）：

```python
sort_idx_view = qd_to_torch(self._collider_state.contact_sort_idx, transpose=True, copy=False)
gather_idx_flat = sort_idx_view[:, :n_contacts_max]
```

修复后（L1017–L1021）：

```python
sort_idx_view = qd_to_torch(self._collider_state.contact_sort_idx, transpose=True, copy=False)
# torch.gather demands an int64 index; the sort permutation is
# stored as int32, so upcast a copy (fixes "Expected dtype
# int64 for index").
gather_idx_flat = sort_idx_view[:, :n_contacts_max].to(torch.int64)
```

要点：仅对送入 `gather` 的索引切片做 `.to(torch.int64)` 无损升型（排序索引
只是排列置换，int32 表示范围绰绰有余），quadrants 原字段与其余 zerocopy
视图不动。

## 替代方案（不打补丁时）

```bash
export GS_ENABLE_ZEROCOPY=0
```

关闭 zerocopy 后接触查询走通用拷贝路径，功能正确。局限：
- 只对设置了该环境变量的进程生效，传感器内部调用、其他脚本不受保护；
- 放弃零拷贝共享内存的性能优势。

## 安全性与验证

- **数值语义不变**：int32→int64 为无损升型，接触查询结果与通用路径一致
  （probe 已对照验证）；
- **不影响训练路径**：可微训练使用 `Scene.get_state()`，不经过
  `get_contacts()`；
- **回归验证**：打补丁并移除 `GS_ENABLE_ZEROCOPY=0` 绕行后，
  `scripts/stage2_grasp_detector_check.py` 30 项检查全部通过
  （Part A 推理 19 项 + Part B 训练 11 项）。

## 注意事项

- 补丁位于 site-packages，**升级 / 重装 genesis 或更换环境后会消失**，
  届时同样的崩溃会重现，需重新打补丁或使用环境变量绕行；
- 建议向 genesis 上游报告该 bug（修复仅需一行升型）；
- 相关记录另见 `doc/stage2.md` §11（排查表最后一行）。
