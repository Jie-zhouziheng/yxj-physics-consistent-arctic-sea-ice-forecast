# 总体策略（先说清楚方向）

你现在的状态非常理想：

* ✔ **论文 story、创新点、结构已经定型**（你现在这版是“能投”的）
* ✔ 方法是 **轻量、可解释、工程友好**
* ✔ GitHub 项目已经存在（不是从 0 开始）

👉 接下来不是“疯狂加新东西”，而是：

> **把“论文里的每一句关键话”，都变成代码里“能跑、能复现、能画图”的事实。**

---

# 阶段 0（现在～2 天）：工程与论文“对齐冻结”

## 🎯 目标

确保 **论文 ≠ 空中楼阁**，工程结构与论文 Method 完全一致。

## ✅ 必做任务

### 0.1 冻结论文方法设定（不要再大改）

* 输入变量：**SIC + SIT proxy + SAT anomaly**
* 预测任务：**1–6 month lead**
* 模型结构：

  * shared encoder
  * winter pooling
  * spring ConvLSTM
  * physics-consistent loss（spatial + temporal）

👉 这四点 **从现在开始是“铁律”**

---

### 0.2 快速 audit 你的 GitHub 项目结构

你的仓库：
👉 [https://github.com/LTL-77/physics-consistent-arctic-sea-ice-forecast](https://github.com/LTL-77/physics-consistent-arctic-sea-ice-forecast)

你现在要确认（不一定已有）：

```text
project/
├── data/
│   ├── raw/
│   ├── processed/
├── models/
│   ├── encoder.py
│   ├── convlstm.py
│   ├── model.py
├── losses/
│   ├── physics_loss.py
├── train.py
├── evaluate.py
├── configs/
│   └── default.yaml
```

📌 **如果现在结构很乱：**

> 先别写新模型，先重构目录（这是最省时间的投资）

---

## 📦 阶段产出

* 工程结构和论文 Method **一一对应**
* README 写一句话：

  > “This repository implements the season-aware, physics-consistent lightweight framework proposed in our TGRS submission.”

---

# 阶段 1（第 1–2 周）：Baseline 全部跑通（非常重要）

## 🎯 目标

让 **Experiments 表格里所有 baseline 都是真实跑出来的**。

---

## ✅ 必做 baseline（按优先级）

### 1.1 统计基线（1 天）

* Climatology
* Persistence

📌 这一步非常重要，因为：

* 审稿人最信这两个
* 可以快速验证数据没问题

---

### 1.2 深度学习 baseline（3–5 天）

必须有：

* U-Net（时间堆叠）
* ConvLSTM（uniform temporal）

**注意**：

* 输入变量要和你方法一致（否则不公平）
* 网络深度/参数量别比你方法大太多

---

## 📦 阶段产出

* 表 I（Main Results）中：

  * Climatology
  * U-Net
  * Uniform ConvLSTM
    都有数值

* GitHub：

  * `scripts/run_baselines.sh`
  * baseline checkpoint 能复现

---

# 阶段 2（第 3–4 周）：实现你的“核心方法”

## 🎯 目标

**把两个创新点 100% 落地成代码**

---

## 🔑 创新点 1：关键变量 + 物理一致性（优先）

### 2.1 Physics-consistent loss（2–3 天）

* spatial TV loss
* temporal smoothness loss
* mask handling（冰区）

📌 注意：

* loss 权重必须写在 config
* 不要 hard-code

---

### 2.2 变量消融接口

你必须能做到：

```yaml
use_sic: true
use_sit: false
use_sat: true
use_physics_loss: false
```

👉 为 **Ablation Table II** 服务

---

## 🔑 创新点 2：Season-aware 时序建模（重中之重）

### 2.3 Winter pooling + Spring ConvLSTM（3–4 天）

关键点：

* winter months index 明确
* spring months index 明确
* winter → pooled feature
* pooled feature → ConvLSTM init

📌 一定要能开关：

```yaml
season_aware: true / false
```

---

## 📦 阶段产出

* Full model 跑通
* 和 uniform ConvLSTM **可直接对比**
* Ablation 至少跑 2–3 个关键版本

---

# 阶段 3（第 5–6 周）：实验补全 + 图表生成

## 🎯 目标

**填满论文里所有 “?? / 待填数值”**

---

## ✅ 必做实验

### 3.1 Lead-time 分析

* 1–6 month RMSE / MAE 曲线
* 特别标注：

  * 跨春季 vs 非跨春季

---

### 3.2 空间可视化（非常重要）

* 同一个月份：

  * GT
  * U-Net
  * Uniform ConvLSTM
  * Ours

📌 冰缘一定要画清楚
📌 这是 TGRS 审稿人最爱看的图

---

### 3.3 消融实验（对应论文结构）

你论文里写了的，就必须有：

* w/o SIT
* w/o SAT
* w/o physics loss
* w/o season-aware

---

## 📦 阶段产出

* 所有表格填满
* Figure 编号能直接放进 LaTeX
* GitHub `results/` 目录清晰

---

# 阶段 4（并行进行）：论文精修（不要等实验全完）

## 🎯 目标

把论文从“能投” → “像 TGRS 论文”

---

## 建议修改顺序

1. **Abstract**（等实验稳定后微调数字）
2. **Experiments**（和结果严格对齐）
3. **Discussion**（强化“为什么有效”）
4. **Introduction 最后一段**（再次对齐贡献）

📌 原则：

> **实验不是证明你对，而是证明你没吹。**

---

# 阶段 5（投稿前 1 周）：审稿人视角自检

## 🎯 目标

防“致命 reviewer comment”

### Checklist

* ❓“季节是不是拍脑袋？”
* ❓“为什么不用 Transformer？”
* ❓“算力受限是否真实？”

👉 这些你现在**都有答案**，只要确保论文里写清楚。

