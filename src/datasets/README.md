# datasets 目录是干什么的？

一句话：这里负责把 raw 的 NetCDF（月文件）变成“训练/评估可直接用的样本”。

你可以把它看成两步：

## 第 1 步：读一个月（sic_reader.py）
每个 .nc 文件里面变量名可能不同：
- 1979 年：N07_ICECON
- 2000 年：F13_ICECON

但是它们都是“海冰浓度”。

所以 `sic_reader.py` 做的事是：
- 自动找到 `*_ICECON` 这个变量
- 去掉 time=1 这个维度
- 返回一个二维数组 `(y, x)`

输出格式（固定）：
- 单月 SIC：shape = (448, 304)

## 第 2 步：切成训练样本（sic_dataset.py）
模型不是一次看一个月，而是看一段历史。

我们定义两个参数：

- input_window：输入要看多少个月
- lead_time：要预测几个月以后

例子：
- input_window=3, lead_time=1
  - 输入 X：3 个月（t-2, t-1, t）
  - 输出 Y：下一个月（t+1）

所以 `SICWindowDataset` 返回：
- X：shape = (input_window, 448, 304)
- Y：shape = (448, 304)
- meta：告诉你这条样本从哪个月到哪个月

## 为什么要这样设计？
因为 baseline、深度学习模型、物理一致性模型都应该用同一套样本。
这样你做出来的对比才公平、可复现。
