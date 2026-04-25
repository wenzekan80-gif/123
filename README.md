# 华中杯 A 题：城市绿色物流配送调度

本仓库用于让 Codex 直接读取赛题 PDF 与四个 Excel 数据文件，完成城市绿色物流配送调度建模代码。

## 数据文件

- `data/A题：城市绿色物流配送调度.pdf`：题干
- `data/订单信息.xlsx`：2169 条订单信息，包含订单编号、重量、体积、目标客户编号
- `data/距离矩阵.xlsx`：配送中心 0 与客户 1-98 的实际道路距离矩阵
- `data/客户坐标信息.xlsx`：配送中心和客户坐标，市中心坐标为 (0,0)
- `data/时间窗.xlsx`：客户 1-98 的最早/最晚到达时间

## 推荐任务

请让 Codex 读取 `CODEX_TASK.md`，按其中要求实现数据预处理、问题一静态调度、问题二限行政策调度、问题三动态重调度、消融实验和敏感性分析。

## 建议依赖

```bash
pip install pandas numpy openpyxl scikit-learn matplotlib
```

## 运行目标

最终希望通过：

```bash
python main.py
```

自动生成 `output/` 下所有结果表和图。
