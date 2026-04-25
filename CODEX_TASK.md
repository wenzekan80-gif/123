你现在是我的数学建模程序手。请读取 data/ 目录下的赛题 PDF 和四个 Excel 文件，完成第十八届华中杯 A 题《城市绿色物流配送调度》的可复现实验代码。

## 核心任务

本题是异构车队车辆路径优化 / VRPTW / 可拆分配送 / 绿色成本 / 限行政策 / 动态重调度问题。

### 车辆参数

1. 燃油车1型：载重 3000kg，容积 13.5m³，数量 60
2. 燃油车2型：载重 1500kg，容积 10.8m³，数量 50
3. 燃油车3型：载重 1250kg，容积 6.5m³，数量 50
4. 新能源车1型：载重 3000kg，容积 15m³，数量 10
5. 新能源车2型：载重 1250kg，容积 8.5m³，数量 15

所有车辆启动成本为 400 元/辆。

### 时间窗

客户有软时间窗，早到等待成本 20 元/小时，晚到惩罚成本 50 元/小时，服务时间 20 分钟。

### 速度

优先使用均值作为确定性速度：

- 顺畅：9:00-10:00、13:00-15:00，55.3 km/h
- 一般：10:00-11:30、15:00-17:00，35.4 km/h
- 拥堵：8:00-9:00、11:30-13:00，9.8 km/h
- 补充：17:00-19:00 按一般，19:00 后按顺畅

### 能耗与排放

燃油车每百公里油耗：

FPK(v)=0.0025*v^2 - 0.2554*v + 31.75

新能源车每百公里电耗：

EPK(v)=0.0014*v^2 - 0.12*v + 36.19

燃油车满载能耗比空载高 40%，新能源车满载能耗比空载高 35%，按当前载重率线性修正。

- 油价：7.61 元/L
- 电价：1.64 元/kW·h
- 燃油碳排放系数：2.547 kg/L
- 电耗碳排放系数：0.501 kg/kW·h
- 碳排放成本：0.65 元/kg

### 目标函数

问题一和问题二最小化：

C_total = C_start + C_energy + C_time + C_carbon

分别输出启动成本、能耗成本、时间窗惩罚成本、碳排放成本、总行驶里程、总碳排放量、时间窗满足率。

## 问题一

在无政策限制下，建立考虑车辆类型、载重体积约束、时间窗约束与速度时变特性的车辆调度模型，以总配送成本最低为目标，输出车辆使用方案、行驶路径、到达时间及成本构成。

要求允许可拆分配送：若单客户需求超过任一车辆容量，必须拆分；若未超过，优先单车服务。

## 问题二

加入绿色配送区限行政策：8:00-16:00 禁止燃油车进入以市中心 (0,0) 为圆心、半径 10 km 的绿色配送区。

请按客户坐标判定绿色配送区：sqrt(x^2+y^2) <= 10。

限行必须作为硬约束进入求解过程。任何燃油车在 8 <= arrival_time < 16 访问绿色区客户均不可行。新能源车不受限制。

必须输出 policy_violation_check.csv，最终 violation 全为 False。

## 问题三

设计事件驱动的动态重调度策略。至少模拟两个场景：

- 场景 A：9:30 出现订单取消 + 新增订单
- 场景 B：10:00 出现客户时间窗提前 + 地址变更

事件发生后锁定已完成任务，更新车辆当前位置、当前时间、剩余载重、剩余容积，对未完成订单局部重调度。动态目标为：remaining_cost + lambda * disruption_cost。扰动成本包括改派客户数、路径变化里程、新增车辆数、客户到达时间变化量。

## 推荐算法

主算法使用改进 ALNS：

1. 数据预处理
2. 时空聚类初始化
3. 最近邻/节约法/贪心插入构造初始路径
4. ALNS 优化
5. 模拟退火接受准则
6. 问题二加入政策感知修复
7. 问题三事件驱动局部重调度

破坏算子建议：随机移除、高成本客户移除、时间窗冲突客户移除、相近客户批量移除、绿色区客户移除。

修复算子建议：贪心插入、遗憾插入、时间窗优先插入、碳排放优先插入、政策感知插入、可拆分配送插入。

## 输出文件

请输出：

- output/problem1/problem1_vehicle_routes.csv
- output/problem1/problem1_customer_arrival_times.csv
- output/problem1/problem1_cost_breakdown.csv
- output/problem1/problem1_vehicle_usage.csv
- output/problem1/problem1_summary.json
- output/problem1/problem1_convergence.csv
- output/problem1/problem1_cost_breakdown.png
- output/problem1/problem1_convergence.png

- output/problem2/problem2_vehicle_routes.csv
- output/problem2/problem2_customer_arrival_times.csv
- output/problem2/problem2_cost_breakdown.csv
- output/problem2/problem2_vehicle_usage.csv
- output/problem2/problem2_emission_summary.csv
- output/problem2/problem2_policy_violation_check.csv
- output/problem2/problem2_summary.json
- output/problem2/problem2_cost_compare.png
- output/problem2/problem2_vehicle_structure_compare.png
- output/problem2/problem2_emission_compare.png

- output/problem3/scenario_A_before.csv
- output/problem3/scenario_A_after.csv
- output/problem3/scenario_A_compare.csv
- output/problem3/scenario_B_before.csv
- output/problem3/scenario_B_after.csv
- output/problem3/scenario_B_compare.csv
- output/problem3/problem3_summary.json

- output/ablation/ablation_results.csv
- output/ablation/ablation_convergence.png

- output/sensitivity/carbon_price_sensitivity.csv
- output/sensitivity/ev_quantity_sensitivity.csv
- output/sensitivity/green_radius_sensitivity.csv

## 消融实验

至少比较：Greedy baseline、ALNS_random_init、ALNS_no_adaptive、ALNS_no_SA、Full_ALNS。输出最终总成本、运行时间、最优解出现迭代数、时间窗满足率、碳排放量、可行解比例。

## 敏感性分析

至少做三类：碳排放成本系数（0.5、0.75、1、1.25、1.5倍）、新能源车数量（0.8、1、1.2、1.4倍）、绿色配送区半径（8、10、12、15km）。

## 代码质量

使用 Python，优先依赖 pandas、numpy、openpyxl、scikit-learn、matplotlib。不要依赖商业求解器。所有随机过程设置 random seed=42。建议代码结构：data_loader.py、cost_model.py、constraints.py、alns_solver.py、policy_solver.py、dynamic_solver.py、experiments.py、visualization.py、main.py。

最终运行 `python main.py` 后自动生成所有结果。

优先保证正确性和可复现性，不要为了炫技牺牲约束可行性。
