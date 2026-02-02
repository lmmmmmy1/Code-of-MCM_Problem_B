import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ===================== 1. 算法核心配置（新增绿植参数，全可修改） =====================
def init_parameters():
    """初始化所有可配置参数（人居+农业+绿植+输水+运输成本）"""
    params = {
        # ---------------- 人居用水参数 ----------------
        "population": 100000,          # 总人数（人）
        "daily_water_per_person_L": 80, # 人均日用水量（升）
        "human_water_cycle_efficiency": 0.95, # 人居水循环效率（95%）
        # ---------------- 农业用水参数 ----------------
        "farm_area_acre": 60,          # 农业种植面积（亩）
        "farm_water_per_acre_year_m3": 330, # 每亩农业年蓄水需求（立方米）
        "farm_water_efficiency": 0.9,  # 农业水利用率（90%）
        "days_per_year": 365,          # 一年天数（用于农业年转日）
        # ---------------- 绿植用水参数 ----------------
        "green_plant_area_acre": 1200, # 绿植种植面积（亩）
        "green_plant_water_per_acre_day_m3": 1, # 每亩绿植日用水量（立方米）
        "green_plant_water_efficiency": 0.9, # 绿植水利用率（90%）
        # ---------------- 输水参数 ----------------
        "max_input_per_month_ton": 41160, # 每月最大输水量（吨）
        "month_days": 30,               # 每月天数（可适配28/31天）
        "decrease_cycles": 12,          # 从最大值递减到基础值的月数
        "total_months": 18,             # 总规划月数（递减期+稳定期）
        # ---------------- 运输成本参数 ----------------
        "ton_transport_cost_usd": 5823.79, # 每吨水运输成本（美元）
        # ---------------- 初始状态 ----------------
        "initial_water_ton": 0.0        # 初始水量（吨），要求为0
    }
    return params

# ===================== 2. 基础耗水量计算（人居+农业+绿植，合并总耗水） =====================
def calculate_basic_consumption(params):
    """计算核心耗水量：区分人居/农业/绿植，输出总耗水量（统一单位：吨）"""
    # ---------------- 人居耗水量计算 ----------------
    daily_water_per_person_ton = params["daily_water_per_person_L"] / 1000
    human_total_daily_water_ton = params["population"] * daily_water_per_person_ton
    human_daily_consumption_ton = human_total_daily_water_ton * (1 - params["human_water_cycle_efficiency"])
    human_monthly_basic_supply_ton = human_daily_consumption_ton * params["month_days"]

    # ---------------- 农业耗水量计算 ----------------
    farm_year_total_water_ton = params["farm_area_acre"] * params["farm_water_per_acre_year_m3"]
    farm_daily_total_water_ton = farm_year_total_water_ton / params["days_per_year"]
    farm_daily_consumption_ton = farm_daily_total_water_ton * (1 - params["farm_water_efficiency"])
    farm_monthly_basic_supply_ton = farm_daily_consumption_ton * params["month_days"]

    # ---------------- 绿植耗水量计算 ----------------
    # 1立方米=1吨，直接转换
    green_plant_daily_total_water_ton = params["green_plant_area_acre"] * params["green_plant_water_per_acre_day_m3"]
    green_plant_daily_consumption_ton = green_plant_daily_total_water_ton * (1 - params["green_plant_water_efficiency"])
    green_plant_monthly_basic_supply_ton = green_plant_daily_consumption_ton * params["month_days"]

    # ---------------- 总耗水量计算 ----------------
    total_daily_consumption_ton = human_daily_consumption_ton + farm_daily_consumption_ton + green_plant_daily_consumption_ton
    total_monthly_basic_supply_ton = human_monthly_basic_supply_ton + farm_monthly_basic_supply_ton + green_plant_monthly_basic_supply_ton

    # 补充到参数集
    params.update({
        # 人居耗水量
        "human_daily_consumption_ton": human_daily_consumption_ton,
        "human_monthly_basic_supply_ton": human_monthly_basic_supply_ton,
        # 农业耗水量
        "farm_daily_consumption_ton": farm_daily_consumption_ton,
        "farm_monthly_basic_supply_ton": farm_monthly_basic_supply_ton,
        # 绿植耗水量
        "green_plant_daily_consumption_ton": green_plant_daily_consumption_ton,
        "green_plant_monthly_basic_supply_ton": green_plant_monthly_basic_supply_ton,
        # 总耗水量
        "total_daily_consumption_ton": total_daily_consumption_ton,
        "total_monthly_basic_supply_ton": total_monthly_basic_supply_ton
    })

    # 输出计算结果
    print("=== 基础耗水量计算结果（人居+农业+绿植） ===")
    print(f"【人居用水】")
    print(f"  人均日用水量：{daily_water_per_person_ton:.4f} 吨/人/天")
    print(f"  日耗水量（需补充）：{human_daily_consumption_ton:.2f} 吨/天 | 月基础补水量：{human_monthly_basic_supply_ton:.2f} 吨/月")
    print(f"【农业用水】")
    print(f"  日耗水量（需补充）：{farm_daily_consumption_ton:.4f} 吨/天 | 月基础补水量：{farm_monthly_basic_supply_ton:.2f} 吨/月")
    print(f"【绿植用水】")
    print(f"  日总需水量：{green_plant_daily_total_water_ton:.2f} 吨/天 | 日耗水量（需补充）：{green_plant_daily_consumption_ton:.2f} 吨/天")
    print(f"  月基础补水量：{green_plant_monthly_basic_supply_ton:.2f} 吨/月")
    print(f"【总用水】")
    print(f"  总日耗水量（需补充）：{total_daily_consumption_ton:.4f} 吨/天 | 总月基础补水量：{total_monthly_basic_supply_ton:.2f} 吨/月")
    return params

# ===================== 3. 生成递减输水量（基于总耗水，首月最大逐步递减） =====================
def generate_decreasing_input(params):
    """生成逐月递减的输水量列表（基于总基础补水量）"""
    max_input = params["max_input_per_month_ton"]
    total_basic_supply = params["total_monthly_basic_supply_ton"]
    decrease_cycles = params["decrease_cycles"]
    total_months = params["total_months"]
    
    # 计算递减步长（线性递减）
    step = (max_input - total_basic_supply) / (decrease_cycles - 1) if decrease_cycles > 1 else 0
    
    # 生成逐月输水量
    monthly_input = []
    for month in range(1, total_months + 1):
        if month <= decrease_cycles:
            current_input = max_input - (month - 1) * step
            current_input = max(round(current_input, 2), total_basic_supply)
        else:
            current_input = round(total_basic_supply, 2)
        monthly_input.append(current_input)
    
    params["monthly_input_list"] = monthly_input
    print(f"\n=== 递减输水量生成结果 ===")
    print(f"递减周期：{decrease_cycles} 个月 | 递减步长：{step:.4f} 吨/月")
    print(f"首月输水量：{monthly_input[0]} 吨 | 稳定期月输水量：{total_basic_supply:.2f} 吨")
    return params

# ===================== 4. 运输成本计算（新增核心模块，逐月+累计+稳定期） =====================
def calculate_transport_cost(params):
    """计算运输成本：逐月成本、累计成本、稳定期成本，关联输水量"""
    monthly_input = params["monthly_input_list"]
    ton_cost = params["ton_transport_cost_usd"]
    total_basic_supply = params["total_monthly_basic_supply_ton"]
    
    # 计算逐月运输成本（当月输水量 × 每吨成本）
    monthly_transport_cost = [round(input_ton * ton_cost, 2) for input_ton in monthly_input]
    
    # 计算累计运输成本（累加）
    cumulative_transport_cost = []
    current_cum = 0.0
    for cost in monthly_transport_cost:
        current_cum += cost
        cumulative_transport_cost.append(round(current_cum, 2))
    
    # 计算稳定期相关成本
    stable_monthly_cost = round(total_basic_supply * ton_cost, 2)  # 稳定期单月成本
    total_transport_cost = cumulative_transport_cost[-1]          # 模拟期总运输成本
    avg_monthly_cost = round(total_transport_cost / len(monthly_input), 2)  # 模拟期平均月成本
    
    # 补充到参数集
    params.update({
        "monthly_transport_cost": monthly_transport_cost,  # 逐月运输成本（美元）
        "cumulative_transport_cost": cumulative_transport_cost,  # 累计运输成本（美元）
        "stable_monthly_transport_cost": stable_monthly_cost,    # 稳定期单月成本（美元）
        "total_transport_cost": total_transport_cost,            # 模拟期总成本（美元）
        "avg_monthly_transport_cost": avg_monthly_cost            # 模拟期平均月成本（美元）
    })
    
    # 输出成本计算结果
    print(f"\n=== 运输成本核心计算结果（每吨{ton_cost}美元） ===")
    print(f"稳定期月输水量：{total_basic_supply:.2f} 吨 | 稳定期单月运输成本：{stable_monthly_cost:,} 美元")
    print(f"模拟期总输水量：{sum(monthly_input):.2f} 吨 | 模拟期总运输成本：{total_transport_cost:,} 美元")
    print(f"模拟期平均月运输成本：{avg_monthly_cost:,} 美元")
    return params

# ===================== 5. 水量+成本模拟（核心算法，整合所有数据） =====================
def simulate_water_and_cost(params):
    """模拟每月水量变化+关联运输成本，返回含成本的详细时间表"""
    initial_water = params["initial_water_ton"]
    total_daily_consumption = params["total_daily_consumption_ton"]
    month_days = params["month_days"]
    monthly_input = params["monthly_input_list"]
    monthly_transport_cost = params["monthly_transport_cost"]
    cumulative_transport_cost = params["cumulative_transport_cost"]
    total_months = params["total_months"]
    
    # 初始化模拟变量
    current_water = initial_water  # 初始水量为0
    total_days = 0
    schedule = []
    
    # 逐月模拟（水量+成本关联）
    for month in range(1, total_months + 1):
        idx = month - 1
        input_ton = monthly_input[idx]
        month_cost = monthly_transport_cost[idx]
        cum_cost = cumulative_transport_cost[idx]
        
        # 当月输水后更新水量
        current_water += input_ton
        month_start_water = round(current_water, 2)
        
        # 模拟当月每日水量消耗
        daily_remaining = []
        for day in range(1, month_days + 1):
            current_water = max(round(current_water, 4), 0.0)
            daily_remaining.append(current_water)
            current_water -= total_daily_consumption
            total_days += 1
        
        # 月末剩余水量（防护：避免负数）
        month_end_water = max(round(daily_remaining[-1], 2), 0.0)
        
        # 记录当月数据（含水量+成本）
        schedule.append({
            "月份": month,
            "累计天数（月初）": total_days - month_days + 1,
            "当月输水量（吨）": input_ton,
            "当月运输成本（美元）": month_cost,
            "累计运输成本（美元）": cum_cost,
            "输水后初始水量（吨）": month_start_water,
            "月末剩余水量（吨）": month_end_water,
            "是否断水": month_end_water < 0
        })
    
    # 转换为DataFrame，方便查看
    schedule_df = pd.DataFrame(schedule)
    params["water_cost_schedule_df"] = schedule_df
    
    # 验证是否断水
    has_water_shortage = any(schedule_df["是否断水"])
    params["water_shortage"] = has_water_shortage
    print(f"\n=== 水量+成本模拟结果 ===")
    print(f"初始水量：{initial_water} 吨 | 模拟总月数：{total_months} 个月")
    print(f"是否存在断水风险：{'是' if has_water_shortage else '否'}")
    return params

# ===================== 6. 结果输出+可视化（英文+三张独立图，含绿植占比） =====================
def output_and_visualize(params):
    """输出结构化结果+可视化趋势（水量+输水量+运输成本，英文+三张独立图）"""
    schedule_df = params["water_cost_schedule_df"]
    monthly_input = params["monthly_input_list"]
    monthly_transport_cost = params["monthly_transport_cost"]
    cumulative_transport_cost = params["cumulative_transport_cost"]
    total_months = params["total_months"]
    # 核心基础数据
    total_basic_supply = params["total_monthly_basic_supply_ton"]
    max_input = params["max_input_per_month_ton"]
    stable_monthly_cost = params["stable_monthly_transport_cost"]
    ton_cost = params["ton_transport_cost_usd"]
    # 人居/农业/绿植占比
    human_monthly = params["human_monthly_basic_supply_ton"]
    farm_monthly = params["farm_monthly_basic_supply_ton"]
    green_plant_monthly = params["green_plant_monthly_basic_supply_ton"]
    
    # 1. 输出含成本的核心时间表
    print(f"\n=== 最终输水+运输成本时间表（人居+农业+绿植总需求） ===")
    core_columns = [
        "月份", "累计天数（月初）", "当月输水量（吨）",
        "当月运输成本（美元）", "累计运输成本（美元）",
        "月末剩余水量（吨）"
    ]
    # 格式化美元为千分位，更易读
    schedule_df_display = schedule_df.copy()
    schedule_df_display["当月运输成本（美元）"] = schedule_df_display["当月运输成本（美元）"].apply(lambda x: f"{x:,}")
    schedule_df_display["累计运输成本（美元）"] = schedule_df_display["累计运输成本（美元）"].apply(lambda x: f"{x:,}")
    print(schedule_df_display[core_columns].to_string(index=False))
    
    # 定义x轴（月份）
    month_x = range(1, total_months + 1)
    # 月末剩余水量列表
    month_end_water = schedule_df["月末剩余水量（吨）"].tolist()

    # ===================== 图1：Monthly Water Input Volume Trend（逐月输水量趋势）=====================
    plt.figure(figsize=(14, 7))
    ax1 = plt.gca()
    ax1.plot(month_x, monthly_input, marker="o", color="#2E86AB", linewidth=2, label="Monthly Water Input")
    # 稳定期输水量线（含人居+农业+绿植占比）
    ax1.axhline(y=total_basic_supply, color="red", linestyle="--", 
                label=f"Stable Period Input: {total_basic_supply:.2f}t\n(Human: {human_monthly:.2f}t + Farm: {farm_monthly:.2f}t + Green Plant: {green_plant_monthly:.2f}t)")
    # 最大输水量线
    ax1.axhline(y=max_input, color="orange", linestyle="--", label=f"Max Monthly Input: {max_input}t")
    # 轴与标题配置
    ax1.set_title("Monthly Water Input Volume Trend (Human + Farm + Green Plant Demand)", fontsize=14, fontweight="bold", pad=20)
    ax1.set_xlabel("Month", fontsize=12)
    ax1.set_ylabel("Water Input Volume (ton)", fontsize=12)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(month_x)
    # 调整布局
    plt.tight_layout()
    # 显示图1
    plt.show()

    # ===================== 图2：Monthly End Remaining Water Volume（月末剩余水量，清晰显示警戒线）=====================
    plt.figure(figsize=(14, 7))
    ax2 = plt.gca()
    ax2.bar(month_x, month_end_water, color="#7E99F4", alpha=0.7, label="Monthly End Remaining Water", width=0.6)
    # 水位警戒线（设为1000吨，避免与X轴重合，更易识别）
    warning_level = 56000
    ax2.axhline(y=warning_level, color="red", linestyle="--", linewidth=2, label=f"Water Warning Level: {warning_level}t")
    # 调整Y轴范围，确保警戒线可见
    ax2.set_ylim(bottom=0)
    # 轴与标题配置
    ax2.set_title("Monthly End Remaining Water Volume", fontsize=14, fontweight="bold", pad=20)
    ax2.set_xlabel("Month", fontsize=12)
    ax2.set_ylabel("Remaining Water Volume (ton)", fontsize=12)
    ax2.legend(loc="lower right", fontsize=10)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_xticks(month_x)
    # 调整布局
    plt.tight_layout()
    # 显示图2
    plt.show()

    # ===================== 图3：Transportation Cost Trend（运输成本趋势，双Y轴）=====================
    plt.figure(figsize=(14, 7))
    ax3 = plt.gca()
    # 双Y轴：右侧累计成本
    ax3_twin = ax3.twinx()
    # 柱状：逐月运输成本
    ax3.bar(month_x, monthly_transport_cost, color="#F18F01", alpha=0.6, label="Monthly Transportation Cost", width=0.6)
    # 折线：累计运输成本
    ax3_twin.plot(month_x, cumulative_transport_cost, marker="s", color="#C73E1D", linewidth=2, label="Cumulative Transportation Cost")
    # 稳定期成本线
    ax3.axhline(y=stable_monthly_cost, color="#8E44AD", linestyle="--", linewidth=2, label=f"Stable Period Monthly Cost: ${stable_monthly_cost:,}")
    # 轴与标题配置
    ax3.set_title(f"Water Transportation Cost Trend (${ton_cost:.2f} per Ton)", fontsize=14, fontweight="bold", pad=20)
    ax3.set_xlabel("Month", fontsize=12)
    ax3.set_ylabel("Monthly Transportation Cost (USD)", color="#F18F01", fontsize=12)
    ax3_twin.set_ylabel("Cumulative Transportation Cost (USD)", color="#C73E1D", fontsize=12)
    # 轴颜色匹配
    ax3.tick_params(axis="y", labelcolor="#F18F01", labelsize=10)
    ax3_twin.tick_params(axis="y", labelcolor="#C73E1D", labelsize=10)
    ax3.tick_params(axis="x", labelsize=10)
    # 合并图例
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
    ax3.grid(True, alpha=0.3, axis="x")
    ax3.set_xticks(month_x)
    # 调整布局
    plt.tight_layout()
    # 显示图3
    plt.show()
    
    # 输出最终核心结论（水量+成本双维度，含绿植）
    print(f"\n=== 算法最终核心结论（人居+农业+绿植+运输成本） ===")
    print(f"【水量维度】")
    print(f"  1. 首月输水量：{monthly_input[0]} 吨 | 首月月末剩余水量：{month_end_water[0]:.2f} 吨")
    print(f"  2. 递减期{params['decrease_cycles']}个月后，月输水量稳定为：{total_basic_supply:.2f} 吨")
    print(f"     - 人居占比：{human_monthly/total_basic_supply*100:.2f}%（{human_monthly:.2f}吨）")
    print(f"     - 农业占比：{farm_monthly/total_basic_supply*100:.2f}%（{farm_monthly:.2f}吨）")
    print(f"     - 绿植占比：{green_plant_monthly/total_basic_supply*100:.2f}%（{green_plant_monthly:.2f}吨）")
    print(f"  3. 模拟期内最大剩余水量：{max(month_end_water):.2f} 吨 | 最小剩余水量：{min(month_end_water):.2f} 吨")
    print(f"  4. 模拟期总输水量：{sum(monthly_input):.2f} 吨")
    print(f"【成本维度（每吨{ton_cost}美元）】")
    print(f"  1. 首月运输成本：{monthly_transport_cost[0]:,} 美元 | 稳定期单月运输成本：{stable_monthly_cost:,} 美元")
    print(f"  2. 模拟期{total_months}个月总运输成本：{params['total_transport_cost']:,} 美元")
    print(f"  3. 模拟期平均月运输成本：{params['avg_monthly_transport_cost']:,} 美元")
    print(f"  4. 模拟期末累计运输成本：{cumulative_transport_cost[-1]:,} 美元")
    return

# ===================== 7. 算法主入口（整合所有模块，一键运行） =====================
def run_full_algorithm():
    """运行完整算法流程：人居+农业+绿植用水+运输成本+水量模拟"""
    print("========== 封闭环境用水+运输成本算法启动（人居+农业+绿植） ==========\n")
    # 步骤1：初始化所有参数
    params = init_parameters()
    # 步骤2：计算人居+农业+绿植基础耗水量
    params = calculate_basic_consumption(params)
    # 步骤3：生成逐月递减输水量
    params = generate_decreasing_input(params)
    # 步骤4：计算运输成本（逐月+累计+稳定期）
    params = calculate_transport_cost(params)
    # 步骤5：模拟水量变化+关联运输成本
    params = simulate_water_and_cost(params)
    # 步骤6：输出结果+可视化（英文+三张独立图）
    if params["water_shortage"]:
        print("\n⚠️  警告：模拟期内存在断水风险，请调整参数（如增加输水量/延长递减周期）！")
    else:
        output_and_visualize(params)
    print("\n========== 封闭环境用水+运输成本算法结束 ==========")
    return params

# ===================== 执行完整算法（一键运行） =====================
if __name__ == "__main__":
    final_params = run_full_algorithm()
