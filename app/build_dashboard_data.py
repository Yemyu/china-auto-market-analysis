#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
中国汽车市场分析看板数据桥 (data bridge)
把 stage3-5 的 CSV 预聚合为数组化 JSON，落到 app/static/data/。
看板只认这些 JSON 的字段名；上游(阶段三/四/五)任何"换数字/换模型/加维度"
的改动，只要列名结构不变，重跑本脚本即可，前端代码零改动。

设计要点（前向兼容）:
  - 模型对比用数组 models:[{name, wmape_vol, ...}] 而非写死列 -> 加模型只是多一条数据
  - 情感维度用数组 aspects:[{key, name_zh, name_en, avg}] -> 加第 11 维只是多一个对象
  - 预警用数组 alerts:[{...}] -> 加规则只是多几条
"""
import os
import re
import json
import math
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
PROC_NEW = os.path.join(ROOT, "data", "processed_new")
SENTIMENT_NEW = os.path.join(ROOT, "data", "sentiment_new", "processed")
OUT = os.path.join(ROOT, "app", "static", "data")

# 销量预测特征名的业务映射（中英 + 一句话解释）
FEATURE_NAME = {
    "lag_1": {"zh": "上月销量", "en": "Sales (last month)"},
    "roll_mean_3": {"zh": "近3月销量均值", "en": "Sales 3-month average"},
    "roll_mean_6": {"zh": "近6月销量均值", "en": "Sales 6-month average"},
    "lag_2": {"zh": "前2月销量", "en": "Sales 2 months ago"},
    "lag_3": {"zh": "前3月销量", "en": "Sales 3 months ago"},
    "month_cos": {"zh": "月份季节性（余弦）", "en": "Month seasonality (cosine)"},
    "month_sin": {"zh": "月份季节性（正弦）", "en": "Month seasonality (sine)"},
    "energy_type_enc": {"zh": "能源类型", "en": "Energy type"},
    "vehicle_class_enc": {"zh": "车辆级别", "en": "Vehicle class"},
    "positive_ratio": {"zh": "正面评论占比", "en": "Positive review ratio"},
    "negative_ratio": {"zh": "负面评论占比", "en": "Negative review ratio"},
    "intelligence_lag2": {"zh": "智能化情感（滞后2月）", "en": "Intelligence sentiment (lag 2M)"},
    "comfort_lag1": {"zh": "舒适性情感（滞后1月）", "en": "Comfort sentiment (lag 1M)"},
    "value_lag1": {"zh": "性价比情感（滞后1月）", "en": "Value sentiment (lag 1M)"},
    "overall": {"zh": "综合口碑", "en": "Overall sentiment"},
    "overall_lag1": {"zh": "综合口碑（滞后1月）", "en": "Overall sentiment (lag 1M)"},
    "overall_lag2": {"zh": "综合口碑（滞后2月）", "en": "Overall sentiment (lag 2M)"},
    "overall_lag3": {"zh": "综合口碑（滞后3月）", "en": "Overall sentiment (lag 3M)"},
}
FEATURE_DESC = {
    "lag_1": {"zh": "销量自身惯性：上个月卖多少，下个月大概率延续", "en": "Sales inertia: last month's sales strongly predict next month"},
    "roll_mean_3": {"zh": "平滑短期波动，捕捉季度趋势", "en": "Smooths short-term noise and captures quarterly trend"},
    "roll_mean_6": {"zh": "平滑中长期波动，捕捉半年趋势", "en": "Smooths medium-term noise and captures half-year trend"},
    "month_cos": {"zh": "用三角函数编码月份，帮助模型识别春节/金九银十等季节性", "en": "Trigonometric month encoding for seasonal patterns"},
    "month_sin": {"zh": "与 month_cos 配合，完整表达一年中各月周期性", "en": "Pairs with month_cos to model annual seasonality"},
    "energy_type_enc": {"zh": "纯电/插混/燃油等能源类型属于车型固有属性", "en": "Fixed vehicle attribute: EV/PHEV/ICE etc."},
    "vehicle_class_enc": {"zh": "轿车/SUV/MPV 等级别属于车型固有属性", "en": "Fixed vehicle attribute: sedan/SUV/MPV etc."},
    "positive_ratio": {"zh": "该车系当月正面评论占比", "en": "Share of positive reviews this month"},
    "negative_ratio": {"zh": "该车系当月负面评论占比", "en": "Share of negative reviews this month"},
    "intelligence_lag2": {"zh": "两个月前的智能化口碑，检验滞后效应", "en": "Two-month lagged intelligence sentiment to test lag effect"},
}

ASPECT_ORDER = ["appearance", "interior", "space", "power", "control",
                "comfort", "fuel_consumption", "configuration", "intelligence", "value"]
ASPECT_ZH = {"appearance": "外观", "interior": "内饰", "space": "空间", "power": "动力",
             "control": "操控", "comfort": "舒适", "fuel_consumption": "油耗",
             "configuration": "配置", "intelligence": "智能化", "value": "性价比"}
ASPECT_EN = {"appearance": "Appearance", "interior": "Interior", "space": "Space",
             "power": "Power", "control": "Control", "comfort": "Comfort",
             "fuel_consumption": "Fuel", "configuration": "Config",
             "intelligence": "Intelligence", "value": "Value"}

# 品牌中英文映射（数据源为中文平台，英文名为常用国际名或官方英文名）
BRAND_EN = {
    "ARCFOX极狐": "ARCFOX", "DS": "DS", "MG名爵": "MG", "Polestar极星": "Polestar",
    "SERES赛力斯": "SERES", "SRM鑫源": "SRM", "SWM斯威汽车": "SWM", "iCAR": "iCAR",
    "smart": "smart", "一汽": "FAW", "东风奕派": "Dongfeng eπ", "东风富康": "Dongfeng Fukang",
    "东风纳米": "Dongfeng Nammi", "东风风光": "Dongfeng Fengguang", "东风风度": "Dongfeng Fengdu",
    "东风风神": "Dongfeng Fengshen", "东风风行": "Dongfeng Fengxing", "中国重汽VGV": "CNHTC VGV",
    "丰田": "Toyota", "五菱": "Wuling", "五菱汽车": "Wuling", "仰望": "Yangwang",
    "凌宝汽车": "Lingbao", "凯翼": "Kaiyi", "凯迪拉克": "Cadillac", "创维汽车": "Skyworth",
    "别克": "Buick", "北京汽车": "Beijing Auto", "北京越野": "Beijing Off-road",
    "北汽制造": "BAW", "合创汽车": "Hycan", "吉利几何": "Geely Geometry", "吉利汽车": "Geely",
    "吉利银河": "Geely Galaxy", "启辰": "Venucia", "哈弗": "Haval", "哪吒汽车": "Neta",
    "坦克": "Tank", "埃安": "Aion", "大众": "Volkswagen", "大运": "Dayun",
    "大通": "Maxus", "奇瑞": "Chery", "奇瑞新能源": "Chery New Energy", "奇瑞风云": "Chery Fulwin",
    "奔腾": "Bestune", "奔驰": "Mercedes-Benz", "奥迪": "Audi", "宝马": "BMW",
    "宝骏": "Baojun", "小米汽车": "Xiaomi Auto", "小虎": "Xiaohu", "小鹏": "XPeng",
    "岚图汽车": "Voyah", "广汽传祺": "GAC Trumpchi", "广汽集团": "GAC Group", "开瑞": "Karry",
    "思皓": "Sehol", "思铭": "Ciimo", "捷豹": "Jaguar", "捷达": "Jetta", "捷途": "Jetour",
    "斯柯达": "Škoda", "新龙马汽车": "Xinlongma", "方程豹": "Fangchengbao", "日产": "Nissan",
    "昊铂": "Hyper", "星途": "Exeed", "智己汽车": "IM Motors", "曹操汽车": "Caocao Auto",
    "本田": "Honda", "极氪": "Zeekr", "极石汽车": "Jishi", "林肯": "Lincoln", "标致": "Peugeot",
    "欧拉": "Ora", "比亚迪": "BYD", "江淮": "JAC", "江淮瑞风": "JAC Ruifeng",
    "江淮钇为": "JAC Yiwei", "江铃集团新能源": "JMC New Energy", "沃尔沃": "Volvo",
    "海马": "Haima", "海马新能源": "Haima New Energy", "海马郑州": "Haima Zhengzhou",
    "深蓝汽车": "Deepal", "特斯拉": "Tesla", "现代": "Hyundai", "理念": "Everus",
    "理想汽车": "Li Auto", "睿蓝汽车": "Livan", "福特": "Ford", "福田": "Foton",
    "红旗": "Hongqi", "腾势": "Denza", "英菲尼迪": "Infiniti", "荣威": "Roewe",
    "蓝电": "Landian", "蔚来": "NIO", "起亚": "Kia", "路虎": "Land Rover", "长安": "Changan",
    "长安凯程": "Changan Kaicheng", "长安启源": "Changan Qiyuan", "长安欧尚": "Changan Oshan",
    "雪佛兰": "Chevrolet", "雪铁龙": "Citroën", "零跑汽车": "Leapmotor", "雷丁": "Letin",
    "领克": "Lynk & Co", "飞凡汽车": "Rising Auto", "马自达": "Mazda", "魏牌": "Wey",
    "鸿蒙智行": "Harmony Intelligent Mobility"
}

def brand_en(name):
    """返回品牌的英文显示名；没有映射时返回原中文名。"""
    return BRAND_EN.get(str(name), str(name))


# 车系中英文映射：数据源来自懂车帝 / 太平洋汽车等中文平台，车系名多为中文。
# 这里手工整理主流车系的官方 / 国际英文名；对于“中文品牌前缀 + 英文车型码”的车系
# （如 奥迪A5L Sportback、捷豹XFL、标致408X），series_en() 会自动剥离中文品牌前缀，
# 直接采用其后的拉丁车型码，无需逐一登记。
SERIES_EN = {
    # 大众 Volkswagen
    "凌渡": "Lamando", "威然": "Viloran", "宝来": "Bora", "帕萨特": "Passat",
    "探岳": "Tayron", "揽境": "Talagon", "揽巡": "Tavendor", "朗逸": "Lavida",
    "迈腾": "Magotan", "途岳": "Tharu", "途昂": "Teramont", "速腾": "Sagitar",
    "高尔夫": "Golf",
    # 丰田 Toyota
    "亚洲龙": "Avalon", "凯美瑞": "Camry", "卡罗拉": "Corolla", "卡罗拉锐放": "Corolla Cross",
    "威兰达": "Wildlander", "威飒": "Venza", "格瑞维亚": "Granvia", "汉兰达": "Highlander",
    "皇冠陆放": "Crown Kluger", "锋兰达": "Frontlander", "雷凌": "Levin",
    # 本田 Honda
    "冠道": "Avancier", "型格": "Integra", "奥德赛": "Odyssey", "思域": "Civic",
    "皓影": "Breeze", "缤智": "HR-V", "英仕派": "Inspire", "雅阁": "Accord",
    # 福特 Ford
    "探险者": "Explorer", "福特烈马": "Bronco", "蒙迪欧": "Mondeo", "锐际": "Escape",
    "领睿": "Equator Sport", "领裕": "Equator",
    # 别克 Buick
    "世纪": "Century", "君威": "Regal", "君越": "LaCrosse", "威朗": "Verano", "昂科威": "Envision",
    # 斯柯达 Škoda
    "明锐": "Octavia", "柯珞克": "Karoq", "柯米克": "Kamiq", "柯迪亚克": "Kodiaq", "速派": "Superb",
    # 日产 Nissan
    "劲客": "Kicks", "天籁": "Teana", "奇骏": "X-Trail", "轩逸": "Sylphy", "逍客": "Qashqai",
    # 吉利 Geely
    "博越": "Boyue", "星瑞": "Preface", "缤瑞": "Binrui", "缤越": "Binyue",
    # 起亚 Kia
    "嘉华": "Carnival", "奕跑": "Stonic", "智跑": "Sportage", "狮铂拓界": "Sportage",
    # 雪佛兰 Chevrolet
    "创酷": "Trax", "开拓者": "Blazer", "探界者": "Equinox", "星迈罗": "Seeker",
    # 东风风神 Dongfeng Fengshen
    "奕炫": "Yixuan", "皓极": "Haoji", "皓瀚": "Haohan",
    # 星途 Exeed
    "星途凌云": "Lingyun", "星途揽月": "Lanyue", "星途瑶光": "Yaoguang",
    # 林肯 Lincoln
    "冒险家": "Corsair", "航海家": "Nautilus", "飞行家": "Aviator",
    # 欧拉 Ora
    "欧拉好猫": "Good Cat", "欧拉芭蕾猫": "Ballet Cat", "欧拉闪电猫": "Lightning Cat",
    # 比亚迪 BYD
    "海豚": "Dolphin", "海豹": "Seal", "海鸥": "Seagull",
    # 现代 Hyundai
    "伊兰特": "Elantra", "库斯途": "Custo", "胜达": "Santa Fe",
    # 广汽传祺 GAC Trumpchi
    "影豹": "Empow", "影酷": "Emkoo",
    # 捷途 Jetour
    "捷途大圣": "Grand Sage", "捷途旅行者": "Traveller",
    # 长安 Changan
    "逸动": "Eado", "逸达": "Yida",
    # 五菱 Wuling
    "五菱之光": "Sunshine",
    # 凯翼 Kaiyi
    "轩度": "Xuandu",
    # 哈弗 Haval
    "哈弗大狗": "Big Dog",
    # 奇瑞 Chery
    "欧萌达": "Omoda",
    # 奔腾 Bestune
    "奔腾小马": "Pony",
    # 思皓 Sehol
    "思皓花仙子": "E10X",
    # 路虎 Land Rover
    "揽胜极光": "Range Rover Evoque",
    # 预警列表中“品牌前缀 + 数字/英文”混合车系，需显式登记
    "风光580": "Fengguang 580",
    "探索06 C-DM": "Tansuo 06 C-DM",
    "吉利几何A": "Geometry A",
    # 奔驰 Mercedes-Benz（级别后缀）
    "奔驰A级": "Mercedes-Benz A-Class", "奔驰C级": "Mercedes-Benz C-Class",
    "奔驰E级": "Mercedes-Benz E-Class", "奔驰V级": "Mercedes-Benz V-Class",
    # 宝马 BMW（级别后缀）
    "宝马3系": "BMW 3 Series", "宝马5系": "BMW 5 Series",
    # 启辰 Venucia
    "启辰大V": "Venucia DaV", "启辰大V DD-i": "Venucia DaV DD-i",
    # 凯迪拉克 IQ 纯电序列
    "IQ傲歌": "IQ Optiq", "IQ锐歌": "IQ Lyriq",
    # 别克 微蓝 / 北京汽车
    "微蓝6": "Velite 6", "北京X7": "Beijing X7",
    # ARCFOX 阿尔法序列 / smart
    "极狐 阿尔法S5": "ARCFOX Alpha S5", "极狐 阿尔法T5": "ARCFOX Alpha T5",
    "smart精灵#1": "smart #1", "smart精灵#3": "smart #3",
    # 东风风行 星海 / 江淮 / 睿蓝
    "星海V9": "Xinghai V9", "钇为3": "Yiwei 3", "瑞风M3": "Refine M3", "睿蓝7": "Livan 7",
    # 长安 锐程 / 方程豹 豹
    "锐程PLUS": "Ruicheng PLUS", "豹5": "Bao 5",
    # 比亚迪 海狮 / 驱逐舰 / 元 / 宋
    "海狮07EV": "Seal U 07 EV", "驱逐舰05": "Destroyer 05",
    "元PLUS": "Yuan PLUS", "元UP": "Yuan UP", "宋L EV": "Song L EV",
    # 马自达
    "马自达3 昂克赛拉": "Mazda 3 Axela", "马自达CX-50行也": "Mazda CX-50",
}

# 中文基名 → 英文基名（带尾随空格），用于“中文基名 + 拉丁后缀”变体，
# 如 博越L → Boyue L、瑞虎3x → Tiggo 3x、银河E8 → Galaxy E8。
SERIES_EN_BASE = {
    "博越": "Boyue ", "探岳": "Tayron ", "迈腾": "Magotan ", "途观": "Tiguan ",
    "途昂": "Teramont ", "速腾": "Sagitar ", "朗逸": "Lavida ", "宝来": "Bora ",
    "威然": "Viloran ", "帕萨特": "Passat ", "凌渡": "Lamando ", "揽巡": "Tavendor ",
    "揽境": "Talagon ", "途岳": "Tharu ", "探歌": "T-Roc ",
    "瑞虎": "Tiggo ", "艾瑞泽": "Arrizo ", "风云": "Fengyun ",
    "智己": "IM ", "理想": "Li Auto ", "深蓝": "Deepal ", "零跑": "Leapmotor ",
    "飞凡": "Rising ", "银河": "Galaxy ", "豪越": "Haoyue ", "星越": "Xingyue ",
    "瑶光": "Yaoguang ", "追风": "Zhuifeng ",
    # 品牌钻取 Top 车系排名里出现的补充项
    "帕拉丁": "Paladin", "菱智": "Lingzhi", "炫界": "Xuanjie", "威霆": "Vito",
    "艾力绅": "Elysion", "索纳塔": "Sonata", "焕驰": "Huanchi", "赛图斯": "Seltos",
}


def series_en(name, brand=None):
    """返回车系的英文显示名。
    - 命中手工映射表（含变体 / 混合车系的显式登记）→ 直接用；
    - 否则剥离中文品牌前缀后，用“中文基名→英文基名”前缀匹配处理“博越L / 瑞虎3x”类带后缀变体；
    - 仍含拉丁字母 / 数字的词元（如 RAV4荣放、传祺GS3、smart精灵#1）→ 抽取拉丁词元（RAV4 / GS3 / smart #1）；
    - 全为中文且无任何映射 → 退回原中文名（不臆造英文名）。
    """
    name = str(name)
    if name in SERIES_EN:
        return SERIES_EN[name]
    rem = name
    if brand:
        b = str(brand)
        if name.startswith(b):
            rem = name[len(b):].strip()
    # 中文基名 + 拉丁后缀
    for base, en in SERIES_EN_BASE.items():
        if rem.startswith(base):
            return en + rem[len(base):]
    # 拉丁 + 中文混排：抽取拉丁 / 数字词元
    if re.search(r"[A-Za-z0-9]", rem):
        toks = re.findall(r"[A-Za-z0-9#.\-+ ]+", rem)
        joined = " ".join(t.strip() for t in toks if t.strip())
        if joined:
            return joined
    return name
PALETTE = ["#2c7be5", "#00a9ae", "#34c38f", "#f6c343", "#ee5b5b",
           "#a55eea", "#7783f5", "#fd7e56", "#26c6da", "#95aac9"]


def j(val):
    """把 numpy 类型转成 JSON 原生类型；NaN/Inf 转成 null。"""
    try:
        if hasattr(val, "item"):
            val = val.item()
    except Exception:
        pass
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return None
    return val


def load(name):
    p = os.path.join(PROC, name)
    if not os.path.exists(p):
        print(f"  [skip] 缺少 {p}")
        return None
    return pd.read_csv(p)


def load_new(name, **kwargs):
    p = os.path.join(PROC_NEW, name)
    if not os.path.exists(p):
        print(f"  [skip] 缺少 {p}")
        return None
    return pd.read_csv(p, **kwargs)


def load_sentiment(name, **kwargs):
    p = os.path.join(SENTIMENT_NEW, name)
    if not os.path.exists(p):
        print(f"  [skip] 缺少 {p}")
        return None
    return pd.read_csv(p, **kwargs)


def read_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def dump(obj, fname, compact=False):
    def clean_nan(x):
        if isinstance(x, float):
            return None if math.isnan(x) or math.isinf(x) else x
        if isinstance(x, list):
            return [clean_nan(i) for i in x]
        if isinstance(x, dict):
            return {k: clean_nan(v) for k, v in x.items()}
        return x
    obj = clean_nan(obj)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, fname)
    with open(path, "w", encoding="utf-8") as f:
        if compact:
            json.dump(obj, f, ensure_ascii=False, separators=(',', ':'))
        else:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"  [ok]   {fname} ({round(os.path.getsize(path)/1024/1024, 2)} MB)")


def build_overview():
    sent = load("stage4/sentiment_sales_monthly.csv")
    if sent is None:
        sent = load("stage4/sentiment_monthly_by_series.csv")
    brand = load("stage4/sentiment_sales_monthly_brand.csv")
    alerts = load("stage5/sentiment_alerts.csv")
    mc = load("stage3/model_comparison.csv")

    coverage = int(sent["series_id"].nunique()) if sent is not None else 0
    brands = int(brand["brand"].nunique()) if brand is not None else 0
    alert_n = int(len(alerts)) if alerts is not None else 0
    best_wmape = round(float(mc.sort_values("WMAPE_vol").iloc[0]["WMAPE_vol"]), 2) if mc is not None else 0
    review_n = int(sent["review_count"].sum()) if sent is not None and "review_count" in sent.columns else 0

    # 月度总销量走势（概览迷你图）
    monthly_trend = []
    if brand is not None:
        mt = brand.groupby("period")["monthly_sales"].sum().sort_index()
        monthly_trend = [{"month": str(k), "sales": round(float(v), 0)} for k, v in mt.items()]

    return {
        "kpis": {
            "coverage_series": coverage,
            "forecast_wmape": best_wmape,
            "forecast_horizon": 24,
            "alert_count": alert_n,
            "brand_count": brands,
            "eval_series": int(mc["n_series"].iloc[0]) if mc is not None else 0,
            "review_count": review_n,
        },
        "monthly_trend": monthly_trend,
        "stages": [
            {"id": 1, "name": "数据准备", "en": "Data Preparation", "status": "done"},
            {"id": 2, "name": "数据筛选与 EDA", "en": "Screening & EDA", "status": "done"},
            {"id": 3, "name": "销量预测建模", "en": "Sales Forecasting", "status": "done"},
            {"id": 4, "name": "舆情 ABSA 与归因", "en": "ABSA & Attribution", "status": "done"},
            {"id": 5, "name": "情感融合·主题·预警", "en": "Fusion · Topics · Alerts", "status": "done"},
            {"id": 6, "name": "看板交付", "en": "Dashboard Delivery", "status": "current"},
        ],
        "findings": [
            {"zh": "销量预测最优基线：XGBoost 体积加权 WMAPE 为 29.26%",
             "en": "Best sales forecast baseline: XGBoost volume-weighted WMAPE 29.26%"},
            {"zh": "舆情与销量存在弱相关，但品牌级 Granger 显著率仅 4%~15%",
             "en": "Weak correlation between sentiment and sales; brand-level Granger significance 4-15%"},
            {"zh": "加入舆情特征后销量归因 R² 提升 +0.211",
             "en": "Adding sentiment features improves sales attribution R² by +0.211"},
            {"zh": "情感融合预测未降低 WMAPE：舆情是同期/滞后指标，非先行指标",
             "en": "Sentiment fusion does not lower WMAPE: sentiment is coincident/lagging, not leading"},
            {"zh": "共触发 17 条口碑恶化预警，可作为声誉风险监测入口",
             "en": "17 sentiment deterioration alerts triggered as reputation risk monitoring entry"},
        ],
    }


def build_forecast():
    mc = load("stage3/model_comparison.csv")
    preds = load("stage3/xgboost_preds.csv")
    feat = load("stage5/feature_importance.csv")
    out = {"models": [], "class_wmape": [], "scatter": [], "features": [], "conclusion": {"zh": "", "en": ""}}
    if mc is not None:
        mc = mc.sort_values("WMAPE_vol")
        for i, (_, r) in enumerate(mc.iterrows()):
            out["models"].append({
                "name": str(r["model"]),
                "name_zh": str(r["model"]),
                "name_en": str(r["model"]),
                "wmape_vol": round(float(r["WMAPE_vol"]), 2),
                "wmape_med": round(float(r["WMAPE_med"]), 2),
                "mae": round(float(r["MAE"]), 1),
                "color": PALETTE[i % len(PALETTE)],
            })
        out["best_model"] = out["models"][0]["name"]

    if feat is not None and "feature" in feat.columns and "importance" in feat.columns:
        feat = feat.sort_values("importance", ascending=False).head(12)
        for i, (_, r) in enumerate(feat.iterrows()):
            raw = str(r["feature"])
            out["features"].append({
                "name": raw,
                "name_zh": FEATURE_NAME.get(raw, {}).get("zh", raw),
                "name_en": FEATURE_NAME.get(raw, {}).get("en", raw),
                "desc_zh": FEATURE_DESC.get(raw, {}).get("zh", ""),
                "desc_en": FEATURE_DESC.get(raw, {}).get("en", ""),
                "importance": round(float(r["importance"]), 4),
                "color": PALETTE[i % len(PALETTE)],
            })

    if preds is not None and len(preds) > 0:
        # 散点图：所有实际 vs 预测点
        valid = preds[(preds["actual"] > 0) & (preds["pred"].notna())].copy()
        if len(valid) > 0:
            out["scatter"] = [
                [round(float(a), 1), round(float(p), 1)]
                for a, p in zip(valid["actual"], valid["pred"])
            ]

        # 按车型级别的 WMAPE
        sales = load("sales_filtered_24m.csv")
        if sales is not None and "series_name" in sales.columns and "category_en" in sales.columns:
            cat_map = sales.drop_duplicates("series_name").set_index("series_name")["category_en"].to_dict()
            valid = valid.copy()
            valid["cat"] = valid["series_name"].map(cat_map)
            for cat, g in valid.groupby("cat"):
                if pd.isna(cat) or len(g) == 0:
                    continue
                wmape = float((g["actual"] - g["pred"]).abs().sum() / g["actual"].sum() * 100)
                out["class_wmape"].append({
                    "category": str(cat),
                    "wmape": round(wmape, 1),
                    "n_series": int(g["series_name"].nunique()),
                })
            out["class_wmape"].sort(key=lambda x: x["wmape"])

    out["conclusion"]["zh"] = ("在 150 系评估集上，XGBoost（销量滞后+车型配置+舆情特征）体积加权 WMAPE 最低。"
                               "特征重要性显示：销量自身滞后（lag_1、roll_mean_3）占绝对主导，说明销量主要由历史惯性驱动；"
                               "舆情特征重要性普遍靠后。这引出一个关键问题：舆情是否在其他环节（如归因）才有价值？")
    out["conclusion"]["en"] = ("On the 150-series evaluation set, XGBoost (sales lags + vehicle config + sentiment features) achieves the lowest volume-weighted WMAPE. "
                               "Feature importance shows sales lag features dominate, confirming sales are mainly driven by historical momentum; "
                               "sentiment features rank low. This raises a key question: does sentiment matter more in attribution than in forecasting?")
    # 给前端的特征解读文字
    out["feature_insight"] = {
        "zh": "前两位都是销量自身滞后/移动平均，说明预测主要依赖「惯性」；"
              "能源类型、车辆级别等配置属性提供车型截面差异；"
              "舆情特征（正/负面占比、智能化滞后）排在末尾，提示动态口碑不是强预测信号。",
        "en": "The top two features are sales lags/moving averages, confirming the forecast relies mostly on momentum. "
              "Energy type and vehicle class provide cross-sectional differences. "
              "Sentiment features (positive/negative ratios, intelligence lag) rank low, suggesting dynamic word-of-mouth is not a strong predictor."
    }
    return out


def build_absa():
    sent = load("stage4/sentiment_sales_monthly.csv")
    if sent is None:
        sent = load("stage4/sentiment_monthly_by_series.csv")
    out = {"aspects": [], "radar": {"indicators": [], "values": []},
           "distribution": [], "variance": [], "monthly_trends": {"months": [], "series": []},
           "conclusion": {"zh": "", "en": ""}}
    if sent is not None:
        for i, k in enumerate(ASPECT_ORDER):
            avg = float(sent[k].mean())
            pos = float((sent[k] > 0.1).mean())
            neg = float((sent[k] < -0.1).mean())
            neu = 1 - pos - neg
            out["aspects"].append({
                "key": k, "name_zh": ASPECT_ZH[k], "name_en": ASPECT_EN[k],
                "avg": round(avg, 3), "color": PALETTE[i % len(PALETTE)],
            })
            out["radar"]["indicators"].append({
                "name": ASPECT_ZH[k], "name_en": ASPECT_EN[k], "max": 1, "min": -1
            })
            out["radar"]["values"].append(round(avg, 3))
            out["distribution"].append({
                "key": k, "name_zh": ASPECT_ZH[k], "name_en": ASPECT_EN[k],
                "positive": round(pos, 3), "neutral": round(neu, 3), "negative": round(neg, 3),
            })
            out["variance"].append({
                "key": k, "name_zh": ASPECT_ZH[k], "name_en": ASPECT_EN[k],
                "std": round(float(sent[k].std()), 3),
            })
        # 月度维度走势：按 period 聚合各维度均值
        if "period" in sent.columns:
            mt = sent.groupby("period")[ASPECT_ORDER].mean().sort_index()
            out["monthly_trends"]["months"] = mt.index.astype(str).tolist()
            for k in ASPECT_ORDER:
                out["monthly_trends"]["series"].append({
                    "key": k, "name_zh": ASPECT_ZH[k], "name_en": ASPECT_EN[k],
                    "data": [round(float(x), 3) if pd.notna(x) else None for x in mt[k]],
                })
    out["conclusion"]["zh"] = ("10 维度平均情感分(−1~1)：外观/内饰偏正面，智能化/性价比偏负面；"
                               "消费者对「看得见的静态产品力」满意，对「智能与价值感」最挑剔。")
    out["conclusion"]["en"] = ("10-dimension average sentiment (−1~1): Appearance/Interior are positive, while Intelligence/Value are negative; "
                               "consumers favor visible static qualities but are most critical of intelligence and value perception.")
    return out


def build_attribution():
    shap = load("stage4/aspect_shap_ranking.csv")
    metr = load("stage4/attribution_metrics.csv")
    preds = load("stage3/xgboost_preds.csv")
    out = {"shap": [], "comparison": {"with": None, "without": None},
           "top_example": None, "conclusion": {"zh": "", "en": ""}}
    if shap is not None:
        # 第一列是无名列(维度名)，重命名为 aspect
        if shap.columns[0].startswith("Unnamed"):
            shap = shap.rename(columns={shap.columns[0]: "aspect"})
        key_col = "aspect" if "aspect" in shap.columns else shap.columns[0]
        for _, r in shap.iterrows():
            k = str(r[key_col])
            if k not in ASPECT_ZH:
                continue
            out["shap"].append({
                "key": k, "name_zh": ASPECT_ZH[k], "name_en": ASPECT_EN[k],
                "importance": round(float(r["mean_abs_shap"]), 4),
            })
        out["shap"].sort(key=lambda x: x["importance"], reverse=True)
        for i, s in enumerate(out["shap"]):
            s["color"] = PALETTE[i % len(PALETTE)]

    if metr is not None:
        for _, r in metr.iterrows():
            m = str(r["model"]).lower()
            if "without" in m:
                tag = "without"
            elif "with" in m:
                tag = "with"
            else:
                tag = "without"
            out["comparison"][tag] = {
                "r2": round(float(r["R2"]), 3),
                "mape": round(float(r["MAPE"]), 3),
            }

    if preds is not None and len(preds) > 0 and "actual" in preds.columns and "pred" in preds.columns:
        # 找预测误差最小的 top 车系作为示例
        sub = preds.copy()
        sub["err"] = (sub["pred"] - sub["actual"]).abs()
        best = sub.groupby("series_name")["err"].mean().sort_values().head(5)
        if len(best) > 0:
            name = best.index[0]
            ssub = sub[sub["series_name"] == name].sort_values("date")
            out["top_example"] = {
                "series": name,
                "dates": ssub["date"].astype(str).tolist(),
                "actual": [round(float(x), 1) for x in ssub["actual"]],
                "pred": [round(float(x), 1) for x in ssub["pred"]],
                "error": [round(float(x), 1) for x in ssub["err"]],
            }

    out["conclusion"]["zh"] = ("加入舆情特征后销量归因 R² 由 −0.073 提升至 0.138（+0.211）；"
                               "SHAP 显示「舒适 > 性价比 > 智能化」是舆情影响销量的关键维度。")
    out["conclusion"]["en"] = ("Adding sentiment features lifts sales attribution R² from −0.073 to 0.138 (+0.211); "
                               "SHAP shows Comfort > Value > Intelligence are the key sentiment drivers of sales.")
    return out


def build_relation():
    gr = load("stage4/granger_brand_summary.csv")
    fc = load("stage5/forecast_comparison.csv")
    bsent = load("stage4/sentiment_sales_monthly_brand.csv")
    sent = load("stage4/sentiment_sales_monthly.csv")
    out = {"granger": {"aspects": [], "sig_rates": []},
           "fusion": [], "timeseries": None,
           "correlation": [], "conclusion": {"zh": "", "en": ""}}
    if gr is not None:
        gr = gr.sort_values("sig_rate", ascending=False)
        for _, r in gr.iterrows():
            k = str(r["aspect"])
            out["granger"]["aspects"].append({"zh": ASPECT_ZH.get(k, k), "en": ASPECT_EN.get(k, k)})
            out["granger"]["sig_rates"].append(round(float(r["sig_rate"]), 3))
    if fc is not None:
        xgb = fc[fc["model"] == "XGBoost"]
        for _, r in xgb.iterrows():
            out["fusion"].append({
                "version": str(r["version"]),
                "version_en": str(r["version"]),
                "wmape_vol": round(float(r["WMAPE_vol"]), 2),
            })
    if bsent is not None:
        aspect_cols = ASPECT_ORDER
        bsent = bsent.copy()
        bsent["overall"] = bsent[aspect_cols].mean(axis=1)
        # 市场级时序：所有品牌按月汇总
        market = bsent.groupby("period").agg(
            sales=("monthly_sales", "sum"),
            sentiment=("overall", "mean"),
        ).sort_index()
        out["timeseries"] = {
            "brand": "全市场",
            "brand_en": "Market",
            "months": market.index.astype(str).tolist(),
            "sales": [round(float(x), 1) for x in market["sales"]],
            "sentiment": [round(float(x), 3) if pd.notna(x) else None for x in market["sentiment"]],
        }

    if sent is not None:
        # 计算各维度与销量的月度相关系数
        tmp = sent.copy()
        tmp = tmp[tmp["monthly_sales"] > 0]
        if len(tmp) > 10:
            for k in ASPECT_ORDER:
                corr = tmp[k].corr(tmp["monthly_sales"])
                if pd.isna(corr):
                    corr = 0
                out["correlation"].append({
                    "key": k, "name_zh": ASPECT_ZH[k], "name_en": ASPECT_EN[k],
                    "corr": round(float(corr), 3),
                })

    out["conclusion"]["zh"] = ("品牌级 Granger 仅 4%~15% 车系显著；把月度情感塞回销量模型后体积加权误差反而略升——"
                               "舆情是销量的「同期/滞后指标」而非「先行指标」，故不能作预测特征。")
    out["conclusion"]["en"] = ("Brand-level Granger is significant for only 4-15% of series; adding monthly sentiment to the sales model slightly raises volume-weighted error — "
                               "sentiment is a coincident/lagging indicator, not a leading one, so it cannot be used as a predictive feature.")
    return out


def build_alerts():
    al = load("stage5/sentiment_alerts.csv")
    out = {"rule": {"zh": "overall < -0.1 且 环比(对上月)下降 > 0.05", "en": "overall < -0.1 & month-over-month drop > 0.05"},
           "alerts": [], "monthly": [], "risk_dist": [], "conclusion": {"zh": "", "en": ""}}
    if al is not None:
        for _, r in al.iterrows():
            b = str(r["brand"])
            out["alerts"].append({
                "series_name": str(r["series_name"]),
                "brand": b,
                "brand_en": brand_en(b),
                "series_en": series_en(str(r["series_name"]), b),
                "period": str(r["period"]),
                "overall": round(float(r["overall"]), 3),
                "overall_drop": round(float(r["overall_drop"]), 3),
            })
        out["alerts"].sort(key=lambda x: x["overall_drop"])
        # 月度趋势
        al2 = al.copy()
        al2["ym"] = al2["period"].astype(str)
        trend = al2.groupby("ym").size().sort_index()
        for k, v in trend.items():
            out["monthly"].append({"month": str(k), "count": int(v)})
        # 风险等级
        al2["risk"] = al2["overall"].apply(lambda x: "高危" if x < -0.4 else ("中危" if x < -0.2 else "低危"))
        risk = al2.groupby("risk").size()
        for lvl, cnt in risk.items():
            out["risk_dist"].append({"level": str(lvl), "count": int(cnt)})
    out["conclusion"]["zh"] = (f"按上述规则共触发 {len(out['alerts'])} 条舆情预警；"
                               "这些车系口碑出现明显负向恶化，建议作为声誉风险重点监测对象。")
    out["conclusion"]["en"] = (f"{len(out['alerts'])} sentiment alerts triggered under the above rule; "
                               "these series show marked negative deterioration and should be prioritized for reputation risk monitoring.")
    return out


def build_drilldown():
    # 用已对齐的口碑+销量表（按车系名 join，规避两平台 series_id 不同源）
    aligned = load("stage4/sentiment_sales_monthly.csv")
    out = {"series": [], "brands": [], "data": {}, "brand_en": {}}
    if aligned is None:
        return out
    aspect_cols = ASPECT_ORDER
    aligned = aligned.copy()
    aligned["series_id"] = aligned["series_id"].astype(str)
    aligned["overall"] = aligned[aspect_cols].mean(axis=1)
    aligned["ym"] = aligned["year"].astype(str) + "-" + aligned["month"].astype(str).str.zfill(2)

    # 过滤：保留销量与口碑均完整的系列（至少 3 个月非 NaN 综合口碑）
    MIN_NON_NAN_SENTIMENT_MONTHS = 3
    valid_series = []
    for sid, g in aligned.groupby("series_id"):
        non_nan = g["overall"].notna().sum()
        if non_nan >= MIN_NON_NAN_SENTIMENT_MONTHS:
            valid_series.append(sid)
    aligned = aligned[aligned["series_id"].isin(valid_series)].copy()

    meta = aligned.drop_duplicates("series_id")[["series_id", "series_name", "brand"]]
    meta = meta.sort_values(["brand", "series_name"])
    for _, r in meta.iterrows():
        sid = str(r["series_id"])
        b = str(r["brand"])
        out["series"].append({
            "id": sid, "name": str(r["series_name"]), "brand": b,
            "brand_en": brand_en(b),
            "series_en": series_en(str(r["series_name"]), b)
        })
        out["brand_en"][b] = brand_en(b)
    out["brands"] = sorted(meta["brand"].unique().tolist())
    for sid, g in aligned.groupby("series_id"):
        sid = str(sid)
        g = g.sort_values("ym")
        d = {
            "months": g["ym"].tolist(),
            "sales": [round(float(x), 1) for x in g["monthly_sales"]],
            "sentiment": [round(float(x), 3) if pd.notna(x) else None for x in g["overall"]],
            "aspects": {},
        }
        for k in aspect_cols:
            d["aspects"][k] = [round(float(x), 3) if pd.notna(x) else None for x in g[k]]
        out["data"][sid] = d
    return out


def build_brand_drilldown():
    """品牌钻取数据桥：品牌级时序 + 维度雷达 + 旗下车系排名。

    与车型钻取互补：车型级舆情稀疏（大量空白格），品牌级聚合后数据基本连续，
    完成度更高，适合从宏观看「品牌口碑 vs 销量」的整体走势与结构。
    """
    brand_monthly = load("stage4/sentiment_sales_monthly_brand.csv")
    model_monthly = load("stage4/sentiment_sales_monthly.csv")
    out = {"brands": [], "brand_en": {}, "market_radar": {}, "data": {}}
    if brand_monthly is None:
        return out
    aspect_cols = ASPECT_ORDER
    brand_monthly = brand_monthly.copy()
    brand_monthly["brand"] = brand_monthly["brand"].astype(str).str.strip()
    brand_monthly["overall"] = brand_monthly[aspect_cols].mean(axis=1)

    # 市场平均雷达（各品牌维度均值的平均，代表典型品牌画像）
    market_radar = {k: round(float(brand_monthly[k].mean(skipna=True)), 3) for k in aspect_cols}
    out["market_radar"] = market_radar

    # 品牌级时序 + 雷达
    for b, g in brand_monthly.groupby("brand"):
        g = g.sort_values("period")
        radar = {k: round(float(g[k].mean(skipna=True)), 3) for k in aspect_cols}
        out["data"][b] = {
            "months": g["period"].astype(str).tolist(),
            "sales": [round(float(x), 1) if pd.notna(x) else None for x in g["monthly_sales"]],
            "sentiment": [round(float(x), 3) if pd.notna(x) else None for x in g["overall"]],
            "radar": radar,
            "series_rank": [],
        }
        out["brand_en"][b] = brand_en(b)
    out["brands"] = sorted(brand_monthly["brand"].unique().tolist())

    # 品牌下各车系排名（车型级 CSV 聚合）
    if model_monthly is not None:
        mm = model_monthly.copy()
        mm["brand"] = mm["brand"].astype(str).str.strip()
        mm["series_id"] = mm["series_id"].astype(str)
        mm["overall"] = mm[aspect_cols].mean(axis=1)

        # 先算每个品牌整体平均综合口碑（用于"与品牌均值比较"标注）
        for b in out["brands"]:
            sub = mm[mm["brand"] == b]
            if sub.empty:
                continue
            # 按车系聚合
            grp = sub.groupby(["series_id", "series_name"]).agg(
                total_sales=("monthly_sales", "sum"),
                n_months=("monthly_sales", lambda s: int((s > 0).sum())),
                avg_sent=("overall", "mean"),
            ).reset_index()
            grp = grp[grp["avg_sent"].notna() & (grp["n_months"] > 0)]
            grp["avg_monthly"] = grp.apply(
                lambda r: round(r["total_sales"] / r["n_months"], 1) if r["n_months"] else 0, axis=1)
            grp = grp.sort_values("avg_monthly", ascending=False).head(15)
            out["data"][b]["series_rank"] = [
                {
                    "id": str(r["series_id"]),
                    "name": str(r["series_name"]),
                    "series_en": series_en(str(r["series_name"]), b),
                    "avg_monthly": float(r["avg_monthly"]),
                    "avg_sent": round(float(r["avg_sent"]), 3),
                }
                for _, r in grp.iterrows()
            ]
    return out


def main():
    print("== 中国汽车市场分析看板数据桥 ==")
    os.makedirs(OUT, exist_ok=True)
    # Keep the existing static app and JSON contracts, but source all payloads
    # from the audited 371/736-series pipeline.  Legacy builder functions above
    # remain temporarily readable as a migration reference and are not invoked.
    from dashboard_data_v2 import DashboardV2
    payloads = DashboardV2(ROOT, brand_en, series_en).payloads()
    for filename, payload in payloads.items():
        dump(payload, filename, compact=filename in {"drilldown.json", "brand_drilldown.json"})
    print("== 完成 ==")


if __name__ == "__main__":
    main()
