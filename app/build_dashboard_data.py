#!/usr/bin/env python3
"""Build the static JSON files consumed by the dashboard."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from dashboard_data import DashboardData


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "app" / "static" / "data"

BRAND_EN = {
    "ARCFOX极狐": "ARCFOX", "DS": "DS", "MG名爵": "MG", "Polestar极星": "Polestar",
    "SERES赛力斯": "SERES", "SRM鑫源": "SRM", "SWM斯威汽车": "SWM", "iCAR": "iCAR",
    "smart": "smart", "一汽": "FAW", "东风奕派": "Dongfeng eπ", "东风富康": "Dongfeng Fukang",
    "东风纳米": "Dongfeng Nammi", "东风风光": "Dongfeng Fengguang", "东风风神": "Dongfeng Fengshen",
    "东风风行": "Dongfeng Fengxing", "丰田": "Toyota", "五菱": "Wuling", "五菱汽车": "Wuling",
    "仰望": "Yangwang", "凯翼": "Kaiyi", "凯迪拉克": "Cadillac", "创维汽车": "Skyworth",
    "别克": "Buick", "北京汽车": "Beijing Auto", "北京越野": "Beijing Off-road", "北汽制造": "BAW",
    "合创汽车": "Hycan", "吉利几何": "Geely Geometry", "吉利汽车": "Geely", "吉利银河": "Geely Galaxy",
    "启辰": "Venucia", "哈弗": "Haval", "哪吒汽车": "Neta", "坦克": "Tank", "埃安": "Aion",
    "大众": "Volkswagen", "大通": "Maxus", "奇瑞": "Chery", "奇瑞新能源": "Chery New Energy",
    "奇瑞风云": "Chery Fulwin", "奔腾": "Bestune", "奔驰": "Mercedes-Benz", "奥迪": "Audi",
    "宝马": "BMW", "宝骏": "Baojun", "小米汽车": "Xiaomi Auto", "小鹏": "XPeng",
    "岚图汽车": "Voyah", "广汽传祺": "GAC Trumpchi", "广汽集团": "GAC Group", "开瑞": "Karry",
    "捷豹": "Jaguar", "捷达": "Jetta", "捷途": "Jetour", "斯柯达": "Škoda",
    "方程豹": "Fangchengbao", "日产": "Nissan", "昊铂": "Hyper", "星途": "Exeed",
    "智己汽车": "IM Motors", "本田": "Honda", "极氪": "Zeekr", "林肯": "Lincoln",
    "标致": "Peugeot", "欧拉": "Ora", "比亚迪": "BYD", "江淮": "JAC", "江淮瑞风": "JAC Ruifeng",
    "江淮钇为": "JAC Yiwei", "沃尔沃": "Volvo", "海马": "Haima", "深蓝汽车": "Deepal",
    "特斯拉": "Tesla", "现代": "Hyundai", "理想汽车": "Li Auto", "睿蓝汽车": "Livan",
    "福特": "Ford", "福田": "Foton", "红旗": "Hongqi", "腾势": "Denza",
    "英菲尼迪": "Infiniti", "荣威": "Roewe", "蔚来": "NIO", "起亚": "Kia", "路虎": "Land Rover",
    "长安": "Changan", "长安凯程": "Changan Kaicheng", "长安启源": "Changan Qiyuan",
    "长安欧尚": "Changan Oshan", "雪佛兰": "Chevrolet", "雪铁龙": "Citroën",
    "零跑汽车": "Leapmotor", "领克": "Lynk & Co", "飞凡汽车": "Rising Auto",
    "马自达": "Mazda", "魏牌": "Wey", "鸿蒙智行": "Harmony Intelligent Mobility",
}

SERIES_EN = {
    "凌渡": "Lamando", "威然": "Viloran", "宝来": "Bora", "帕萨特": "Passat",
    "探岳": "Tayron", "揽境": "Talagon", "揽巡": "Tavendor", "朗逸": "Lavida",
    "迈腾": "Magotan", "途岳": "Tharu", "途昂": "Teramont", "速腾": "Sagitar",
    "高尔夫": "Golf", "亚洲龙": "Avalon", "凯美瑞": "Camry", "卡罗拉": "Corolla",
    "卡罗拉锐放": "Corolla Cross", "威兰达": "Wildlander", "威飒": "Venza",
    "格瑞维亚": "Granvia", "汉兰达": "Highlander", "皇冠陆放": "Crown Kluger",
    "锋兰达": "Frontlander", "雷凌": "Levin", "冠道": "Avancier", "型格": "Integra",
    "奥德赛": "Odyssey", "思域": "Civic", "皓影": "Breeze", "缤智": "HR-V",
    "英仕派": "Inspire", "雅阁": "Accord", "探险者": "Explorer", "福特烈马": "Bronco",
    "蒙迪欧": "Mondeo", "锐际": "Escape", "领睿": "Equator Sport", "领裕": "Equator",
    "世纪": "Century", "君威": "Regal", "君越": "LaCrosse", "威朗": "Verano",
    "昂科威": "Envision", "明锐": "Octavia", "柯珞克": "Karoq", "柯米克": "Kamiq",
    "柯迪亚克": "Kodiaq", "速派": "Superb", "劲客": "Kicks", "天籁": "Teana",
    "奇骏": "X-Trail", "轩逸": "Sylphy", "逍客": "Qashqai", "博越": "Boyue",
    "星瑞": "Preface", "缤瑞": "Binrui", "缤越": "Binyue", "嘉华": "Carnival",
    "智跑": "Sportage", "狮铂拓界": "Sportage", "探界者": "Equinox", "星迈罗": "Seeker",
    "冒险家": "Corsair", "航海家": "Nautilus", "飞行家": "Aviator",
    "欧拉好猫": "Good Cat", "欧拉芭蕾猫": "Ballet Cat", "欧拉闪电猫": "Lightning Cat",
    "海豚": "Dolphin", "海豹": "Seal", "海鸥": "Seagull", "伊兰特": "Elantra",
    "库斯途": "Custo", "胜达": "Santa Fe", "影豹": "Empow", "影酷": "Emkoo",
    "捷途旅行者": "Traveller", "逸动": "Eado", "逸达": "Yida", "哈弗大狗": "Big Dog",
    "揽胜极光": "Range Rover Evoque", "风光580": "Fengguang 580",
    "探索06 C-DM": "Tansuo 06 C-DM", "吉利几何A": "Geometry A",
    "奔驰A级": "Mercedes-Benz A-Class", "奔驰C级": "Mercedes-Benz C-Class",
    "奔驰E级": "Mercedes-Benz E-Class", "奔驰V级": "Mercedes-Benz V-Class",
    "宝马3系": "BMW 3 Series", "宝马5系": "BMW 5 Series", "微蓝6": "Velite 6",
    "北京X7": "Beijing X7", "极狐 阿尔法S5": "ARCFOX Alpha S5",
    "极狐 阿尔法T5": "ARCFOX Alpha T5", "smart精灵#1": "smart #1",
    "smart精灵#3": "smart #3", "海狮07EV": "Seal U 07 EV", "驱逐舰05": "Destroyer 05",
    "元PLUS": "Yuan PLUS", "元UP": "Yuan UP", "宋L EV": "Song L EV",
}

SERIES_EN_BASE = {
    "博越": "Boyue ", "探岳": "Tayron ", "迈腾": "Magotan ", "途观": "Tiguan ",
    "途昂": "Teramont ", "速腾": "Sagitar ", "朗逸": "Lavida ", "宝来": "Bora ",
    "威然": "Viloran ", "帕萨特": "Passat ", "凌渡": "Lamando ", "揽巡": "Tavendor ",
    "揽境": "Talagon ", "途岳": "Tharu ", "探歌": "T-Roc ", "瑞虎": "Tiggo ",
    "艾瑞泽": "Arrizo ", "风云": "Fengyun ", "智己": "IM ", "理想": "Li Auto ",
    "深蓝": "Deepal ", "零跑": "Leapmotor ", "飞凡": "Rising ", "银河": "Galaxy ",
    "豪越": "Haoyue ", "星越": "Xingyue ", "瑶光": "Yaoguang ", "追风": "Zhuifeng ",
}


def brand_en(name: str) -> str:
    return BRAND_EN.get(str(name), str(name))


def series_en(name: str, brand: str | None = None) -> str:
    name = str(name)
    if name in SERIES_EN:
        return SERIES_EN[name]
    remainder = name
    if brand and name.startswith(str(brand)):
        remainder = name[len(str(brand)):].strip()
    for base, translated in SERIES_EN_BASE.items():
        if remainder.startswith(base):
            return translated + remainder[len(base):]
    if re.search(r"[A-Za-z0-9]", remainder):
        tokens = re.findall(r"[A-Za-z0-9#.\-+ ]+", remainder)
        translated = " ".join(token.strip() for token in tokens if token.strip())
        if translated:
            return translated
    return name


def _json_safe(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    return value


def _write_json(filename: str, payload: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / filename
    compact = filename in {"drilldown.json", "brand_drilldown.json"}
    options = {"ensure_ascii": False, "separators": (",", ":")} if compact else {
        "ensure_ascii": False, "indent": 2,
    }
    path.write_text(json.dumps(_json_safe(payload), **options), encoding="utf-8")
    print(f"  [ok] {filename} ({path.stat().st_size / 1024 / 1024:.2f} MB)")


def main() -> None:
    print("== 构建看板数据 ==")
    payloads = DashboardData(ROOT, brand_en, series_en).payloads()
    for filename, payload in payloads.items():
        _write_json(filename, payload)
    print("== 完成 ==")


if __name__ == "__main__":
    main()
