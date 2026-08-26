"""Collect recent sales for configured series missing from the monthly panel.

The source exposes only a recent window, so results are written to a separate
audit file and are not merged into the historical panel automatically.
"""
import os, re, time, sys
import pandas as pd
import requests
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "data" / "raw"

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
      "Accept-Language": "zh-CN,zh;q=0.9"}

SLEEP = 1.0  # request interval in seconds


def norm(s):
    return re.sub(r"[\s\-]", "", str(s)).lower()


def get(url, session):
    for _ in range(3):
        try:
            r = session.get(url, timeout=15, headers=UA)
            if r.status_code == 200:
                return r.text
        except Exception:
            pass
        time.sleep(2)
    return ""


def build_brand_map(session):
    """爬销量目录首页 + 分页，收集 (品牌名 -> nb/sg 品牌级 id)。"""
    brands = {}
    page = 1
    while True:
        url = "https://price.pcauto.com.cn/salescar/" if page == 1 else \
              f"https://price.pcauto.com.cn/salescar/?page={page}"
        t = get(url, session)
        if not t:
            break
        # 品牌入口形如 <a href="/salescar/nb845/">哈弗销量</a>
        for bid, bname in re.findall(r'href="/salescar/(nb\d+|sg\d+)/"[^>]*>([^<]+?)销量</a>', t):
            name = bname.strip()
            brands[name] = bid
        # 是否还有下一页
        if f"?page={page+1}" not in t and page > 1:
            break
        page += 1
        if page > 30:
            break
        time.sleep(SLEEP)
    return brands


def build_series_map(session, brands):
    """对每个品牌页，收集 (车系名 -> sg/nb 系列级 id)。"""
    smap = {}
    for bname, bid in brands.items():
        t = get(f"https://price.pcauto.com.cn/salescar/{bid}/", session)
        if not t:
            continue
        for sid, sname in re.findall(r'href="/salescar/(sg\d+|nb\d+)/"[^>]*>([^<]+)</a>', t):
            smap[norm(sname)] = (sid, sname.strip(), bname)
        time.sleep(SLEEP)
    return smap


def parse_sales_page(html, sid):
    """从销量页解析近期月度销量 + 年内累计 + 上年累计。"""
    # 表头月度链接（按顺序）
    months = re.findall(r'/salescar/' + re.escape(sid) + r'/y(\d{4})-m(\d{1,2})/', html)
    # 数据行
    m = re.search(r'<tr class="tr-ser">(.*?)</tr>', html, re.S)
    if not m or not months:
        return []
    row = m.group(1)
    tds = re.findall(r'<td[^>]*>(.*?)</td>', row, re.S)
    # tds[0] = 车系名；之后依次为：各月、年内累计(1-N月)、上年累计
    vals = [re.sub(r"<.*?>", "", x).strip().replace(",", "") for x in tds]
    name = vals[0]
    monthly_vals = vals[1:1 + len(months)]
    tail = vals[1 + len(months):]
    ytd = tail[0] if len(tail) > 0 else ""
    cum = tail[1] if len(tail) > 1 else ""
    out = []
    for (y, mo), v in zip(months, monthly_vals):
        try:
            v = int(v)
        except Exception:
            v = None
        out.append({"year": int(y), "month": int(mo), "monthly_sales": v})
    out.append({"year": int(months[0][0]) if months else None, "month": 0,
                "monthly_sales": ytd, "_note": "ytd_1-N月"})
    out.append({"year": int(months[0][0]) - 1 if months else None, "month": 0,
                "monthly_sales": cum, "_note": "cum_prev_year"})
    return out


def main():
    session = requests.Session()
    feat = pd.read_csv(RAW / "feature.csv")
    feat["series_name"] = feat["series_name"].astype(str)
    ms = pd.read_csv(RAW / "monthly_sales.csv")
    ms["series_name"] = ms["series_name"].astype(str)

    # Reuse known series-level IDs.
    ms_map = {}
    for _, r in ms.iterrows():
        sid = str(r.get("source_series_id", "")).strip()
        if sid and sid.lower() != "nan":
            ms_map[norm(r["series_name"])] = (sid, r["series_name"], "")
    covered = set(ms_map.keys())

    # 缺销量车系
    feat_names = feat["series_name"].unique().tolist()
    missing = [n for n in feat_names if norm(n) not in covered]
    print(f"[01b] feature 车系 {len(feat_names)} | 已由 monthly_sales 覆盖 {len(feat_names)-len(missing)} "
          f"| 缺销量 {len(missing)}")

    # Resolve the remaining IDs from the sales directory.
    map_cache = RAW / "pcauto_series_map.csv"
    if map_cache.exists():
        sm = pd.read_csv(map_cache)
        pc_map = {norm(r["series_name"]): (str(r["sg_id"]), r["series_name"], r["brand"])
                  for _, r in sm.iterrows()}
        print(f"[01b] 载入缓存 pcauto_series_map.csv: {len(pc_map)} 车系")
    else:
        print("[01b] 构建品牌映射(爬销量目录)...")
        brands = build_brand_map(session)
        print(f"[01b] 品牌数 {len(brands)}")
        pc_map = build_series_map(session, brands)
        pd.DataFrame(
            [{"norm_name": k, "sg_id": v[0], "series_name": v[1], "brand": v[2]}
             for k, v in pc_map.items()]
        ).to_csv(map_cache, index=False)
        print(f"[01b] 已建 pcauto_series_map.csv: {len(pc_map)} 车系")

    # 解析 + 抓取
    rows = []
    resolved = 0
    for n in missing:
        kn = norm(n)
        if kn in ms_map:
            sid, sname, _ = ms_map[kn]
            src = "monthly_sales"
        elif kn in pc_map:
            sid, sname, _ = pc_map[kn]
            src = "pcauto_crawl"
        else:
            rows.append({"series_name": n, "pcauto_salescar_id": "", "year": None,
                         "month": None, "monthly_sales": None, "resolved": False,
                         "note": "UNRESOLVED_id"})
            continue
        resolved += 1
        html = get(f"https://price.pcauto.com.cn/salescar/{sid}/", session)
        if not html:
            rows.append({"series_name": n, "pcauto_salescar_id": sid, "year": None,
                         "month": None, "monthly_sales": None, "resolved": True,
                         "note": f"{src}_fetch_fail"})
            continue
        recs = parse_sales_page(html, sid)
        for rec in recs:
            rows.append({"series_name": n, "pcauto_salescar_id": sid,
                         "year": rec.get("year"), "month": rec.get("month"),
                         "monthly_sales": rec.get("monthly_sales"),
                         "resolved": True,
                         "note": f"{src}_{rec.get('_note','monthly')}"})
        time.sleep(SLEEP)

    out = pd.DataFrame(rows)
    out_path = RAW / "sales_crawl_raw.csv"
    out.to_csv(out_path, index=False)
    got = out[(out["resolved"]) & (out["monthly_sales"].notna()) & (out["month"].notna())]
    print(f"[01b] 解析到 id: {resolved}/{len(missing)} | 落盘 {len(out)} 行 -> {out_path}")
    print(f"[01b] 其中真正拿到月度数值的行: {len(got)} | UNRESOLVED: "
          f"{int((~out['resolved']).sum())}")


if __name__ == "__main__":
    main()
