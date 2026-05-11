# =========================================================
# AI 行为量化系统 v9.1 - Web 版 (Streamlit)
# 使用方法：在终端运行 streamlit run 本文件名.py
# =========================================================

import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
import datetime
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange

# =========================================================
# 网页配置（必须是第一个 Streamlit 命令）
# =========================================================
st.set_page_config(
    page_title="AI 行为量化系统 v9.1",
    page_icon="🤖",
    layout="wide"
)

# =========================================================
# 参数区
# =========================================================

SCAN_INTERVAL = 30
SIGNAL_SCORE = 75
STRONG_SIGNAL_SCORE = 90
MAX_WORKERS = 6

FLASH_CRASH_THRESHOLD = 0.03
BLACKLIST_COOLDOWN = 3600 * 4

# =========================================================
# 主流币池
# =========================================================

SYMBOLS = [
    'BTCUSDT','ETHUSDT','SOLUSDT','BNBUSDT','XRPUSDT',
    'DOGEUSDT','ADAUSDT','AVAXUSDT','LINKUSDT',
    'NEARUSDT','APTUSDT','ARBUSDT','OPUSDT',
    'SUIUSDT','INJUSDT','SEIUSDT','TIAUSDT',
    'WIFUSDT','1000PEPEUSDT','FETUSDT'
]

# =========================================================
# 黑名单
# =========================================================

blacklist = {}

# =========================================================
# 防限流
# =========================================================

def rate_limit():
    time.sleep(random.uniform(0.08, 0.18))

# =========================================================
# 获取K线 (多源熔断)
# =========================================================

def get_klines(symbol, interval='5m', limit=150):
    urls = [
        'https://fapi.binance.com/fapi/v1/klines',
        'https://api.binance.com/api/v3/klines'
    ]
    params = {'symbol': symbol, 'interval': interval, 'limit': limit}
    for url in urls:
        try:
            r = requests.get(url, params=params, timeout=8)
            rate_limit()
            data = r.json()
            if isinstance(data, list):
                return data
        except:
            continue
    return None

# =========================================================
# 辅助数据
# =========================================================

def get_ticker(symbol):
    url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
    try:
        r = requests.get(url, params={'symbol': symbol}, timeout=6)
        rate_limit()
        return r.json()
    except:
        return None

def get_funding(symbol):
    url = "https://fapi.binance.com/fapi/v1/premiumIndex"
    try:
        r = requests.get(url, params={'symbol': symbol}, timeout=6)
        rate_limit()
        data = r.json()
        return float(data['lastFundingRate'])
    except:
        return 0.0

def get_fear_greed():
    url = "https://api.alternative.me/fng/"
    try:
        r = requests.get(url, timeout=6)
        data = r.json()
        return int(data['data'][0]['value'])
    except:
        return 50

# =========================================================
# DataFrame
# =========================================================

def klines_to_df(klines):
    df = pd.DataFrame(klines, columns=[
        'time','open','high','low','close','volume',
        'close_time','qav','trades',
        'tb_base','tb_quote','ignore'
    ])
    for col in ['open','high','low','close','volume']:
        df[col] = df[col].astype(float)
    return df

# =========================================================
# 技术指标计算
# =========================================================

def calc_atr(df, period=14):
    atr = AverageTrueRange(df['high'], df['low'], df['close'], period)
    return atr.average_true_range().iloc[-1]

def calc_slope(series, periods=5):
    if len(series) < periods:
        return 0
    y = series[-periods:].values
    x = np.arange(periods)
    slope = np.polyfit(x, y, 1)[0]
    return slope

# =========================================================
# 位置系统
# =========================================================

def analyze_position(df):
    last = df.iloc[-1]
    ema99 = last['ema99']
    close = last['close']
    deviation_pct = (close - ema99) / ema99 if ema99 > 0 else 0
    recent_high = df['high'][-30:].max()
    recent_low = df['low'][-30:].min()
    high_break = close >= recent_high * 0.998
    low_break = close <= recent_low * 1.002

    position = "中位"
    if deviation_pct > 0.05:       position = "高位"
    elif deviation_pct > 0.15:     position = "极高位"
    elif deviation_pct < -0.05:    position = "低位"
    elif deviation_pct < -0.15:    position = "极低位"

    return {
        'position': position,
        'deviation': deviation_pct,
        'is_high_break': high_break,
        'is_low_break': low_break
    }

# =========================================================
# EMA 斜率与趋势动能
# =========================================================

def analyze_ema_momentum(df):
    ema7 = df['ema7']
    ema25 = df['ema25']
    slope7 = calc_slope(ema7, 5)
    slope25 = calc_slope(ema25, 5)

    if ema7.iloc[-1] > ema25.iloc[-1]:
        gap = ema7.iloc[-1] - ema25.iloc[-1]
        gap_prev = ema7.iloc[-5] - ema25.iloc[-5]
        if gap > gap_prev and slope7 > 0:
            return "趋势加速"
        elif gap < gap_prev:
            return "趋势衰减"
        else:
            return "稳定多头"
    elif ema7.iloc[-1] < ema25.iloc[-1]:
        gap = ema25.iloc[-1] - ema7.iloc[-1]
        gap_prev = ema25.iloc[-5] - ema7.iloc[-5]
        if gap > gap_prev and slope7 < 0:
            return "趋势加速"
        elif gap < gap_prev:
            return "趋势衰减"
        else:
            return "稳定空头"
    else:
        return "震荡纠缠"

# =========================================================
# 连续量能系统
# =========================================================

def analyze_volume_continuity(df):
    volumes = df['volume']
    ma20 = volumes.tail(20).mean()
    recent_vol = volumes.tail(5)
    surge_count = sum(1 for v in recent_vol if v > ma20 * 1.3)
    big_surge_count = sum(1 for v in recent_vol if v > ma20 * 1.8)
    current_vol = volumes.iloc[-1]
    ratio20 = current_vol / ma20 if ma20 > 0 else 0

    if big_surge_count >= 3:
        continuity = "主力持续进场"
    elif surge_count >= 4:
        continuity = "连续放量"
    elif surge_count >= 2:
        continuity = "温和放量"
    elif ratio20 > 2.0:
        continuity = "单根脉冲"
    else:
        continuity = "缩量"

    return {
        'ratio20': ratio20,
        'continuity': continuity,
        'surge_count': surge_count,
        'is_shrink': ratio20 < 0.7,
        'is_surge': ratio20 > 1.3,
        'is_big_volume': ratio20 > 1.8,
        'is_huge_volume': ratio20 > 2.5
    }

# =========================================================
# 基础结构与假突破
# =========================================================

def analyze_structure(df):
    last = df.iloc[-1]
    open_, close, high, low = last['open'], last['close'], last['high'], last['low']
    body = abs(close - open_)
    total = high - low
    if total == 0: total = 0.00001
    upper = high - max(open_, close)
    lower = min(open_, close) - low

    return {
        'bullish': close > open_,
        'bearish': close < open_,
        'upper_ratio': upper / total,
        'lower_ratio': lower / total,
        'rejection': upper > body * 1.5,
        'support': lower > body * 1.5,
        'hammer': lower > total * 0.6 and close > open_,
        'shooting_star': upper > total * 0.6 and close < open_
    }

def detect_fake_breakout(df, atr):
    last = df.iloc[-1]
    open_, close, high, low = last['open'], last['close'], last['high'], last['low']
    body = abs(close - open_)
    total = high - low
    if total == 0: return None
    upper = high - max(open_, close)
    lower = min(open_, close) - low
    base_ratio = 0.35 + (atr / last['close']) * 5
    base_ratio = min(base_ratio, 0.55)
    if upper / total > base_ratio and close < open_: return "fake_up"
    if lower / total > base_ratio and close > open_: return "fake_down"
    return None

def smart_money_behavior(volume_info, structure):
    if volume_info['is_big_volume'] and structure['shooting_star']: return "疑似出货"
    if volume_info['is_huge_volume'] and structure['bullish']: return "主力拉升"
    if volume_info['is_shrink']: return "缩量观察"
    return "正常"

# =========================================================
# 动态市场环境
# =========================================================

def get_market_regime(btc_env, fear):
    if fear > 80 or btc_env == 'bear':
        regime = '谨慎'
        score_adj = 5
        vol_adj = 0.2
    elif fear < 30 or btc_env == 'bull':
        regime = '积极'
        score_adj = -5
        vol_adj = -0.2
    else:
        regime = '中性'
        score_adj = 0
        vol_adj = 0.0
    return regime, score_adj, vol_adj

# =========================================================
# BTC环境
# =========================================================

def btc_environment():
    k = get_klines('BTCUSDT', '30m', 100)
    if not k: return 'neutral'
    df = klines_to_df(k)
    ema7 = EMAIndicator(df['close'], 7).ema_indicator()
    ema25 = EMAIndicator(df['close'], 25).ema_indicator()
    macd = MACD(df['close'])
    dif = macd.macd().iloc[-1]
    dea = macd.macd_signal().iloc[-1]
    if ema7.iloc[-1] > ema25.iloc[-1] and dif > dea: return 'bull'
    elif ema7.iloc[-1] < ema25.iloc[-1] and dif < dea: return 'bear'
    return 'neutral'

# =========================================================
# 熔断检查
# =========================================================

def check_flash_crash(symbol):
    global blacklist
    if symbol in blacklist and time.time() < blacklist[symbol]: return True
    k1 = get_klines(symbol, '1m', 3)
    if k1 and len(k1) >= 2:
        p1 = float(k1[-1][4])
        p2 = float(k1[-3][4])
        if p2 != 0 and abs(p1 / p2 - 1) > FLASH_CRASH_THRESHOLD:
            blacklist[symbol] = time.time() + BLACKLIST_COOLDOWN
            return True
    return False

# =========================================================
# AI 综合分析引擎（核心逻辑与你 v9.1 完全相同）
# =========================================================

def analyze_symbol(symbol, btc_env, fear, market_regime, score_adj, vol_adj):
    if check_flash_crash(symbol): return None
    try:
        k5 = get_klines(symbol, '5m', 150)
        k30 = get_klines(symbol, '30m', 150)
        if not k5 or not k30: return None

        df5 = klines_to_df(k5)
        df30 = klines_to_df(k30)

        atr5 = calc_atr(df5, 14)
        if atr5 == 0: atr5 = 1e-9
        atr30 = calc_atr(df30, 14)

        for df in [df30, df5]:
            df['ema7'] = EMAIndicator(df['close'], 7).ema_indicator()
            df['ema25'] = EMAIndicator(df['close'], 25).ema_indicator()
            df['ema99'] = EMAIndicator(df['close'], 99).ema_indicator()

        macd5 = MACD(df5['close'])
        dif5, dea5 = macd5.macd().iloc[-1], macd5.macd_signal().iloc[-1]
        macd30 = MACD(df30['close'])
        dif30, dea30 = macd30.macd().iloc[-1], macd30.macd_signal().iloc[-1]

        position = analyze_position(df30)
        momentum = analyze_ema_momentum(df30)
        vol_info = analyze_volume_continuity(df5)

        structure = analyze_structure(df5)
        fake_break = detect_fake_breakout(df5, atr5)
        funding = get_funding(symbol)
        behavior = smart_money_behavior(vol_info, structure)

        # ----- 双向评分 -----
        bull_score, bear_score = 0, 0
        bull_reasons, bear_reasons = [], []

        # 1. 位置系统
        if position['position'] in ['低位', '极低位']:
            bull_score += 25; bull_reasons.append(f"低位({position['position']})")
        if position['position'] in ['高位', '极高位']:
            bear_score += 25; bear_reasons.append(f"高位({position['position']})")
        if position['is_high_break']:
            bull_score += 10; bull_reasons.append("突破近期高")
        if position['is_low_break']:
            bear_score += 10; bear_reasons.append("跌破近期低")

        # 2. 趋势动能
        if momentum == "趋势加速":
            if dif30 > 0 and dea30 > 0: bull_score += 20; bull_reasons.append("多头加速")
            elif dif30 < 0 and dea30 < 0: bear_score += 20; bear_reasons.append("空头加速")
        elif momentum == "趋势衰减":
            if dif30 > 0: bull_score += 5; bull_reasons.append("多头减速")
            elif dif30 < 0: bear_score += 5; bear_reasons.append("空头减速")
        else:
            if dif30 > 0: bull_score += 10
            elif dif30 < 0: bear_score += 10

        # 3. 连续量能
        if vol_info['continuity'] == "主力持续进场":
            bull_score += 30; bear_score += 30
        elif vol_info['continuity'] == "连续放量":
            bull_score += 20; bear_score += 20
        elif vol_info['continuity'] == "单根脉冲":
            bull_score += 5; bear_score += 5

        # 4. 裸K与假突破
        if structure['hammer'] and position['is_low_break']:
            bull_score += 25; bull_reasons.append("低位锤子线")
        if structure['shooting_star'] and position['is_high_break']:
            bear_score += 25; bear_reasons.append("高位射击星")
        if structure['support']:
            bull_score += 10; bull_reasons.append("下影支撑")
        if structure['rejection']:
            bear_score += 15; bear_reasons.append("上影抛压")

        if fake_break == 'fake_down': bull_score += 25; bull_reasons.append("诱空陷阱")
        elif fake_break == 'fake_up': bull_score -= 25
        if fake_break == 'fake_up': bear_score += 25; bear_reasons.append("诱多陷阱")
        elif fake_break == 'fake_down': bear_score -= 25

        # 5. MACD
        if dif5 > dea5: bull_score += 5; bull_reasons.append("金叉")
        if dif5 < dea5: bear_score += 5; bear_reasons.append("死叉")

        # 6. 环境与情绪
        if btc_env == 'bull': bull_score += 10
        elif btc_env == 'bear': bear_score += 10
        if fear < 35: bull_score += 10; bear_score -= 10
        elif fear > 75: bull_score -= 10; bear_score += 10
        if funding < -0.001: bull_score += 8; bear_score -= 8
        elif funding > 0.001: bull_score -= 8; bear_score += 8

        # ----- 放量硬性开关 -----
        has_volume = vol_info['is_surge'] or vol_info['is_big_volume'] or vol_info['is_huge_volume']
        effective_threshold = SIGNAL_SCORE + score_adj

        final_signal, final_score, reasons = None, 0, []
        if bull_score >= effective_threshold and bull_score >= bear_score and dif5 > dea5 and has_volume:
            final_signal, final_score, reasons = "做多", bull_score, bull_reasons
        elif bear_score >= effective_threshold and bear_score > bull_score and dif5 < dea5 and has_volume:
            final_signal, final_score, reasons = "做空", bear_score, bear_reasons

        if final_signal:
            signal_grade = "🔥 S级" if final_score >= STRONG_SIGNAL_SCORE else "⭐ A级"
            return {
                'symbol': symbol, 'direction': final_signal,
                'score': final_score, 'signal': signal_grade,
                'price': round(df5['close'].iloc[-1], 4),
                'position': position, 'momentum': momentum,
                'continuity': vol_info['continuity'], 'behavior': behavior,
                'fear': fear, 'funding': funding,
                'volume_ratio': round(vol_info['ratio20'], 2),
                'atr5': round(atr5, 4), 'reasons': reasons
            }
    except: pass
    return None

# =========================================================
# 网页界面（Streamlit）
# =========================================================

st.title("🤖 AI 行为量化交易系统 v9.1")
st.markdown("##### 核心引擎：位置系统 · EMA动能 · 连续量能 · 金叉死叉放量硬性开关")

# 侧边栏
with st.sidebar:
    st.header("⚙️ 控制面板")
    scan_status = st.empty()
    progress_bar = st.progress(0)

# 主区域
col1, col2, col3 = st.columns(3)
market_info = st.empty()
signal_table = st.empty()

# 运行控制
if "running" not in st.session_state:
    st.session_state.running = False

def start_scan():
    st.session_state.running = True

def stop_scan():
    st.session_state.running = False

start_button = st.button("▶️ 开始扫描", on_click=start_scan, disabled=st.session_state.running)
stop_button = st.button("⏹️ 停止扫描", on_click=stop_scan, disabled=not st.session_state.running)

if st.session_state.running:
    round_num = 0
    while st.session_state.running:
        round_num += 1

        btc_env = btc_environment()
        fear = get_fear_greed()
        regime, score_adj, vol_adj = get_market_regime(btc_env, fear)

        with col1:
            st.metric("BTC环境", btc_env.upper())
        with col2:
            st.metric("市场情绪", f"{fear} (Fear & Greed)")
        with col3:
            st.metric("市场状态", f"{regime} (阈值{score_adj:+d})")

        results = []
        total_symbols = len(SYMBOLS)
        completed = 0

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(analyze_symbol, s, btc_env, fear, regime, score_adj, vol_adj) for s in SYMBOLS]
            for future in as_completed(futures):
                result = future.result()
                completed += 1
                progress_bar.progress(completed / total_symbols)
                if result:
                    results.append(result)

        progress_bar.empty()

        results = sorted(results, key=lambda x: x['score'], reverse=True)

        if results:
            signal_text = ""
            for r in results:
                signal_text += f"### {r['signal']} {r['symbol']} | {r['direction']} | 评分: {r['score']} | 价格: {r['price']}\n"
                signal_text += f"- 位置: {r['position']['position']} | 动能: {r['momentum']} | 量能: {r['continuity']}\n"
                signal_text += f"- 逻辑: {' | '.join(r['reasons'])}\n"
                signal_text += "---\n"
            signal_table.markdown(signal_text)
        else:
            signal_table.info(f"第 {round_num} 轮扫描完成，暂无信号。等待 {SCAN_INTERVAL} 秒后继续...")

        scan_status.info(f"✅ 第 {round_num} 轮扫描完成 | 发现 {len(results)} 个信号 | {datetime.datetime.now().strftime('%H:%M:%S')}")
        time.sleep(SCAN_INTERVAL)
