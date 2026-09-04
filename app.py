import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="NIFTY Institutional Dashboard",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
.main {background:#0b0f14;}
.block-container {padding-top:1rem;}
.metric-card {padding:14px;border-radius:12px;background:#121923;border:1px solid #263241;}
.small {color:#9aa7b5;font-size:0.85rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=300)
def yf_download(ticker, period="1y", interval="1d", start=None):
    kwargs = dict(progress=False, auto_adjust=False)
    if start:
        df = yf.download(ticker, start=start, interval=interval, **kwargs)
    else:
        df = yf.download(ticker, period=period, interval=interval, **kwargs)

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        if len(set(df.columns.get_level_values(-1))) == 1:
            df.columns = df.columns.get_level_values(0)
        else:
            df.columns = [c[0] for c in df.columns]
    return df


def series(df, col):
    x = df[col]
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0]
    return pd.to_numeric(x, errors="coerce").dropna()


@st.cache_data(ttl=300)
def core_data():
    nifty = yf_download("^NSEI", "1y")
    vix = yf_download("^INDIAVIX", "1y")
    bank = yf_download("^NSEBANK", "1y")
    return nifty, vix, bank


def iv_metrics(vix):
    c = series(vix, "Close")
    cur, hi, lo = float(c.iloc[-1]), float(c.max()), float(c.min())
    ivr = ((cur - lo) / (hi - lo) * 100) if hi != lo else np.nan
    ivp = float((c < cur).sum() / len(c) * 100)
    regime = "HIGH VOLATILITY" if ivr > 50 else "LOW VOLATILITY"
    return cur, hi, lo, ivr, ivp, regime


def expected_move(nifty, vix):
    n = series(nifty, "Close")
    v = series(vix, "Close")
    spot, cvix = float(n.iloc[-1]), float(v.iloc[-1])
    move = spot * (cvix / 100) * np.sqrt(1 / 365)
    return spot, cvix, move, spot + move, spot - move


def parkinson(nifty):
    h = series(nifty, "High")
    l = series(nifty, "Low")
    c = series(nifty, "Close")
    n = len(h)
    var = (np.log(h / l) ** 2).sum() / (4 * n * np.log(2))
    pvol = np.sqrt(var) * np.sqrt(252) * 100
    c2c = np.log(c / c.shift(1)).std() * np.sqrt(252) * 100
    return float(pvol), float(c2c), n


def vol_cone(nifty):
    c = series(nifty, "Close")
    r = np.log(c / c.shift(1))
    windows = [10, 20, 30, 60, 90, 120, 180, 252]
    rows = []
    for w in windows:
        rv = r.rolling(w).std() * np.sqrt(252) * 100
        rows.append([w, rv.min(), rv.median(), rv.max(), rv.iloc[-1]])
    return pd.DataFrame(rows, columns=["Window", "Min", "Median", "Max", "Current"])


def hurst_calc(ts):
    if len(ts) < 20:
        return np.nan
    lags = range(2, 20)
    vals = [np.std(ts[lag:] - ts[:-lag]) for lag in lags]
    if any(v <= 0 for v in vals):
        return np.nan
    return float(np.polyfit(np.log(list(lags)), np.log(vals), 1)[0])


def hurst(nifty):
    c = series(nifty, "Close")
    hp = np.log(c).rolling(60).apply(
        lambda x: hurst_calc(x.values), raw=False
    ).dropna()
    cur = float(hp.iloc[-1])
    if cur < 0.45:
        reg = "MEAN REVERTING"
    elif cur > 0.55:
        reg = "TRENDING"
    else:
        reg = "RANDOM WALK"
    return c, hp, cur, reg


def vrp(nifty, vix):
    n = series(nifty, "Close")
    v = series(vix, "Close")
    hv = np.log(n / n.shift(1)).rolling(20).std() * np.sqrt(252) * 100
    d = pd.concat([v.rename("VIX"), hv.rename("HV")], axis=1).dropna()
    d["VRP"] = d["VIX"] - d["HV"]
    cur = d.iloc[-1]
    return d, float(cur.VIX), float(cur.HV), float(cur.VRP)


def correlation(nifty, bank):
    n = series(nifty, "Close")
    b = series(bank, "Close")
    d = pd.concat([n.rename("Nifty"), b.rename("Bank")], axis=1).dropna()
    ret = np.log(d / d.shift(1)).dropna()
    corr = ret["Nifty"].rolling(20).corr(ret["Bank"])
    cur = float(corr.dropna().iloc[-1])
    if cur > 0.80:
        reg = "HIGH CORRELATION"
    elif cur < 0.50:
        reg = "SEVERE DIVERGENCE"
    else:
        reg = "MODERATE DIVERGENCE"
    return d, corr, cur, reg


def liquidity():
    d = yf_download("^NSEI", "5d", "15m")
    if d.empty:
        return d, None

    d["Prev_High"] = d["High"].rolling(20).max().shift(1)
    d["Prev_Low"] = d["Low"].rolling(20).min().shift(1)
    d["Supply_Sweep"] = (
        (d["High"] > d["Prev_High"]) &
        (d["Close"] < d["Prev_High"])
    )
    d["Demand_Sweep"] = (
        (d["Low"] < d["Prev_Low"]) &
        (d["Close"] > d["Prev_Low"])
    )

    last = d.iloc[-1]
    if bool(last.Supply_Sweep):
        reg = "SUPPLY LIQUIDITY SWEPT"
    elif bool(last.Demand_Sweep):
        reg = "DEMAND LIQUIDITY SWEPT"
    else:
        reg = "PRICE DISCOVERY PHASE"
    return d, reg


def regime_score(nifty, bank, ivr, vrpcur, hcur, corrcur, lreg):
    n = series(nifty, "Close")
    b = series(bank, "Close")

    ne20 = n.ewm(span=20, adjust=False).mean().iloc[-1]
    ne50 = n.ewm(span=50, adjust=False).mean().iloc[-1]
    be20 = b.ewm(span=20, adjust=False).mean().iloc[-1]
    be50 = b.ewm(span=50, adjust=False).mean().iloc[-1]

    trend_points = (
        (15 if n.iloc[-1] > ne20 else 0) +
        (15 if ne20 > ne50 else 0) +
        (10 if n.iloc[-1] > ne50 else 0)
    )
    bank_points = (
        (10 if b.iloc[-1] > be20 else 0) +
        (5 if be20 > be50 else 0)
    )
    corr_points = 10 if corrcur >= 0.80 else (5 if corrcur >= 0.50 else 2)
    liq_points = (
        15 if lreg == "DEMAND LIQUIDITY SWEPT"
        else (0 if lreg == "SUPPLY LIQUIDITY SWEPT" else 8)
    )
    hurst_points = 10 if hcur > 0.55 else (6 if hcur >= 0.45 else 4)
    vol_points = (6 if vrpcur > 0 else 2) + (4 if ivr < 50 else 0)

    score = int(round(
        trend_points + bank_points + corr_points +
        liq_points + hurst_points + vol_points
    ))
    score = max(0, min(100, score))

    label = (
        "STRONG BULLISH" if score >= 75 else
        "BULLISH" if score >= 60 else
        "NEUTRAL / WAIT" if score >= 45 else
        "BEARISH" if score >= 30 else
        "STRONG BEARISH"
    )

    reasons = [
        f"NIFTY: {'above' if n.iloc[-1] > ne20 else 'below'} 20-EMA and "
        f"{'20-EMA above' if ne20 > ne50 else '20-EMA below'} 50-EMA.",
        f"BANK NIFTY: "
        f"{'bullish confirmation' if (b.iloc[-1] > be20 and be20 > be50) else 'not fully confirming the NIFTY trend'}.",
        f"Correlation r={corrcur:.2f}: "
        f"{'good cross-index confirmation' if corrcur >= .80 else 'divergence reduces conviction' if corrcur < .50 else 'moderate confirmation'}.",
        f"Liquidity: {lreg.lower()}.",
        f"Volatility: IVR {ivr:.1f}% and VRP {vrpcur:+.2f}% "
        f"({'option premium above realized vol' if vrpcur > 0 else 'option premium below realized vol'}).",
        f"Hurst {hcur:.3f}: "
        f"{'trend-friendly' if hcur > .55 else 'mean-reverting' if hcur < .45 else 'mixed/random'} regime."
    ]

    parts = dict(
        trend=trend_points,
        bank=bank_points,
        correlation=corr_points,
        liquidity=liq_points,
        hurst=hurst_points,
        volatility=vol_points
    )
    return score, label, reasons, parts, ne20, ne50


def market_decision(nifty, score):
    c = series(nifty, "Close")
    high = series(nifty, "High")
    low = series(nifty, "Low")

    ema20 = float(c.ewm(span=20, adjust=False).mean().iloc[-1])
    ema50 = float(c.ewm(span=50, adjust=False).mean().iloc[-1])

    tr = pd.concat([
        high.rename("H"),
        low.rename("L"),
        c.shift(1).rename("PC")
    ], axis=1).dropna()

    tr["TR"] = np.maximum(
        tr.H - tr.L,
        np.maximum(abs(tr.H - tr.PC), abs(tr.L - tr.PC))
    )
    atr = float(tr.TR.rolling(14).mean().iloc[-1])
    spot = float(c.iloc[-1])

    # Previous 5-session extremes, excluding today's candle.
    prior_5_high = float(high.shift(1).rolling(5).max().iloc[-1])
    prior_5_low = float(low.shift(1).rolling(5).min().iloc[-1])

    if not np.isfinite(atr) or atr <= 0:
        return (
            "WAIT", spot, ema20, ema50, atr,
            np.nan, np.nan, np.nan, np.nan, np.nan,
            np.nan, "ATR(14) is unavailable or invalid. Wait for valid volatility data."
        )

    # Anti-chase filter: do not initiate a fresh trade when price is
    # already more than 1 ATR away from EMA20.
    extension_atr = abs(spot - ema20) / atr
    too_extended = extension_atr > 1.0

    bullish_structure = spot > ema20 > ema50
    bearish_structure = spot < ema20 < ema50

    # The entry zone is deliberately centered around EMA20.
    # This makes it a pullback/retest zone rather than a chase zone.
    zone_half = 0.25 * atr

    if score >= 60 and bullish_structure and not too_extended:
        decision = "BUY"

        # Long: wait for a pullback toward EMA20.
        entry_low = ema20 - zone_half
        entry_high = ema20 + zone_half
        planned_entry = ema20

        sl = entry_low - 1.0 * atr
        t1 = planned_entry + 1.5 * atr
        t2 = planned_entry + 2.5 * atr

        # Fresh breakout confirmation is above the recent 5-session high.
        trigger_level = max(entry_high, prior_5_high)

        reason = (
            "Bullish regime and EMA structure are aligned. "
            f"Prefer a pullback into {entry_low:,.0f}–{entry_high:,.0f}; "
            f"then require a close above {trigger_level:,.0f} for breakout confirmation."
        )

    elif score <= 44 and bearish_structure and not too_extended:
        decision = "SELL / SHORT"

        # Short: wait for a pullback toward EMA20.
        entry_low = ema20 - zone_half
        entry_high = ema20 + zone_half
        planned_entry = ema20

        sl = entry_high + 1.0 * atr
        t1 = planned_entry - 1.5 * atr
        t2 = planned_entry - 2.5 * atr

        # Fresh breakdown confirmation is below the recent 5-session low.
        trigger_level = min(entry_low, prior_5_low)

        reason = (
            "Bearish regime and EMA structure are aligned. "
            f"Prefer a pullback into {entry_low:,.0f}–{entry_high:,.0f}; "
            f"then require a close below {trigger_level:,.0f} for breakdown confirmation."
        )

    else:
        decision = "WAIT"
        entry_low = entry_high = sl = t1 = t2 = np.nan
        planned_entry = np.nan
        trigger_level = np.nan

        if too_extended and bearish_structure and score <= 44:
            reason = (
                f"Bearish setup is valid but NIFTY is {extension_atr:.2f} ATR below EMA20. "
                "Avoid chasing. Wait for a pullback toward EMA20 and then a fresh breakdown."
            )
        elif too_extended and bullish_structure and score >= 60:
            reason = (
                f"Bullish setup is valid but NIFTY is {extension_atr:.2f} ATR above EMA20. "
                "Avoid chasing. Wait for a pullback toward EMA20 and then a fresh breakout."
            )
        elif 45 <= score < 60:
            reason = "Market regime is mixed/neutral. Wait for stronger confirmation."
        elif bullish_structure:
            reason = "Bullish EMA structure exists, but regime score is not strong enough."
        elif bearish_structure:
            reason = "Bearish EMA structure exists, but regime score is not weak enough."
        else:
            reason = "EMA trend structure is not aligned. Wait for clearer direction."

    return (
        decision, spot, ema20, ema50, atr,
        entry_low, entry_high, sl, t1, t2,
        trigger_level, reason
    )


def signal_details(
    decision, score, spot, entry_low, entry_high,
    sl, t1, t2, trigger_level
):
    if decision == "BUY":
        strength = "STRONG BUY" if score >= 75 else "MODERATE BUY"
        trigger = (
            f"BUY setup: pullback to {entry_low:,.0f}–{entry_high:,.0f}; "
            f"then close above {trigger_level:,.0f}"
        )
        planned_entry = (entry_low + entry_high) / 2
        risk = planned_entry - sl
        reward_t1 = t1 - planned_entry
        reward_t2 = t2 - planned_entry

    elif decision == "SELL / SHORT":
        strength = "STRONG SELL" if score <= 29 else "MODERATE SELL"
        trigger = (
            f"SHORT setup: pullback to {entry_low:,.0f}–{entry_high:,.0f}; "
            f"then close below {trigger_level:,.0f}"
        )
        planned_entry = (entry_low + entry_high) / 2
        risk = sl - planned_entry
        reward_t1 = planned_entry - t1
        reward_t2 = planned_entry - t2

    else:
        strength = "NO TRADE"
        trigger = "Wait for BUY or SHORT confirmation."
        risk = reward_t1 = reward_t2 = np.nan

    rr_t1 = reward_t1 / risk if risk > 0 else np.nan
    rr_t2 = reward_t2 / risk if risk > 0 else np.nan

    return strength, trigger, rr_t1, rr_t2


# ---------------- MAIN DATA ----------------

nifty, vix, bank = core_data()

if nifty.empty or vix.empty or bank.empty:
    st.error("Yahoo Finance data unavailable. Try Refresh Data in the sidebar.")
    st.stop()

with st.sidebar:
    st.title("⚙️ Dashboard Controls")
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    st.caption("Data refresh/cache: ~5 minutes")
    st.divider()
    st.info(
        "This dashboard combines the uploaded notebooks into one statistical "
        "market-structure view. It is not investment advice."
    )

st.title("📊 NIFTY Institutional Market Dashboard")
st.caption(
    "Unified: IVR/IVP • Expected Move • Parkinson • Volatility Cone • "
    "Hurst • VRP • Correlation • Liquidity"
)

spot = float(series(nifty, "Close").iloc[-1])
cvix, hi, lo, ivr, ivp, ivreg = iv_metrics(vix)
_, _, move, upper, lower = expected_move(nifty, vix)
pv, c2c, N = parkinson(nifty)
_, _, hcur, hreg = hurst(nifty)
_, _, _, vrpcur = vrp(nifty, vix)
_, _, corrcur, correg = correlation(nifty, bank)

_, lreg = liquidity()
if lreg is None:
    lreg = "PRICE DISCOVERY PHASE"

reg_score, reg_label, reg_reasons, reg_parts, reg_ema20, reg_ema50 = regime_score(
    nifty, bank, ivr, vrpcur, hcur, corrcur, lreg
)

(
    market_decision_value, decision_spot, decision_ema20,
    decision_ema50, decision_atr, decision_entry_low,
    decision_entry_high, decision_sl, decision_t1,
    decision_t2, decision_trigger_level, decision_reason
) = market_decision(nifty, reg_score)

(
    signal_strength, entry_trigger, rr_t1, rr_t2
) = signal_details(
    market_decision_value,
    reg_score,
    decision_spot,
    decision_entry_low,
    decision_entry_high,
    decision_sl,
    decision_t1,
    decision_t2,
    decision_trigger_level
)

# ---------------- TOP SNAPSHOT ----------------

cols = st.columns(6)
for col, title, value, helptext in [
    (cols[0], "NIFTY", f"{spot:,.2f}", "Latest NIFTY 50 close"),
    (cols[1], "INDIA VIX", f"{cvix:.2f}", "Latest India VIX"),
    (cols[2], "IVR", f"{ivr:.1f}%", "1-year IV Rank"),
    (cols[3], "IVP", f"{ivp:.1f}%", "1-year IV Percentile"),
    (cols[4], "VRP", f"{vrpcur:+.2f}%", "VIX minus 20-day realized volatility"),
    (cols[5], "HURST", f"{hcur:.3f}", "60-day rolling Hurst"),
]:
    col.metric(title, value, help=helptext)

st.divider()

score_col, why_col = st.columns([1, 2])
with score_col:
    st.metric("Institutional Regime Score", f"{reg_score}/100")
    st.subheader(reg_label)

with why_col:
    st.subheader("Why this score?")
    for r in reg_reasons:
        st.write("• " + r)

with st.expander("Score breakdown"):
    bd = pd.DataFrame({
        "Component": [
            "NIFTY trend", "BANK NIFTY", "Correlation",
            "Liquidity", "Hurst regime", "Volatility context"
        ],
        "Points": [
            reg_parts["trend"], reg_parts["bank"],
            reg_parts["correlation"], reg_parts["liquidity"],
            reg_parts["hurst"], reg_parts["volatility"]
        ]
    })
    st.dataframe(bd, use_container_width=True, hide_index=True)

# ---------------- MARKET DECISION ----------------

st.subheader("🚦 Market Decision")

dc1, dc2 = st.columns([1, 2])
with dc1:
    st.metric("BUY / SELL / WAIT", market_decision_value)

with dc2:
    st.info(decision_reason)

if market_decision_value == "BUY":
    st.write(
        f"**🟢 BUY Entry Zone:** "
        f"{decision_entry_low:,.0f} – {decision_entry_high:,.0f}"
    )
    st.write(
        f"**🛑 BUY Stop Loss:** {decision_sl:,.0f}  |  "
        f"**T1:** {decision_t1:,.0f}  | **T2:** {decision_t2:,.0f}"
    )

elif market_decision_value == "SELL / SHORT":
    st.write(
        f"**🔴 SHORT Entry Zone:** "
        f"{decision_entry_low:,.0f} – {decision_entry_high:,.0f}"
    )
    st.write(
        f"**🛑 SHORT Stop Loss:** {decision_sl:,.0f}  |  "
        f"**T1:** {decision_t1:,.0f}  | **T2:** {decision_t2:,.0f}"
    )

else:
    st.warning(
        "WAIT — no fresh BUY or SHORT entry until the required confirmation appears."
    )

st.subheader("💪 Signal Strength")
st.metric("Signal", signal_strength)

st.write(f"**🎯 Entry Trigger:** {entry_trigger}")

if np.isfinite(rr_t1):
    st.write(
        f"**⚖️ Risk : Reward:** "
        f"T1 = 1:{rr_t1:.2f}  | T2 = 1:{rr_t2:.2f}"
    )

if market_decision_value == "SELL / SHORT":
    st.caption(
        "Do not chase the short after a large move. Prefer the stated entry zone "
        "and confirmation trigger."
    )
elif market_decision_value == "BUY":
    st.caption(
        "Do not chase the long after a large move. Prefer the stated entry zone "
        "and confirmation trigger."
    )

# ---------------- TRADING SETUP ----------------

st.subheader("🎯 Trading Setup")

tc = st.columns(5)
tc[0].metric("NIFTY", f"{decision_spot:,.2f}")
tc[1].metric("EMA 20", f"{decision_ema20:,.2f}")
tc[2].metric("EMA 50", f"{decision_ema50:,.2f}")
tc[3].metric("ATR 14", f"{decision_atr:,.1f}")
tc[4].metric("Expected Move", f"±{move:,.1f}")

# ---------------- TABS ----------------

tabs = st.tabs([
    "🏠 Overview", "🌡️ Volatility", "📈 Regimes",
    "💧 Liquidity", "🎯 OI Profile"
])

with tabs[0]:
    st.subheader("Market Structure Snapshot")
    x1, x2, x3 = st.columns(3)
    x1.metric("Regime Score", f"{reg_score}/100")
    x2.metric("Regime", reg_label)
    x3.metric("Trading Decision", market_decision_value)

    a, b, c = st.columns(3)
    a.metric("Expected Daily Move", f"±{move:,.1f} pts")
    b.metric("Parkinson Volatility", f"{pv:.2f}%")
    c.metric("Close-to-Close Volatility", f"{c2c:.2f}%")

    st.write("### Statistical Regimes")
    table = pd.DataFrame({
        "Module": ["IVR / IVP", "VRP", "Hurst", "Nifty–Bank Correlation"],
        "Current": [
            ivreg,
            "POSITIVE VRP" if vrpcur > 0 else "NEGATIVE VRP",
            hreg,
            correg
        ],
        "Value": [
            f"IVR {ivr:.1f}% / IVP {ivp:.1f}%",
            f"{vrpcur:+.2f}%",
            f"H={hcur:.3f}",
            f"r={corrcur:.2f}"
        ]
    })
    st.dataframe(table, use_container_width=True, hide_index=True)

with tabs[1]:
    st.subheader("IV Rank & IV Percentile")
    c = series(vix, "Close")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=c.index, y=c, name="India VIX", line=dict(width=2)
    ))
    fig.add_hline(
        y=hi, line_dash="dash", annotation_text=f"52W High {hi:.2f}"
    )
    fig.add_hline(
        y=lo, line_dash="dash", annotation_text=f"52W Low {lo:.2f}"
    )
    fig.add_hline(y=cvix, annotation_text=f"Current {cvix:.2f}")
    fig.update_layout(template="plotly_dark", height=430)
    st.plotly_chart(fig, use_container_width=True)

    st.write(
        f"**Regime:** {ivreg}  |  **IVR:** {ivr:.1f}%  |  **IVP:** {ivp:.1f}%"
    )

    st.subheader("Expected Move")
    nd = series(nifty, "Close").tail(30)
    fig = go.Figure(go.Scatter(x=nd.index, y=nd, name="NIFTY"))
    fig.add_hline(
        y=upper, line_dash="dash", annotation_text=f"+1 SD {upper:.0f}"
    )
    fig.add_hline(
        y=lower, line_dash="dash", annotation_text=f"-1 SD {lower:.0f}"
    )
    fig.add_hline(y=spot, annotation_text=f"Spot {spot:.0f}")
    fig.update_layout(template="plotly_dark", height=380)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"VIX {cvix:.2f} → estimated 1-day 1σ move ±{move:.1f} points. "
        "The original notebook uses √365."
    )

    st.subheader("Volatility Cone")
    cone = vol_cone(nifty)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cone.Window, y=cone.Max, name="Max", mode="lines+markers"
    ))
    fig.add_trace(go.Scatter(
        x=cone.Window, y=cone.Min, name="Min", mode="lines+markers"
    ))
    fig.add_trace(go.Scatter(
        x=cone.Window, y=cone.Median, name="Median", mode="lines+markers"
    ))
    fig.add_trace(go.Scatter(
        x=cone.Window, y=cone.Current, name="Current", mode="lines+markers"
    ))
    fig.update_layout(
        template="plotly_dark",
        height=430,
        xaxis_title="Trading Days",
        yaxis_title="Annualized Volatility %"
    )
    st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    st.subheader("Hurst Regime")
    hc, hs, hcur, hreg = hurst(nifty)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hs.index, y=hs, name="60D Hurst"))
    fig.add_hline(
        y=.55, line_dash="dash", annotation_text="Trending > 0.55"
    )
    fig.add_hline(
        y=.45, line_dash="dash", annotation_text="Mean Reverting < 0.45"
    )
    fig.add_hline(y=.50, line_dash="dot")
    fig.update_layout(
        template="plotly_dark", height=430,
        yaxis=dict(range=[.3, .7])
    )
    st.plotly_chart(fig, use_container_width=True)
    st.info(f"Current Hurst = {hcur:.3f} → **{hreg}**")

    st.subheader("Volatility Risk Premium")
    vd, vixcur, hvcur, vrpcur = vrp(nifty, vix)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=vd.index, y=vd.VIX, name="India VIX"))
    fig.add_trace(go.Scatter(
        x=vd.index, y=vd.HV, name="20D Realized Vol"
    ))
    fig.add_trace(go.Bar(
        x=vd.index, y=vd.VRP, name="VRP", opacity=.45
    ))
    fig.update_layout(template="plotly_dark", height=430)
    st.plotly_chart(fig, use_container_width=True)
    st.info(
        f"VRP = {vrpcur:+.2f}% → "
        f"{'POSITIVE VRP' if vrpcur > 0 else 'NEGATIVE VRP'}"
    )

    st.subheader("NIFTY vs BANK NIFTY Correlation")
    cd, cs, curr, creg = correlation(nifty, bank)
    norm = cd / cd.iloc[0] * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=norm.index, y=norm.Nifty, name="NIFTY (base 100)"
    ))
    fig.add_trace(go.Scatter(
        x=norm.index, y=norm.Bank, name="BANK NIFTY (base 100)"
    ))
    fig.add_hline(y=100, line_dash="dot")
    fig.update_layout(template="plotly_dark", height=350)
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure(go.Scatter(
        x=cs.index, y=cs, name="20D Correlation"
    ))
    fig2.add_hline(
        y=.8, line_dash="dash", annotation_text="0.80"
    )
    fig2.add_hline(
        y=.5, line_dash="dash", annotation_text="0.50"
    )
    fig2.update_layout(
        template="plotly_dark",
        height=300,
        yaxis=dict(range=[-.2, 1.1])
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.info(f"Current 20D correlation = {curr:.2f} → **{creg}**")

    st.subheader("Parkinson vs Close-to-Close")
    x, y, n = parkinson(nifty)
    p1, p2, p3 = st.columns(3)
    p1.metric("Trading Days", n)
    p2.metric("Parkinson", f"{x:.2f}%")
    p3.metric("Close-to-Close", f"{y:.2f}%")
    st.caption(
        "Parkinson emphasizes intraday high-low information; "
        "close-to-close uses daily log returns."
    )

with tabs[3]:
    st.subheader("15-Minute Liquidity Sweep Detector")
    ld, lreg = liquidity()
    if ld.empty:
        st.warning("15-minute Yahoo Finance data unavailable.")
    else:
        recent = ld.tail(80)
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=recent.index,
            open=recent.Open,
            high=recent.High,
            low=recent.Low,
            close=recent.Close,
            name="NIFTY"
        ))
        sup = recent[recent.Supply_Sweep]
        dem = recent[recent.Demand_Sweep]
        fig.add_trace(go.Scatter(
            x=sup.index, y=sup.High, mode="markers",
            name="Supply Sweep",
            marker_symbol="triangle-down",
            marker_size=11
        ))
        fig.add_trace(go.Scatter(
            x=dem.index, y=dem.Low, mode="markers",
            name="Demand Sweep",
            marker_symbol="triangle-up",
            marker_size=11
        ))
        fig.update_layout(
            template="plotly_dark",
            height=550,
            xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig, use_container_width=True)
        st.metric("Current Microstructure Regime", lreg)
        st.caption(
            "Supply sweep = structural high pierced but closed below; "
            "demand sweep = structural low pierced but closed above."
        )

with tabs[4]:
    st.subheader("Live NIFTY Options OI Profile")
    st.warning(
        "This module needs a valid Zerodha Kite API key + access token. "
        "No credentials are embedded."
    )

    key = st.text_input("Kite API Key", type="password")
    token = st.text_input("Kite Access Token", type="password")

    if key and token:
        try:
            from kiteconnect import KiteConnect

            kite = KiteConnect(api_key=key)
            kite.set_access_token(token)

            q = kite.quote(["NSE:NIFTY 50"])
            s = float(q["NSE:NIFTY 50"]["last_price"])

            inst = pd.DataFrame(kite.instruments("NFO"))
            opt = inst[
                (inst["name"] == "NIFTY") &
                (inst["segment"] == "NFO-OPT")
            ].copy()

            opt["expiry"] = pd.to_datetime(opt["expiry"])
            expiry = opt["expiry"].min()
            opt = opt[opt["expiry"] == expiry]
            opt = opt[
                (opt["strike"] >= s - 1000) &
                (opt["strike"] <= s + 1000)
            ]

            symbols = ["NFO:" + x for x in opt["tradingsymbol"]]
            quotes = kite.quote(symbols)

            opt["OI"] = opt["tradingsymbol"].map(
                lambda x: quotes.get("NFO:" + x, {}).get("oi", np.nan)
            )

            calls = opt[opt["instrument_type"] == "CE"].groupby(
                "strike"
            )["OI"].sum()

            puts = opt[opt["instrument_type"] == "PE"].groupby(
                "strike"
            )["OI"].sum()

            strikes = sorted(set(calls.index) | set(puts.index))
            calls = calls.reindex(strikes).fillna(0)
            puts = puts.reindex(strikes).fillna(0)

            pain = {}
            strike_array = np.array(strikes)

            for k in strikes:
                call_pain = (
                    np.maximum((k - strike_array) * calls.values, 0)
                    .sum()
                )
                put_pain = (
                    np.maximum((strike_array - k) * puts.values, 0)
                    .sum()
                )
                pain[k] = call_pain + put_pain

            mp = min(pain, key=pain.get)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=strikes, x=-puts.values, name="Put OI"
            ))
            fig.add_trace(go.Bar(
                y=strikes, x=calls.values, name="Call OI"
            ))
            fig.add_hline(y=s, annotation_text=f"Spot {s:.0f}")
            fig.add_hline(
                y=mp,
                line_dash="dash",
                annotation_text=f"Max Pain {mp}"
            )
            fig.update_layout(
                template="plotly_dark",
                barmode="relative",
                height=650,
                xaxis_title="OI (Put negative / Call positive)",
                yaxis_title="Strike"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.write(
                f"**Highest Put Wall:** {puts.idxmax():.0f}  |  "
                f"**Highest Call Wall:** {calls.idxmax():.0f}  |  "
                f"**Max Pain:** {mp}"
            )

        except Exception as e:
            st.error(f"Kite connection/data error: {e}")

    else:
        st.info(
            "Enter credentials only if you want the live OI module. "
            "All other dashboard modules work from Yahoo Finance."
        )

st.divider()
st.caption(
    "Source engines: Yahoo Finance for NIFTY/India VIX/Bank NIFTY; "
    "optional Zerodha Kite for live options OI. Statistical outputs should "
    "be interpreted with market context."
)
