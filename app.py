"""
NIFTY 50 DATA-DRIVEN INSTITUTIONAL DASHBOARD — V6
Combines the user's uploaded quantitative notebooks into one decision engine:
VRP, IVR/IVP, Hurst, Expected Move, Correlation, Volatility Cone,
Parkinson Estimator and 15m Liquidity Sweep, with optional Zerodha OI Profile.

The uploaded notebooks are preserved as the source of the calculations.
The signal/entry/exit layer is an added execution framework: it does not
claim predictive certainty and should be backtested before live use.
"""
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="NIFTY Data-Driven V6 FINAL", page_icon="📊", layout="wide")

TICKERS = {"NIFTY":"^NSEI", "BANKNIFTY":"^NSEBANK", "VIX":"^INDIAVIX"}


def flat(df):
    if df is None or df.empty: return pd.DataFrame()
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex): df.columns=[c[0] for c in df.columns]
    return df

def col(df, name):
    return pd.to_numeric(df[name].squeeze(), errors="coerce")

@st.cache_data(ttl=300, show_spinner=False)
def daily_data():
    n=flat(yf.download(TICKERS["NIFTY"], start="2015-01-01", progress=False, auto_adjust=False))
    b=flat(yf.download(TICKERS["BANKNIFTY"], start="2015-01-01", progress=False, auto_adjust=False))
    v=flat(yf.download(TICKERS["VIX"], start="2015-01-01", progress=False, auto_adjust=False))
    return n,b,v

@st.cache_data(ttl=300, show_spinner=False)
def intraday_data():
    return flat(yf.download(TICKERS["NIFTY"], period="5d", interval="15m", progress=False, auto_adjust=False))

# ---------------- Uploaded-notebook calculations ----------------
def ivr_ivp(vix):
    c=col(vix,"Close").dropna().tail(252)
    if len(c)<20: return {"current":np.nan,"high":np.nan,"low":np.nan,"ivr":np.nan,"ivp":np.nan,"regime":"DATA N/A","series":c}
    cur=float(c.iloc[-1]); hi=float(c.max()); lo=float(c.min())
    ivr=((cur-lo)/(hi-lo)*100) if hi>lo else 0.0
    ivp=float((c<cur).mean()*100)
    regime="HIGH VOLATILITY" if ivr>50 else "LOW VOLATILITY"
    return {"current":cur,"high":hi,"low":lo,"ivr":ivr,"ivp":ivp,"regime":regime,"series":c}

def vrp(nifty,vix):
    nc=col(nifty,"Close"); vc=col(vix,"Close")
    hv=np.log(nc/nc.shift(1)).rolling(20).std()*np.sqrt(252)*100
    d=pd.DataFrame({"VIX":vc,"HV":hv}).dropna()
    if d.empty: return {"vix":np.nan,"hv":np.nan,"vrp":np.nan,"df":d}
    d["VRP"]=d.VIX-d.HV
    return {"vix":float(d.VIX.iloc[-1]),"hv":float(d.HV.iloc[-1]),"vrp":float(d.VRP.iloc[-1]),"df":d.tail(126)}

def hurst_calc(ts):
    if len(ts)<20:return np.nan
    arr=np.asarray(ts,float); lags=range(2,20)
    reg=[np.std(arr[lag:]-arr[:-lag]) for lag in lags]
    if any(x<=0 or not np.isfinite(x) for x in reg): return np.nan
    return float(np.polyfit(np.log(list(lags)),np.log(reg),1)[0])

def hurst(nifty):
    c=col(nifty,"Close").dropna().tail(400); lp=np.log(c)
    hs=lp.rolling(60).apply(hurst_calc,raw=False); d=pd.DataFrame({"Close":c,"Hurst":hs}).dropna()
    h=float(d.Hurst.iloc[-1]) if not d.empty else np.nan
    reg="TRENDING" if h>.55 else ("MEAN REVERTING" if h<.45 else "RANDOM WALK")
    return {"df":d,"price":float(c.iloc[-1]),"hurst":h,"regime":reg}

def expected_move(nifty,vix):
    c=col(nifty,"Close").dropna(); vc=col(vix,"Close").dropna()
    spot=float(c.iloc[-1]); vv=float(vc.iloc[-1]); move=spot*(vv/100)*np.sqrt(1/365)
    return {"spot":spot,"vix":vv,"move":move,"upper":spot+move,"lower":spot-move,"series":c.tail(30)}

def correlation(nifty,bank):
    a=pd.DataFrame({"NIFTY":col(nifty,"Close"),"BANK":col(bank,"Close")}).dropna().tail(252)
    r=np.log(a/a.shift(1)).dropna(); rc=r.NIFTY.rolling(20).corr(r.BANK); cur=float(rc.iloc[-1]) if not rc.dropna().empty else np.nan
    reg="HIGH CORRELATION" if cur>.80 else ("SEVERE DIVERGENCE" if cur<.50 else "MODERATE DIVERGENCE")
    return {"data":a,"rolling":rc,"corr":cur,"regime":reg}

def vol_cone(nifty):
    d=nifty.copy(); d["ret"]=np.log(col(d,"Close")/col(d,"Close").shift(1))
    wins=[10,20,30,60,90,120,180,252]; out={k:[] for k in ["max","min","median","current"]}
    for w in wins:
        rv=d.ret.rolling(w).std()*np.sqrt(252)*100
        out["max"].append(float(rv.max())); out["min"].append(float(rv.min())); out["median"].append(float(rv.median())); out["current"].append(float(rv.iloc[-1]))
    return {"windows":wins,**out}

def parkinson(nifty):
    d=nifty.tail(252); h=col(d,"High"); l=col(d,"Low"); c=col(d,"Close"); N=len(d)
    pv=np.sqrt((np.log(h/l)**2).sum()/(4*N*np.log(2)))*np.sqrt(252)
    c2c=np.log(c/c.shift(1)).std()*np.sqrt(252)
    return {"parkinson":float(pv),"c2c":float(c2c),"N":N}

def liquidity(df):
    if df is None or df.empty:return {"available":False}
    d=df.copy(); w=20
    d["Prev_High"]=col(d,"High").rolling(w).max().shift(1); d["Prev_Low"]=col(d,"Low").rolling(w).min().shift(1)
    d["Supply_Sweep"]=(col(d,"High")>d.Prev_High)&(col(d,"Close")<d.Prev_High)
    d["Demand_Sweep"]=(col(d,"Low")<d.Prev_Low)&(col(d,"Close")>d.Prev_Low)
    x=d.iloc[-1]
    return {"available":True,"df":d,"price":float(x.Close),"prev_high":float(x.Prev_High) if pd.notna(x.Prev_High) else None,"prev_low":float(x.Prev_Low) if pd.notna(x.Prev_Low) else None,"supply":bool(x.Supply_Sweep),"demand":bool(x.Demand_Sweep)}


def premium_bias(iv, vr):
    """Independent options-premium read from IVR + VRP.
    This does NOT change BUY/SHORT/WAIT direction.
    """
    ivr = iv.get("ivr", np.nan)
    vrpv = vr.get("vrp", np.nan)
    if not (np.isfinite(ivr) and np.isfinite(vrpv)):
        return "NEUTRAL (Data N/A)", "IVR/VRP data unavailable."
    if ivr > 50 and vrpv > 0:
        return ("SELL PREMIUM",
                f"IVR {ivr:.0f}% + VRP {vrpv:+.2f}%: implied volatility is relatively rich.")
    if ivr < 50 and vrpv < 0:
        return ("BUY PREMIUM",
                f"IVR {ivr:.0f}% + VRP {vrpv:+.2f}%: implied volatility is relatively cheap.")
    return ("NEUTRAL",
            f"IVR {ivr:.0f}% / VRP {vrpv:+.2f}%: no clean premium edge.")

# ---------------- Execution layer: added on top of notebook metrics ----------------
def atr14(nifty):
    h,l,c=col(nifty,"High"),col(nifty,"Low"),col(nifty,"Close"); pc=c.shift(1)
    tr=pd.concat([(h-l),(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    return float(tr.rolling(14).mean().iloc[-1])

def ema_levels(nifty):
    c=col(nifty,"Close"); return float(c.iloc[-1]),float(c.ewm(span=20,adjust=False).mean().iloc[-1]),float(c.ewm(span=50,adjust=False).mean().iloc[-1]),float(c.ewm(span=200,adjust=False).mean().iloc[-1])

def oi_profile_from_kite():
    try:
        from kiteconnect import KiteConnect
    except Exception:return {"available":False,"reason":"kiteconnect not installed"}
    key=st.session_state.get("kite_key","").strip(); token=st.session_state.get("kite_token","").strip()
    if not key or not token:return {"available":False,"reason":"Kite credentials not entered"}
    try:
        kite=KiteConnect(api_key=key); kite.set_access_token(token)
        spot=float(kite.quote(["NSE:NIFTY 50"])["NSE:NIFTY 50"]["last_price"])
        inst=pd.DataFrame(kite.instruments("NFO")); opt=inst[(inst.name=="NIFTY")&(inst.segment=="NFO-OPT")].copy()
        opt.expiry=pd.to_datetime(opt.expiry); expiry=opt.expiry.min(); opt=opt[opt.expiry==expiry]
        opt=opt[(opt.strike>=spot-1000)&(opt.strike<=spot+1000)]
        quotes=kite.quote(["NFO:"+x for x in opt.tradingsymbol])
        rows=[]
        for _,r in opt.iterrows():
            q=quotes.get("NFO:"+r.tradingsymbol)
            if q: rows.append({"Strike":float(r.strike),"Type":r.instrument_type,"OI":float(q.get("oi",0))})
        od=pd.DataFrame(rows)
        calls=od[od.Type=="CE"].groupby("Strike").OI.sum(); puts=od[od.Type=="PE"].groupby("Strike").OI.sum(); strikes=np.sort(np.unique(np.r_[calls.index,puts.index]))
        calls=calls.reindex(strikes,fill_value=0); puts=puts.reindex(strikes,fill_value=0)
        pain=[]
        for k in strikes:pain.append(np.sum(np.maximum(0,k-strikes)*calls.values)+np.sum(np.maximum(0,strikes-k)*puts.values))
        mp=float(strikes[int(np.argmin(pain))]); put_wall=float(puts.idxmax()) if len(puts) else np.nan; call_wall=float(calls.idxmax()) if len(calls) else np.nan
        return {"available":True,"spot":spot,"expiry":expiry,"data":pd.DataFrame({"Strike":strikes,"CallOI":calls.values,"PutOI":puts.values}),"max_pain":mp,"put_wall":put_wall,"call_wall":call_wall}
    except Exception as e:return {"available":False,"reason":f"Kite error: {e}"}

def signal_engine(nifty,bank,vix,iv,vr,h,em,corr,cone,pk,liq,oi):
    spot,ema20,ema50,ema200=ema_levels(nifty); atr=atr14(nifty); close=col(nifty,"Close");
    prev5h=float(col(nifty,"High").shift(1).rolling(5).max().iloc[-1]); prev5l=float(col(nifty,"Low").shift(1).rolling(5).min().iloc[-1])
    score=0; reasons=[]; quality=100
    bullish=spot>ema20>ema50 and ema50>ema200; bearish=spot<ema20<ema50 and ema50<ema200
    mom10=float((spot/close.iloc[-11]-1)*100) if len(close)>11 else 0
    # Trend structure is execution-layer context, not from the notebooks.
    if bullish: score+=30; reasons.append("NIFTY EMA structure is bullish (20 > 50 > 200).")
    elif bearish: score-=30; reasons.append("NIFTY EMA structure is bearish (20 < 50 < 200).")
    else: quality-=10; reasons.append("NIFTY EMA structure is mixed; directional conviction reduced.")
    # Hurst decides whether trend-following or mean-reversion logic has priority.
    if h["hurst"]>.55:
        if mom10>0: score+=20; reasons.append(f"Hurst {h['hurst']:.3f} is TRENDING and 10D momentum is positive.")
        else: score-=20; reasons.append(f"Hurst {h['hurst']:.3f} is TRENDING and 10D momentum is negative.")
    elif h["hurst"]<.45:
        quality-=15; reasons.append(f"Hurst {h['hurst']:.3f} is MEAN REVERTING; breakout chasing is filtered.")
        if liq.get("demand"): score+=18; reasons.append("15m demand sweep supports a mean-reversion long.")
        if liq.get("supply"): score-=18; reasons.append("15m supply sweep supports a mean-reversion short.")
    else: quality-=8; reasons.append(f"Hurst {h['hurst']:.3f} is RANDOM WALK; no strong trend edge.")
    # Cross-index confirmation.
    if corr["corr"]>.80:
        bclose=float(col(bank,"Close").iloc[-1]); b20=float(col(bank,"Close").ewm(span=20,adjust=False).mean().iloc[-1]); b50=float(col(bank,"Close").ewm(span=50,adjust=False).mean().iloc[-1])
        if bclose>b20>b50: score+=15; reasons.append("Bank Nifty confirms the bullish direction with high correlation.")
        elif bclose<b20<b50: score-=15; reasons.append("Bank Nifty confirms the bearish direction with high correlation.")
        else: quality-=8; reasons.append("High Nifty–Bank correlation, but Bank Nifty trend is not confirming.")
    elif corr["corr"]<.50:
        quality-=18; reasons.append(f"Nifty–Bank correlation {corr['corr']:.2f}: severe divergence, so conviction is reduced.")
    else: quality-=8; reasons.append(f"Nifty–Bank correlation {corr['corr']:.2f}: moderate divergence.")
    # Volatility context: affects quality/risk, not direction.
    if iv["ivr"]>50 and vr["vrp"]>0: reasons.append(f"IVR {iv['ivr']:.1f}% + VRP {vr['vrp']:+.2f}%: volatility is relatively rich; prefer defined-risk options structures.")
    elif iv["ivr"]<50 and vr["vrp"]<0: reasons.append(f"IVR {iv['ivr']:.1f}% + VRP {vr['vrp']:+.2f}%: volatility is relatively cheap.")
    else: reasons.append(f"IVR {iv['ivr']:.1f}% / VRP {vr['vrp']:+.2f}%: mixed volatility signal.")
    cone_ratio=cone["current"][1]/cone["median"][1] if cone["median"][1] else 1
    if cone_ratio>1.25: quality-=10; reasons.append("20D realized volatility is well above its cone median; use wider risk or wait for cleaner structure.")
    elif cone_ratio<.75: quality-=5; reasons.append("20D realized volatility is below its cone median; breakout follow-through may need confirmation.")
    if pk["parkinson"]>pk["c2c"]: quality-=5; reasons.append("Parkinson volatility exceeds close-to-close volatility: intraday range risk is elevated.")
    # Liquidity confirmation / conflict.
    if liq.get("available"):
        if liq.get("demand") and score<0: quality-=10; reasons.append("Demand sweep conflicts with bearish direction.")
        if liq.get("supply") and score>0: quality-=10; reasons.append("Supply sweep conflicts with bullish direction.")
    # Location + anti-chase.
    ext=abs(spot-ema20)/atr if atr>0 else np.inf
    trend_candidate = (score>=55 and bullish) or (score<=-55 and bearish)
    decision="WAIT"; entry_low=entry_high=trigger=sl=t1=t2=np.nan; setup_type="NO TRADE"
    if trend_candidate and ext<=1.0 and quality>=60:
        half=.25*atr; entry_low=ema20-half; entry_high=ema20+half
        if score>0:
            decision="BUY"; setup_type="TREND PULLBACK BUY"; trigger=max(entry_high,prev5h); sl=entry_low-atr; t1=min(ema20+1.5*atr,em["upper"]); t2=min(ema20+2.5*atr,em["upper"]+0.5*em["move"])
            if oi.get("available") and np.isfinite(oi.get("call_wall",np.nan)): t1=min(t1,oi["call_wall"])
            reasons.append("Trend setup: wait for pullback into EMA20 zone, then daily close above confirmation trigger.")
        else:
            decision="SHORT"; setup_type="TREND PULLBACK SHORT"; trigger=min(entry_low,prev5l); sl=entry_high+atr; t1=max(ema20-1.5*atr,em["lower"]); t2=max(ema20-2.5*atr,em["lower"]-0.5*em["move"])
            if oi.get("available") and np.isfinite(oi.get("put_wall",np.nan)): t1=max(t1,oi["put_wall"])
            reasons.append("Trend setup: wait for pullback into EMA20 zone, then daily close below confirmation trigger.")
    elif h["hurst"]<.45 and quality>=60 and liq.get("available") and (liq.get("demand") or liq.get("supply")):
        # Mean-reversion entry is only allowed after a confirmed sweep near the expected-move boundary.
        if liq.get("demand") and spot<=em["upper"] and score>=10:
            decision="BUY"; setup_type="MEAN-REVERSION BUY"; entry_low=max(em["lower"],liq.get("prev_low",em["lower"])); entry_high=spot; trigger=spot; sl=(liq.get("prev_low") or em["lower"])-.35*atr; t1=min(em["spot"],em["upper"]); t2=em["upper"]; reasons.append("Mean-reversion setup: demand sweep is confirmed; buy only after price closes back above the swept level.")
        elif liq.get("supply") and spot>=em["lower"] and score<=-10:
            decision="SHORT"; setup_type="MEAN-REVERSION SHORT"; entry_low=spot; entry_high=min(em["upper"],liq.get("prev_high",em["upper"])); trigger=spot; sl=(liq.get("prev_high") or em["upper"])+.35*atr; t1=max(em["spot"],em["lower"]); t2=em["lower"]; reasons.append("Mean-reversion setup: supply sweep is confirmed; short only after price closes back below the swept level.")
    if ext>1.0 and ((score>0 and bullish) or (score<0 and bearish)):
        reasons.append(f"Anti-chase filter: NIFTY is {ext:.2f} ATR from EMA20, so a fresh entry is blocked until a pullback.")
    # R:R gate
    planned=(entry_low+entry_high)/2 if np.isfinite(entry_low) and np.isfinite(entry_high) else np.nan
    if decision=="BUY": risk=planned-sl; rr1=(t1-planned)/risk if risk>0 else np.nan; rr2=(t2-planned)/risk if risk>0 else np.nan
    elif decision=="SHORT": risk=sl-planned; rr1=(planned-t1)/risk if risk>0 else np.nan; rr2=(planned-t2)/risk if risk>0 else np.nan
    else: rr1=rr2=np.nan
    if decision!="WAIT" and (not np.isfinite(rr1) or rr1<1.2):
        reasons.append("R:R gate failed: T1 does not offer at least 1.2R from the planned entry, so the trade is downgraded to WAIT.")
        decision="WAIT"; setup_type="NO TRADE"; entry_low=entry_high=trigger=sl=t1=t2=np.nan; rr1=rr2=np.nan
    confidence=int(np.clip(50+abs(score)*.45+(quality-60)*.25,0,95)) if decision!="WAIT" else int(np.clip(50+abs(score)*.25+(quality-60)*.15,0,75))
    return {"decision":decision,"setup":setup_type,"score":score,"quality":max(0,quality),"confidence":confidence,"spot":spot,"ema20":ema20,"ema50":ema50,"ema200":ema200,"atr":atr,"entry_low":entry_low,"entry_high":entry_high,"trigger":trigger,"sl":sl,"t1":t1,"t2":t2,"rr1":rr1,"rr2":rr2,"reasons":reasons,"ext":ext}

# ---------------- UI ----------------
def main():
    st.title("📊 NIFTY 50 Data-Driven Institutional Dashboard — V6")
    st.caption("Notebook metrics → regime → direction → location → confirmation → risk-managed setup + independent premium-bias lens")
    with st.sidebar:
        st.header("Controls")
        if st.button("🔄 Refresh data now"):
            st.cache_data.clear(); st.rerun()
        st.caption("Data cache: 5 minutes")
        st.divider(); st.subheader("Optional Zerodha OI")
        st.text_input("Kite API Key", type="password", key="kite_key")
        st.text_input("Kite Access Token", type="password", key="kite_token")
        st.caption("OI is optional. Leave blank to run without live OI.")
    with st.spinner("Fetching market data..."):
        try:nifty,bank,vix=daily_data(); n15=intraday_data()
        except Exception as e: st.error(f"Data fetch failed: {e}"); st.stop()
    if nifty.empty or bank.empty or vix.empty: st.error("Required Yahoo Finance data is unavailable."); st.stop()
    iv=ivr_ivp(vix); vr=vrp(nifty,vix); h=hurst(nifty); em=expected_move(nifty,vix); co=correlation(nifty,bank); cone=vol_cone(nifty); pk=parkinson(nifty); liq=liquidity(n15); oi=oi_profile_from_kite(); sig=signal_engine(nifty,bank,vix,iv,vr,h,em,co,cone,pk,liq,oi)
    pbias,pnote=premium_bias(iv,vr)
    a,b,c,d,e,f=st.columns(6)
    for x,t,v in [(a,"NIFTY",f"{sig['spot']:,.2f}"),(b,"INDIA VIX",f"{iv['current']:.2f}"),(c,"IVR",f"{iv['ivr']:.1f}%"),(d,"IVP",f"{iv['ivp']:.1f}%"),(e,"VRP",f"{vr['vrp']:+.2f}%"),(f,"HURST",f"{h['hurst']:.3f}")]: x.metric(t,v)
    st.divider()
    sc1,sc2,sc3=st.columns(3)
    sc1.metric("Institutional Direction Score",f"{sig['score']:+d}")
    sc2.metric("Setup Quality",f"{sig['quality']}/100")
    sc3.metric("Confidence",f"{sig['confidence']}%")
    st.subheader("💊 Options Premium Bias")
    pb1,pb2=st.columns([1,2])
    with pb1: st.metric("Premium Bias",pbias)
    with pb2: st.caption(pnote)
    st.caption("Independent of BUY/SHORT/WAIT. This is an IVR + VRP premium-richness lens, not a directional signal.")

    title={"BUY":"🟢 BUY","SHORT":"🔴 SHORT","WAIT":"🟡 WAIT"}[sig['decision']]
    st.subheader(title)
    st.info(f"**{sig['setup']}** — {('NIFTY is '+format(sig['ext'],'.2f')+' ATR from EMA20.') if np.isfinite(sig['ext']) else ''}")
    if sig['decision']!="WAIT":
        q=st.columns(6)
        for x,t,v in [(q[0],"Entry Zone",f"{sig['entry_low']:,.0f}–{sig['entry_high']:,.0f}"),(q[1],"Trigger",f"{sig['trigger']:,.0f}"),(q[2],"Stop Loss",f"{sig['sl']:,.0f}"),(q[3],"T1",f"{sig['t1']:,.0f}"),(q[4],"T2",f"{sig['t2']:,.0f}"),(q[5],"R:R",f"{sig['rr1']:.2f}R / {sig['rr2']:.2f}R")]: x.metric(t,v)
    else: st.warning("No fresh entry. Wait for the stated confirmation rather than chasing price.")
    with st.expander("Why this decision?",expanded=True):
        for r in sig["reasons"]: st.write("• "+r)
    st.subheader("📐 Quantitative Snapshot")
    rows=[
        ["Hurst",f"{h['hurst']:.3f}",h['regime']],
        ["IVR / IVP",f"{iv['ivr']:.1f}% / {iv['ivp']:.1f}%",iv['regime']],
        ["VRP",f"{vr['vrp']:+.2f}%", "POSITIVE" if vr['vrp']>0 else "NEGATIVE"],
        ["Expected Move",f"±{em['move']:.1f} pts",f"{em['lower']:.0f}–{em['upper']:.0f}"],
        ["Nifty–Bank Corr",f"{co['corr']:.2f}",co['regime']],
        ["Vol Cone 20D",f"{cone['current'][1]:.1f}%",f"Median {cone['median'][1]:.1f}%"],
        ["Parkinson / C2C",f"{pk['parkinson']*100:.2f}% / {pk['c2c']*100:.2f}%","Intraday range risk"],
        ["Liquidity", "N/A" if not liq['available'] else ("DEMAND SWEEP" if liq['demand'] else "SUPPLY SWEEP" if liq['supply'] else "PRICE DISCOVERY"), "15m"],
        ["OI", "AVAILABLE" if oi['available'] else "OPTIONAL / NOT CONNECTED", (f"Put {oi['put_wall']:.0f} | Call {oi['call_wall']:.0f} | Max Pain {oi['max_pain']:.0f}" if oi['available'] else oi['reason'])]
    ]
    st.dataframe(pd.DataFrame(rows,columns=["Metric","Value","Interpretation"]),use_container_width=True,hide_index=True)
    tabs=st.tabs(["Volatility","Regimes","Liquidity","OI Profile"])
    with tabs[0]:
        fig,ax=plt.subplots(figsize=(10,4)); ax.plot(iv['series']); ax.axhline(iv['current'],ls='--'); ax.set_title('India VIX — IVR/IVP'); st.pyplot(fig); plt.close(fig)
        fig,ax=plt.subplots(figsize=(10,4)); ax.plot(cone['windows'],cone['median'],label='Median'); ax.plot(cone['windows'],cone['current'],label='Current'); ax.fill_between(cone['windows'],cone['min'],cone['max'],alpha=.15); ax.legend(); ax.set_title('Volatility Cone'); st.pyplot(fig); plt.close(fig)
    with tabs[1]:
        fig,ax=plt.subplots(figsize=(10,4)); ax.plot(h['df'].index,h['df']['Hurst']); ax.axhline(.55,ls='--'); ax.axhline(.45,ls='--'); ax.set_title('60D Hurst Regime'); st.pyplot(fig); plt.close(fig)
        st.write(f"**EMA20:** {sig['ema20']:,.2f}  |  **EMA50:** {sig['ema50']:,.2f}  |  **EMA200:** {sig['ema200']:,.2f}")
    with tabs[2]:
        if liq['available']:
            d=liq['df'].tail(60); fig,ax=plt.subplots(figsize=(10,4)); ax.plot(col(d,'Close')); ax.plot(d['Prev_High'],ls='--'); ax.plot(d['Prev_Low'],ls='--'); ax.set_title('15m Liquidity Sweep'); st.pyplot(fig); plt.close(fig)
        else: st.info("15m data unavailable.")
    with tabs[3]:
        if oi['available']:
            od=oi['data']; fig,ax=plt.subplots(figsize=(10,5)); ax.barh(od.Strike,od.CallOI/1e5,alpha=.7,label='Call OI'); ax.barh(od.Strike,-od.PutOI/1e5,alpha=.7,label='Put OI'); ax.axhline(oi['spot'],ls='-'); ax.axhline(oi['max_pain'],ls='--'); ax.legend(); ax.set_title(f"Live NIFTY OI Profile — {oi['expiry'].date()}"); st.pyplot(fig); plt.close(fig)
        else: st.info("Live OI is optional. Enter Kite API key + access token in the sidebar.")
    st.caption(f"Last run: {datetime.now().strftime('%d-%b-%Y %H:%M:%S')} | Research/education only. No signal guarantees future returns.")

if __name__=="__main__": main()
