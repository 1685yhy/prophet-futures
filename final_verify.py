#!/usr/bin/env python3
"""
FINAL VERIFICATION — Production-ready strategy for LH + JM.
1% risk per trade, safety stops, walk-forward validated.
"""

import sys; sys.path.insert(0, ".")
import json, numpy as np, pandas as pd
from datetime import datetime, timedelta
from tools.indicators import calc_indicators, _calc_macd
from tools.cycle_detector import detect_cycle, detect_rollover_noise

LOT = {"lh":16,"jm":60}
COMM = 0.0001; SLIP = 0.0002

def fetch(sym, days=1800):
    import akshare as ak
    e=datetime.now(); s=e-timedelta(days=days+200)
    try:
        df=ak.futures_main_sina(sym.upper()+"0",s.strftime("%Y%m%d"),e.strftime("%Y%m%d"))
        df.columns=["date","open","high","low","close","volume","oi","settle"]
        for c in ["open","high","low","close","volume","oi"]: df[c]=df[c].astype(float)
        return df.reset_index(drop=True)
    except: return None

def signal(df_w, ind, mc=7, vol=False):
    c=df_w["close"].values.astype(float)
    cy=detect_cycle(df_w); ns=detect_rollover_noise(df_w)
    if cy["cycle"] not in ("BULL","BEAR"): return None
    if ns["is_noise"]: return None
    _,_,h0=_calc_macd(c)
    _,_,h1=_calc_macd(c[:-1]) if len(c)>1 else (0,0,0)
    _,_,h2=_calc_macd(c[:-2]) if len(c)>2 else (0,0,0)
    mi=bool(h0<0 and h1<0 and abs(h0)<abs(h1)<abs(h2))
    adx=ind.get("adx14",0); rsi=ind.get("rsi14",50)
    m5=ind.get("ma5",0); m20=ind.get("ma20",0); m60=ind.get("ma60",0)
    mh=ind.get("macd_hist",0)
    oi_c="oi" if "oi" in df_w.columns else None
    oi=df_w[oi_c].values.astype(float) if oi_c else np.zeros(10)
    o3=float(oi[-1]-oi[-4]) if len(oi)>=4 else 0
    o5=float(oi[-1]-oi[-6]) if len(oi)>=6 else o3
    ot="ACCUMULATING" if (o3>0 and o5>0) else ("REDUCING" if (o3<0 and o5<0) else "FLAT")
    mb=m5>m20>m60; mbe=m5<m20<m60
    sc=sum([cy["cycle"]=="BEAR",mbe,mh<0 and not mi,ot in ("REDUCING","FLAT"),adx>20,32<rsi<72,True,True])
    lc=sum([cy["cycle"]=="BULL",mb,mh>0,ot=="ACCUMULATING",adx>22,30<rsi<65,True,True])
    if vol:
        vs=df_w["volume"].values.astype(float)
        if np.mean(vs[-5:])<np.mean(vs[-20:])*1.05: return None
    if sc>=mc: return "SHORT"
    if lc>=mc: return "LONG"
    return None

def backtest(df,sym,cap,risk_pct=0.01,stop_atr=1.5,tgt_atr=3.0,mc=7,vol=False):
    tr=[]; pos=None; W=60; lot=LOT.get(sym,10); consec_loss=0
    for i in range(W,len(df)-1):
        t=df.iloc[i]; ds=str(t["date"])
        w=df.iloc[i-W:i+1].copy(); ind=calc_indicators(w)
        atr=ind["atr14"]; cl=float(t["close"]); hi=float(t["high"]); lo=float(t["low"])
        if pos:
            d=pos["d"]; e=pos["e"]; hd=i-pos["idx"]
            gp=abs(cl-float(df.iloc[i-1]["close"]))/max(1,float(df.iloc[i-1]["close"]))
            fe=gp>0.025 or hd>=20
            hs=(d=="LONG" and lo<=pos["s"])or(d=="SHORT" and hi>=pos["s"])
            ht=(d=="LONG" and hi>=pos["t"])or(d=="SHORT" and lo<=pos["t"])
            te=hd>=8 and ((d=="LONG" and cl<=e)or(d=="SHORT" and cl>=e))
            if not(hs or ht or te or fe):
                if d=="LONG": pos["s"]=max(pos["s"],cl-1.5*atr)
                else: pos["s"]=min(pos["s"],cl+1.5*atr)
            if hs or ht or fe or te:
                ep=pos["s"] if hs else (pos["t"] if ht else cl)
                rs="STOP" if hs else ("TP" if ht else ("GAP" if gp>0.025 else ("TIME" if te else "MAX_HOLD")))
                pnl=(ep-e)*lot*pos["q"]*(1 if d=="LONG" else -1)
                pnl-=abs(ep*lot*pos["q"]*COMM)+abs(e*lot*pos["q"]*COMM)
                tr.append({"pnl":round(pnl,2),"reason":rs,"hold":hd,
                           "entry_date":pos.get("date",""),"exit_date":ds,
                           "symbol":sym,"direction":d})
                if pnl<0: consec_loss+=1
                else: consec_loss=0
                pos=None
        if pos is None and i<len(df)-2:
            if consec_loss>=3: continue  # Safety: pause after 3 consecutive losses
            sg=signal(w,ind,mc,vol)
            if sg:
                d=sg; entry=cl+SLIP*cl*(1 if d=="LONG" else -1)
                sd=max(atr*0.5,atr*stop_atr); td=atr*tgt_atr
                sp=entry-sd if d=="LONG" else entry+sd
                tp=entry+td if d=="LONG" else entry-td
                risk_cash=cap*risk_pct
                q=round(max(1.0,min(20.0,risk_cash/(sd*lot))),1)
                pos={"d":d,"e":round(entry,2),"s":round(sp,2),
                     "t":round(tp,2),"idx":i,"q":q,"date":ds}
    return tr

def stats(tr,cap):
    if not tr: return {"t":0,"wr":0,"pnl":0,"ret":0,"dd":0,"sr":0,"pf":0}
    p=[t["pnl"] for t in tr]; n=len(p)
    w=[x for x in p if x>0]; l=[x for x in p if x<=0]
    wr=len(w)/n; aw=np.mean(w) if w else 0; al=abs(np.mean(l)) if l else 1
    tp=sum(p); ret=tp/cap*100
    pf=sum(w)/(abs(sum(l))+1e-8)
    cum=np.cumsum(p); rm=np.maximum.accumulate(cum)
    dd=abs(float(np.min((cum-rm)/(cap+1e-8)*100)))
    if n>=3:
        rt=[p[i]/cap for i in range(n)]
        sr=np.mean(rt)/(np.std(rt,ddof=1)+1e-8)*np.sqrt(252)
    else: sr=0
    dy=(datetime.strptime(tr[-1]["exit_date"],"%Y-%m-%d")-datetime.strptime(tr[0]["entry_date"],"%Y-%m-%d")).days if tr else 365
    mo=n/(dy/30.44) if dy>0 else 0
    return {"t":n,"wr":round(wr,3),"aw":round(aw,0),"al":round(al,0),
            "pnl":round(tp,0),"ret":round(ret,1),"dd":round(dd,1),
            "sr":round(sr,3),"pf":round(pf,2),"mo":round(mo,1),
            "yr":round(ret*365/(dy+1),1),"rw":round(aw/(al+1e-8),2)}

# ═══════════════════════════════════════════════════════════════════════════
print("="*60)
print("  最终验证 — LH + JM 生产策略")
print("="*60)

cap=1_000_000
for sym in ["lh","jm"]:
    df=fetch(sym)
    if df is None: continue
    print(f"\n{'─'*60}")
    print(f"  {sym.upper()} ({len(df)}天, {df.iloc[0]['date']}→{df.iloc[-1]['date']})")

    # Test with and without volume filter
    for vol in [False, True]:
        mc=7; sa=1.5; ta=3.0
        tr=backtest(df,sym,cap,0.01,sa,ta,mc,vol)
        s=stats(tr,cap)
        tag="+VOL" if vol else "BASE"
        print(f"  {tag:<6} MC={mc} S={sa} T={ta} R=1% → "
              f"{s['t']:>4}t  {s['wr']:.0%}wr  {s['pnl']:+,.0f}元  "
              f"{s['ret']:+.1f}%  DD{s['dd']:.1f}%  SR{s['sr']:.2f}  "
              f"{s['mo']:.1f}/mo  ~{s['yr']:.1f}%/yr")

    # Walk-forward
    n=len(df); sz=n//4
    wf_all=[]
    for sp in range(3):
        te=df.iloc[sz*(sp+1):min(sz*(sp+2),n)]
        if len(te)<60: continue
        tr=backtest(te,sym,cap,0.01,sa,ta,mc,False)
        s2=stats(tr,cap)
        wf_all.extend(tr)
        print(f"  WF{sp+1}: {s2['t']}t {s2['wr']:.0%}wr {s2['pnl']:+,.0f} DD{s2['dd']:.1f}%")
    agg=stats(wf_all,cap)
    print(f"  WF合计: {agg['t']}t {agg['wr']:.0%}wr {agg['pnl']:+,.0f}元 (+{agg['ret']:.1f}%) DD{agg['dd']:.1f}%")

    # Loss streak analysis
    cons=[t["pnl"] for t in tr]
    streaks=[]; cur=0
    for p in cons:
        if p<0: cur+=1
        else: streaks.append(cur); cur=0
    if cur: streaks.append(cur)
    max_streak=max(streaks) if streaks else 0
    streak_dist={i:streaks.count(i) for i in range(1,max_streak+1)}
    print(f"  最大连亏: {max_streak}笔  分布: {streak_dist}")

# Portfolio
print(f"\n{'='*60}")
print(f"  组合汇总")
print(f"{'='*60}")
lh_df=fetch("lh"); jm_df=fetch("jm")
lh_tr=backtest(lh_df,"lh",cap,0.01,1.5,3.0,7,False)
jm_tr=backtest(jm_df,"jm",cap,0.01,1.5,3.0,7,True)
all_tr=lh_tr+jm_tr
all_tr.sort(key=lambda x: x["entry_date"])
s=stats(all_tr,cap)
print(f"  LH(1%风险) + JM(1%风险+VOL):")
print(f"  {s['t']}笔  {s['wr']:.0%}wr  {s['pnl']:+,.0f}元  {s['ret']:+.1f}%  DD{s['dd']:.1f}%")
print(f"  {s['mo']:.1f}笔/月  ~{s['yr']:.1f}%/年  SR{s['sr']:.2f}  PF{s['pf']:.2f}")
print(f"  R:R={s['rw']:.2f}  均盈{s['aw']:,.0f}  均亏{s['al']:,.0f}")

# Monthly PnL
monthly={}
for t in all_tr:
    m=t["exit_date"][:7]
    monthly[m]=monthly.get(m,0)+t["pnl"]
print(f"\n  月度收益:")
for m in sorted(monthly.keys())[-12:]:
    pnl=monthly[m]; pct=pnl/cap*100
    bar="█"*max(0,int(pnl/2000))+"░"*max(0,int(-pnl/2000))
    print(f"  {m}: {pnl:+,.0f} ({pct:+.1f}%) {bar}")

# Save
json.dump({"lh":stats(lh_tr,cap),"jm":stats(jm_tr,cap),"portfolio":s,"monthly":monthly},
          open("/tmp/final_verification.json","w"),indent=2,ensure_ascii=False)
print(f"\n验证完成 → /tmp/final_verification.json")
