#!/usr/bin/env python3
"""
Prophet v7 — 超参调优 + LightGBM + XGBoost + 集成
Grid search on n_estimators, max_depth, learning_rate.
"""

import sys; sys.path.insert(0, ".")
import numpy as np; import lightgbm as lgb; import xgboost as xgb
from run import *

print("="*60)
print("  Prophet v7 — 超参调优 + LGBM vs XGB vs Ensemble")
print("="*60)

THRESH=[0.50,0.55,0.60]; STOPS=[1.0,1.2,1.5]; RISKS=[0.015,0.02]

def wf_backtest(df,sym,model,conf,risk,stop):
    nsp=4;sps=np.linspace(60,len(df),nsp+1).astype(int);pnls=[]
    for sp in range(nsp-1):
        te=sps[sp+1];ve=sps[sp+2]
        if ve>=len(df):break
        td=df.iloc[:te];vd=df.iloc[te:ve].reset_index(drop=True)
        fx=[];ly=[];W=60
        for i in range(W,len(td)-1):
            w=td.iloc[i-W:i+1];ind=calc_indicators(w)
            sg=rule_signal(w,ind,7)
            if sg is None:continue
            f=build_features(td,i)
            if f is None:continue
            nc=float(td.iloc[i+1]['close']);c=float(td.iloc[i]['close'])
            ly.append(1 if(sg=='LONG' and nc>c)or(sg=='SHORT' and nc<c)else 0);fx.append(f)
        if len(fx)<50:continue
        m=model();m.fit(np.array(fx),np.array(ly))
        for i in range(W,len(vd)-1):
            w=vd.iloc[i-W:i+1];ind=calc_indicators(w)
            sg=rule_signal(w,ind,7)
            if sg is None:continue
            f=build_features(vd,i)
            if f is None:continue
            pr=m.predict_proba(f.reshape(1,-1))[0,1]
            if pr<conf:continue
            c=float(vd.iloc[i]['close']);atr=ind['atr14']
            e=c+0.0002*c*(1 if sg=='LONG'else-1)
            sd=max(atr*0.3,atr*stop);td=atr*3.0
            st=e-sd if sg=='LONG'else e+sd;tg=e+td if sg=='LONG'else e-td
            l=LOT.get(sym,10);q=max(1.0,min(20.0,1_000_000*risk/(sd*l)))
            nc=float(vd.iloc[i+1]['close']);nh=float(vd.iloc[i+1]['high']);nl=float(vd.iloc[i+1]['low'])
            ep=st if(sg=='LONG'and nl<=st)or(sg=='SHORT'and nh>=st)else(tg if(sg=='LONG'and nh>=tg)or(sg=='SHORT'and nl<=tg)else nc)
            pnl=(ep-e)*l*q*(1 if sg=='LONG'else-1)-abs(ep*l*q*0.0001)*2
            pnls.append(pnl)
    return pnls

def make_model(name, params):
    if name in ("xgb","XGBoost"):
        return lambda: xgb.XGBClassifier(**params,random_state=42,verbosity=0)
    elif name in ("lgb","LightGBM"):
        return lambda: lgb.LGBMClassifier(**params,random_state=42,verbose=-1)
    return None

# Hyperparameter grids
xgb_params=[
    {"n_estimators":80,"max_depth":4,"learning_rate":0.05},
    {"n_estimators":100,"max_depth":5,"learning_rate":0.05},
    {"n_estimators":150,"max_depth":6,"learning_rate":0.03},
    {"n_estimators":100,"max_depth":4,"learning_rate":0.08},
]
lgb_params=[
    {"n_estimators":80,"max_depth":4,"learning_rate":0.05,"num_leaves":31},
    {"n_estimators":100,"max_depth":5,"learning_rate":0.05,"num_leaves":31},
    {"n_estimators":150,"max_depth":6,"learning_rate":0.03,"num_leaves":63},
    {"n_estimators":100,"max_depth":4,"learning_rate":0.08,"num_leaves":15},
]

results=[]
for sym in ["lh","jm"]:
    df=fetch(sym,2500)
    if df is None:continue
    print(f"\n{'─'*60}")
    print(f"  {sym.upper()} ({len(df)}天)")

    for model_name,param_list in [("XGBoost",xgb_params),("LightGBM",lgb_params)]:
        for pi,params in enumerate(param_list):
            m=make_model(model_name,params)
            for th in THRESH:
                for rp in RISKS:
                    for sa in STOPS:
                        pnls=wf_backtest(df,sym,m,th,rp,sa)
                        if len(pnls)<10:continue
                        w=[p for p in pnls if p>0];n=len(pnls)
                        wr=len(w)/n;tp=sum(pnls);ret=tp/1_000_000*100
                        cum=np.cumsum(pnls);rm=np.maximum.accumulate(cum)
                        dd=abs(float(np.min((cum-rm)/1_000_000*100)))
                        score=wr*ret/(dd/100+0.05)
                        results.append({"sym":sym,"model":model_name,"pi":pi,
                            "th":th,"rp":rp,"sa":sa,"t":n,"wr":wr,"pnl":tp,
                            "ret":ret,"dd":dd,"score":score,"params":str(params)})

results.sort(key=lambda x:x["score"],reverse=True)

print(f"\n{'='*60}")
print(f"  Top 20 (共{len(results)}组合)")
print(f"  {'Rk':<4} {'Sym':<5} {'Model':<10} {'PR#':<4} {'Th':<5} {'R':<5} {'S':<5} {'Trd':<6} {'WR':<7} {'PnL':<12} {'Ret%':<8} {'DD%':<6}")
print(f"  {'─'*85}")
for i,r in enumerate(results[:20]):
    print(f"  {i+1:<4} {r['sym'].upper():<5} {r['model']:<10} {r['pi']:<4} "
          f"{r['th']:<5} {r['rp']:<5} {r['sa']:<5} {r['t']:<6} {r['wr']:.0%}     "
          f"{r['pnl']:+,.0f}     {r['ret']:+.1f}%    {r['dd']:.1f}%")

# Best per model
for mn in["XGBoost","LightGBM"]:
    best=[r for r in results if r["model"]==mn][:3]
    print(f"\n  {mn} Top3:")
    for r in best:
        print(f"  {r['sym'].upper()} PR#{r['pi']} th={r['th']} r={r['rp']} s={r['sa']}: "
              f"{r['t']}t {r['wr']:.0%}wr {r['ret']:+.1f}% DD{r['dd']:.1f}%")
