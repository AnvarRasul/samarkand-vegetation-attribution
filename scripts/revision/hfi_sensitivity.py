"""Sensitivity of the attribution to the 2018->2019 HFI discontinuity."""
import sys, hashlib, json
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, rasterio, pymannkendall as mk
from scipy import stats
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import os
os.chdir(os.environ.get("DATA_ROOT", "."))  # folder holding the data and outputs (see README)

def rd(p):
    with rasterio.open(p) as s: return s.read(1).astype(np.float64)
d=np.load("analysis_stacks.npz"); vm=d['valid_mask']; t=d['transform']
hfp=d['hfp_annual'].astype(np.float64)[:, vm].T          # (n,10)
ndvi_s=rd("Trends/ndvi_sen_slope.tif")[vm]
CC=rd("RESTREND/CC.tif")[vm]
res=np.load("RESTREND/residuals.npz")["residuals"].astype(np.float64)[:, vm].T
res_s=rd("RESTREND/residual_sen_slope.tif")[vm]
hfi_s0=rd("Trends/hfi_sen_slope.tif")[vm]

# --- discontinuity diagnostics
dif=np.diff(hfp,axis=1)          # 9 annual changes
J=dif[:,3]                       # 2018->2019
other=np.delete(dif,3,axis=1)
print("Province mean HFI by year:", np.round(hfp.mean(0),2).tolist())
print(f"Mean jump 2018->2019: {J.mean():+.2f}; mean |other annual change|: {np.abs(other).mean():.2f}")
print(f"Share of pixels where |J| > max |other change|: {np.mean(np.abs(J)>np.abs(other).max(1))*100:.1f}%")
print(f"Share of pixels with J>0: {np.mean(J>0)*100:.1f}%")
# share of total 2015-2024 net change due to the jump (pixels with rising HFI)
net=hfp[:,-1]-hfp[:,0]

def sen_mk(X):
    def f(row):
        r=mk.original_test(row); return r.slope, r.p
    out=Parallel(n_jobs=-1,batch_size=512)(delayed(f)(x) for x in X)
    return np.array(out)

def fit_e(H):
    n=H.shape[0]; X=np.stack([H,np.ones_like(H)],2)
    XtX=np.einsum("pti,ptj->pij",X,X); Xty=np.einsum("pti,pt->pi",X,res)
    return np.einsum("pij,pj->pi",np.linalg.pinv(XtX),Xty)[:,0]

def classify(CC,HA):
    aC,aH=np.abs(CC),np.abs(HA); mx=np.maximum(aC,aH)
    with np.errstate(all='ignore'): ratio=np.where(mx>0,np.abs(CC-HA)/mx,np.inf)
    c=np.zeros(CC.shape,np.uint8); rest=ndvi_s>0; degr=ndvi_s<0
    c[rest&(ratio<0.2)]=3; c[degr&(CC<0)&(HA<0)&(ratio<0.2)]=6
    f=rest&(c==0); c[f&(CC>0)&(HA>0)&(CC>HA)]=1; c[f&(CC>0)&(HA>0)&(HA>=CC)]=2
    f=degr&(c==0); c[f&(CC<0)&(aC>aH)]=4; c[f&(HA<0)&(aH>aC)]=5
    return c

rows,cols=np.where(vm); lats=t[5]+(rows+0.5)*t[4]; lons=t[2]+(cols+0.5)*t[0]
split=np.array([int(hashlib.md5(f"{a:.6f}_{b:.6f}".encode()).hexdigest()[:8],16)%100<80 for a,b in zip(lats,lons)])
def rf_imp(H):
    n=H.shape[0]; yrs=np.arange(2015,2025)
    cols_=[d[k].astype(np.float64)[:,vm].T.ravel() for k in ["lst_annual","precip_annual","srad_annual"]]
    X=np.column_stack(cols_+[H.ravel(),np.repeat(lats,10),np.repeat(lons,10),np.tile(yrs,n)])
    y=d['ndvi_annual'].astype(np.float64)[:,vm].T.ravel(); tr=np.repeat(split,10)
    rf=RandomForestRegressor(n_estimators=300,max_depth=12,n_jobs=-1,random_state=42).fit(X[tr],y[tr])
    imp=rf.feature_importances_; r2=r2_score(y[~tr],rf.predict(X[~tr]))
    phys=imp[:4]; return imp[3], phys[3]/phys.sum()*100, r2

scen={}
scen["S0 original"]=hfp.copy()
adj=hfp.copy(); adj[:,4:]-=J[:,None]; scen["S1 step-adjusted"]=adj
fro=hfp.copy(); fro[:,4:]=hfp[:,[3]]; scen["S2 frozen at 2018"]=fro
out={}
for name,H in scen.items():
    sp=sen_mk(H); hs,hp=sp[:,0],sp[:,1]
    e=fit_e(H); HA=e*hs; c=classify(CC,HA)
    tot=np.abs(CC)+np.abs(HA)
    with np.errstate(all='ignore'): hap=np.where(tot>0,np.abs(HA)/tot*100,np.nan)
    pr=stats.pearsonr(res_s,hs); sr=stats.spearmanr(res_s,hs)
    imp,imp_phys,r2=rf_imp(H)
    cnt={k:int((c==k).sum()) for k in range(7)}
    # longitude of class 5
    lon5=lons[c==5].mean() if cnt[5] else float('nan')
    o=dict(hfi_slope_mean=hs.mean(),hfi_slope_median=np.median(hs),hfi_sig=np.mean(hp<0.05)*100,
           hfi_sig_pos=np.mean((hp<0.05)&(hs>0))*100, hfi_sig_neg=np.mean((hp<0.05)&(hs<0))*100,
           HA_mean=HA.mean(),HA_median=np.median(HA),HApct_median=np.nanmedian(hap),
           classes={k:(v,round(v/len(c)*100,1)) for k,v in cnt.items()},lon_class5=lon5,
           pearson=pr[0],spearman=sr[0],rf_hfi_imp=imp,rf_hfi_phys_pct=imp_phys,rf_r2=r2,
           west_class5=int(((c==5)&(lons<66.1)).sum()))
    if name.startswith("S0"): print("check vs original slope maxdiff:", np.nanmax(np.abs(hs-hfi_s0)))
    out[name]=o
    print("\n==",name); [print(f"  {k}: {v}") for k,v in o.items()]
json.dump(out,open(sys.argv[1],"w"),indent=1,default=float)
