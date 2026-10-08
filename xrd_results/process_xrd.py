import os
os.environ['MPLCONFIGDIR']='/tmp/xrd_mpl'
os.environ['XDG_CACHE_HOME']='/tmp/xrd_cache'
import zipfile,io,json,sys,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np,pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter,find_peaks
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path(__file__).resolve().parent
src=Path(sys.argv[1]) if len(sys.argv)>1 else Path('/workspace/attachments/07578cc9-0951-4a99-b2de-00c664de00c8/XRD.zip')
z=zipfile.ZipFile(src)
def baseline(y,lam=1e6,p=.02):
 D=sparse.diags([np.ones(len(y)-2),-2*np.ones(len(y)-2),np.ones(len(y)-2)],[0,1,2],shape=(len(y)-2,len(y)),format='csc');w=np.ones(len(y))
 for _ in range(20):
  b=spsolve(sparse.diags(w,format='csc')+lam*(D.T@D),w*y);w=np.where(y>b,p,1-p)
 return b

def pv(x,b,m,h,c,f,e):
 t=(x-c)/f
 return b+m*(x-c)+h*(e/(1+4*t*t)+(1-e)*np.exp(-4*np.log(2)*t*t))
sets=[];metas=[];fits=[];candidates=[];curves=[]
for i,angle in enumerate([.2,.5,1.],1):
 name=f'1_{i:03d}';r=zipfile.ZipFile(io.BytesIO(z.read(name+'.rasx')));root=ET.fromstring(r.read('Data0/MesurementConditions0.xml'))
 a=np.loadtxt(io.StringIO(r.read('Data0/Profile0.txt').decode('utf-8-sig')))
 txt=z.read(name+'.txt').decode('utf-8-sig');a2=np.loadtxt(io.StringIO('\n'.join(l for l in txt.splitlines() if l and l[0].isdigit())))
 x,y=a2.T;assert len(x)==4001 and np.allclose(np.diff(x),.02) and np.all(np.isfinite(y)) and np.all(y>=0)
 assert np.allclose(x,a[:,0]) and np.allclose(y,12.5*a[:,1],atol=.00011)
 bg=baseline(y);net=y-bg;sm=savgol_filter(net,21,3);norm=100*sm/max(sm)
 frame=pd.DataFrame({'two_theta_deg':x,'raw_export_intensity':y,'rasx_intensity':a[:,1],'background_AsLS':bg,'net_unsmoothed':net,'net_SG21_poly3':sm,'normalized_smoothed_max100':norm});frame.to_csv(out/f'{name}_processed.csv',index=False)
 sets.append((name,angle,x,y,bg,net,sm,norm,frame))
 meta={'file':name,'incidence_deg':angle,'n_points':len(x),'start_deg':x[0],'stop_deg':x[-1],'step_deg':.02,'scan_speed_deg_min':15,'target':'Cu','lambda_Ka1_A':1.540593,'lambda_Ka2_A':1.544414,'voltage_kV':40,'current_mA':40,'TXT_RASX_ratio':12.5,'max_export_intensity':float(y.max()),'unit_note':'TXT likely counts/s; RASX metadata says counts; retain native columns'};metas.append(meta)
 # Empirical high-frequency noise; correlated continuous-scan points make nominal significance approximate.
 noise=1.4826*np.median(np.abs(net-savgol_filter(net,11,2)-np.median(net-savgol_filter(net,11,2))))
 peaks,pr=find_peaks(sm,prominence=max(.15*y.max(),3*noise),distance=25,width=10)
 for j,k in enumerate(peaks):
  candidates.append({'file':name,'incidence_deg':angle,'candidate_2theta_deg':x[k],'height_smoothed':sm[k],'prominence':pr['prominences'][j],'status':'candidate only; inspect raw data, not phase identification'})
 for lo,hi,hkl in [(42,46.5,'111'),(49,53.5,'200'),(73,77.5,'220')]:
  mask=(x>=lo)&(x<=hi);xx=x[mask];yy=y[mask];k=np.argmax(sm[mask]);center=xx[k]
  try:
   p,cov=curve_fit(pv,xx,yy,p0=[np.median(yy),0,max(1,np.ptp(yy)/2),center,.5,.5],bounds=([0,-100,0,lo+.2,.08,0],[y.max()*2,100,y.max()*3,hi-.2,2.5,1]),maxfev=30000)
   err=np.sqrt(np.maximum(0,np.diag(cov)));pred=pv(xx,*p);res=yy-pred
   snr=p[2]/max(np.std(res),1e-9);ok=snr>=2 and p[2]/max(err[2],1e-9)>=3 and .081<p[4]<2.49 and lo+.21<p[3]<hi-.21
   d=1.540593/(2*np.sin(np.deg2rad(p[3]/2)))
   fits.append({'file':name,'incidence_deg':angle,'tentative_gamma_hkl':hkl,'fit_2theta_deg':p[3],'center_fit_SE_deg':err[3],'FWHM_observed_deg':p[4],'FWHM_fit_SE_deg':err[4],'height_export_units':p[2],'height_fit_SE':err[2],'height_residual_SD_ratio':snr,'d_apparent_A_using_Ka1':d,'a_apparent_A_if_FCC':d*np.sqrt(sum(int(v)**2 for v in hkl)),'status':'resolved_candidate' if ok else 'weak_or_unstable_do_not_quantify'})
   curves.append((name,hkl,xx,yy,pred,ok))
  except Exception as ex:fits.append({'file':name,'incidence_deg':angle,'tentative_gamma_hkl':hkl,'status':'fit_failed','error':str(ex)})
pd.DataFrame(metas).to_csv(out/'scan_metadata.csv',index=False);pd.DataFrame(fits).to_csv(out/'peak_fits.csv',index=False);pd.DataFrame(candidates).to_csv(out/'candidate_peaks.csv',index=False)
with pd.ExcelWriter(out/'XRD_processed.xlsx') as w:
 pd.DataFrame(metas).to_excel(w,sheet_name='scan_metadata',index=False);pd.DataFrame(fits).to_excel(w,sheet_name='peak_fits',index=False);pd.DataFrame(candidates).to_excel(w,sheet_name='candidates',index=False)
 for s in sets:s[-1].to_excel(w,sheet_name=s[0],index=False)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(3,1,figsize=(10,9),sharex=True)
for a,s in zip(ax,sets):
 name,ang,x,y,bg,net,sm,norm,_=s;a.plot(x,y,color='0.6',lw=.55,label='Raw TXT export');a.plot(x,bg,color='C3',lw=1.5,label='Estimated background');a.plot(x,sm+bg,color='C0',lw=1,label='SG display (21 points)');a.set_ylabel('Export intensity');a.set_title(f'{name}: incidence {ang:g} deg');a.legend(fontsize=8)
ax[-1].set_xlabel('2 theta (deg)');fig.tight_layout();fig.savefig(out/'raw_and_background.png',dpi=220);fig.savefig(out/'raw_and_background.pdf');plt.close(fig)
fig,axs=plt.subplots(2,1,figsize=(10,8),sharex=True)
for name,ang,x,y,bg,net,sm,norm,_ in sets:
 axs[0].plot(x,sm,label=f'{ang:g} deg');axs[1].plot(x,norm,label=f'{ang:g} deg')
for a in axs:
 a.axhline(0,color='0.7',lw=.5);a.legend();a.set_xlim(10,90)
axs[0].set_ylabel('Background-subtracted intensity');axs[1].set_ylabel('Within-scan maximum = 100');axs[1].set_xlabel('2 theta (deg)');axs[0].set_title('IN718 GI-XRD: exploratory scans (SG smoothing for display)')
fig.tight_layout();fig.savefig(out/'comparison.png',dpi=220);fig.savefig(out/'comparison.pdf');plt.close(fig)
fig,axs=plt.subplots(3,3,figsize=(12,9))
for j,(name,hkl,x,y,pred,ok) in enumerate(curves):
 a=axs.flat[j];a.plot(x,y,'.',ms=2,label='Raw');a.plot(x,pred,lw=1.5,label='Single pseudo-Voigt');a.set_title(f'{name}, FCC ({hkl}) region\n'+('resolved candidate' if ok else 'weak / unstable'));a.set_xlabel('2 theta (deg)');a.set_ylabel('Export intensity')
fig.tight_layout();fig.savefig(out/'peak_fit_diagnostics.png',dpi=200);fig.savefig(out/'peak_fit_diagnostics.pdf');plt.close(fig)
print(pd.DataFrame(fits)[['file','tentative_gamma_hkl','fit_2theta_deg','status']].to_string(index=False));print('screening candidates:',len(candidates))
