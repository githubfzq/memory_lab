# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'data/patch/analysis/20190928'))
	print(os.getcwd())
except:
	pass
# %%
from IPython import get_ipython

# %% [markdown]
# # Patch clamp analysis
# %% [markdown]
# ## sEPSC demo trace

# %%
import pyabf


# %%
import os.path
import os


# %%
def getAllFiles(rootDir):
    FilePath=[]
    Files=[]
    for (root, sub, file) in os.walk(rootDir):
        if not(sub) or all([s.startswith('.') for s in sub]):
            FilePath+=[os.path.join(root, f) for f in file]
            Files+=[f.split('.')[0] for f in file]
    return(FilePath,Files)


# %%
mFiles,_=getAllFiles("../../mini EPSC/")

# %% [markdown]
# ### representative trace

# %%
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes


# %%
get_ipython().run_line_magic('matplotlib', 'inline')


# %%
mDemo=pyabf.ABF(mFiles[0])

plt.figure(figsize=(10,3))
plt.plot(mDemo.sweepX, mDemo.sweepY, color='C0', alpha=.8)
plt.ylabel(mDemo.sweepLabelY)
plt.xlabel(mDemo.sweepLabelX)
plt.axis('off')

ax=plt.gca()
axin=zoomed_inset_axes(ax, 10, 4, axes_kwargs={'xlabel':'10 s','ylabel':'10 pA','xticks':[],
                                              'yticks':[]})
axin.spines['top'].set_visible(False)
axin.spines['right'].set_visible(False)

plt.savefig('sEPSC_demo_trace',dpi=600)

# %% [markdown]
# ### average trace

# %%
import pandas as pd
import numpy as np


# %%
mDemoTrace=pd.read_csv('../../process/mEPSC_trace/19315000.txt',
                       skiprows=6,sep='\t',header=None)


# %%
plt.figure(figsize=(5,5))
plt.plot(mDemoTrace[0],mDemoTrace[1], 'c-', linewidth=3, alpha=.8)
plt.plot(mDemoTrace[0],mDemoTrace.iloc[:,2:], color='grey' ,alpha=.03)

plt.axis('off')
plt.ylim(bottom=-20)

ax=plt.gca()
axin=zoomed_inset_axes(ax,10,4,axes_kwargs={'xlabel':'10 ms','ylabel':'5 pA',
                                            'xticks':[],'yticks':[],'ylim':(0,.5)})
axin.spines['top'].set_visible(False)
axin.spines['right'].set_visible(False)

plt.savefig('sEPSC_average',dpi=600)

# %% [markdown]
# ## cumulative distribution

# %%
itiFiles,_=getAllFiles('../../process/mEPSC_interEvent/')
ampFiles,_=getAllFiles('../../process/mEPSC_amplitude_data/')


# %%
itiDemoFile=itiFiles[0]
ampDemoFile=ampFiles[0]


# %%
itiDemoData=pd.read_csv(itiDemoFile,skiprows=3,sep='\t')
plt.plot(itiDemoData['Inter-event Interval (msec)'],itiDemoData['Cumulative Fraction'])
plt.xlabel('Inter-event Interval (msec)')
plt.ylabel('Cumulative Fraction')
plt.savefig('cumulative_sEPSP_ITI',dpi=600)


# %%
ampDemoData=pd.read_csv(ampDemoFile,skiprows=3,sep='\t')
plt.plot(ampDemoData['Amplitude (pA)'],ampDemoData['Cumulative Fraction'])
plt.xlabel('Amplitude (pA)')
plt.ylabel('Cumulative Fraction')
plt.savefig('cumulative_sEPSP_amplitute',dpi=600)

# %% [markdown]
# ## AP demo trace
# %% [markdown]
# ### raw trace

# %%
apFiles=[]
for root,subDir,file in os.walk('../../AP/'):
    if not(subDir):
        apFiles+=[os.path.join(root, f) for f in file]


# %%
apDemoFile='../../AP/20190408/cell 2/19408000.abf'
apDemoAbf=pyabf.ABF(apDemoFile)


# %%
import pyabf.plot


# %%
fig = plt.figure(figsize=(10,5))
ax1=fig.add_subplot(211)
ax2=fig.add_subplot(212)
for sweep in apDemoAbf.sweepList:
    apDemoAbf.setSweep(sweep)
    ax1.plot(apDemoAbf.sweepX, apDemoAbf.sweepY, 'C0')
    ax2.plot(apDemoAbf.sweepX, apDemoAbf.sweepC, 'C1')
ax1.axis('off')
ax2.axis('off')

plt.savefig('AP_demo',dpi=600)

# %% [markdown]
# ### AP distribution

# %%
fitFiles,_=getAllFiles('../../process/AP_fit/')

demofitFile='../../process/AP_fit/demo.csv'
fitFiles.remove(demofitFile)

fitDemoFile=fitFiles[0]


# %%
def readTitle(demoFile):
    with open(demoFile,'r') as f:
        title=f.readline().split(',')
    for ind,txt in enumerate(title):
        if txt=='' or txt=='\n':
            title[ind]='col_'+str(ind+1)
    return(title)


# %%
title=readTitle(demofitFile)
def readFitFile(file, head):
    fitDemoData=pd.read_csv(file,sep='\t',names=head)
    new=fitDemoData['Identifier'].str.extract('(?P<filename>.*abf) G(?P<sweep>\d+) T (?P<AP_ID>\d+) AP')
    rlt=pd.concat([new,fitDemoData.drop('Identifier',axis=1)],axis=1)
    rlt[['sweep','AP_ID']]=rlt[['sweep','AP_ID']].astype('int64')
    return(rlt)
fitDemoData=readFitFile(fitDemoFile,title)

# %% [markdown]
# ### AP frequency

# %%
apFreqRoot='../../process/AP_freq_data/'
apFreqFiles,_=getAllFiles(apFreqRoot)
apFreqDemoFile=apFreqFiles[0]


# %%
def getCellID(filePath):
    return(os.path.split(filePath))[-1].split('.')[0]


# %%
def readOneAPfreqData(filepath):
    dt=pd.read_csv(filepath,sep='\t',skiprows=1)
    dt.dropna(how='all',inplace=True)
    dt.rename({'Unnamed: 0':'sweep'},axis=1,inplace=True)
    dt.sweep=dt.sweep.str.extract('Group # (\d+)').astype('int64')
    dt.n=dt.n.astype('int64')
    dt['CellID']=getCellID(filepath)
    dt=dt[np.roll(dt.columns,1)]
    return(dt)


# %%
apFreqDemoData=readOneAPfreqData(apFreqDemoFile)


# %%
fig=apFreqDemoData.plot('sweep','n',kind='bar',legend=False)
plt.ylabel('AP number')
plt.savefig('AP_number',dpi=600)


# %%
apAllFreqData=pd.concat(map(readOneAPfreqData,apFreqFiles))

# %% [markdown]
# ### AP amplitude

# %%
apAmpRoot='../../process/AP_amplitude_data/'
apAmpFiles,_=getAllFiles(apAmpRoot)
apAmpDemoFile=apAmpFiles[0]
apAmpDemoData=readOneAPfreqData(apAmpDemoFile)


# %%
apAmpDemoData.plot('sweep','Average',kind='bar',legend=False)
plt.ylabel('Mean AP amplitude (mV)')
plt.savefig('AP_amplitute',dpi=600)

# %% [markdown]
# ## time constant (tau)

# %%
import re
def readAtfText(filename):
    with open(filename, 'r') as f:
        txt=f.readlines()
    title=re.findall(r'"([^"]*)"',txt[2])
    datastr=txt[3].split('\t')
    sz=(len(datastr)//len(title), len(title))
    dt=pd.DataFrame(np.array(datastr).reshape(sz), columns=title)
    return(dt)

# %% [markdown]
# Read tau results

# %%
tauPath='../../process/AP_tau/'
tauDirFiles=np.array(os.listdir(tauPath))
tauFileName=tauDirFiles[np.char.endswith(tauDirFiles,'.atf')][-1] # tau results are all combined
tauFile=os.path.join(tauPath,tauFileName)
tauData=readAtfText(tauFile)
tauData.insert(1,'cell_ID',tauData['File Name'].str.split('.').str[0])
tauData[['A','tau','C']]=tauData[['A','tau','C']].astype('float')


# %%
tauRes=tauData[['cell_ID','tau']].groupby('cell_ID').mean().reset_index()


# %%
tauRes['tau'].plot.hist(alpha=.7)
plt.xlabel('Time constant (ms)')
plt.ylabel('Number of neuron')
plt.savefig('tau',dpi=600)

# %% [markdown]
# ## filter outlier

# %%
from scipy.stats import variation
outlier=tauData[['cell_ID','tau','C']].groupby('cell_ID').agg({'tau':np.mean,'C':variation}).reset_index()
outlier['isOutlier']=(np.abs(outlier['C'])>0.2)|(outlier['tau']<=0)

# %% [markdown]
# ## I/V relationship
# %% [markdown]
# ### plot I-V traces

# %%
ivFileDemo='../../process/IV_data/19408008.atf'


# %%
import numpy as np


# %%
ivDemo=readAtfText(ivFileDemo)
ivDemo.iloc[:,[1,2,6,7,8,9]]=ivDemo.iloc[:,[1,2,6,7,8,9]].apply(pd.to_numeric)


# %%
def get_APCurrentStep(filename):
    apDemoAbf=pyabf.ABF(filename)
    cur_step=[]
    for sweep in apDemoAbf.sweepList:
        apDemoAbf.setSweep(sweep)
        cur_step.append(apDemoAbf.sweepEpochs.levels[2])
    return(cur_step)


# %%
plt.plot(get_APCurrentStep(apDemoFile)[:6],ivDemo['S1R1 Mean (mV)'],'-o')
plt.xlabel('I (nA)')
plt.ylabel('Vm (mV)')
plt.show()


# %%
apFilePath,apFiles=getAllFiles("../../AP/")
ivFiles,apCellIDs=getAllFiles('../../process/IV_data/')


# %%
def IVfileMatchABF(ivFile):
    return np.array(apFilePath)[np.array(getCellID(ivFile))==np.array(apFiles)][0]
def AbfMatchIVfile(abfFile):
    matched=np.array(getCellID(abfFile))
    return(np.array(ivFiles)[matched==np.array(apCellIDs)][0]
           if np.any(matched==np.array(apCellIDs)) else None)


# %%
def getOneIVdata(filePath):
    ivRead=readAtfText(filePath)
    apFile=IVfileMatchABF(filePath)
    sweepNum=ivRead.shape[0]
    curStep=get_APCurrentStep(apFile)[:sweepNum]
    dt=pd.DataFrame({'CellID':getCellID(filePath),'I':curStep,
                     'Vm':ivRead['S1R1 Mean (mV)']})
    dt.I=pd.to_numeric(dt.I)
    dt.Vm=pd.to_numeric(dt.Vm)
    return(dt)


# %%
getOneIVdata(ivFileDemo)


# %%
ivAllData=pd.concat(map(getOneIVdata, ivFiles),ignore_index=True)


# %%
for key,tb in ivAllData.groupby('CellID'):
    plt.plot(tb['I'],tb['Vm'],'-o',alpha=.5,color='C0')
plt.xlabel('I (nA)')
plt.ylabel('Vm (mV)')
plt.savefig('IV_trace',dpi=600)

# %% [markdown]
# ### Compute membrane resistance

# %%
ivDemoDt=getOneIVdata(ivFileDemo)


# %%
from sklearn.linear_model import LinearRegression


# %%
x=ivDemoDt[['I']]
y=ivDemoDt['Vm']
reg=LinearRegression().fit(x,y)


# %%
reg.score(x,y)


# %%
reg.coef_


# %%
reg.intercept_


# %%
reg.predict(x)


# %%
plt.plot(x.values,y,'-o',x.values,reg.predict(x),'-')
plt.xlabel('I (nA)')
plt.ylabel('Vm (mV)')
plt.text(-0.15,0,r'$Rm=%.4f M\Omega$' % reg.coef_)
plt.text(-0.15,-5,r'$R^2=%.4f$' % reg.score(x,y))
plt.savefig('IV_fit_demo',dpi=600)


# %%
def computeRm(IVdata):
    x=IVdata[['I']]
    y=IVdata['Vm']
    reg=LinearRegression().fit(x,y)
    score=reg.score(x,y)
    coef=reg.coef_[0]
    return {'Rm':coef,'R_score':score}


# %%
computeRm(ivDemoDt)


# %%
res=ivAllData.groupby('CellID').apply(computeRm)
RmRes=pd.DataFrame(res.to_list(),index=res.index).reset_index()


# %%
RmRes[RmRes.R_score>0.9].Rm.plot.hist(alpha=.75)
plt.xlabel(r'$Rm\ (M\Omega)$')
plt.ylabel('Number of neuron')
plt.savefig('Rm_hist',dpi=600)

# %% [markdown]
# ## relationship of intrinsic membrane properties

# %%


# %% [markdown]
# # Morphological analysis

# %%
import metakernel


# %%
metakernel.register_ipython_magics()
get_ipython().run_line_magic('kernel', 'matlab_kernel.kernel MatlabKernel')


# %%
get_ipython().run_cell_magic('kx', '', "addpath('../../../reconstruction/traces/')\naddpath('../../../reconstruction/functions/')\naddpath(genpath('/usr/local/matlab2017b/toolbox/shared/TREES1.15/'),'-end')")


# %%
get_ipython().run_cell_magic('kx', '', 'load morpho_data.mat')


# %%
get_ipython().run_cell_magic('kx', '', "plot_multi_tree(tr,treeColor,[3 floor(size(tr,2)/3+1)]);\ndraw_scale_bar(100,'xy');")

# %% [markdown]
# Put apical dendrites upside:

# %%
get_ipython().run_cell_magic('kx', '', "for n=1:size(tr,2)\n    trUpsided(n)=apical_upside(tr(n));\nend\nplot_multi_tree(trUpsided,treeColor,[3 floor(size(trUpsided,2)/3+1)]);\ndraw_scale_bar(100,'xy');\nsaveas(gcf,'neuron_traces','png');")

# %% [markdown]
# ## Sholl analysis
# %% [markdown]
# Analyse by Matlab:

# %%
get_ipython().run_cell_magic('kx', '', "sholl_res=arrayfun(@sholl_parts,trUpsided,ones(size(trUpsided)),'UniformOutput',false);\nsholl_tab=[];\nfor n=1:length(sholl_res)\n    sholl_tab=[sholl_tab;sholl_res{n}];\nend")


# %%
get_ipython().run_cell_magic('kx', '', "writetable(sholl_tab,'sholl_tmp.csv');")


# %%
shollParts=pd.read_csv('sholl_tmp.csv')


# %%
from itertools import product
def zero_padding(tab,*padding_rows):
    keep_rows=tab.columns.drop(list(padding_rows))
    base=pd.DataFrame(product(*[set(tab[r]) for r in keep_rows]),columns=keep_rows)
    res=pd.merge(base,tab,'outer',on=keep_rows.tolist()).fillna(0)
    return(res)


# %%
from scipy.stats import sem
shollPartsCompute=(zero_padding(shollParts,'intersections').groupby(['label','radius']).
                   agg([np.mean,sem]))
shollPartsCompute.columns=["_".join(x) for x in shollPartsCompute.columns.ravel()]
shollPartsCompute.reset_index(inplace=True)


# %%
shollParts2=shollPartsCompute.pivot(index='radius',columns='label',
                                    values=['intersections_mean','intersections_sem'])
plt.plot(shollParts2.index,shollParts2.intersections_mean.apical,
         shollParts2.index,shollParts2.intersections_mean.basal,label='')
ax1=plt.fill_between(shollParts2.index,shollParts2.loc[:,("intersections_mean","apical")]+shollParts2.loc[:,("intersections_sem","apical")],
                     shollParts2.loc[:,("intersections_mean","apical")]-shollParts2.loc[:,("intersections_sem","apical")],
                     alpha=.6,label='apical')
ax2=plt.fill_between(shollParts2.index,shollParts2.loc[:,("intersections_mean","basal")]+shollParts2.loc[:,("intersections_sem","basal")],
                     shollParts2.loc[:,("intersections_mean","basal")]-shollParts2.loc[:,("intersections_sem","basal")],
                     alpha=.6,label='basal')
plt.ylabel('Sholl intersections')
plt.xlabel('$Radius\ (\mu m)$')
plt.legend()
plt.savefig('sholl_part',dpi=600)

# %% [markdown]
# Analyse by morphological data exported by Imaris:

# %%
allShollStatFile,_=getAllFiles('../../../reconstruction/stat/')
statFiles,somaStatFiles=[],[]
for f in allShollStatFile:
    if 'Detailed.csv' in f: 
#  filter Detailed labeld files
        if 'soma' in f:
            somaStatFiles.append(f)
        else:
            statFiles.append(f)


# %%
def readOneMorphoData(file):
    tb=pd.read_csv(file,skiprows=2,header=1)
    tb['neuron_ID']=os.path.split(file)[-1].split('_Detailed.')[0]
    return(tb[np.roll(tb.columns,1)])


# %%
morphoData=pd.concat(map(readOneMorphoData,statFiles),sort=True)


# %%
shollData=morphoData[morphoData.Variable=='Filament No. Sholl Intersections'].loc[:,['neuron_ID','Radius','Value']]


# %%
shollPlotData=zero_padding(shollData,'Value').groupby('Radius').agg([np.mean,sem]).reset_index()
plt.plot(shollPlotData["Radius"],shollPlotData[('Value','mean')])
plt.fill_between(shollPlotData["Radius"],
                 shollPlotData[('Value','mean')]+shollPlotData[('Value','sem')],
                 shollPlotData[('Value','mean')]-shollPlotData[('Value','sem')],
                 alpha=.6)
plt.ylabel('Sholl intersections')
plt.xlabel('$Radius\ (\mu m)$')
plt.savefig('sholl_both',dpi=600)

# %% [markdown]
# ## angle distribution
# %% [markdown]
# ## branch order

# %%
depthData=(morphoData[morphoData.Variable=='Dendrite Branch Depth']
           .loc[:,['neuron_ID','Depth','Level','Value']]
           .groupby(['neuron_ID','Depth']).count().reset_index()
           .drop('Level',axis=1).rename({'Value':'counts'},axis=1))
depthData.Depth=depthData.Depth.astype('int64')


# %%
depthPlotData=depthData.groupby('Depth').agg([np.mean,sem]).reset_index()
plt.bar(depthPlotData['Depth']+1,depthPlotData[('counts','mean')],alpha=.6)
plt.errorbar(depthPlotData['Depth']+1,depthPlotData[('counts','mean')],
             depthPlotData[('counts','sem')],linestyle='')
plt.ylabel('Number of filaments')
plt.xlabel('Branch order')
plt.savefig('branch_depth',dpi=600)

# %% [markdown]
# # Cluster Analysis
# %% [markdown]
# ## Electrophysiological Parameters

# %%
def getElectroPara(file):
    name=getCellID(file)
    abf=pyabf.ABF(file)
    rmp=abf.sweepY[abf.sweepEpochs.p1s[0]:abf.sweepEpochs.p2s[0]].mean()
    ivFile=AbfMatchIVfile(file)
    if ivFile:
        ivData=getOneIVdata(ivFile)
        RmRes=computeRm(ivData)
        return({'ID':name,'RMP':rmp,'Rm':RmRes['Rm']})
    else:
        return None

# %% [markdown]
# Get all electrophysiological data

# %%
electroParaData=pd.Series(apFilePath).map(getElectroPara)
electroParaData=electroParaData[np.not_equal(electroParaData,None)]
electroParaData=pd.DataFrame(electroParaData.to_list())
electroParaData=pd.merge(electroParaData,outlier.drop('C',axis=1),'outer',left_on='ID',right_on='cell_ID').drop('ID',axis=1)

# %% [markdown]
# ## analysis
# %% [markdown]
# ### K-Means

# %%
from sklearn.cluster import KMeans


# %%
kmeans=KMeans(n_clusters=2)


# %%
CAdata=electroParaData[np.logical_not(electroParaData['isOutlier'])].drop(['cell_ID','isOutlier'],axis=1)
kmeans.fit(CAdata)


# %%
CAdata['Cluster']=kmeans.fit_predict(CAdata)


# %%
def plotClusterScatter(ax,column1,column2):
    ax.scatter(CAdata.iloc[CAdata.Cluster.values==0,column1],CAdata.iloc[CAdata.Cluster.values==0,column2],color='C1',label='cluster 1')
    ax.scatter(CAdata.iloc[CAdata.Cluster.values==1,column1],CAdata.iloc[CAdata.Cluster.values==1,column2],color='C2',label='cluster 2')
    curColumn=CAdata.columns[[column1,column2]]
    ax.scatter(electroParaData.loc[electroParaData.isOutlier.tolist(),curColumn[0]],
                electroParaData.loc[electroParaData.isOutlier.tolist(),curColumn[1]],marker='x',color='C0',label='outlier')
    ax.legend()
#     plt.show()


# %%
fig,axs=plt.subplots(3,1)
fig.set_size_inches((6,12))
plotClusterScatter(axs[0],0,1)
axs[0].set_xlabel(electroParaData.columns[0]+' (mV)')
axs[0].set_ylabel(electroParaData.columns[1]+'$\ (M\Omega)$')
plotClusterScatter(axs[1],0,2)
axs[1].set_xlabel(CAdata.columns[0]+' (mV)')
axs[1].set_ylabel(CAdata.columns[2]+' (ms)')
plotClusterScatter(axs[2],1,2)
axs[2].set_xlabel(CAdata.columns[1]+'$\ (M\Omega)$')
axs[2].set_ylabel(CAdata.columns[2]+' (ms)')
plt.savefig('cluster_scatter',dpi=600)


# %%
def plotClusterHist(ax,col):
    range_=CAdata.iloc[:,col].min(),CAdata.iloc[:,col].max()
    ax.hist(CAdata.iloc[CAdata.Cluster.values==0,col],alpha=.6,bins=20,
            label='cluster 1',range=range_)
    ax.hist(CAdata.iloc[CAdata.Cluster.values==1,col],alpha=.6,bins=20,
            label='cluster 2',range=range_)
    ax.legend()


# %%
fig,axs=plt.subplots(3,1)
fig.set_size_inches((6,12))
plotClusterHist(axs[0],0)
axs[0].set_xlabel('RMP (mV)')
plotClusterHist(axs[1],1)
axs[1].set_xlabel('Rm '+'$(M\Omega)$')
axs[1].set_ylabel('Number of neurons')
plotClusterHist(axs[2],2)
axs[2].set_xlabel('time constant (ms)')
plt.savefig('cluster_histogram',dpi=600)


# %%
moreInfo=pd.read_excel('../../../reconstruction/stat/neuron information.xlsx')
moreInfo.patch_date=moreInfo.patch_date.astype('str')


# %%
apFileArr=pd.Series(apFilePath)
apFileArr=apFileArr[apFileArr.str.match(r'.*\d{8}\.abf')]
apFileInfo=apFileArr.str.extract(r'.*/(?P<patch_date>\d{8})/cell\s?(?P<cellN>\d)/(?P<cell_ID>\d{8})\.abf')
apFileInfo.cellN='cell'+apFileInfo.cellN


# %%
linkInfo=apFileInfo.merge(moreInfo.rename(columns={'patch_ID':'cellN'}),
                          on=['patch_date','cellN'])


# %%


# %% [markdown]
# ### Hierarchical Cluster

# %%
from scipy.cluster.hierarchy import dendrogram,linkage


# %%
HierData=electroParaData[np.logical_not(electroParaData['isOutlier'])].drop(['cell_ID','isOutlier'],axis=1)


# %%
z=linkage(HierData)
dendroData=dendrogram(z)
plt.savefig('dendrogram',dpi=600)

