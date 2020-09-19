from ij import WindowManager
from ij.gui import ShapeRoi,PointRoi
from itertools import chain,combinations
from ij.measure import ResultsTable

roiManager=WindowManager.getFrame("ROI Manager")
imgName=WindowManager.getCurrentImage().getTitle().split('.')[0]
curSelInd=roiManager.getSelectedIndex()
curSelName=roiManager.getName(curSelInd)
N=roiManager.getCount()
otherInd=range(curSelInd)+range(curSelInd+1,N)
otherName=[roiManager.getName(i) for i in otherInd]
otherName.sort()
def nameCombination(ls, pre):
	ls=[pre]+ls
	res=chain(*map(lambda x: combinations(ls,x), range(0, len(ls)+1)))
	res=list(res)
	return(['_'.join(s) for s in res if pre in s])
classNames=nameCombination(otherName,curSelName)

roiComp=roiManager.getRoi(curSelInd)
rois=roiComp.getRois()
rois=map(ShapeRoi,rois)

otherComp={roiManager.getName(i):roiManager.getRoi(i) for i in otherInd}
def isIntersect(roi1,roi2):
	"Roi1 must be class ShapeRoi, but Roi2 could be class ShapeRoi or PointRoi."
	if(isinstance(roi2,PointRoi)):
		resRoi=roi2.containedPoints(roi1)
		if resRoi!=None:
			Icounter=resRoi.getCounter()
			N=resRoi.getCount(Icounter)
		else:
			N=0
	else:
		resRoi=roi1.and(roi2)
		resRois=resRoi.getRois()
		N=len(resRois)
	return(N)
def isIntersectComps(roi,roiRefs):
	res={nm:isIntersect(roi,roiRef) for nm,roiRef in otherComp.items()}
	return(res)
def getRoiClass(roi):
	cls_tmp=isIntersectComps(roi,otherComp)
	for cls in classNames:
		posNames,negNames=[],[]
		for nm,isCls in cls_tmp.items():
			if isCls>0:
				posNames.append(nm)
			else:
				negNames.append(nm)
		if all([p in cls for p in posNames]) and all([n not in cls for n in negNames]):
			cls_tmp['Class']=cls
			return(cls_tmp)
def showRoiSummary(table):
	res=ResultsTable()
	for i,val in table.items():
		res.setValue('id',i,i+1)
		valInd=val.keys()
		valInd.remove('Class')
		for ind in valInd:
			res.setValue(ind,i,val[ind])
		res.setValue('Class',i,val['Class'])
	res.show('[ROI Summary]'+imgName)
res={i:getRoiClass(roi) for i,roi in enumerate(rois)}
showRoiSummary(res)
def showClassSummary(table):
	resClass=dict()
	resTable=ResultsTable()
	for i,d in table.items():
		curClass=d['Class']
		resClass.setdefault(curClass,[]).append(i)
	resClassName=sorted(resClass.keys())
	for i,clsName in enumerate(resClassName):
		resTable.setValue('Class',i,clsName)
		resTable.setValue('Counts',i,len(resClass[clsName]))
	resTable.show('[Class Summary]'+imgName)
showClassSummary(res)