from ij.plugin.frame import RoiManager
from ij import WindowManager
from ij.gui import PointRoi,ShapeRoi

roiManager=WindowManager.getFrame("ROI Manager")
roiSel=roiManager.getSelectedRoisAsArray()
if len(roiSel)==1:
	roiSel=roiSel[0].getRois()
	roiSel=map(ShapeRoi,roiSel) # convert roi composite to many roi
x,y=[],[]
for sel in roiSel:
	stat=sel.getStatistics()
	x.append(stat.xCentroid)
	y.append(stat.yCentroid)
roi_new=PointRoi(x,y)
roiManager.addRoi(roi_new)