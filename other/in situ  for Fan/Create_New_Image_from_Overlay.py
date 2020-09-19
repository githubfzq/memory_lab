from ij import IJ
from ij.gui import PointRoi,ShapeRoi,Line,Overlay
from ij.plugin import Duplicator
from ij.plugin.filter import Filler

class OverlayCropper:
	def __init__(self):
		self.imp=IJ.getImage()
		self.rois=self.getOverlay()
		self.getOutlineRoi()
	def getOverlay(self,imp=None):
		if imp is None:
			imp=self.imp
		self.overlay=imp.getOverlay()
		return(self.overlay.toArray())
	def getOutlineRoi(self):
		result=[(ind,roi) for ind,roi in enumerate(self.rois) 
				if not(isinstance(roi,PointRoi)) and not(isinstance(roi,ShapeRoi)) and not(isinstance(roi,Line))]
		self.outlineRoiInd,self.outlineRoi=result[0]
	def crop(self):
		self.imp.setRoi(self.outlineRoi)
		duplicator=Duplicator()
		self.imp.setHideOverlay(True)
		self.impCrop=duplicator.run(self.imp)
		self.impCrop.show()
		self.imp.setHideOverlay(False)
		
	def clearOutside(self,imp=None):
		imp=IJ.getImage() if (imp is None) else self.impCrop
		ip=imp.getProcessor()
		self.translateRoi()
		outlineRoiNew=self.newRois[self.outlineRoiInd]
		self.impCrop.setRoi(outlineRoiNew)
		sliceN=self.impCrop.getStackSize()
		for i in range(1,sliceN+1):
			imp.setSlice(i)
			ip.fillOutside(outlineRoiNew)
		imp.setSlice(1)
		
	def translateRoi(self):
		bound=self.outlineRoi.getBounds()
		dx,dy=bound.x,bound.y
		self.newRois=[]
		for roi in self.rois:
			rect=roi.getFloatBounds()
			roiNew=roi.clone()
			roiNew.setLocation(rect.getX()-dx,rect.getY()-dy)
			self.newRois.append(roiNew)
	def restoreRoi(self):
		overlay=Overlay()
		rois=[r for ind,r in enumerate(self.newRois) if ind!=self.outlineRoiInd]
		for roi in rois:
			overlay.add(roi)
		self.impCrop.setOverlay(overlay)
cropper=OverlayCropper()
cropper.crop()
cropper.clearOutside()
cropper.restoreRoi()