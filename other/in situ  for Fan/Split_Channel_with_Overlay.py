from ij import IJ
from ij.plugin import ChannelSplitter as rawSplitter

class ChannelSplitter:
	def __init__(self):
		self.img=IJ.getImage()
		self.overlay=self.img.getOverlay()
		self.rois=self.overlay.toArray()
	def getRoiInfo(self):
		pos=[roi.getPosition() for roi in self.rois]
		nm=[roi.getName() for roi in self.rois]
		strokeColor=[roi.getStrokeColor() for roi in self.rois]
		strokeWidth=[int(roi.getStrokeWidth()) for roi in self.rois]
		fillColor=[roi.getFillColor() for roi in self.rois]
		return({'Position':pos,'Name':nm,'StrokeColor':strokeColor,
				'StrokeWidth':strokeWidth,'FillColor':fillColor})
	def split(self):
		spliter=rawSplitter()
		imgcopy=self.img.clone()
		info=self.getRoiInfo()
		roiPos=info['Position']
		imgs=spliter.split(imgcopy)
		for im in imgs:
			imc=self.getSplittedImageC(im)
			matchRoi=[roiInd for roiInd,roiP in enumerate(roiPos) if roiP==imc][0]
			matchRoiObj=self.rois[matchRoi]
			matchRoiObj.setPosition(0)
			overlay=im.getOverlay()
			im.setOverlay(matchRoiObj,info['StrokeColor'][matchRoi],
						  info['StrokeWidth'][matchRoi],info['FillColor'][matchRoi])
			im.show()
	def getSplittedImageC(self,img):
		title=img.getShortTitle()
		c=title.split('-')[0]
		return(int(c.lstrip('C')))
obj=ChannelSplitter()
obj.split()