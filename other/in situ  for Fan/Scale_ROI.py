from ij import WindowManager,IJ
from ij.gui import PointRoi,PolygonRoi,Roi,GenericDialog,ShapeRoi
from java.awt.event import TextListener
from ij.plugin.frame import RoiManager

imp=IJ.getImage()

class ScaleRoiDialog(TextListener):
	xScale,yScale=1.0,1.0
	def __init__(self):
		self.impXpixel=imp.getWidth()
		self.impYpixel=imp.getHeight()
		self.toXpixel,self.toYpixel=self.impXpixel,self.impYpixel
	def inputScale(self):
		gd=GenericDialog("Scale ROI")
		gd.addStringField("X Scale:",str(self.xScale))
		gd.addStringField("Y Scale:",str(self.yScale))
		gd.addStringField("Scale X to pixel:",str(self.impXpixel))
		gd.addStringField("Scale Y to pixel:",str(self.impYpixel))
		fields=gd.getStringFields()
		self.XscaleField,self.YscaleField=fields[0],fields[1]
		self.toXfield,self.toYfield=fields[2],fields[3]
		self.XscaleField.addTextListener(self)
		self.YscaleField.addTextListener(self)
		self.toXfield.addTextListener(self)
		self.toYfield.addTextListener(self)
		gd.showDialog()
		if gd.wasOKed():
			return(self.xScale,self.yScale)
		else:
			return(1.0,1.0)
	def textValueChanged(self,e):
		source=e.getSource()
		if source==self.XscaleField:
			self.xScale=float(source.getText())
			self.yScale=self.xScale
			self.YscaleField.setText(str(self.yScale))
			self.toXfield.setText(str(self.xScale*self.impXpixel))
			self.toYfield.setText(str(self.yScale*self.impYpixel))
		elif source==self.YscaleField:
			self.yScale=float(source.getText())
			self.toYfield.setText(str(self.yScale*self.impYpixel))
		elif source==self.toXfield:
			self.toXpixel=float(source.getText())
			self.xScale=self.toXpixel/self.impXpixel
		elif source==self.toYfield:
			self.toYpixel=float(source.getText())
			self.yScale=self.toYpixel/self.impYpixel
dg=ScaleRoiDialog()
scale=dg.inputScale()
roiMng=WindowManager.getWindow('ROI Manager')

def scaleTypeROI(roi):
	if isinstance(roi,PointRoi):
		p=roi.getFloatPolygon()
		x,y=list(p.xpoints),list(p.ypoints)
		xNew,yNew=map(lambda c:c*scale[0],x),map(lambda c:c*scale[1],y)
		roiNew=PointRoi(xNew,yNew)
	elif isinstance(roi,ShapeRoi):
		roiSels=roi.getRois()
		roiNews=map(scaleTypeROI,roiSels)
		roiNew=0
		for roi in roiNews:
			if roiNew==0:
				roiNew=ShapeRoi(roi)
			else:
				roiNew=roiNew.or(ShapeRoi(roi))
	else:
		tp=roi.getType()
		if tp!=0:
			p=roi.getPolygon()
			x,y=list(p.xpoints),list(p.ypoints)
			posNew=map(lambda pos:(pos[0]*scale[0],pos[1]*scale[1]),zip(x,y))
			xNew,yNew=zip(*posNew)
			roiNew=PolygonRoi(xNew,yNew,tp)
		else:
			x,y,w,h=roi.getXBase(),roi.getYBase(),roi.getFloatWidth(),roi.getFloatHeight()
			xNew,yNew,wNew,hNew=x*scale[0],y*scale[1],w*scale[0],h*scale[1]
			roiNew=Roi(xNew,yNew,wNew,hNew)
	return(roiNew)
def doScaleROI():
	ind=roiMng.getSelectedIndex()
	roiSel=roiMng.getRoi(ind)
	roiNm=roiMng.getName(ind)
	imp=IJ.getImage()
	roiNew=scaleTypeROI(roiSel)
	imp.deleteRoi()
	roiMng.setRoi(roiNew,ind)
	imp.setRoi(roiNew)
doScaleROI()