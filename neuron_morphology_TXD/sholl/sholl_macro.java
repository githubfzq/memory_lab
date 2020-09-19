import ij.*;
import ij.process.*;
import ij.gui.*;
import java.awt.*;
import ij.plugin.*;

public class sholl_macro implements PlugIn {

	public void run(String arg) {
		// Recording Sholl Analysis version 3.7.3
		// Visit https://imagej.net/Sholl_Analysis#Batch_Processing for scripting tips
		// Recording Sholl Analysis version 3.7.3
		// Visit https://imagej.net/Sholl_Analysis#Batch_Processing for scripting tips
		IJ.run("Sholl Analysis (Tracings)...", "traces/(e)swc=[J:/神经元追踪备份/trace/GFP+//(GFP+)slice_1_red neuron1.swc] image=[] load center=[Start of main path] radius=0 enclosing=1 #_primary=[] infer linear polynomial=[Best fitting degree] linear-norm semi-log log-log normalizer=Area/Volume directory=[]");
		  <<warning: the options string contains one or more non-ascii characters>>
	}

}
