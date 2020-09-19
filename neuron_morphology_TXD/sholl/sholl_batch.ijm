#@ File (label = "Input directory", style = "directory") input
#@ String (label = "File suffix", value = ".swc") suffix

// See also Process_Folder.py for a version of this code
// in the Python scripting language.

processFolder(input);

// function to scan folders/subfolders/files to find files with correct suffix
function processFolder(input) {
	list = getFileList(input);
	list = Array.sort(list);
	for (i = 0; i < list.length; i++) {
		if(File.isDirectory(input + File.separator + list[i]))
			processFolder(input + File.separator + list[i]);
		if(endsWith(list[i], suffix))
			// Do the processing here by adding your own code.
			run("Sholl Analysis (Tracings)...", "traces/(e)swc=["+input+File.separator+list[i]+"] image=[] load center=[Start of main path] radius=1 enclosing=1 _primary=[] infer linear polynomial=[Best fitting degree] linear-norm semi-log log-log normalizer=Area/Volume directory=[]");
			// Leave the print statements until things work, then remove them.
			print("Processing: " + input + File.separator + list[i]);
	}
}

function processFile(input, file) {
	
}
