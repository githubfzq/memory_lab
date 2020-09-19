javaaddpath("C:\Program Files\Bitplane\Imaris x64 9.0.1\XT\matlab\ImarisLib.jar");
vImarisLib=ImarisLib;
aImarisApplication=vImarisLib.GetApplication(0);
dtset=aImarisApplication.GetDataSet;
tran=dtset.GetExtendMaxZ;
tr=load_tree;
trNew=tran_tree(tr,[0,0,tran/2],'-s'); 
% Two tif image stack at z direction
