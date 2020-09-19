% function repair_imaris_tif_voxelsize(direction,voxelsize)
% To resize the tif image opened in imaris surpass, in order to be a new
% size as x*y*z.
% direction::string: the direction to be set ('x','y' or 'z')
% voxelsize::double: the voxel size relavant to the direction, which equals
% to (ExtendMaxX/Y/Z)/(SizeX/Y/Z)

function repair_imaris_tif_voxelsize(direction,voxelsize)
javaaddpath("C:\Program Files\Bitplane\Imaris x64 9.0.1\XT\matlab\ImarisLib.jar");
vImarisLib=ImarisLib;
aImarisApplication=vImarisLib.GetApplication(0);
dtset=aImarisApplication.GetDataSet;
if direction=='z'
    dtset.SetExtendMaxZ(dtset.GetSizeZ*voxelsize);
elseif direction=='x'
    dtset.SetExtendMaxX(dtset.GetSizeX*voxelsize);
elseif direction=='y'
    dtset.SetExtendMaxY(dtset.GetSizeY*voxelsize);
end
