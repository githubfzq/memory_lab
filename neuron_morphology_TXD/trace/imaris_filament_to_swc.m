function varargout=imaris_filament_to_swc(filename)

% 格式：
% tree_loaded=imaris_filament_to_swc(filename)
%
% 在Imaris中选中重构好的Filament
%
% 输入参数：
% filename: 需要导出的swc格式文件名
% 
% 输出参数：
% tree_loaded: 加载到工作区的变量，用于tree分析
%
% Written by FanZuquan on July,16,2018
%
global trees

javaaddpath("C:\Program Files\Bitplane\Imaris x64 9.0.1\XT\matlab\ImarisLib.jar");
vImarisLib=ImarisLib;
aImarisApplication=vImarisLib.GetApplication(0);
vFactory=aImarisApplication.GetFactory;
vFila=vFactory.ToFilaments(aImarisApplication.GetSurpassSelection);
% vPosi=vFila.GetPositionsXYZ(0);
% vRadi=vFila.GetRadii(0);
% vEdge=vFila.GetEdges(0);vEdge=vEdge+1;
BeginVerInd=vFila.GetBeginningVertexIndex(0);
% vEdge(vEdge(:,2)==BeginVerInd,1)=-1;

vFilaList=vFila.GetFilamentsList(0); %list的mEdge/mPositionsXYZ/mType/mRadii有用
edg=[-2 0;vFilaList.mEdges]+1; %增加一行[-1 beginpoint=1]
swc_tr=[cast(edg(:,2),'single'),cast(vFilaList.mTypes,'single'),...
    vFilaList.mPositionsXYZ,vFilaList.mRadii,cast(edg(:,1),'single')];
swc_tr=sortrows(swc_tr);

fileID=fopen(filename,'w+');
fprintf(fileID,'%5d %1d %5.10f %5.10f %5.10f %5.10f %5d\n',swc_tr');
fclose(fileID);
old_tr=load_tree(filename);
[tree_loaded,~]=redirect_tree(old_tr,BeginVerInd+1); % set new root to BeginVerInd from Imaris
swc_tree(tree_loaded,filename);

if nargout==1
    varargout{1}=tree_loaded;
else
    trees {length (trees) + 1} =tree_loaded;
end
