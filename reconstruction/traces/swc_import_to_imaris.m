% swc_import_to_imaris()
% To import the tree of '*.swc' format to imaris surpass
% 
% format:
% swc_import_to_imaris([tr]);
% tr: a tree to import;
% -----
% Written by FanZuquan on July 18,2018
function swc_import_to_imaris(varargin)
javaaddpath("C:\Program Files\Bitplane\Imaris x64 9.0.1\XT\matlab\ImarisLib.jar");
vImarisLib=ImarisLib;
aImarisApplication=vImarisLib.GetApplication(0);
vFactory=aImarisApplication.GetFactory;

% compute Edge
if nargin==1
    tr=varargin{1};
else
    tr=load_tree;
end
edg=idpar_tree(tr);
edg=[edg,reshape(1:length(edg),length(edg),1)];
edg=edg(2:end,:)-1;

% create filament
newFila=vFactory.CreateFilaments;
newFila.AddFilament([tr.X,tr.Y,tr.Z],tr.D/2,zeros(size(tr.R)),edg,0);

selec=aImarisApplication.GetSurpassSelection;
if vFactory.IsDataContainer(selec)
    selec=vFactory.ToDataContainer(selec);
    selec.AddChild(newFila,-1);
else
    selec_par=vFactory.ToDataContainer(selec.GetParent);
    selec_par.AddChild(newFila,-1);
end
