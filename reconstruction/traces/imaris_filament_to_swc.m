function imaris_filament_to_swc(filename)

% 格式：
% imaris_filament_to_swc(filename)
%
% 在Imaris中选中重构好的Filament
%
% 输入参数：
% filename: 需要导出的swc格式文件名
% 
%
% Written by FanZuquan on July,16,2018
% Edited at July,15,2019
% Modified at:
%   - September,4,2019:fix region of nodes.

javaaddpath('C:\Program Files\Bitplane\Imaris x64 9.0.1\XT\matlab\ImarisLib.jar');
vImarisLib=ImarisLib;
vImaris=vImarisLib.GetApplication(0);
if isempty(vImaris)
    fprintf('Could not connect to Imaris\n');
    return;
end
vFactory=vImaris.GetFactory;
vFilaments=vFactory.ToFilaments(vImaris.GetSurpassSelection);
if isempty(vFilaments)
    fprintf('Pick a filament first\n');
    return;
end

V=vImaris.GetDataSet;
if ~exist('filename')
    filename=fullfile(pwd,'export.swc');
end

head=0;
swcs=zeros(0,7);

vCount=vFilaments.GetNumberOfFilaments;
for vFilamentsIndex=0:vCount-1
    vFilamentsXYZ=vFilaments.GetPositionsXYZ(vFilamentsIndex);
    vFilamentsEdges=vFilaments.GetEdges(vFilamentsIndex);
    vFilamentsRadius=vFilaments.GetRadii(vFilamentsIndex);
    
    % get label of each edges
    segIds=vFilaments.GetEdgesSegmentId(vFilamentsIndex);
    segGroup=unique(segIds);[~,segGroupInd]=ismember(segIds,segGroup);
    segGroupLabelsTmp=arrayfun(@(seg) vFilaments.GetLabelsOfId(seg),segGroup,'UniformOutput',false);
    segGroupLabels=cellfun(@(obj) string(obj(1).mLabelValue),segGroupLabelsTmp);
    segLabels=segGroupLabels(segGroupInd);
    
    % get label of each point
    labelgroup=unique(segLabels);
    for lb=1:length(labelgroup)
        segLabelInd=find(segLabels==labelgroup(lb));
        segPoints=unique(vFilamentsEdges(segLabelInd,:));
        if labelgroup(lb)=="apical"
            vFilamentsTypes(segPoints+1,1)=4;
        elseif labelgroup(lb)=="basal"
            vFilamentsTypes(segPoints+1,1)=3;
        else
            vFilamentsTypes(segPoints+1,1)=0;
        end
        beginVertex=vFilaments.GetBeginningVertexIndex(vFilamentsIndex);
        vFilamentsTypes(beginVertex+1)=1;
        % 0 - undefined, 1 - soma, 3 - basal dendrite, 4 - apical dendrite
    end
    
    N=size(vFilamentsXYZ,1);
    G=zeros([N,N],'logical');
    visited=zeros([N,1],'logical');
    G(sub2ind(size(G),vFilamentsEdges(:,1)+1,vFilamentsEdges(:,2)+1))=1;
    G(sub2ind(size(G),vFilamentsEdges(:,2)+1,vFilamentsEdges(:,1)+1))=1;
    
    head=1;
    swc=zeros(N,7);
    visited(1,:)=1;
    queue=1;
    prevs=-1;
    while ~isempty(queue)
        cur=queue(end);queue(end)=[];
        prev=prevs(end);prevs(end)=[];
        swc(head,:)=[head,vFilamentsTypes(cur),vFilamentsXYZ(cur,:),vFilamentsRadius(cur,:),prev];
        for idx=find(G(cur,:))
            if ~visited(idx)
                visited(idx)=1;
                queue=[queue,idx];
                prevs=[prevs,head];
            end
        end
        head=head+1;
    end
    swcs=[swcs;swc];
end

fileID=fopen(filename,'w');
fprintf(fileID,'%d %d %f %f %f %f %d\n',swcs');
fclose(fileID);
fprintf('Export to %s completed.\n',filename);
end