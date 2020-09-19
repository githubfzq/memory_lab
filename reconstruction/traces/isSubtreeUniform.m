% detect whether all subtrees from soma point is uniform. eg: if some points within a subtrees from 
% soma is labeled with "apical" but the other is labeled with "basal", this function will
% return false, otherwise will return true.
% format: result=isSubtreeUniform(tree,[plot])
% plot: true or false, specific whether plot each subtree in different
% color.
% result:: logical ture(1) or false(0)
% by FanZuquan on Dec,15,2019

function res=isSubtreeUniform(intr,varargin)
    somaRInd=find(ismember(intr.rnames,'1')); % tree.rnames(somaRInd)='1'
    somaInd=find(intr.R==somaRInd);
    ipar=ipar_tree(intr);
    somaChild=setdiff(ipar(ismember(ipar(:,2),somaInd),1),somaInd);
    subTrees=zeros([size(ipar,1),length(somaChild)]);
    res=true;
    for m=1:length(somaChild)
        subTrees(:,m)=sub_tree(intr,somaChild(m));
        if length(unique(intr.R(logical(subTrees(:,m)))))>1
            res=false;
        end
    end
    if nargin==2 && varargin{1}==true
        color=[1,0,0;0,1,0;0,0,1;1,0,1;1,1,0;0,1,1;0.5,0.5,0;0.5,0,0.5]; % at most 8 subtree
        for n=1:length(somaChild)
            plot_tree(intr,color(n,:),[],find(subTrees(:,n)),[],'-2l');
        end
    end
end