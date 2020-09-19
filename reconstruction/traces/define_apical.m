% [apical_tree,basal_tree]=define_apical(intree)
% 
% apical_tree: apical dendrites part of intree;
% basal_tree: basal dendrites part of intree;

function [apical_tree,basal_tree]=define_apical(intr)
    rind=rindex_tree(intr);
    ipar=ipar_tree(intr);
    begins=ipar(ipar(:,2)==1,1);
    beginR=intr.R(begins);
    [~,apicalRind]=ismember('4',intr.rnames);
    [~,basalRind]=ismember('3',intr.rnames);
    apical_begin=begins(beginR==apicalRind);
    basal_begin=begins(beginR==basalRind);% '3':basal, '4':apical
    % create apical tree
    subind=[];
    for bg=1:length(basal_begin)
        [delInd,~]=sub_tree(intr,basal_begin(bg));
        subind=[subind;find(delInd)];
    end
    apical_tree=delete_tree(intr,subind);
    % create basal tree
    subind=[];
    for bg=apical_begin
        [delInd,~]=sub_tree(intr,bg);
        subind=[subind;find(delInd)];
    end
    basal_tree=delete_tree(intr,subind);
end