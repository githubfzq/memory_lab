% [defined_tree,apical_tree,basal_tree]=define_apical(intree)
% 
% defined_tree:add 'soma,apical,basal' region to intree;
% apical_tree: apical dendrites part of intree;
% basal_tree: basal dendrites part of intree;

function varargout=define_apical(intr)
    rooted=find_root(intr);
    rind=rindex_tree(rooted);
    begins=find(rind==1);
    beginR=rooted.R(begins);
    apical_begin=begins(beginR==1);
    basal_begin=begins(beginR==2); % begins include apical_begin & basal_begin
    [subind,apicaltr]=sub_tree(rooted,apical_begin);
    intr.rnames={'apical','basal'};
    
    varargout{1}=intr;
    if nargout==2
        varargout{2}=apicaltr;
    end
    if nargout==3
        varargout{2}=apicaltr;
        varargout{3}=delete_tree(intr,find(subind));
    end
end