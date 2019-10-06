% [defined_tree,apical_tree,basal_tree]=define_apical(intree)
% 
% defined_tree:add 'soma,apical,basal' region to intree;
% apical_tree: apical dendrites part of intree;
% basal_tree: basal dendrites part of intree;

function varargout=define_apical(intr)
    eucl=eucl_tree(intr);
    [~,max_ind]=max(eucl);
    ipar=ipar_tree(intr);
    path=ipar(max_ind,:);
    path=path(path~=0);
    apical_begin=path(end-30); %先从root外30作为临时root 避免引入多余的basal dendrites
    
    path_append=ipar(apical_begin,:);path_append=path_append(path_append~=0);
    branch_pns=find(B_tree(intr));
    [~,apical_begin_ind]=intersect(path_append,branch_pns);
    if apical_begin_ind(end)==1
        apical_begin=path_append(apical_begin_ind(end-1)-1);
    else
        apical_begin=path_append(apical_begin_ind(end)-1); %寻找真正的apical的root
    end
    
    [subind,apicaltr]=sub_tree(intr,apical_begin);
    intr.R=ones([length(subind),1])*2;
    intr.R(logical(subind))=1;
    intr.R(path(end))=0;
    intr.rnames={'soma','apical','basal'};
    
    varargout{1}=intr;
    if nargout==2
        varargout{2}=apicaltr;
    end
    if nargout==3
        varargout{2}=apicaltr;
        varargout{3}=delete_tree(intr,find(subind));
    end
end