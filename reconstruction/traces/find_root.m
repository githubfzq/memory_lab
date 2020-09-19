<<<<<<< Updated upstream
% repaire wrong root
% format: tr=find_root(intr)
% 
% tr::the tree returned
% intr::the input tree where intr.R=1 represent apical and intr.R=2
% represent basal.

function tr=find_root(intr)
    ipars=ipar_tree(intr);
    rind=rindex_tree(intr);
    branch_roots=find(rind==1); % find the root of apical and basal parts
    if isNearby(ipars,branch_roots) % two roots are nearby
        if intr.R(branch_roots(1))==2 % root of basal is root
            tr=intr;
        else % root of apical is root
            tr=redirect_tree(intr,branch_roots(2));
        end
    else
        if intr.R(branch_roots(1))==2
            tr=redirect_tree(intr,branch_roots(1)); % always redirect root to basal root
        else
            tr=redirect_tree(intr,branch_roots(2));
        end
    end
end
function near=isNearby(ipar,ind)
    pars=ipar(ind,:);
    nearPoint=(pars(1,1:end-1)==pars(2,2:end)).*(sum(pars(:,1:end-1)==0)==0);
    nearPoint2=(pars(1,2:end)==pars(2,1:end-1)).*(sum(pars(:,1:end-1)==0)==0); % find nearby point
    near=~isempty(find(nearPoint))||~isempty(find(nearPoint2));
=======
% repaire wrong root
% format: tr=find_root(intr)
% 
% tr::the tree returned
% intr::the input tree where intr.rname=4 represent apical and intr.rname=3
% represent basal.

function tr=find_root(intr)
    ipars=ipar_tree(intr);
    rind=rindex_tree(intr);
    
    % find the root of soma, apical and basal parts seperately
    [roots,rloc]=getPartRoots(rind, intr.R, intr.rnames); 
    
    if roots(1)~=1
        newR=intr.R(1);
        intr.R(roots(1))=newR; % set soma to apical/basal
        intr.R(roots(2))=rloc(1); 
        tr=redirect_tree(intr, roots(2)); % set apical root as soma
    else
        tr=intr;
    end
end

function varargout=getPartRoots(rind, R, rnames)
    [~,rlocSoma]=ismember('1',rnames);
    [~,rlocApical]=ismember('4',rnames);
    [~,rlocBasal]=ismember('3',rnames);
    somaRoot=find((rind==1)&(R==rlocSoma));
    apicalRoot=find((rind==1)&(R==rlocApical));
    basalRoot=find((rind==1)&(R==rlocBasal));
    roots=[somaRoot, apicalRoot, basalRoot];
    R_loc=[rlocSoma,rlocApical,rlocBasal];
    varargout{1}=roots;
    if nargout==2
        varargout{2}=R_loc;
    end
>>>>>>> Stashed changes
end