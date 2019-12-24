% Find a new soma point in a neuron tree, so that the label of each subtree
% is uniform.
% Format: outTree=find_new_soma(inTree)
% output:
% a tree whose region tr.R is modified where soma point locate and soma is redirected to root.
function res=find_new_soma(intr)
% res=sort_tree(intr);
    res=intr;
    somaRInd=find(ismember(res.rnames,'1')); % tree.rnames(somaRInd)='1'
    somaInd=find(res.R==somaRInd);
    ipar=ipar_tree(res);
    idpar=idpar_tree(res);
    rind=rindex_tree(res);
    secRoots=sort(setdiff(find(rind==1),somaInd));
    [secRootsPath,passover]=findSectionPath(ipar,secRoots,somaInd);
    
% get branch nodes
    bct=typeN_tree(res);
    Bnodes=secRootsPath(bct(secRootsPath)==2);
    for node=Bnodes
        child=ipar(ipar(:,2)==node,1);
        parent=idpar(node);parent(parent==somaInd)=[];
        if length(unique(res.R([child(:);parent])))>1
            break;
        end
    end
    
    if node~=somaInd
    
        % get new label of original soma node
        somaChild=ipar(ipar(:,2)==somaInd,1);
        childLabel=unique(res.R(somaChild));
        if length(childLabel)==1
            somaNewR=childLabel;
        end

        % set new label
        res.R(somaInd)=somaNewR;
        res.R(node)=somaRInd;

        % redirect to new soma
        res=redirect_to_soma(res);
    end
end
function varargout=findSectionPath(ipar,sect,origin)
% Find a path between two nodes.
% sect: two elements.
% origin: soma point, ususally equal to 1.
    res=ipar(sect(2),find(ipar(sect(2),:)==sect(2)):find(ipar(sect(2),:)==sect(1)));
    passover = false;
    if isempty(res) % the section between two nodes include origin node (soma)
        passover=true;
        x=ipar(sect(2),find(ipar(sect(2),:)==sect(2)):find(ipar(sect(2),:)==origin)-1);
        y=ipar(sect(1),find(ipar(sect(1),:)==sect(1)):find(ipar(sect(1),:)==origin));
        res=[x y(end:-1:1)];
    end
    varargout{1}=res;
    if nargout==2
        varargout{2}=passover;
    end
end