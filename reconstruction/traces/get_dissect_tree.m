% get all sections in a tree like the function of dissect_tree, but the
% section nodes are not in continuation.
function sections=get_dissect_tree(tree)
    iparBT=ipar_BT_tree(tree);
    sections=iparBT(iparBT(:,2)~=0,[2,1]);
end
function BTipar=ipar_BT_tree(tree)
% Returns for the tree a matrix ipar of indices to the parent of branch or
% terminal nodes.
    ipar=ipar_tree(tree);
    bct=typeN_tree(tree);
    ipar(bct(ipar(:,1))==1,:)=[];
    BTipar=zeros(size(ipar));
    for row=1:size(ipar,1)
        cur=ipar(row,bct(ipar(row,ipar(row,:)~=0))~=1);
        BTipar(row,1:length(cur))=cur;
    end
    BTipar(:,sum(BTipar)==0)=[];
end