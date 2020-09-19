function Nprimary=get_primary_branch(tr)
    rs_tr=resample_tree(tr,30);
    sh=sholl_tree(rs_tr,10);
    Nprimary=sh(2);
end
