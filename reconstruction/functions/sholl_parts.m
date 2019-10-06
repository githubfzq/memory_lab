% sholl_profile=sholl_parts(tree,dd);
%
% Do Sholl analysis on apical dendrites and basal dendrites seperately.
% dd::interger:incredient radius;
% sholl_profile::table:include variable r, lable, sholl interactions 
% and id name.

function tb=sholl_parts(tr,delta_r)
    [~,apical,basal]=define_apical(tr); % addpath('..\trace\')
    apical=root_tree(apical);
    apical.X(1)=basal.X(1);apical.Y(1)=basal.Y(1);apical.Z(1)=basal.Z(1);
    [sholl_apic,dd_apic]=sholl_tree(apical,2*delta_r);
    [sholl_bas,dd_bas]=sholl_tree(basal,2*delta_r);
    tb_apic=table(dd_apic'/2,sholl_apic',repmat({'apical'},length(dd_apic),1),...
        'VariableNames',{'radius','intersections','label'});
    tb_bas=table(dd_bas'/2,sholl_bas',repmat({'basal'},length(dd_bas),1),...
        'VariableNames',{'radius','intersections','label'});
    tb=[tb_apic;tb_bas];
    tb.id=repmat({tr.name},length(tb.radius),1);
end