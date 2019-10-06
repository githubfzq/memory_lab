% out_tree=apical_upside(in_tree)
%
% Rotate the neuron in x-y plate so as to put the apical dendrites upside.

function outtr=apical_upside(intr)
    xyz=[intr.X,intr.Y,intr.Z];
    traned=tran_tree(intr,-xyz(1,:));
    angle=compute_apical_angle(traned);
    outtr=rot_tree(traned,[0,0,angle]);
end
function angle_reslt=compute_apical_angle(tr)
    tr=define_apical(tr);
    apical_ind=tr.R==1;
    xyz=[tr.X,tr.Y,tr.Z];
    xyz=xyz(apical_ind,:);
    center_xyz=mean(xyz);
    [theta,~]=cart2pol(center_xyz(1),center_xyz(2));
    angle_reslt=rad2deg(theta)-90;
end