% data save as 'display_neurons.mat'

sholl_tree(demo_trees(1),30,'-s');
f_sholl=gcf;
axs_sholl=f_sholl.Children([2,3]);
set(axs_sholl,'Visible','off');
legend off
sholl_circles=findobj('Type','Line');
inters_line=sholl_circles(3);
sholl_circles=sholl_circles(4:end);
set(sholl_circles([2:3:30,3:3:30]),'Visible','off');
set(sholl_circles,'Color','k');
set(sholl_circles,"LineWidth",0.2);
set(sholl_circles,"Color",[0.5882353 0.5882353 0.5882353]);
delete(findobj('Type','Patch'));
[~,demo_apical,demo_basal]=define_apical(demo_trees(1));
apical_patch=plot_tree(demo_apical);
basal_patch=plot_tree(demo_basal);
apical_patch.LineWidth=1;
basal_patch.LineWidth=1;
apical_patch.FaceColor=[0.5960784 0.3058824 0.6392157];
basal_patch.FaceColor=[0.2156863 0.4941176 0.7215686];
inters_line.Visible='off';
text(-150,120,0,'Apical','Color',[0.5960784 0.3058824 0.6392157]);
text(150,-100,0,'Basal','Color',[0.2156863 0.4941176 0.7215686]);

set(axs_sholl,'Position',[0,0,1,1]);
f_sholl.Position(3)=f_sholl.Position(4);