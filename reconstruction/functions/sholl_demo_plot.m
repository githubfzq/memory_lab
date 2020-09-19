<<<<<<< Updated upstream
sholl_tree(gfp_trees{1},20,'-s');
%减少同心圆密集度，每5个同心圆只保留一个，画出半径=50递增的效果
f=gcf;h=findobj(f,'Type','Line');
circle=h([4:end]);
circle_hide=circle(setdiff(1:end,1:5:end));
arrayfun(@(x) set(x,'Visible','off'),circle_hide);
%改变标注位置
lg=f.Children(1);
lg.Position(1)=1-lg.Position(3);
lg.Box='off';
%去掉格子
ax=f.Children(3);
=======
sholl_tree(gfp_trees{1},20,'-s');
%减少同心圆密集度，每5个同心圆只保留一个，画出半径=50递增的效果
f=gcf;h=findobj(f,'Type','Line');
circle=h([4:end]);
circle_hide=circle(setdiff(1:end,1:5:end));
arrayfun(@(x) set(x,'Visible','off'),circle_hide);
%改变标注位置
lg=f.Children(1);
lg.Position(1)=1-lg.Position(3);
lg.Box='off';
%去掉格子
ax=f.Children(3);
>>>>>>> Stashed changes
ax.XGrid='off';ax.YGrid='off';