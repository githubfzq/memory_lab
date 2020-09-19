figure;p=plot_tree(j,[],[],[],[],'-2l-thin');
f=gcf;
f.Color='white';
ax=gca;
ax.XAxis.Visible='off';
ax.YAxis.Visible='off';
ax.Title.Interpreter='none';
ax.Title.String=trees{j}.name;
ax.Title.Position(2)=150;
for i=1:max(size(p))
    p(i).Color=[0 0.5 0.5];
end
saveas(gcf,[trees{j}.name '.png']);
close(gcf);