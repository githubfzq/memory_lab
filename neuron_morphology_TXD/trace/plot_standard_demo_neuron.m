%工作区为plot_standard_neuron
tr=[green_tr,red_tr];
gfpcolor=[40 110 80]/256;
colr={gfpcolor,'r'};
for op=[1,2]
    %计算旋转角度并对神经元进行旋转变换
    euc_dist=eucl_tree(tr(op));
    cor=[tr(op).X,tr(op).Y,tr(op).Z];
    maxdist=cor(euc_dist==max(euc_dist),:);
    triangle=maxdist-cor(1,:);
    angle=atan(triangle(2)/triangle(1))*180/pi;
    rot=rot_tree(tr(op),[0 0 270+angle]);
    %平移胞体至原点
    cor_rot=[rot.X,rot.Y,rot.Z];
    tran=tran_tree(rot,-cor_rot(1,:,:));
    %绘图
    subplot(1,2,op)
    ax(op)=gca;
    ln=plot_tree(tran,[],[],[],[],'-2l');
    arrayfun(@(l) set(l,'Color',colr{op}),ln);
    xyax=get(gca,{'XAxis','YAxis'});
    cellfun(@(a) set(a,'Visible','off'),xyax);
end
%调整坐标轴
ax(2).Position(1)=ax(1).Position(1)+ax(1).Position(3); %靠近
ax(2).Position([3,4])=ax(1).Position([3,4]).*[range(ax(2).XLim),range(ax(2).YLim)]...
    ./[range(ax(1).XLim),range(ax(1).YLim)];%调整第二个坐标轴尺度为第一个大小
scal=axes('Position',[ax(2).Position(1)+ax(2).Position(3),ax(2).Position(2),...
    200*ax(1).Position([3,4])./[range(ax(1).XLim),range(ax(1).YLim)]]);%添加比例尺
scal.XAxis.TickLabels={'';'200\mum';''};
scal.YAxis.TickLabels={'';'200\mum';''};
scal.LineWidth=1;
scal.Position(1)=scal.Position(1)-scal.Position(3);
scal.YAxisLocation='right';
set(gcf,'Color','w');