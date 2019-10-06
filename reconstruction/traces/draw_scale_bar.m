% draw_scale_bar(scale_length,direction)
%
% draw scale bar on the right bottom, annotated with 'scale_length' um.
% scale_length::number:scale bar length(um)
% direction::string:'x','y','xy'

function draw_scale_bar(len,dir)
    ax=findobj(gcf,'Type','Axes');
    pos=cell2mat(get(ax,'Position'));
    corner=[min(pos(:,1:2)),max(pos(:,1:2)+pos(:,3:4))]; %[x_min,y_min,x_max,y_max]
    corner(2)=max(0.05,corner(2));corner(3)=min(corner(3),0.9);
    scale=get_axis_scale(ax(1));
    bar_len=scale.*[len,len];
    bar_ori=corner([3,2])-[bar_len(1),0];
    bar_pos=[bar_ori,bar_len];
    B=axes('Position',bar_pos);
    B.LineWidth=1;
    B.YAxisLocation='right';
    B.XColor='k';B.YColor='k';
    B.XTickLabel={char(0),[num2str(len),'\mum'],char(0)};
    B.YTickLabel={char(0),[num2str(len),'\mum'],char(0)};
    B.TickLength=[0 0];
    B.Color='none';
    if dir=='x'
        B.YAxis.Visible='off';
    elseif dir=='y'
        B.XAxis.Visible='off';
    end
end