function scale=get_axis_scale(ax)
    wh=ax.Position(3:4);
    lim=range([ax.XLim',ax.YLim']);
    scale=wh./lim; % scale unit: normalized units/um
end