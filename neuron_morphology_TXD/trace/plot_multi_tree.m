% plot_multi_tree(trees,color,layout,text)
%
% trees::trees struct array;
% color::string: a color type used for all neuron;
%        or cell: each string in the cell is used as the color of n-th
%        neuron.
% layout::2-number array;
% text::(option) text cell
function plot_multi_tree(trees,color,layout,varargin)
    if ischar(color)||((isnumeric(color)&&length(color)==3))
        color_gr=repmat({color},layout);
    elseif iscell(color)
        color_gr=color;
    end
    for n=1:length(trees)
        subplot(layout(1),layout(2),n);
        if nargin==4
            ax(n)=plot_tree_color(trees(n),color_gr{n},varargin{1}{n});
        else
            ax(n)=plot_tree_color(trees(n),color_gr{n});
        end
        scale(n,:)=get_axis_scale(ax(n));
    end
    universal_scale=max(scale);
    universal_scale_xy=ones([1,2])*max(universal_scale); % x and y use the same scale
    for n=1:length(ax)
        set_axis_scale(ax(n),universal_scale_xy);
    end
    axis_compact(ax,layout);
    set(gcf,'Color','w');
end
function set_axis_scale(ax,scale)
    lim=range([ax.XLim',ax.YLim']);
    ax.Position(3:4)=scale.*lim;
end
function axis_compact(axes,layout)
    % zoom axes
    pos=cell2mat(get(axes,'Position'));
    if layout(1)*layout(2)>length(axes)
        pos(end+1:layout(1)*layout(2),:)=0;
    end
    Ws=reshape(pos(:,3),[layout(2),layout(1)])';W_group=max(Ws,[],1);W_sum=sum(W_group);
    Hs=reshape(pos(:,4),[layout(2),layout(1)])';H_group=max(Hs,[],2);H_sum=sum(H_group);
    zoom_fold=1/max([W_sum,H_sum]);
    for n=1:length(axes)
        axes(n).Position(3:4)=axes(n).Position(3:4)*zoom_fold;
    end
    
    % adjust left-bottom position
    pos=cell2mat(get(axes,'Position'));
    if layout(1)*layout(2)>length(axes)
        pos(end+1:layout(1)*layout(2),:)=0;
    end
    Ws=reshape(pos(:,3),[layout(2),layout(1)])';W_group=max(Ws,[],1);W_sum=sum(W_group);
    Hs=reshape(pos(:,4),[layout(2),layout(1)])';H_group=max(Hs,[],2);H_sum=sum(H_group);
    W_margin=(1-W_sum)/2;
    H_margin=(1-H_sum)/2;
    W_diff=(ones([layout(1),1])*W_group-Ws)/2;
    H_diff=(H_group*ones([1,layout(2)])-Hs)/2;
    x_adjust=ones([layout(1),1])*cumsum([W_margin,W_group(1:end-1)])+W_diff;
    y_adjust=cumsum([H_group(2:end);H_margin],'reverse')*ones([1,layout(2)])+H_diff;
    x_adjust=reshape(x_adjust',[layout(1)*layout(2),1]);
    y_adjust=reshape(y_adjust',[layout(1)*layout(2),1]);
    pos_adjust=[x_adjust,y_adjust,pos(:,3:4)];
    for n=1:length(axes)
        axes(n).Position=pos_adjust(n,:);
    end
end