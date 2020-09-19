<<<<<<< Updated upstream
% axis=plot_tree_color(tree,color,option)
%
% plot tree as defined color; set axis limit as bounds; set axis invisible.
% tree::structured tree: neuron tree to be ploted.
% color::vector:RGB 3-vector or string:RGB value.
% option::string:
%   '2l':plot as thin on x-y plate.
%   others: label this string at left-bottom corner.
% axis::graphic axis object: returned plot axis.

function varargout=plot_tree_color(tree,color,varargin)
    if nargin==3&&strcmp(varargin{1},'2l')
        lines=plot_tree(tree,[],[],[],[],varargin{1});
        arrayfun(@(L) set(L,'Color',color),lines);
    else
        p=plot_tree(tree,[],[],[],[],'-b');
        set(p,'FaceColor',color);
        set(p,'EdgeColor',color);
    end
    xyax=get(gca,{'XAxis','YAxis'});
    cellfun(@(a) set(a,'Visible','off'),xyax);
    xyz=[tree.X,tree.Y,tree.Z];
    [minlim,maxlim]=bounds(xyz);
    lim=[minlim;maxlim]';
    xlim(lim(1,:));ylim(lim(2,:));
    if nargin==3&&~strcmp(varargin{1},'2l')
        if isnumeric(varargin{1})
            txt=num2str(varargin{1});
        else
            txt=varargin{1};
        end
        text(minlim(1),minlim(2),minlim(3),txt,'FontSize',5,'VerticalAlignment','bottom');
    end
    if nargout==1
        varargout{1}=gca;
    end
=======
% axis=plot_tree_color(tree,color,option)
%
% plot tree as defined color; set axis limit as bounds; set axis invisible.
% tree::structured tree: neuron tree to be ploted.
% color::vector:RGB 3-vector or string:RGB value.
% option::string:
%   '2l':plot as thin on x-y plate.
%   others: label this string at left-bottom corner.
% axis::graphic axis object: returned plot axis.

function varargout=plot_tree_color(tree,color,varargin)
    if nargin==3&&strcmp(varargin{1},'2l')
        lines=plot_tree(tree,[],[],[],[],varargin{1});
        arrayfun(@(L) set(L,'Color',color),lines);
    else
        p=plot_tree(tree,[],[],[],[],'-b');
        set(p,'FaceColor',color);
        set(p,'EdgeColor',color);
    end
    xyax=get(gca,{'XAxis','YAxis'});
    cellfun(@(a) set(a,'Visible','off'),xyax);
    xyz=[tree.X,tree.Y,tree.Z];
    [minlim,maxlim]=bounds(xyz);
    lim=[minlim;maxlim]';
    xlim(lim(1,:));ylim(lim(2,:));
    if nargin==3&&~strcmp(varargin{1},'2l')
        if isnumeric(varargin{1})
            txt=num2str(varargin{1});
        else
            txt=varargin{1};
        end
        text(minlim(1),minlim(2),minlim(3),txt,'FontSize',5,'VerticalAlignment','bottom');
    end
    if nargout==1
        varargout{1}=gca;
    end
>>>>>>> Stashed changes
end