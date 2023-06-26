import numpy as np
from matplotlib import pyplot as plt
from matplotlib import transforms
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes


def plot_unified_scale_grid(fig, axs):
    """Input a layout (row, column), returns multiple axes, which has the same scale.
    Scale means the ratio of actual length and numeric length"""
    centers, factors = compute_gridspec_layout(fig, axs)
    for ind, ax in np.ndenumerate(axs):
        set_bound_center(fig, ax, centers[:, ind[0], ind[1]], system="ratio")
    scale_ax(fig, axs, factors)


def get_aspect_scale(ax):
    """Get scale of an axes(or axes array).
    Scale is the ratio of pixel and data x-, y- coordinate system.
    If axes array given, result[0,:,:] and result[1,:,:] store aspect scale of x- and y- dimension separetively."""
    if not isinstance(ax, np.ndarray):
        mat = ax.transData.get_matrix()
        result = np.diag(mat)[:2]
    else:
        result = np.empty((2, ax.shape[0], ax.shape[1]))
        for ind, a in np.ndenumerate(ax):
            result[:, ind[0], ind[1]] = get_aspect_scale(a)
    return result


def add_scalebar(ax, num, unit, fig, bbox_to_anchor=(0,0,1,1), x_label=None, y_label=None):
    """Add scalebar to axes `ax`.
    num: number or tuple (number, number). Scalebar length.
    unit: String, or (Sring, string).
    x_label, y_label: set defined x-label(y-label). Default: num+unit."""
    if isinstance(num, tuple):
        num_x, num_y = num[0], num[1]
    else:
        num_x, num_y = num, num
    if isinstance(num, tuple):
        unit_x, unit_y = unit[0], unit[1]
    else:
        unit_x, unit_y = unit, unit
    x_label = str(num_x)+unit_x if x_label is None else x_label
    y_label = str(num_y)+unit_y if y_label is None else y_label
    axin = zoomed_inset_axes(
        ax, 1, "lower right",
        bbox_to_anchor=bbox_to_anchor,
        bbox_transform=fig.transFigure,
        axes_kwargs={
            "xlabel": x_label,
            "ylabel": y_label,
            "xticks": [],
            "yticks": [],
            "xlim": (0, num_x),
            "ylim": (0, num_y)
        },
    )
    # axin.set_ylim(top=num_y/num_x)
    axin.yaxis.set_label_position("right")
    axin.spines["top"].set_visible(False)
    axin.spines["left"].set_visible(False)
    axin.spines["right"].set_visible(True)
    axin.patch.set_alpha(0)


def get_xy_lim_range(ax, fig=None, unit="data"):
    """Get range of xlim and ylim for given axes(or axes array).
    unit: {'data', 'pixel', 'ratio'}. Default: 'data'.
    If unit = 'ratio', fig object must be specific."""
    if not isinstance(ax, np.ndarray):
        if unit == "data":
            xlim, ylim = ax.get_xlim(), ax.get_ylim()
            return xlim[1] - xlim[0], ylim[1] - ylim[0]
        elif unit == "pixel":
            trans = ax.transAxes
            return trans.transform((1, 1)) - trans.transform((0, 0))
        elif unit == "ratio":
            trans = ax.transAxes + fig.transFigure.inverted()
            return trans.transform((1, 1)) - trans.transform((0, 0))
    else:
        result = np.empty((2, ax.shape[0], ax.shape[1]))
        for ind, a in np.ndenumerate(ax):
            result[:, ind[0], ind[1]] = get_xy_lim_range(a, fig, unit)
        return result


def get_optimal_gridspec_ratio(axs):
    """Compute optimal gridspec ratio."""
    lim = get_xy_lim_range(axs)
    return lim[0, :, :].max(axis=0), lim[1, :, :].max(axis=1)


def get_bound_center(ax, fig=None, unit="pixel"):
    """Get bounding box center in figure pixel system (or figure ratio system).
    unit: {'pixel','ratio'}.
    for pixel unit, fig object is not needed;
    for ratio unit, fig object must be specific."""
    if not isinstance(ax, np.ndarray):
        if unit == "pixel":
            return ax.transAxes.transform((0.5, 0.5))
        elif unit == "ratio":
            comp = ax.transAxes + fig.transFigure.inverted()
            return comp.transform((0.5, 0.5))
    else:
        result = np.empty((2, ax.shape[0], ax.shape[1]))
        for ind, a in np.ndenumerate(ax):
            result[:, ind[0], ind[1]] = get_bound_center(a, fig, unit)
        return result


def set_bound_center(fig, ax, center, system="pixel"):
    """Set a given axes to specific center.
    center: (x, y). 
    system: unit, include {"pixel", "ratio"}"""
    if system == "pixel":
        w_before, h_before = get_xy_lim_range(ax, unit="pixel")
        position = [
            center[0] - w_before / 2,
            center[1] - h_before / 2,
            center[0] + w_before / 2,
            center[1] + h_before / 2,
        ]
        position[:2], position[2:] = fig.transFigure.inverted().transform(
            [position[:2], position[2:]]
        )
        position[2:] = [position[2] - position[0], position[3] - position[1]]
        ax.set_position(position)
    elif system == "ratio":
        bbox = ax.get_position()
        com_trans = ax.transAxes + fig.transFigure.inverted()
        center0 = com_trans.transform((0.5, 0.5))
        dx, dy = center[0] - center0[0], center[1] - center0[1]
        mat = [[1, 0, dx], [0, 1, dy], [0, 0, 1]]
        tran = transforms.Affine2D(mat)
        bbox_new = bbox.transformed(tran)
        ax.set_position(bbox_new)


def scale_ax(fig, ax, factor):
    """Scale both x-,y- axes without moving center.
    factor. scale factor within (0,1), 
    format: if only one axes, [x-scale, y-scale], or (2*w*h)"""
    if not isinstance(ax, np.ndarray):
        bbox = ax.get_position()
        mat = [
            [factor[0], 0, (1 - factor[0]) / 2],
            [0, factor[1], (1 - factor[1]) / 2],
            [0, 0, 1],
        ]
        trans_scale = transforms.Affine2D(mat)
        trans = (
            fig.transFigure
            + ax.transAxes.inverted()
            + trans_scale
            + ax.transAxes
            + fig.transFigure.inverted()
        )
        bbox_new = bbox.transformed(trans)
        ax.set_position(bbox_new)
    else:
        for ind, a in np.ndenumerate(ax):
            scale_ax(fig, a, factor[:, ind[0], ind[1]])


def compute_gridspec_layout(fig, axs):
    """Compute new gridspec layout to put axes compact and keep the same aspect.
    Return: new center point (figure ratio unit) and new scale factor of each axes.
    Both `center` and `scale factor` are 3-d array (2*#width*#height).
    For `center` center[0,:,:] and center[1,:,:] store x- and y- seperately, unit is figure ratio.
    For `scale factor`, factor[0,:,:] and factor[1,:,:] store x- and y- factor seperately."""
    # unify aspect scale
    aspects = get_aspect_scale(axs)
    zoom = np.min(aspects) / aspects
    ratio_raw = get_xy_lim_range(axs, fig, "ratio")
    ratio_new = zoom * ratio_raw
    scale = 1 / get_max_ratio_sum_compact(ratio_new)
    factors = zoom * scale
    # compute center
    ratio1 = ratio_new * scale
    ratio_x1, ratio_y1 = compute_compact_ratio(ratio1)
    ratio_wh = np.sum(ratio_x1) / sum(ratio_y1)
    fig_size = fig.get_size_inches()
    fig_ratio_wh = fig_size[0] / fig_size[1]
    if ratio_wh < fig_ratio_wh:
        dx = (1 - ratio_wh) / 2
        center_x = dx + np.cumsum(ratio_x1) - ratio_x1 / 2
        center_y = np.cumsum(ratio_y1) - ratio_y1 / 2
    else:
        dy = (1 - 1 / ratio_wh) / 2
        center_x = np.cumsum(ratio_x1) - ratio_x1 / 2
        center_y = dy + np.cumsum(ratio_y1) - ratio_y1 / 2
    center_y = center_y[::-1]
    center1, center2 = np.meshgrid(center_x, center_y)
    centers = np.stack([center1, center2])
    return centers, factors


def compute_compact_ratio(ratio):
    """Given ratios of axes array, compute compact grid layout along x-axis and y-axis. 
    ratio: (2*w*h) array, each element belongs to (0,1)"""
    return ratio[0, :, :].max(axis=0), ratio[1, :, :].max(axis=1)


def get_max_ratio_sum_compact(ratio):
    """Given ratios of axes array, get the max ratio summation based on compact grid layout.
    ratio: (2*w*h) array."""
    ratio_x, ratio_y = compute_compact_ratio(ratio)
    return np.array([ratio_x.sum(), ratio_y.sum()]).max()


def plot_signif_marker(prop):
    """Plot significant marker.
    prop: boxplot component returned by `pyplot.boxplot()`."""

def to_save_figure(to_save, formats=['png','eps','pdf']):
    """Save figure to file.
    to_save: String. Filename of figure file to save.
    formats: List. The format of figure, each saved as a file.
    """
    for fmt in formats:

        plt.savefig(to_save+'.'+fmt, dpi=600, transparent=True, bbox_inches='tight')

def get_subplots_position(axs):
    """
    Get the bounding box position for all subplots.
    axs: the numpy array of 2d-axes.
    return: (x_min, y_min, x_max, y_max)
    """
    shape_ = axs.shape
    pos = np.empty((2, 2, shape_[0], shape_[1]))
    for ind, ax in np.ndenumerate(axs):
        p = ax.get_position()
        pos[:, :, ind[0], ind[1]] = [[p.xmin, p.xmax], [p.ymin, p.ymax]]
    p1 = pos[:, 0, :, :].min(axis=(1, 2))
    p2 = pos[:, 1, :, :].max(axis=(1, 2))
    return (*p1, *p2)
