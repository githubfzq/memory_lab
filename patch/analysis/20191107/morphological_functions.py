import os.path
import warnings
import numpy as np
import pandas as pd
from scipy.stats import sem
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import neurom as nm
from neurom.io.utils import load_morphologies
from neurom.view.matplotlib_impl import plot_morph
from plot_helper import plot_unified_scale_grid, add_scalebar, to_save_figure, get_subplots_position
from analysis_functions import zero_padding, getAllFiles
import os.path
from cache_util import cache
from functools import partial

class morpho_parser:

    selected_imaris_parameters = [
        "Filament Area (sum)",
        "Filament Dendrite Area (sum)",
        "Filament Dendrite Length (sum)",
        "Filament Dendrite Volume (sum)",
        "Filament Distance from Origin",
        "Filament Full Branch Depth",
        "Filament Full Branch Level",
        "Filament Length (sum)",
        "Filament No. Dendrite Branch Pts",
        "Filament No. Dendrite Segments",
        "Filament No. Dendrite Terminal Pts",
        "Filament Volume (sum)",
    ]

    def __init__(self, path_root=""):
        """Create morphological analysis processor.
        path_root: set path root for neuron morphology.
        Default path_root is empty, use default path_root."""
        if not (path_root):
            self.path_root = {
                "swc": "../../../reconstruction/traces/",
                "imaris": "../../../reconstruction/stat/",
            }
        else:
            self.path_root = path_root
        self.neurons = load_morphologies(self.path_root["swc"], cache=True)
        self.files = {}
        self.files["swc"], _ = getAllFiles(self.path_root["swc"], ends="swc")
        self.files["imaris"] = {"stat": [], "soma": []}
        shollFiles, _ = getAllFiles(self.path_root["imaris"])
        for f in shollFiles:
            if "Detailed.csv" in f:
                # filter Detailed labled files
                if "soma" in f:
                    self.files["imaris"]["soma"].append(f)
                else:
                    self.files["imaris"]["stat"].append(f)

    def plot_all_neurons(self, layout=None, to_save="", neurons=None, **kwargs):
        """Plot all neurons in array manner.
        layout: 1x2 tuple. including the row number and the colunm number.
        to_save: the file to save as. If left empty, not to save.
        **kwargs: parameters passed to `plot_multi_neuron()`"""
        if not neurons:
            neurons = self.neurons
        if not (layout):
            s = len(neurons)
            r = int(np.floor(np.sqrt(s)))
            c = int(np.ceil(s / r))
            layout = (r, c)
        return plot_multi_neuron(neurons, layout, to_save, **kwargs)

    def plot_apical_upside(self, neurons=None, **kwargs):
        """Plot all neurons apical-upsided.
        neurons: neurons to plot.
        **kwargs: other parameters passed to `morpho_parser.plot_all_neurons()`."""
        if neurons is None:
            neurons = self.neurons
        roted = [apical_upside(tr) for tr in neurons]
        self.plot_all_neurons(neurons=roted, **kwargs)

    @property
    @cache.shelve_cache
    def sholl_part_data(self):
        """get sholl interactions including apical and basal parts with increaing 1 um for all neurons."""
        result = pd.concat(map(lambda x: get_sholl_parts(x, 1), self.neurons))
        return result

    @property
    @cache.shelve_cache
    def sholl_part_stat(self):
        """Compute sholl analysis result including parts."""
        sholl_dat = self.sholl_part_data
        shollPartsCompute = (
            zero_padding(sholl_dat, "intersections")
            .groupby(["label", "radius"])[["intersections"]]
            .agg([np.mean, sem])
        )
        shollPartsCompute.columns = [
            "_".join(x) for x in shollPartsCompute.columns.ravel()
        ]
        shollPartsCompute.reset_index(inplace=True)
        return shollPartsCompute

    def plot_sholl(self, sholl_part=True, to_save=""):
        """Plot sholl analysis of apical and basal parts.
        sholl_part: logical. plot domain or plot whole."""
        if not sholl_part:
            imarisData = self.imarisData
            shollData = imarisData[
                imarisData.Variable == "Filament No. Sholl Intersections"
            ].loc[:, ["neuron_ID", "Radius", "Value"]]
            shollPlotData = (
                zero_padding(shollData, "Value")
                .groupby("Radius")[["Value"]]
                .agg([np.mean, sem])
                .reset_index()
            )
            plt.plot(shollPlotData["Radius"], shollPlotData[("Value", "mean")])
            plt.fill_between(
                shollPlotData["Radius"],
                shollPlotData[("Value", "mean")] + shollPlotData[("Value", "sem")],
                shollPlotData[("Value", "mean")] - shollPlotData[("Value", "sem")],
                alpha=0.6,
            )
            plt.ylabel("Sholl intersections")
            plt.xlabel("Radius ($\mu m$)")
        else:
            dat = self.sholl_part_stat
            dat = dat.pivot(
                index="radius",
                columns="label",
                values=["intersections_mean", "intersections_sem"],
            )
            plt.plot(
                dat.index,
                dat.intersections_mean.apical,
                dat.index,
                dat.intersections_mean.basal,
                label="",
            )
            ax1 = plt.fill_between(
                dat.index,
                dat.loc[:, ("intersections_mean", "apical")]
                + dat.loc[:, ("intersections_sem", "apical")],
                dat.loc[:, ("intersections_mean", "apical")]
                - dat.loc[:, ("intersections_sem", "apical")],
                alpha=0.6,
                label="apical",
            )
            ax2 = plt.fill_between(
                dat.index,
                dat.loc[:, ("intersections_mean", "basal")]
                + dat.loc[:, ("intersections_sem", "basal")],
                dat.loc[:, ("intersections_mean", "basal")]
                - dat.loc[:, ("intersections_sem", "basal")],
                alpha=0.6,
                label="basal",
            )
            plt.ylabel("Sholl intersections")
            plt.xlabel("$Radius\ (\mu m)$")
            plt.legend()
        if to_save:
            to_save_figure(to_save)

    @property
    @cache.shelve_cache
    def imarisData(self):
        """Read morphological statistic data generated by Imaris software."""
        imarisData = pd.concat(
            map(read_demo_imaris_stat, self.files["imaris"]["stat"]), sort=True
        )
        return imarisData

    @property
    @cache.shelve_cache
    def depthData(self):
        """Get branch depth data."""
        imarisData = self.imarisData
        depthData_ = (
            imarisData[imarisData.Variable == "Dendrite Branch Depth"]
            .loc[:, ["neuron_ID", "Depth", "Level", "Value"]]
            .groupby(["neuron_ID", "Depth"])
            .count()
            .reset_index()
            .drop("Level", axis=1)
            .rename({"Value": "counts"}, axis=1)
        )
        depthData_.Depth = depthData_.Depth.astype("int64")
        return depthData_

    def plot_depth(self, to_save=""):
        """Plot branch order distribution."""
        depthData = self.depthData
        depthPlotData = depthData.groupby("Depth")[["counts"]].agg([np.mean, sem]).reset_index()
        plt.bar(
            depthPlotData["Depth"] + 1, depthPlotData[("counts", "mean")], alpha=0.6
        )
        plt.errorbar(
            depthPlotData["Depth"] + 1,
            depthPlotData[("counts", "mean")],
            depthPlotData[("counts", "sem")],
            linestyle="",
        )
        plt.ylabel("Number of filaments")
        plt.xlabel("Branch order")
        if to_save:
            to_save_figure(to_save)

    @cache.shelve_cache
    def get_all_data(self, item='all'):
        """Get morphological parameters of all neuron for clustering analysis.
        item: {'all', 'imaris', 'python'}. Get only imaris data or stats of python, or all include."""
        if item == 'imaris':
            dat = self.imarisData
            result = dat[dat["Variable"].isin(self.selected_imaris_parameters)][
                ["neuron_ID", "Variable", "Value", "Unit"]
            ]
            result = result.pivot_table("Value", "neuron_ID", "Variable").reset_index()
        elif item == 'python':
            res_map=(
                (
                    os.path.splitext(neuron.name)[0], 
                    res[0], 
                    res[1].real if isinstance(res[1], complex) else res[1]
                ) 
                for neuron in self.neurons for res in self.get_demo_morpho_parameter(neuron)
            )
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                result = pd.DataFrame(res_map, columns=['reconstruction_ID', 'item', 'value'])
            result = result.pivot_table('value','reconstruction_ID','item', dropna=False)
        elif item == 'all':
            imaris_dat = self.get_all_data('imaris')
            imaris_dat.rename(columns={'neuron_ID':'reconstruction_ID'}, inplace=True)
            py_dat = self.get_all_data('python')
            py_dat = py_dat.reset_index()
            result = imaris_dat.merge(py_dat, 'outer', 'reconstruction_ID')
        else:
            raise ValueError('item must be one of {"all", "imaris", "python"}')
        return result

    @cache.shelve_cache
    def get_demo_morpho_parameter(self, neuron):
        return list(self._get_demo_morpho_parameter_(neuron))


    def _get_demo_morpho_parameter_(self, neuron):
        """Get morphology parameters of a neuron (max_sholl_intercept).
        neuron: `Neuron` object."""
        def f1_1(name,part='all', filter_func=lambda x:True):
            if part=='all':
                cur_res = np.array(nm.get(name, neuron))
                cur_res = cur_res[filter_func(cur_res)].mean()
                cur_key = 'mean_' + name
            elif part=='apical':
                cur_res = np.array(nm.get(name, neuron, neurite_type=nm.APICAL_DENDRITE))
                cur_res = cur_res[filter_func(cur_res)].mean()
                cur_key = 'mean_' + name + '(apical)'
            elif part=='basal':
                cur_res = np.array(nm.get(name, neuron, neurite_type=nm.BASAL_DENDRITE))
                cur_res = cur_res[filter_func(cur_res)].mean()
                cur_key = 'mean_' + name + '(basal)'
            yield (cur_key, cur_res)
        def sholl_f(name, part='all'):
            par = {'all':nm.ANY_NEURITE, 'apical':nm.APICAL_DENDRITE,'basal':nm.BASAL_DENDRITE}
            freq = np.array(nm.get(name, neuron,neurite_type=par[part],step_size=1))
            suffix_ = {'all':'', 'apical':'(apical)', 'basal':'(basal)'}
            yield ('max_sholl_intercept'+suffix_[part], (freq.max() if freq.size>0 else np.nan))
            yield ('max_sholl_intercept_radii'+suffix_[part], (freq.argmax() if freq.size>0 else np.nan))
        def f2(name):
            cur_res = nm.get(name, neuron)
            yield (name, cur_res)
        def f1(name, filter_func=lambda x:True):
            yield from f1_1(name, 'all', filter_func)
            yield from f1_1(name, 'apical', filter_func)
            yield from f1_1(name, 'basal', filter_func)
        def f3(name):
            cur_res0=np.array(nm.get(name, neuron))
            cur_res1=np.array(nm.get(name, neuron, neurite_type=nm.APICAL_DENDRITE))
            cur_res2=np.array(nm.get(name, neuron, neurite_type=nm.BASAL_DENDRITE))
            yield (name, (cur_res0.sum() if cur_res0.size>0 else np.nan))
            yield (name+'(apical)', (cur_res1.sum() if cur_res1.size>0 else np.nan))
            yield (name+'(basal)', (cur_res2.sum() if cur_res2.size>0 else np.nan))
        def f4(name):
            yield from sholl_f(name)
            yield from sholl_f(name,'apical')
            yield from sholl_f(name,'basal')
        def f5(name):
            new_name = {
                "neurite_lengths": "total_length_per_neurite",
                "neurite_volumes": "total_volume_per_neurite"
            }.get(name)
            for new_key, res in f1(new_name):
                key = new_key.replace(new_name, name)
                yield (key, res)
        neurom_metric = {'local_bifurcation_angles':f1,
                         'neurite_lengths': f5,
                    'neurite_volume_density': partial(f1, filter_func=lambda x:x<5),
                    'neurite_volumes': f5,
                    'number_of_bifurcations':f2,
                    'number_of_forking_points':f2,
                    'number_of_neurites':f2,
                    'number_of_sections':f2,
                    'number_of_sections_per_neurite':f1,
                    'number_of_segments':f2,
                    'number_of_leaves':f2,
                    'principal_direction_extents':f1,
                    'remote_bifurcation_angles':f1,
                    'section_areas':f1,
                        'section_bif_branch_orders':f1,
                        'section_branch_orders':f1,
                        'section_end_distances':f1,
                        'section_lengths':f1,
                        'total_length_per_neurite':f3,
                        'total_volume_per_neurite':f3,
                        'sholl_frequency':f4}
        for item, fun_ in neurom_metric.items():
            yield from fun_(item)
            
def read_demo_imaris_stat(file):
    """read one morphological data generated by Imaris software."""
    tb = pd.read_csv(file, skiprows=2, header=1)
    tb["neuron_ID"] = os.path.split(file)[-1].split("_Detailed.")[0]
    return tb[np.roll(tb.columns, 1)]


def get_sholl_parts(neuron, step_size):
    """return a dataframe including sholl frequency of apical and basal dendrites
    'step_size': the increasing radius (um)"""
    shollf1 = nm.get(
        "sholl_frequency",
        neuron,
        step_size=step_size,
        neurite_type=nm.NeuriteType.apical_dendrite,
    )
    shollf2 = nm.get(
        "sholl_frequency",
        neuron,
        step_size=step_size,
        neurite_type=nm.NeuriteType.basal_dendrite,
    )
    r1 = np.arange(len(shollf1)) * step_size
    r2 = np.arange(len(shollf2)) * step_size
    df1 = pd.DataFrame(
        {"radius": r1, "intersections": shollf1, "label": "apical", "id": neuron.name}
    )
    df2 = pd.DataFrame(
        {"radius": r2, "intersections": shollf2, "label": "basal", "id": neuron.name}
    )
    df = pd.concat([df1, df2])
    return df[df.intersections > 0]


def plot_multi_neuron(neurons, layout, to_save="", scalebar=(200,"$\mu m$"),fig_size=None):
    """Plot mutiple neurons in array manner.
    neurons: neurom.population.Population.
    layout: 1x2 tuple. including the row number and the colunm number.
    to_save: the file to save as. If left empty, not to save.
    fig_size: default: 2x2 inch multiple with layout."""
    if fig_size is None:
        fig_size=(2*layout[0], 2*layout[1])
    fig, axs = plt.subplots(*layout, figsize=fig_size)
    if layout == (1, 1):
        axs = np.array([[axs]])
    elif layout[1]==1 and axs.ndim==1:
        axs=axs[:,np.newaxis]
    elif layout[0]==1 and axs.ndim==1:
        axs=axs[np.newaxis,:]
    for (ind, ax), neuron in zip(np.ndenumerate(np.array(axs)), neurons):
        plot_morph(neuron, ax)
        ax.set_aspect("equal")
        ax.autoscale()
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_axis_off()
    left = len(neurons) - layout[0] * layout[1]
    if left<0:
        for ax in axs.flat[left:]:
            ax.set_visible(False)
    # force draw so that the following adjustment get proper positions
    plt.draw() 

    plot_unified_scale_grid(fig, axs)
    add_scalebar(axs[0,0], scalebar[0], scalebar[1], fig,
                 bbox_to_anchor=get_subplots_position(axs))
    if bool(to_save):
        to_save_figure(to_save)


def apical_upside(neuron):
    """Put apical part of a neuron upside."""
    traned = nm.geom.translate(neuron, -neuron.soma.center)
    apic = [
        nrt for nrt in traned.neurites if nrt.type == nm.NeuriteType.apical_dendrite
    ]
    if apic:
        apic_center = np.mean(
            np.concatenate(list(map(lambda a: a.points[:, :3], apic))), axis=0
        )
        angle = np.pi / 2 - np.angle(complex(apic_center[0], apic_center[1]))
        roted = nm.geom.rotate(traned, (0, 0, 1), angle)
        return roted
    else:
        return traned

def plot_sholl_demo(neuron, step_size=30, label_dict=None, to_save=""):
    """Plot sholl analysis demo figure.
    Display the apical and basal part of a neuron, and concentric circles of sholl analysis.
    Args:
    - neuron: [neurom.fst._core.FstNeuron], a neuron object.
    - step_size: [int], the interval radius of the concentric circles.
    - label_dict: [dict], a dictionary indicate the position of label.
        the keys of dict are the label name, and the values are the (x, y) position.
        Default: {'Apical': (x1, y1), 'Basal': (x2, y2)}, where (x1,y1) and (x2,y2) are computed automatically.
    """
    neuron = apical_upside(neuron)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, aspect='equal')
    plot_morph(neuron, ax)

    # draw circles
    center = neuron.soma.center[:2]
    _dist = np.linalg.norm(neuron.points[:,:2]-center, axis=1).max()
    radii = np.arange(step_size, _dist, step_size)
    patches = []
    for rad in radii:
        circle = mpatches.Circle(center, rad, fill=False, edgecolor='dimgray')
        patches.append(circle)
    p = PatchCollection(patches, match_original=True)
    ax.add_collection(p)

    # add labels
    if label_dict is None:
        apical_points = np.concatenate([x.points for x in nm.iter_neurites(neuron, filt=lambda t: t.type==nm.APICAL_DENDRITE)])
        apical_label_pos_xs = [apical_points[:,0].min()-center[0], apical_points[:,0].max()-center[0]]
        apical_label_pos_x = apical_label_pos_xs[0] if np.abs(apical_label_pos_xs[0])>np.abs(apical_label_pos_xs[1]) else apical_label_pos_xs[1]
        apical_label_pos_y = center[1]+(apical_points[:,1].max()-center[1])/2
        basal_points = np.concatenate([x.points for x in nm.iter_neurites(neuron, filt=lambda t: t.type==nm.BASAL_DENDRITE)])
        basal_label_pos_xs = [basal_points[:,0].min()-center[0],basal_points[:,0].max()-center[0]]
        basal_label_pos_x = basal_label_pos_xs[0] if np.abs(basal_label_pos_xs[0])>np.abs(basal_label_pos_xs[1]) else basal_label_pos_xs[1]
        basal_label_pos_y = center[1]+(basal_points[:,1].min()-center[1])/2
        label_dict = {'Apical':(apical_label_pos_x, apical_label_pos_y), 'Basal': (basal_label_pos_x, basal_label_pos_y)}

    for name,pos in label_dict.items():
        plt.annotate(name, pos)

    ax.autoscale()
    ax.set_axis_off()
    plt.title(None)
    if to_save:
        to_save_figure(to_save)

def plot_single_neuron(neuron, put_apical_upside=False, to_save=""):
    """Plot a neuron.
    neuron: [neurom.fst._core.FstNeuron] The neuron to plot.
    put_apical_upside: logical. Whether put apical dendrite upside."""
    fig, ax = plt.subplots(subplot_kw={'aspect':'equal'})
    if put_apical_upside:
        neuron=apical_upside(neuron)
    plot_morph(neuron, ax)
    ax.autoscale()
    ax.set_title("")
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_axis_off()
    if to_save:
        to_save_figure(to_save)