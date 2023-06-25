import os.path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
from brokenaxes import brokenaxes
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from scipy.stats import normaltest, f, ttest_ind, ranksums, sem
from morphological_functions import morpho_parser
from patch_clamp_functions import electro_parser
from analysis_functions import merge_dicts, fillna_unique, zero_padding
from plot_helper import plot_signif_marker, to_save_figure
from functools import cache


class cluster_processor(electro_parser, morpho_parser):
    """Inherate electro_parser and morpho_parser, integrate patch clamp data and morphological data. Do cluster analysis.
    There should be a csv file named `id_table.csv` in imaris statistic folder, for mapping electro_data and morpho_data."""

    # Cache returning value of function`get_all_data()`
    dat = None
    # Cache Cluster analysis result table
    ca_dat = None

    elec_ps = electro_parser()
    mpp = morpho_parser()

    def __init__(self, electro_parser_=None, morpho_parser_=None):
        """Parameters: 
        electro_parser_, morpho_parser_:
        Include electro_parser and morpho_parser.
        class::electro_parser, morpho_parser."""
        electro_parser.__init__(self)
        path_root = self.path_root
        files = self.files
        morpho_parser.__init__(self)
        self.path_root = merge_dicts(path_root, self.path_root)
        self.files = merge_dicts(files, self.files)
        self.files["id_tab"] = os.path.join(self.path_root["imaris"], "id_table.csv")
        self.elec_ps = (
            electro_parser_ if electro_parser_ is not None else electro_parser()
        )
        self.mpp = morpho_parser_ if morpho_parser_ is not None else morpho_parser()

    @cache
    def get_all_data(self, how="outer"):
        """Merge electrophysiological data and morphological data.
        Parameters:
        - select_trace: Select one sweep of AP electrophysiological recording to represent its neuron.
        - how: passed to DataFrame.merge()"""
        elec_dat = self.elec_ps.get_all_data(method="all")
        morpho_dat = self.mpp.get_all_data()
        map_dat = self.get_id_table()
        map_dat.rename(columns={"CellID": "cellID_ap"}, inplace=True)
        elec_dat = fillna_unique(elec_dat, 'cellID_ap', 'elec_empty_')
        map_dat = fillna_unique(map_dat, 'cellID_ap', 'map_empty_')
        result = elec_dat.merge(
            map_dat, how, on="cellID_ap"
        )
        result = fillna_unique(result, 'reconstruction_ID', 'res_empty_')
        morpho_dat.rename(columns={'neuron_ID':"reconstruction_ID"}, inplace=True)
        morpho_dat = fillna_unique(morpho_dat, 'reconstruction_ID', 'morpho_empty_')
        result = result.merge(morpho_dat, how, on="reconstruction_ID")
        result.replace(r'.*empty.*', np.nan, regex=True, inplace=True)
        return result
    
    def get_id_table(self):
        """Read excel spreadsheet about electrophysiological and morphological cellIDs and cell position"""
        map_dat = pd.read_csv(
            self.files["id_tab"], encoding="unicode_escape", na_values="-"
        )
        map_dat = map_dat.rename({"patch_ID": "CellID"}, axis=1)
        return map_dat

    def k_means(self, dat=None, n_cluster=2, return_scaled=False, return_filled=False, return_model=False, random_state=19,
                label_mapping=None):
        """Compute k-means.
        dat: data used to do k-means algorithm.
        return_filled: Whether return NA_filled data.
        label_mapping: dict. Re-mapping the cluster ids.
        """
        kmeans = KMeans(n_clusters=n_cluster, init='random', n_init=10, random_state=random_state)
        if dat is None:
            dat = self.get_all_data("inner")
            dat = dat[dat["cellID_ap"].notna()]
        ca_dat = k_means_preprocess(dat)
        ca_values = ca_dat.drop(columns='cellID_ap')
        ca_dat_filled = ca_values.fillna(ca_values.mean())
        ca_dat_scaled = preprocessing.scale(ca_dat_filled)
        kmeans.fit(ca_dat_scaled)
        pred = kmeans.fit_predict(ca_dat_scaled)
        if label_mapping:
            pred = np.vectorize(lambda k: label_mapping.get(k, k))(pred)
        if return_model:
            return kmeans
        if not return_filled:
            ca_dat["cluster"] = pred
            self.ca_dat = ca_dat.copy()
            return ca_dat
        elif not return_scaled:
            ca_dat_filled["cluster"] = pred
            return ca_dat_filled
        else:
            ca_dat_scaled = pd.DataFrame(ca_dat_scaled, columns=ca_dat_filled.columns)
            ca_dat_scaled["cluster"] = pred
            return ca_dat_scaled

    def plot_cluster_scatter(self, item1, item2, to_save="", ca_dat=None):
        """Plot cluster analysis scatter plot `item1` vs. `item2`.
        Parameters:
        item1, item2: columns string from cluster_processor.k_means() or cluster_processor.ca_dat
        ca_dat: result from cluster_processor.k_means()"""
        if ca_dat is None:
            ca_dat = self.k_means() if self.ca_dat is None else self.ca_dat.copy()
        _color = ("C{}".format(i) for i in range(9))
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for clust in ca_dat.cluster.unique():
            clust_row = ca_dat.cluster.values == clust
            ax.scatter(
                ca_dat[clust_row][item1],
                ca_dat[clust_row][item2],
                color=next(_color),
                label='cluster {}'.format(clust + 1),
            )
        ax.legend()
        plt.xlabel(item1)
        plt.ylabel(item2)
        if bool(to_save):
            to_save_figure(to_save)

    def plot_decomposition_scatter(
        self, ca_dat=None, dim1=0, dim2=1, to_save="", plot_3d=False, fig_size=(3,3), show_legend=True,
        label_map=None, colors=None,
    ):
        """Plot PCA scatter.
        dim1, dim2: The number of dimension after PCA.
        plot_3d: 3D scatter plot. If true, auto-select first 3 dimensions.
        label_map: dict. map cluster ID to label string. Default: {0: 'cluster 1', 1: 'cluster 2', ...}.
        colors: list. default to C1, C2, C3...
        """
        if ca_dat is None:
            ca_dat = self.k_means(return_filled=True, return_scaled=True)
        X = ca_dat.to_numpy()[:, :-1]
        pca = PCA(n_components=0.8)
        newX = pca.fit_transform(X)
        _color = ("C{}".format(i) for i in range(9))
        fig = plt.figure(figsize=fig_size)
        if label_map is None:
            label_map={clust:'cluster {}'.format(clust + 1) for clust in np.sort(ca_dat.cluster.unique())}
        if plot_3d:
            ax = fig.add_subplot(projection='3d')
            ax.view_init(30,30)
            dim1, dim2, dim3 = (0, 1, 2)
            for clust in np.sort(ca_dat.cluster.unique()):
                clust_row = ca_dat.cluster.values == clust
                ax.scatter(
                    newX[clust_row, 0],
                    newX[clust_row, 1],
                    newX[clust_row, 2],
                    color=next(_color),
                    label=label_map[clust],
                )
            ax.dist = 12 # avoid labels incomplete
            ax.legend()
            ax.set_xlabel("Component {}".format(dim1 + 1))
            ax.set_ylabel("Component {}".format(dim2 + 1))
            ax.set_zlabel("Component {}".format(dim3 + 1))
            plt.yticks(rotation=30,horizontalalignment='center',
                        verticalalignment='baseline',rotation_mode='anchor')
            # plt.tight_layout()
        else:
            ax = fig.add_subplot(1, 1, 1)
            for clust in np.sort(ca_dat.cluster.unique()):
                clust_row = ca_dat.cluster.values == clust
                ax.scatter(
                    newX[clust_row, dim1],
                    newX[clust_row, dim2],
                    color=next(_color),
                    label=label_map[clust],
                )
            if show_legend:
                ax.legend()
            ax.set_xlabel("Component {}".format(dim1 + 1))
            ax.set_ylabel("Component {}".format(dim2 + 1))
            plt.tight_layout()
        if to_save:
            to_save_figure(to_save)

    def stat_analyse(
        self, tab, group_variable="cluster", return_valid_variable=True, **kwargs
    ):
        """See stat_analyse().
        Return: DataFrame.
        return_valid_variable: Only return results of valid test(not None).
        **kwargs: variable passed to `stat_analyse()`."""
        res = []
        for v in tab.columns.drop(group_variable):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cul_res = stat_analyse(tab, v, group_variable, **kwargs)
            res.append((v, *cul_res))
        result = pd.DataFrame(res, columns=["variable", "stat_method", "p_value"])
        if return_valid_variable:
            result = result[result.p_value < 1]
        return result
    
    def svm_train(self, train_data):
        """Linear SVM classifier model.
        `train_data`: DataFrame. It should have column `cluster`.
        Return: a predicting function `predict_fun(predict_data)`. predict_data: A DataFrame of morphology without 'cluster'."""
        def preprocess(dat):
            to_drop = ['cellID_ap', 'reconstruction_ID', 'neuron_ID']
            drop = set(to_drop) & set(dat.columns)
            train_data = dat.drop(drop, axis=1)
            if 'cluster' in dat.columns:
                X = train_data.drop(columns='cluster').to_numpy()
                y = train_data.cluster.values
                return X,y
            else:
                X = train_data.to_numpy()
                return X
        X_train ,y_train = preprocess(train_data)
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        clf = make_pipeline(imp, preprocessing.StandardScaler(), LinearSVC())
        clf.fit(X_train, y_train)
        def predict_fun(dat):
            X_new = preprocess(dat)
            return clf.predict(X_new)
        return predict_fun
    
    def get_morpho_cluster_label(self, electro_data, cluster_data):
        """Given electrophysiological dataset with cluster labels, return relavant morphological dataset with labels.
        Parameters:
        electro_data: dataFrame with cluster label (eg. returned by k_means algorithm), should have columns ['cellID_ap', 'cluster'].
        cluster_data: A large dataset containing electrophysiological data as well as morphological data."""
        dat = electro_data[['cellID_ap', 'cluster']].dropna()
        morpho_para_names = self.mpp.get_all_data().columns.drop('reconstruction_ID')
        to_merge = cluster_data[['cellID_ap', 'reconstruction_ID'] + morpho_para_names.tolist()]
        result = to_merge.merge(dat, 'left', 'cellID_ap')
        return result



def stat_analyse(tab, compare_variable, group_variable="cluster", group1=0, group2=1, test_small_sample=False):
    """Do stat analysis for each variable except group_variable.
    group_variable: a column in tab, values containing {0, 1, ...},
    tab: DataFrame.
    compara_variable: The variable in tab to stat.
    group1, group2: two value in group_variable, used to filter data for stats.
    test_small_sample: If sample<20, do not compare (default).
    Return: (stat_method, p_value)."""
    dat1 = tab[tab[group_variable] == group1][compare_variable].to_numpy()
    dat2 = tab[tab[group_variable] == group2][compare_variable].to_numpy()
    dat1_notna_ind = np.where(np.logical_not(np.isnan(dat1)))[0]
    dat2_notna_ind = np.where(np.logical_not(np.isnan(dat2)))[0]
    try:
        _, norm1 = normaltest(dat1, nan_policy="omit")
        _, norm2 = normaltest(dat2, nan_policy="omit")
        both_normal = norm1 > 0.05 and norm2 > 0.05
    except ValueError:
        both_normal = False
    if both_normal:
        stat_method = "t_test"
        varp = f_test(dat1, dat2)
        if varp > 0.05:
            _, resultp = ttest_ind(dat1, dat2, nan_policy="omit")
        else:
            _, resultp = ttest_ind(dat1, dat2, equal_var=False, nan_policy="omit")
    elif (dat1_notna_ind.size > 20 and dat2_notna_ind.size > 20 and not test_small_sample) or (test_small_sample):
        stat_method = "Wilcoxon rank-sum test"
        dat1 = dat1[dat1_notna_ind]
        dat2 = dat2[dat2_notna_ind]
        res = ranksums(dat1, dat2)
        resultp = res.pvalue
    else:
        stat_method = "None"
        resultp = 1
    return (stat_method, resultp)


def k_means_preprocess(dat):
    """Remove some column from `dat` for k-means analysis."""
    to_drop = [
        "recording_date",
        "recording_cell_id",
        "cellID_epsc",
        "cellID_ramp",
        "reconstruction_ID",
        "patch_date",
        "CellID",
        "layer",
        "species",
        "virus",
        "weight",
        "neuron_ID",
        "isOutlier",
        "signal",
        "ramp_slope",
    ]
    if "isOutlier" in dat.columns:
        dat = dat[np.logical_not(dat["isOutlier"])]
    drop = set(dat.columns) & set(to_drop)
    ca_dat = dat.drop(drop, axis=1)
    return ca_dat


def f_test(a, b):
    """Test whether var of a and b are equal."""
    a = a[np.logical_not(np.isnan(a))]
    b = b[np.logical_not(np.isnan(b))]
    varp = f.cdf(a.var() / b.var(), a.size - 1, b.size - 1)
    return varp


def plot_cluster_hist(
    dat,
    col,
    group1=0,
    group2=1,
    to_save="",
    fig_size=(3,3),
    alpha=0.6,
    bins=20,
    xlabel=None,
    ylabel="No. of neuron",
    plot_cumulative=False,
    legend_kw=None,
    show_legend=True,
    **kwargs
):
    """Plot histogram of two clusters.
    dat: CA result data, whose columns contains `cluster`.
    col: column in `dat` to be plotted.
    xlabel: x-label string. Default is equal to `col`.
    plot_cumulative: {True, False}. If True, draw cumulative curve inset with `plot_cumulative_curve()`.
    axin_kw: parameters passed to `figure.add_axes()`.
    **kwargs: other parameters passed to `axes.hist()`
    """
    range_ = dat[col].min(), dat[col].max()
    plt.figure(figsize=fig_size)
    ax = plt.axes()
    ax.hist(
        dat[dat.cluster.values == group1][col],
        alpha=alpha,
        bins=bins,
        label="cluster {}".format(group1 + 1),
        range=range_,
        **kwargs
    )
    ax.hist(
        dat[dat.cluster.values == group2][col],
        alpha=alpha,
        bins=bins,
        label="cluster {}".format(group2 + 1),
        range=range_,
        **kwargs
    )
    if plot_cumulative:
        fig = plt.gcf()
        axin = fig.add_axes([0.7, 0.6, 0.17, 0.2])
        plot_cumulative_curve(dat, col, (group1, group2), ax=axin)
    if show_legend:
        ax.legend(**legend_kw)
    if xlabel is None:
        xlabel = col
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        plt.tight_layout()
    if bool(to_save):
        to_save_figure(to_save)


def select_demo_cell_id(tab, feature, clusters=(0,1)):
    """Select two cell_id, each for a cluster, representing appearently the `feature`.
    Parameters:
    tab: DataFrame. Should contain columns {`feature`, 'cellID_ap', 'cluster'}.
    feature: String. A column in `tab`.
    clusters: 2-element tuple. The cluster ID to be fit.
    return:
    A dict eg. {0:`cell_id_1`, 1:`cell_id_2`}"""
    dat1 = tab[tab.cluster==clusters[0]][['cellID_ap', feature]]
    dat2 = tab[tab.cluster==clusters[1]][['cellID_ap', feature]]
    if dat1[feature].mean()<dat2[feature].mean():
        res1 = dat1[dat1[feature]==dat1[feature].min()]['cellID_ap'].values[0]
        res2 = dat2[dat2[feature]==dat2[feature].max()]['cellID_ap'].values[0]
    else:
        res1 = dat1[dat1[feature]==dat1[feature].max()]['cellID_ap'].values[0]
        res2 = dat2[dat2[feature]==dat2[feature].min()]['cellID_ap'].values[0]
    return {clusters[0]:res1, clusters[1]:res2} 

def plot_cluster_stat(tab, feature, clusters=(0,1), fig_size=(2.5,2), xticks=None, capsize=8, plot_box=False, 
                      ylabel=None, to_save="", signif=None, 
                      brokenaxes_dict=None, **kwargs):
    """Plot bar plot of feature for 2 clusters.
    Parameters:
    tab: DataFrame. Should contain columns {`feature`, 'cluster'}.
    feature: String. A column in `tab`.
    capsize: cap size of errorbar. Default 8 points.
    plot_box: Draw boxplot instead of barplot with errorbar.
    ylabel: ylabel.
    to_save: filename to save.
    signif: significant marker, default: None.
    brokenaxes_dict: Dictionary. Parameters passed to `brokenaxes()`.
                    If None, not draw broken axes. Default: None.
    **kwargs: other parameters passed to `matplotlib.pyplot.bar()` or `matplotlib.pyplot.boxplot()`."""
    plt.figure(figsize=fig_size)
    if brokenaxes_dict is None:
        ax = plt.gca()
    else:
        ax = brokenaxes(**brokenaxes_dict)
    if plot_box:
        x = [ind+1 for ind, _ in enumerate(clusters)]
        y = [tab[tab['cluster']==c][feature].dropna().values for c in clusters]
        box_prop=ax.boxplot(y, **kwargs)
    else:
        x = [ind for ind, _ in enumerate(clusters)]
        y = [np.nanmean(tab[tab['cluster']==c][feature]) for c in clusters]
        err = [sem(tab[tab['cluster']==c][feature], nan_policy='omit') for c in clusters]
        ax.bar(x, y, yerr=err, capsize=capsize, **kwargs)
    if xticks is None:
        xticks = ['cluster {}'.format(c+1) for c in clusters]
    plt.xticks(x, xticks)
    if ylabel is None:
        ylabel=feature
    plt.ylabel(ylabel)
    plt.tight_layout()
    if bool(to_save):
        to_save_figure(to_save)
    
def plot_cumulative_curve(tab, feature, clusters=(0,1), n_bins=50, ax=None,fig_size=(3,3), to_save="", **kwargs):
    """Plot cumulative density distribution curve."""
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
    x1 = tab[tab.cluster==clusters[0]][feature]
    x2 = tab[tab.cluster==clusters[1]][feature]
    range_ = tab[feature].min(), tab[feature].max()
    ax.hist(x1, n_bins, cumulative=True, histtype='step', range=range_, density=True)
    ax.hist(x2, n_bins, cumulative=True, histtype='step', range=range_, density=True)
    if bool(to_save):
        to_save_figure(to_save)

    
def plot_cluster_sholl(
    morpho_parser,
    cluster1_ids, cluster2_ids,
    clusters=(0,1),
    label_map=None,
    sholl_part=True,
    fig_size=(3,3),
    to_save="",
):
    """Plot sholl analysis of apical and basal parts between two clusters.
    morpho_parser: `morpho_parser` object.
    clusters: clusters to show in plot labels.
    cluster1_ids, cluster2_ids: neurons IDs of two clusters.
    """
    if not sholl_part:
        imarisData = morpho_parser.imarisData
        shollData_clust1 = imarisData[
            (imarisData.Variable == "Filament No. Sholl Intersections") & (imarisData.neuron_ID+".swc").isin(cluster1_ids)
        ].loc[:, ["neuron_ID", "Radius", "Value"]]
        shollData_clust1['cluster'] = clusters[0]
        shollData_clust1 = zero_padding(shollData_clust1, 'Value')
        shollData_clust2 = imarisData[
            (imarisData.Variable == "Filament No. Sholl Intersections") & (imarisData.neuron_ID+".swc").isin(cluster2_ids)
        ].loc[:, ["neuron_ID", "Radius", "Value"]]
        shollData_clust2['cluster'] = clusters[1]
        shollData_clust2 = zero_padding(shollData_clust2, 'Value')
        shollData = pd.concat(
            [shollData_clust1, shollData_clust2], sort=False,
            ignore_index=True
        )
        shollPlotData = (
            shollData
            .groupby(["cluster","Radius"])[['Value']]
            .agg([np.mean, sem])
        )
        plt.figure(figsize=fig_size)
        line1,=plt.plot(
            shollPlotData.loc[clusters[0],:].index,
            shollPlotData.loc[clusters[0],('Value','mean')],
        )
        line2,=plt.plot(
            shollPlotData.loc[clusters[1],:].index,
            shollPlotData.loc[clusters[1],('Value','mean')],
        )
        plt.fill_between(
            shollPlotData.loc[clusters[0],:].index,
            shollPlotData.loc[clusters[0],('Value', 'mean')]+shollPlotData.loc[clusters[0],('Value','sem')],
            shollPlotData.loc[clusters[0],('Value', 'mean')]-shollPlotData.loc[clusters[0],('Value','sem')],
            alpha=0.6,
        )
        p1=mpatch.Patch(color='C0', alpha=0.6)
        plt.fill_between(
            shollPlotData.loc[clusters[1],:].index,
            shollPlotData.loc[clusters[1],('Value', 'mean')]+shollPlotData.loc[clusters[1],('Value','sem')],
            shollPlotData.loc[clusters[1],('Value', 'mean')]-shollPlotData.loc[clusters[1],('Value','sem')],
            alpha=0.6,
        )
        p2=mpatch.Patch(color='C1', alpha=0.6)
        plt.ylabel('Sholl intersections')
        plt.xlabel('$Radius\ (\mu m)$')
        ax=plt.gca()
        if label_map is None:
            label_map={c:'clusters {}'.format(c+1) for c in clusters}
        plt.legend(
            ((line1, p1),(line2,p2)),
            (label_map[clusters[0]], label_map[clusters[1]])
        )
        if bool(to_save):
            to_save_figure(to_save)