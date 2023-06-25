# ---
# jupyter:
#   jupytext:
#     comment_magics: false
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import autoreload

%load_ext autoreload
%autoreload 2

# %%
import dill

with open('session_data.pkl','rb') as f:
    mpp = dill.load(f)
    shollParts = dill.load(f)
    elec_ps = dill.load(f)
    electroParaData_py = dill.load(f)
    morphoData = dill.load(f)
    pred_mpp = dill.load(f)
    gfp_morpho_dat = dill.load(f)
    pred_mpp2 = dill.load(f)
    control_morpho_dat = dill.load(f)

# %%
import matplotlib as mpl

mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False

# %% [markdown] toc-hr-collapsed=true toc-hr-collapsed=true toc-hr-collapsed=true toc-hr-collapsed=true toc-hr-collapsed=true toc-hr-collapsed=true toc-hr-collapsed=true toc-hr-collapsed=true toc-hr-collapsed=true toc-hr-collapsed=true toc-hr-collapsed=true toc-hr-collapsed=true toc-hr-collapsed=true toc-hr-collapsed=true toc-hr-collapsed=true toc-hr-collapsed=true toc-hr-collapsed=true toc-hr-collapsed=true toc-hr-collapsed=true toc-hr-collapsed=true toc-hr-collapsed=true toc-hr-collapsed=true toc-hr-collapsed=true toc-hr-collapsed=true toc-hr-collapsed=true toc-hr-collapsed=true toc-hr-collapsed=true
#  # Patch clamp analysis

# %%
import matplotlib.pyplot as plt

from patch_clamp_functions import *

# %% [markdown] toc-hr-collapsed=true toc-hr-collapsed=true
#  ## sEPSC demo trace

# %%
sEPSCparser = epsc_parser()
print(len(sEPSCparser.files["abf"]))

# %% [markdown]
#  ### representative trace

# %%
sEPSCparser.plot_demo_epsc(80)

# %% [markdown] toc-hr-collapsed=true toc-hr-collapsed=true toc-hr-collapsed=true
#  ## cumulative distribution

# %%
itiDemoData = sEPSCparser.read_demo_stat_data("iti", 0)

plt.plot(itiDemoData["Inter-event Interval (msec)"], itiDemoData["Cumulative Fraction"])
plt.xlabel("Inter-event Interval (msec)")
plt.ylabel("Cumulative Fraction")

# %%
ampDemoData = sEPSCparser.read_demo_stat_data("amp", 0)
plt.plot(ampDemoData["Amplitude (pA)"], ampDemoData["Cumulative Fraction"])
plt.xlabel("Amplitude (pA)")
plt.ylabel("Cumulative Fraction")
# plt.savefig('cumulative_sEPSP_amplitute',dpi=600)

# %% [markdown] toc-hr-collapsed=true toc-hr-collapsed=true toc-hr-collapsed=true
#  ## AP demo trace

# %% [markdown]
#  ### raw trace

# %%
ap_ps = ap_parser()
print(len(ap_ps.files["abf"]))

# %%
ap_ps.plot_demo_ap(82, sweepYcolor="C0", sweepCcolor="C1")

# %%
ap_ps.plot_demo_ap(54, sweep=12, sweepYcolor="C0", sweepCcolor="C1")

# %% [markdown]
#  ### AP distribution

# %%
fitDemoData = ap_ps.read_demo_fit_stat(0)
fitDemoData.head()

# %%
EvDemoData = ap_ps.read_demo_event_stat(0)
EvDemoData.head()

# %%
IFdemoData = ap_ps.get_demo_IF_data(0)
IFallData = ap_ps.get_all_data("IF")

# %%
ap_ps.plot_demo_IF(4)

# %%
IFstat = get_CV_of_IF(IFallData)

# %%
plot_IF_CV_distribution(IFstat)

# %% [markdown]
#  ### AP frequency

# %%
ap_ps.plot_demo_distribution("freq", 0)

# %%
apAllFreqData = ap_ps.get_all_data("freq")

# %% [markdown]
#  ### AP amplitude

# %%
ap_ps.plot_demo_distribution("amp", 12)

# %% [markdown] toc-hr-collapsed=true toc-hr-collapsed=true
#  ## time constant (tau)

# %% [markdown]
#  Read tau results

# %%
tauData = ap_ps.get_all_data("tau")
tauData.head()

# %% [markdown]
# ### filter outlier

# %%
ap_ps.is_outlier().head()

# %% [markdown]
# ### time constant statistic

# %%
ap_ps.plot_tau_distribution()

# %% [markdown] toc-hr-collapsed=true toc-hr-collapsed=true
#  ## I/V relationship

# %% [markdown]
#  ### plot I-V traces

# %%
ivAllData = ap_ps.get_all_data("iv")
ivAllData.head()

# %%
plot_iv_traces(ivAllData, to_save="IV_trace")

# %% [markdown]
#  ### Compute membrane resistance

# %%
RmRes = compute_Rm(ivAllData)
RmRes.head()

# %%
plot_Rm_distribution(RmRes, to_save="Rm_hist")

# %% [markdown]
#  ## ramp current triggered voltage

# %%
ramp_ps = ramp_parser()

# %%
ramp_ps.plot_demo_ramp(20)

# %% [markdown] toc-hr-collapsed=true toc-hr-collapsed=true toc-hr-collapsed=true toc-hr-collapsed=true toc-hr-collapsed=true toc-hr-collapsed=true toc-hr-collapsed=true
#  # Morphological analysis

# %%
from morphological_functions import morpho_parser

if 'mpp' not in locals():
    mpp = morpho_parser()

# %%
from morphological_functions import plot_sholl_demo

plot_sholl_demo(mpp.neurons[7], to_save='Sholl_demo')
# %%
mpp.plot_all_neurons()

# %% [markdown]
#  Put apical dendrites upside:

# %%
mpp.plot_apical_upside(to_save='all neurons')

# %% [markdown]
#  ## Sholl analysis

# %% [markdown]
#  Analyse by neurom:

# %%
if 'shollParts' not in locals():
    shollParts = mpp.get_sholl_parts_stat()  # time consuming
shollParts.head()

# %%
mpp.plot_sholl(sholl_part=True)

# %% [markdown]
#  Analyse by morphological data exported by Imaris:

# %%
morphoData = mpp.imarisData
morphoData.head()

# %%
mpp.plot_sholl(sholl_part=False)

# %% [markdown]
#  ## angle distribution

# %% [markdown]
#  ## branch order

# %%
depthData = mpp.depthData
depthData.head()

# %%
mpp.plot_depth()

# %% [markdown] toc-hr-collapsed=true
#  # Cluster Analysis

# %% [markdown]
#  ## Electrophysiological Parameters

# %% [markdown]
#  Get all electrophysiological data

# %%
if 'elec_ps' not in locals():
    elec_ps = electro_parser()
if 'electroParaData_py' not in locals():
    electroParaData_py = elec_ps.get_all_data()  # time consuming
electroParaData_py.head()

# %%
electroParaData_loc = elec_ps.get_all_data(method="local")
electroParaData_loc.head()

# %%
electroParaData = elec_ps.get_all_data(method="all")
electroParaData.head()

# %% [markdown]
# ## Morphological Parameters

# %%
if 'morphoData' not in locals():
    morphoData = mpp.get_all_data()  # time consuming
morphoData.head()

# %% [markdown]
#  ## analysis

# %%
from neuron_cluster_analysis import cluster_processor

analysor = cluster_processor(elec_ps, mpp)
cluster_data = analysor.get_all_data("inner")
cluster_data.head()

# %% [markdown]
#  ### K-Means

# %% [markdown]
# Analyze electrophysiological data and morphological data:

# %%
km = analysor.k_means()
km.head()

# %%
analysor.plot_cluster_scatter("Rm", "tau")

# %% [markdown]
# Not well.

# %% [markdown]
# Only analyze electrophysiological data:

# %%
cluster_data_e = analysor.elec_ps.get_all_data(method="all")
cluster_data_e.head()

# %% [markdown]
# Filter neuron with RMP<-40mV:

# %%
cluster_data_e=cluster_data_e[cluster_data_e.RMP<-40]

# %%
km_e = analysor.k_means(cluster_data_e, n_cluster=3, random_state=15)
km_e.head()

# %% [markdown]
# ### PCA scatter plot

# %%
pca_dat = analysor.k_means(
    cluster_data_e, n_cluster=3, return_filled=True, random_state=15, return_scaled=True
)
analysor.plot_decomposition_scatter(
    ca_dat=pca_dat, to_save="PCA_dim1_2", show_legend=False
)

# %%
analysor.plot_decomposition_scatter(
    ca_dat=pca_dat, dim2=2, to_save="PCA_dim1_3", show_legend=False
)

# %%
analysor.plot_decomposition_scatter(
    ca_dat=pca_dat, dim1=1, dim2=2, to_save="PCA_dim2_3", show_legend=False
)

# %%
analysor.plot_decomposition_scatter(
    ca_dat=pca_dat,
    to_save="PCA_dim123",
    plot_3d=True,
    label_map={0: "ET-1", 1: "ET-2", 2: "ET-3"},
)

# %% [markdown]
# ### Inter-class comparation

# %%
inter_class_stat_e = analysor.stat_analyse(
    km_e.drop(columns="cellID_ap"),
    #     test_small_sample=True,
    group1=0,
    group2=1,
)
inter_class_stat_e

# %% [markdown]
# Which parameter show significant:

# %%
inter_class_stat_e[inter_class_stat_e.p_value < 0.05].variable.to_list()

# %%
from neuron_cluster_analysis import plot_cluster_hist

plot_cluster_hist(
    km_e,
    "Rm",
    xlabel="Rm " + "$(M\Omega)$",
    group1=0,
    group2=1,
    to_save="Rm_clusters",
    plot_cumulative=True,
    #     legend_kw={"loc": 4}
    show_legend=False,
)

# %%
plot_cluster_hist(
    km_e,
    "RMP",
    xlabel="rest membrane potential (mV)",
    group1=0,
    group2=1,
    plot_cumulative=True,
    to_save="RMP_clusters",
    #     legend_kw={"loc": 5},
    show_legend=False,
)

# %%
plot_cluster_hist(
    km_e,
    "inv_second_ISI",
    xlabel="inverse of\n second ISI (Hz)",
    group1=0,
    group2=1,
    plot_cumulative=True,
    #     legend_kw={"loc": 5}
    show_legend=False,
    fig_size=(2, 2),
    to_save="inv_second_ISI_hist",
)

# %%
plot_cluster_hist(
    km_e,
    "ISI_log_slope",
    xlabel="ISI log slope",
    #     group1=0, group2=2,
    plot_cumulative=True,
    #     legend_kw={"loc": 4},
    show_legend=False,
    fig_size=(2, 2),
    to_save="ISI_log_slope",
)

# %%
clust1_id = km_e[km_e.cluster == 0].cellID_ap.values
clust2_id = km_e[km_e.cluster == 1].cellID_ap.values

# %%
from neuron_cluster_analysis import select_demo_cell_id

id_select = select_demo_cell_id(km_e, "inv_first_ISI")
analysor.plot_demo_ap(
    sweep=13, cell_id=id_select[0], to_save="demo_ET1_ap", fig_size=(3, 1.5)
)

# %%
analysor.plot_demo_ap(
    sweep=13,
    cell_id=id_select[1],
    to_save="demo_ET2_ap",
    sweepYcolor="C1",
    sweepCcolor="C1",
    fig_size=(3, 1.5),
)

# %%
from neuron_cluster_analysis import plot_cluster_stat

plot_cluster_stat(
    km_e,
    "Freq (Hz)",
    plot_box=True,
    clusters=(0, 1),
    xticks=["ET-1", "ET-2"],
    ylabel="Freq (Hz)",
    to_save="Freq_clusters",
)

# %%
from neuron_cluster_analysis import plot_cluster_stat

plot_cluster_stat(
    km_e,
    "inv_first_ISI",
    plot_box=True,
    xticks=["ET-1", "ET-2"],
    ylabel="inverse of\n time to first ISI (Hz)",
    to_save="inv_first_ISI_clusters",
)

# %%
plot_cluster_stat(
    km_e,
    "inv_second_ISI",
    plot_box=True,
    xticks=["ET-1", "ET-2"],
    ylabel="inverse of\n second ISI (Hz)",
    to_save="inv_second_ISI_clusters",
)

# %%
plot_cluster_stat(
    km_e,
    "n",
    plot_box=True,
    xticks=["ET-1", "ET-2"],
    ylabel="No. spike",
    to_save="spike_n_clusters",
)

# %%
plot_cluster_stat(
    km_e,
    "AP1_amp",
    plot_box=True,
    xticks=["ET-1", "ET-2"],
    ylabel="AP1 amplitude (mV)",
    to_save="AP1_amp_clusters",
)

# %%
plot_cluster_stat(
    km_e,
    "ISI_log_slope",
    plot_box=True,
    xticks=["ET-1", "ET-2"],
    ylabel="ISI log slope",
    to_save="ISI_log_slope_clusters",
)

# %%
plot_cluster_stat(
    km_e,
    "ISI_semilog_slope",
    plot_box=True,
    xticks=["ET-1", "ET-2"],
    ylabel="ISI semilog slope",
    to_save="ISI_semilog_slope_clusters",
)

# %%
plot_cluster_stat(
    km_e,
    "mean_AP_width",
    xticks=["ET-1", "ET-2"],
    ylabel="mean of AP width (ms)",
    to_save="mean_AP_width_clusters",
    plot_box=True
)

# %%
plot_cluster_stat(
    km_e,
    "mean_AHP_depth_abs",
    xticks=["ET-1", "ET-2"],
    ylabel="mean of AHP depth (mV)",
    to_save="mean_AHP_depth_abs_clusters",
    plot_box=True,
)

# %% [markdown]
# Plot ramp current of cluster 1

# %%
to_plot_id = cluster_data_e[cluster_data_e.cellID_ap==clust1_id[-2]].cellID_ramp.values[0]
analysor.elec_ps.plot_demo_ramp(cell_id=to_plot_id, sweep=2, to_save="demo_ET1_ramp")

# %% [markdown]
# Plot ramp current of cluster 2

# %%
to_plot_id = cluster_data_e[
    cluster_data_e.cellID_ap==clust2_id[-2]
].cellID_ramp.values[0]
analysor.elec_ps.plot_demo_ramp(
    cell_id=to_plot_id,
    sweep=2,
    to_save="demo_ET2_ramp",
    sweepYcolor="C1",
    sweepCcolor="C1",
)

# %%
plot_cluster_stat(
    km_e,
    "threshold_I",
    xticks=["ET-1", "ET-2"],
    ylabel="rheobase (pA)",
    to_save="rheobase_clusters",
    plot_box=True,
)

# %%
plot_cluster_stat(
    km_e,
    "threshold_V",
    xticks=["ET-1", "ET-2"],
    ylabel="threshold voltage (mV)",
    to_save="threshold_V_clusters",
    plot_box=True,
)

# %% [markdown]
# ### Morphological classifier

# %% [markdown]
# Prepare morphology dataset for SVM training:

# %%
from neuron_cluster_analysis import k_means_preprocess

train_dat = analysor.get_morpho_cluster_label(km_e, cluster_data)
morpho_train = train_dat.set_index("cluster")["reconstruction_ID"]
train_dat = k_means_preprocess(train_dat)
train_dat = train_dat.drop(columns="cellID_ap").fillna(train_dat.mean())
train_dat.head()

# %%
svm_f = analysor.svm_train(train_dat)
y_test = svm_f(train_dat.drop(columns="cluster"))
correct = y_test == train_dat.cluster.values
print("Correct prediction rate: {}".format(correct.sum() / correct.size))

# %%
train_c1 = [n for n in mpp.neurons if n.name in morpho_train[0].values]
mpp.plot_apical_upside(train_c1, to_save="train_ET1")

# %%
train_c2 = [n for n in mpp.neurons if n.name in morpho_train[1].values]
mpp.plot_apical_upside(train_c2, to_save="train_ET2")

# %% [markdown]
# ### predict new morphology

# %% [markdown]
# #### GFP+ neurons

# %%
if 'pred_mpp' not in locals():
    pred_mpp = morpho_parser(
        {
            "swc": "../../../reconstruction/predict/traces/GFP+_repair",
            "imaris": "../../../reconstruction/predict/stat/GFP+",
        }
    )

# %%
to_plot_neurons = list(pred_mpp.neurons[1:])+[pred_mpp.neurons[0]]
pred_mpp.plot_apical_upside(to_plot_neurons,to_save="GFP+ neurons")

# %%
if 'gfp_morpho_dat' not in locals():
    gfp_morpho_dat = pred_mpp.get_all_data()  # time consuming
gfp_morpho_dat.head()

# %%
gfp_cluster = svm_f(gfp_morpho_dat)
gfp_morpho_dat["cluster"] = gfp_cluster
gfp_morpho_dat.head()

# %%
gfp_morpho_inter = analysor.stat_analyse(
    gfp_morpho_dat.drop(columns="reconstruction_ID"),
    group1=0,
    group2=1,
    test_small_sample=True,
)
gfp_morpho_inter[gfp_morpho_inter.p_value < 0.05].variable

# %%
gfp_c1_neurons = [
    n
    for n in pred_mpp.neurons.neurons
    if n.name in gfp_morpho_dat[gfp_morpho_dat.cluster == 0].reconstruction_ID.values
]
gfp_c2_neurons = [
    n
    for n in pred_mpp.neurons.neurons
    if n.name in gfp_morpho_dat[gfp_morpho_dat.cluster == 1].reconstruction_ID.values
]

# %%
to_plot_neurons = list(gfp_c1_neurons[1:])+[gfp_c1_neurons[0]]
pred_mpp.plot_apical_upside(to_plot_neurons, to_save="GFP+ ET1", fig_size=(7.5, 5.5))

# %%
mpp.plot_apical_upside(gfp_c2_neurons, to_save="GFP+ ET-2")

# %%
plot_cluster_stat(
    gfp_morpho_dat,
    "mean_neurite_lengths(apical)",
    ylabel="mean apical neurite\n length (" + "$\mu m)$",
    clusters=(0, 1),
    xticks=["ET-1", "ET-2"],
    plot_box=True,
    to_save="mean_neurite_lengths(apical)_gfp_clusters",
)

# %%
plot_cluster_stat(
    gfp_morpho_dat,
    "Filament Dendrite Length (sum)",
    ylabel="total dendrite\n length (" + "$\mu m)$",
    clusters=(0, 1),
    xticks=["ET-1", "ET-2"],
    plot_box=True,
    to_save="dendrite_lengths_gfp_clusters",
)

# %%
plot_cluster_stat(
    gfp_morpho_dat,
    "Filament Full Branch Depth",
    ylabel="full branch depth",
    clusters=(0, 1),
    xticks=["ET-1", "ET-2"],
    plot_box=True,
    to_save="branch_depth_gfp_clusters",
)

# %%
plot_cluster_stat(
    gfp_morpho_dat,
    "mean_neurite_volume_density(apical)",
    ylabel="mean neurite\n volume density (apical)",
    clusters=(0, 1),
    xticks=["ET-1", "ET-2"],
    plot_box=True,
    to_save="mean_neurite_volume_density(apical)_gfp_clusters",
)

# %%
plot_cluster_stat(
    gfp_morpho_dat,
    "max_sholl_intercept_radii",
    ylabel="Radius with max\n Sholl intersection " + "$(\mu m)$",
    clusters=(0, 1),
    xticks=["ET-1", "ET-2"],
    plot_box=True,
    to_save="max_sholl_intercept_radii_gfp_clusters",
)

# %% [markdown]
# Sholl intersection of two clusters:

# %%
from neuron_cluster_analysis import plot_cluster_sholl

plot_cluster_sholl(
    pred_mpp,
    [c.name for c in gfp_c1_neurons],
    [c.name for c in gfp_c2_neurons],
    clusters=(0, 1),
    sholl_part=False,
    label_map={0: "GFP+:ET-1", 1: "GFP+:ET-2"},
    to_save="GFP+ c1_c2 sholl",
)

# %% [markdown]
# #### GFP- neurons

# %%
if 'pred_mpp2' not in locals():
    pred_mpp2 = morpho_parser(
        {
            "swc": "../../../reconstruction/predict/traces/GFP-_repair",
            "imaris": "../../../reconstruction/predict/stat/GFP-",
        }
    )

# %%
pred_mpp2.plot_all_neurons()

# %%
pred_mpp2.plot_apical_upside(to_save='GFP- neurons')

# %%
if 'control_morpho_dat' not in locals():
    control_morpho_dat = pred_mpp2.get_all_data()  # time consuming
control_morpho_dat.head()

# %%
control_cluster = svm_f(control_morpho_dat)
control_morpho_dat["cluster"] = control_cluster
control_morpho_dat.head()

# %%
control_c1_neurons = [
    n
    for n in pred_mpp2.neurons.neurons
    if n.name
    in control_morpho_dat[control_morpho_dat.cluster == 0].reconstruction_ID.values
]
control_c2_neurons = [
    n
    for n in pred_mpp2.neurons.neurons
    if n.name
    in control_morpho_dat[control_morpho_dat.cluster == 1].reconstruction_ID.values
]

# %%
pred_mpp2.plot_apical_upside(control_c1_neurons)

# %%
pred_mpp2.plot_apical_upside(control_c2_neurons)

# %% [markdown]
# Save partial data:

# %%
with open('session_data.pkl','wb') as f:
    dill.dump(mpp, f)
with open('session_data.pkl','ab') as f:
    dill.dump(shollParts, f)
    dill.dump(elec_ps, f)
    dill.dump(electroParaData_py, f)
    dill.dump(morphoData, f)
    dill.dump(pred_mpp, f)
    dill.dump(gfp_morpho_dat, f)
    dill.dump(pred_mpp2, f)
    dill.dump(control_morpho_dat, f)
# %%
