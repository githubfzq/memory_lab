# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython


# %%
import autoreload
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# %% [markdown]
#  # Patch clamp analysis

# %%
import matplotlib.pyplot as plt
from patch_clamp_functions import *

# %% [markdown]
#  ## sEPSC demo trace

# %%
sEPSCparser = epsc_parser()
print(len(sEPSCparser.files["abf"]))

# %% [markdown]
#  ### representative trace

# %%
sEPSCparser.plot_demo_epsc(80)

# %% [markdown]
#  ## cumulative distribution

# %%
itiDemoData = sEPSCparser.read_demo_stat_data('iti', 0)

plt.plot(itiDemoData["Inter-event Interval (msec)"], itiDemoData["Cumulative Fraction"])
plt.xlabel("Inter-event Interval (msec)")
plt.ylabel("Cumulative Fraction")
# plt.savefig('cumulative_sEPSP_ITI',dpi=600)


# %%
ampDemoData = sEPSCparser.read_demo_stat_data('amp', 0)
plt.plot(ampDemoData["Amplitude (pA)"], ampDemoData["Cumulative Fraction"])
plt.xlabel("Amplitude (pA)")
plt.ylabel("Cumulative Fraction")
# plt.savefig('cumulative_sEPSP_amplitute',dpi=600)

# %% [markdown]
#  ## AP demo trace
# %% [markdown]
#  ### raw trace

# %%
ap_ps = ap_parser()
print(len(ap_ps.files['abf']))


# %%
ap_ps.plot_demo_ap(82)


# %%
ap_ps.plot_demo_ap(54, sweep=12)

# %% [markdown]
#  ### AP distribution

# %%
ap_ps = ap_parser()
fitDemoData = ap_ps.read_demo_fit_stat(0)
fitDemoData.head()


# %%
EvDemoData = ap_ps.read_demo_event_stat(0)
EvDemoData.head()


# %%
IFdemoData = ap_ps.get_demo_IF_data(0)
IFallData = ap_ps.get_all_data('IF')


# %%
ap_ps.plot_demo_IF(4)


# %%
IFstat = get_CV_of_IF(IFallData)


# %%
plot_IF_CV_distribution(IFstat)

# %% [markdown]
#  ### AP frequency

# %%
ap_ps.plot_demo_distribution('freq', 0)


# %%
apAllFreqData = ap_ps.get_all_data('freq')

# %% [markdown]
#  ### AP amplitude

# %%
ap_ps.plot_demo_distribution('amp', 12)

# %% [markdown]
#  ## time constant (tau)
# %% [markdown]
#  Read tau results

# %%
tauData = ap_ps.get_all_data('tau')
tauData.head()

# %% [markdown]
# ### filter outlier

# %%
ap_ps.is_outlier().head()

# %% [markdown]
# ### time constant statistic

# %%
ap_ps.plot_tau_distribution()

# %% [markdown]
#  ## I/V relationship
# %% [markdown]
#  ### plot I-V traces

# %%
ivAllData = ap_ps.get_all_data('iv')
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
from patch_clamp_functions import ramp_parser
ramp_ps = ramp_parser()


# %%
ramp_ps.plot_demo_ramp(20)

# %% [markdown]
#  # Morphological analysis

# %%
from morphological_functions import morpho_parser
mpp = morpho_parser()


# %%
mpp.plot_all_neurons()

# %% [markdown]
#  Put apical dendrites upside:

# %%
mpp.plot_apical_upside()

# %% [markdown]
#  ## Sholl analysis
# %% [markdown]
#  Analyse by neurom:

# %%
shollParts = mpp.get_sholl_parts_stat()
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
depthData=mpp.depthData
depthData.head()


# %%
mpp.plot_depth()

# %% [markdown]
#  # Cluster Analysis
# %% [markdown]
#  ## Electrophysiological Parameters
# %% [markdown]
#  Get all electrophysiological data

# %%
elec_ps=electro_parser()
electroParaData_py=elec_ps.get_all_data()
electroParaData_py.head()


# %%
electroParaData_loc = elec_ps.get_all_data(method='local')
electroParaData_loc.head()


# %%
electroParaData = elec_ps.get_all_data(method='all')
electroParaData.head()

# %% [markdown]
# ## Morphological Parameters

# %%
morphoData = mpp.get_all_data()
morphoData.head()

# %% [markdown]
#  ## analysis

# %%
from neuron_cluster_analysis import cluster_processor
analysor = cluster_processor(elec_ps,mpp)
cluster_data = analysor.get_all_data()
cluster_data.head()


# %%
cluster_data = cluster_data[cluster_data['cellID_ap'].notna()]

# %% [markdown]
#  ### K-Means
# %% [markdown]
# Analyze electrophysiological data and morphological data:

# %%
km = analysor.k_means()
km.head()


# %%
analysor.plot_cluster_scatter('Rm','tau')

# %% [markdown]
# Not well.
# %% [markdown]
# Only analyze electrophysiological data:

# %%
cluster_data_e = analysor.elec_ps.get_all_data(method="all")
cluster_data_e.head()


# %%
km_e = analysor.k_means(cluster_data_e, n_cluster=3)
km_e.head()

# %% [markdown]
# ### PCA scatter plot

# %%
pca_dat = analysor.k_means(cluster_data_e, n_cluster=3,return_filled=True, return_scaled=True)
analysor.plot_decomposition_scatter(ca_dat=pca_dat)


# %%
analysor.plot_decomposition_scatter(ca_dat=pca_dat,dim2=2)


# %%
analysor.plot_decomposition_scatter(ca_dat=pca_dat,dim1=1,dim2=2)


# %%
analysor.plot_decomposition_scatter(ca_dat=pca_dat, plot_3d=True)

# %% [markdown]
# ### Inter-class comparation

# %%
inter_class_stat_e=analysor.stat_analyse(km_e.drop(columns='cellID_ap'))
inter_class_stat_e


# %%
inter_class_stat_e[inter_class_stat_e.p_value<0.05].variable.to_list()


# %%
from neuron_cluster_analysis import plot_cluster_hist

plot_cluster_hist(
    km_e, "Rm", xlabel="Rm " + "$(M\Omega)$", 
    plot_cumulative=True, legend_kw={"loc": 4}
)


# %%
plot_cluster_hist(
    km_e, "tau", xlabel="time constant (ms)", 
    plot_cumulative=True, legend_kw={"loc": 2}
)


# %%
clust1_id = km_e[km_e.cluster == 0].cellID_ap.values
clust2_id = km_e[km_e.cluster == 1].cellID_ap.values


# %%
from neuron_cluster_analysis import select_demo_cell_id
id_select = select_demo_cell_id(km_e, 'ISI_log_slope')
analysor.plot_demo_ap(sweep=13, cell_id=id_select[0])


# %%
analysor.plot_demo_ap(sweep=13, cell_id=id_select[1])


# %%
from neuron_cluster_analysis import plot_cluster_stat
plot_cluster_stat(km_e, 'ISI_log_slope', alpha=.8)


# %%
plot_cluster_stat(km_e, 'mean_AHP_depth_abs_slow', alpha=0.8)


# %%
plot_cluster_stat(km_e, 'mean_AHP_time_from_peak', alpha=0.8)


# %%
plot_cluster_stat(km_e, 'steady_state_voltage_stimend', alpha=0.8)

# %% [markdown]
# Plot ramp current of cluster 1

# %%
analysor.elec_ps.plot_demo_ramp(cell_id='19n11001', sweep=2)

# %% [markdown]
# Plot ramp current of cluster 2

# %%
analysor.elec_ps.plot_demo_ramp(cell_id='19n14006', sweep=2)

# %% [markdown]
# ### Morphological classifier

# %%


