import os.path
import re
import warnings
import numpy as np
import pandas as pd
from scipy.stats import variation
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
import pyabf
import efel
from analysis_functions import (
    getAllFiles,
    getCellID,
    getFileName,
    merge_dicts,
    get_cellID_info,
    fillna_unique
)
from plot_helper import add_scalebar, get_aspect_scale, to_save_figure


class electro_base(object):
    """default parameters:
        path_root = {'abf':None, 'amp':None}
        files = {'abf':None, 'amp':None}"""

    # Cache `get_id_table()` function
    id_table = None

    def __init__(self):
        self.__reset_parameter()

    def get_parameter(self, item):
        """Get parameter of current class.
        item: {'path_root','files'}"""
        if item == "path_root":
            return self.path_root
        elif item == "files":
            return self.files

    def get_id_table(self):
        """get recording date info and cell_ID info for each abf file, according to its path."""
        if self.id_table is None:
            _map = map(get_cellID_info, self.files["abf"])
            result = pd.DataFrame(
                _map, columns=["cellID", "recording_date", "recording_cell_id"]
            )
            result = result.drop_duplicates()
        else:
            result = self.id_table
        return result
    
    def get_file_id(self, cell_id):
        """Return the index of self.files['abf'] for `cell_id`"""
        for ind, f in enumerate(self.files['abf']):
            if cell_id in f:
                return ind


    def __reset_parameter(self):
        self.path_root = {"abf": None, "amp": None}
        self.files = {"abf": None, "amp": None}


class epsc_parser(electro_base):
    def __init__(self, path_root=""):
        """EPSC parser
        path_root: a dict of abf, iti, amp path root 
        -abf file: default is "../../mini EPSC"
        -iti file: default is "../../process/mEPSC_interEvent/"
        -amp file: default is "../../process/mEPSC_amplitude_data/"
        abfFiles: all EPSC abf files in path_rort
        """
        electro_base.__init__(self)
        if not (path_root):
            self.path_root = {
                "abf": "../../mini EPSC/",
                "iti": "../../process/mEPSC_interEvent/",
                "amp": "../../process/mEPSC_amplitude_data/",
            }
        else:
            self.path_root = path_root
        self.files["abf"], _ = getAllFiles(self.path_root["abf"])
        self.files["iti"], _ = getAllFiles(self.path_root["iti"])
        self.files["amp"], _ = getAllFiles(self.path_root["amp"])

    def plot_demo_epsc(self, file_id, to_save=""):
        """Input EPSC file path, plot demo figure.
        file_id:  The n-th EPSC abf file selected as demo;
        to_save: Specify the filename to save as 600 dpi figure.
        Default is empty string which means not to save.
        """
        mDemo = pyabf.ABF(self.files["abf"][file_id])

        plt.figure(figsize=(10, 3))
        plt.plot(mDemo.sweepX, mDemo.sweepY, color="C0", alpha=0.8)
        plt.ylabel(mDemo.sweepLabelY)
        plt.xlabel(mDemo.sweepLabelX)
        plt.axis("off")

        ax = plt.gca()
        axin = zoomed_inset_axes(
            ax, 1, 4,
            axes_kwargs={
                "xlabel": "10 s",
                "ylabel": "10 pA",
                "xticks": [],
                "yticks": [],
                "xlim": (0, 10),
                "ylim": (0, 10)
            },
        )
        axin.spines["top"].set_visible(False)
        axin.spines["right"].set_visible(False)
        if to_save:
            to_save_figure(to_save)

    def read_demo_stat_data(self, item, file_id):
        """item : String: 'iti' or 'amp'. Read iti stat data or amplitute stat data.
        file_id: Integral. The n-th file to read.
        """
        to_read = self.files[item][file_id]
        return pd.read_csv(to_read, skiprows=3, sep="\t")


class ap_parser(electro_base):
    def __init__(self, path_root=""):
        """AP parser.
        path_root: dict of string. Keys include: 'abf', 'fit', 'event', 'freq', 'amp', 'tau', 'iv'.
        A file named 'demo.csv' must exist in fit/event file root for reading title.
        """
        electro_base.__init__(self)
        if not (path_root):
            self.path_root = {
                "abf": "../../AP",
                "fit": "../../process/AP_fit/",
                "event": "../../process/AP_event/",
                "freq": "../../process/AP_freq_data/",
                "amp": "../../process/AP_amplitude_data/",
                "tau": "../../process/AP_tau/",
                "iv": "../../process/IV_data",
            }
        else:
            self.path_root = path_root
        for term in ["abf", "fit", "event", "freq", "amp"]:
            self.files[term], _ = getAllFiles(self.path_root[term])
        for term in ["tau", "iv"]:
            self.files[term], _ = getAllFiles(self.path_root[term], ends="atf")

        # remove redundant files
        self.__title_file = {
            "fit": os.path.join(self.path_root["fit"], "demo.csv"),
            "event": os.path.join(self.path_root["event"], "demo.csv"),
        }
        self.__title = dict()
        for item in ["fit", "event"]:
            if self.__title_file[item] in self.files[item]:
                self.files[item].remove(self.__title_file[item])
                self.__title[item] = self.__readTitle(item)
    
    def plot_demo_ap(self, file_id=None, to_save="", sweep="all", cell_id=None, fig_size=(3,3), aspect_ratio=150, sweepYcolor='C0',
                    sweepCcolor='C0'):
        """plot AP demo trace from a file.
        file_id: Interger. The n-th file in root_path.
        to_save: String. The figure to be saved. Default: empty (not to save).
        sweep: String or interger. The n-th sweep to be plotted. 
        aspect_ratio: y-axis scale ratio of voltage and current.
        sweepYcolor, sweepCcolor: the color of sweepY axes and sweepC axes.
        Default: 'all', all sweeps will be plotted."""
        if cell_id is not None:
            file_id = self.get_file_id(cell_id)
        fig = plt.figure(figsize=(3,3))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        demo = pyabf.ABF(self.files["abf"][file_id])
        if sweep == "all":
            for sweep in demo.sweepList:
                demo.setSweep(sweep)
                ax1.plot(demo.sweepX[500:5500], demo.sweepY[500:5500], sweepYcolor)
                ax2.plot(demo.sweepX[500:5500], demo.sweepC[500:5500], sweepCcolor)
        else:
            if not isinstance(sweep,int):
                sweep = int(sweep)
            demo.setSweep(sweep-1)
            ax1.plot(demo.sweepX[500:5500], demo.sweepY[500:5500], sweepYcolor)
            ax2.plot(demo.sweepX[500:5500], demo.sweepC[500:5500], sweepCcolor)
        
        plt.draw()
        asp1=get_aspect_scale(ax1)
        ax2.set_aspect(asp1[1]*aspect_ratio/asp1[0])
        add_scalebar(
            ax2, (.1, .2), ('s', 'nA'), fig, 
            y_label='200pA/\n30mV')
        ax1.axis("off")
        ax2.axis("off")
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            plt.tight_layout()
        if to_save:
            to_save_figure(to_save)

    def __readTitle(self, item):
        "read fit/event demo csv title"
        with open(self.__title_file[item], "r") as f:
            title = f.readline().split(",")
        for ind, txt in enumerate(title):
            if txt == "" or txt == "\n":
                title[ind] = "col_" + str(ind + 1)
        return title

    def read_demo_fit_stat(self, file_id):
        """Read fit statistic data from a fit file.
        file_id: Integer. The n-th fit file to be read."""
        file = self.files["fit"][file_id]
        fitDemoData = pd.read_csv(file, sep="\t", names=self.__title["fit"])
        new = fitDemoData["Identifier"].str.extract(
            "(?P<filename>.*abf) G(?P<sweep>\d+) T (?P<AP_ID>\d+) AP"
        )
        rlt = pd.concat([new, fitDemoData.drop("Identifier", axis=1)], axis=1)
        rlt[["sweep", "AP_ID"]] = rlt[["sweep", "AP_ID"]].astype("int64")
        return rlt

    def read_demo_event_stat(self, file_id):
        file = self.files["event"][file_id]
        dat = pd.read_csv(file, sep="\t", names=self.__title["event"])
        identifier = os.path.split(file)[-1].split(".")[0]
        dat["Identifier"] = identifier
        dat.drop(self.__title["event"][0], axis=1, inplace=True)
        dat["Time (ms)"] = dat["Time (ms)"].str.replace(",", "").astype("float64")
        return dat

    def read_demo_stat(self, term, file_ID):
        """Read statistic data from txt file generated by mini.
        term: String. Include 'freq', 'amp'.
        file_ID: demo file order."""
        if term in ["freq", "amp"]:
            filepath = self.files[term][file_ID]
            dt = pd.read_csv(filepath, sep="\t", skiprows=1)
            dt.dropna(how="all", inplace=True)
            dt.rename({"Unnamed: 0": "sweep"}, axis=1, inplace=True)
            dt.sweep = dt.sweep.str.extract("Group # (\d+)").astype("int64")
            dt.n = dt.n.astype("int64")
            dt["CellID"] = getCellID(filepath)
            dt = dt[np.roll(dt.columns, 1)]
            return dt

    def match_file(self, item1, item2):
        """Match pairwise files for 'abf', 'iti', etc.
        item1, item2: String. one of 'abf', 'iti', 'event', 'fit', 'freq', 'amp';
        but exclude 'tau', 'iv'.
        return a map of matched indexes of item2, for each item1.
        map: key: files[item1].index-->values: files[item2].index_array
        """
        items = ["abf", "iti", "event", "fit", "freq", "amp"]
        if (item1 in items) & (item2 in items):
            identifiers = pd.Series(self.files[item1]).apply(getCellID)
            matched = pd.Series(self.files[item2])
            _map = {}
            for ind in range(len(identifiers)):
                _map[ind] = matched.index.values[matched.str.contains(identifiers[ind])]
            return _map

    def get_matched_file(self, item1, item2, file_id=0, cell_id=""):
        """Match pairwise file for 'abf', 'iti', 'event', 'fit', 'freq', 'amp',
        return actual path instead of a map. Input either file_id in item1 or cell_id.
        file_id: interger. The n-th file in item1 group.
        cell_id: string. A cell id what item1_files contains.
        See `ap_parser.match_file()`"""
        if not (cell_id):
            identifier = getCellID(self.files[item1][file_id])
        else:
            identifier = cell_id
        matched = pd.Series(self.files[item2])
        result = matched[matched.str.contains(identifier)].values
        if len(result) == 0:
            return ""
        elif len(result) >= 2:
            return result.tolist()
        else:
            return result[0]

    def get_current_step(self, file_id=0, file=""):
        """Returns the current step of the n-th abf file."""
        if not (file):
            file = self.files["abf"][file_id]
            check = self.check_abf_valid(file_id)
        else:
            check = self.check_abf_valid(file=file)
        apDemoAbf = pyabf.ABF(file)
        cur_step = []
        if check:
            for sweep in apDemoAbf.sweepList:
                apDemoAbf.setSweep(sweep)
                cur_step.append(apDemoAbf.sweepEpochs.levels[2])
        return cur_step

    def get_demo_IF_data(self, file_id):
        """Compute instantaneous frequency data from the n-th event file"""
        dat = self.read_demo_event_stat(file_id)
        rst = dat[["Identifier", "Time (ms)", "Group"]].groupby("Group")
        rst = (
            rst.apply(lambda x: x.sort_values(by="Time (ms)"))
            .drop("Group", axis=1)
            .reset_index()
            .drop("level_1", axis=1)
        )
        rst["InterEventTime (ms)"] = (
            rst.groupby("Group")
            .apply(lambda x: pd.Series(np.insert(np.diff(x["Time (ms)"]), 0, 0)))
            .values.flatten()
        )
        rst["instantaneous_freq (Hz)"] = rst["InterEventTime (ms)"].transform(
            lambda s: 1000 / s if s != 0 else 0
        )
        rst["relative_time"] = (
            rst.groupby("Group")
            .apply(lambda x: x["Time (ms)"] - min(x["Time (ms)"]))
            .values.flatten()
        )
        matched = self.get_matched_file("event", "abf", file_id)
        if matched:
            if isinstance(matched, list):
                # only  get the first matched if  multiple matched
                current = self.get_current_step(file=matched[0])
            else:
                current = self.get_current_step(file=matched)
            rst["current"] = rst["Group"].transform(
                lambda x: current[x - 1] if x != 255 else None
            )
            return rst

    def get_all_data(self, item):
        """Concantenate statistic data in term of instant frequency, AP frequency & amplitute, time constant, I-V data.
        item: String. 'IF', 'freq', 'amp', 'tau', 'iv'.
        return pandas dataframe."""
        if item == "IF":
            result = pd.concat(
                map(self.get_demo_IF_data, range(len(self.files["event"])))
            )
        elif item in ["freq", "amp"]:
            result = pd.concat(
                map(
                    lambda x: self.read_demo_stat(item, x), range(len(self.files[item]))
                )
            )
        elif item in ["tau", "iv"]:
            result = []
            for file in self.files[item]:
                dat_tmp = read_atf(file)
                result.append(dat_tmp)
            result = pd.concat(result, sort=False)
            result.insert(1, "CellID", result["File Name"].str.split(".").str[0])
            if item == "tau":
                result[["A", "tau", "C"]] = result[["A", "tau", "C"]].astype("float")
            elif item == "iv":
                result.rename(columns={"S1R1 Mean (mV)": "Vm"}, inplace=True)
                result.Vm = pd.to_numeric(result.Vm)
                result.Trace = pd.to_numeric(result.Trace)
                current = self.get_iv_current()
                result = pd.merge(
                    result, current, how="left", on=["File Name", "Trace"]
                )
                result = result[result.Vm.notna() & result.I.notna()]
        return result

    def get_iv_current(self):
        """Get current of I-V curve for all abf file.
        Returns a dataframe """
        files = pd.Series(self.files["abf"])
        result = files.apply(lambda x: pd.Series(self.get_current_step(file=x)))
        result.insert(0, "File Name", files.apply(getFileName))
        result = pd.melt(
            result,
            id_vars="File Name",
            value_vars=result.columns[1:],
            value_name="I",
            var_name="Trace",
        )
        result["Trace"] = pd.to_numeric(result["Trace"]) + 1
        return result

    def is_outlier(self, **kwargs):
        """Return a dataframe including whether a neuron is an oulier, 
        according to variation of tau parameter.
        Parameter:
        **kwargs: other parameters passed to self.get_all_data()"""
        dat = self.get_all_data("tau", **kwargs)
        outlier = (
            dat[["CellID", "tau", "C"]]
            .groupby("CellID")
            .agg({"tau": np.mean, "C": variation})
            .reset_index()
        )
        outlier["isOutlier"] = (
            (np.abs(outlier["C"]) > 0.2)
            | (outlier["tau"] <= 0)
            | (outlier["tau"] > 500)
        )
        return outlier

    def plot_demo_IF(self, file_id, to_save=""):
        """Plot instant frequency of each current level for a demo neuron.
        file_id: the n-th 'event' file.
        to_save: String. The figure to be saved.
          Default: empty (not to save)."""
        IFdemoData = self.get_demo_IF_data(file_id)
        if IFdemoData is not None:
            sns_line = sns.lineplot(
                x="relative_time",
                y="instantaneous_freq (Hz)",
                hue="Group",
                data=IFdemoData[
                    IFdemoData["instantaneous_freq (Hz)"].between(0, 500, "neither")
                    & (IFdemoData["Group"] != 0)
                ],
                estimator=None,
            )
            plt.xlabel("AP onset time (ms)")
            plt.ylabel("Instantaneous frequency (Hz)")
            title = getCellID(self.files["event"][file_id])
            plt.title(title)
            plt.show()
            figIFdemo = sns_line.get_figure()
            if bool(to_save):
                figIFdemo.savefig("instantaneous freq of" + title, dpi=600)

    def plot_demo_distribution(self, term, file_id, to_save=""):
        """Plot a bar plot of AP number vs. sweep number.
        term: String. Include 'freq', 'amp'.
        file_id: Interger. the n-th 'freq' file.
        to_save: file path. If left empty, not to save."""
        demoData = self.read_demo_stat(term, file_id)
        demoData.plot("sweep", "n", kind="bar", legend=False)
        identifier = getCellID(self.files[term][file_id])
        plt.title(identifier)
        plt.ylabel("AP number")
        if bool(to_save):
            to_save_figure(term + " distribution of " + to_save)

    def get_tau_stat(self):
        """Return a dataframe computed from get_all_data('tau'), exclude outliers"""
        outlier = self.is_outlier()
        return outlier[np.logical_not(outlier.isOutlier)].drop("isOutlier", axis=1)

    def plot_tau_distribution(self, to_save=""):
        """Plot a bar plot of tau parameters, exclude outlier examples.
        to_save: file path. If left empty, not to save."""
        dat = self.get_tau_stat()
        dat.tau[dat.tau.between(0, 500)].plot.hist(
            alpha=0.7
        )  # tau<0 or tau is too large, imnormal
        plt.xlabel("Time constant (ms)")
        plt.ylabel("Number of neuron")
        if bool(to_save):
            to_save_figure(to_save)

    def check_abf_valid(self, file_id=0, file=""):
        """Check if the n-th abf file has normal data organized"""
        if not (file):
            abf = pyabf.ABF(self.files["abf"][file_id])
        else:
            abf = pyabf.ABF(file)
        if len(abf.sweepEpochs.levels) <= 2:
            return False
        else:
            return True


class ramp_parser(electro_base):
    # Cache computed threshold data from `get_threshold_current()`
    threshold_dat = None

    def __init__(self, path_root=None):
        electro_base.__init__(self)
        if path_root is None:
            self.path_root["abf"] = "../../Ramp"
            self.files["abf"], _ = getAllFiles(self.path_root["abf"])

    def plot_demo_ramp(self, file_id=0, cell_id=None, sweep='all', fig_size=(3,3), to_save="", aspect_ratio=150, 
                       sweepYcolor='C0', sweepCcolor='C0'):
        """Plot voltage reaction to ramp current.
        parameters: see `ap_parser().plot_demo_ap()`."""
        if cell_id is not None:
            file_id = self.get_file_id(cell_id)
        fig = plt.figure(figsize=fig_size)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        abf = pyabf.ABF(self.files["abf"][file_id])
        if sweep=='all':
            for sweep in abf.sweepList:
                abf.setSweep(sweep)
                ax1.plot(abf.sweepX, abf.sweepY, sweepYcolor)
                ax2.plot(abf.sweepX, abf.sweepC, sweepCcolor)
        else:
            abf.setSweep(sweep - 1)
            ax1.plot(abf.sweepX, abf.sweepY, sweepYcolor)
            ax2.plot(abf.sweepX, abf.sweepC, sweepCcolor)
        plt.draw()
        asp1=get_aspect_scale(ax1)
        ax2.set_aspect(asp1[1]*aspect_ratio/asp1[0])
        ax1.axis("off")
        ax2.axis("off")
        add_scalebar(ax2, (1,0.2),('s','pA'), fig, y_label='200pA/\n30mV')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            plt.tight_layout()
        if to_save:
            to_save_figure(to_save)

    def get_threshold_current(self):
        """Get the threshold current to initialize AP for all neurons.
        Return:
        A table containing threshold current and threshold voltage."""
        if self.threshold_dat is None:
            _map = (res for f in self.files["abf"] for res in get_threshold_current(f))
            result = pd.DataFrame(
                _map,
                columns=[
                    "cellID",
                    "sweep",
                    "ramp_slope",
                    "threshold_I",
                    "threshold_V",
                    "ramp_spiking_time",
                ],
            )
            self.threshold_dat = result.copy()
        else:
            result = self.threshold_dat.copy()
        return result


class electro_parser(epsc_parser, ap_parser, ramp_parser):
    elec_features = [
        "inv_time_to_first_spike",
        "inv_first_ISI",
        "inv_second_ISI",
        "inv_third_ISI",
        "inv_last_ISI",
        "time_to_last_spike",
        "ISI_semilog_slope",
        "ISI_log_slope",
        "number_initial_spikes",
        "AP1_amp",
        "AP2_amp",
        "APlast_amp",
        "steady_state_voltage_stimend",
    ]
    elec_features_avg = [
        "AHP_time_from_peak",
        "AHP_depth_abs_slow",
        "AHP_depth_abs",
        "AP_duration_half_width",
        "AP_width",
    ]
    elec_data = {"local": None, "python": None, 'all':None}

    def __init__(self):
        """Concantenate EPSC_parser and AP_parser class."""
        epsc_parser.__init__(self)
        path_root = self.path_root
        files = self.files
        ap_parser.__init__(self)
        path_root = merge_dicts(path_root, self.path_root)
        files = merge_dicts(files, self.files, merge_list=False)
        ramp_parser.__init__(self)
        self.path_root = merge_dicts(path_root, self.path_root)
        self.files = merge_dicts(files, self.files)
        self.__backup = None

    
    def get_all_data(self, item="all", method="python", select_trace=13):
        """Get all electrophisiological data, including RMP(rest membrane potential), Rm(membrane resistance).
        method: {'python', 'local', 'all'}, extract data by python package or manual data (for AP data) or merging both.
        Return:
        If method is 'all', return merged result of:
        - Result of `method = 'local'`
        - For python computed result, default only select data of trace 13 (`select_trace`=13).
        - For ramp current result, only select data of sweep 2."""
        if method == "local":
            if self.elec_data["local"] is None:
                if not self.__backup:
                    self.__change_restore()
                if item == "all":
                    iv_dat = ap_parser.get_all_data(self, "iv")
                    rm_dat = compute_Rm(iv_dat).drop("R_score", axis=1)
                    rmp_dat = iv_dat[np.isclose(iv_dat.I, 0)][["CellID", "Vm"]].rename(
                        columns={"Vm": "RMP"}
                    )
                    result = rm_dat.merge(rmp_dat, "outer", on="CellID")
                    tau_dat = self.is_outlier().drop("C", axis=1)
                    result = result.merge(tau_dat, "outer", on="CellID")
                    freq_dat = ap_parser.get_all_data(self, 'freq')
                    freq_dat = freq_dat[freq_dat.sweep==select_trace][['CellID','n','Freq (Hz)']]
                    result = result.merge(freq_dat, "outer", on='CellID')
                    self.reset_all_data('local')
                    if_dat = get_CV_of_IF(self.get_all_data('IF', method='local'))
                    if_dat = if_dat[np.isclose(if_dat.current, 0.35)][['Identifier','CV_of_IF']].rename(columns={'Identifier':'CellID'})
                    result = result.merge(if_dat, "outer", on='CellID')
                else:
                    result = super().get_all_data(item)
                if self.__backup:
                    self.__change_restore("restore")
                self.elec_data["local"] = result.copy()
            else:
                result = self.elec_data["local"].copy()
        elif method == "python":
            if self.elec_data["python"] is None:
                ap_abfs = [
                    f
                    for f in self.files["abf"][1]
                    if ap_parser.check_abf_valid(self, file=f)
                ]
                result = extract_demo_ap_features(ap_abfs, self.elec_features)
                result_mean = extract_demo_ap_features(
                    ap_abfs, self.elec_features_avg, extract_average=True
                )
                result = pd.concat([result, result_mean], ignore_index=True)
                result["featureValue"] = pd.to_numeric(result["featureValue"])
                result = result.pivot_table(
                    "featureValue", ["CellID", "trace"], "featureName"
                ).reset_index()
                self.elec_data["python"] = result.copy()
            else:
                result = self.elec_data["python"].copy()
        elif method == 'all':
            if self.elec_data['all'] is None:
                res1 = self.get_all_data(method='local')
                res2 = self.get_all_data(method='python')
                res2 = res2[res2['trace']==select_trace].drop(columns='trace')
                res3 = self.get_threshold_current()
                res3 = res3[res3['sweep']==2].drop(columns='sweep')
                id_tab = self.get_id_table()
                id_tab = fillna_unique(id_tab, 'cellID_ap', prefix='tab_empty_')
                res1 = fillna_unique(res1.rename(columns={'CellID': 'cellID_ap'}), 'cellID_ap', 'local_empty_')
                result = id_tab.merge(res1, how='outer', on='cellID_ap')
                res2 = fillna_unique(res2.rename(columns={'CellID': 'cellID_ap'}), 'cellID_ap', 'py_empty_')
                result = result.merge(res2, how='outer', on='cellID_ap')
                res3 = fillna_unique(res3.rename(columns={'cellID':'cellID_ramp'}), 'cellID_ramp', 'ramp_empty_')
                result = fillna_unique(result, 'cellID_ramp', 'res_empty_')
                result = result.merge(res3, how='outer',on='cellID_ramp' )
                result.replace(r'.*empty.*',np.nan,regex=True,inplace=True)
                self.elec_data['all']=result.copy()
            else:
                result = self.elec_data['all'].copy()
        return result

    def reset_all_data(self, item="all"):
        """Reset the caching result from `self.get_all_data()` funciton.
        Parameters:
        item:{'all', 'local', 'python'}. Reset all item or only an item."""
        if item == "all":
            self.elec_data = {"local": None, "python": None, 'all':None}
        else:
            self.elec_data[item] = None

    def get_id_table(self):
        """Merge id table for ap_parser, epsc_parser, ramp_parser."""
        if self.id_table is None:
            fun_map = {
                "ap": ap_parser.get_id_table,
                "epsc": epsc_parser.get_id_table,
                "ramp": ramp_parser.get_id_table,
            }
            result_tab = []
            for term, get_id_fun in fun_map.items():
                self.__change_restore(change=term)
                cur_tab = get_id_fun(self)
                result_tab.append(cur_tab)
                self.__change_restore("restore")
            result = (
                result_tab[0]
                .merge(
                    result_tab[1],
                    "outer",
                    on=["recording_date", "recording_cell_id"],
                    suffixes=("_ap", "_epsc"),
                )
                .merge(
                    result_tab[2], "outer", on=["recording_date", "recording_cell_id"]
                )
                .rename(columns={"cellID":"cellID_ramp"})
            )
        else:
            result = self.id_table
        return result
    
    def is_outlier(self):
        """Wrap ap_parser.is_outlier()."""
        return ap_parser.is_outlier(self, method='local')
    
    def get_threshold_current(self):
        """Wrap ramp_parser.get_threshold_current(self)"""
        self.__change_restore(change='ramp')
        res = ramp_parser.get_threshold_current(self)
        self.__change_restore("restore")
        return res
    
    def plot_demo_ap(self, **kwargs):
        """Wrapper of ap_parser.plot_demo_ap()
        **kwargs: parameters passed to ap_parser.plot_demo_ap."""
        ap_ps=ap_parser()
        ap_ps.plot_demo_ap(**kwargs)
        
    def plot_demo_ramp(self, cell_id=None, **kwargs):
        """Wrapper of ramp_parser.plot_demo_ramp()
        **kwargs: parameters passed to ramp_parser.plot_demo_ramp()"""
        ramp_ps = ramp_parser()
        ramp_ps.plot_demo_ramp(cell_id=cell_id, **kwargs)

    def __change_restore(self, func="change", change="ap"):
        """Change instance parameter (`path_root` and `files`) temperally to satisfy the format of parents conflicted.
        func: {'change', 'restore'}
        change: {'epsc', 'ap', 'ramp'}. only change to parameters of parent epsc_parser or ap_parser."""
        if func == "change":
            self.__backup = self.path_root.copy(), self.files.copy()
            ind = {"ap": 1, "epsc": 0, "ramp": 2}
            for term in ["abf", "amp"]:
                self.path_root[term] = self.path_root[term][ind[change]]
                self.files[term] = self.files[term][ind[change]]
        elif func == "restore":
            self.path_root, self.files = self.__backup
            self.__backup = None


def get_CV_of_IF(IFdata):
    """Compute CV of IF vs. current for all neuron.
    IFdata: pandas dataframe. Instantaneous frequency of all neuron,
    for example, generated by ap_parser.get_all_neuron('IF').
    Return pandas dataframe."""
    IFstat = IFdata.loc[
        IFdata["instantaneous_freq (Hz)"] > 0,
        ["Identifier", "instantaneous_freq (Hz)", "current"],
    ]
    IFstat = (
        IFstat.groupby(["Identifier", "current"])
        .agg({"instantaneous_freq (Hz)": variation})
        .reset_index()
    )
    IFstat.rename(columns={"instantaneous_freq (Hz)": "CV_of_IF"}, inplace=True)
    IFstat["I"] = IFstat["current"].transform(lambda x: "%.2f" % (x))
    return IFstat


def plot_IF_CV_distribution(IFstat, to_save=""):
    """Plot CV of IF vs. current for all neuron.
    IFstat: pandas dataframe. Instantaneous frequency variance(CV of IF),
    for example, generated by function get_CV_of_IF(IFdata).
    Result returned as a pandas dataframe.
    """
    figIFdist = sns.catplot(IFstat, x="I", y="CV_of_IF")
    plt.xlabel("Current step (nA)")
    plt.ylabel("CV of instantaneous frequency")
    plt.show()
    if bool(to_save):
        figIFdist.savefig("IF_distribution", dpi=600)


def read_atf(filename):
    """Read atf file exported by clampfit software.
    filename: full path of an .atf file."""
    with open(filename, "r") as f:
        txt = f.readlines()
    title = np.array(re.findall(r'"([^"]*)"', txt[2]))
    title = np.where(title=='R1S1 Mean (mV)','S1R1 Mean (mV)',title)
    datastr = txt[3].split("\t")
    sz = (len(datastr) // len(title), len(title))
    dt = pd.DataFrame(np.array(datastr).reshape(sz), columns=title)
    return dt


def plot_iv_traces(IVdata, to_save=""):
    """Plot I-V traces for all abf file.
    IVdata: dataframe. Format from generation by function ap_parser.get_all_data('iv')"""
    for key, tb in IVdata.groupby("CellID"):
        plt.plot(tb["I"], tb["Vm"], "-o", alpha=0.5, color="C0")
    plt.xlabel("I (nA)")
    plt.ylabel("Vm (mV)")
    if bool(to_save):
        to_save_figure(to_save=to_save)


def compute_Rm(IVdata):
    """Compute input resistance according to I-V curve.
    IVdata: dataframe. Format from generation by function ap_parser.get_all_data('iv').
    Return: dataframe. Rm of each neuron."""
    result = IVdata.groupby("CellID").apply(compute_demo_Rm)
    result = pd.DataFrame(result.to_list(), index=result.index).reset_index()
    return result


def compute_demo_Rm(IVdata):
    """Compute input resistance for only one neuron, accordig to its I-V curve.
    IVdata: dataframe. Information must include only one neuron.
    Return: dict of {'Rm', "R_score"} as linear regression result."""
    x = IVdata[["I"]]
    y = IVdata["Vm"]
    reg = LinearRegression().fit(x, y)
    score = reg.score(x, y)
    coef = reg.coef_[0]
    return {"Rm": coef, "R_score": score}


def plot_Rm_distribution(Rm_data, to_save=""):
    """Bar plot of Rm for all neurons.
    Rm_data: dataframe. Format from generation by function compute_Rm().
    to_save: string. Filename to be saved."""
    # Filter R_score>0.9
    Rm_data[Rm_data.R_score > 0.9].Rm.plot.hist(alpha=0.75)
    plt.xlabel(r"$Rm\ (M\Omega)$")
    plt.ylabel("Number of neuron")
    if bool(to_save):
        to_save_figure(to_save)


def extract_demo_epsc_features(file, features):
    """Extract electrophysiological features from an abf file about EPSC recording.
    features: list. All features to be extracted."""
    pass


def extract_demo_ap_features(file, features, extract_average=False):
    """Extract electrophysiological features from an abf file about AP recording.
    features: list. All features to be extracted.
    extrace_average: Average the values of the same sweep."""
    result = []
    visited = []
    for f in file:
        traces = abf2trace(f)
        cellid = getCellID(f)
        if cellid in visited:
            continue
        cur_result = efel.getFeatureValues(traces, features, raise_warnings=False)
        cur_result = {cellid: cur_result}
        visited.append(cellid)
        if not extract_average:
            result.append(
                pd.DataFrame(
                    (
                        (cellid, ind + 1, itemk, itemv[0])
                        if itemv
                        else (cellid, ind + 1, itemk, None)
                        for cellid, v in cur_result.items()
                        for ind, vv in enumerate(v)
                        for itemk, itemv in vv.items()
                    ),
                    columns=["CellID", "trace", "featureName", "featureValue"],
                )
            )
        else:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                result.append(
                    pd.DataFrame(
                        (
                            (cellid, ind + 1, "mean_" + itemk, itemv.mean())
                            if isinstance(itemv, np.ndarray)
                            else (cellid, ind + 1, itemk, None)
                            for cellid, v in cur_result.items()
                            for ind, vv in enumerate(v)
                            for itemk, itemv in vv.items()
                        ),
                        columns=["CellID", "trace", "featureName", "featureValue"],
                    )
                )
    return pd.concat(result, ignore_index=True)


def abf2trace(file):
    """Corvert an abf file for AP recording to trace format for eFEL."""
    f = pyabf.ABF(file)
    traces = []
    for sweep in f.sweepList:
        f.setSweep(sweep)
        t = f.sweepX * 10 ** 3
        v = f.sweepY
        start = f.sweepEpochs.p1s[2] * f.dataSecPerPoint * 10 ** 3
        end = f.sweepEpochs.p2s[2] * f.dataSecPerPoint * 10 ** 3
        traces.append({"T": t, "V": v, "stim_start": [start], "stim_end": [end]})
    return traces


def get_threshold_current(file):
    """Get the threshold current to initialize AP for a neuron.
    Input:
    file: An abf file with voltage under ramp current recorded.
    Return:
    A generator containing (cell_ID, trace/sweep, ramp slope(pA/s), threshold current(pA), threshold voltage(mV), spiking_time(s))."""
    abf = pyabf.ABF(file)
    cellid = getCellID(file)
    dt = abf.dataSecPerPoint
    start, end = abf.sweepEpochs.p1s[2], abf.sweepEpochs.p2s[2]
    for sweep in abf.sweepList:
        abf.setSweep(sweep)
        dy = np.diff(abf.sweepY[start:end])
        dc = np.diff(abf.sweepC[start:end])
        dyt = dy / dt
        dct = dc / dt
        try:
            if dyt.max() / dyt.std() < 10:
                continue
            # if dv/dt > 5*std(dy/dt), set as threshold point
            spike_p = np.where(dyt > 5 * dyt.std())[0]
            th_i = abf.sweepC[spike_p[0] + start]
            th_v = abf.sweepY[spike_p[0] + start]
            th_t = abf.sweepX[spike_p[0] + start]
            yield (cellid, sweep + 1, dct[0], th_i, th_v, th_t)
        except ValueError:
            print(
                "Cannot compute rheobase for `{}` trace {}".format(
                    getFileName(file), sweep + 1
                )
            )
