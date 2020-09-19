import os
import os.path
import re
import numpy as np
import pandas as pd
from itertools import product


def getAllFiles(rootDir, ends=""):
    """get all files in a folder recursively, exclude hidden folders.
    ends: file format. Default ''(empty string, all format)."""
    FilePath = []
    Files = []
    for (root, sub, file) in os.walk(rootDir):
        if not (sub) or all([s.startswith(".") for s in sub]):
            if not os.path.abspath(root).split("/")[-1].startswith("."):
                FilePath += [os.path.join(root, f) for f in file if f.endswith(ends)]
                Files += [f.split(".")[0] for f in file if f.endswith(ends)]
    return (FilePath, Files)


def getCellID(filePath):
    """Given a file path, return the cell ID generated by Clampex software.
    Each measured result has a unique ID."""
    result = (os.path.split(filePath))[-1].split(".")[0]
    return re.findall("\d{2}[0-9ond]\d{5}", result)[0]

def get_cellID_info(filePath):
    """get recording date info and cell_ID info for given abf file, according to its path."""
    pat = '/(20\d{6})/cell.?(\d{1,2})/(\d{2}[0-9ond]\d{5}).*.abf'
    _date, _cell_id, _id = re.findall(pat, filePath)[0]
    return _id, _date, _cell_id

def getFileName(fullpath):
    "Input fullpath, return filename"
    return os.path.split(fullpath)[-1]


def zero_padding(tab, *padding_rows):
    """Zero-padding several rows in a dataframe, the other columns show all possible combinations."""
    keep_rows = tab.columns.drop(list(padding_rows))
    base = pd.DataFrame(product(*[set(tab[r]) for r in keep_rows]), columns=keep_rows)
    res = pd.merge(base, tab, "outer", on=keep_rows.tolist()).fillna(0)
    return res


def merge_dicts(dict1, dict2, merge_list=True):
    """
    Merge two dict and merge values with the same key.
    Parameters:
    - merge_list: logic. Whether flat lists inside dict to merge them.
    """
    out = {**dict1, **dict2}
    for key, value in out.items():
        if key in dict1 and key in dict2:
            if isinstance(dict1[key], list) & merge_list:
                out[key]=[*dict1[key], value]
            else:
                out[key]=[dict1[key], value]
    return out

def fillna_unique(dat, col, prefix='empty_'):
    """Fill NA value in a column of DataFrame with unique string.
    prefix： The prefix of filled value. Results of filled values are 'empty_1', 'empty_2', ... (eg. when prefix is 'empty')."""
    col_val = dat[col]
    new_val = np.where(~col_val.isna(), col_val.values, prefix+col_val.isna().cumsum().astype('str'))
    copy_ = dat.copy()
    copy_[col] = new_val
    return copy_