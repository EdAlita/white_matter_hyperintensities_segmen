import sys, os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold


def histedges_equalN(x, nbin=10):
    """Computes histogram with bin-edges containing equal samples each

    Parameters
    ----------
    x : iterable
        Data
    nbin : integer
        Number of bins

    Returns
    -------
    List of bin-edges
    """
    npt = len(x)
    return np.interp(np.linspace(0, npt, nbin + 1),
                     np.arange(npt),
                     np.sort(x))


def compute_stratified_class(rs_df, num_classes=30, sex_var='BASE_SEX_R1', age_var='BASE_AGE_R1'):
    """Returns a series with class info stratified by sex and age
    Parameters
    ----------
        rs_df : pandas dataframe
            contains all data
        num_classes: int
            number of classes to create. Maximum should be the number of expected samples in the smallest fold.
            For N=1000 and nested CV of 1/4, 1/3, the max number is approx 80. Choose 20 or 30 to be safe.
        sex_var : string
            id of sex variable
        age_var : string
            id of age variable
    Returns
    -------
        pandas series with stratified class variable
    """
    y = np.zeros(rs_df.shape[0], dtype=np.int32)

    sex = rs_df[sex_var].astype('category').cat.codes  # get numeric values
    sex_codes = sex.unique()
    nbins0 = np.int32((num_classes * (sex == sex_codes[0]).sum()) / rs_df.shape[
        0])  # proportional bins for men and women accoding to prevalence
    nbins1 = np.int32((num_classes * (sex == sex_codes[1]).sum()) / rs_df.shape[0])
    bins0 = histedges_equalN(rs_df.loc[sex == sex_codes[0], age_var].values, nbins0)
    bins1 = histedges_equalN(rs_df.loc[sex == sex_codes[1], age_var].values, nbins1)

    bins0[-1] += 1.e-3  # slightly extend last bin to avoid highest sample falling in new bin by digitize function below
    bins1[-1] += 1.e-3

    y[sex == sex_codes[0]] = np.digitize(rs_df.loc[sex == sex_codes[0], age_var], bins0)
    y[sex == sex_codes[1]] = np.digitize(rs_df.loc[sex == sex_codes[1], age_var],
                                         bins1) + nbins0  # to not overlap with men's class-ids

    return pd.Series(y, index=rs_df.index)


def compute_stratification(csv_file='',sep=',',save_dir='',num_classes=30,n_splits=5,sex_var='sex',age_var='age',id_var='imageid'):

    array_df = pd.read_csv(csv_file, sep=sep)

    demo_arr = array_df.values
    # Stratify only the single_index ones
    vols_df = compute_stratified_class(array_df, num_classes=num_classes, sex_var=sex_var, age_var=age_var)
    #print(len(vols_df.unique()))

    labels = vols_df.values

    skf = StratifiedKFold(n_splits=n_splits, random_state=5,shuffle=True)

    train_index = {}
    test_index = {}
    i = 0

    sub_column= array_df.columns.get_loc(id_var)
    for tr_index, te_index in skf.split(np.zeros(labels.shape[0]), labels):
        print("TRAIN:", len(tr_index), "VALIDATION:", len(te_index))
        train_index[i] = tr_index
        test_index[i] = te_index
        i += 1
        # train_subs=np.zeros((len(tr_index),3),dtype=object)
        train_subs = demo_arr[tr_index, sub_column]

        # test_subs = np.zeros((len(tr_index), 3),dtype=object)
        test_subs = demo_arr[te_index, sub_column]

        df = pd.DataFrame(train_subs, columns=[id_var])
        df.to_csv(os.path.join(save_dir,'train_split_'+ str(i) +'.csv'), sep=',', index=False)

        df = pd.DataFrame(test_subs, columns=[id_var])
        df.to_csv(os.path.join(save_dir ,'val_split_'+str(i)+'.csv'), sep=',', index=False)


def compute_stratification_index(array_df,num_classes=30,n_splits=5,sex_var='sex',age_var='age',id_var='imageid'):

    # Stratify only the single_index ones
    vols_df = compute_stratified_class(array_df, num_classes=num_classes, sex_var=sex_var, age_var=age_var)
    #print(len(vols_df.unique()))

    labels = vols_df.values

    skf = StratifiedKFold(n_splits=n_splits)

    train_index = {}
    test_index = {}
    i = 0

    sub_column= array_df.columns.get_loc(id_var)
    for tr_index, te_index in skf.split(np.zeros(labels.shape[0]), labels):
        print("TRAIN:", len(tr_index), "VALIDATION:", len(te_index))
        train_index[i] = tr_index
        test_index[i] = te_index
        i += 1

    return train_index,test_index
