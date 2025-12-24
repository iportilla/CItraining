import os
from os import path
import pandas
import numpy
from numpy import inf, nan, isnan, isinf, isposinf, isneginf
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
    

SCHEMA_NETFLOW_inp = """
FlowID STRING,         # 1
src_ip STRING,
src_port INTEGER,
dest_ip STRING,
dest_port INTEGER,
protocol INTEGER,
t_stamp TIMESTAMP,
duration BIGINT,

fw_npkt_tot BIGINT,
bw_npkt_tot BIGINT,    # 10
fw_plen_tot DOUBLE,
bw_plen_tot DOUBLE,
fw_plen_max DOUBLE,
fw_plen_min DOUBLE,
fw_plen_mean DOUBLE,
fw_plen_std DOUBLE,
bw_plen_max DOUBLE,
bw_plen_min DOUBLE,
bw_plen_mean DOUBLE,
bw_plen_std DOUBLE,    # 20

bd_Bps DOUBLE,
bd_Pps DOUBLE,

bd_IAT_mean DOUBLE,
bd_IAT_std DOUBLE,
bd_IAT_max DOUBLE,
bd_IAT_min DOUBLE,

fw_IAT_tot DOUBLE,
fw_IAT_mean DOUBLE,
fw_IAT_std DOUBLE,
fw_IAT_max DOUBLE,     # 30
fw_IAT_min DOUBLE,

bw_IAT_tot DOUBLE,
bw_IAT_mean DOUBLE,
bw_IAT_std DOUBLE,
bw_IAT_max DOUBLE,
bw_IAT_min DOUBLE,

fw_nflg_PSH INTEGER,
bw_nflg_PSH INTEGER,
fw_nflg_URG INTEGER,
bw_nflg_URG INTEGER,    # 40

fw_len_hdr INTEGER,
bw_len_hdr INTEGER,
fw_Pps DOUBLE,
bw_Pps DOUBLE,

plen_min DOUBLE,
plen_max DOUBLE,
plen_mean DOUBLE,
plen_std DOUBLE,
plen_var DOUBLE,

nflg_FIN INTEGER,        # 50
nflg_SYN INTEGER,
nflg_RST INTEGER,
nflg_PSH INTEGER,
nflg_ACK INTEGER,
nflg_URG INTEGER,
nflg_CWE INTEGER,
nflg_ECE INTEGER,

down_up_ratio DOUBLE,
pkt_size_avg DOUBLE,
fw_seg_size_avg DOUBLE,  # 60
bw_seg_size_avg DOUBLE,

fw_len_hdr61 INTEGER,    # 62--duplicated of 41

fw_Bpb_avg DOUBLE,
fw_Ppb_avg DOUBLE,
fw_bulkrate_avg DOUBLE,
bw_Bpb_avg DOUBLE,
bw_Ppb_avg DOUBLE,
bw_bulkrate_avg DOUBLE,

fw_npkt_subflow BIGINT,
fw_nbytes_subflow BIGINT, # 70
bw_npkt_subflow BIGINT,
bw_nbytes_subflow BIGINT,

fw_init_win_bytes INTEGER,
bw_init_win_bytes INTEGER,
fw_act_data_pkt BIGINT,
fw_min_seg_size BIGINT,

active_mean DOUBLE,
active_std DOUBLE,
active_max DOUBLE,
active_min DOUBLE,        # 80

idle_mean DOUBLE,
idle_std DOUBLE,
idle_max DOUBLE,
idle_min DOUBLE,

label STRING              # 85
"""

COLS_FEATURE_1_inp = """
duration
fw_npkt_tot
bw_npkt_tot
fw_plen_tot
bw_plen_tot
fw_plen_max
fw_plen_min
fw_plen_mean
fw_plen_std
bw_plen_max
bw_plen_min
bw_plen_mean
bw_plen_std
bd_Bps
bd_Pps
bd_IAT_mean
bd_IAT_std
bd_IAT_max
bd_IAT_min
fw_IAT_tot
fw_IAT_mean
fw_IAT_std
fw_IAT_max
fw_IAT_min
bw_IAT_tot
bw_IAT_mean
bw_IAT_std
bw_IAT_max
bw_IAT_min
fw_nflg_PSH
bw_nflg_PSH
fw_nflg_URG
bw_nflg_URG
fw_len_hdr
bw_len_hdr
fw_Pps
bw_Pps
plen_min
plen_max
plen_mean
plen_std
plen_var
nflg_FIN
nflg_SYN
nflg_RST
nflg_PSH
nflg_ACK
nflg_URG
nflg_CWE
nflg_ECE
down_up_ratio
pkt_size_avg
fw_seg_size_avg
bw_seg_size_avg
fw_Bpb_avg
fw_Ppb_avg
fw_bulkrate_avg
bw_Bpb_avg
bw_Ppb_avg
bw_bulkrate_avg
fw_npkt_subflow
fw_nbytes_subflow
bw_npkt_subflow
bw_nbytes_subflow
fw_init_win_bytes
bw_init_win_bytes
fw_act_data_pkt
fw_min_seg_size
active_mean
active_std
active_max
active_min
idle_mean
idle_std
idle_max
idle_min
"""

# Converts the schema form above to something that is palatable to
# Pandas.
dtype_mapper = {
    'STRING': numpy.str_,
    'DOUBLE': numpy.float64,
    'INTEGER': numpy.int32,
    'BIGINT': numpy.int64,
    'TIMESTAMP': numpy.str_,  # FIXME: Should be --> numpy.dtype("datetime64[ms]"),
}
_schema = []
for L in SCHEMA_NETFLOW_inp.splitlines():
    L = L.split("#")[0].strip()
    if L != "":
        colname, dtype = L.rstrip(",").split(None, maxsplit=1)
        _schema.append((colname, dtype_mapper[dtype]))

# The datastructure containing the schema of the dataset:
SCHEMA_NETFLOW_PD = dict(
    name=[ n[0] for n in _schema ],
    dtype=dict(_schema),
    raw=(_schema),
)
#del _schema, L


def categorical_to_numerics(a, cats=None):
    """Converts array or Series of categorical (or string values)
    to numerical enumeration.
    The input array `a` must be one-dimensional in nature.
    Unless the input categories are given (in `cats` argument),
    the labels will be auto-detected by this algorithm.

    Sometimes this step is needed because not all ML algorithms can
    take categoricals and go with it.
    """
    # HACK NOTE: Actually, an Pandas.Categorical object has the `_ndarray_values`
    # member which contains exactly the array we are seeking.
    # But I won't use that because it is not a documented behavior.

    if cats is not None:
        # assume that cats is a valid categories
        pass
        # Otherwise, extract the categories: hopefully one of these
        # ways gets it:
    elif isinstance(a, pandas.Series):
        if isinstance(a.dtypes, pandas.api.types.CategoricalDtype):
            cats = a.dtypes.categories
        else:
            # general approach for array of strings
            cats = sorted(a.unique())
    elif isinstance(a, pandas.Categorical):
        cats = a.categories
    else:
        # general iterable case
        cats = sorted(pandas.Series(a).unique())
    #    raise TypeError("Don't know how to handle {}".format(a.__class__))

    # mapping: category -> numerics
    cat_map = dict((c, i) for (i,c) in enumerate(cats))
    # mapping: numerics -> category
    cat_revmap = list(cats)

    return (numpy.array([cat_map[c] for c in a]), cat_revmap)


def delete_bad_numerics(df, inplace=False):
    """Returns a clean dataframe by deleting rows that have bad numerical values
    (infinity or NaN values).
    """
   
    if not inplace:
        df_new = df.replace([inf, -inf], nan).dropna()
        return df_new
    else:
        df.replace([inf, -inf], nan, inplace=True)
        df.dropna(inplace=True)
        return df


def detect_bad_numeric_columns(df):
    """Detects if certain columns are giving us headache down the road because
    of infinity or NaN values.
    """    

    cols = df.columns
    print("Detecting NaN values...")
    cols_has_nan = []
    for c in cols:
        try:
            nan_df = df[isnan(df[c])]
        except TypeError:
            continue
        if nan_df.shape[0] != 0:
            cols_has_nan.append((c, nan_df.shape[0]))
    if len(cols_has_nan) > 0:
        print("  {} columns detected as having NaN:".format(len(cols_has_nan)))
        print("  ", str(cols_has_nan))

    print("Detecting +/-Inf values...")
    cols_has_inf = []
    for c in cols:
        try:
            inf_df = df[isneginf(df[c]) | isposinf(df[c])]
        except TypeError:
            continue
        if inf_df.shape[0] != 0:
            cols_has_inf.append((c, inf_df.shape[0]))
    if len(cols_has_inf) > 0:
        print("  {} columns detected as having +/-Inf:".format(len(cols_has_inf)))
        print("  ", str(cols_has_inf))

    return (cols_has_nan, cols_has_inf)


def flowmeter_load_csv_pd(filename):
    """Loads CICFlowMeter data file (in CSV format) into a Pandas
    dataframe."""

    col_names = SCHEMA_NETFLOW_PD['name']
    col_dtype = SCHEMA_NETFLOW_PD['dtype']

    # For some reason the C reader engine choked when parsing thru "NaN"
    # (column 20 = bd_Bps has some NaN values).
    df = pandas.read_csv(filename,
                         skiprows=1,
                         names=col_names,
                         dtype=col_dtype,
                         index_col=False,
                         # Also disregard all infinity values--these strings cause
                         # the C engine to choke
                         na_values=['Infinity', '-Infinity', '+Infinity',
                                    'Inf', '-Inf', '+Inf'],
                         header=None, engine='c')
    return df


def load_data(data_home, filename):
    """Experiment X1: using only Tuesday dataset.

    """    
    data_path = os.path.join(data_home, filename)
    print("Loading IDS2017 dataset file: {}".format(filename))
    DF_TUE = flowmeter_load_csv_pd(data_path)
    print("Raw dataset dimensions: {}".format(DF_TUE.shape))
    r1 = detect_bad_numeric_columns(DF_TUE)
    DF_TUE_cleaned = delete_bad_numerics(DF_TUE, inplace=True)
    print("Post-cleaning dataset dimensions: {}".format(DF_TUE_cleaned.shape))
    
    return DF_TUE

def prepare_input(data_frame, split_testing=0.2):
    """Experiment X1: using only Tuesday dataset.

    Step 2: Prepares input data (training & testing).
    """    

    DF_TUE = data_frame
    
    X1_SPLIT = (1.0 - split_testing, split_testing)
    
    COLS_FEATURE_1 = COLS_FEATURE_1_inp.split()

    DF_TUE['label_val'] = DF_TUE['label'].astype('category')
    DF_TUE_FEATURES = DF_TUE[COLS_FEATURE_1 + ['label_val']]
    print("Randomly split dataset to training/testing dataset: test fraction = {}" \
              .format(X1_SPLIT[1]))
    X1_TRAINING, X1_TESTING = train_test_split(DF_TUE_FEATURES, test_size=X1_SPLIT[1])

    # Converts this to numerical-value matrix
    X1_TRAINING_FM = X1_TRAINING[COLS_FEATURE_1].values
    X1_TRAINING_L = X1_TRAINING['label_val'].values
    X1_TRAINING_LL, X1_LABELS = categorical_to_numerics(X1_TRAINING_L)

    X1_TESTING_FM = X1_TESTING[COLS_FEATURE_1].values
    X1_TESTING_L = X1_TESTING['label_val'].values

    # Don't re-create labels: this will help detect weird erroneous
    # situation where the training data doesn't contain some
    # categories existing in the test data
    X1_TESTING_LL, labels2 = categorical_to_numerics(X1_TESTING_L, X1_LABELS)

    print("Training data label stats:")
    print(X1_TRAINING[[COLS_FEATURE_1[0], 'label_val']].groupby('label_val').count())

    print("Testing data label stats:")
    print(X1_TESTING[[COLS_FEATURE_1[0], 'label_val']].groupby('label_val').count())
    
    return X1_TRAINING_FM, X1_TRAINING_L, X1_TESTING_FM, X1_TESTING_L


def svm_model_fit(training_feature_matrix, training_labels, verbose=True):
    """Experiment X1: using only Tuesday dataset.

    Step '13': Train a SVM classifier.
    """    
    

    # Feature matrix and label vector   
    X1_TRAINING_FM = training_feature_matrix
    X1_TRAINING_L = training_labels
         
    print("Fitting an SVC model...")
    X1_MODEL_SVM = SVC(verbose=verbose)
    X1_MODEL_SVM.fit(X1_TRAINING_FM, X1_TRAINING_L)   

    return X1_MODEL_SVM

def model_test(test_feature_matrix, model):
    """Experiment X1: using only Tuesday dataset.

    Step 4: Test the quality of the trained model.

    NOTE: This function has been generalized to handle *any*
    model defined by sklearn.
    """

    # Feature matrix and label vector    
    X1_TESTING_FM = test_feature_matrix

    # Extracts the model class name:
    model_class = model.__class__.__name__
    
    X1_MODEL_DT = model

    print("Quality-checking the {} model...".format(model_class))
    X1_TESTING_Lpred = X1_MODEL_DT.predict(X1_TESTING_FM)

    return X1_TESTING_Lpred

DATA_HOME = "/scratch-lustre/DeapSECURE/module02/CIC-IDS-2017/TrafficLabeling/"
file_name = "Tuesday-WorkingHours.pcap_ISCX.csv"

TU_DF = load_data(DATA_HOME, file_name)
training_fm, training_l, testing_fm, testing_l = prepare_input(TU_DF, split_testing=.9)
model = svm_model_fit(training_fm, training_l)
prediction = model_test(testing_fm, model)

Confusion_matrix = confusion_matrix(testing_l, prediction)
Accuracy_score = accuracy_score(testing_l, prediction)

print("Testing confusion matrix: ")
print(Confusion_matrix)
print("Testing acuracy score : ", Accuracy_score)
