"""Preparation-only scriplet.
"""

from analysis_sherlock_ML import *

# CUSTOMIZATIONS (optional)
numpy.set_printoptions(linewidth=1000)


df = pd.read_csv("sherlock_apps_yhe_test.csv")
summarize_dataset(df)
df2 = preprocess_sherlock_19F17C(df)

print()
Rec = step0_label_features(df2)

tmp = step_onehot_encoding(Rec)
tmp = step_feature_scaling(Rec)
print("After scaling:")
print(Rec.df_features.head(10))

print()
tmp = step_train_test_split(Rec, test_size=0.2, random_state=34)

print("Now the dataset is ready for machine learning!")
