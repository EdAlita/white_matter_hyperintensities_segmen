import pandas as pd
import os
import sys

sys.path.append('../')
sys.path.append('../..')

from stratify import compute_stratification_index
file='MD'
save_dir ='/localmount/volume-hd/users/uline/data_sets/UK-biobank/'
# load data containing demographics and hypothalamus process volumes
demo_file = f'/localmount/volume-hd/users/uline/data_sets/UK-biobank/biobank_{file}.csv'
df=pd.read_csv(demo_file,sep=',')
df_subset = df[['eid','iid','aid','Sex','Age_Visit2','WMH_load']]
df_values = df_subset.values
train_idx,val_idx=compute_stratification_index(df_subset,num_classes=30,n_splits=80,sex_var='Sex',age_var='Age_Visit2',id_var='eid')

#save values
df_case_study=df_subset[df_subset['eid'].isin(df_values[train_idx[0], 0])]
df_train = df_case_study.sample(n=60,random_state=42)
df_reducido = df[~df['eid'].isin(df_train['eid'])]
df_train.to_csv(os.path.join(save_dir, f'subset_test_{file}.csv'), sep=',', index=False)


#df_case2_study=df_subset[df_subset['eid'].isin(df_values[val_idx[0], 0])]
#df_val = df_case2_study.sample(n=20,random_state=42)
#df_val.to_csv(os.path.join(save_dir, f'subset_val_{file}.csv'), sep=',', index=False)
#df_reducido = df_reducido[~df_reducido['eid'].isin(df_val['eid'])]

print(df['eid'].is_unique)

df_reducido.to_csv(demo_file, sep=',', index=False)

print(df.shape,df_reducido.shape)