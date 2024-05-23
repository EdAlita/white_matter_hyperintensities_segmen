import pandas as pd
from scipy.stats import wilcoxon
import itertools

df = pd.read_csv('/localmount/volume-hd/users/uline/segmentation_results/aug_test/test_flairt1.csv')

dice_columns = [col for col in df.columns if 'Recall' in col]
print(dice_columns)

# Calculate Wilcoxon tests for all unique pairs of 'DICE' columns
p_values = pd.DataFrame(index=dice_columns, columns=dice_columns)

for col1, col2 in itertools.combinations(dice_columns, 2):
    valid_data = df[[col1, col2]].dropna()

    stat, p_value = wilcoxon(valid_data[col1],
                             valid_data[col2])

    p_values.at[col1, col2] = p_value
    p_values.at[col2, col1] = p_value

print(p_values)

p_values.to_csv('wilcoxon_p_values_Recall_t1.csv')
