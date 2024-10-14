import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Use an interactive backend
matplotlib.use('TkAgg')

# Read data from CSV file
data = pd.read_csv('data.csv')

# Identify columns for each group
sta_cols = [col for col in data.columns if col.endswith('STA')]
per_cols = [col for col in data.columns if col.endswith('PER')]
cad_cols = [col for col in data.columns if col.endswith('CAD')]
rel_cols = [col for col in data.columns if col.endswith('REL')]
did_cols = [col for col in data.columns if col.endswith('DID')]

# Combine data for each group up to row 87
sta_data = data[sta_cols][:87].values.flatten()
per_data = data[per_cols][:87].values.flatten()
cad_data = data[cad_cols][:87].values.flatten()
rel_data = data[rel_cols][:87].values.flatten()
did_data = data[did_cols][:87].values.flatten()

# Ensure all arrays have the same length
min_length = min(len(sta_data), len(per_data), len(cad_data), len(rel_data), len(did_data), len(data['Sexe'][:87]))

sta_data = sta_data[:min_length]
per_data = per_data[:min_length]
cad_data = cad_data[:min_length]
rel_data = rel_data[:min_length]
did_data = did_data[:min_length]
sexe_data = data['Sexe'][:min_length]

# Convert to numeric, forcing errors to NaN
sta_data = pd.to_numeric(sta_data, errors='coerce')
per_data = pd.to_numeric(per_data, errors='coerce')
cad_data = pd.to_numeric(cad_data, errors='coerce')
rel_data = pd.to_numeric(rel_data, errors='coerce')
did_data = pd.to_numeric(did_data, errors='coerce')

# Drop NaN values
sta_data = sta_data[~pd.isna(sta_data)]
per_data = per_data[~pd.isna(per_data)]
cad_data = cad_data[~pd.isna(cad_data)]
rel_data = rel_data[~pd.isna(rel_data)]
did_data = did_data[~pd.isna(did_data)]

# Create a DataFrame for the combined data
combined_data = pd.DataFrame({
    'STA': sta_data,
    'PER': per_data,
    'CAD': cad_data,
    'REL': rel_data,
    'DID': did_data,
    'Sexe': sexe_data[:len(sta_data)]  # Ensure Sexe data is the same length as the numeric data
})

# Reshape the data to long format
long_data = pd.melt(combined_data, id_vars=['Sexe'], var_name='Group', value_name='Value')

# Convert to numeric, forcing errors to NaN
long_data['Value'] = pd.to_numeric(long_data['Value'], errors='coerce')

# Drop NaN values
long_data = long_data.dropna()

# Perform two-way ANOVA
model = ols('Value ~ C(Group) + C(Sexe) + C(Group):C(Sexe)', data=long_data).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

# Calculate means for radar chart
means = [
    np.mean(sta_data),
    np.mean(per_data),
    np.mean(cad_data),
    np.mean(rel_data),
    np.mean(did_data)
]

# Radar chart labels
labels = ['STA', 'PER', 'CAD', 'REL', 'DID']

# Number of variables
num_vars = len(labels)

# Compute angle of each axis
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

# The plot is a circle, so we need to "complete the loop"
# and append the start value to the end.
means += means[:1]
angles += angles[:1]

# Plot
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.fill(angles, means, color='red', alpha=0.25)
ax.plot(angles, means, color='red', linewidth=2)

# Labels
ax.set_yticklabels([])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)

plt.title('Mean Comparison of Groups')
plt.show()

# Perform pairwise t-tests comparing REL to each of the other groups
groups = {'STA': sta_data, 'PER': per_data, 'CAD': cad_data, 'DID': did_data}
for group_name, group_data in groups.items():
    t_stat, p_val = stats.ttest_ind(rel_data, group_data, equal_var=False)
    print(f"REL vs {group_name}: t-statistic = {t_stat}, p-value = {p_val}")
    if p_val < 0.05 and np.mean(rel_data) > np.mean(group_data):
        print(f"REL is statistically higher than {group_name} (p < 0.05)")
    else:
        print(f"REL is not statistically higher than {group_name} (p >= 0.05)")