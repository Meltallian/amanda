import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import re
import scikit_posthocs as sp
from scipy.stats import kruskal
from bioinfokit.analys import stat
import pingouin as pg


# Load and clean the data
data = pd.read_csv('data.csv')

# Isolate the columns of interest
columns_of_interest = data.loc[:, '1STA':'44PER']

# Convert all columns of interest to numeric, coerce errors to NaN
columns_of_interest = columns_of_interest.apply(pd.to_numeric, errors='coerce')
# Remove rows that contain only 1s or only 4s
filtered_data = columns_of_interest[
    ~columns_of_interest.eq(1).all(axis=1) & ~columns_of_interest.eq(4).all(axis=1)
]

sex_data = data['Sexe']  # Assuming 'Sexe' column exists in the original dataset
# Align sex_data with filtered_data indices
sex_data_aligned = sex_data.loc[filtered_data.index]

# Add 'Sex' column to filtered_data
filtered_data = filtered_data.assign(Sex=sex_data_aligned.values)

# Verify the column addition
# print(filtered_data.columns)




# Define a function to check the conditions with NaN handling
def check_conditions(row):
    conditions_failed = 0
    
    # Condition 1: Check 30th and 32nd columns
    if pd.notna(row.iloc[29]) and pd.notna(row.iloc[31]):
        if ((row.iloc[29] == 1 and row.iloc[31] == 4) or (row.iloc[29] == 4 and row.iloc[31] == 1)):
            conditions_failed += 1
    
    # Condition 2: Check 23rd and 34th columns
    if pd.notna(row.iloc[22]) and pd.notna(row.iloc[33]):
        if ((row.iloc[22] == 1 and row.iloc[33] == 4) or (row.iloc[22] == 4 and row.iloc[33] == 1)):
            conditions_failed += 1
    
    # Condition 3: Check 36th and 19th columns
    if pd.notna(row.iloc[35]) and pd.notna(row.iloc[18]):
        if ((row.iloc[35] == 1 and row.iloc[18] == 4) or (row.iloc[35] == 4 and row.iloc[18] == 1)):
            conditions_failed += 1
    
    # Omit the row if it fails at least 2 out of 3 conditions
    return conditions_failed < 2

# Apply the function to filter the DataFrame
final_filtered_data = filtered_data[filtered_data.apply(check_conditions, axis=1)]
# print(filtered_data.columns)
# Rearrange columns by dimensions: STA, PER, CAD, REL, DID
dimensions = ['STA', 'PER', 'CAD', 'REL', 'DID']
# excluding sex from the sorting
sorted_columns = sorted(
    [col for col in final_filtered_data.columns if col != 'Sex'], 
    key=lambda x: dimensions.index(x[-3:])
)
final_filtered_data = final_filtered_data[sorted_columns + ['Sex']]

# print("final ?",final_filtered_data)

# Define a color mapping for each dimension
dimension_colors = {
    'STA': 'dodgerblue',
    'PER': 'tomato',
    'CAD': 'mediumseagreen',
    'REL': 'gold',
    'DID': 'violet'
}

# Assign colors to columns based on their dimension
colors = [dimension_colors[col[-3:]] for col in sorted_columns]

# Plotting the bar chart of column means
column_means = final_filtered_data.loc[:, final_filtered_data.columns != 'Sex'].mean()

plt.figure(figsize=(14, 8))
column_means.plot(kind='bar', color=colors, alpha=0.8)
plt.title('Mean Values of Each Column by Dimensions', fontsize=18, fontweight='bold')
plt.xlabel('Columns', fontsize=14)
plt.ylabel('Mean Value', fontsize=14)
plt.xticks(rotation=90, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
plt.tight_layout()
# plt.show()
plt.savefig('bar_chart.png', dpi=300, bbox_inches='tight')


# Aggregate mean values by dimension
aggregate_means = {
    dim: final_filtered_data[[col for col in final_filtered_data.columns if col.endswith(dim)]].mean().mean()
    for dim in dimensions
}

# Create a list of colors for the aggregate dimensions
aggregate_colors = [dimension_colors[dim] for dim in dimensions]

# Convert the dictionary to a Series for easy plotting
aggregate_means_series = pd.Series(aggregate_means)

# Plotting the aggregate mean values by dimension
plt.figure(figsize=(10, 6))
aggregate_means_series.plot(kind='bar', color=aggregate_colors, alpha=0.8)
plt.title('Aggregate Mean Values by Dimensions', fontsize=18, fontweight='bold')
plt.xlabel('Dimensions', fontsize=14)
plt.ylabel('Aggregate Mean Value', fontsize=14)
plt.xticks(rotation=0, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
plt.tight_layout()
# plt.show()
plt.savefig('aggregated_bar_chart.png', dpi=300, bbox_inches='tight')

# print("Final:", final_filtered_data.columns)

aggregate_means_male = {
    dim: final_filtered_data[final_filtered_data['Sex'] == 'M'][
        [col for col in final_filtered_data.columns if col.endswith(dim)]
    ].mean().mean()
    for dim in dimensions
}

aggregate_means_female = {
    dim: final_filtered_data[final_filtered_data['Sex'] == 'F'][
        [col for col in final_filtered_data.columns if col.endswith(dim)]
    ].mean().mean()
    for dim in dimensions
}

# Convert the aggregated means into a DataFrame
aggregate_means_df = pd.DataFrame({
    'Male': aggregate_means_male,
    'Female': aggregate_means_female
})

# Plotting the bar chart
aggregate_means_df.plot(kind='bar', figsize=(10, 6), color=['pink', 'blue'], alpha=0.8)
plt.title('Aggregate Mean Values by Dimensions and Sex', fontsize=18, fontweight='bold')
plt.xlabel('Dimensions', fontsize=14)
plt.ylabel('Aggregate Mean Value', fontsize=14)
plt.xticks(rotation=0, fontsize=12)  # Rotate ticks for better label visibility
plt.yticks(fontsize=12)
plt.legend(title='Sex', fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
plt.tight_layout()
plt.savefig('aggregate_means_sex_barplot.png', dpi=300, bbox_inches='tight')
# plt.show()


# Radar Chart
# Prepare data for the radar chart
df = pd.DataFrame(list(aggregate_means.items()), columns=['Dimension', 'Mean'])
df['Angle'] = np.linspace(0, 2 * pi, len(df), endpoint=False)

# Append the start value to close the radar chart
df = pd.concat([df, df.iloc[[0]]]).reset_index(drop=True)

fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(8, 8))
ax.plot(df['Angle'], df['Mean'], 'o-', linewidth=2, label='Mean Value')
ax.fill(df['Angle'], df['Mean'], alpha=0.25)

# Set the dimension names as labels
ax.set_xticks(df['Angle'][:-1])
ax.set_xticklabels(df['Dimension'][:-1])

# Annotate values
for i, txt in enumerate(df['Mean']):
    ax.annotate(f"{txt:.2f}", (df['Angle'][i], df['Mean'][i]), textcoords="offset points", xytext=(0,10), ha='center')

plt.title('Mean Comparison of Dimensions', fontsize=16, fontweight='bold')
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
# plt.show()
plt.savefig('radar_chart.png', dpi=300, bbox_inches='tight')


# Clean data for one-way ANOVA

def clean_dimension(data, keyword):
    values = data.filter(like=keyword).values.flatten()
    numeric_values = pd.to_numeric(values, errors='coerce')
    return numeric_values[~pd.isna(numeric_values)]

STA = clean_dimension(data, "STA")
PER = clean_dimension(data, "PER")
CAD = clean_dimension(data, "CAD")
REL = clean_dimension(data, "REL")
DID = clean_dimension(data, "DID")

# Print number of observations
# print(f"STA: {len(STA)} observations")
# print(f"PER: {len(PER)} observations")
# print(f"CAD: {len(CAD)} observations")
# print(f"REL: {len(REL)} observations")
# print(f"DID: {len(DID)} observations")

"""""
# Conduct one-way ANOVA
f_value, p_value = stats.f_oneway(STA, PER, CAD, REL, DID)
# print(f"F-Value: {f_value}, P-Value: {p_value}")

# Detailed ANOVA table using statsmodels
anova_data = pd.DataFrame({
    'Value': np.concatenate([STA, PER, CAD, REL, DID]),
    'Group': (
        ['STA'] * len(STA) +
        ['PER'] * len(PER) +
        ['CAD'] * len(CAD) +
        ['REL'] * len(REL) +
        ['DID'] * len(DID)
    )
})

model = ols('Value ~ Group', data=anova_data).fit()
anova_table = sm.stats.anova_lm(model, typ=2)  # Type II ANOVA Table
print(anova_table)
with open('results.txt', 'w') as file:
    file.write("ANOVA Results:\n")
    file.write(anova_table.to_string())
    file.write("\n")

# Perform Tukey's HSD test
tukey = pairwise_tukeyhsd(endog=anova_data['Value'], groups=anova_data['Group'], alpha=0.05)
print(tukey.summary())

# Visualize Tukey's HSD results
tukey.plot_simultaneous(figsize=(8, 6))

# Append Tukey's HSD results to the file
with open('results.txt', 'a') as file:  # Open in append mode
    file.write("\nTukey's HSD Results:\n")
    file.write(tukey.summary().as_text())

# Reshape data for two ways ANOVA
anova_data = pd.melt(
    final_filtered_data, 
    id_vars=['Sex'], 
    value_vars=[col for col in final_filtered_data.columns if col != 'Sex'],
    var_name='Dimension', 
    value_name='Value'
)
anova_data.dropna(subset=['Value'], inplace=True)  # Remove NaN values

model = ols('Value ~ C(Sex) + C(Dimension) + C(Sex):C(Dimension)', data=anova_data).fit()
anova_results = anova_lm(model, typ=2)  # Type II ANOVA table
print(anova_results)

# Write the ANOVA table to a text file
with open('results.txt', 'a') as file:
    file.write("ANOVA Results:\n")
    file.write(anova_results.to_string())
    file.write("\n") """

############### Kruskal-Wallis on the five dimensions without other arguments ########


# 1) Kruskal–Wallis on the five dimensions
kw_stat, kw_pvalue = kruskal(STA, PER, CAD, REL, DID)
print(f"Kruskal–Wallis statistic = {kw_stat:.3f}, p-value = {kw_pvalue:.9e}")

# 2) If significant, do Dunn’s post-hoc test
# For Dunn’s, we need a single array of values plus an array of group labels:
anova_data = pd.DataFrame({
    'Value': np.concatenate([STA, PER, CAD, REL, DID]),
    'Group': (
        ['STA'] * len(STA) +
        ['PER'] * len(PER) +
        ['CAD'] * len(CAD) +
        ['REL'] * len(REL) +
        ['DID'] * len(DID)
    )
})

if kw_pvalue < 0.05:
    print("\nDunn’s Test (post-hoc) because Kruskal–Wallis is significant:")
    dunn_results = sp.posthoc_dunn(
        anova_data, 
        val_col='Value', 
        group_col='Group', 
        p_adjust='bonferroni'  # or 'holm', etc.
    )
    print(dunn_results)
else:
    print("No significant difference found by Kruskal–Wallis (p ≥ 0.05).")

""" 
# Step 1: Melt your DataFrame
long_df = pd.melt(
    final_filtered_data,
    id_vars=['Sex'],  # keep 'Sex' as an identifier
    var_name='Item',  # old column names
    value_name='Value'
)
# Now each row is: [Sex, Item, Value], where Item might be '1STA','5STA','10PER', etc.

# Step 2: Map each Item to a dimension suffix: 'STA','PER','CAD','REL','DID'
def get_dimension_suffix(item_name):
    # e.g. '1STA' -> 'STA'
    #      '10PER' -> 'PER'
    # Possibly do more robust checks if needed (e.g., item_name[-3:])
    return item_name[-3:]  # last 3 characters

long_df['Dimension'] = long_df['Item'].apply(get_dimension_suffix)

# Step 3: (Optional) Filter out rows that have other suffixes or NaNs
valid_dims = ['STA','PER','CAD','REL','DID']
long_df = long_df[long_df['Dimension'].isin(valid_dims)].copy()


# Suppose your long DataFrame is `long_df` with columns:
# 'Sex' and 'Dimension' (factors) and 'Value' (numeric response)
# You want a two-way nonparametric test for 'Sex' × 'Dimension' on 'Value'.

results = pg.scheirer_ray_hare(
    data=long_df,
    dv='Value',            # the numeric dependent variable
    between=['Sex','Dimension']  # the two factors
)

print(results) """

######### Classes #######

def map_classe_to_group(classe_value):
    """
    Returns '9' if the value starts with 9,
            '10' if it starts with 10,
            '11' if it starts with 11,
            np.nan otherwise.
    """
    if pd.notnull(classe_value):
        # Using regex or str.startswith to check the beginning:
        if str(classe_value).startswith("9"):
            return "9"
        elif str(classe_value).startswith("10"):
            return "10"
        elif str(classe_value).startswith("11"):
            return "11"
    return np.nan

final_filtered_data['ClassGroup'] = data['Classe'].apply(map_classe_to_group)

final_filtered_data = final_filtered_data.dropna(subset=['ClassGroup'])

counts = final_filtered_data['ClassGroup'].value_counts()
print("Counts by ClassGroup:")
print(counts)

dimensions = ['STA', 'PER', 'CAD', 'REL', 'DID']

grouped_means = {}  # Dictionary to store means for each subgroup
for group_label, group_df in final_filtered_data.groupby('ClassGroup'):
    # Calculate the mean for each dimension by flattening all columns that end with that dimension
    dim_means = {}
    for dim in dimensions:
        # Filter columns that end with the dimension keyword, then compute their overall mean
        dim_cols = [col for col in group_df.columns if col.endswith(dim)]
        dim_means[dim] = group_df[dim_cols].mean().mean()
    grouped_means[group_label] = dim_means
    
# Convert to a DataFrame for easier inspection / plotting
grouped_means_df = pd.DataFrame(grouped_means)
print("Means of each dimension for each ClassGroup (9, 10, 11):")
print(grouped_means_df)


def melt_subgroup(data, subgroup_label):
    """
    Filters 'data' to only rows with ClassGroup == subgroup_label.
    Gathers all columns ending in each dimension
    into a single long-format DataFrame:
    Columns: ['Value', 'Dimension', 'ClassGroup'].
    """
    # 1) Filter to keep only this group's rows
    subgroup_data = data[data['ClassGroup'] == subgroup_label].copy()
    
    # 2) Create a list of melted DataFrames—one per dimension
    melted_parts = []
    for dim in dimensions:
        # Gather all columns that end with the dimension suffix
        dim_cols = [col for col in subgroup_data.columns if col.endswith(dim)]
        
        # Melt these columns into rows -> (variable, value)
        melted = subgroup_data[dim_cols].melt(var_name='OriginalColumn', value_name='Value')
        
        # Tag the dimension
        melted['Dimension'] = dim
        
        # We only keep the 'Value' and 'Dimension' columns
        # (Though you could keep OriginalColumn if you want to know from which question it came.)
        melted_parts.append(melted[['Value', 'Dimension']])
    
    # 3) Concatenate all dimension data
    melted_df = pd.concat(melted_parts, axis=0, ignore_index=True)
    
    # 4) Remove NaN rows
    melted_df.dropna(subset=['Value'], inplace=True)
    
    # 5) Add the subgroup label
    melted_df['ClassGroup'] = subgroup_label
    
    return melted_df

# final_filtered_data should already have the "ClassGroup" column with values '9', '10', or '11'.
melted_9 = melt_subgroup(final_filtered_data, '9')
melted_10 = melt_subgroup(final_filtered_data, '10')
melted_11 = melt_subgroup(final_filtered_data, '11')

""" 
def run_anova_and_tukey(melted_df, subgroup_label):
    print(f"\n=== ANOVA for subgroup '{subgroup_label}' ===")
    
    # 1) Fit a one-way ANOVA model: Value ~ C(Dimension)
    #    We treat Dimension as a categorical factor
    model = ols('Value ~ C(Dimension)', data=melted_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)  # Type II ANOVA
    print(anova_table)
    
    # 2) Tukey HSD for pairwise comparisons among dimensions
    tukey = pairwise_tukeyhsd(endog=melted_df['Value'], 
                              groups=melted_df['Dimension'], 
                              alpha=0.05)
    print("\n--- Tukey HSD results ---")
    print(tukey.summary())
    
    # Optional: Plot confidence intervals
    # import matplotlib.pyplot as plt
    # tukey.plot_simultaneous()
    # plt.show()

# Now run the function for each subgroup
run_anova_and_tukey(melted_9,  '9')
run_anova_and_tukey(melted_10, '10')
run_anova_and_tukey(melted_11, '11') """


def run_kruskal_and_dunn(melted_df, subgroup_label):
    print(f"\n=== Kruskal–Wallis for subgroup '{subgroup_label}' ===")
    
    # 1) We need to gather 'Value' arrays for each dimension
    #    e.g. all STA values in one list, all PER in another, etc.
    dimension_groups = []
    unique_dims = melted_df['Dimension'].unique()
    
    for dim in unique_dims:
        # Extract all 'Value' entries for that dimension
        vals = melted_df.loc[melted_df['Dimension'] == dim, 'Value']
        dimension_groups.append(vals.dropna())
    
    # 2) Kruskal–Wallis test
    kw_stat, kw_pvalue = kruskal(*dimension_groups)
    print(f"Kruskal–Wallis statistic = {kw_stat:.4f}, p-value = {kw_pvalue:.4e}")
    
    # 3) If significant, do Dunn’s post-hoc test
    if kw_pvalue < 0.05:
        print("\n--- Dunn’s Test (post-hoc) ---")
        # scikit-posthocs has a direct method for Dunn’s test:
        #   posthoc_dunn() expects a DataFrame, plus 'val_col' and 'group_col'
        #   specifying which columns contain numeric values and group labels.

        dunn_result = sp.posthoc_dunn(
            melted_df, 
            val_col='Value',     # Column name for numeric data
            group_col='Dimension',  # Column name for groups
            p_adjust='bonferroni'   # or 'holm', 'fdr_bh', etc.
        )
        print(dunn_result)
    else:
        print("No significant difference found by Kruskal–Wallis (p ≥ 0.05).")
        
run_kruskal_and_dunn(melted_9,  '9')
run_kruskal_and_dunn(melted_10, '10')
run_kruskal_and_dunn(melted_11, '11')

def compute_dimension_means_for_group(data, group_label):
    """
    Given a DataFrame and a class group (e.g. '9'), this function
    computes the aggregate mean of each dimension's columns within that group.
    """
    group_data = data[data['ClassGroup'] == group_label]
    means_dict = {}
    for dim in dimensions:
        dim_cols = [col for col in group_data.columns if col.endswith(dim)]
        if len(dim_cols) > 0:
            means_dict[dim] = group_data[dim_cols].mean().mean()
        else:
            # If no columns found for some reason, default to NaN
            means_dict[dim] = np.nan
    return means_dict

def plot_radar_chart(means_dict, group_label):
    """
    Plots a radar chart for the dimension means in 'means_dict'
    and saves it to a file (e.g. 'radar_chart_9.png').
    """
    # Convert the dictionary to a DataFrame with columns ['Dimension', 'Mean']
    df = pd.DataFrame(list(means_dict.items()), columns=['Dimension', 'Mean'])
    
    # Create an angle for each dimension
    df['Angle'] = np.linspace(0, 2 * pi, len(df), endpoint=False)
    
    # Append the first row at the bottom so the radar is "closed"
    df = pd.concat([df, df.iloc[[0]]]).reset_index(drop=True)
    
    fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(8, 8))
    
    # Plot the data
    ax.plot(df['Angle'], df['Mean'], 'o-', linewidth=2, label='Mean Value')
    ax.fill(df['Angle'], df['Mean'], alpha=0.25)
    
    # Set dimension labels around the circle (excluding the duplicated last row)
    ax.set_xticks(df['Angle'][:-1])
    ax.set_xticklabels(df['Dimension'][:-1])
    
    # Annotate each point with its numeric value
    for i, txt in enumerate(df['Mean']):
        ax.annotate(f"{txt:.2f}", 
                    (df['Angle'][i], df['Mean'][i]), 
                    textcoords="offset points", 
                    xytext=(0, 10), 
                    ha='center')
    
    # Title and legend
    plt.title(f"Mean Comparison of Dimensions for ClassGroup {group_label}",
              fontsize=16, fontweight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Save (and/or show) the figure
    plt.savefig(f'radar_chart_{group_label}.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure so it won't pop up repeatedly

# ----------------------------
# Main Execution
# ----------------------------
# Suppose 'final_filtered_data' is your cleaned DataFrame 
# which already has the "ClassGroup" column = ['9','10','11'].

# For each class group, compute dimension means and plot
for grp in ['9', '10', '11']:
    group_means = compute_dimension_means_for_group(final_filtered_data, grp)
    plot_radar_chart(group_means, grp)
    

# 1) Identify all columns ending with STA, PER, CAD, REL, DID
dimension_suffixes = ('STA', 'PER', 'CAD', 'REL', 'DID')
dimension_cols = [
    col for col in final_filtered_data.columns 
    if col.endswith(dimension_suffixes)  # ends with any of the dimension suffixes
]

# 2) Compute the "overall score" as the row-wise mean of these columns.
final_filtered_data['OverallScore'] = final_filtered_data[dimension_cols].mean(axis=1)

# 3) Optional: Inspect the first few rows
#print(final_filtered_data[['ClassGroup', 'OverallScore']])

# Suppose final_filtered_data has columns:
# 'ClassGroup' in ['9','10','11'] and 'OverallScore' (or similar).

group_9  = final_filtered_data.loc[final_filtered_data['ClassGroup'] == '9',  'OverallScore'].dropna()
group_10 = final_filtered_data.loc[final_filtered_data['ClassGroup'] == '10', 'OverallScore'].dropna()
group_11 = final_filtered_data.loc[final_filtered_data['ClassGroup'] == '11', 'OverallScore'].dropna()

""" # One-way ANOVA
anova_res = stats.f_oneway(group_9, group_10, group_11)
print(f"ANOVA F={anova_res.statistic:.3f}, p={anova_res.pvalue:.4f}")

if anova_res.pvalue < 0.05:
    # Tukey HSD
    all_vals = pd.concat([group_9, group_10, group_11], ignore_index=True)
    groups   = (['9'] * len(group_9)
              + ['10'] * len(group_10)
              + ['11'] * len(group_11))
    tukey = pairwise_tukeyhsd(endog=all_vals, groups=groups, alpha=0.05)
    print(tukey.summary()) """


# 1) Kruskal–Wallis
kw_stat, kw_pvalue = kruskal(group_9, group_10, group_11)
print(f"Kruskal–Wallis (comparing the 3 subClasses againt each other) = {kw_stat:.3f}, p-value = {kw_pvalue:.4f}.")

# 2) If significant, do Dunn’s
if kw_pvalue < 0.05:
    # We combine all scores into a single series, plus a parallel group label
    all_scores = pd.concat([group_9, group_10, group_11], ignore_index=True)
    labels = (['9'] * len(group_9)
            + ['10'] * len(group_10)
            + ['11'] * len(group_11))
    
    # We can create a small DataFrame to pass into posthoc_dunn
    df_overall = pd.DataFrame({'Score': all_scores, 'Group': labels})
    
    print("\nDunn’s Test Results:")
    dunn_result = sp.posthoc_dunn(
        df_overall, 
        val_col='Score',   # numeric column
        group_col='Group', # group labels
        p_adjust='bonferroni'
    )
    print(dunn_result)
else:
    print("No significant difference found among '9','10','11' by Kruskal–Wallis.")