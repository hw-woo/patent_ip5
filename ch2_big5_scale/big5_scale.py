# Patent Scaling for Big 5 (Kyoungown Kim and Hyunwoo Woo)
# 2024-09-21

import pandas as pd
import numpy as np
import pickle
import os
import datetime
from datetime import datetime
import glob
import warnings
warnings.filterwarnings(action='ignore')

# import big5_scale as bs_kw

############################################################################################
# 1. Make scale values for each country X year X {application-based vs. publication-based} #
############################################################################################
def make_scale(select_country, year_start, year_end): 
    # (Code Example) final_df_appln, final_df_appln_count, final_df_appln_grantY, final_df_appln_grantY_count, final_df_publn, final_df_publn_count, final_df_publn_grantY, final_df_publn_grantY_count 
    # = bs_kw.make_scale("US", 1975, 2023)
    # = bs_kw.make_scale("EP", 1978, 2023)
    # = bs_kw.make_scale("CN", 1994, 2023)
    # = bs_kw.make_scale("KR", 1990, 2023)
    # = bs_kw.make_scale("JP", 1975, 2023)

    # Required Files (N): all pkl files in one country folder (e.g., {pat_IPC4_allC_us_1975.pkl, pat_IPC4_allC_us_1976.pkl, ..., pat_IPC4_allC_us_2022.pkl})
    initial_path = os.getcwd()
    ### 1. Patent Filing Year (appln_filing_year) based ###
    print(datetime.now())
    folder_path = f'./patstat_each_nation_year_vars/{select_country.upper()}' # Set importing Path
    os.chdir(folder_path)
    print(os.getcwd())
    file_paths = glob.glob(f'{folder_path}/*.pkl')

    # Initialize empty lists to store results
    all_df_appln, all_df_appln_count, all_df_appln_grantY, all_df_appln_grantY_count = [], [], [], []

    # Process each file and accumulate the results
    for file_path in file_paths:
        df_appln, df_appln_count, df_appln_grantY, df_appln_grantY_count = repeat_file_appln(file_path)
        all_df_appln.append(df_appln)
        all_df_appln_count.append(df_appln_count)
        all_df_appln_grantY.append(df_appln_grantY)
        all_df_appln_grantY_count.append(df_appln_grantY_count)
        print("@ ", datetime.now(), "@ ", file_path)

    # Concatenate all results into single DataFrames
    final_df_appln = pd.concat(all_df_appln, ignore_index=True)
    final_df_appln_count = pd.concat(all_df_appln_count, ignore_index=True)
    final_df_appln_grantY = pd.concat(all_df_appln_grantY, ignore_index=True)
    final_df_appln_grantY_count = pd.concat(all_df_appln_grantY_count, ignore_index=True)


    ### 2. Patent Publication Year (earliest_publn_year) based ###
    folder_path = f'./patstat_each_nation_year_vars/{select_country}' #Set importing Path
    years = range(year_start, year_end) # Set year range

    # Initialize empty lists to store results
    all_df_publn, all_df_publn_count, all_df_publn_grantY, all_df_publn_grantY_count = [], [], [], []

    # Process each file and accumulate the results
    for year in years:
        df_publn, df_publn_count, df_publn_grantY, df_publn_grantY_count = repeat_file_publn(folder_path, year)
        all_df_publn.append(df_publn)
        all_df_publn_count.append(df_publn_count)
        all_df_publn_grantY.append(df_publn_grantY)
        all_df_publn_grantY_count.append(df_publn_grantY_count)
        print("@ ", datetime.now(), "@ ", year)

    # Concatenate all results into single DataFrames
    final_df_publn = pd.concat(all_df_publn, ignore_index=True)
    final_df_publn_count = pd.concat(all_df_publn_count, ignore_index=True)
    final_df_publn_grantY = pd.concat(all_df_publn_grantY, ignore_index=True)
    final_df_publn_grantY_count = pd.concat(all_df_publn_grantY_count, ignore_index=True)
    
    print(datetime.now())

    # Save
    os.chdir('../../')
    folder_path = f'./scale_results_big5_each' # Set saving Path
    os.chdir(folder_path) # Change to the saving directory
    final_df_appln.to_pickle(f'{select_country.lower()}_patstatvar_appln.pkl')
    final_df_appln_count.to_pickle(f'{select_country.lower()}_patstatvar_appln_count.pkl')
    final_df_appln_grantY.to_pickle(f'{select_country.lower()}_patstatvar_appln_grantY.pkl')
    final_df_appln_grantY_count.to_pickle(f'{select_country.lower()}_patstatvar_appln_grantY_count.pkl')
    
    final_df_publn.to_pickle(f'{select_country.lower()}_patstatvar_publn.pkl')
    final_df_publn_count.to_pickle(f'{select_country.lower()}_patstatvar_publn_count.pkl')
    final_df_publn_grantY.to_pickle(f'{select_country.lower()}_patstatvar_publn_grantY.pkl')
    final_df_publn_grantY_count.to_pickle(f'{select_country.lower()}_patstatvar_publn_grantY_count.pkl')

    os.chdir(initial_path)
    print("Reverted to initial directory:", os.getcwd())
    return final_df_appln, final_df_appln_count, final_df_appln_grantY, final_df_appln_grantY_count, final_df_publn, final_df_publn_count, final_df_publn_grantY, final_df_publn_grantY_count


############################################################################################
# 2. Make scale values for each country X year X {application-based vs. publication-based} #
############################################################################################
def nber_match_field(select_country, year_start, year_end): 
    # = bs_kw.nber_match_field("US", 1975, 2023)
    # = bs_kw.nber_match_field("EP", 1978, 2023)
    # = bs_kw.nber_match_field("CN", 1994, 2023)
    # = bs_kw.nber_match_field("KR", 1990, 2023)
    # = bs_kw.nber_match_field("JP", 1975, 2023) 
    
    print(datetime.now())
    initial_path = os.getcwd()
    
    NBER_match = pd.read_csv('NBER_Match_IPC_Field_240330.csv')
    NBER_match.rename(columns={'ipc_code':'ipc_sub_3'}, inplace = True)
    NBER_match
    
    ### 1. Patent Filing Year (appln_filing_year) based ###
    folder_path = f'./patstat_each_nation_year_vars/{select_country.upper()}' # Set importing Path
    os.chdir(folder_path)
    file_paths = glob.glob(f'{folder_path}/*.pkl')
    
    # Initialize empty lists to store results
    all_df_appln, all_df_appln_count, all_df_appln_grantY, all_df_appln_grantY_count = [], [], [], []

    # Process each file and accumulate the results
    for file_path in file_paths:
        df_appln, df_appln_count, df_appln_grantY, df_appln_grantY_count = repeat_file_appln_nber(file_path)
        all_df_appln.append(df_appln)
        all_df_appln_count.append(df_appln_count)
        all_df_appln_grantY.append(df_appln_grantY)
        all_df_appln_grantY_count.append(df_appln_grantY_count)
        print("@ ", datetime.now(), "@ ", file_path)

    # Concatenate all results into single DataFrames
    final_df_appln = pd.concat(all_df_appln, ignore_index=True)
    final_df_appln_count = pd.concat(all_df_appln_count, ignore_index=True)
    final_df_appln_grantY = pd.concat(all_df_appln_grantY, ignore_index=True)
    final_df_appln_grantY_count = pd.concat(all_df_appln_grantY_count, ignore_index=True)
    
    ### 2. Patent Publication Year (earliest_publn_year) based ###
    years = range(year_start, year_end) #Set year range
    
    # Initialize empty lists to store results
    all_df_publn, all_df_publn_count, all_df_publn_grantY, all_df_publn_grantY_count = [], [], [], []

    # Process each file and accumulate the results
    for year in years:
        df_publn, df_publn_count, df_publn_grantY, df_publn_grantY_count = repeat_file_publn_nber(folder_path, year)
        all_df_publn.append(df_publn)
        all_df_publn_count.append(df_publn_count)
        all_df_publn_grantY.append(df_publn_grantY)
        all_df_publn_grantY_count.append(df_publn_grantY_count)
        print("@ ", datetime.now(), "@ ", year)

    # Concatenate all results into single DataFrames
    final_df_publn = pd.concat(all_df_publn, ignore_index=True)
    final_df_publn_count = pd.concat(all_df_publn_count, ignore_index=True)
    final_df_publn_grantY = pd.concat(all_df_publn_grantY, ignore_index=True)
    final_df_publn_grantY_count = pd.concat(all_df_publn_grantY_count, ignore_index=True)
    
    print(datetime.now())
    os.chdir(initial_path)
    os.chdir('./nber_field_match') # Write Your dataset Path
    print(os.getcwd())
    
    final_df_appln.to_pickle(f'{select_country.lower()}_patstatvar_appln_NBER.pkl')
    final_df_appln_count.to_pickle(f'{select_country.lower()}_appln_count_NBER.pkl')
    final_df_appln_grantY.to_pickle(f'{select_country.lower()}_appln_grantY_NBER.pkl')
    final_df_appln_grantY_count.to_pickle(f'{select_country.lower()}_appln_grantY_count_NBER.pkl')
    final_df_publn.to_pickle(f'{select_country.lower()}_publn_NBER.pkl')
    final_df_publn_count.to_pickle(f'{select_country.lower()}_publn_count_NBER.pkl')
    final_df_publn_grantY.to_pickle(f'{select_country.lower()}_publn_grantY_NBER.pkl')
    final_df_publn_grantY_count.to_pickle(f'{select_country.lower()}_publn_grantY_count_NBER.pkl')
    
    os.chdir(initial_path)
    print("Reverted to initial directory:", os.getcwd())
    
    return final_df_appln, final_df_appln_count, final_df_appln_grantY, final_df_appln_grantY_count, final_df_publn, final_df_publn_count, final_df_publn_grantY, final_df_publn_grantY_count
    

#################################################
# 3. Combine the IP5-patents into one dataframe #
#################################################
def combine_nber():
    # bs_kw.combine_nber()

    print(datetime.now())
    initial_path = os.getcwd()
    
    ### 1. Patent Filing Year (appln_filing_year) based ###
    os.chdir('./nber_field_match') # Path
    print(os.getcwd())
    df_us = pd.read_pickle('us_patstatvar_appln_NBER.pkl')
    df_ep = pd.read_pickle('ep_patstatvar_appln_NBER.pkl')
    df_kr = pd.read_pickle('kr_patstatvar_appln_NBER.pkl')
    df_cn = pd.read_pickle('cn_patstatvar_appln_NBER.pkl')
    df_jp = pd.read_pickle('jp_patstatvar_appln_NBER.pkl')
    
    df_us["nation"] = "US"
    df_ep["nation"] = "EP"
    df_kr["nation"] = "KR"
    df_cn["nation"] = "CN"
    df_jp["nation"] = "JP"
    
    df_total = pd.concat([df_us, df_ep, df_kr, df_cn, df_jp]).reset_index(drop = True)
    
    df_total["NBER_no"] = df_total["NBER"]
    change_value_dict = {1: "Chemical", 
                         2: "Computer and Communications", 
                         3: "Pharmaceuticals and Medical", 
                         4: "Electrical and Electronics", 
                         5: "Mechanical", 
                         6: "Others"
                         }
    df_total = df_total.replace({"NBER": change_value_dict})
    
    # Save
    os.chdir(initial_path)
    os.chdir('../output/nber_match') # Path
    print(os.getcwd())
    df_total.to_pickle("big5_pat_appln_nber.pkl") # Dataset based on each patent application
    
    ### 2. Patent Publication Year (earliest_publn_year) based ###
    os.chdir(initial_path)
    os.chdir('./nber_field_match') # Path
    print(os.getcwd())
    df_us = pd.read_pickle('us_patstatvar_publn_NBER.pkl')
    df_ep = pd.read_pickle('ep_patstatvar_publn_NBER.pkl')
    df_kr = pd.read_pickle('kr_patstatvar_publn_NBER.pkl')
    df_cn = pd.read_pickle('cn_patstatvar_publn_NBER.pkl')
    df_jp = pd.read_pickle('jp_patstatvar_publn_NBER.pkl')
    
    df_us["nation"] = "US"
    df_ep["nation"] = "EP"
    df_kr["nation"] = "KR"
    df_cn["nation"] = "CN"
    df_jp["nation"] = "JP"
    
    df_total = pd.concat([df_us, df_ep, df_kr, df_cn, df_jp]).reset_index(drop = True)
    df_total["NBER_no"] = df_total["NBER"]
    change_value_dict = {1: "Chemical", 
                         2: "Computer and Communications", 
                         3: "Pharmaceuticals and Medical", 
                         4: "Electrical and Electronics", 
                         5: "Mechanical", 
                         6: "Others"
                         }

    df_total = df_total.replace({"NBER": change_value_dict})
    df_total
    
    # Save
    os.chdir(initial_path)
    os.chdir('../output/nber_match') # Path
    print(os.getcwd())
    df_total.to_pickle("big5_pat_publn_nber.pkl") # Dataset based on each patent publication
    
    ### 3. Patent Filing Year (appln_filing_year) based: Only Granted (Y) ###
    os.chdir(initial_path)
    os.chdir('./nber_field_match') # Path
    print(os.getcwd())
    df_us = pd.read_pickle('us_patstatvar_appln_grantY_NBER.pkl')
    df_ep = pd.read_pickle('ep_patstatvar_appln_grantY_NBER.pkl')
    df_kr = pd.read_pickle('kr_patstatvar_appln_grantY_NBER.pkl')
    df_cn = pd.read_pickle('cn_patstatvar_appln_grantY_NBER.pkl')
    df_jp = pd.read_pickle('jp_patstatvar_appln_grantY_NBER.pkl')

    df_us["nation"] = "US"
    df_ep["nation"] = "EP"
    df_kr["nation"] = "KR"
    df_cn["nation"] = "CN"
    df_jp["nation"] = "JP"
    
    df_total = pd.concat([df_us, df_ep, df_kr, df_cn, df_jp]).reset_index(drop = True)
    df_total["NBER_no"] = df_total["NBER"]
    change_value_dict = {1: "Chemical", 
                         2: "Computer and Communications", 
                         3: "Pharmaceuticals and Medical", 
                         4: "Electrical and Electronics", 
                         5: "Mechanical", 
                         6: "Others"
                        }

    df_total = df_total.replace({"NBER": change_value_dict})
    df_total
    
    # Save
    os.chdir(initial_path)
    os.chdir('../output/nber_match') # Path
    print(os.getcwd())
    df_total.to_pickle("big5_pat_appln_grantY_nber.pkl") # Dataset based on each patent application

    ### 4. Patent Publication Year (earliest_publn_year) based: Only Granted (Y) ###
    os.chdir(initial_path)
    os.chdir('./nber_field_match') # Path
    print(os.getcwd())
    df_us = pd.read_pickle('us_patstatvar_publn_grantY_NBER.pkl')
    df_ep = pd.read_pickle('ep_patstatvar_publn_grantY_NBER.pkl')
    df_kr = pd.read_pickle('kr_patstatvar_publn_grantY_NBER.pkl')
    df_cn = pd.read_pickle('cn_patstatvar_publn_grantY_NBER.pkl')
    df_jp = pd.read_pickle('jp_patstatvar_publn_grantY_NBER.pkl')
    
    df_us["nation"] = "US"
    df_ep["nation"] = "EP"
    df_kr["nation"] = "KR"
    df_cn["nation"] = "CN"
    df_jp["nation"] = "JP"
    
    df_total = pd.concat([df_us, df_ep, df_kr, df_cn, df_jp]).reset_index(drop = True)
    df_total["NBER_no"] = df_total["NBER"]
    change_value_dict = {1: "Chemical", 
                         2: "Computer and Communications", 
                         3: "Pharmaceuticals and Medical", 
                         4: "Electrical and Electronics", 
                         5: "Mechanical", 
                         6: "Others"
                        }

    df_total = df_total.replace({"NBER": change_value_dict})
    df_total

    # Save
    os.chdir(initial_path)
    os.chdir('../output/nber_match') # Path
    print(os.getcwd())
    df_total.to_pickle("big5_pat_publn_grantY_nber.pkl") # Dataset based on each patent publication
    
    print(datetime.now())
    os.chdir(initial_path)


#########################################################
### Sub functions (in the main functions: make_scale) ###
#########################################################

def calculate_group_stats(df, group_columns, values, granted1, granted2):
    # Filter the DataFrame based on the 'granted' column values
    filtered_df = df[(df['granted'] == granted1) | (df['granted'] == granted2)]
    
    # Add 'count' as an aggregation operation for the first column in values
    # It doesn't matter which column you count because you're counting rows in each group
    agg_operations = {col: ['mean', 'std'] for col in values}
    agg_operations[values[-1]].append('count')  # Add count operation for the last column ('count') in the list
    
    # Group the filtered DataFrame by the specified columns and calculate mean, std, and count for the specified columns
    grouped = filtered_df.groupby(group_columns).agg(agg_operations)
    
    # Flatten the MultiIndex in columns and adjust new column names for mean, std, and count
    grouped.columns = [f'{col[0]}_{col[1]}' if col[1] != 'count' else 'count' for col in grouped.columns]
    
    # Reset the index to turn the grouping columns back into regular columns
    grouped = grouped.copy().reset_index()
    
    # Arrange the columns to ensure 'count' appears right after the grouping columns
    # The exact order will depend on the structure of your grouped DataFrame
    # This step might need adjustment based on your specific DataFrame structure
    final_columns = group_columns + ['count'] + [col for col in grouped.columns if col not in group_columns and col != 'count']
    grouped = grouped[final_columns].drop(columns = ['count_mean', 'count_std']).copy().reset_index(drop = True)
    
    return grouped


def count_group(df, group_column, values, granted1, granted2):
    # Filter the DataFrame based on the 'granted' column values
    filtered_df = df[(df['granted'] == granted1) | (df['granted'] == granted2)]
    
    # Group the DataFrame by the specified column and calculate mean and std
    grouped = filtered_df.groupby(group_column)[values].agg(['count'])
    
    # Create new column names by appending '_mean' and '_std'
    # The resulting column names will be a MultiIndex. The following lines of code
    # iterate over the levels of the MultiIndex to create the new column names.
    new_columns = [f'{col[0]}_{col[1]}' for col in grouped.columns]
    
    # Assign the new column names to the DataFrame
    grouped.columns = new_columns
    
    # Optionally, reset the index if you want the grouping column to be a regular column in the output
    grouped = grouped.drop(columns = ['count_count']).copy().reset_index()
    
    return grouped


def repeat_file_appln(file_path):
    df = pd.read_pickle(file_path)
    df = df[(df['appln_auth'] == df['publn_auth']) & (df['appln_auth'].isnull() == False) & (df['publn_auth'].isnull() == False)]
    df['count'] = 1 # for counting
    column_list = df.columns.tolist()
    column_list_for_groupby = [e for e in column_list if e not in ('appln_filing_year', 'earliest_publn_year', 'appln_id', 'publn_nr', 'ipc_sub', 'granted', 'appln_auth', 'publn_auth')]
    
    # Application based & regardless of grant
    df_appln = calculate_group_stats(df, ['appln_filing_year', 'ipc_sub'], column_list_for_groupby, 'N', 'Y')
    df_appln_count = count_group(df, ['appln_filing_year', 'ipc_sub'], column_list_for_groupby, 'N', 'Y')

    # Application based & grant == Y
    df_appln_grantY = calculate_group_stats(df, ['appln_filing_year', 'ipc_sub'], column_list_for_groupby, 'Y', 'Y')
    df_appln_grantY_count = count_group(df, ['appln_filing_year', 'ipc_sub'], column_list_for_groupby, 'Y', 'Y')
    
    del df
    
    return df_appln, df_appln_count, df_appln_grantY, df_appln_grantY_count


def repeat_file_publn(folder_path, year):

    # Initialize an empty list to store DataFrames
    dfs = []

    # Loop through the files, read each into a DataFrame, and append to the list
    folder_path = folder_path
    files = glob.glob(f'{folder_path}/*.pkl')
    
    for file in files:
        df_year = pd.read_pickle(file)
        df_year = df_year[df_year['earliest_publn_year'] == year].copy()
        dfs.append(df_year)
        del df_year

    # Concatenate all the DataFrames in the list vertically
    df = pd.concat(dfs, ignore_index=True)
    del dfs

    df = df[(df['appln_auth'] == df['publn_auth']) & (df['appln_auth'].isnull() == False) & (df['publn_auth'].isnull() == False)]
    df['count'] = 1 # for counting
    column_list = df.columns.tolist()
    column_list_for_groupby = [e for e in column_list if e not in ('appln_filing_year', 'earliest_publn_year', 'appln_id', 'publn_nr', 'ipc_sub', 'granted', 'appln_auth', 'publn_auth')]
    
    # publication based & regardless of grant
    df_publn = calculate_group_stats(df, ['earliest_publn_year', 'ipc_sub'], column_list_for_groupby, 'N', 'Y')
    df_publn_count = count_group(df, ['earliest_publn_year', 'ipc_sub'], column_list_for_groupby, 'N', 'Y')

    # publication based & grant == Y
    df_publn_grantY = calculate_group_stats(df, ['earliest_publn_year', 'ipc_sub'], column_list_for_groupby, 'Y', 'Y')
    df_publn_grantY_count = count_group(df, ['earliest_publn_year', 'ipc_sub'], column_list_for_groupby, 'Y', 'Y')
    
    del df
    
    return df_publn, df_publn_count, df_publn_grantY, df_publn_grantY_count


###############################################################
### Sub functions (in the main functions: nber_match_field) ###
###############################################################

def repeat_file_appln_nber(file_path):
    df = pd.read_pickle(file_path)
    
    # Field Match for NBER #
    df['ipc_sub_3'] = df['ipc_sub'].copy().str.slice(start=0, stop=3)
    df = df[(df['ipc_sub_3'].isin(NBER_match['ipc_sub_3'])) | (df['ipc_sub'].isin(NBER_match['ipc_sub_3']))].reset_index(drop = True)

    df = pd.merge(df.copy(), NBER_match, on = 'ipc_sub_3', how = 'left')
    df["NBER"] = np.where(df["ipc_sub"]=="A01N", 1, df["NBER"])
    df.drop(["ipc_sub_3"], axis = 1, inplace=True)
    # Field Match for NBER #
    
    df = df[(df['appln_auth'] == df['publn_auth']) & (df['appln_auth'].isnull() == False) & (df['publn_auth'].isnull() == False)]
    df['count'] = 1 # count 하기 위해서 추가
    column_list = df.columns.tolist()
    column_list_for_groupby = [e for e in column_list if e not in ('appln_filing_year', 'earliest_publn_year', 
                                                                   'appln_id', 'publn_nr', 'ipc_sub', 'granted', 
                                                                   'appln_auth', 'publn_auth', 'NBER')]
    
    #출원연도 기준 & grant 유무 무관
    df_appln = calculate_group_stats(df, ['appln_filing_year', 'NBER'], column_list_for_groupby, 'N', 'Y')
    df_appln_count = count_group(df, ['appln_filing_year', 'NBER'], column_list_for_groupby, 'N', 'Y')

    #출원연도 기준 & grant == Y
    df_appln_grantY = calculate_group_stats(df, ['appln_filing_year', 'NBER'], column_list_for_groupby, 'Y', 'Y')
    df_appln_grantY_count = count_group(df, ['appln_filing_year', 'NBER'], column_list_for_groupby, 'Y', 'Y')
    
    del df
    
    return df_appln, df_appln_count, df_appln_grantY, df_appln_grantY_count


def repeat_file_publn_nber(folder_path, year):

    # Initialize an empty list to store DataFrames
    dfs = []

    # Loop through the files, read each into a DataFrame, and append to the list
    folder_path = folder_path
    files = glob.glob(f'{folder_path}/*.pkl')
    
    for file in files:
        df_year = pd.read_pickle(file)
        df_year = df_year[df_year['earliest_publn_year'] == year].copy()
        dfs.append(df_year)
        del df_year

    # Concatenate all the DataFrames in the list vertically
    df = pd.concat(dfs, ignore_index=True)
    del dfs

    # Field Match for NBER #
    df['ipc_sub_3'] = df['ipc_sub'].copy().str.slice(start=0, stop=3)
    df = df[(df['ipc_sub_3'].isin(NBER_match['ipc_sub_3'])) | (df['ipc_sub'].isin(NBER_match['ipc_sub_3']))].reset_index(drop = True)

    df = pd.merge(df.copy(), NBER_match, on = 'ipc_sub_3', how = 'left')
    df["NBER"] = np.where(df["ipc_sub"]=="A01N", 1, df["NBER"])
    df.drop(["ipc_sub_3"], axis = 1, inplace=True)
    # Field Match for NBER #
    
    df = df[(df['appln_auth'] == df['publn_auth']) & (df['appln_auth'].isnull() == False) & (df['publn_auth'].isnull() == False)]
    df['count'] = 1 # count 하기 위해서 추가
    column_list = df.columns.tolist()
    column_list_for_groupby = [e for e in column_list if e not in ('appln_filing_year', 'earliest_publn_year', 
                                                                   'appln_id', 'publn_nr', 'ipc_sub', 'granted', 
                                                                   'appln_auth', 'publn_auth', 'NBER')]
    
    #출원연도 기준 & grant 유무 무관
    df_publn = calculate_group_stats(df, ['earliest_publn_year', 'NBER'], column_list_for_groupby, 'N', 'Y')
    df_publn_count = count_group(df, ['earliest_publn_year', 'NBER'], column_list_for_groupby, 'N', 'Y')

    #출원연도 기준 & grant == Y
    df_publn_grantY = calculate_group_stats(df, ['earliest_publn_year', 'NBER'], column_list_for_groupby, 'Y', 'Y')
    df_publn_grantY_count = count_group(df, ['earliest_publn_year', 'NBER'], column_list_for_groupby, 'Y', 'Y')
    
    del df
    
    return df_publn, df_publn_count, df_publn_grantY, df_publn_grantY_count
