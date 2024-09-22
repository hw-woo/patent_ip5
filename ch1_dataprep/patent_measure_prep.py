# Patent Data (Kyoungown Kim and Hyunwoo Woo)
# 2024-09-13

import pandas as pd
import numpy as np
import pickle
import os
from collections import Counter
import time
from datetime import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta

# import patent_measure_prep as pmp_kw

###############################################
# 1. IPC Code Cleansing: Make an IPC subclass #
###############################################
def ipc_filter(): # df_ipc_simple = pmp_kw.ipc_filter()
    # Required File (1): "table4_ipc.pkl"
    initial_path = os.getcwd()
    
    tot_start_time = time.time() # Record the start time
    tot_start = time.strftime('%Y.%m.%d - %H:%M:%S')
    
    PATH_ipc_filter = './patstat'
    os.chdir(PATH_ipc_filter)
    print(os.getcwd()) # present working directory
    df = pd.read_pickle("table4_ipc.pkl")

    df.loc[:, 'ipc_sub'] = df['ipc_class_symbol'].astype(str).str[:4] # Make ipc "subclass" (4 digits) symbols
    
    # Problem check: Some 'appln_id' have more than one 'ipc_sub' (not unique)
    print("# of total 'ipc_class_symbol's:", len(df)) # 
    print("# of unique 'appln_id':", len(df['appln_id'].unique()))
    print("# of unique combination of ['appln_id', 'ipc_sub']:", df.groupby(['appln_id', 'ipc_sub']).size().shape[0])

    ### IPC subclass Rule: ###
    # (step 0) Only select "ipc_value" == 'I' AND "ipc_position" != 'L': *Already completed in the PATSTAT website downloading stage
    # (step 1) If "ipc_position" == 'F', then assign the IPC subclass to the corresponding 'appln_id'
    # (step 2-a) If "ipc_position" != 'F' BUT the 'appln_id' only has one IPC code, then assign its subclass to the 'appln_id'
    # (step 2-b) If "ipc_position" != 'F' BUT the 'appln_id' has more than one IPC code, then assgin a subclass with the biggest frequencies (max appearance) within its subclass group to the 'appln_id'
    # (Step 3) Drop other cases (No any assignment)

    # "ipc_value": technological role of an IPC code ~ 'I' (inventive); 'N' (Non-inventive)
    # "ipc_position": where an IPC code is located in the patent document ~ 'F' (First/Primary/Main); 'L' (Last)
    print("# IPC_position categories and counts (F/L/empty)")
    print(df['ipc_position'].value_counts(dropna=False))

    unique_appln_ids = df['appln_id'].unique() # grouping based on "appln_id"
    total_appln_ids = len(unique_appln_ids)

    # Set chunk size for memory efficiency
    chunk_size = 1000000  # Number of "appln_id"s for each chunck
    chunks = [unique_appln_ids[i:i + chunk_size] for i in range(0, total_appln_ids, chunk_size)]

    # Chunk processing for IPC-selection filtering
    dfs = []
    for idx, chunk in enumerate(chunks):
        df_small = df[df['appln_id'].isin(chunk)] # Extract data in the chunk
        # Step 1
        df_noF = df_small.groupby('appln_id').filter(lambda g: (g.ipc_position != 'F').all())
        list_noF = pd.DataFrame(df_noF['appln_id'].unique(), columns=['appln_id'])
        list_noF['position_F'] = 0
        df_small = df_small.merge(list_noF, on='appln_id', how='left')
        df_small['position_F'] = df_small['position_F'].fillna(1)
        df_small['position_F'] = df_small['position_F'].astype(int)
        # Step 2 (a+b)
        df_unique_ipc = df_small.groupby('appln_id')['ipc_sub'].nunique().to_frame().reset_index().rename(columns={'ipc_sub': 'n_unique_ipc'})
        df_small = df_small.merge(df_unique_ipc, how='left', on='appln_id')
        df_ipc_sub_unique = df_small.groupby('appln_id')['ipc_sub'].agg(lambda x: list(x.mode())).to_frame().reset_index().rename(columns={'ipc_sub': 'unique_ipc'})
        df_ipc_sub_unique['max_n_unique_ipc'] = df_ipc_sub_unique['unique_ipc'].apply(len)
        df_ipc_sub_unique['max_appearance'] = [','.join(map(str, l)) for l in df_ipc_sub_unique['unique_ipc']]
        df_small = df_small.merge(df_ipc_sub_unique, on='appln_id', how='left')
        
        # Final Column Settings
        df_small['criteria_1'] = np.where((df_small['position_F'] == 1), 1, 0) # Step 1
        df_small['criteria_2'] = np.where((df_small['position_F'] != 1) & (df_small['n_unique_ipc'] == 1), 1, 0) # Step 2-a
        df_small['criteria_3'] = np.where((df_small['position_F'] != 1) & (df_small['n_unique_ipc'] != 1) & (df_small['max_n_unique_ipc'] == 1), 1, 0) # Step 2-b
        df_small['selected'] = np.where(df_small['criteria_1'] + df_small['criteria_2'] + df_small['criteria_3'] > 0, 1, 0)
        
         # Step 3: Select final rows and drop duplicates
        df_final = df_small[df_small['selected'] == 1].drop_duplicates('appln_id')[['appln_id', 'ipc_class_symbol', 'ipc_position', 'max_appearance', 'criteria_1', 'criteria_2', 'criteria_3']].reset_index(drop=True)
        df_final = df_final.rename(columns={'max_appearance': 'ipc_sub'})  # max_appearance -> ipc_sub

        # Iteration
        dfs.append(df_final)
        # Print the ongoing process
        processed_appln_ids = (idx + 1) * chunk_size
        percent_complete = min(100, (processed_appln_ids / total_appln_ids) * 100)
        print(f"Processed {processed_appln_ids} / {total_appln_ids} appln_ids ({percent_complete:.2f}% complete), current chunk size: {df_final.shape[0]}")

    df_ipc = pd.concat(dfs)
    df_ipc = df_ipc.reset_index(drop = True)
    df_ipc.drop_duplicates(subset=['appln_id'], keep='first', inplace=True, ignore_index=False) # Delete duplicates
    df_ipc = df_ipc.reset_index(drop = True)

    df_ipc = df_ipc.sort_values("appln_id", ascending = True) # ascending order
    df_ipc.reset_index(drop = True, inplace=True)
    df_ipc_simple = df_ipc[["appln_id", "ipc_sub"]] # small file size version
    df_ipc_simple

    # Summary & Basic descriptions

    print("### IPC subclass Rule: ###")
    print("# (step 0) Only select \"ipc_value\" == 'I' AND \"ipc_position\" != 'L': *Already completed in the PATSTAT website downloading stage")
    print("# (step 1) If \"ipc_position\" == 'F', then assign the IPC subclass to the corresponding 'appln_id'")
    print("# (step 2-a) If \"ipc_position\" != 'F' BUT the 'appln_id' only has one IPC code, then assign its subclass to the 'appln_id'")
    print("# (step 2-b) If \"ipc_position\" != 'F' BUT the 'appln_id' has more than one IPC code, then assgin a subclass with the biggest frequencies (max appearance) within its subclass group to the 'appln_id'")
    print("# (Step 3) Drop other cases (No any assignment)")

    n0 = len(df['appln_id'].unique())
    n1 = len(df_ipc['appln_id'].unique())
    print( "============================================" )
    print( 'Initial unique appln_id:', n0 )
    print( 'Number of chosen unique appln_id:', n1 )
    print( "# Surviving Percentage = ", round(100*n1/n0, 5), "% \n" )

    print( '# Number of unique IPC Subclass:', len(df_ipc['ipc_sub'].unique()) )
    c1 = len(df_ipc[ (df_ipc['criteria_1'] == 1) ]['appln_id'].unique())
    c2 = len(df_ipc[ (df_ipc['criteria_2'] == 1) ]['appln_id'].unique())
    c3 = len(df_ipc[ (df_ipc['criteria_3'] == 1) ]['appln_id'].unique())
    print('- Chosen by STEP 1:', c1)
    print('- Chosen by STEP 2-a:', c2)
    print('- Chosen by STEP 2-b:', c3)
    print(f"SUM of total chosen unique 'appln_id's = {c1+c2+c3}")
    
    tot_end_time = time.time() # Record the end time
    duration = tot_end_time - tot_start_time
    print("@ Start time =", tot_start)
    print("@ End time =", time.strftime('%Y.%m.%d - %H:%M:%S'))
    print(f"The execution time: {round(duration, 3)} seconds.")

    df_ipc_simple.to_pickle('table4_ipc_clean_simple.pkl') # TO SAVE THE IPC CLEANSING DATA (.pkl)
    df_ipc_simple.to_csv('table4_ipc_clean_simple.csv', index=False) # TO SAVE THE IPC CLEANSING DATA (.csv)

    os.chdir(initial_path)
    print("Reverted to initial directory:", os.getcwd())

    return df_ipc_simple



###########################################################################
# 2. Patent Measure dataset: Make an dataset with various patent measures #
#    *country_select = {"us", "ep", "cn", "kr", "jp"}                     #
###########################################################################
def make_measures(country_select, year_select, big5_vs_domestic = True): # (e.g.) df_main = pmp_kw.make_measures('us', 2017, big5_vs_domestic = True)
    # *country_select = {"us", "ep", "cn", "kr", "jp"} 
    # *year_select = {1975, 1976, ..., 2023}
    # *big5_vs_domestic = True (citations within Big5 countries) or False (citations within only the same one country)
    
    # Required Files (n): "table4_ipc_clean_simple.pkl", "PATSTAT_Tables_raw_each.zip (10.7GB)"
    initial_path = os.getcwd()
    
    tot_start_time = time.time() # Record the start time
    tot_start = time.strftime('%Y.%m.%d - %H:%M:%S')
    
    ### (1) Data Load ###
    PATH_make_measures = './patstat'
    os.chdir(PATH_make_measures)
    print(os.getcwd()) # present working directory
    df_table4 = pd.read_pickle('table4_ipc_clean_simple.pkl') # Table 4: Application ID ("appln_id") & IPC Code ("ipc_sub")

    PATH_make_measures = f'./patstat/{country_select.upper()}_1975_2023'
    os.chdir(PATH_make_measures)
    df_table1 = pd.read_pickle(f'{country_select.lower()}_table1_{year_select}.pkl') # Table 1: Basic Patent information
    df_table2 = pd.read_pickle(f'{country_select.lower()}_table2_{year_select}.pkl') # Table 2: Forward ciations of "Cited" Patents in Table 1 ("Cited" Patents in Table 2 = Patents in Table 1)
    df_table3 = pd.read_pickle(f'{country_select.lower()}_table3_{year_select}.pkl') # Table 3: Backward ciations of "Citing" Patents in Table 1 ("Citing" Patents in Table 3 = Patents in Table 1)
    
    # "appln_auth" == "publn_auth"
    df_table1 = df_table1[df_table1["appln_auth"] == df_table1["publn_auth"]].reset_index(drop = True)
    
    # Filter for IPC Subclass ("ipc_sub" in df_table4)
    unique_ids = pd.concat([df_table1['appln_id'], df_table2['appln_id_citing'], df_table3['appln_id_cited']]).unique()
    df_table4_filtered = df_table4[df_table4['appln_id'].isin(unique_ids)]

    # Merge "ipc_sub" to each table (1, 2, 3)
    df_table1 = df_table1.merge(df_table4_filtered[['appln_id', 'ipc_sub']], on = 'appln_id', how = 'left') # Basic
    df_table2 = df_table2.merge(df_table4_filtered[['appln_id', 'ipc_sub']].rename(columns = {'appln_id' : 'appln_id_citing', 'ipc_sub' : 'ipc_sub_citing'}), on = 'appln_id_citing', how = 'left') # FWD Citations
    df_table3 = df_table3.merge(df_table4_filtered[['appln_id', 'ipc_sub']].rename(columns = {'appln_id' : 'appln_id_cited', 'ipc_sub' : 'ipc_sub_cited'}), on = 'appln_id_cited', how = 'left') # BWD Citations

    del df_table4_filtered; del df_table4 # Delete
    
    # "appln_auth" == "publn_auth" (for Table 2 and 3)
    df_table2 = df_table2[df_table2["appln_auth_citing"] == df_table2["publn_auth_citing"]].reset_index(drop = True)
    df_table3 = df_table3[df_table3["appln_auth_cited"] == df_table3["publn_auth_cited"]].reset_index(drop = True)
    # MAIN BIG 5 COUNTRIES: {'EP', 'US', 'CN', 'JP', 'KR'}: The country (authority) criteria is "appln_auth (patent application location)" as we define each patent application as an individual patent
    # We do not consider 'WO': World Intellectual Property Organization (WIPO), which refers to international patent applications filed under the Patent Cooperation Treaty (PCT)
    if big5_vs_domestic == True: # citations "between BIG 5" countries
        df_table2 = df_table2[df_table2['appln_auth_citing'].isin(['EP', 'US', 'CN', 'JP', 'KR'])].reset_index(drop = True)
        df_table3 = df_table3[df_table3['appln_auth_cited'].isin(['EP', 'US', 'CN', 'JP', 'KR'])].reset_index(drop = True)
    else: # citations "within a country"
        df_table2 = df_table2[df_table2['appln_auth_citing'] == df_table1['appln_auth'].unique()[0]].reset_index(drop = True)
        df_table3 = df_table3[df_table3['appln_auth_cited'] == df_table1['appln_auth'].unique()[0]].reset_index(drop = True)
    
    # Date (time) variable format: Consider "appln_filing_date" and "earliest_publn_date" (filing date & publication date)
    # appln_id (df_table1) = appln_id_cited (df_table2)
    df_table2 = df_table2.merge(df_table1[['appln_id', 'appln_filing_date']].rename(columns = {'appln_id' : 'appln_id_cited'}), on = 'appln_id_cited', how = 'left')
    df_table2 = df_table2.merge(df_table1[['appln_id', 'earliest_publn_date']].rename(columns = {'appln_id' : 'appln_id_cited'}), on = 'appln_id_cited', how = 'left')
    df_table2 = df_table2.rename(columns = {'appln_filing_date' : 'appln_filing_date_cited', 'earliest_publn_date' : 'publn_date_cited'})
    # appln_id (df_table1) = appln_id_citing (df_table3)
    df_table3 = df_table3.merge(df_table1[['appln_id', 'appln_filing_date']].rename(columns = {'appln_id' : 'appln_id_citing'}), on = 'appln_id_citing', how = 'left')
    df_table3 = df_table3.merge(df_table1[['appln_id', 'earliest_publn_date']].rename(columns = {'appln_id' : 'appln_id_citing'}), on = 'appln_id_citing', how = 'left')
    df_table3 = df_table3.rename(columns = {'appln_filing_date' : 'appln_filing_date_citing', 'earliest_publn_date' : 'publn_date_citing'})
    
    # Extract Year (yyyy) from Date (yyyy-mm-dd)
    # Table 1
    df_table1['appln_filing_year'] = df_table1['appln_filing_date'].str[:4].astype(int)
    df_table1['earliest_publn_year'] = df_table1['earliest_publn_date'].str[:4].astype(int)
    # Table 2
    df_table2['appln_filing_year_citing'] = df_table2['appln_filing_date_citing'].str[:4].astype(int)
    df_table2['publn_year_citing'] = df_table2['publn_date_citing'].str[:4].astype(int)
    df_table2['appln_filing_year_cited'] = df_table2['appln_filing_date_cited'].str[:4].astype(int)
    df_table2['publn_year_cited'] = df_table2['publn_date_cited'].str[:4].astype(int)
    # Table 3
    df_table3['appln_filing_year_cited'] = df_table3['appln_filing_date_cited'].str[:4].astype(int)
    df_table3['publn_year_cited'] = df_table3['publn_date_cited'].str[:4].astype(int)
    df_table3['appln_filing_year_citing'] = df_table3['appln_filing_date_citing'].str[:4].astype(int)
    df_table3['publn_year_citing'] = df_table3['publn_date_citing'].str[:4].astype(int)
    
    # Format Change (Delete "-"): Date (yyyy-mm-dd) --> Date (yyyymmdd)
    # Table 1
    df_table1['appln_filing_date'] = df_table1['appln_filing_date'].replace('-', '', regex=True).astype(int)
    df_table1['earliest_filing_date'] = df_table1['earliest_filing_date'].replace('-', '', regex=True).astype(int)
    df_table1['earliest_publn_date'] = df_table1['earliest_publn_date'].replace('-', '', regex=True).astype(int)
    # Table 2
    df_table2['appln_filing_date_citing'] = df_table2['appln_filing_date_citing'].replace('-', '', regex=True).astype(int)
    df_table2['publn_date_citing'] = df_table2['publn_date_citing'].replace('-', '', regex=True).astype(int)
    df_table2['appln_filing_date_cited'] = df_table2['appln_filing_date_cited'].replace('-', '', regex=True).astype(int)
    df_table2['publn_date_cited'] = df_table2['publn_date_cited'].replace('-', '', regex=True).astype(int)
    # Table 3
    df_table3['appln_filing_date_citing'] = df_table3['appln_filing_date_citing'].replace('-', '', regex=True).astype(int)
    df_table3['publn_date_citing'] = df_table3['publn_date_citing'].replace('-', '', regex=True).astype(int)
    df_table3['appln_filing_date_cited'] = df_table3['appln_filing_date_cited'].replace('-', '', regex=True).astype(int)
    df_table3['publn_date_cited'] = df_table3['publn_date_cited'].replace('-', '', regex=True).astype(int)
    
    # Make "after-t-years" Variables: (Filing or Publication) [Year] + "t" [years]
    # **Only Table 2 (Table 3 does not need time points because backward citations are fixed at the filing date)
    # "filing_date" #
    # Table 2: Convert the "appln_filing_date_cited" column to datetime format to calculate time changes
    df_table2['appln_filing_date_cited'] = pd.to_datetime(df_table2['appln_filing_date_cited'], format='%Y%m%d')
    # Calculate new date variables using vectorized operations (date t +{1, 2, 3, 4, 5} years)
    df_table2['appln_filing_date_cited_t+1'] = df_table2['appln_filing_date_cited'] + pd.DateOffset(years=1)
    df_table2['appln_filing_date_cited_t+2'] = df_table2['appln_filing_date_cited'] + pd.DateOffset(years=2)
    df_table2['appln_filing_date_cited_t+3'] = df_table2['appln_filing_date_cited'] + pd.DateOffset(years=3)
    df_table2['appln_filing_date_cited_t+4'] = df_table2['appln_filing_date_cited'] + pd.DateOffset(years=4)
    df_table2['appln_filing_date_cited_t+5'] = df_table2['appln_filing_date_cited'] + pd.DateOffset(years=5)
    # Convert dates back to the original format (integer): appln_filing_date_cited_t +{1, 2, 3, 4, 5} years
    date_columns = ['appln_filing_date_cited_t+1', 
                    'appln_filing_date_cited_t+2', 
                    'appln_filing_date_cited_t+3', 
                    'appln_filing_date_cited_t+4', 
                    'appln_filing_date_cited_t+5']
    for col in date_columns:
        df_table2[col] = df_table2[col].dt.strftime('%Y%m%d').astype(int)
    # Convert dates back to the original format (integer): appln_filing_date_cited
    df_table2['appln_filing_date_cited'] = df_table2['appln_filing_date_cited'].dt.strftime('%Y%m%d').astype(int)

    # "publn_date" #
    # Table 2: Convert the "publn_date_cited" column to datetime format to calculate time changes
    df_table2['publn_date_cited'] = pd.to_datetime(df_table2['publn_date_cited'], format='%Y%m%d')
    # Calculate new date variables using vectorized operations (date t +{1, 2, 3, 4, 5} years)
    df_table2['publn_date_cited_t+1'] = df_table2['publn_date_cited'] + pd.DateOffset(years=1)
    df_table2['publn_date_cited_t+2'] = df_table2['publn_date_cited'] + pd.DateOffset(years=2)
    df_table2['publn_date_cited_t+3'] = df_table2['publn_date_cited'] + pd.DateOffset(years=3)
    df_table2['publn_date_cited_t+4'] = df_table2['publn_date_cited'] + pd.DateOffset(years=4)
    df_table2['publn_date_cited_t+5'] = df_table2['publn_date_cited'] + pd.DateOffset(years=5)
    # Convert dates back to the original format (integer): publn_date_cited_t +{1, 2, 3, 4, 5} years
    date_columns = ['publn_date_cited_t+1', 
                    'publn_date_cited_t+2', 
                    'publn_date_cited_t+3', 
                    'publn_date_cited_t+4', 
                    'publn_date_cited_t+5']
    for col in date_columns:
        df_table2[col] = df_table2[col].dt.strftime('%Y%m%d').astype(int)
    # Convert dates back to the original format (integer): publn_date_cited
    df_table2['publn_date_cited'] = df_table2['publn_date_cited'].dt.strftime('%Y%m%d').astype(int)
    
    ### (2) Make scaling values (patent measures) based on each patent application id ###
    
    # Forward citations (by Table 2)
    # - Fiscal (true) vs. Calendar (only year) = 2 cases
    # - Time between [appln/publn date (cited) vs. appln/publn date (citing)] = 2*2 cases = 4 cases
    # - "date_cited_t"+{k} where k={1,2,3,4,5} = 5 cases
    k_values = [2, 3, 5] # Define k values: how many citations are during the "k" years
    df_fwd_calendar = calculate_conditions_calendar(df_table2.copy(), k_values) # Apply the optimized function to calculate conditions based on calendar years
    df_fwd_fiscal = calculate_conditions_fiscal(df_table2.copy(), k_values) # Apply the optimized function to calculate conditions based on fiscal years
    
    # Generality (by Table 2)
    # - based on "Forward" citations
    # - where N_i denotes the number of "forward citations" to a patent i, N_ij is the number received from patents in class j, and N is the number of different technological fields (classes) j's
    # - HHI_i = ∑[ (N_ij/N_i)^2 ] (e.g. [(3/5)^2 + (1/5)^2 + (1/5)^2])
    # - Generality_i = (N / (N-1)) * (1- HHI_i)
    # - Do not include citing patents with IPC values of NaN (Null)
    
    # Originality (by Table 3)
    # - based on "Backward" citations
    # - where N_i denotes the number of "backward citations" to a patent i, N_ij is the number received from patents in class j, and N is the number of different technological fields (classes) j's
    # - HHI_i = ∑[ (N_ij/N_i)^2 ] (e.g. [(3/5)^2 + (1/5)^2 + (1/5)^2])
    # - Generality_i = (N / (N-1)) * (1- HHI_i)
    
    # Forward/Backward citation lags (by Table 2&3)
    # - The (Median or Mean) value of time differences between cited and citing patents
    # - Forward citation lag is calculated "within 2/3/5 years-based"
    
    ## Forward Citation Lag (Calendar-based) - Calculate the time difference
    fwd_lag2_clndr_aa = calculate_fwd_citation_lag(df_table2, "appln_filing_year_citing", "appln_filing_year_cited", 2, "fwd_lag_clndr_aa")
    fwd_lag2_clndr_pa = calculate_fwd_citation_lag(df_table2, "appln_filing_year_citing", "publn_year_cited", 2, "fwd_lag_clndr_pa")
    fwd_lag2_clndr_ap = calculate_fwd_citation_lag(df_table2, "publn_year_citing", "appln_filing_year_cited", 2, "fwd_lag_clndr_ap")
    fwd_lag2_clndr_pp = calculate_fwd_citation_lag(df_table2, "publn_year_citing", "publn_year_cited", 2, "fwd_lag_clndr_pp")
    fwd_lag3_clndr_aa = calculate_fwd_citation_lag(df_table2, "appln_filing_year_citing", "appln_filing_year_cited", 3, "fwd_lag_clndr_aa")
    fwd_lag3_clndr_pa = calculate_fwd_citation_lag(df_table2, "appln_filing_year_citing", "publn_year_cited", 3, "fwd_lag_clndr_pa")
    fwd_lag3_clndr_ap = calculate_fwd_citation_lag(df_table2, "publn_year_citing", "appln_filing_year_cited", 3, "fwd_lag_clndr_ap")
    fwd_lag3_clndr_pp = calculate_fwd_citation_lag(df_table2, "publn_year_citing", "publn_year_cited", 3, "fwd_lag_clndr_pp")
    fwd_lag5_clndr_aa = calculate_fwd_citation_lag(df_table2, "appln_filing_year_citing", "appln_filing_year_cited", 5, "fwd_lag_clndr_aa")
    fwd_lag5_clndr_pa = calculate_fwd_citation_lag(df_table2, "appln_filing_year_citing", "publn_year_cited", 5, "fwd_lag_clndr_pa")
    fwd_lag5_clndr_ap = calculate_fwd_citation_lag(df_table2, "publn_year_citing", "appln_filing_year_cited", 5, "fwd_lag_clndr_ap")
    fwd_lag5_clndr_pp = calculate_fwd_citation_lag(df_table2, "publn_year_citing", "publn_year_cited", 5, "fwd_lag_clndr_pp")
    fwd_lag2_clndr_ap_a = calculate_fwd_citation_lag_ap_a(df_table2, "appln_filing_year_citing", "appln_filing_year_cited", "publn_year_cited", 2, "fwd_lag_clndr_ap_a")
    fwd_lag2_clndr_ap_p = calculate_fwd_citation_lag_ap_p(df_table2, "publn_year_citing", "appln_filing_year_cited", "publn_year_cited", 2, "fwd_lag_clndr_ap_p")
    fwd_lag3_clndr_ap_a = calculate_fwd_citation_lag_ap_a(df_table2, "appln_filing_year_citing", "appln_filing_year_cited", "publn_year_cited", 3, "fwd_lag_clndr_ap_a")
    fwd_lag3_clndr_ap_p = calculate_fwd_citation_lag_ap_p(df_table2, "publn_year_citing", "appln_filing_year_cited", "publn_year_cited", 3, "fwd_lag_clndr_ap_p")
    fwd_lag5_clndr_ap_a = calculate_fwd_citation_lag_ap_a(df_table2, "appln_filing_year_citing", "appln_filing_year_cited", "publn_year_cited", 5, "fwd_lag_clndr_ap_a")
    fwd_lag5_clndr_ap_p = calculate_fwd_citation_lag_ap_p(df_table2, "publn_year_citing", "appln_filing_year_cited", "publn_year_cited", 5, "fwd_lag_clndr_ap_p")
    
    ## Forward Citation Lag (Fiscal-based) - Calculate the time difference
    fwd_lag2_fscl_aa = calculate_fwd_citation_lag_fiscal(df_table2, "appln_filing_date_citing", "appln_filing_date_cited", "appln_filing_date_cited_t+2", "fwd_lag_M_fscl_aa")
    fwd_lag2_fscl_pa = calculate_fwd_citation_lag_fiscal(df_table2, "appln_filing_date_citing", "publn_date_cited", "publn_date_cited_cited_t+2", "fwd_lag_M_fscl_pa")
    fwd_lag2_fscl_ap = calculate_fwd_citation_lag_fiscal(df_table2, "publn_date_citing", "appln_filing_date_cited", "appln_filing_date_cited_cited_t+2", "fwd_lag_M_fscl_ap")
    fwd_lag2_fscl_pp = calculate_fwd_citation_lag_fiscal(df_table2, "publn_date_citing", "publn_date_cited", "publn_date_cited_t+2", "fwd_lag_M_fscl_pp")
    fwd_lag3_fscl_aa = calculate_fwd_citation_lag_fiscal(df_table2, "appln_filing_date_citing", "appln_filing_date_cited", "appln_filing_date_cited_t+3", "fwd_lag_M_fscl_aa")
    fwd_lag3_fscl_pa = calculate_fwd_citation_lag_fiscal(df_table2, "appln_filing_date_citing", "publn_date_cited", "publn_date_cited_cited_t+3", "fwd_lag_M_fscl_pa")
    fwd_lag3_fscl_ap = calculate_fwd_citation_lag_fiscal(df_table2, "publn_date_citing", "appln_filing_date_cited", "appln_filing_date_cited_cited_t+3", "fwd_lag_M_fscl_ap")
    fwd_lag3_fscl_pp = calculate_fwd_citation_lag_fiscal(df_table2, "publn_date_citing", "publn_date_cited", "publn_date_cited_t+3", "fwd_lag_M_fscl_pp")
    fwd_lag5_fscl_aa = calculate_fwd_citation_lag_fiscal(df_table2, "appln_filing_date_citing", "appln_filing_date_cited", "appln_filing_date_cited_t+5", "fwd_lag_M_fscl_aa")
    fwd_lag5_fscl_pa = calculate_fwd_citation_lag_fiscal(df_table2, "appln_filing_date_citing", "publn_date_cited", "publn_date_cited_cited_t+5", "fwd_lag_M_fscl_pa")
    fwd_lag5_fscl_ap = calculate_fwd_citation_lag_fiscal(df_table2, "publn_date_citing", "appln_filing_date_cited", "appln_filing_date_cited_cited_t+5", "fwd_lag_M_fscl_ap")
    fwd_lag5_fscl_pp = calculate_fwd_citation_lag_fiscal(df_table2, "publn_date_citing", "publn_date_cited", "publn_date_cited_t+5", "fwd_lag_M_fscl_pp")
    fwd_lag2_fscl_ap_a = calculate_fwd_citation_lag_fscl_ap_a(df_table2, "appln_filing_date_citing", "appln_filing_date_cited", "publn_date_cited_t+2", "fwd_lag_M_fscl_ap_a")
    fwd_lag2_fscl_ap_p = calculate_fwd_citation_lag_fscl_ap_p(df_table2, "publn_date_citing", "appln_filing_date_cited", "publn_date_cited_t+2", "fwd_lag_M_fscl_ap_p")
    fwd_lag3_fscl_ap_a = calculate_fwd_citation_lag_fscl_ap_a(df_table2, "appln_filing_date_citing", "appln_filing_date_cited", "publn_date_cited_t+3", "fwd_lag_M_fscl_ap_a")
    fwd_lag3_fscl_ap_p = calculate_fwd_citation_lag_fscl_ap_p(df_table2, "publn_date_citing", "appln_filing_date_cited", "publn_date_cited_t+3", "fwd_lag_M_fscl_ap_p")
    fwd_lag5_fscl_ap_a = calculate_fwd_citation_lag_fscl_ap_a(df_table2, "appln_filing_date_citing", "appln_filing_date_cited", "publn_date_cited_t+5", "fwd_lag_M_fscl_ap_a")
    fwd_lag5_fscl_ap_p = calculate_fwd_citation_lag_fscl_ap_p(df_table2, "publn_date_citing", "appln_filing_date_cited", "publn_date_cited_t+5", "fwd_lag_M_fscl_ap_p")
    
    ## Backward Citation Lag (Calendar-based) - Calculate the time difference
    df_bwd_lag= df_table3[["appln_id_citing", 
                        "appln_filing_year_citing", "appln_filing_year_cited", "publn_year_citing", "publn_year_cited"
                        ]].copy()
    bwd_lag_clndr_aa = (df_bwd_lag[(df_bwd_lag["appln_filing_year_citing"] >= df_bwd_lag["appln_filing_year_cited"])
                                ])[["appln_id_citing", "appln_filing_year_citing", "appln_filing_year_cited"]]
    bwd_lag_clndr_pa = (df_bwd_lag[(df_bwd_lag["appln_filing_year_citing"] >= df_bwd_lag["publn_year_cited"])
                                ])[["appln_id_citing", "appln_filing_year_citing", "publn_year_cited"]]
    bwd_lag_clndr_ap = (df_bwd_lag[(df_bwd_lag["publn_year_citing"] >= df_bwd_lag["appln_filing_year_cited"])
                                ])[["appln_id_citing", "publn_year_citing", "appln_filing_year_cited"]]
    bwd_lag_clndr_pp = (df_bwd_lag[(df_bwd_lag["publn_year_citing"] >= df_bwd_lag["publn_year_cited"])
                                ])[["appln_id_citing", "publn_year_citing", "publn_year_cited"]]
    # Calculate the time difference in year
    bwd_lag_clndr_aa['bwd_lag_clndr_aa'] = bwd_lag_clndr_aa['appln_filing_year_citing'] - bwd_lag_clndr_aa['appln_filing_year_cited']
    bwd_lag_clndr_pa['bwd_lag_clndr_pa'] = bwd_lag_clndr_pa['appln_filing_year_citing'] - bwd_lag_clndr_pa['publn_year_cited']
    bwd_lag_clndr_ap['bwd_lag_clndr_ap'] = bwd_lag_clndr_ap['publn_year_citing'] - bwd_lag_clndr_ap['appln_filing_year_cited']
    bwd_lag_clndr_pp['bwd_lag_clndr_pp'] = bwd_lag_clndr_pp['publn_year_citing'] - bwd_lag_clndr_pp['publn_year_cited']
    # Group by "appln_id_citing" and calculate the median time difference
    bwd_lag_clndr_aa = bwd_lag_clndr_aa.groupby('appln_id_citing')['bwd_lag_clndr_aa'].agg(['median', 'mean']).reset_index()
    bwd_lag_clndr_pa = bwd_lag_clndr_pa.groupby('appln_id_citing')['bwd_lag_clndr_pa'].agg(['median', 'mean']).reset_index()
    bwd_lag_clndr_ap = bwd_lag_clndr_ap.groupby('appln_id_citing')['bwd_lag_clndr_ap'].agg(['median', 'mean']).reset_index()
    bwd_lag_clndr_pp = bwd_lag_clndr_pp.groupby('appln_id_citing')['bwd_lag_clndr_pp'].agg(['median', 'mean']).reset_index()
    
    ## Backward Citation Lag (Fiscal-based) - Calculate the time difference
    df_bwd_lag= df_table3[["appln_id_citing", 
                        "appln_filing_date_citing", "appln_filing_date_cited", "publn_date_citing", "publn_date_cited"
                        ]].copy()
    # Time format
    df_bwd_lag['appln_filing_date_citing'] = pd.to_datetime(df_bwd_lag['appln_filing_date_citing'], format='%Y%m%d')
    df_bwd_lag['appln_filing_date_cited'] = pd.to_datetime(df_bwd_lag['appln_filing_date_cited'], format='%Y%m%d')
    df_bwd_lag['publn_date_citing'] = pd.to_datetime(df_bwd_lag['publn_date_citing'], format='%Y%m%d')
    df_bwd_lag['publn_date_cited'] = pd.to_datetime(df_bwd_lag['publn_date_cited'], format='%Y%m%d')
    bwd_lag_fscl_aa = (df_bwd_lag[(df_bwd_lag["appln_filing_date_citing"] >= df_bwd_lag["appln_filing_date_cited"])
                                ])[["appln_id_citing", "appln_filing_date_citing", "appln_filing_date_cited"]]
    bwd_lag_fscl_pa = (df_bwd_lag[(df_bwd_lag["appln_filing_date_citing"] >= df_bwd_lag["publn_date_cited"])
                                ])[["appln_id_citing", "appln_filing_date_citing", "publn_date_cited"]]
    bwd_lag_fscl_ap = (df_bwd_lag[(df_bwd_lag["publn_date_citing"] >= df_bwd_lag["appln_filing_date_cited"])
                                ])[["appln_id_citing", "publn_date_citing", "appln_filing_date_cited"]]
    bwd_lag_fscl_pp = (df_bwd_lag[(df_bwd_lag["publn_date_citing"] >= df_bwd_lag["publn_date_cited"])
                                ])[["appln_id_citing", "publn_date_citing", "publn_date_cited"]]
    # Calculate the time difference in months
    df_bwd_lag['bwd_lag_M_fscl_aa'] = (df_bwd_lag['appln_filing_date_citing'].dt.year - \
                                    df_bwd_lag['appln_filing_date_cited'].dt.year
                                    )*12 + (df_bwd_lag['appln_filing_date_citing'].dt.month - \
                                            df_bwd_lag['appln_filing_date_cited'].dt.month)
    df_bwd_lag['bwd_lag_M_fscl_pa'] = (df_bwd_lag['appln_filing_date_citing'].dt.year - \
                                    df_bwd_lag['publn_date_cited'].dt.year
                                    )*12 + (df_bwd_lag['appln_filing_date_citing'].dt.month - \
                                            df_bwd_lag['publn_date_cited'].dt.month)
    df_bwd_lag['bwd_lag_M_fscl_ap'] = (df_bwd_lag['publn_date_citing'].dt.year - \
                                    df_bwd_lag['appln_filing_date_cited'].dt.year
                                    )*12 + (df_bwd_lag['publn_date_citing'].dt.month - \
                                            df_bwd_lag['appln_filing_date_cited'].dt.month)
    df_bwd_lag['bwd_lag_M_fscl_pp'] = (df_bwd_lag['publn_date_citing'].dt.year - \
                                    df_bwd_lag['publn_date_cited'].dt.year
                                    )*12 + (df_bwd_lag['publn_date_citing'].dt.month - \
                                            df_bwd_lag['publn_date_cited'].dt.month)
    # Group by "appln_id_cited and calculate the median time difference
    bwd_lag_fscl_aa = df_bwd_lag.groupby('appln_id_citing')['bwd_lag_M_fscl_aa'].agg(['median', 'mean']).reset_index()
    bwd_lag_fscl_pa = df_bwd_lag.groupby('appln_id_citing')['bwd_lag_M_fscl_pa'].agg(['median', 'mean']).reset_index()
    bwd_lag_fscl_ap = df_bwd_lag.groupby('appln_id_citing')['bwd_lag_M_fscl_ap'].agg(['median', 'mean']).reset_index()
    bwd_lag_fscl_pp = df_bwd_lag.groupby('appln_id_citing')['bwd_lag_M_fscl_pp'].agg(['median', 'mean']).reset_index()
    
    
    # First forward citation time
    # (calendar: year)
    fwd_1st_clndr_aa = (df_table2[(df_table2["appln_filing_year_citing"] >= df_table2["appln_filing_year_cited"]) & \
                                (df_table2["appln_filing_year_citing"] <= df_table2["appln_filing_year_cited"] +5)
                                ])[["appln_id_cited", "appln_filing_year_citing", "appln_filing_year_cited"]]
    fwd_1st_clndr_pa = (df_table2[(df_table2["appln_filing_year_citing"] >= df_table2["publn_year_cited"]) & \
                                (df_table2["appln_filing_year_citing"] <= df_table2["publn_year_cited"] +5)
                                ])[["appln_id_cited", "appln_filing_year_citing", "publn_year_cited"]]
    fwd_1st_clndr_ap = (df_table2[(df_table2["publn_year_citing"] >= df_table2["appln_filing_year_cited"]) & \
                                (df_table2["publn_year_citing"] <= df_table2["appln_filing_year_cited"] +5)
                                ])[["appln_id_cited", "publn_year_citing", "appln_filing_year_cited"]]
    fwd_1st_clndr_pp = (df_table2[(df_table2["publn_year_citing"] >= df_table2["publn_year_cited"]) & \
                                (df_table2["publn_year_citing"] <= df_table2["publn_year_cited"] +5)
                                ])[["appln_id_cited", "publn_year_citing", "publn_year_cited"]]
    # filing date ~ until publication date + (3 or 5)years: total citations
    fwd_1st_clndr_ap_a = (df_table2[(df_table2["appln_filing_year_citing"] >= df_table2["appln_filing_year_cited"]) & \
                                    (df_table2["appln_filing_year_citing"] <= df_table2["publn_year_cited"] +5)
                                ])[["appln_id_cited", "appln_filing_year_citing", "appln_filing_year_cited"]]
    fwd_1st_clndr_ap_p = (df_table2[(df_table2["publn_year_citing"] >= df_table2["appln_filing_year_cited"]) & \
                                    (df_table2["publn_year_citing"] <= df_table2["publn_year_cited"] +5)
                                ])[["appln_id_cited", "publn_year_citing", "appln_filing_year_cited"]]
    # Calculate the time difference in year
    fwd_1st_clndr_aa['Y_fwd_1st_aa'] = fwd_1st_clndr_aa['appln_filing_year_citing'] - fwd_1st_clndr_aa['appln_filing_year_cited']
    fwd_1st_clndr_pa['Y_fwd_1st_pa'] = fwd_1st_clndr_pa['appln_filing_year_citing'] - fwd_1st_clndr_pa['publn_year_cited']
    fwd_1st_clndr_ap['Y_fwd_1st_ap'] = fwd_1st_clndr_ap['publn_year_citing'] - fwd_1st_clndr_ap['appln_filing_year_cited']
    fwd_1st_clndr_pp['Y_fwd_1st_pp'] = fwd_1st_clndr_pp['publn_year_citing'] - fwd_1st_clndr_pp['publn_year_cited']
    fwd_1st_clndr_ap_a['Y_fwd_1st_ap_a'] = fwd_1st_clndr_ap_a['appln_filing_year_citing'] - fwd_1st_clndr_ap_a['appln_filing_year_cited']
    fwd_1st_clndr_ap_p['Y_fwd_1st_ap_p'] = fwd_1st_clndr_ap_p['publn_year_citing'] - fwd_1st_clndr_ap_p['appln_filing_year_cited']
    # Group by "appln_id_cited" and calculate the minimum time difference
    fwd_1st_clndr_aa = fwd_1st_clndr_aa.groupby('appln_id_cited')['Y_fwd_1st_aa'].agg(['min']).reset_index()
    fwd_1st_clndr_pa = fwd_1st_clndr_pa.groupby('appln_id_cited')['Y_fwd_1st_pa'].agg(['min']).reset_index()
    fwd_1st_clndr_ap = fwd_1st_clndr_ap.groupby('appln_id_cited')['Y_fwd_1st_ap'].agg(['min']).reset_index()
    fwd_1st_clndr_pp = fwd_1st_clndr_pp.groupby('appln_id_cited')['Y_fwd_1st_pp'].agg(['min']).reset_index()
    fwd_1st_clndr_ap_a = fwd_1st_clndr_ap_a.groupby('appln_id_cited')['Y_fwd_1st_ap_a'].agg(['min']).reset_index()
    fwd_1st_clndr_ap_p = fwd_1st_clndr_ap_p.groupby('appln_id_cited')['Y_fwd_1st_ap_p'].agg(['min']).reset_index()
    # Rename
    fwd_1st_clndr_aa.rename(columns = {'min' : 'Y_fwd_1st_aa'}, inplace=True)
    fwd_1st_clndr_pa.rename(columns = {'min' : 'Y_fwd_1st_pa'}, inplace=True)
    fwd_1st_clndr_ap.rename(columns = {'min' : 'Y_fwd_1st_ap'}, inplace=True)
    fwd_1st_clndr_pp.rename(columns = {'min' : 'Y_fwd_1st_pp'}, inplace=True)
    fwd_1st_clndr_ap_a.rename(columns = {'min' : 'Y_fwd_1st_ap_a'}, inplace=True)
    fwd_1st_clndr_ap_p.rename(columns = {'min' : 'Y_fwd_1st_ap_p'}, inplace=True)
    # Combine into a single DataFrame to reduce the number of merges
    df_Y_fwd_1st = pd.merge(fwd_1st_clndr_aa, fwd_1st_clndr_pa, on='appln_id_cited', how='outer')
    df_Y_fwd_1st = pd.merge(df_Y_fwd_1st, fwd_1st_clndr_ap, on='appln_id_cited', how='outer')
    df_Y_fwd_1st = pd.merge(df_Y_fwd_1st, fwd_1st_clndr_pp, on='appln_id_cited', how='outer')
    df_Y_fwd_1st = pd.merge(df_Y_fwd_1st, fwd_1st_clndr_ap_a, on='appln_id_cited', how='outer')
    df_Y_fwd_1st = pd.merge(df_Y_fwd_1st, fwd_1st_clndr_ap_p, on='appln_id_cited', how='outer')
    
    # (fiscal: month)
    # Convert 'appln_filing_date_citing' and 'appln_filing_date_cited' to datetime
    df_fwd_cit_fiscal = df_table2[["appln_id_cited", 
                                "appln_filing_date_citing", "appln_filing_date_cited",
                                "publn_date_citing", "publn_date_cited", 
                                "appln_filing_date_cited_t+5", "publn_date_cited_t+5"]].copy()
    df_fwd_cit_fiscal['appln_filing_date_citing'] = pd.to_datetime(df_fwd_cit_fiscal['appln_filing_date_citing'], format='%Y%m%d')
    df_fwd_cit_fiscal['appln_filing_date_cited'] = pd.to_datetime(df_fwd_cit_fiscal['appln_filing_date_cited'], format='%Y%m%d')
    df_fwd_cit_fiscal['publn_date_citing'] = pd.to_datetime(df_fwd_cit_fiscal['publn_date_citing'], format='%Y%m%d')
    df_fwd_cit_fiscal['publn_date_cited'] = pd.to_datetime(df_fwd_cit_fiscal['publn_date_cited'], format='%Y%m%d')
    df_fwd_cit_fiscal['publn_date_cited_t+5'] = pd.to_datetime(df_fwd_cit_fiscal['publn_date_cited_t+5'], format='%Y%m%d')
    df_fwd_cit_fiscal['appln_filing_date_cited_t+5'] = pd.to_datetime(df_fwd_cit_fiscal['appln_filing_date_cited_t+5'], format='%Y%m%d')

    fwd_1st_fscl_aa = (df_fwd_cit_fiscal[(df_fwd_cit_fiscal["appln_filing_date_citing"] >= df_fwd_cit_fiscal['appln_filing_date_cited']) & \
                                        (df_fwd_cit_fiscal["appln_filing_date_citing"] <= df_fwd_cit_fiscal['appln_filing_date_cited_t+5'])
                                        ])[["appln_id_cited", "appln_filing_date_citing", "appln_filing_date_cited"]]
    fwd_1st_fscl_pa = (df_fwd_cit_fiscal[(df_fwd_cit_fiscal["appln_filing_date_citing"] >= df_fwd_cit_fiscal["publn_date_cited"]) & \
                                        (df_fwd_cit_fiscal["appln_filing_date_citing"] <= df_fwd_cit_fiscal["publn_date_cited_t+5"])
                                        ])[["appln_id_cited", "appln_filing_date_citing", "publn_date_cited"]]
    fwd_1st_fscl_ap = (df_fwd_cit_fiscal[(df_fwd_cit_fiscal["publn_date_citing"] >= df_fwd_cit_fiscal["appln_filing_date_cited"]) & \
                                        (df_fwd_cit_fiscal["publn_date_citing"] <= df_fwd_cit_fiscal["appln_filing_date_cited_t+5"])
                                        ])[["appln_id_cited", "publn_date_citing", "appln_filing_date_cited"]]
    fwd_1st_fscl_pp = (df_fwd_cit_fiscal[(df_fwd_cit_fiscal["publn_date_citing"] >= df_fwd_cit_fiscal["publn_date_cited"])& \
                                        (df_fwd_cit_fiscal["publn_date_citing"] <= df_fwd_cit_fiscal["publn_date_cited_t+5"])
                                        ])[["appln_id_cited", "publn_date_citing", "publn_date_cited"]]
    fwd_1st_fscl_ap_a = (df_fwd_cit_fiscal[(df_fwd_cit_fiscal["appln_filing_date_citing"] >= df_fwd_cit_fiscal['appln_filing_date_cited']) & \
                                        (df_fwd_cit_fiscal["appln_filing_date_citing"] <= df_fwd_cit_fiscal['publn_date_cited_t+5'])
                                        ])[["appln_id_cited", "appln_filing_date_citing", "appln_filing_date_cited"]]
    fwd_1st_fscl_ap_p = (df_fwd_cit_fiscal[(df_fwd_cit_fiscal["publn_date_citing"] >= df_fwd_cit_fiscal['appln_filing_date_cited']) & \
                                        (df_fwd_cit_fiscal["publn_date_citing"] <= df_fwd_cit_fiscal['publn_date_cited_t+5'])
                                        ])[["appln_id_cited", "publn_date_citing", "appln_filing_date_cited"]]

    # Calculate the time difference in months
    fwd_1st_fscl_aa['M_fwd_1st_aa'] = (fwd_1st_fscl_aa['appln_filing_date_citing'].dt.year - \
                                    fwd_1st_fscl_aa['appln_filing_date_cited'].dt.year
                                    )*12 + (fwd_1st_fscl_aa['appln_filing_date_citing'].dt.month - \
                                            fwd_1st_fscl_aa['appln_filing_date_cited'].dt.month)
    fwd_1st_fscl_pa['M_fwd_1st_pa'] = (fwd_1st_fscl_pa['appln_filing_date_citing'].dt.year - \
                                    fwd_1st_fscl_pa['publn_date_cited'].dt.year
                                    )*12 + (fwd_1st_fscl_pa['appln_filing_date_citing'].dt.month - \
                                            fwd_1st_fscl_pa['publn_date_cited'].dt.month)
    fwd_1st_fscl_ap['M_fwd_1st_ap'] = (fwd_1st_fscl_ap['publn_date_citing'].dt.year - \
                                    fwd_1st_fscl_ap['appln_filing_date_cited'].dt.year
                                    )*12 + (fwd_1st_fscl_ap['publn_date_citing'].dt.month - \
                                            fwd_1st_fscl_ap['appln_filing_date_cited'].dt.month)
    fwd_1st_fscl_pp['M_fwd_1st_pp'] = (fwd_1st_fscl_pp['publn_date_citing'].dt.year - \
                                    fwd_1st_fscl_pp['publn_date_cited'].dt.year
                                    )*12 + (fwd_1st_fscl_pp['publn_date_citing'].dt.month - \
                                            fwd_1st_fscl_pp['publn_date_cited'].dt.month)
    fwd_1st_fscl_ap_a['M_fwd_1st_ap_a'] = (fwd_1st_fscl_ap_a['appln_filing_date_citing'].dt.year - \
                                        fwd_1st_fscl_ap_a['appln_filing_date_cited'].dt.year
                                        )*12 + (fwd_1st_fscl_ap_a['appln_filing_date_citing'].dt.month - \
                                                fwd_1st_fscl_ap_a['appln_filing_date_cited'].dt.month)
    fwd_1st_fscl_ap_p['M_fwd_1st_ap_p'] = (fwd_1st_fscl_ap_p['publn_date_citing'].dt.year - \
                                        fwd_1st_fscl_ap_p['appln_filing_date_cited'].dt.year
                                        )*12 + (fwd_1st_fscl_ap_p['publn_date_citing'].dt.month - \
                                                fwd_1st_fscl_ap_p['appln_filing_date_cited'].dt.month)

    # Group by "appln_id_cited" and calculate the minimum time difference
    fwd_1st_fscl_aa = fwd_1st_fscl_aa.groupby('appln_id_cited')['M_fwd_1st_aa'].agg(['min']).reset_index()
    fwd_1st_fscl_pa = fwd_1st_fscl_pa.groupby('appln_id_cited')['M_fwd_1st_pa'].agg(['min']).reset_index()
    fwd_1st_fscl_ap = fwd_1st_fscl_ap.groupby('appln_id_cited')['M_fwd_1st_ap'].agg(['min']).reset_index()
    fwd_1st_fscl_pp = fwd_1st_fscl_pp.groupby('appln_id_cited')['M_fwd_1st_pp'].agg(['min']).reset_index()
    fwd_1st_fscl_ap_a = fwd_1st_fscl_ap_a.groupby('appln_id_cited')['M_fwd_1st_ap_a'].agg(['min']).reset_index()
    fwd_1st_fscl_ap_p = fwd_1st_fscl_ap_p.groupby('appln_id_cited')['M_fwd_1st_ap_p'].agg(['min']).reset_index()
    # Rename
    fwd_1st_fscl_aa.rename(columns = {'min' : 'M_fwd_1st_aa'}, inplace=True)
    fwd_1st_fscl_pa.rename(columns = {'min' : 'M_fwd_1st_pa'}, inplace=True)
    fwd_1st_fscl_ap.rename(columns = {'min' : 'M_fwd_1st_ap'}, inplace=True)
    fwd_1st_fscl_pp.rename(columns = {'min' : 'M_fwd_1st_pp'}, inplace=True)
    fwd_1st_fscl_ap_a.rename(columns = {'min' : 'M_fwd_1st_ap_a'}, inplace=True)
    fwd_1st_fscl_ap_p.rename(columns = {'min' : 'M_fwd_1st_ap_p'}, inplace=True)
    # Combine into a single DataFrame to reduce the number of merges
    df_M_fwd_1st = pd.merge(fwd_1st_fscl_aa, fwd_1st_fscl_pa, on='appln_id_cited', how='outer')
    df_M_fwd_1st = pd.merge(df_M_fwd_1st, fwd_1st_fscl_ap, on='appln_id_cited', how='outer')
    df_M_fwd_1st = pd.merge(df_M_fwd_1st, fwd_1st_fscl_pp, on='appln_id_cited', how='outer')
    df_M_fwd_1st = pd.merge(df_M_fwd_1st, fwd_1st_fscl_ap_a, on='appln_id_cited', how='outer')
    df_M_fwd_1st = pd.merge(df_M_fwd_1st, fwd_1st_fscl_ap_p, on='appln_id_cited', how='outer')
    
    ### (3) Merge ###
    df_main = df_table1[["appln_id", "publn_nr", "appln_filing_year", "earliest_publn_year", 
                         "granted", "ipc_sub", "appln_auth", "publn_auth", "docdb_family_size"]]
    df_main = df_main[df_main["ipc_sub"].notnull()]
    df_main = df_main.reset_index(drop = True)
    
    # Originality #
    org_removeNULL = compute_Originality(df_table3, 'appln_id_citing', 'ipc_sub_cited', 1)
    org_removeNULL = org_removeNULL[["appln_id_citing", "Originality"]].copy()
    org_removeNULL.rename(columns = {'Originality' : 'Originality_removeNULL'}, inplace=True)
    org_save1NULL = compute_Originality(df_table3, 'appln_id_citing', 'ipc_sub_cited', 0)
    org_save1NULL = org_save1NULL[["appln_id_citing", "Originality"]].copy()
    org_save1NULL.rename(columns = {'Originality' : 'Originality_save1NULL'}, inplace=True)
    org = pd.merge(org_removeNULL, org_save1NULL, on='appln_id_citing', how='outer')

    # Generality - "Calendar/Fiscal" Year based #
    # Prepare gen_2, gen_3, gen_5, and org DataFrames (RemoveNULL)
    gen_2_clndr_removeNULL = compute_Generality(df_table2, "appln_id_cited", "ipc_sub_citing", 1, 1, 2)[[
        "appln_id_cited", "Generality_removeNULL_clndr_2_aa", "Generality_removeNULL_clndr_2_pa", "Generality_removeNULL_clndr_2_ap", 
        "Generality_removeNULL_clndr_2_pp", "Generality_removeNULL_clndr_2_ap_a", "Generality_removeNULL_clndr_2_ap_p"]]
    gen_3_clndr_removeNULL = compute_Generality(df_table2, "appln_id_cited", "ipc_sub_citing", 1, 1, 3)[[
        "appln_id_cited", "Generality_removeNULL_clndr_3_aa", "Generality_removeNULL_clndr_3_pa", "Generality_removeNULL_clndr_3_ap", 
        "Generality_removeNULL_clndr_3_pp", "Generality_removeNULL_clndr_3_ap_a", "Generality_removeNULL_clndr_3_ap_p"]]
    gen_5_clndr_removeNULL = compute_Generality(df_table2, "appln_id_cited", "ipc_sub_citing", 1, 1, 5)[[
        "appln_id_cited", "Generality_removeNULL_clndr_5_aa", "Generality_removeNULL_clndr_5_pa", "Generality_removeNULL_clndr_5_ap", 
        "Generality_removeNULL_clndr_5_pp", "Generality_removeNULL_clndr_5_ap_a", "Generality_removeNULL_clndr_5_ap_p"]]
    gen_2_fscl_removeNULL = compute_Generality(df_table2, "appln_id_cited", "ipc_sub_citing", 1, 0, 2)[[
        "appln_id_cited", "Generality_removeNULL_fscl_2_aa", "Generality_removeNULL_fscl_2_pa", "Generality_removeNULL_fscl_2_ap", 
        "Generality_removeNULL_fscl_2_pp", "Generality_removeNULL_fscl_2_ap_a", "Generality_removeNULL_fscl_2_ap_p"]]
    gen_3_fscl_removeNULL = compute_Generality(df_table2, "appln_id_cited", "ipc_sub_citing", 1, 0, 3)[[
        "appln_id_cited", "Generality_removeNULL_fscl_3_aa", "Generality_removeNULL_fscl_3_pa", "Generality_removeNULL_fscl_3_ap", 
        "Generality_removeNULL_fscl_3_pp", "Generality_removeNULL_fscl_3_ap_a", "Generality_removeNULL_fscl_3_ap_p"]]
    gen_5_fscl_removeNULL = compute_Generality(df_table2, "appln_id_cited", "ipc_sub_citing", 1, 0, 5)[[
        "appln_id_cited", "Generality_removeNULL_fscl_5_aa", "Generality_removeNULL_fscl_5_pa", "Generality_removeNULL_fscl_5_ap", 
        "Generality_removeNULL_fscl_5_pp", "Generality_removeNULL_fscl_5_ap_a", "Generality_removeNULL_fscl_5_ap_p"]]
    gen_2_clndr_save1NULL = compute_Generality(df_table2, "appln_id_cited", "ipc_sub_citing", 0, 1, 2)[[
        "appln_id_cited", "Generality_save1NULL_clndr_2_aa", "Generality_save1NULL_clndr_2_pa", "Generality_save1NULL_clndr_2_ap", 
        "Generality_save1NULL_clndr_2_pp", "Generality_save1NULL_clndr_2_ap_a", "Generality_save1NULL_clndr_2_ap_p"]]
    gen_3_clndr_save1NULL = compute_Generality(df_table2, "appln_id_cited", "ipc_sub_citing", 0, 1, 3)[[
        "appln_id_cited", "Generality_save1NULL_clndr_3_aa", "Generality_save1NULL_clndr_3_pa", "Generality_save1NULL_clndr_3_ap", 
        "Generality_save1NULL_clndr_3_pp", "Generality_save1NULL_clndr_3_ap_a", "Generality_save1NULL_clndr_3_ap_p"]]
    gen_5_clndr_save1NULL = compute_Generality(df_table2, "appln_id_cited", "ipc_sub_citing", 0, 1, 5)[[
        "appln_id_cited", "Generality_save1NULL_clndr_5_aa", "Generality_save1NULL_clndr_5_pa", "Generality_save1NULL_clndr_5_ap", 
        "Generality_save1NULL_clndr_5_pp", "Generality_save1NULL_clndr_5_ap_a", "Generality_save1NULL_clndr_5_ap_p"]]
    gen_2_fscl_save1NULL = compute_Generality(df_table2, "appln_id_cited", "ipc_sub_citing", 0, 0, 2)[[
        "appln_id_cited", "Generality_save1NULL_fscl_2_aa", "Generality_save1NULL_fscl_2_pa", "Generality_save1NULL_fscl_2_ap", 
        "Generality_save1NULL_fscl_2_pp", "Generality_save1NULL_fscl_2_ap_a", "Generality_save1NULL_fscl_2_ap_p"]]
    gen_3_fscl_save1NULL = compute_Generality(df_table2, "appln_id_cited", "ipc_sub_citing", 0, 0, 3)[[
        "appln_id_cited", "Generality_save1NULL_fscl_3_aa", "Generality_save1NULL_fscl_3_pa", "Generality_save1NULL_fscl_3_ap", 
        "Generality_save1NULL_fscl_3_pp", "Generality_save1NULL_fscl_3_ap_a", "Generality_save1NULL_fscl_3_ap_p"]]
    gen_5_fscl_save1NULL = compute_Generality(df_table2, "appln_id_cited", "ipc_sub_citing", 0, 0, 5)[[
        "appln_id_cited", "Generality_save1NULL_fscl_5_aa", "Generality_save1NULL_fscl_5_pa", "Generality_save1NULL_fscl_5_ap", 
        "Generality_save1NULL_fscl_5_pp", "Generality_save1NULL_fscl_5_ap_a", "Generality_save1NULL_fscl_5_ap_p"]]
    # Rename columns for merge
    gen_2_clndr_removeNULL.rename(columns = {'appln_id_cited' : 'appln_id'}, inplace=True)
    gen_3_clndr_removeNULL.rename(columns = {'appln_id_cited' : 'appln_id'}, inplace=True)
    gen_5_clndr_removeNULL.rename(columns = {'appln_id_cited' : 'appln_id'}, inplace=True)
    gen_2_fscl_removeNULL.rename(columns = {'appln_id_cited' : 'appln_id'}, inplace=True)
    gen_3_fscl_removeNULL.rename(columns = {'appln_id_cited' : 'appln_id'}, inplace=True)
    gen_5_fscl_removeNULL.rename(columns = {'appln_id_cited' : 'appln_id'}, inplace=True)
    gen_2_clndr_save1NULL.rename(columns = {'appln_id_cited' : 'appln_id'}, inplace=True)
    gen_3_clndr_save1NULL.rename(columns = {'appln_id_cited' : 'appln_id'}, inplace=True)
    gen_5_clndr_save1NULL.rename(columns = {'appln_id_cited' : 'appln_id'}, inplace=True)
    gen_2_fscl_save1NULL.rename(columns = {'appln_id_cited' : 'appln_id'}, inplace=True)
    gen_3_fscl_save1NULL.rename(columns = {'appln_id_cited' : 'appln_id'}, inplace=True)
    gen_5_fscl_save1NULL.rename(columns = {'appln_id_cited' : 'appln_id'}, inplace=True)
    org.rename(columns = {'appln_id_citing' : 'appln_id'}, inplace=True)
    df_fwd_calendar.rename(columns = {'appln_id_cited' : 'appln_id'}, inplace=True)
    df_fwd_fiscal.rename(columns = {'appln_id_cited' : 'appln_id'}, inplace=True)
    # Combine gen_3, gen_5, and org into a single DataFrame to reduce the number of merges
    combined_gen_org = pd.merge(gen_2_clndr_removeNULL, gen_3_clndr_removeNULL, on='appln_id', how='outer')
    combined_gen_org = pd.merge(combined_gen_org, gen_5_clndr_removeNULL, on='appln_id', how='outer')
    combined_gen_org = pd.merge(combined_gen_org, gen_2_fscl_removeNULL, on='appln_id', how='outer')
    combined_gen_org = pd.merge(combined_gen_org, gen_3_fscl_removeNULL, on='appln_id', how='outer')
    combined_gen_org = pd.merge(combined_gen_org, gen_5_fscl_removeNULL, on='appln_id', how='outer')
    combined_gen_org = pd.merge(combined_gen_org, gen_2_clndr_save1NULL, on='appln_id', how='outer')
    combined_gen_org = pd.merge(combined_gen_org, gen_3_clndr_save1NULL, on='appln_id', how='outer')
    combined_gen_org = pd.merge(combined_gen_org, gen_5_clndr_save1NULL, on='appln_id', how='outer')
    combined_gen_org = pd.merge(combined_gen_org, gen_2_fscl_save1NULL, on='appln_id', how='outer')
    combined_gen_org = pd.merge(combined_gen_org, gen_3_fscl_save1NULL, on='appln_id', how='outer')
    combined_gen_org = pd.merge(combined_gen_org, gen_5_fscl_save1NULL, on='appln_id', how='outer')
    combined_gen_org = pd.merge(combined_gen_org, org, on='appln_id', how='outer')
    
    # Perform a single merge operation
    df_main = df_main.merge(df_fwd_calendar, on='appln_id', how='left') # w/ forward citations
    df_main = df_main.merge(df_fwd_fiscal, on='appln_id', how='left') # w/ forward citations
    df_main = df_main.merge(combined_gen_org, on='appln_id', how='left') # w/ generality & originality

    # Make NaN (forward citation value) zero
    # Columns where NaN values should be replaced with 0
    columns_to_replace_nan = [
        'fwd_aa_clndr_2', 'fwd_pa_clndr_2', 'fwd_ap_clndr_2', 'fwd_pp_clndr_2',
        'fwd_aa_clndr_3', 'fwd_pa_clndr_3', 'fwd_ap_clndr_3', 'fwd_pp_clndr_3',
        'fwd_aa_clndr_5', 'fwd_pa_clndr_5', 'fwd_ap_clndr_5', 'fwd_pp_clndr_5',
        'fwd_aa_fscl_2', 'fwd_pa_fscl_2', 'fwd_ap_fscl_2', 'fwd_pp_fscl_2',
        'fwd_aa_fscl_3', 'fwd_pa_fscl_3', 'fwd_ap_fscl_3', 'fwd_pp_fscl_3',
        'fwd_aa_fscl_5', 'fwd_pa_fscl_5', 'fwd_ap_fscl_5', 'fwd_pp_fscl_5', 
        'fwd_ap_a_clndr_2', 'fwd_ap_a_clndr_3', 'fwd_ap_a_clndr_5', 'fwd_ap_a_fscl_2', 'fwd_ap_a_fscl_3', 'fwd_ap_a_fscl_5'
        ]
    # Replace NaN values with 0
    df_main[columns_to_replace_nan] = df_main[columns_to_replace_nan].fillna(0)
    
    # FWD/BWD Lags (Median, Mean) - "Calendar/Fiscal" Year based #
    # FWD
    # Rename columns for merge
    fwd_lag2_clndr_aa.rename(columns = {'median' : 'fwd_lag2_clndr_aa_median', 'mean': 'fwd_lag2_clndr_aa_mean'}, inplace=True)
    fwd_lag2_clndr_pa.rename(columns = {'median' : 'fwd_lag2_clndr_pa_median', 'mean': 'fwd_lag2_clndr_pa_mean'}, inplace=True)
    fwd_lag2_clndr_ap.rename(columns = {'median' : 'fwd_lag2_clndr_ap_median', 'mean': 'fwd_lag2_clndr_ap_mean'}, inplace=True)
    fwd_lag2_clndr_pp.rename(columns = {'median' : 'fwd_lag2_clndr_pp_median', 'mean': 'fwd_lag2_clndr_pp_mean'}, inplace=True)
    fwd_lag2_clndr_ap_a.rename(columns = {'median' : 'fwd_lag2_clndr_ap_a_median', 'mean': 'fwd_lag2_clndr_ap_a_mean'}, inplace=True)
    fwd_lag2_clndr_ap_p.rename(columns = {'median' : 'fwd_lag2_clndr_ap_p_median', 'mean': 'fwd_lag2_clndr_ap_p_mean'}, inplace=True)
    fwd_lag2_fscl_aa.rename(columns = {'median' : 'fwd_lag2_M_fscl_aa_median', 'mean': 'fwd_lag2_M_fscl_aa_mean'}, inplace=True)
    fwd_lag2_fscl_pa.rename(columns = {'median' : 'fwd_lag2_M_fscl_pa_median', 'mean': 'fwd_lag2_M_fscl_pa_mean'}, inplace=True)
    fwd_lag2_fscl_ap.rename(columns = {'median' : 'fwd_lag2_M_fscl_ap_median', 'mean': 'fwd_lag2_M_fscl_ap_mean'}, inplace=True)
    fwd_lag2_fscl_pp.rename(columns = {'median' : 'fwd_lag2_M_fscl_pp_median', 'mean': 'fwd_lag2_M_fscl_pp_mean'}, inplace=True)
    fwd_lag2_fscl_ap_a.rename(columns = {'median' : 'fwd_lag2_M_fscl_ap_a_median', 'mean': 'fwd_lag2_M_fscl_ap_a_mean'}, inplace=True)
    fwd_lag2_fscl_ap_p.rename(columns = {'median' : 'fwd_lag2_M_fscl_ap_p_median', 'mean': 'fwd_lag2_M_fscl_ap_p_mean'}, inplace=True)
    fwd_lag3_clndr_aa.rename(columns = {'median' : 'fwd_lag3_clndr_aa_median', 'mean': 'fwd_lag3_clndr_aa_mean'}, inplace=True)
    fwd_lag3_clndr_pa.rename(columns = {'median' : 'fwd_lag3_clndr_pa_median', 'mean': 'fwd_lag3_clndr_pa_mean'}, inplace=True)
    fwd_lag3_clndr_ap.rename(columns = {'median' : 'fwd_lag3_clndr_ap_median', 'mean': 'fwd_lag3_clndr_ap_mean'}, inplace=True)
    fwd_lag3_clndr_pp.rename(columns = {'median' : 'fwd_lag3_clndr_pp_median', 'mean': 'fwd_lag3_clndr_pp_mean'}, inplace=True)
    fwd_lag3_clndr_ap_a.rename(columns = {'median' : 'fwd_lag3_clndr_ap_a_median', 'mean': 'fwd_lag3_clndr_ap_a_mean'}, inplace=True)
    fwd_lag3_clndr_ap_p.rename(columns = {'median' : 'fwd_lag3_clndr_ap_p_median', 'mean': 'fwd_lag3_clndr_ap_p_mean'}, inplace=True)
    fwd_lag3_fscl_aa.rename(columns = {'median' : 'fwd_lag3_M_fscl_aa_median', 'mean': 'fwd_lag3_M_fscl_aa_mean'}, inplace=True)
    fwd_lag3_fscl_pa.rename(columns = {'median' : 'fwd_lag3_M_fscl_pa_median', 'mean': 'fwd_lag3_M_fscl_pa_mean'}, inplace=True)
    fwd_lag3_fscl_ap.rename(columns = {'median' : 'fwd_lag3_M_fscl_ap_median', 'mean': 'fwd_lag3_M_fscl_ap_mean'}, inplace=True)
    fwd_lag3_fscl_pp.rename(columns = {'median' : 'fwd_lag3_M_fscl_pp_median', 'mean': 'fwd_lag3_M_fscl_pp_mean'}, inplace=True)
    fwd_lag3_fscl_ap_a.rename(columns = {'median' : 'fwd_lag3_M_fscl_ap_a_median', 'mean': 'fwd_lag3_M_fscl_ap_a_mean'}, inplace=True)
    fwd_lag3_fscl_ap_p.rename(columns = {'median' : 'fwd_lag3_M_fscl_ap_p_median', 'mean': 'fwd_lag3_M_fscl_ap_p_mean'}, inplace=True)
    fwd_lag5_clndr_aa.rename(columns = {'median' : 'fwd_lag5_clndr_aa_median', 'mean': 'fwd_lag5_clndr_aa_mean'}, inplace=True)
    fwd_lag5_clndr_pa.rename(columns = {'median' : 'fwd_lag5_clndr_pa_median', 'mean': 'fwd_lag5_clndr_pa_mean'}, inplace=True)
    fwd_lag5_clndr_ap.rename(columns = {'median' : 'fwd_lag5_clndr_ap_median', 'mean': 'fwd_lag5_clndr_ap_mean'}, inplace=True)
    fwd_lag5_clndr_pp.rename(columns = {'median' : 'fwd_lag5_clndr_pp_median', 'mean': 'fwd_lag5_clndr_pp_mean'}, inplace=True)
    fwd_lag5_clndr_ap_a.rename(columns = {'median' : 'fwd_lag5_clndr_ap_a_median', 'mean': 'fwd_lag5_clndr_ap_a_mean'}, inplace=True)
    fwd_lag5_clndr_ap_p.rename(columns = {'median' : 'fwd_lag5_clndr_ap_p_median', 'mean': 'fwd_lag5_clndr_ap_p_mean'}, inplace=True)
    fwd_lag5_fscl_aa.rename(columns = {'median' : 'fwd_lag5_M_fscl_aa_median', 'mean': 'fwd_lag5_M_fscl_aa_mean'}, inplace=True)
    fwd_lag5_fscl_pa.rename(columns = {'median' : 'fwd_lag5_M_fscl_pa_median', 'mean': 'fwd_lag5_M_fscl_pa_mean'}, inplace=True)
    fwd_lag5_fscl_ap.rename(columns = {'median' : 'fwd_lag5_M_fscl_ap_median', 'mean': 'fwd_lag5_M_fscl_ap_mean'}, inplace=True)
    fwd_lag5_fscl_pp.rename(columns = {'median' : 'fwd_lag5_M_fscl_pp_median', 'mean': 'fwd_lag5_M_fscl_pp_mean'}, inplace=True)
    fwd_lag5_fscl_ap_a.rename(columns = {'median' : 'fwd_lag5_M_fscl_ap_a_median', 'mean': 'fwd_lag5_M_fscl_ap_a_mean'}, inplace=True)
    fwd_lag5_fscl_ap_p.rename(columns = {'median' : 'fwd_lag5_M_fscl_ap_p_median', 'mean': 'fwd_lag5_M_fscl_ap_p_mean'}, inplace=True)
    # Combine into a single DataFrame to reduce the number of merges
    fwd_lag_stats = pd.merge(fwd_lag2_clndr_aa, fwd_lag2_clndr_pa, on='appln_id_cited', how='outer')
    fwd_lag_stats = pd.merge(fwd_lag_stats, fwd_lag2_clndr_ap, on='appln_id_cited', how='outer')
    fwd_lag_stats = pd.merge(fwd_lag_stats, fwd_lag2_clndr_pp, on='appln_id_cited', how='outer')
    fwd_lag_stats = pd.merge(fwd_lag_stats, fwd_lag2_clndr_ap_a, on='appln_id_cited', how='outer')
    fwd_lag_stats = pd.merge(fwd_lag_stats, fwd_lag2_clndr_ap_p, on='appln_id_cited', how='outer')
    fwd_lag_stats = pd.merge(fwd_lag_stats, fwd_lag2_fscl_aa, on='appln_id_cited', how='outer')
    fwd_lag_stats = pd.merge(fwd_lag_stats, fwd_lag2_fscl_pa, on='appln_id_cited', how='outer')
    fwd_lag_stats = pd.merge(fwd_lag_stats, fwd_lag2_fscl_ap, on='appln_id_cited', how='outer')
    fwd_lag_stats = pd.merge(fwd_lag_stats, fwd_lag2_fscl_pp, on='appln_id_cited', how='outer')
    fwd_lag_stats = pd.merge(fwd_lag_stats, fwd_lag2_fscl_ap_a, on='appln_id_cited', how='outer')
    fwd_lag_stats = pd.merge(fwd_lag_stats, fwd_lag2_fscl_ap_p, on='appln_id_cited', how='outer')
    fwd_lag_stats = pd.merge(fwd_lag_stats, fwd_lag3_clndr_aa, on='appln_id_cited', how='outer')
    fwd_lag_stats = pd.merge(fwd_lag_stats, fwd_lag3_clndr_pa, on='appln_id_cited', how='outer')
    fwd_lag_stats = pd.merge(fwd_lag_stats, fwd_lag3_clndr_ap, on='appln_id_cited', how='outer')
    fwd_lag_stats = pd.merge(fwd_lag_stats, fwd_lag3_clndr_pp, on='appln_id_cited', how='outer')
    fwd_lag_stats = pd.merge(fwd_lag_stats, fwd_lag3_clndr_ap_a, on='appln_id_cited', how='outer')
    fwd_lag_stats = pd.merge(fwd_lag_stats, fwd_lag3_clndr_ap_p, on='appln_id_cited', how='outer')
    fwd_lag_stats = pd.merge(fwd_lag_stats, fwd_lag3_fscl_aa, on='appln_id_cited', how='outer')
    fwd_lag_stats = pd.merge(fwd_lag_stats, fwd_lag3_fscl_pa, on='appln_id_cited', how='outer')
    fwd_lag_stats = pd.merge(fwd_lag_stats, fwd_lag3_fscl_ap, on='appln_id_cited', how='outer')
    fwd_lag_stats = pd.merge(fwd_lag_stats, fwd_lag3_fscl_pp, on='appln_id_cited', how='outer')
    fwd_lag_stats = pd.merge(fwd_lag_stats, fwd_lag3_fscl_ap_a, on='appln_id_cited', how='outer')
    fwd_lag_stats = pd.merge(fwd_lag_stats, fwd_lag3_fscl_ap_p, on='appln_id_cited', how='outer')
    fwd_lag_stats = pd.merge(fwd_lag_stats, fwd_lag5_clndr_aa, on='appln_id_cited', how='outer')
    fwd_lag_stats = pd.merge(fwd_lag_stats, fwd_lag5_clndr_pa, on='appln_id_cited', how='outer')
    fwd_lag_stats = pd.merge(fwd_lag_stats, fwd_lag5_clndr_ap, on='appln_id_cited', how='outer')
    fwd_lag_stats = pd.merge(fwd_lag_stats, fwd_lag5_clndr_pp, on='appln_id_cited', how='outer')
    fwd_lag_stats = pd.merge(fwd_lag_stats, fwd_lag5_clndr_ap_a, on='appln_id_cited', how='outer')
    fwd_lag_stats = pd.merge(fwd_lag_stats, fwd_lag5_clndr_ap_p, on='appln_id_cited', how='outer')
    fwd_lag_stats = pd.merge(fwd_lag_stats, fwd_lag5_fscl_aa, on='appln_id_cited', how='outer')
    fwd_lag_stats = pd.merge(fwd_lag_stats, fwd_lag5_fscl_pa, on='appln_id_cited', how='outer')
    fwd_lag_stats = pd.merge(fwd_lag_stats, fwd_lag5_fscl_ap, on='appln_id_cited', how='outer')
    fwd_lag_stats = pd.merge(fwd_lag_stats, fwd_lag5_fscl_pp, on='appln_id_cited', how='outer')
    fwd_lag_stats = pd.merge(fwd_lag_stats, fwd_lag5_fscl_ap_a, on='appln_id_cited', how='outer')
    fwd_lag_stats = pd.merge(fwd_lag_stats, fwd_lag5_fscl_ap_p, on='appln_id_cited', how='outer')
    fwd_lag_stats.rename(columns = {'appln_id_cited' : 'appln_id'}, inplace=True)

    # BWD
    # Rename columns for merge
    bwd_lag_clndr_aa.rename(columns = {'median' : 'bwd_lag_clndr_aa_median', 'mean': 'bwd_lag_clndr_aa_mean'}, inplace=True)
    bwd_lag_clndr_pa.rename(columns = {'median' : 'bwd_lag_clndr_pa_median', 'mean': 'bwd_lag_clndr_pa_mean'}, inplace=True)
    bwd_lag_clndr_ap.rename(columns = {'median' : 'bwd_lag_clndr_ap_median', 'mean': 'bwd_lag_clndr_ap_mean'}, inplace=True)
    bwd_lag_clndr_pp.rename(columns = {'median' : 'bwd_lag_clndr_pp_median', 'mean': 'bwd_lag_clndr_pp_mean'}, inplace=True)
    bwd_lag_fscl_aa.rename(columns = {'median' : 'bwd_lag_M_fscl_aa_median', 'mean': 'bwd_lag_M_fscl_aa_mean'}, inplace=True)
    bwd_lag_fscl_pa.rename(columns = {'median' : 'bwd_lag_M_fscl_pa_median', 'mean': 'bwd_lag_M_fscl_pa_mean'}, inplace=True)
    bwd_lag_fscl_ap.rename(columns = {'median' : 'bwd_lag_M_fscl_ap_median', 'mean': 'bwd_lag_M_fscl_ap_mean'}, inplace=True)
    bwd_lag_fscl_pp.rename(columns = {'median' : 'bwd_lag_M_fscl_pp_median', 'mean': 'bwd_lag_M_fscl_pp_mean'}, inplace=True)
    # Combine into a single DataFrame to reduce the number of merges
    bwd_lag_stats = pd.merge(bwd_lag_clndr_aa, bwd_lag_clndr_pa, on='appln_id_citing', how='outer')
    bwd_lag_stats = pd.merge(bwd_lag_stats, bwd_lag_clndr_ap, on='appln_id_citing', how='outer')
    bwd_lag_stats = pd.merge(bwd_lag_stats, bwd_lag_clndr_pp, on='appln_id_citing', how='outer')
    bwd_lag_stats = pd.merge(bwd_lag_stats, bwd_lag_fscl_aa, on='appln_id_citing', how='outer')
    bwd_lag_stats = pd.merge(bwd_lag_stats, bwd_lag_fscl_pa, on='appln_id_citing', how='outer')
    bwd_lag_stats = pd.merge(bwd_lag_stats, bwd_lag_fscl_ap, on='appln_id_citing', how='outer')
    bwd_lag_stats = pd.merge(bwd_lag_stats, bwd_lag_fscl_pp, on='appln_id_citing', how='outer')
    bwd_lag_stats.rename(columns = {'appln_id_citing' : 'appln_id'}, inplace=True)
    # FWD & BWD: Combine into a single DataFrame to reduce the number of merges
    combined_lag = pd.merge(fwd_lag_stats, bwd_lag_stats, on='appln_id', how='outer')

    # Perform a single merge operation
    df_main = df_main.merge(combined_lag, on='appln_id', how='left')
    
    # How many years a patent takes to receive its first forward citation
    df_Y_fwd_1st.rename(columns = {'appln_id_cited' : 'appln_id'}, inplace=True)
    df_M_fwd_1st.rename(columns = {'appln_id_cited' : 'appln_id'}, inplace=True)
    combined_fwd_1st = pd.merge(df_Y_fwd_1st, df_M_fwd_1st, on='appln_id', how='outer')
    df_main = df_main.merge(combined_fwd_1st, on='appln_id', how='left')
    
    tot_end_time = time.time() # Record the end time
    duration = tot_end_time - tot_start_time
    print("@ Start time =", tot_start)
    print("@ End time =", time.strftime('%Y.%m.%d - %H:%M:%S'))
    print(f"The execution time: {round(duration, 3)} seconds.")
    
    os.chdir(initial_path)
    folder_path = f'../ch2_big5_scale/patstat_each_nation_year_vars/{country_select.upper()}' # Set saving path
    os.chdir(folder_path)
    df_main.to_pickle(f'pat_IPC4_allC_{country_select.lower_case()}_{year_select}.pkl') # Dataset based on each patent application

    os.chdir(initial_path)
    print("Reverted to initial directory:", os.getcwd())
    
    return df_main


######################################################
### Checking the statistics of "df_main" dataframe ###
######################################################
def df_main_stat(df_main):
    print("(1) # of unique patent applications =", len(df_main) )
    print("(2) # of unique patent applications (w/ IPC code) =", len(df_main[df_main["ipc_sub"].notnull()]) )

    print("=====================================================")
    print("(3) # of unique patent applications (same authorities in application and publication) =", 
        len( df_main[df_main["appln_auth"] == df_main["publn_auth"]] ) )
    print("(4) # of unique patent applications (w/ Granted) =", len(df_main[df_main["granted"]=="Y"]) )

    print("=====================================================")
    print("(5) # of unique patent applications (satisfying (2)&(3)) =", 
        len( df_main[(df_main["appln_auth"] == df_main["publn_auth"]) & \
                    (df_main["ipc_sub"].notnull())
                    ] 
            ) 
        )
    print("(6) # of unique patent applications (satisfying (2)&(3)&(4)) =", 
        len( df_main[(df_main["appln_auth"] == df_main["publn_auth"]) & \
                    (df_main["ipc_sub"].notnull()) & \
                    (df_main["granted"]=="Y")
                    ] 
            ) 
        )

    print("=====================================================")
    print("(7) # of unique IPC codes =", len(df_main["ipc_sub"].unique()) )
    print("(8) Unique application years =", df_main["appln_filing_year"].unique())
    print("(9) Unique application years =", df_main["earliest_publn_year"].unique())


############################################################
### Sub functions (in the main functions: make_measures) ###
############################################################

### Forward Citations-related ###
# (A) Implementing the optimized function for fwd citations (calendar year) calculations
def calculate_conditions_calendar(df, k_values):
    for k in k_values:
        df[f'cond_a_{k}'] = ((df['appln_filing_year_citing'] >= df['appln_filing_year_cited']) &
                                (df['appln_filing_year_citing'] <= df['appln_filing_year_cited'] + k)) # appln (cited) ~ appln +k (cited) <- appln (citing)
        df[f'cond_b_{k}'] = ((df['appln_filing_year_citing'] >= df['publn_year_cited']) &
                                (df['appln_filing_year_citing'] <= df['publn_year_cited'] + k)) # publn (cited) ~ publn +k (cited) <- appln (citing)
        df[f'cond_c_{k}'] = ((df['publn_year_citing'] >= df['appln_filing_year_cited']) &
                                (df['publn_year_citing'] <= df['appln_filing_year_cited'] + k)) # appln (cited) ~ appln +k (cited) <- publn (citing)
        df[f'cond_d_{k}'] = ((df['publn_year_citing'] >= df['publn_year_cited']) &
                                (df['publn_year_citing'] <= df['publn_year_cited'] + k)) # publn (cited) ~ publn +k (cited) <- publn (citing)
        df[f'cond_e_{k}'] = ((df['appln_filing_year_citing'] >= df['appln_filing_year_cited']) &
                            (df['appln_filing_year_citing'] <= df['publn_year_cited'] + k)) # appln (cited) ~ publn +k (cited) <- appln (citing)
        df[f'cond_f_{k}'] = ((df['publn_year_citing'] >= df['appln_filing_year_cited']) &
                                (df['publn_year_citing'] <= df['publn_year_cited'] + k)) # appln (cited) ~ publn +k (cited) <- publn (citing)

    cols_to_sum = [col for col in df.columns if col.startswith('cond_')]
    df_grouped = df.groupby('appln_id_cited')[cols_to_sum].sum()

    rename_map = {}
    for k in k_values:
        rename_map.update({
            f'cond_a_{k}': f'fwd_aa_clndr_{k}',
            f'cond_b_{k}': f'fwd_pa_clndr_{k}',
            f'cond_c_{k}': f'fwd_ap_clndr_{k}',
            f'cond_d_{k}': f'fwd_pp_clndr_{k}',
            f'cond_e_{k}': f'fwd_ap_a_clndr_{k}',
            f'cond_f_{k}': f'fwd_ap_p_clndr_{k}',
            })
    df_grouped.rename(columns=rename_map, inplace=True)
    
    return df_grouped.reset_index()

# (B) Implementing the optimized function for fwd citations (fiscal year) calculations
def calculate_conditions_fiscal(df, k_values):
    for k in k_values:
        df[f'cond_a_{k}'] = ((df['appln_filing_date_citing'] >= df['appln_filing_date_cited']) &
                             (df['appln_filing_date_citing'] <= df[f'appln_filing_date_cited_t+{k}'])) # appln (cited) ~ appln +k (cited) <- appln (citing)
        df[f'cond_b_{k}'] = ((df['appln_filing_date_citing'] >= df['publn_date_cited']) &
                             (df['appln_filing_date_citing'] <= df[f'publn_date_cited_t+{k}'])) # publn (cited) ~ publn +k (cited) <- appln (citing)
        df[f'cond_c_{k}'] = ((df['publn_date_citing'] >= df['appln_filing_date_cited']) &
                             (df['publn_date_citing'] <= df[f'appln_filing_date_cited_t+{k}'])) # appln (cited) ~ appln +k (cited) <- publn (citing)
        df[f'cond_d_{k}'] = ((df['publn_date_citing'] >= df['publn_date_cited']) &
                             (df['publn_date_citing'] <= df[f'publn_date_cited_t+{k}'])) # publn (cited) ~ publn +k (cited) <- publn (citing)
        df[f'cond_e_{k}'] = ((df['appln_filing_date_citing'] >= df['appln_filing_date_cited']) &
                             (df['appln_filing_date_citing'] <= df[f'publn_date_cited_t+{k}'])) # appln (cited) ~ publn +k (cited) <- appln (citing)
        df[f'cond_f_{k}'] = ((df['publn_date_citing'] >= df['appln_filing_date_cited']) &
                             (df['publn_date_citing'] <= df[f'publn_date_cited_t+{k}'])) # appln (cited) ~ publn +k (cited) <- publn (citing)

    cols_to_sum = [col for col in df.columns if col.startswith('cond_')]
    df_grouped = df.groupby('appln_id_cited')[cols_to_sum].sum()

    rename_map = {}
    for k in k_values:
        rename_map.update({
            f'cond_a_{k}': f'fwd_aa_fscl_{k}',
            f'cond_b_{k}': f'fwd_pa_fscl_{k}',
            f'cond_c_{k}': f'fwd_ap_fscl_{k}',
            f'cond_d_{k}': f'fwd_pp_fscl_{k}',
            f'cond_e_{k}': f'fwd_ap_a_fscl_{k}',
            f'cond_f_{k}': f'fwd_ap_p_fscl_{k}',
        })
    df_grouped.rename(columns=rename_map, inplace=True)
    
    return df_grouped.reset_index()

### Generality-related ###
# (A) Implementing the function for Generality calculations (forward ciation based)
def compute_Generality(df, unique_id, counting_col, Select_NULL_option, Y_criteria, k):
    original_rows = len(df)
    if Select_NULL_option == 1: # Delete all null values
        df = df[df[counting_col].notnull()].reset_index(drop=True)
        print(f"We deleted rows which have NULL values in the column \"{counting_col}\". \nSo the dataframe: previous # of rows={original_rows} and now, # of rows={len(df)}")
    else: # Survive when only one unknown IPC code
        unique_id_counts = df[unique_id].value_counts() # Count the occurrence of each unique_id
        # Mark rows that have a unique `unique_id` or non-NULL `counting_col`
        keep_mask = df[unique_id].isin(unique_id_counts[unique_id_counts == 1].index) | df[counting_col].notnull()
        # Apply the mask to keep desired rows
        df = df.loc[keep_mask].reset_index(drop=True)
        print(f"Adjusted NULL handling (Only one cited patent with unknown IPC code: OK). \n previous # of rows={original_rows}, now # of rows={len(df)}")
    # Calendar Year condition checks
    if Y_criteria == 1:
        conditions = {
            'aa': (df["appln_filing_year_citing"] >= df["appln_filing_year_cited"]) & \
            (df["appln_filing_year_citing"] <= df["appln_filing_year_cited"] + k),
            'pa': (df["appln_filing_year_citing"] >= df["publn_year_cited"]) & \
            (df["appln_filing_year_citing"] <= df["publn_year_cited"] + k),
            'ap': (df["publn_year_citing"] >= df["appln_filing_year_cited"]) & \
            (df["publn_year_citing"] <= df["appln_filing_year_cited"] + k),
            'pp': (df["publn_year_citing"] >= df["publn_year_cited"]) & \
            (df["publn_year_citing"] <= df["publn_year_cited"] + k),
            # filing year ~ until publication year + (3 or 5)years: total citations
            'ap_a': (df["appln_filing_year_citing"] >= df["appln_filing_year_cited"]) & \
            (df["appln_filing_year_citing"] <= df["publn_year_cited"] + k),
            'ap_p': (df["publn_year_citing"] >= df["appln_filing_year_cited"]) & \
            (df["publn_year_citing"] <= df["publn_year_cited"] + k)
        }
    else:  # Fiscal Year requires specific date adjustments
        # Placeholder for actual fiscal year date calculations
        # Assuming df contains columns like 'appln_filing_date_cited_t+k' for fiscal year adjustments
        conditions = {
            'aa': (df["appln_filing_date_citing"] >= df["appln_filing_date_cited"]) & \
            (df["appln_filing_date_citing"] <= df[f'appln_filing_date_cited_t+{k}']),
            'pa': (df["appln_filing_date_citing"] >= df["publn_date_cited"]) & \
            (df["appln_filing_date_citing"] <= df[f'publn_date_cited_t+{k}']),
            'ap': (df["publn_date_citing"] >= df["appln_filing_date_cited"]) & \
            (df["publn_date_citing"] <= df[f'appln_filing_date_cited_t+{k}']),
            'pp': (df["publn_date_citing"] >= df["publn_date_cited"]) & \
            (df["publn_date_citing"] <= df[f'publn_date_cited_t+{k}']),
            # filing date ~ until publication date + (3 or 5)years: total citations
            'ap_a': (df["appln_filing_date_citing"] >= df["appln_filing_date_cited"]) & \
            (df["appln_filing_date_citing"] <= df[f'publn_date_cited_t+{k}']),
            'ap_p': (df["publn_date_citing"] >= df["appln_filing_date_cited"]) & \
            (df["publn_date_citing"] <= df[f'publn_date_cited_t+{k}'])
        }
    # Initialize result DataFrame
    result_df = pd.DataFrame(df[unique_id].unique(), columns=[unique_id]).set_index(unique_id)
    # Function to calculate HHI and Generality for each condition
    def calculate_HHI_Generality(group):
        N = group.size
        counter = Counter(group[counting_col])
        if N == 0:
            HHI = np.nan
            Generality = np.nan
        else:
            HHI = sum((n / N) ** 2 for n in counter.values())
            Generality = ((N) / (N - 1)) * (1 - HHI) if N > 1 else 0
        return pd.Series([HHI, Generality], index=['HHI', 'Generality'])
    # Compute metrics for each condition and merge results
    for condition_key, condition in conditions.items():
        filtered_df = df[condition]
        metrics_df = filtered_df.groupby(unique_id)[[counting_col]].apply(calculate_HHI_Generality).reset_index()
        if Select_NULL_option == 1: # Delete all null values
            if Y_criteria == 1: # Calendar
                metrics_df.rename(columns={'HHI': f'HHI_removeNULL_clndr_{k}_{condition_key}', 
                                           'Generality': f'Generality_removeNULL_clndr_{k}_{condition_key}'}, inplace = True)
            else: # Fiscal
                metrics_df.rename(columns={'HHI': f'HHI_removeNULL_fscl_{k}_{condition_key}', 
                                           'Generality': f'Generality_removeNULL_fscl_{k}_{condition_key}'}, inplace = True)
        else: # Survive when only one unknown IPC code
            if Y_criteria == 1: # Calendar
                metrics_df.rename(columns={'HHI': f'HHI_save1NULL_clndr_{k}_{condition_key}', 
                                           'Generality': f'Generality_save1NULL_clndr_{k}_{condition_key}'}, inplace = True)
            else: # Fiscal
                metrics_df.rename(columns={'HHI': f'HHI_save1NULL_fscl_{k}_{condition_key}', 
                                           'Generality': f'Generality_save1NULL_fscl_{k}_{condition_key}'}, inplace = True)
        result_df = result_df.merge(metrics_df, on=unique_id, how='left')

    return result_df.reset_index()

### Originality-related ###
# (A) Implementing the function for Originality calculations (backward ciation based)
def compute_Originality(df, unique_id, counting_col, Select_NULL_option):
    original_rows = len(df)
    if Select_NULL_option == 1: # Delete all null values
        df = df[df[counting_col].notnull()].reset_index(drop=True)
        print(f"We deleted rows which have at least one NULL value in the column \"{counting_col}\". \nSo the dataframe: previous # of rows={original_rows} and now, # of rows={len(df)}")
    else: # Survive when only one unknown IPC code
        unique_id_counts = df[unique_id].value_counts() # Count the occurrence of each unique_id
        # Mark rows that have a unique `unique_id` or non-NULL `counting_col`
        keep_mask = df[unique_id].isin(unique_id_counts[unique_id_counts == 1].index) | df[counting_col].notnull()        
        # Apply the mask to keep desired rows
        df = df[keep_mask].reset_index(drop=True)
        print(f"Adjusted NULL handling (Only one cited patent with unknown IPC code: OK). \n previous # of rows={original_rows}, now # of rows={len(df)}")

    # Ensure counting_col is a category for efficient computation
    df[counting_col] = df[counting_col].astype('category').cat.add_categories(["NULL"]).fillna("NULL")
    # Calculate counts of each category within groups, including "NULL" as a category
    counts = df.groupby([unique_id, counting_col]).size().unstack(fill_value=0)
    # Calculate N (the total counts of categories per group)
    N = counts.sum(axis=1)
    # Initialize HHI and Originality with NaN values
    HHI = pd.Series(np.nan, index=N.index)
    Originality = pd.Series(np.nan, index=N.index)
    # Only calculate HHI and Originality for N > 0
    N_gt_0 = N > 0
    proportions = counts[N_gt_0].divide(N[N_gt_0], axis=0)
    HHI[N_gt_0] = (proportions**2).sum(axis=1)
    Originality[N_gt_0] = (N[N_gt_0] / (N[N_gt_0] - 1)) * (1 - HHI[N_gt_0])
    Originality[N == 1] = 0  # Set Originality to 0 when N = 1
    # Remove the "NULL" category before finalizing results
    if "NULL" in counts.columns:
        counts = counts.drop(columns=["NULL"])
    # Prepare the final DataFrame
    results = pd.DataFrame({'HHI': HHI, 'Originality': Originality}).reset_index()
    
    return results

### Citation Lag-related ###
def calculate_fwd_citation_lag(df, year_citing, year_cited, year_diff, lag_type):
    fwd_lag = (df[(df[year_citing] >= df[year_cited]) & 
                  (df[year_citing] <= df[year_cited] + year_diff)])[["appln_id_cited", year_citing, year_cited]]
    fwd_lag[lag_type] = fwd_lag[year_citing] - fwd_lag[year_cited]
    fwd_lag_agg = fwd_lag.groupby('appln_id_cited')[lag_type].agg(['median', 'mean']).reset_index()
    
    return fwd_lag_agg

def calculate_fwd_citation_lag_ap_a(df, year_citing, year_cited_app, year_cited_pub, year_diff, lag_type):
    fwd_lag = (df[(df[year_citing] >= df[year_cited_app]) & 
                  (df[year_citing] <= df[year_cited_pub] + year_diff)])[
        ["appln_id_cited", year_citing, year_cited_app, year_cited_pub]]
    fwd_lag[lag_type] = fwd_lag[year_citing] - fwd_lag[year_cited_app]
    fwd_lag_agg = fwd_lag.groupby('appln_id_cited')[lag_type].agg(['median', 'mean']).reset_index()
    
    return fwd_lag_agg

def calculate_fwd_citation_lag_ap_p(df, year_citing_pub, year_cited_app, year_cited_pub, year_diff, lag_type):
    fwd_lag = (df[(df[year_citing_pub] >= df[year_cited_app]) & 
                  (df[year_citing_pub] <= df[year_cited_pub] + year_diff)])[
        ["appln_id_cited", year_citing_pub, year_cited_app, year_cited_pub]]
    fwd_lag[lag_type] = fwd_lag[year_citing_pub] - fwd_lag[year_cited_app]
    fwd_lag_agg = fwd_lag.groupby('appln_id_cited')[lag_type].agg(['median', 'mean']).reset_index()
    
    return fwd_lag_agg

def calculate_fwd_citation_lag_fiscal(df, date_citing, date_cited, date_cited_t_plus, lag_type):
    df[date_citing] = pd.to_datetime(df[date_citing], format='%Y%m%d')
    df[date_cited] = pd.to_datetime(df[date_cited], format='%Y%m%d')
    df[date_cited_t_plus] = pd.to_datetime(df[date_cited_t_plus], format='%Y%m%d')
    fwd_lag = (df[(df[date_citing] >= df[date_cited]) & 
                  (df[date_citing] <= df[date_cited_t_plus])])[["appln_id_cited", date_citing, date_cited]]
    fwd_lag[lag_type] = (fwd_lag[date_citing].dt.year - fwd_lag[date_cited].dt.year) * 12 + \
                        (fwd_lag[date_citing].dt.month - fwd_lag[date_cited].dt.month)
    fwd_lag_agg = fwd_lag.groupby('appln_id_cited')[lag_type].agg(['median', 'mean']).reset_index()
    
    return fwd_lag_agg

def calculate_fwd_citation_lag_fscl_ap_a(df, date_citing, date_cited_app, date_cited_t_plus, lag_type):
    df[date_citing] = pd.to_datetime(df[date_citing], format='%Y%m%d')
    df[date_cited_app] = pd.to_datetime(df[date_cited_app], format='%Y%m%d')
    df[date_cited_t_plus] = pd.to_datetime(df[date_cited_t_plus], format='%Y%m%d')
    fwd_lag = (df[(df[date_citing] >= df[date_cited_app]) & 
                  (df[date_citing] <= df[date_cited_t_plus])])[["appln_id_cited", date_citing, date_cited_app]]
    fwd_lag[lag_type] = (fwd_lag[date_citing].dt.year - fwd_lag[date_cited_app].dt.year) * 12 + \
                        (fwd_lag[date_citing].dt.month - fwd_lag[date_cited_app].dt.month)
    fwd_lag_agg = fwd_lag.groupby('appln_id_cited')[lag_type].agg(['median', 'mean']).reset_index()
    
    return fwd_lag_agg

def calculate_fwd_citation_lag_fscl_ap_p(df, date_citing_pub, date_cited_app, date_cited_t_plus, lag_type):
    df[date_citing_pub] = pd.to_datetime(df[date_citing_pub], format='%Y%m%d')
    df[date_cited_app] = pd.to_datetime(df[date_cited_app], format='%Y%m%d')
    df[date_cited_t_plus] = pd.to_datetime(df[date_cited_t_plus], format='%Y%m%d')
    fwd_lag = (df[(df[date_citing_pub] >= df[date_cited_app]) & 
                  (df[date_citing_pub] <= df[date_cited_t_plus])])[["appln_id_cited", date_citing_pub, date_cited_app]]
    fwd_lag[lag_type] = (fwd_lag[date_citing_pub].dt.year - fwd_lag[date_cited_app].dt.year) * 12 + \
                        (fwd_lag[date_citing_pub].dt.month - fwd_lag[date_cited_app].dt.month)
    fwd_lag_agg = fwd_lag.groupby('appln_id_cited')[lag_type].agg(['median', 'mean']).reset_index()
    
    return fwd_lag_agg
