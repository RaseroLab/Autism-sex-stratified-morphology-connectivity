import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join as opj
import pingouin as pg


# File paths
BASE_PATH = "/path/to/project"

MODEL_PATH = os.path.join(BASE_PATH, "demo.csv")
CBCL_EXCEL_PATH = os.path.join(BASE_PATH, "Master_Spreadsheet_5-29-2018.xlsx")
RES_MALE_PATH = os.path.join(BASE_PATH, "res_cohort_male_thrP005.csv")
RES_FEMALE_PATH = os.path.join(BASE_PATH, "res_cohort_female_thrP005.csv")
CONNECTIVITY_FOLDER = os.path.join(BASE_PATH, "CT_MC_Vol_SD_SA")

def load_valid_connectivity_matrices(subject_ids):
    con_mats = []
    valid_ids = []
    for sid in tqdm(subject_ids):
        file_path = os.path.join(CONNECTIVITY_FOLDER, f"{sid}_atlas-aparc_mind.csv")
        if os.path.exists(file_path):
            mat = pd.read_csv(file_path).iloc[:, 1:].to_numpy()
            con_mats.append(mat)
            valid_ids.append(sid)
    return np.array(con_mats), valid_ids

def compute_connectivity_strength(con_mats, sig_edges):
    return [np.sum([mat[i, j] for i, j in sig_edges]) for mat in con_mats]


demo_nbs = pd.read_csv(MODEL_PATH)
demo_nbs = pd.get_dummies(demo_nbs, columns=["site"], drop_first=True, dtype=int)
# Load demographic data

cbcl_headers_1 = pd.read_excel(CBCL_EXCEL_PATH, skiprows=9).columns.to_list()
cbcl_headers_2 = pd.read_excel(CBCL_EXCEL_PATH, skiprows=12).columns.to_list()

headers = [x.split(".")[0] + "-" + y.split(".")[0] for x,y in zip(cbcl_headers_1, cbcl_headers_2)]

demo_dat = pd.read_excel(CBCL_EXCEL_PATH, skiprows=12)
demo_dat.columns = headers
demo_dat = demo_dat.rename(columns={"src_subject_id-Site ID":"Site ID"})
demo_dat["SUB_ID"] = demo_dat["Site ID"].apply(lambda x: "sub-" + str(x))
demo_dat = pd.merge(demo_nbs, demo_dat, on="SUB_ID")

severity_cols = ['ADI-R-Social Total',
                 'ADI-R-Communication Total',
                 'ADI-R-Behavioral Total',
                 'ADOS-Social Affect Total - New Algorithm (Mod4)',
                 'ADOS-Behavioral Total - New Algorithm (Mod4)']#,
                 #'ADOS-ADOS Total - New Algorithm (Mod4)']

cbcl_cols = ['CBCL-Affective Problems',
             'CBCL-Anxiety Problems',
             'CBCL-Attention Deficit/Hyperactivity',
             'CBCL-Oppositional Defiant Problems',
             'CBCL-Conduct Problems',
             'CBCL-Internalizing Problems']

rename_columns={'ADI-R-Social Total': 'ADI-R_social',
            'ADI-R-Communication Total': 'ADI-R_communication',
            'ADI-R-Behavioral Total': 'ADI-R_behavioral',
            'ADOS-Social Affect Total - New Algorithm (Mod4)':'ADOS_social',
            'ADOS-Behavioral Total - New Algorithm (Mod4)':'ADOS_communication',
            #'ADOS-ADOS Total - New Algorithm (Mod4)':'ADOS_total',
            "CBCL-Affective Problems": "CBCL_Affective",
            'CBCL-Anxiety Problems': "CBCL_Anxiety",
            'CBCL-Attention Deficit/Hyperactivity': "CBCL_ADHD",
            'CBCL-Oppositional Defiant Problems':"CBCL_Oppositional",
            'CBCL-Conduct Problems': "CBCL_Conduct",
            'CBCL-Internalizing Problems': "CBCL_Internalizing"
            }

conn_mats, _ = load_valid_connectivity_matrices(demo_dat["SUB_ID"].to_list())

# Compute strengh

covars = ["Age", "ICV", "site_SCR", "site_UCL", "site_YAL"]


corrs_df = []
for case, gender in zip(["male", "female"], ["M", "F"]):
    
    demo_case = demo_dat[(demo_dat.Gender == gender) & (demo_dat.Cohort == "ASD")]
    conn_mats_case,_ =  load_valid_connectivity_matrices(demo_case["SUB_ID"].to_list())
    res_nbs = pd.read_csv(opj(BASE_PATH, f"res_cohort_{case}_thrP005.csv"))
    
    for sign in ["positive", "negative"]:
        if sign == "positive":
            mask = res_nbs.strn > 0
        else:
            mask = res_nbs.strn < 0
            
        sig_edges = list(zip(res_nbs[mask]["3Drow"]-1, 
                             res_nbs[mask]["3Dcol"]-1)) # Here NOOShin had a mistake with the indexes!
        strengh = compute_connectivity_strength(conn_mats_case, sig_edges)
        demo_case[f"{sign}_strength"]  = strengh
        
        beh_data = demo_case.loc[:, [f"{sign}_strength"] + severity_cols + cbcl_cols + covars]
        
        beh_data = beh_data.replace(to_replace=999, value=np.nan)
        
        beh_data = beh_data.rename(columns=rename_columns)
        
       
        for col in rename_columns.values():
            temp_df = pg.partial_corr(data=beh_data, 
                                      x=f"{sign}_strength", 
                                      y=col, 
                                      covar=covars)
            temp_df["var"] = col
            temp_df["sign"] = sign
            temp_df["gender"] = case
            corrs_df.append(temp_df.reset_index(drop=True))
        

corrs_df = pd.concat(corrs_df)
corrs_df = corrs_df.reset_index(drop=True)    
corrs_df_group = corrs_df.groupby("gender")


# Set Seaborn style
sns.set(style="white", context="talk")

for gender in ["male", "female"]:

    data_plot = corrs_df_group.get_group(gender)
    data_plot = data_plot.sort_values(by="var").reset_index(drop=True)
    data_plot['CI_lower'] = data_plot["CI95%"].apply(lambda x:x[0])
    data_plot['CI_upper'] = data_plot["CI95%"].apply(lambda x:x[1])
    
    n_vars = len(data_plot.loc[:, "var"].unique())
    data_plot['y_pos'] = np.concatenate([[y, y+0.5] for y in np.arange(0, 2*n_vars, 2)])
    
    # Color mapping
    palette = {'positive': "red", 'negative': "blue"}
    colors = data_plot['sign'].map(palette)
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Horizontal line at zero
    ax.axvline(0, color='k', linestyle='--', linewidth=2)
    
    # Error bars
    ax.hlines(
        y=data_plot['y_pos'],
        xmin=data_plot['CI_lower'],
        xmax=data_plot['CI_upper'],
        color=colors,
        linewidth=2
    )
    
    # Points
    ax.scatter(
        data_plot['r'],
        data_plot['y_pos'],
        color=colors,
        s=100,   
        zorder=3,
        #edgecolor='black'
    )
    

    # Labels and ticks
    ax.set_yticks(data_plot['y_pos'].iloc[np.arange(0, 2*n_vars, 2)] + 0.25)
    ax.set_yticklabels(data_plot['var'].unique())
    ax.set_xlabel('Partial correlation (r)', size=30)
    
    
    # Custom legend
    #Custom label map
    legend_labels = {
        'negative': 'TDC > ASD',
        'positive': 'ASD > TDC'
    }
    for label in palette:
        ax.scatter([], [], color=palette[label], label=legend_labels[label], s=70)
    ax.legend(title='', loc='best', edgecolor="k")
    
    ax.tick_params(labelsize=20)
    
    sns.despine(offset=10, trim=True, left=True)
    plt.tight_layout()
    plt.savefig(opj(BASE_PATH, "plots", f"forest_behavior_{gender}.png"), dpi=300)
    plt.savefig(opj(BASE_PATH, "plots", f"forest_behavior_{gender}.svg"), dpi=300)
    
