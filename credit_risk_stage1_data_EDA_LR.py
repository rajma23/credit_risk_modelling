from pathlib import Path
import warnings, sqlite3, pickle
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 140, "font.size": 11})
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "lending_club_loan_two.csv"
OUT = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"
OUT.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

print("=== DATA LOADING ===", flush=True)
df_raw = pd.read_csv(DATA_PATH)
print(f"Loaded: {df_raw.shape[0]:,} rows x {df_raw.shape[1]} cols", flush=True)
conn = sqlite3.connect(":memory:")
df_raw.to_sql("loans", conn, if_exists="replace", index=False)
sql_result = pd.read_sql("""SELECT loan_status, COUNT(*) AS total_loans,
    ROUND(100.0*COUNT(*)/SUM(COUNT(*)) OVER(),2) AS pct
    FROM loans GROUP BY loan_status""", conn)
print(sql_result.to_string(index=False), flush=True)
dr = sql_result.loc[sql_result.loan_status=="Charged Off","pct"].values[0]
print(f"Default rate: {dr}%", flush=True)

print("\n=== EDA ===", flush=True)
df = df_raw.copy()
df["default"] = (df["loan_status"] == "Charged Off").astype(int)
df["Status"]  = df["default"].map({0:"Fully Paid", 1:"Charged Off"})

fig = plt.figure(figsize=(18, 15))
fig.suptitle("Lending Club - Exploratory Data Analysis", fontsize=16, fontweight="bold", y=1.01)
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])
cnts = df["Status"].value_counts()
bars = ax1.bar(cnts.index, cnts.values, color=["#2ecc71","#e74c3c"], width=0.5, edgecolor="white")
for b, v in zip(bars, cnts.values):
    ax1.text(b.get_x()+b.get_width()/2, b.get_height()+2500, f"{v:,}\n({v/len(df)*100:.1f}%)", ha="center", fontsize=10)
ax1.set_title("Target Distribution", fontweight="bold"); ax1.set_ylabel("Count")
ax1.set_ylim(0, max(cnts.values)*1.15)

ax2 = fig.add_subplot(gs[0, 1])
sns.boxplot(data=df, x="Status", y="loan_amnt", ax=ax2,
    palette={"Fully Paid":"#2ecc71","Charged Off":"#e74c3c"}, showfliers=False)
ax2.set_title("Loan Amount vs Default", fontweight="bold"); ax2.set_xlabel("")

ax3 = fig.add_subplot(gs[1, 0])
sns.boxplot(data=df, x="Status", y="int_rate", ax=ax3,
    palette={"Fully Paid":"#2ecc71","Charged Off":"#e74c3c"}, showfliers=False)
ax3.set_title("Interest Rate vs Default", fontweight="bold"); ax3.set_xlabel("")

ax4 = fig.add_subplot(gs[1, 1])
sns.boxplot(data=df, x="Status", y="annual_inc", ax=ax4,
    palette={"Fully Paid":"#2ecc71","Charged Off":"#e74c3c"}, showfliers=False)
ax4.set_yscale("log"); ax4.set_title("Annual Income vs Default (Log Scale)", fontweight="bold"); ax4.set_xlabel("")

ax5 = fig.add_subplot(gs[2, :])
key_cols = ["loan_amnt","int_rate","installment","annual_inc","dti",
            "open_acc","revol_bal","revol_util","total_acc","mort_acc","default"]
corr = df[key_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", ax=ax5, cmap="RdYlGn",
    center=0, linewidths=0.5, annot_kws={"size":8})
ax5.set_title("Correlation Heatmap - Key Numerical Features", fontweight="bold")

plt.savefig(OUT / "01_eda.png", bbox_inches="tight"); plt.close()
print("EDA saved -> 01_eda.png", flush=True)

print("\n=== FEATURE ENGINEERING ===", flush=True)
df_eng = df_raw.copy()
df_eng["term"] = df_eng["term"].str.strip().str.extract(r"(\d+)").astype(float)

def parse_emp(s):
    if pd.isna(s): return np.nan
    s=str(s)
    if "10+" in s: return 10.0
    if "< 1" in s: return 0.0
    d="".join(filter(str.isdigit,s)); return float(d) if d else np.nan

df_eng["emp_length"] = df_eng["emp_length"].apply(parse_emp)
df_eng["issue_d"]    = pd.to_datetime(df_eng["issue_d"],         format="%b-%Y", errors="coerce")
df_eng["earliest_cr_line"] = pd.to_datetime(df_eng["earliest_cr_line"], format="%b-%Y", errors="coerce")
df_eng["credit_history_years"] = (df_eng["issue_d"] - df_eng["earliest_cr_line"]).dt.days / 365.25

EPS=1e-6
df_eng["income_to_loan_ratio"]  = df_eng["annual_inc"]/(df_eng["loan_amnt"]+EPS)
df_eng["loan_to_income"]        = df_eng["loan_amnt"]/(df_eng["annual_inc"]+EPS)
df_eng["installment_to_income"] = df_eng["installment"]/(df_eng["annual_inc"]+EPS)
df_eng["debt_per_account"]      = df_eng["revol_bal"]/(df_eng["open_acc"]+EPS)
df_eng["high_risk_interaction"] = df_eng["int_rate"]*df_eng["loan_to_income"]
df_eng["log_annual_inc"]  = np.log1p(df_eng["annual_inc"])
df_eng["log_revol_bal"]   = np.log1p(df_eng["revol_bal"])
df_eng["zip_prefix"] = df_eng["address"].str.extract(r"\b(\d{5})\b")[0].str[:2].fillna("00")
print("Feature engineering complete", flush=True)

print("\n=== PIPELINE + SPLIT ===", flush=True)
NUM_FEATURES=["loan_amnt","int_rate","term","dti","open_acc","pub_rec","revol_util",
    "total_acc","mort_acc","pub_rec_bankruptcies","emp_length","income_to_loan_ratio",
    "loan_to_income","installment_to_income","credit_history_years","debt_per_account",
    "high_risk_interaction","log_annual_inc","log_revol_bal"]
CAT_FEATURES=["grade","home_ownership","verification_status","purpose",
    "application_type","initial_list_status","zip_prefix"]

y = (df_eng["loan_status"]=="Charged Off").astype(int)
X = df_eng[NUM_FEATURES+CAT_FEATURES].copy()

num_pipe = Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler())])
cat_pipe = Pipeline([("imp",SimpleImputer(strategy="most_frequent")),
                     ("ohe",OneHotEncoder(handle_unknown="ignore",sparse_output=False))])
preprocessor = ColumnTransformer([("num",num_pipe,NUM_FEATURES),("cat",cat_pipe,CAT_FEATURES)],remainder="drop")

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
print(f"Train: {X_train.shape[0]:,} (default {y_train.mean()*100:.1f}%)", flush=True)
print(f"Test:  {X_test.shape[0]:,}  (default {y_test.mean()*100:.1f}%)", flush=True)

print("\n=== LOGISTIC REGRESSION ===", flush=True)
lr_pipeline = Pipeline([
    ("prep", preprocessor),
    ("model", LogisticRegression(class_weight="balanced",max_iter=500,solver="lbfgs",random_state=42,n_jobs=-1)),
])
lr_pipeline.fit(X_train, y_train)
lr_probs = lr_pipeline.predict_proba(X_test)[:,1]
lr_auc   = roc_auc_score(y_test, lr_probs)
print(f"LR ROC-AUC: {lr_auc:.4f}", flush=True)

with open(MODEL_DIR / "stage1_state.pkl", "wb") as f:
    pickle.dump({
        "X_train":X_train,"X_test":X_test,"y_train":y_train,"y_test":y_test,
        "NUM_FEATURES":NUM_FEATURES,"CAT_FEATURES":CAT_FEATURES,
        "lr_pipeline":lr_pipeline,"lr_probs":lr_probs,"lr_auc":lr_auc,
        "preprocessor":preprocessor
    },f)
print("Stage 1 state saved.", flush=True)

