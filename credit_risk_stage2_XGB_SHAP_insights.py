from pathlib import Path
import sys, warnings, pickle
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
import shap

sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi":140,"font.size":11})
BASE_DIR = Path(__file__).resolve().parent
OUT = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"
OUT.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

with open(MODEL_DIR / "stage1_state.pkl", "rb") as f:
    s = pickle.load(f)
X_train       = s["X_train"]
X_test        = s["X_test"]
y_train       = s["y_train"]
y_test        = s["y_test"]
NUM_FEATURES  = s["NUM_FEATURES"]
CAT_FEATURES  = s["CAT_FEATURES"]
lr_pipeline   = s["lr_pipeline"]
lr_probs      = s["lr_probs"]
lr_auc        = s["lr_auc"]
preprocessor  = s["preprocessor"]
print(f"Stage 1 state loaded. LR AUC={lr_auc:.4f}", flush=True)

print("\n=== XGBOOST TRAINING ===", flush=True)
neg_n = int((y_train==0).sum()); pos_n = int((y_train==1).sum())
spw   = neg_n/pos_n
print(f"scale_pos_weight = {spw:.2f}  ({neg_n:,} paid / {pos_n:,} defaults)", flush=True)

xgb_pipeline = Pipeline([
    ("prep", preprocessor),
    ("model", XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        scale_pos_weight=spw, subsample=0.8, colsample_bytree=0.8,
        eval_metric="auc", random_state=42, n_jobs=-1,
    )),
])
xgb_pipeline.fit(X_train, y_train)
xgb_probs = xgb_pipeline.predict_proba(X_test)[:,1]
xgb_auc   = roc_auc_score(y_test, xgb_probs)
print(f"XGBoost ROC-AUC: {xgb_auc:.4f}", flush=True)

print("\n=== EVALUATION ===", flush=True)

def evaluate(name, probs, y_t, thr=0.5):
    preds = (probs>=thr).astype(int)
    auc   = roc_auc_score(y_t, probs)
    cm    = confusion_matrix(y_t, preds)
    print("", flush=True)
    print(f"{name} | Threshold={thr} | AUC={auc:.4f}", flush=True)
    print(classification_report(y_t, preds, target_names=["Fully Paid","Charged Off"]), flush=True)
    tn,fp,fn,tp = cm.ravel()
    print(pd.DataFrame([[tn,fp],[fn,tp]],
        index=["Actual Paid","Actual Default"],
        columns=["Pred Paid","Pred Default"]).to_string(), flush=True)
    return auc

evaluate("Logistic Regression", lr_probs,  y_test)
evaluate("XGBoost", xgb_probs, y_test)

print("\n=== THRESHOLD ANALYSIS ===", flush=True)
thresholds = np.arange(0.20, 0.55, 0.05)
thr_results = []
print(f"{'Threshold':>10}{'TP':>8}{'FP':>8}{'FN':>8}{'TN':>8}{'Recall':>9}{'Precision':>11}", flush=True)
print("-"*65, flush=True)
for t in thresholds:
    p = (xgb_probs>=t).astype(int)
    tn,fp,fn,tp = confusion_matrix(y_test,p).ravel()
    rec=tp/(tp+fn+1e-9); prec=tp/(tp+fp+1e-9)
    thr_results.append(dict(t=t,tp=tp,fp=fp,fn=fn,tn=tn,recall=rec,prec=prec))
    print(f"{t:>10.2f}{tp:>8,}{fp:>8,}{fn:>8,}{tn:>8,}{rec:>9.3f}{prec:>11.3f}", flush=True)

candidates = [r for r in thr_results if r["recall"]>=0.75 and r["prec"]>=0.28]
best_t = candidates[0]["t"] if candidates else 0.30
print(f"\nSelected threshold: {best_t:.2f}", flush=True)
best_fn = [r["fn"] for r in thr_results if abs(r["t"]-best_t)<0.001][0]
print(f"FN at this threshold: {best_fn:,}", flush=True)

evaluate("XGBoost - Optimised Threshold", xgb_probs, y_test, thr=best_t)

print("\n=== MODEL COMPARISON ===", flush=True)
print(f"LR AUC: {lr_auc:.4f} | XGBoost AUC: {xgb_auc:.4f} | Lift: +{(xgb_auc-lr_auc)*100:.2f} pts", flush=True)
print(f"Selected model: XGBoost, threshold={best_t:.2f}", flush=True)

print("\n=== ROC CURVES ===", flush=True)
fig, ax = plt.subplots(figsize=(9,6))
for name, probs, auc, color in [
    ("Logistic Regression", lr_probs,  lr_auc,  "#3498db"),
    ("XGBoost",             xgb_probs, xgb_auc, "#e74c3c"),
]:
    fpr,tpr,_ = roc_curve(y_test, probs)
    ax.plot(fpr, tpr, lw=2.5, color=color, label=f"{name}  (AUC = {auc:.4f})")

ax.plot([0,1],[0,1],"k--",lw=1.2, label="Random Classifier (AUC = 0.50)")
ax.fill_between(*roc_curve(y_test,xgb_probs)[:2], alpha=0.07, color="#e74c3c")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
ax.set_title("ROC Curves - Logistic Regression vs XGBoost", fontsize=13, fontweight="bold")
ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "02_roc_curves.png", bbox_inches="tight"); plt.close()
print("ROC curves saved -> 02_roc_curves.png", flush=True)

print("\n=== SHAP EXPLAINABILITY ===", flush=True)
fitted_prep = xgb_pipeline.named_steps["prep"]
fitted_xgb  = xgb_pipeline.named_steps["model"]

ohe_names = (fitted_prep.named_transformers_["cat"]
             .named_steps["ohe"].get_feature_names_out(CAT_FEATURES).tolist())
all_feat_names = NUM_FEATURES + ohe_names

rng = np.random.default_rng(42)
idx = rng.choice(X_test.shape[0], size=2000, replace=False)
X_shap = fitted_prep.transform(X_test.iloc[idx])

print("Computing SHAP values on 2,000 samples...", flush=True)
explainer   = shap.TreeExplainer(fitted_xgb)
shap_values = explainer.shap_values(X_shap)
print("SHAP values computed.", flush=True)

shap.summary_plot(shap_values, X_shap, feature_names=all_feat_names,
                  max_display=20, show=False, plot_type="dot")
plt.title("SHAP Feature Importance - XGBoost (Top 20 Features)",
          fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "03_shap_summary.png", bbox_inches="tight"); plt.close()
print("SHAP summary saved -> 03_shap_summary.png", flush=True)

mean_shap = pd.DataFrame({
    "feature":  all_feat_names,
    "shap_mean": np.abs(shap_values).mean(axis=0),
}).sort_values("shap_mean", ascending=False)

top20 = mean_shap.head(20)
bar_colors = ["#f39c12" if f.startswith("zip") else "#e74c3c" for f in top20["feature"]]
fig, ax = plt.subplots(figsize=(10,7))
ax.barh(top20["feature"][::-1], top20["shap_mean"][::-1], color=bar_colors[::-1])
ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
ax.set_title("Feature Importance - XGBoost (SHAP)\nOrange = geographic feature",
             fontsize=12, fontweight="bold")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "04_shap_bar.png", bbox_inches="tight"); plt.close()
print("SHAP bar chart saved -> 04_shap_bar.png", flush=True)

zip_pct = (mean_shap[mean_shap.feature.str.startswith("zip_prefix")]["shap_mean"].sum()
           / mean_shap["shap_mean"].sum() * 100)
print(f"\nzip_prefix SHAP importance: {zip_pct:.1f}%", flush=True)
print("\nTop 15 features:", flush=True)
print(mean_shap.head(15).to_string(index=False), flush=True)

print("\n=== THRESHOLD SENSITIVITY PLOT ===", flush=True)
thr_df = pd.DataFrame(thr_results)
fig, axes = plt.subplots(1,2,figsize=(14,5))

axes[0].plot(thr_df.t, thr_df.recall, "o-", color="#e74c3c", lw=2.5, label="Recall (Default)")
axes[0].plot(thr_df.t, thr_df.prec,   "s-", color="#3498db", lw=2.5, label="Precision (Default)")
axes[0].axvline(best_t, ls="--", color="gray", lw=1.5, label=f"Selected={best_t:.2f}")
axes[0].set_xlabel("Threshold"); axes[0].set_ylabel("Score")
axes[0].set_title("Recall & Precision vs Threshold\n(XGBoost - Default class)", fontweight="bold")
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(thr_df.t, thr_df.fn, "o-", color="#e74c3c", lw=2.5, label="False Negatives")
axes[1].plot(thr_df.t, thr_df.fp, "s-", color="#f39c12", lw=2.5, label="False Positives")
axes[1].axvline(best_t, ls="--", color="gray", lw=1.5, label=f"Selected={best_t:.2f}")
axes[1].set_xlabel("Threshold"); axes[1].set_ylabel("Count")
axes[1].set_title("False Negatives vs False Positives", fontweight="bold")
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.suptitle("Threshold Optimisation", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUT / "05_threshold_analysis.png", bbox_inches="tight"); plt.close()
print("Threshold analysis saved -> 05_threshold_analysis.png", flush=True)
print("Pipeline complete.", flush=True)
