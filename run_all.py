import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

for script in ["credit_risk_stage1_data_EDA_LR.py", "credit_risk_stage2_XGB_SHAP_insights.py"]:
    print(f"\n=== Running {script} ===", flush=True)
    subprocess.run([sys.executable, str(BASE_DIR / script)], check=True, cwd=BASE_DIR)

print("\nDone. Outputs are in the outputs folder; model state is in the models folder.", flush=True)
