# %% [markdown]
# Converted from jubodeath_v9_puredata_paper.ipynb
# Source notebook retained for provenance. Notebook shell/magic commands are commented out for Python syntax compatibility.

# %% [markdown]
# <a href="https://colab.research.google.com/github/peculab/AI4JUBO/blob/main/jubodeath_v9_puredata_paper.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# %% [markdown]
# #### 訓練資料是 mortality_2020_2023_1014/training_data_1014
# #### 外部驗證資料是 mortality_2024_1014/external_validation_1014
#
# #### 次族群
#
# - <= 85 & > 85
# - ADL 變好 & ADL 變差
# - 男性 & 女性
#
# #### 由於各項量測數值有限制在６個月內的量測值，且有新增體重的變化，因此 ADL 沒有值被排除的人比較多。
# %%
# Notebook magic omitted in .py conversion: !pip install shap plotly xgboost --quiet

# %%
# Notebook magic omitted in .py conversion: !pip uninstall shap -y
# Notebook magic omitted in .py conversion: !pip install shap --no-deps

# %%
# Notebook magic omitted in .py conversion: !pip install ace_tools

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from IPython.display import display
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, mean_absolute_error, r2_score
)

# %%
from google.colab import auth
auth.authenticate_user()

import gspread
from google.auth import default
creds, _ = default()

gc = gspread.authorize(creds)

# %% [markdown]
# 外部資料讀入
# %%
# read data and put it in a dataframe
# 在 google 工作表載入外部資料 gsheets

gsheets = gc.open_by_url('https://docs.google.com/spreadsheets/d/1NFAhP8NUVsxzEq55siFA0yHvnXY5GWqiKGSOKC4y1Qg/edit?usp=sharing')
worksheet = gsheets.worksheet("external_validation_1014")  # 指定分頁名稱

worksheet = worksheet.get_all_records()
external = pd.DataFrame(worksheet)
external = external.apply(lambda col: pd.to_numeric(col.astype(str).str.replace(',', '').str.strip(), errors='coerce'))
external.head()

# %%
external.describe(include='all').T

# %%
ex_missing_info = external.isnull().sum().to_frame(name='Missing Count')
ex_missing_info['Missing Ratio'] = (ex_missing_info['Missing Count'] / len(external)).round(4)
ex_missing_info = ex_missing_info.sort_values(by='Missing Ratio', ascending=True)
ex_missing_info

# %% [markdown]
# 訓練資料讀入
# %%
# read data and put it in a dataframe
# 在 google 工作表載入訓練資料 gsheets

gsheets = gc.open_by_url('https://docs.google.com/spreadsheets/d/1qljyp9lq3QsZ7O2O7FQxm7taEWQi3F3bZgNMcQ7NJeE/edit?usp=sharing')
worksheet = gsheets.worksheet("training_data_1014")  # 指定分頁名稱

worksheet = worksheet.get_all_records()
df = pd.DataFrame(worksheet)
df = df.apply(lambda col: pd.to_numeric(col.astype(str).str.replace(',', '').str.strip(), errors='coerce'))
df.head()

# %%
df.describe(include='all').T

# %%
import pandas as pd
import numpy as np
from scipy import stats

outcome_col = "死亡標記"  # 依實際欄位名調整

# === 分組 ===
group0 = df[df[outcome_col] == 0]
group1 = df[df[outcome_col] == 1]

summary = []

for col in df.columns:
    if col == outcome_col or df[col].isna().all():
        continue

    x = df[col].dropna()

    # --- 數值變項 ---
    if np.issubdtype(df[col].dtype, np.number):
        g0, g1 = group0[col].dropna(), group1[col].dropna()
        # 統計量（中位數與四分位）
        def q(x): return np.percentile(x, [25, 50, 75]) if len(x) > 0 else [np.nan, np.nan, np.nan]
        q0, q1, qall = q(g0), q(g1), q(x)

        # 檢定（Mann–Whitney U）
        try:
            p = stats.mannwhitneyu(g0, g1, alternative="two-sided").pvalue
        except ValueError:
            p = np.nan

        summary.append({
            "Variable": col,
            "Total": f"{np.median(x):.2f} ({qall[0]:.2f}, {qall[2]:.2f})",
            "Survival": f"{np.median(g0):.2f} ({q0[0]:.2f}, {q0[2]:.2f})",
            "Death": f"{np.median(g1):.2f} ({q1[0]:.2f}, {q1[2]:.2f})",
            "P": f"{p:.4f}" if not np.isnan(p) else "-"
        })

    # --- 類別變項 ---
    else:
        ct = pd.crosstab(df[col], df[outcome_col])
        for level in ct.index:
            total_n = ct.loc[level].sum()
            g0_n = ct.loc[level].get(0, 0)
            g1_n = ct.loc[level].get(1, 0)
            try:
                chi2, p, _, _ = stats.chi2_contingency(ct)
            except ValueError:
                p = np.nan
            summary.append({
                "Variable": f"{col}={level}",
                "Total": f"{total_n} ({total_n/len(df)*100:.1f}%)",
                "Survival": f"{g0_n} ({g0_n/len(group0)*100:.1f}%)",
                "Death": f"{g1_n} ({g1_n/len(group1)*100:.1f}%)",
                "P": f"{p:.4f}" if not np.isnan(p) else "-"
            })

# === 匯出與顯示 ===
table1 = pd.DataFrame(summary)
table1.to_csv("Table1_summary.csv", index=False, encoding="utf-8-sig")
print(f"✅ Table 1 generated, total variables: {len(table1)}")
table1

# %%
df_missing_info = df.isnull().sum().to_frame(name='Missing Count')
df_missing_info['Missing Ratio'] = (df_missing_info['Missing Count'] / len(df)).round(4)
df_missing_info = df_missing_info.sort_values(by='Missing Ratio', ascending=True)
df_missing_info

# %%
features = df_missing_info[df_missing_info['Missing Ratio']<0.3].index.tolist()

# %%
features

# %%
dfNew = df[features]

# %%
dfNew = dfNew.fillna(0)

# %%
from sklearn.base import BaseEstimator, ClassifierMixin

class HybridXGBRF(BaseEstimator, ClassifierMixin):
    def __init__(self, xgb_model=None, rf_model=None, alpha=0.5):
        self.xgb_model = xgb_model
        self.rf_model = rf_model
        self.alpha = alpha
        self._init_models()

    def _init_models(self):
        # Best Parameters: {'colsample_bytree': 0.8, 'learning_rate': 0.01, 'max_depth': 4, 'n_estimators': 800, 'subsample': 1.0}
        # "XGBClassifier": XGBClassifier(n_estimators=200, learning_rate=0.01, max_depth=5, random_state=42, eval_metric='logloss'),

        self.xgb = self.xgb_model or XGBClassifier(
            eval_metric="logloss",
            random_state=42,
            colsample_bytree=0.8,     # ✅ 降低每棵樹看到的特徵比例 → 提高多樣性
            learning_rate=0.01,       # ✅ 稍微提升學習率搭配更早停止
            max_depth=5,              # ✅ 降低單棵樹複雜度 → 降低過擬合
            n_estimators=200,         # ✅ 總樹數可略減以免累積錯誤
            subsample=1.0,            # ✅ 樣本隨機抽樣 → 提升隨機性
            verbosity=0,
            use_label_encoder=False
        )
        self.rf = self.rf_model or RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

    def fit(self, X, y):
        self._init_models()  # 每次 fit 要重設模型
        self.xgb.fit(X, y)
        self.rf.fit(X, y)
        return self

    def predict_proba(self, X):
        xgb_prob = self.xgb.predict_proba(X)[:, 1]
        rf_prob = self.rf.predict_proba(X)[:, 1]
        blended = self.alpha * xgb_prob + (1 - self.alpha) * rf_prob
        return np.vstack([1 - blended, blended]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

    def get_params(self, deep=True):
        return {
            'xgb_model': self.xgb_model,
            'rf_model': self.rf_model,
            'alpha': self.alpha
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        self._init_models()  # 重新初始化模型
        return self

# %%
# 要移除的欄位，是代表身分標記，以及天數
drop_columns = ['H01_NUM', '觀察天數']

# 丟掉這些欄位
dfNew = dfNew.drop(columns=drop_columns)

# %%
dfNew

# %%
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, classification_report
)
import plotly.graph_objects as go
import plotly.express as px

# %%
# === 資料準備 ===
X = dfNew.drop(columns=['死亡標記'])
y = df['死亡標記']

# %%
X_missing_info = X.isnull().sum().to_frame(name='Missing Count')
X_missing_info['Missing Ratio'] = (X_missing_info['Missing Count'] / len(X)).round(4)
X_missing_info = X_missing_info.sort_values(by='Missing Ratio', ascending=True)
X_missing_info

# %% [markdown]
# # 開始進行訓練
# %%
X.describe(include='all').T

# %%
# Notebook magic omitted in .py conversion: !pip install lifelines

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lifelines import CoxPHFitter
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV

# 假設 HybridXGBRF 已定義
all_models = {
    #"HybridXGBRF (Our Approach)": HybridXGBRF(alpha=1),
    "HybridXGBRF (Our Approach)": XGBClassifier(n_estimators=500, learning_rate=0.01, max_depth=3, random_state=42, eval_metric='logloss', subsample=1.0, verbosity=0),
    "LogisticRegression (max_iter=200)": LogisticRegression(max_iter=200),
    "XGBClassifier": XGBClassifier(n_estimators=200, learning_rate=0.01, max_depth=5, random_state=42, eval_metric='logloss'),
    "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=42),
    "LogisticRegression (max_iter=1000)": LogisticRegression(max_iter=1000),

    # 🔽 新增未測試模型
    "Ridge": make_pipeline(StandardScaler(), LogisticRegression(penalty='l2', solver='saga', max_iter=1000, random_state=42)),
    "Lasso": make_pipeline(StandardScaler(), LogisticRegression(penalty='l1', solver='saga', max_iter=1000, random_state=42)),
    "Elastic": make_pipeline(StandardScaler(), LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000, random_state=42)),
}

# %%
import copy
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix, brier_score_loss
)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
import plotly.graph_objs as go
from sklearn.calibration import calibration_curve

# ===== Helper: 取得機率分數（沒有 predict_proba 時用 decision_function 維持到 [0,1]）=====
def get_positive_proba(estimator, X):
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)[:, 1]
    elif hasattr(estimator, "decision_function"):
        # 將 margin 以 logistic 轉為 [0,1] 近似機率（避免不同尺度造成 brier/roc 差異）
        margin = estimator.decision_function(X)
        proba = 1.0 / (1.0 + np.exp(-margin))
    else:
        # 保底方案：用預測標籤當作機率（會降低 Brier/ROC 解釋性，但不會拋錯）
        proba = estimator.predict(X).astype(float)
    return proba

# ====== Cross-Validation 與 ROC ======
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

fig_roc = go.Figure()
mean_fpr = np.linspace(0, 1, 100)

results = []
trained_models = {}

# --- 準備 OOF 容器（放在 for model_name, model in all_models.items(): 之前）---
n_samples = len(y)
oof_probs_all = {}   # {model_name: np.array shape (n_samples,)}
oof_true = y.values  # 之後各模型共用同一組 y_true（OOF真值）

for model_name, model in all_models.items():
    print(f"▶ Running CV for: {model_name}")

    # OOF 機率緩衝區（每個樣本在它所屬的fold被當成測試集時計算到一次）
    oof_probs = np.zeros(n_samples, dtype=float)

    accs, precs, recalls, f1s, aucs = [], [], [], [], []
    specs, briers = [], []
    tprs = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model_fold = clone(model)

        # —— 缺失值處理 ——
        try:
            model_fold.fit(X_train, y_train)
        except ValueError as e:
            if "Input X contains NaN" in str(e):
                print(f"⚠️ Missing value detected for {model_name} (fold {fold+1}) — applying median imputation.")
                imputer = SimpleImputer(strategy='median')
                X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
                X_test  = pd.DataFrame(imputer.transform(X_test),  columns=X.columns)
                model_fold.fit(X_train, y_train)
            else:
                raise e

        # —— 預測與機率 ——
        y_pred = model_fold.predict(X_test)
        y_prob = get_positive_proba(model_fold, X_test)

        # ★ 回填 OOF 機率（用「位置索引」test_idx 對應到原資料順序）
        oof_probs[test_idx] = y_prob

        # —— ROC（逐 fold）——
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)
        aucs.append(roc_auc)

        # —— 指標（逐 fold）——
        accs.append(accuracy_score(y_test, y_pred))
        precs.append(precision_score(y_test, y_pred, zero_division=0))
        recalls.append(recall_score(y_test, y_pred, zero_division=0))
        f1s.append(f1_score(y_test, y_pred, zero_division=0))

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0,1]).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specs.append(spec)

        briers.append(brier_score_loss(y_test, y_prob))

    # —— 保存最後一個 fold 的已訓練模型（你的原邏輯）——
    trained_models[model_name] = copy.deepcopy(model_fold)

    # —— 保存 OOF 機率（供 Calibration/DCA 使用）——
    oof_probs_all[model_name] = oof_probs

    accs, precs, recalls, f1s, aucs = [], [], [], [], []
    specs, briers = [], []
    tprs = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model_fold = clone(model)

        # —— 缺失值處理：優先原生支援，否則以 median 插補 ——
        try:
            model_fold.fit(X_train, y_train)
        except ValueError as e:
            if "Input X contains NaN" in str(e):
                print(f"⚠️ Missing value detected for {model_name} (fold {fold+1}) — applying median imputation.")
                imputer = SimpleImputer(strategy='median')
                X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
                X_test  = pd.DataFrame(imputer.transform(X_test),  columns=X.columns)
                model_fold.fit(X_train, y_train)
            else:
                raise e

        # —— 預測與機率 ——
        y_pred = model_fold.predict(X_test)
        y_prob = get_positive_proba(model_fold, X_test)

        # —— ROC（逐 fold）——
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)
        aucs.append(roc_auc)

        # —— 指標（逐 fold）——
        accs.append(accuracy_score(y_test, y_pred))
        precs.append(precision_score(y_test, y_pred, zero_division=0))
        recalls.append(recall_score(y_test, y_pred, zero_division=0))
        f1s.append(f1_score(y_test, y_pred, zero_division=0))

        # Specificity：TN / (TN + FP)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0,1]).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specs.append(spec)

        # Brier Score：機率校準指標（越小越好）
        briers.append(brier_score_loss(y_test, y_prob))

    # —— 保留最後一個 fold 訓練好的模型（或可改成 refit 全資料）——
    trained_models[model_name] = copy.deepcopy(model_fold)

    # —— ROC 平均曲線（跨 fold）——
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc_curve = auc(mean_fpr, mean_tpr)   # 用平均曲線再算一次 AUC（展示用）
    fig_roc.add_trace(go.Scatter(
        x=mean_fpr, y=mean_tpr, mode='lines',
        name=f"{model_name} (mean AUC={mean_auc_curve:.3f})"
    ))

    # —— 聚合統計（表格輸出）——
    acc_mean, acc_std = np.mean(accs), np.std(accs, ddof=1)
    prec_mean        = np.mean(precs)
    rec_mean         = np.mean(recalls)
    f1_mean          = np.mean(f1s)
    spec_mean        = np.mean(specs)
    brier_mean       = np.mean(briers)
    auc_mean, auc_std = np.mean(aucs), np.std(aucs, ddof=1)

    # AUROC 95% 信賴區間（以 fold 間常態近似）：mean ± 1.96 * std
    auc_ci_low  = auc_mean - 1.96 * auc_std
    auc_ci_high = auc_mean + 1.96 * auc_std
    auc_ci_text = f"{auc_mean:.3f} ({max(0, auc_ci_low):.3f}-{min(1, auc_ci_high):.3f})"

    results.append({
        'Model': model_name,
        'Accuracy Mean': acc_mean,
        'Accuracy Std': acc_std,
        'Precision Mean': prec_mean,
        'Recall Mean': rec_mean,          # = Sensitivity
        'Sensitivity Mean': rec_mean,     # 額外以 Table 2 用語呈現
        'Specificity Mean': spec_mean,    # ★ 新增
        'F1 Score Mean': f1_mean,
        'ROC AUC Mean': auc_mean,
        'ROC AUC Std': auc_std,
        'AUROC (95%CL)': auc_ci_text,     # ★ 新增（字串含 CI）
        'Brier Score Mean': brier_mean    # ★ 新增
    })

# —— Random Baseline 線 ——
fig_roc.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1], mode='lines',
    line=dict(dash='dash'), name='Random Baseline'
))

fig_roc.update_layout(
    title="ROC Curve Comparison (Cross-Validation Mean)",
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
    width=800, height=600
)
fig_roc.show()

# 最終表格
df_results = pd.DataFrame(results).sort_values(by="ROC AUC Mean", ascending=False)
df_results

# %%
import numpy as np
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

# ===== 你要呈現的模型順序（名字要跟 df_results / oof_probs_all 的 key 一樣）=====
model_order = [
    ("HybridXGBRF (Our Approach)", "A"),
    ("XGBClassifier", "B"),
    ("RandomForestClassifier", "C"),
    ("LogisticRegression (max_iter=1000)", "D"),
    ("Ridge", "E"),
    ("Elastic", "F"),
    ("Lasso", "G"),
    ("LogisticRegression (max_iter=200)", "H"),
]

# ---- 先為每個模型算好 cm、百分比、accuracy ----
cm_infos = []
for model_name, panel_letter in model_order:
    probs = np.asarray(oof_probs_all[model_name])
    y_true = np.asarray(oof_true)
    y_pred = (probs >= 0.5).astype(int)

    # 1) 標準版 confusion matrix：rows=[true0,true1], cols=[pred0,pred1]
    cm_full = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN, FP = cm_full[0]
    FN, TP = cm_full[1]

    # 2) 正確的 Accuracy（不能用換過順序的 cm）
    acc = (TN + TP) / cm_full.sum() if cm_full.sum() > 0 else 0.0

    # 3) 為畫圖把 row 順序改成 [true1,true0]，對應 Actual 1 在上、Actual 0 在下
    #    cm = [[FN, TP],   這一列是真實為 1
    #          [TN, FP]]   這一列是真實為 0
    cm = cm_full[[1, 0], :]

    cm_counts = cm.astype(float)
    row_sums = cm_counts.sum(axis=1, keepdims=True)
    cm_pct = np.divide(
        cm_counts,
        row_sums,
        out=np.zeros_like(cm_counts),
        where=row_sums != 0
    ) * 100.0

    text = [
        [f"{cm_pct[i, j]:.1f}%<br>({int(cm[i, j])})" for j in range(2)]
        for i in range(2)
    ]

    cm_infos.append(
        dict(
            name=model_name,
            letter=panel_letter,
            cm_pct=cm_pct,
            text=text,
            acc=acc,
        )
    )

# ===== 建立 Plotly subplots =====
n_models = len(cm_infos)
n_cols = 2
n_rows = math.ceil(n_models / n_cols)

subplot_titles = [
    f"{info['letter']}. {info['name']}<br>Accuracy = {info['acc']*100:.1f}%"
    for info in cm_infos
]

fig = make_subplots(
    rows=n_rows,
    cols=n_cols,
    subplot_titles=subplot_titles,
    horizontal_spacing=0.12,
    vertical_spacing=0.12,
)

# 共用 coloraxis（百分比 0~100）
for idx, info in enumerate(cm_infos):
    row = idx // n_cols + 1
    col = idx % n_cols + 1

    fig.add_trace(
        go.Heatmap(
            z=info["cm_pct"],
            text=info["text"],
            texttemplate="%{text}",
            x=["Predicted 0", "Predicted 1"],
            y=["Actual 1", "Actual 0"],   # 1 在上、0 在下
            coloraxis="coloraxis",        # 共用顏色軸
            hovertemplate=(
                "True = %{y}<br>"
                "Pred = %{x}<br>"
                "Percent = %{z:.1f}%<extra></extra>"
            ),
        ),
        row=row,
        col=col,
    )

# 設定共用的顏色條
fig.update_layout(
    coloraxis=dict(
        colorscale="Blues",
        cmin=0,
        cmax=100,
        colorbar=dict(title="Percentage (%)"),
    ),
    title=dict(
        text="Confusion matrix comparisons between machine learning classifiers (Training CV)",
        x=0.5,
        xanchor="center"
    ),
    plot_bgcolor="white",
    paper_bgcolor="white",
    width=900,
    height=350 * n_rows,
)

# 每個 subplot 的軸標籤
for i in range(n_models):
    r = i // n_cols + 1
    c = i % n_cols + 1
    fig.update_xaxes(
        title_text="Predicted label",
        row=r,
        col=c,
        tickmode="array",
        tickvals=[0, 1],
        ticktext=["Predicted 0", "Predicted 1"],
    )
    fig.update_yaxes(
        title_text="True label",
        row=r,
        col=c,
        tickmode="array",
        tickvals=[0, 1],
        ticktext=["Actual 1", "Actual 0"],
    )

fig.show()

# 若要存成高解析度圖片（論文用），確保已安裝 kaleido: pip install -U kaleido
# fig.write_image("Figure_C04_confusion_matrices_training_plotly.png", scale=3)

# %%
import plotly.graph_objs as go
import numpy as np

def plot_calibration_curves(oof_true, oof_probs_all, n_bins=10, title="Calibration Curve (OOF)"):
    fig = go.Figure()

    # 參考虛線：完美校準 y=x
    fig.add_trace(go.Scatter(
        x=[0,1], y=[0,1], mode='lines',
        line=dict(dash='dash'),
        name='Perfect calibration'
    ))

    ece_records = []  # 收集各模型的 ECE

    for model_name, probs in oof_probs_all.items():
        # 分箱校準
        frac_pos, mean_pred = calibration_curve(oof_true, probs, n_bins=n_bins, strategy='quantile')
        fig.add_trace(go.Scatter(
            x=mean_pred, y=frac_pos, mode='lines+markers',
            name=f"{model_name}"
        ))

        # ECE（以相同分箱權重計算）
        # 權重用各箱樣本數 / 總樣本數
        bin_ids = np.digitize(probs, np.quantile(probs, np.linspace(0, 1, n_bins+1)[1:-1]), right=True)
        weights = np.bincount(bin_ids, minlength=n_bins) / len(probs)
        # 對齊箱的 mean_pred / frac_pos
        ece = np.sum(weights[:len(mean_pred)] * np.abs(frac_pos - mean_pred))
        ece_records.append((model_name, ece))

    fig.update_layout(
        title=title,
        xaxis_title="Mean predicted probability",
        yaxis_title="Fraction of positives",
        width=800, height=600
    )
    fig.show()

    # 另外附上 Brier（你已在 CV 計過）與 ECE 的小表建議
    ece_df = pd.DataFrame(ece_records, columns=["Model", "ECE"])
    return ece_df.sort_values("ECE")

# %%
ece_table = plot_calibration_curves(oof_true, oof_probs_all, n_bins=10)
ece_table

# %%
def decision_curve(oof_true, oof_probs_all, thresholds=None, title="Decision Curve Analysis (OOF)"):
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)  # 避免 0/1 發散
    N = len(oof_true)
    prevalence = np.mean(oof_true)

    # 基準線
    treat_none = np.zeros_like(thresholds, dtype=float)
    treat_all  = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=thresholds, y=treat_none, mode='lines', name='Treat None', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=thresholds, y=treat_all,  mode='lines', name='Treat All',  line=dict(dash='dot')))

    # 各模型 NB
    for model_name, probs in oof_probs_all.items():
        nbs = []
        for pt in thresholds:
            preds = (probs >= pt).astype(int)
            TP = np.sum((preds == 1) & (oof_true == 1))
            FP = np.sum((preds == 1) & (oof_true == 0))
            nb = (TP / N) - (FP / N) * (pt / (1 - pt))
            nbs.append(nb)
        fig.add_trace(go.Scatter(x=thresholds, y=nbs, mode='lines', name=model_name))

    fig.update_layout(
        title=title,
        xaxis_title="Threshold probability (pt)",
        yaxis_title="Net benefit",
        width=800, height=600
    )
    fig.show()

# %%
decision_curve(oof_true, oof_probs_all, thresholds=np.linspace(0.05, 0.95, 19))

# %%
import numpy as np
import shap

xgb_model = all_models["HybridXGBRF (Our Approach)"]
xgb_model.fit(X, y)

# 抽樣背景（避免 KernelExplainer 太慢）
rng = np.random.default_rng(42)
idx_bg = rng.choice(len(X), size=min(100, len(X)), replace=False)
X_bg = X.iloc[idx_bg] if hasattr(X, "iloc") else np.array(X)[idx_bg]

# 若是二元分類，使用 predict_proba 的第1欄（正類機率）
def f_prob(X_in):
    import numpy as np
    try:
        p = xgb_model.predict_proba(X_in)
        # 二元分類: 取正類
        return p[:, 1]
    except Exception:
        # 回歸或沒有 predict_proba：退回 predict
        return xgb_model.predict(X_in)

# KernelExplainer（或 shap.Explainer(..., algorithm="permutation") 也可）
explainer = shap.KernelExplainer(f_prob, X_bg)
# 可先只解釋一小段以測速，例如前 100 筆
X_eval = X.iloc[:min(1000, len(X))] if hasattr(X, "iloc") else np.array(X)[:min(1000, len(X))]
shap_values = explainer.shap_values(X_eval)

# 重要性
shap_abs_mean = np.abs(shap_values).mean(axis=0)
feat_names = X.columns if hasattr(X, "columns") else [f"f{i}" for i in range(shap_abs_mean.shape[0])]
top20 = sorted(zip(feat_names, shap_abs_mean), key=lambda x: x[1], reverse=True)[:20]
print("[Top-20 (Kernel) SHAP |abs| mean]")
for name, val in top20:
    print(f"{name}: {val:.6f}")

# %%
import pandas as pd
import numpy as np
import plotly.express as px

# --- 0) 安全檢查（可留可刪）---
assert len(shap_abs_mean) == X.shape[1], "shap_abs_mean 長度需與特徵數一致"

# --- 1) 做出完整排名（尚未翻譯前）---
importance_df = (
    pd.DataFrame({
        "Feature": X.columns,
        "Mean |SHAP Value|": np.asarray(shap_abs_mean, dtype=float)
    })
    .sort_values(by="Mean |SHAP Value|", ascending=False)
    .reset_index(drop=True)
)

# 印出所有特徵的完整排名（畫圖前）
print("\n=== Full SHAP Ranking (Original Feature Names) ===")
print(importance_df.to_string(index=True))

# %%
# --- 2) 你手動填翻譯對照表（中文→英文）---
#    例：把左邊的中文欄位名換成你的實際欄位名，右邊填你要顯示的英文
feature_name_map = {
    "六個月內住院次數": "Hospitalizations within 6 Months",
    "ADL_last_score": "ADL Last Score",
    "BW_diff_seq": "Body Weight Change (Sequential)",
    "ADL_std": "ADL Standard Deviation",
    "ADL_Min": "ADL Minimum",
    "性別_is_male": "Male",
    "BW_last": "Body Weight (Last)",
    "ADL_總分_max": "ADL Total Max",
    "BW_first": "Body Weight (First)",
    "DNR_flag": "DNR Flag",
    "ADL_first_score": "ADL First Score",
    "意識總分_diff": "Consciousness Score Difference",
    "預估年齡": "Estimated Age",
    "last_ 意識總分": "Consciousness Score (Last)",
    "使用呼吸輔具": "Use of Respiratory Aid",
    "ADL_diff_seq": "ADL Change (Sequential)",
    "diff_has_feeding_tube": "Feeding Tube Change",
    "ADL_last_CouldNot": "ADL Last - Could Not Perform",
    "ADL_明顯惡化": "ADL Significant Deterioration",
    "ADL_Max": "ADL Maximum",
    "ADL_first_CouldNot": "ADL First - Could Not Perform",
    "diff_has_denture": "Denture Change",
    "last_has_denture": "Has Denture (Last)",
    "had_fall": "Had Fall",
    "last_has_feeding_tube": "Has Feeding Tube (Last)",
    "first_has_denture": "Has Denture (First)",
    "意識總分Max": "Consciousness Score Max",
    "first_has_feeding_tube": "Has Feeding Tube (First)",
    "first_ 意識總分": "Consciousness Score (First)"
}

# --- 3) 產生「英文欄位名」欄位（沒有翻到的就保留原名）---
importance_df["Feature_EN"] = importance_df["Feature"].map(
    lambda f: feature_name_map.get(f, f)
)

# --- 4) 畫圖（用英文欄位名），預設取前 20 名 ---
top_n = 20
fig_bar = px.bar(
    importance_df.head(top_n),
    x="Mean |SHAP Value|",
    y="Feature_EN",
    orientation="h",
    title="Top SHAP Features by Mean |SHAP|",
)
fig_bar.update_layout(
    yaxis=dict(categoryorder="total ascending"),
    xaxis_title="Mean |SHAP Value|",
    yaxis_title="Feature",
)
fig_bar.show()

# %%
# =========================
# A–D Panels from existing results
# =========================
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.base import clone
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------- 共用：英文欄名映射 ----------
try:
    feature_name_map  # 如果你已定義就用你的
except NameError:
    feature_name_map = {}

# ---- 安全 AUC scorer（避免 needs_proba 參數問題）----
def roc_auc_proba_scorer(estimator, X_val, y_val):
    if hasattr(estimator, "predict_proba"):
        y_score = estimator.predict_proba(X_val)
        y_score = y_score[:, 1] if y_score.ndim == 2 else y_score
    elif hasattr(estimator, "decision_function"):
        y_score = estimator.decision_function(X_val)
    else:
        y_score = estimator.predict(X_val)
    return roc_auc_score(y_val, y_score)

scorer = roc_auc_proba_scorer

# %%
importance_df = importance_df.copy()
importance_df["Feature_EN"] = importance_df["Feature"].map(lambda f: feature_name_map.get(f, f))
feat_order      = importance_df["Feature"].tolist()           # 重要度高→低（原名）
feat_order_en   = importance_df["Feature_EN"].tolist()        # 對應英文名
feat2en         = dict(zip(importance_df["Feature"], importance_df["Feature_EN"]))


# ---------- A. 累積特徵路徑（CV AUC vs #features） ----------
def cumulative_feature_cv_curve(X, y, base_model, ordered_feats, max_k=20, cv=5, random_state=42, n_jobs=-1):
    ks, means, stds = [], [], []
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    max_k = min(max_k, len(ordered_feats))
    for k in range(1, max_k + 1):
        cols = ordered_feats[:k]
        m = clone(base_model)
        scores = cross_val_score(m, X[cols], y, cv=splitter, scoring=scorer, n_jobs=n_jobs)
        ks.append(k)
        means.append(scores.mean())
        stds.append(scores.std(ddof=1))
    return pd.DataFrame({"k": ks, "auc_mean": means, "auc_std": stds})

df_path = cumulative_feature_cv_curve(X, y, xgb_model, feat_order, max_k=25, cv=5)

figA = go.Figure()
figA.add_trace(go.Scatter(x=df_path["k"], y=df_path["auc_mean"], mode="lines+markers", name="CV AUC"))
figA.add_trace(go.Scatter(x=df_path["k"], y=df_path["auc_mean"]+df_path["auc_std"],
                          mode="lines", name="+1 SD", line=dict(dash="dot")))
figA.add_trace(go.Scatter(x=df_path["k"], y=df_path["auc_mean"]-df_path["auc_std"],
                          mode="lines", name="-1 SD", line=dict(dash="dot")))
k_star = int(df_path.loc[df_path["auc_mean"].idxmax(), "k"])
figA.add_vline(x=k_star, line_dash="dash", annotation_text=f"k*={k_star}", annotation_position="top right")
figA.update_layout(title="A. Cumulative Top-K Features Path (by SHAP)",
                   xaxis_title="# of Top Features", yaxis_title="CV AUC")

# %%
# ---------- B. 單參數交叉驗證曲線（以 max_depth 為例） ----------
def single_param_cv_curve(X, y, base_model, param_name, grid, cv=5, random_state=42, n_jobs=-1):
    means, stds = [], []
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    for v in grid:
        m = clone(base_model)
        # Pipeline 需用 set_params；純模型也 OK
        try:
            m.set_params(**{param_name: v})
        except Exception:
            setattr(m, param_name, v)
        scores = cross_val_score(m, X, y, cv=splitter, scoring=scorer, n_jobs=n_jobs)
        means.append(scores.mean())
        stds.append(scores.std(ddof=1))
    return pd.DataFrame({param_name: grid, "auc_mean": means, "auc_std": stds})

param_name = "max_depth"             # 你也可以改成 'n_estimators' / 'learning_rate' ...
grid_vals  = [2,3,4,5,6,7,8]
df_cv = single_param_cv_curve(X, y, xgb_model, param_name, grid_vals, cv=5)

# x 軸用 log10 視覺上更像你的示意圖；若參數可能為 0/小數，可以直接用線性 x 軸
xvals = np.log10(df_cv[param_name].astype(float))
figB = go.Figure()
figB.add_trace(go.Scatter(x=xvals, y=df_cv["auc_mean"], mode="lines+markers", name="CV AUC"))
figB.add_trace(go.Scatter(x=xvals, y=df_cv["auc_mean"]+df_cv["auc_std"], mode="lines",
                          name="+1 SD", line=dict(dash="dot")))
figB.add_trace(go.Scatter(x=xvals, y=df_cv["auc_mean"]-df_cv["auc_std"], mode="lines",
                          name="-1 SD", line=dict(dash="dot")))
best_idx = df_cv["auc_mean"].idxmax()
figB.add_vline(x=float(xvals.iloc[best_idx]), line_dash="dash",
               annotation_text=f"best {param_name}={df_cv.loc[best_idx, param_name]}",
               annotation_position="top right")
figB.update_layout(title=f"B. Cross-Validation Curve (log10 {param_name})",
                   xaxis_title=f"log10({param_name})", yaxis_title="CV AUC")

# %%
# ---------- C. SHAP 森林圖（Mean |SHAP| ± 95% CI，bootstrap） ----------
def shap_forest_bootstrap(shap_values, feature_names_en, B=500, random_state=42):
    rng = np.random.default_rng(random_state)
    n, p = shap_values.shape
    mean_abs = np.abs(shap_values).mean(axis=0)
    boot = np.empty((B, p))
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        boot[b, :] = np.abs(shap_values[idx]).mean(axis=0)
    ci_low  = np.percentile(boot, 2.5, axis=0)
    ci_high = np.percentile(boot, 97.5, axis=0)
    df = pd.DataFrame({
        "Feature_EN": feature_names_en,
        "MeanAbsSHAP": mean_abs,
        "CI_low": ci_low,
        "CI_high": ci_high
    })
    # 這裡先不排序，讓我們後面用 importance_df 的順序來決定
    return df

# === 1) 和 Beeswarm 統一的 top_k & 特徵順序（用 importance_df） ===
top_k = 8   # 🟡 請和 Beeswarm 那格的 top_k 設一樣
top_feats_zh = importance_df.head(top_k)["Feature"].tolist()              # 中文原欄名
top_feats_en = [feature_name_map.get(f, f) for f in top_feats_zh]         # 對應英文名（Beeswarm 也是用這個）

# === 2) 計算整體 Mean |SHAP| + 95% CI ===
forest_df = shap_forest_bootstrap(
    np.asarray(shap_values),
    [feat2en[f] for f in X.columns],
    B=400
)

# === 3) 用 top_feats_en 來選出 & 排序森林圖的特徵 ===
forest_top = (
    forest_df.set_index("Feature_EN")
             .loc[top_feats_en]     # 只取 top_k，並依照 importance_df.head(top_k) 的順序
             .reset_index()
)

# === 4) 繪圖 ===
figC = go.Figure()
figC.add_trace(go.Scatter(
    x=forest_top["MeanAbsSHAP"], y=forest_top["Feature_EN"],
    mode="markers", name="Mean |SHAP|",
    error_x=dict(
        type="data", symmetric=False,
        array=forest_top["CI_high"] - forest_top["MeanAbsSHAP"],
        arrayminus=forest_top["MeanAbsSHAP"] - forest_top["CI_low"]
    )
))
figC.update_layout(
    title="C. SHAP Forest Plot with 95% CI",
    xaxis_title="Mean |SHAP|",
    yaxis_title="Feature"
)
figC.show()

# %%
# ---------- D. 連續變項 Spearman 相關熱圖 ----------
# 從 SHAP 前 12 名挑「數值型」欄位
topk = 12
top_feats = [f for f in feat_order[:topk] if np.issubdtype(X[f].dtype, np.number)]
corr = X[top_feats].corr(method="spearman")
corr.index   = [feat2en[f] for f in top_feats]
corr.columns = [feat2en[f] for f in top_feats]
figD = px.imshow(corr, text_auto=True, aspect="auto",
                 title="D. Spearman Correlation Heatmap (Continuous Features)")
figD

# %%
# ---------- 可選：合併 2×2 子圖 ----------
grid = make_subplots(rows=2, cols=2, subplot_titles=("A", "B", "C", "D"))
for tr in figA.data: grid.add_trace(tr, row=1, col=1)
for tr in figB.data: grid.add_trace(tr, row=1, col=2)
for tr in figC.data: grid.add_trace(tr, row=2, col=1)
for tr in figD.data: grid.add_trace(tr, row=2, col=2)
grid.update_layout(height=950, width=1000, title_text="Model-centric Panels A–D (HybridXGBRF)")
grid.show()

# %%
import numpy as np
import pandas as pd

def diagnose_shap(X, shap_values, shap_abs_mean, top_n=20):
    # 1) 轉成矩陣（支援 shap.Explanation）
    shap_mat = np.asarray(getattr(shap_values, "values", shap_values))
    n_rows_shap, n_cols_shap = shap_mat.shape
    n_rows_X, n_cols_X = X.shape

    print("=== SHAP vs X 尺寸對齊檢查 ===")
    print(f"X shape           : {X.shape}")
    print(f"shap_values shape : {shap_mat.shape}")

    # 2) 基本一致性檢查
    print("\n[1] 列數（樣本數）一致？", n_rows_shap == n_rows_X)
    print("[2] 欄數（特徵數）一致？", n_cols_shap == n_cols_X)

    # 3) 檢查 shap_abs_mean 長度與欄數一致
    try:
        L = len(shap_abs_mean)
    except Exception:
        L = None
    print("[3] shap_abs_mean 長度   :", L, "（應該等於特徵數）")
    print("    是否等於 X.columns 數：", L == n_cols_X)

    # 4) 重新用 shap_mat（依 X.columns 順序）計算一次 mean|SHAP|
    recomputed = np.abs(shap_mat).mean(axis=0)  # shape = (p,)
    # 將你 bar 圖的 importance_df 取出來對比前 top_n 名的排名與值
    imp_sorted = (
        pd.DataFrame({"Feature": X.columns, "MeanAbs_from_bar": shap_abs_mean})
        .sort_values("MeanAbs_from_bar", ascending=False)
        .reset_index(drop=True)
    )
    check_sorted = (
        pd.DataFrame({"Feature": X.columns, "MeanAbs_recomputed": recomputed})
        .sort_values("MeanAbs_recomputed", ascending=False)
        .reset_index(drop=True)
    )

    # 合併前 top_n 名看看是否一致
    merged_top = imp_sorted.head(top_n).merge(
        check_sorted.head(top_n), on="Feature", how="outer", indicator=True
    )
    rank_match_rate = (merged_top["_merge"] == "both").mean()

    print("\n[4] 以 X.columns 順序重算 mean|SHAP| 的前", top_n, "名一致率：", f"{rank_match_rate:.2%}")
    if rank_match_rate < 1.0:
        print("    -> 前幾名特徵排序不完全一致，可能是 shap_abs_mean 與 X.columns 對齊方式不同。")

    # 5) 額外提醒：欄名重覆 or 類別型沒處理
    print("\n[5] 欄名是否唯一？", X.columns.is_unique)
    non_numeric = [c for c in X.columns if not np.issubdtype(X[c].dtype, np.number)]
    if non_numeric:
        print("    注意：下列欄位不是數值型，後續做 beeswarm 時要確認 SHAP 是否對應到編碼後的欄：")
        print("    ", non_numeric[:10], "..." if len(non_numeric) > 10 else "")

    # 6) 若列數不一致，給出可安全對齊的 n
    if n_rows_shap != n_rows_X:
        n = min(n_rows_shap, n_rows_X)
        print(f"\n[6] SHAP rows != X rows，安全對齊方案：使用前 n={n} 筆。")
    else:
        print("\n[6] 列數一致，後續可直接用 X 與 shap_values 生成 beeswarm。")

    return {
        "imp_sorted": imp_sorted,
        "check_sorted": check_sorted,
    }

diag = diagnose_shap(X, shap_values, shap_abs_mean, top_n=20)

# %%
importance_df

# %%
# === SHAP Beeswarm（依數值著色，顯示英文欄位名，順序對齊 Forest Plot）===
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# 1) 轉成 ndarray，處理 shap.Explanation 或 ndarray 兩種情況
shap_mat = np.asarray(getattr(shap_values, "values", shap_values))

# 2) 與 SHAP 對齊的 X（用前 n 筆）
n = shap_mat.shape[0]
X_shap = X.iloc[:n].copy()

# 3) 取前 top_k 名特徵：順序直接沿用「SHAP Plot with 95% CI」的 forest_top
#    ➜ 確保兩張圖左邊的 Feature 名稱完全一致
top_k = len(forest_top)   # 或固定 8 / 10，也可以
top_feats_en = forest_top["Feature_EN"].tolist()
# 如果你有 feature_name_map = {中文:英文}，這裡先建一個反向對照
feature_name_map_rev = {v: k for k, v in feature_name_map.items()}
top_feats_zh = [feature_name_map_rev.get(f, f) for f in top_feats_en]

# 4) 做成 long-form，並加入抖動座標（模擬 beeswarm）
recs = []
for rank, (f_zh, f_en) in enumerate(zip(top_feats_zh, top_feats_en)):
    j = X.columns.get_loc(f_zh)  # 對應的欄位位置（用中文原欄名取）
    recs.append(pd.DataFrame({
        "Feature_ZH": f_zh,
        "Feature_EN": f_en,
        "rank": rank,                                   # y 軸基準（0=最重要）
        "SHAP": shap_mat[:, j],                         # 該特徵的 SHAP 值
        "Value": X_shap[f_zh].values                    # 對應的特徵原始值
    }))

df_long = pd.concat(recs, ignore_index=True)

# y 軸抖動（避免點重疊）
rng = np.random.default_rng(42)
df_long["y_jitter"] = df_long["rank"] + rng.normal(0, 0.08, size=len(df_long))

# 讓中間顏色對應在「特徵值的中位數」
v_min = df_long["Value"].min()
v_max = df_long["Value"].max()
v_mid = np.median(df_long["Value"])

# 5) 繪圖（顏色依特徵值連續著色；低值藍、中間紫、高值紅）
fig_bee = go.Figure()
fig_bee.add_trace(go.Scattergl(
    x=df_long["SHAP"],
    y=df_long["y_jitter"],
    mode="markers",
    marker=dict(
        size=4,  # ⭐ 點點變小一點
        color=df_long["Value"],
        # 自訂顏色帶：0=藍, 0.5=紫, 1=紅
        colorscale=[
            [0.0, "blue"],
            [0.5, "purple"],
            [1.0, "red"]
        ],
        cmin=v_min,
        cmax=v_max,
        cmid=v_mid,       # ⭐ 中間值對應紫色
        showscale=True,
        colorbar=dict(title="Feature value")
    ),
    # hover 顯示中英文 + 數值
    text=df_long["Feature_EN"],
    hovertemplate=(
        "Feature=%{text}<br>"
        "（原名：%{customdata[0]}）<br>"
        "SHAP=%{x:.4f}<br>"
        "Value=%{marker.color:.4f}<extra></extra>"
    ),
    customdata=np.stack([df_long["Feature_ZH"]], axis=1)
))

# 6) y 軸改成英文刻度（從上到下照 forest_top 的重要度）
fig_bee.update_yaxes(
    tickmode="array",
    tickvals=list(range(len(top_feats_en))),
    ticktext=top_feats_en,
    title="",
    showgrid=False        # optional：不要背景橫線
)

fig_bee.update_xaxes(
    title="SHAP value (impact on model output)",
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor="black"
)

# 7) 背景改白色 + 整體 layout
fig_bee.update_layout(
    title="SHAP Beeswarm (Top Features, English Labels)",
    height=380 + 26 * len(top_feats_en),
    plot_bgcolor="white",   # ⭐ 圖裡背景白
    paper_bgcolor="white"   # ⭐ 整張圖背景白
)

fig_bee.show()

# %%
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.base import clone

# ---------- 工具函數 ----------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def ensure_ndarray_shap(shap_values):
    """同時相容 shap.Explanation / ndarray"""
    return np.asarray(getattr(shap_values, "values", shap_values))

def pick_cases_by_pred(proba, hi_thr=0.6, lo_thr=0.4):
    """挑一個高風險樣本與一個低風險樣本；若沒有門檻符合，就取最大/最小"""
    hi_idx = np.where(proba >= hi_thr)[0]
    lo_idx = np.where(proba <= lo_thr)[0]
    idx_pos = int(hi_idx[0]) if len(hi_idx) else int(np.argmax(proba))
    idx_neg = int(lo_idx[0]) if len(lo_idx) else int(np.argmin(proba))
    return idx_pos, idx_neg

def make_forceplot_plotly(
    idx, X_part, shap_mat, base_val, model_proba,
    feature_map, top_m=8, title="(A) positively predicted patient"
):
    """
    以「逐步 logit→prob」方式建立水平方向 decision/force plot：
    - 先取 |SHAP| 排序前 top_m，其餘併為 Others
    - 顏色：紅=提高風險(推高機率)，藍=降低風險
    - x 軸顯示 0~1 機率；標示 base value 與 f(x)
    """
    # 1) 取當筆 SHAP & 特徵值
    phi = shap_mat[idx, :]                 # 該樣本每個特徵的 SHAP（logit 空間）
    row = X_part.iloc[idx]

    # 2) 依 |SHAP| 排序，取前 top_m
    order = np.argsort(np.abs(phi))[::-1]
    top_idx = order[:top_m]
    rest_idx = order[top_m:]
    feats = X_part.columns.values

    items = []
    for j in top_idx:
        f_zh = feats[j]
        f_en = feature_map.get(f_zh, f_zh)
        items.append((f_en, row[f_zh], phi[j]))

    if len(rest_idx) > 0:
        items.append(("Others", np.nan, phi[rest_idx].sum()))

    # 3) 逐步從 base（logit）加總，並換算成機率差（寬度＝每步機率變化）
    steps = []
    logit_now = float(base_val)
    prob_now  = sigmoid(logit_now)
    for name_en, val, dlogit in items:
        logit_next = logit_now + float(dlogit)
        prob_next  = sigmoid(logit_next)
        dprob      = prob_next - prob_now
        steps.append({
            "name": name_en,
            "value": val,
            "dlogit": float(dlogit),
            "start_prob": prob_now,
            "end_prob": prob_next,
            "delta_prob": dprob
        })
        logit_now = logit_next
        prob_now  = prob_next

    final_prob = float(model_proba[idx])

    # 4) 組裝水平 bar/segment（右紅左藍）
    seg_x = []
    seg_w = []
    seg_c = []
    texts = []

    for s in steps:
        # 片段起點與寬度（機率尺度）
        x0 = s["start_prob"]
        w  = s["delta_prob"]
        color = "crimson" if w >= 0 else "steelblue"
        label_v = "" if np.isnan(s["value"]) else f"{s['value']:.2f}"
        seg_x.append(x0)
        seg_w.append(w)
        seg_c.append(color)
        texts.append(f"{s['name']} = {label_v}")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=seg_w,
        y=[""] * len(seg_w),
        base=seg_x,
        orientation="h",
        marker_color=seg_c,
        hovertext=texts,
        hovertemplate=(
            "%{hovertext}<br>Δp=%{x:.3f}<br>"
            "from %{base:.3f} to %{customdata:.3f}<extra></extra>"
        ),
        customdata=[s["end_prob"] for s in steps],
        showlegend=False
    ))

    # 5) 垂直輔助線：base 與 f(x)
    fig.add_vline(x=float(sigmoid(base_val)), line_width=1, line_dash="dot", line_color="gray",
                  annotation_text="base value", annotation_position="top left")
    fig.add_vline(x=final_prob, line_width=2, line_color="black",
                  annotation_text=f"f(x) = {final_prob:.2f}", annotation_position="top right")

    # 6) 上方圖例說明（higher/lower）
    fig.add_annotation(x=0.82, y=1.14, xref="paper", yref="paper",
                       text="<b>higher</b>  ⟶", showarrow=False, font=dict(color="crimson"))
    fig.add_annotation(x=0.18, y=1.14, xref="paper", yref="paper",
                       text="⟵  <b>lower</b>", showarrow=False, font=dict(color="steelblue"))

    fig.update_layout(
        title=title,
        xaxis=dict(range=[0,1], title="Probability", tickformat=".2f"),
        yaxis=dict(showticklabels=False),
        bargap=0.1,
        height=220,
        margin=dict(l=40, r=30, t=60, b=40)
    )
    return fig

# ---------- 準備資料（對齊 SHAP 與 X、取得機率、base） ----------
shap_mat = ensure_ndarray_shap(shap_values)
n = shap_mat.shape[0]
X_part = X.iloc[:n].copy()

# 模型預測機率（陽性類別）
proba = xgb_model.predict_proba(X_part)[:, 1]

# SHAP base value（若是 shap.Explanation 可讀 .base_values；否則近似用「目標比例的 logit」）
base_val = getattr(shap_values, "base_values", None)
if base_val is None or np.ndim(base_val) == 0:
    # 近似：用樣本平均機率取 logit 當 base
    base_val = float(np.log(proba.mean() / (1 - proba.mean())))
else:
    # 若是向量，取該類別/或第一個
    base_val = float(np.ravel(base_val)[0])

# 挑一正一負個案
idx_pos, idx_neg = pick_cases_by_pred(proba, hi_thr=0.6, lo_thr=0.4)

fig_pos = make_forceplot_plotly(
    idx_pos, X_part, shap_mat, base_val, proba,
    feature_map=feature_name_map, top_m=8, title="(A) positively predicted patient"
)
fig_neg = make_forceplot_plotly(
    idx_neg, X_part, shap_mat, base_val, proba,
    feature_map=feature_name_map, top_m=8, title="(B) negatively predicted patient"
)

fig_pos.show()
fig_neg.show()

# %%
# Convert results to DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="ROC AUC Mean", ascending=False).reset_index(drop=True)
results_df

# %% [markdown]
# # 測試外部資料在 XGBoost 模型下的結果
# %%
external.head()

# %%
ex_X = external[features].drop(columns=['死亡標記'])
ex_y = external['死亡標記']

# %%
ex_X = ex_X.fillna(0)

# %%
# 丟掉這些欄位
ex_X = ex_X.drop(columns=drop_columns)

# %%
ex_X.describe().T

# %%
eX_missing_info = ex_X.isnull().sum().to_frame(name='Missing Count')
eX_missing_info['Missing Ratio'] = (eX_missing_info['Missing Count'] / len(ex_X)).round(4)
eX_missing_info = eX_missing_info.sort_values(by='Missing Ratio', ascending=False)
eX_missing_info

# %%
from sklearn.impute import SimpleImputer
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
import plotly.subplots as sp

def evaluate_all_models_visual(models: dict, X_val, y_val):
    mean_fpr = np.linspace(0, 1, 100)
    fig_roc = go.Figure()
    results = []

    # 建立混淆矩陣子圖
    num_models = len(models)
    cols = 3
    rows = int(np.ceil(num_models / cols))

    fig_cm = sp.make_subplots(
        rows=rows, cols=cols,
        subplot_titles=list(models.keys()),
        horizontal_spacing=0.15,
        vertical_spacing=0.15
    )

    for i, (model_name, model) in enumerate(models.items()):
        print(f"🔍 Evaluating {model_name}...")

        # 嘗試使用原始資料
        X_input = X_val.copy()
        y_input = y_val

        # 若模型不支援 NaN，則補值
        try:
            # 嘗試呼叫 predict_proba
            _ = model.predict_proba(X_input)
        except ValueError as e:
            if "Input X contains NaN" in str(e):
                print(f"⚠️  {model_name} 不支援 NaN，自動補值中...")
                imputer = SimpleImputer(strategy="median")
                X_input = pd.DataFrame(imputer.fit_transform(X_input), columns=X_val.columns)
            else:
                raise e

        # 預測
        y_pred = model.predict(X_input)
        y_prob = model.predict_proba(X_input)[:, 1]

        # 指標
        acc = accuracy_score(y_input, y_pred)
        prec = precision_score(y_input, y_pred)
        rec = recall_score(y_input, y_pred)
        f1 = f1_score(y_input, y_pred)
        auc_val = roc_auc_score(y_input, y_prob)

        # ROC
        fpr, tpr, _ = roc_curve(y_input, y_prob)
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0], tpr_interp[-1] = 0.0, 1.0

        fig_roc.add_trace(go.Scatter(
            x=mean_fpr, y=tpr_interp,
            mode='lines',
            name=f"{model_name} (AUC={auc_val:.3f})"
        ))

        # 混淆矩陣
        cm = confusion_matrix(y_input, y_pred)
        row, col = i // cols + 1, i % cols + 1
        fig_cm.add_trace(
            go.Heatmap(
                z=cm,
                x=["Predicted Negative", "Predicted Positive"],
                y=["Actual Negative", "Actual Positive"],
                colorscale='Blues',
                text=cm,
                texttemplate="%{text}",
                showscale=False
            ),
            row=row, col=col
        )

        results.append({
            "Model": model_name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "ROC AUC": auc_val
        })

    # 隨機基準線
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        line=dict(dash='dash'),
        name='Random Baseline'
    ))

    fig_roc.update_layout(
        title="ROC Curve Comparison",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        width=800,
        height=600
    )
    fig_roc.show()

    fig_cm.update_layout(
        title="Confusion Matrices of All Models",
        width=400 * cols,
        height=300 * rows,
        showlegend=False
    )
    fig_cm.show()

    # 指標表格
    df_result = pd.DataFrame(results)

    return df_result

# %%
# 假設已經訓練完模型並存在 trained_models 中
evaluate_all_models_visual(trained_models, ex_X, ex_y)

# %%
# ===== 次族群實驗：年齡、ADL 變化、性別 =====
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def _first_existing_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _make_sex_masks(df):
    """
    回傳 {'男性': mask, '女性': mask}；若無法判斷，回傳空 dict。
    支援：
      1) '性別' 欄位（值可能為 '男'/'女' 或 1/0/2…）
      2) one-hot 欄位 '性別_男' / '性別_女'
    """
    masks = {}
    if '性別_is_male' in df.columns:
        col = df['性別_is_male']
        # 嘗試各種常見標記
        male_mask = col.astype(str).str.contains('男') | (col == 1) | (col.astype(str).str.lower().isin(['m','male']))
        female_mask = col.astype(str).str.contains('女') | (col == 0) | (col.astype(str).str.lower().isin(['f','female']))
        if male_mask.any(): masks['男性'] = male_mask
        if female_mask.any(): masks['女性'] = female_mask
    else:
        male_col = _first_existing_column(df, ['性別_男','男','male','Male','M'])
        female_col = _first_existing_column(df, ['性別_女','女','female','Female','F'])
        if male_col is not None:
            masks['男性'] = df[male_col] == 1
        if female_col is not None:
            masks['女性'] = df[female_col] == 1
    return masks

def _make_adl_change_masks(df):
    """
    建立 ADL 變好/變差遮罩：
    'ADL_明顯惡化'（=0 視為變好，=1 變差）
    """
    masks = {}
    masks['ADL 變好'] = df['ADL_明顯惡化'] == 0
    masks['ADL 變差'] = df['ADL_明顯惡化'] == 1
    return masks

def _compute_metrics(y_true, y_prob, y_pred):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'ROC AUC': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        'Support (n)': int(len(y_true)),
        'Positives (n)': int(y_true.sum())
    }

def evaluate_subgroups(models: dict, X_all: pd.DataFrame, y_all: pd.Series, raw_df_for_masks: pd.DataFrame):
    """
    models: 已訓練好的模型字典 trained_models
    X_all, y_all: 用於評估的特徵與標記（例如 ex_X, ex_y）
    raw_df_for_masks: 與 X_all 對齊、包含「年齡/ADL/性別」原始欄位的 DataFrame（例如 external）
    """
    # 年齡遮罩（需 '預估年齡'）
    subgroup_masks = {}
    if '預估年齡' in raw_df_for_masks.columns:
        subgroup_masks['年齡 > 85'] = raw_df_for_masks['預估年齡'] > 85
        subgroup_masks['年齡 <= 85'] = raw_df_for_masks['預估年齡'] <= 85
    else:
        print("⚠️ 找不到欄位『預估年齡』，跳過年齡分組。")

    # ADL 變化遮罩
    adl_masks = _make_adl_change_masks(raw_df_for_masks)
    if adl_masks:
        subgroup_masks.update(adl_masks)
    else:
        print("⚠️ 找不到可推算 ADL 變化的欄位，跳過 ADL 分組。")

    # 性別遮罩
    sex_masks = _make_sex_masks(raw_df_for_masks)
    if sex_masks:
        subgroup_masks.update(sex_masks)
    else:
        print("⚠️ 找不到可用的性別欄位，跳過性別分組。")

    rows = []
    for subgroup_name, mask in subgroup_masks.items():
        mask = mask.fillna(False).astype(bool)  # 安全轉型
        if mask.sum() == 0:
            print(f"⚠️ 次族群「{subgroup_name}」資料筆數為 0，略過。")
            continue

        X_sub = X_all.loc[mask]
        y_sub = y_all.loc[mask]

        # 若模型不支援 NaN，與你上面一致，統一補值策略（中位數）
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy="median")
        X_sub_imp = pd.DataFrame(imputer.fit_transform(X_sub), columns=X_sub.columns, index=X_sub.index)

        for model_name, model in models.items():
            # 預測
            y_prob = model.predict_proba(X_sub_imp)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)

            metrics = _compute_metrics(y_sub, y_prob, y_pred)
            metrics.update({'Subgroup': subgroup_name, 'Model': model_name})
            rows.append(metrics)

    result_df = pd.DataFrame(rows)
    if not result_df.empty:
        # 排序：先按次族群，再按 ROC AUC 由高到低
        result_df = result_df.sort_values(by=['Subgroup','ROC AUC'], ascending=[True, False]).reset_index(drop=True)
    return result_df

# === 執行：以外部驗證集為例 ===
subgroup_results = evaluate_subgroups(trained_models, ex_X, ex_y, external)
display(subgroup_results)

# %%
# 若只想看主力模型
display(subgroup_results[subgroup_results['Model'] == 'HybridXGBRF (Our Approach)'])

# %%
# ===== 複合次族群：年齡 × 性別 × ADL是否明顯惡化 =====
from itertools import product
import pandas as pd
from sklearn.impute import SimpleImputer

def evaluate_age_sex_composites(
    models: dict,
    X_all: pd.DataFrame,
    y_all: pd.Series,
    raw_df_for_masks: pd.DataFrame,
    age_threshold: int = 85,
    min_support: int = 10   # 次族群最少樣本數，太小就略過避免不穩定
):
    """
    針對「年齡(>threshold / <=threshold) × 性別(男性/女性) × ADL(變好/變差)」的交叉次族群做評估。
    會輸出每個模型在各交叉次族群的 Accuracy / Precision / Recall / F1 / ROC-AUC 等。
    依賴你已定義的: _make_sex_masks, _make_adl_change_masks, _compute_metrics。
    """
    if '預估年齡' not in raw_df_for_masks.columns:
        print("⚠️ 找不到欄位『預估年齡』，無法建立年齡遮罩。")
        return pd.DataFrame()

    # 年齡 bins
    age_masks = {
        f'年齡 > {age_threshold}': (raw_df_for_masks['預估年齡'] > age_threshold),
        f'年齡 <= {age_threshold}': (raw_df_for_masks['預估年齡'] <= age_threshold),
    }

    # 性別 masks（沿用你上面的 _make_sex_masks）
    sex_masks = _make_sex_masks(raw_df_for_masks)
    if not sex_masks:
        print("⚠️ 找不到可用的性別欄位，無法建立性別遮罩。")
        return pd.DataFrame()

    # ADL 變化 masks（沿用你上面的 _make_adl_change_masks）
    if 'ADL_明顯惡化' not in raw_df_for_masks.columns:
        print("⚠️ 找不到欄位『ADL_明顯惡化』，無法建立 ADL 遮罩。")
        return pd.DataFrame()
    adl_masks = _make_adl_change_masks(raw_df_for_masks)  # {'ADL 變好': mask, 'ADL 變差': mask}

    rows = []
    for (age_name, age_mask), (sex_name, sex_mask), (adl_name, adl_mask) in product(
        age_masks.items(), sex_masks.items(), adl_masks.items()
    ):
        combo_name = f"{age_name} & {sex_name} & {adl_name}"
        mask = (
            age_mask.fillna(False).astype(bool)
            & sex_mask.fillna(False).astype(bool)
            & adl_mask.fillna(False).astype(bool)
        )
        n = int(mask.sum())
        if n < min_support:
            print(f"ℹ️ 複合次族群「{combo_name}」樣本數 {n} < min_support={min_support}，略過。")
            continue

        X_sub = X_all.loc[mask]
        y_sub = y_all.loc[mask]

        # 與你現有策略一致：補缺失值（中位數）
        imputer = SimpleImputer(strategy="median")
        X_sub_imp = pd.DataFrame(imputer.fit_transform(X_sub), columns=X_sub.columns, index=X_sub.index)

        for model_name, model in models.items():
            y_prob = model.predict_proba(X_sub_imp)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)

            metrics = _compute_metrics(y_sub, y_prob, y_pred)
            metrics.update({'Subgroup': combo_name, 'Model': model_name})
            rows.append(metrics)

    result_df = pd.DataFrame(rows)
    if not result_df.empty:
        result_df['Prevalence'] = result_df['Positives (n)'] / result_df['Support (n)']
        result_df = result_df.sort_values(by=['Subgroup', 'ROC AUC'], ascending=[True, False]).reset_index(drop=True)
    return result_df

# %%
age_sex_results = evaluate_age_sex_composites(
    trained_models, ex_X, ex_y, external,
    age_threshold=85,
    min_support=10
)
display(age_sex_results)

pivot_auc = age_sex_results.pivot_table(index='Subgroup', columns='Model', values='ROC AUC')
display(pivot_auc)

# %%
# 每個模型在哪個次族群表現最好（同樣以 ROC AUC 為主）
def best_subgroup_per_model(results_df: pd.DataFrame,
                            primary='ROC AUC',
                            tie_breakers=('F1','Recall','Precision','Accuracy','Support (n)')):
    sort_cols = ['Model', primary, *tie_breakers]
    sort_asc  = [True, False, *([False]*len(tie_breakers))]
    df_sorted = results_df.sort_values(by=sort_cols, ascending=sort_asc)
    best_df = df_sorted.groupby('Model', as_index=False).head(1).reset_index(drop=True)
    return best_df

best_subgroup_each_model = best_subgroup_per_model(age_sex_results)
display(best_subgroup_each_model)

# %%
# 直接列出「整體表現最高的 (Subgroup, Model) Top-K」
def top_k_overall(results_df: pd.DataFrame, k=10,
                  primary='ROC AUC',
                  tie_breakers=('F1','Recall','Precision','Accuracy','Support (n)')):
    sort_cols = [primary, *tie_breakers]
    sort_asc  = [False, *([False]*len(tie_breakers))]
    return results_df.sort_values(by=sort_cols, ascending=sort_asc).head(k).reset_index(drop=True)

top10 = top_k_overall(age_sex_results, k=10)
display(top10)

# %%
# 每個次族群下，表現最好的模型（以 ROC AUC 為主，F1/Recall/Precision/Accuracy/Support 作為平手時的次序）
def best_model_per_subgroup(results_df: pd.DataFrame,
                            primary='ROC AUC',
                            tie_breakers=('F1','Recall','Precision','Accuracy','Support (n)')):
    sort_cols = ['Subgroup', primary, *tie_breakers]
    sort_asc  = [True, False, *([False]*len(tie_breakers))]
    df_sorted = results_df.sort_values(by=sort_cols, ascending=sort_asc)
    # 取每個 Subgroup 的第一列（即最佳模型）
    best_df = df_sorted.groupby('Subgroup', as_index=False).head(1).reset_index(drop=True)
    return best_df

best_by_subgroup = best_model_per_subgroup(age_sex_results)
display(best_by_subgroup)

# %%
# =========================
# External Validation — Multi-Model Comparison (Colab last cell)
# Assumes: trained_models(dict[str, estimator]), ex_X(DataFrame), ex_y(Series/array),
#          feature_name_map(dict[zh->en]) already exist.
# =========================

import re, numpy as np, pandas as pd
import shap, plotly.graph_objects as go, plotly.express as px
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    brier_score_loss, confusion_matrix, f1_score, precision_score, recall_score
)
from sklearn.calibration import calibration_curve
from sklearn.utils import check_array
from math import exp

# ---------- Utility: pretty English labels ----------
def to_en(name):
    return feature_name_map.get(name, name)

# ---------- Utility: robust proba ----------
def _sigmoid(z):
    try:
        # 防爆 overflow
        z = np.clip(z, -50, 50)
    except Exception:
        pass
    return 1.0 / (1.0 + np.exp(-z))

def get_proba(model, X):
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        return p[:, 1] if p.ndim == 2 else p
    if hasattr(model, "decision_function"):
        z = model.decision_function(X)
        return _sigmoid(z)
    # 最後退路：有些模型 .predict 就是機率或0/1
    pred = model.predict(X)
    pred = check_array(pred, ensure_2d=False)
    # 假設是0/1，轉成float
    return pred.astype(float)

# ---------- SHAP: Safe TreeExplainer (xgboost base_score fix) ----------
def make_tree_explainer_safe(xgb_sklearn_model, model_output="probability"):
    booster = getattr(xgb_sklearn_model, "get_booster", lambda: None)()
    if booster is None and hasattr(xgb_sklearn_model, "attr"):
        booster = xgb_sklearn_model  # already Booster
    if booster is None:
        raise ValueError("Not a tree booster.")

    bs = booster.attr("base_score")
    if isinstance(bs, str) and bs.startswith("[") and bs.endswith("]"):
        try:
            bs_clean = re.split(r"[,\s]+", bs.strip("[]").strip())[0]
            float(bs_clean)  # validate
            booster.set_attr(base_score=bs_clean)
        except Exception:
            pass

    return shap.TreeExplainer(
        booster,
        model_output=model_output,
        feature_perturbation="interventional"
    )

# ---------- SHAP backend chooser ----------
def compute_shap_on_external(model, X, max_sample=1000, random_state=42):
    # 抽樣以加速
    n = min(max_sample, len(X))
    sample = X.sample(n=n, random_state=random_state)
    # 先試樹模型專用 Explainer
    try:
        explainer = make_tree_explainer_safe(model, model_output="probability")
        shap_vals = explainer.shap_values(sample)
        # 二元分類可能回傳 [class0, class1]
        if isinstance(shap_vals, list) and len(shap_vals) == 2:
            shap_vals = shap_vals[1]
        return explainer, sample, np.asarray(shap_vals)
    except Exception as e:
        # 通用 Explainer（速度較慢；背景點縮小）
        try:
            bg = shap.sample(X, min(100, len(X)), random_state=random_state)
            explainer = shap.Explainer(model, bg, feature_names=X.columns)
            exp = explainer(sample)
            shap_vals = getattr(exp, "values", exp)  # Explanation 或 ndarray
            return explainer, sample, np.asarray(shap_vals)
        except Exception as e2:
            print(f"[WARN] SHAP failed for model {getattr(model,'__class__',type(model)).__name__}: {e2}")
            return None, sample, None

# ---------- Metrics per model on external ----------
def summarize_threshold_metrics(y_true, prob, label, pos_label=1):
    # Youden J 找最佳閾值
    fpr, tpr, thr = roc_curve(y_true, prob, pos_label=pos_label)
    youden = tpr - fpr
    i_star = int(np.argmax(youden))
    thr_star = float(thr[i_star]) if i_star < len(thr) else 0.5

    y_hat_05 = (prob >= 0.5).astype(int)
    y_hat_star = (prob >= thr_star).astype(int)

    def metrics_at(yh):
        cm = confusion_matrix(y_true, yh, labels=[0,1])
        tn, fp, fn, tp = cm.ravel()
        sens = tp/(tp+fn) if (tp+fn)>0 else 0.0
        spec = tn/(tn+fp) if (tn+fp)>0 else 0.0
        ppv  = tp/(tp+fp) if (tp+fp)>0 else 0.0
        npv  = tn/(tn+fn) if (tn+fn)>0 else 0.0
        f1   = f1_score(y_true, yh, zero_division=0)
        return dict(TP=tp, FP=fp, TN=tn, FN=fn, Sensitivity=sens, Specificity=spec, PPV=ppv, NPV=npv, F1=f1)

    row05  = metrics_at(y_hat_05)
    rowstar= metrics_at(y_hat_star)
    return {
        "model": label,
        "thr@star": thr_star,
        **{f"{k}@0.5": v for k,v in row05.items()},
        **{f"{k}@star": v for k,v in rowstar.items()},
    }

# ========== 1) Run external inference for all models ==========
ext_results = {}  # name -> dict
for name, model in trained_models.items():
    # 1) probs
    prob = get_proba(model, ex_X)
    # 2) ROC / PR / Calibration
    fpr, tpr, _ = roc_curve(ex_y, prob)
    roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(ex_y, prob)
    ap = average_precision_score(ex_y, prob)
    # Calibration bins
    frac_pos, mean_pred = calibration_curve(ex_y, prob, n_bins=10, strategy="quantile")
    brier = brier_score_loss(ex_y, prob)

    # 3) SHAP (sampled)
    explainer, sample_df, shap_vals = compute_shap_on_external(model, ex_X, max_sample=1000, random_state=42)

    ext_results[name] = dict(
        prob=prob, fpr=fpr, tpr=tpr, roc_auc=roc_auc,
        prec=prec, rec=rec, ap=ap,
        cal_x=mean_pred, cal_y=frac_pos, brier=brier,
        shap_explainer=explainer, shap_sample=sample_df, shap_values=shap_vals
    )

# ========== 2) Overlay ROC (External) ==========
fig_roc = go.Figure()
for name, res in ext_results.items():
    fig_roc.add_trace(go.Scatter(x=res["fpr"], y=res["tpr"], mode="lines",
                                 name=f"{name} (AUC={res['roc_auc']:.3f})"))
fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Chance", line=dict(dash="dash")))
fig_roc.update_layout(title="External ROC — All Models", xaxis_title="False Positive Rate",
                      yaxis_title="True Positive Rate", height=500)
fig_roc.show()

# ========== 3) Overlay PR (External) ==========
fig_pr = go.Figure()
baseline = float(np.mean(ex_y))
for name, res in ext_results.items():
    fig_pr.add_trace(go.Scatter(x=res["rec"], y=res["prec"], mode="lines",
                                name=f"{name} (AP={res['ap']:.3f})"))
fig_pr.add_trace(go.Scatter(x=[0,1], y=[baseline, baseline], mode="lines",
                            name=f"Baseline={baseline:.3f}", line=dict(dash="dash")))
fig_pr.update_layout(title="External Precision–Recall — All Models",
                     xaxis_title="Recall", yaxis_title="Precision", height=500)
fig_pr.show()

# ========== 4) Overlay Calibration (External) ==========
fig_cal = go.Figure()
for name, res in ext_results.items():
    fig_cal.add_trace(go.Scatter(x=res["cal_x"], y=res["cal_y"], mode="lines+markers",
                                 name=f"{name} (Brier={res['brier']:.3f})"))
fig_cal.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Perfect", line=dict(dash="dash")))
fig_cal.update_layout(title="External Calibration — All Models",
                      xaxis_title="Mean Predicted Probability",
                      yaxis_title="Fraction of Positives", height=500)
fig_cal.show()

# ========== 5) Threshold summary table (0.5 vs Youden* per model) ==========
rows = []
for name, res in ext_results.items():
    rows.append(summarize_threshold_metrics(ex_y, res["prob"], name))
df_thr = pd.DataFrame(rows).sort_values("model")
display(df_thr)

# ========== 6) SHAP mean|abs| bar (External, per model facet) ==========
# 只對成功產生 SHAP 的模型畫圖
bar_records = []
for name, res in ext_results.items():
    shap_vals = res["shap_values"]
    sample_df = res["shap_sample"]
    if shap_vals is None or sample_df is None:
        continue
    # 對齊欄位順序
    shap_vals = np.asarray(shap_vals)
    assert shap_vals.shape[1] == sample_df.shape[1], f"SHAP shape mismatch for {name}"
    mean_abs = np.abs(shap_vals).mean(axis=0)
    for zh, m in zip(sample_df.columns, mean_abs):
        bar_records.append({
            "Model": name,
            "Feature_EN": to_en(zh),
            "MeanAbsSHAP": float(m)
        })

df_bar = pd.DataFrame(bar_records)
if not df_bar.empty:
    # 每個模型取前 N
    topN = 12
    df_bar_top = (df_bar.sort_values(["Model","MeanAbsSHAP"], ascending=[True, False])
                        .groupby("Model", as_index=False, group_keys=False)
                        .apply(lambda d: d.head(topN)))
    fig_imp = px.bar(df_bar_top, x="MeanAbsSHAP", y="Feature_EN",
                     orientation="h", facet_col="Model", facet_col_wrap=2,
                     title="External Mean |SHAP| Top Features (per Model)")
    fig_imp.update_layout(height=400 + 160 * len(df_bar_top["Model"].unique()))
    # 讓每個 facet 由小到大呈現
    fig_imp.update_yaxes(categoryorder="total ascending")
    fig_imp.show()
else:
    print("[INFO] No SHAP bars drawn (no model produced SHAP successfully).")

# ========== 7) (Optional) SHAP Beeswarm for ANY model you pick ==========
def plot_beeswarm_for_model(model_name, top_k=12, colorscale="RdBu", max_sample=1000):
    res = ext_results.get(model_name)
    if res is None or res["shap_values"] is None:
        print(f"[WARN] No SHAP for model '{model_name}'")
        return
    shap_vals = np.asarray(res["shap_values"])
    Xs = res["shap_sample"].copy()
    # 依外部驗證上的 |SHAP| 排序取前 k
    mean_abs = np.abs(shap_vals).mean(axis=0)
    order_idx = np.argsort(-mean_abs)[:top_k]
    feats_zh = Xs.columns[order_idx]
    feats_en = [to_en(f) for f in feats_zh]

    # long-form + jitter
    rng = np.random.default_rng(42)
    recs = []
    for rank, f in enumerate(feats_zh):
        j = Xs.columns.get_loc(f)
        recs.append(pd.DataFrame({
            "Feature_ZH": f,
            "Feature_EN": to_en(f),
            "rank": rank,
            "SHAP": shap_vals[:, j],
            "Value": Xs[f].values
        }))
    df_long = pd.concat(recs, ignore_index=True)
    df_long["y_jitter"] = df_long["rank"] + rng.normal(0, 0.08, size=len(df_long))

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=df_long["SHAP"], y=df_long["y_jitter"],
        mode="markers",
        marker=dict(
            size=6, color=df_long["Value"],
            colorscale=colorscale, reversescale=True, showscale=True,
            colorbar=dict(title="Feature value")
        ),
        text=df_long["Feature_EN"],
        hovertemplate=("Feature=%{text}<br>"
                       "原名：%{customdata[0]}<br>"
                       "SHAP=%{x:.4f}<br>"
                       "Value=%{marker.color:.4f}<extra></extra>"),
        customdata=np.stack([df_long["Feature_ZH"]], axis=1)
    ))
    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(len(feats_en))),
        ticktext=feats_en,
        title=""
    )
    fig.update_xaxes(title="SHAP value (impact on model output)")
    fig.update_layout(
        title=f"External SHAP Beeswarm — {model_name}",
        height=380 + 26 * len(feats_en)
    )
    fig.show()

# %%
# =========================================
# Final plotting cell: 4 figures with square data area
# 1) CV ROC (OOF)
# 2) CV Calibration (OOF)
# 3) External ROC
# 4) External Calibration
# =========================================

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.utils import check_array

# --------- 統一模型順序 & 短名稱（圖例用） ----------
MODEL_ORDER = [
    "HybridXGBRF (Our Approach)",
    "XGBClassifier",
    "RandomForestClassifier",
    "LogisticRegression (max_iter=1000)",
    "Ridge",
    "Elastic",
    "Lasso",
    "LogisticRegression (max_iter=200)",
]

name_map = {
    "HybridXGBRF (Our Approach)": "HybridXGBRF",
    "XGBClassifier": "XGB",
    "RandomForestClassifier": "RF",
    "LogisticRegression (max_iter=1000)": "LR-1000",
    "LogisticRegression (max_iter=200)": "LR-200",
    "Ridge": "Ridge",
    "Elastic": "Elastic",
    "Lasso": "Lasso",
}

def existing_models(source_dict):
    """只留在指定 dict 裡實際存在的模型名稱"""
    return [m for m in MODEL_ORDER if m in source_dict]

# --------- 共用：讓數值區變正方形，右邊留 legend ----------
def style_square(fig, title, x_title, y_title):
    # 數值區正方形
    fig.update_xaxes(
        title_text=x_title,
        range=[0, 1],
        constrain="domain"           # 不被 legend 壓縮
    )
    fig.update_yaxes(
        title_text=y_title,
        range=[0, 1],
        scaleanchor="x",             # y 比例綁定 x
        scaleratio=1                 # 1:1 → 數值區正方形
    )
    # 整張圖長方形，右邊留空間給 legend
    fig.update_layout(
        title=title,
        width=900,
        height=600,
        margin=dict(l=80, r=260, t=60, b=60),
        legend=dict(
            x=1.02,
            y=0.5,
            xanchor="left",
            yanchor="middle",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=10)
        )
    )
    return fig

# --------- 外部機率工具（給 external 用） ----------
def _sigmoid(z):
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))

def get_proba_external(model, X):
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        return p[:, 1] if p.ndim == 2 else p
    if hasattr(model, "decision_function"):
        return _sigmoid(model.decision_function(X))
    pred = model.predict(X)
    pred = check_array(pred, ensure_2d=False)
    return pred.astype(float)

# =========================================
# 1) CV ROC（用 OOF 預測）
# =========================================
fig_cv_roc = go.Figure()
y_true_oof = np.asarray(oof_true)

for name in existing_models(oof_probs_all):
    proba = np.asarray(oof_probs_all[name])
    fpr, tpr, _ = roc_curve(y_true_oof, proba)
    auc_val = auc(fpr, tpr)
    short = name_map.get(name, name)
    fig_cv_roc.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode="lines",
        line=dict(width=2),
        name=f"{short} (AUC={auc_val:.3f})"
    ))

# chance 線
fig_cv_roc.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1],
    mode="lines",
    line=dict(dash="dash", width=1.5),
    name="Chance"
))

fig_cv_roc = style_square(
    fig_cv_roc,
    "CV ROC (OOF predictions)",
    "False Positive Rate (1 − Specificity)",
    "True Positive Rate (Sensitivity)"
)
fig_cv_roc.show()


# =========================================
# 2) CV Calibration（OOF）
# =========================================
fig_cv_cal = go.Figure()
fig_cv_cal.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1],
    mode="lines",
    line=dict(dash="dash"),
    name="Perfect"
))

ece_records = []
for name in existing_models(oof_probs_all):
    proba = np.asarray(oof_probs_all[name])
    y_true = y_true_oof

    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10, strategy="quantile")
    short = name_map.get(name, name)

    fig_cv_cal.add_trace(go.Scatter(
        x=mean_pred, y=frac_pos,
        mode="lines+markers",
        name=short
    ))

    # ECE
    bin_edges = np.quantile(proba, np.linspace(0, 1, 11))
    bin_edges[0], bin_edges[-1] = 0.0, 1.0
    bin_ids = np.digitize(proba, bin_edges[1:-1], right=True)
    weights = np.bincount(bin_ids, minlength=10) / len(proba)
    ece = np.sum(weights[:len(frac_pos)] * np.abs(frac_pos - mean_pred))
    ece_records.append((short, ece))

fig_cv_cal = style_square(
    fig_cv_cal,
    "CV Calibration (OOF predictions)",
    "Mean predicted probability",
    "Fraction of positives"
)
fig_cv_cal.show()

ece_df = pd.DataFrame(ece_records, columns=["Model", "ECE"]).sort_values("ECE")
display(ece_df)


# =========================================
# 3) External ROC（用 trained_models, ex_X, ex_y）
# =========================================
ext_results = {}
y_true_ext = np.asarray(ex_y)

for name in existing_models(trained_models):
    model = trained_models[name]
    prob = get_proba_external(model, ex_X)

    fpr, tpr, _ = roc_curve(y_true_ext, prob)
    roc_auc_val = auc(fpr, tpr)

    # calibration（順便算，給下一張圖用）
    frac_pos, mean_pred = calibration_curve(y_true_ext, prob, n_bins=10, strategy="quantile")
    brier = brier_score_loss(y_true_ext, prob)

    ext_results[name] = dict(
        prob=prob,
        fpr=fpr, tpr=tpr, roc_auc=roc_auc_val,
        cal_x=mean_pred, cal_y=frac_pos, brier=brier
    )

fig_ext_roc = go.Figure()
for name in MODEL_ORDER:
    if name not in ext_results:
        continue
    res = ext_results[name]
    short = name_map.get(name, name)
    fig_ext_roc.add_trace(go.Scatter(
        x=res["fpr"], y=res["tpr"],
        mode="lines",
        line=dict(width=2),
        name=f"{short} (AUC={res['roc_auc']:.3f})"
    ))

fig_ext_roc.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1],
    mode="lines",
    line=dict(dash="dash", width=1.5),
    name="Chance"
))

fig_ext_roc = style_square(
    fig_ext_roc,
    "External ROC — All Models",
    "False Positive Rate (1 − Specificity)",
    "True Positive Rate (Sensitivity)"
)
fig_ext_roc.show()


# =========================================
# 4) External Calibration（和 ROC 同一批 ext_results）
# =========================================
fig_ext_cal = go.Figure()
fig_ext_cal.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1],
    mode="lines",
    line=dict(dash="dash"),
    name="Perfect"
))

for name in MODEL_ORDER:
    if name not in ext_results:
        continue
    res = ext_results[name]
    short = name_map.get(name, name)
    fig_ext_cal.add_trace(go.Scatter(
        x=res["cal_x"], y=res["cal_y"],
        mode="lines+markers",
        name=f"{short} (Brier={res['brier']:.3f})"
    ))

fig_ext_cal = style_square(
    fig_ext_cal,
    "External Calibration — All Models",
    "Mean predicted probability",
    "Fraction of positives"
)
fig_ext_cal.show()

print("✅ 4 figures generated: CV ROC, CV Calibration, External ROC, External Calibration.")

# %%
# =========================
# External validation: add SD + 95% CI (bootstrap)
# Put this cell at the END of your external validation section
# =========================

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

# ---- Helper: compute point metrics ----
def compute_metrics(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_proba),
    }

# ---- Helper: stratified bootstrap for SD + 95% CI ----
def bootstrap_ci_sd(y_true, y_proba, threshold=0.5, n_boot=2000, random_state=42):
    rng = np.random.default_rng(random_state)

    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    # Store bootstrap metrics
    boot = {k: [] for k in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]}

    for _ in range(n_boot):
        # stratified resample: sample positives and negatives separately
        bs_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        bs_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        bs_idx = np.concatenate([bs_pos, bs_neg])

        yt = y_true[bs_idx]
        yp = y_proba[bs_idx]

        m = compute_metrics(yt, yp, threshold=threshold)
        for k, v in m.items():
            boot[k].append(v)

    # Summaries
    out = {}
    for k, arr in boot.items():
        arr = np.asarray(arr, dtype=float)
        out[k] = {
            "SD": float(arr.std(ddof=1)),
            "CI_low": float(np.percentile(arr, 2.5)),
            "CI_high": float(np.percentile(arr, 97.5)),
        }
    return out

# ---- Assumptions (match your notebook variables) ----
# You already have:
#   X_train, y_train  (development set)
#   X_test,  y_test   (external validation set)
# And you have:
#   all_models = {model_name: model_object, ...}
#
# We fit each model on full training set, then evaluate on external set.
# For HybridXGBRF (Version B), your class should implement fit() and predict_proba().

THRESHOLD = 0.5
N_BOOT = 2000
RANDOM_STATE = 42

external_summary = []

for model_name, model in all_models.items():
    # Fit on full dev data
    model.fit(X_train, y_train)

    # Predict proba on external set
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        # Fallback: some models expose decision_function only
        scores = model.decision_function(X_test)
        # Convert to (0,1) with logistic; keeps ranking for AUC
        y_proba = 1 / (1 + np.exp(-scores))

    # Point estimate on full external set
    point = compute_metrics(y_test, y_proba, threshold=THRESHOLD)

    # Bootstrap SD + 95% CI
    ci = bootstrap_ci_sd(
        y_test, y_proba,
        threshold=THRESHOLD,
        n_boot=N_BOOT,
        random_state=RANDOM_STATE
    )

    row = {"Model": model_name}
    for k, v in point.items():
        row[f"{k}"] = v
        row[f"{k} SD"] = ci[k]["SD"]
        row[f"{k} 95% CI Low"] = ci[k]["CI_low"]
        row[f"{k} 95% CI High"] = ci[k]["CI_high"]

    external_summary.append(row)

external_summary_df = pd.DataFrame(external_summary)

# Optional: nicer formatting for a paper table
def fmt_ci(mean, lo, hi, sd, digits=3):
    return f"{mean:.{digits}f} ({lo:.{digits}f}–{hi:.{digits}f}), SD={sd:.{digits}f}"

pretty_cols = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
pretty_rows = []
for _, r in external_summary_df.iterrows():
    pr = {"Model": r["Model"]}
    for c in pretty_cols:
        pr[c] = fmt_ci(r[c], r[f"{c} 95% CI Low"], r[f"{c} 95% CI High"], r[f"{c} SD"])
    pretty_rows.append(pr)

external_summary_pretty = pd.DataFrame(pretty_rows)

print("=== External validation (point estimate + SD + 95% CI) ===")
display(external_summary_df)

print("\n=== External validation (paper-friendly strings) ===")
display(external_summary_pretty)

# %%
import pandas as pd

pd.set_option("display.max_columns", None)   # 不截斷欄位
pd.set_option("display.max_rows", None)      # 不截斷列（你如果列很多就先不要開）
pd.set_option("display.width", None)         # 自動用可用寬度
pd.set_option("display.max_colwidth", None)  # 不截斷儲存格文字

# %%
print("=== External validation (point estimate + SD + 95% CI) ===")
display(external_summary_df)

print("\n=== External validation (paper-friendly strings) ===")
display(external_summary_pretty)

# %%
external_summary_df.to_excel("external_validation_full.xlsx", index=False)
external_summary_pretty.to_excel("external_validation_pretty.xlsx", index=False)
print("Saved: external_validation_full.xlsx, external_validation_pretty.xlsx")

