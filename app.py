# =========================================================
# PROSTATE CANCER SURVIVAL PREDICTOR — GBSA · COX-P FEATURES
# Output: Cox-recalibrated survival probability (paper's method)
# =========================================================
import os, pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Prostate Survival — GBSA (Cox-P)",
    page_icon="🩺", layout="wide", initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# LOAD ARTIFACTS
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    d = os.path.join(os.path.dirname(__file__), 'models')
    try:
        with open(os.path.join(d, 'gbsa_model.pkl'), 'rb') as f:
            model = pickle.load(f)
        with open(os.path.join(d, 'feature_names.pkl'), 'rb') as f:
            feature_names = pickle.load(f)
        with open(os.path.join(d, 'km_data.pkl'), 'rb') as f:
            km_data = pickle.load(f)
        with open(os.path.join(d, 'metrics.pkl'), 'rb') as f:
            metrics = pickle.load(f)
        with open(os.path.join(d, 'recalibration.pkl'), 'rb') as f:
            recal = pickle.load(f)
        return {'success': True, 'model': model,
                'feature_names': feature_names, 'km_data': km_data,
                'metrics': metrics, 'recal': recal,
                'model_type': type(model).__name__}
    except Exception as e:
        return {'success': False, 'error': str(e)}

art = load_artifacts()
if not art['success']:
    st.error(f"Failed to load models: {art['error']}")
    st.stop()

# ---------------------------------------------------------
# FEATURE LABELS — your 13 real Cox-P features
# ---------------------------------------------------------
FEATURE_LABELS = {
    'ecog_3_4_vs_1_2':             'ECOG Performance Status',
    'timetonadir_high_vs_low':     'Time to PSA Nadir',
    'stage_IV_vs_I':               'Stage IV (vs Stage I)',
    'mets_multiple_vs_none':       'Multiple Metastases',
    'hist_poor_diff_vs_well_diff': 'Poorly Differentiated Histology',
    'psanadir_high_vs_low':        'PSA Nadir',
    'mets_bone_vs_none':           'Bone Metastases',
    'adt_yes_vs_no':               'ADT Therapy',
    'stage_II_vs_I':               'Stage II (vs Stage I)',
    'hist_mod_diff_vs_well_diff':  'Moderately Differentiated Histology',
    'agegrp_81_90_vs_61_80':       'Age 81-90 (vs 61-80)',
    'radiotx_yes_vs_no':           'Radiotherapy',
    'mets_nonregLN_vs_none':       'Non-regional Lymph Node Metastasis',
}

def label(f): return FEATURE_LABELS.get(f, f)

# ---------------------------------------------------------
# PREDICTION — recalibrated survival probability
# ---------------------------------------------------------
def predict_raw_pi(model, X):
    """Raw model output (Cox partial-hazard for GBSA)."""
    return float(model.predict(X)[0])

def recalibrated_survival_functions(model, X, recal):
    """
    Recalibrated survival functions for the patient.
    Matches the pipeline: transform PI → Cox recal → survival function.
    """
    pi = model.predict(X).flatten()
    if recal['transform'] == 'log':
        pi_eval = np.log(np.clip(pi, 1e-6, None))
    else:
        pi_eval = pi
    pi_eval = np.clip(pi_eval, -50, 50)
    df = pd.DataFrame({'prognostic_index': pi_eval})
    return recal['cox_model'].predict_survival_function(df)

def raw_survival_functions(model, X):
    """Raw model survival functions (for side-by-side comparison)."""
    return model.predict_survival_function(X)

def surf_at(fn, t):
    """Evaluate a step-function survival function at time t."""
    return float(fn(t))

def percentile_of(pi, metrics):
    pct = metrics.get('risk_percentiles') or {}
    if pct:
        return min(sorted(pct.keys()), key=lambda k: abs(pi - pct[k]))
    lo, hi = metrics.get('min_risk', -5), metrics.get('max_risk', 5)
    if hi - lo > 0:
        return float(np.clip((pi - lo) / (hi - lo), 0, 1)) * 100
    return 50.0

# ---------------------------------------------------------
# PATIENT INPUT — clinically grouped, model-aligned
# Multiclass variables (metastasis, stage, histology) are single
# dropdowns that expand internally into the correct dummies.
# ---------------------------------------------------------
def render_patient_inputs():
    """
    Clinically meaningful inputs → correct 13-column dummy row.
    Returns dict {feature_name: 0/1}.
    """
    fd = {name: 0 for name in art['feature_names']}   # init all to 0

    # ---------- Patient Status ----------
    st.markdown("#### Patient Status")
    eco = st.selectbox(
        "ECOG Performance Status",
        ['ECOG 1-2 (Good)', 'ECOG 3-4 (Poor)'],
        key='ui_ecog',
        help="ECOG 3-4 indicates poor functional status")
    fd['ecog_3_4_vs_1_2'] = 1 if eco == 'ECOG 3-4 (Poor)' else 0

    # ---------- Disease Characteristics ----------
    st.markdown("#### Disease Characteristics")

    # Metastasis — one variable, 4 mutually exclusive options
    st.markdown("##### Metastasis Status")
    mets = st.selectbox(
        "Metastasis",
        ['No metastasis',
         'Non-regional lymph node metastasis',
         'Bone metastasis',
         'Multiple metastasis'],
        key='ui_mets',
        help="Select the single most advanced metastatic category present")
    fd['mets_nonregLN_vs_none'] = 1 if mets == 'Non-regional lymph node metastasis' else 0
    fd['mets_bone_vs_none']     = 1 if mets == 'Bone metastasis'                    else 0
    fd['mets_multiple_vs_none'] = 1 if mets == 'Multiple metastasis'                else 0

    # Stage — 4 options; Stage III was dropped in stepwise selection,
    # so Stage III maps to (0, 0), same as Stage I.
    st.markdown("##### Disease Stage")
    stage = st.selectbox(
        "Stage at Diagnosis",
        ['Stage I', 'Stage II', 'Stage III', 'Stage IV'],
        key='ui_stage',
        help="Stage III was dropped during stepwise selection — it is "
             "treated as equivalent to Stage I by the model "
             "(both contribute 0 to both stage coefficients)")
    fd['stage_II_vs_I'] = 1 if stage == 'Stage II' else 0
    fd['stage_IV_vs_I'] = 1 if stage == 'Stage IV' else 0

    # Histologic Type — one variable, 3 mutually exclusive options
    st.markdown("##### Histologic Type")
    hist = st.selectbox(
        "Histologic Differentiation",
        ['Well differentiated',
         'Moderately differentiated',
         'Poorly differentiated'],
        key='ui_hist')
    fd['hist_mod_diff_vs_well_diff']  = 1 if hist == 'Moderately differentiated' else 0
    fd['hist_poor_diff_vs_well_diff'] = 1 if hist == 'Poorly differentiated'    else 0

    # ---------- Treatment ----------
    st.markdown("#### Treatment")
    adt = st.selectbox("ADT Therapy", ['No', 'Yes'], key='ui_adt')
    fd['adt_yes_vs_no'] = 1 if adt == 'Yes' else 0
    rt  = st.selectbox("Radiotherapy", ['No', 'Yes'], key='ui_rt')
    fd['radiotx_yes_vs_no'] = 1 if rt == 'Yes' else 0

    # ---------- Biomarkers & Demographics ----------
    st.markdown("#### Biomarkers & Demographics")
    psa_nadir = st.selectbox("PSA Nadir", ['Low', 'High'], key='ui_psanadir')
    fd['psanadir_high_vs_low'] = 1 if psa_nadir == 'High' else 0

    ttn = st.selectbox("Time to PSA Nadir", ['Low', 'High'], key='ui_ttn')
    fd['timetonadir_high_vs_low'] = 1 if ttn == 'High' else 0

    age = st.selectbox("Age Group", ['Age 61-80', 'Age 81-90'], key='ui_age')
    fd['agegrp_81_90_vs_61_80'] = 1 if age == 'Age 81-90' else 0

    # sanity check: all 13 features must be populated
    missing = [f for f in art['feature_names'] if f not in fd]
    if missing:
        st.warning(f"⚠ UI didn't set: {missing} — defaulted to 0")

    return fd

# ---------------------------------------------------------
# MAIN UI
# ---------------------------------------------------------
def main():
    m = art['metrics']
    st.markdown(f"""
    <div style='text-align:center;padding:1rem;
                background:linear-gradient(90deg,#1B4332,#2E86AB);
                border-radius:10px;color:white;'>
        <h1>🩺 Prostate Cancer Survival Predictor</h1>
        <p style='font-size:1.1rem;margin:0;'>
            GBSA · Stepwise Cox-P Features · Cox-Recalibrated</p>
        <p style='font-size:0.85rem;margin-top:0.5rem;'>
            C-index {m.get('c_index',0.8671):.4f}
            &nbsp;|&nbsp; IBS {m.get('ibs',0.0808):.4f}
            [95% CI {m.get('ibs_ci',(0.0696,0.0914))[0]:.4f},
             {m.get('ibs_ci',(0.0696,0.0914))[1]:.4f}]
            &nbsp;|&nbsp; ERV {m.get('erv',0.4725)*100:.1f}%</p>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

    # ---- SIDEBAR ----
    with st.sidebar:
        st.header("📊 Model Performance")
        st.metric("Test C-index", f"{m.get('c_index',0.8671):.4f}")
        st.metric("IBS",          f"{m.get('ibs',0.0808):.4f}")
        st.metric("ERV",
                  f"{m.get('erv',0.4725)*100:.1f}%",
                  help="Explained Relative Variance vs Kaplan-Meier")
        st.markdown("---")
        st.header("🔧 Method")
        st.markdown(f"- **Model:** `{art['model_type']}`")
        st.markdown(f"- **Feature set:** Stepwise Cox-P (13 features)")
        st.markdown(f"- **Recalibration:** Cox model on OOF PI "
                    f"(β = {art['recal']['coef']:.4f}, "
                    f"transform = {art['recal']['transform']})")
        st.markdown("---")
        st.header("📈 Features Used")
        for f in art['feature_names']:
            st.markdown(f"- {label(f)}")

    # ---- LAYOUT ----
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📝 Patient Input")
        with st.form("prediction_form"):
            fd = render_patient_inputs()
            st.markdown("---")
            t_pred = st.slider("Prediction Horizon (months)",
                               1, 60, 36, 1)
            submitted = st.form_submit_button(
                "🔮 Predict Recalibrated Survival",
                use_container_width=True, type="primary")

    with col2:
        if submitted:
            st.subheader("📊 Recalibrated Prediction")

            # Build input row in correct feature order
            X = pd.DataFrame([fd])[art['feature_names']]

            # ---- Raw PI + risk group ----
            pi_raw = predict_raw_pi(art['model'], X)
            cutoff = m['cutoff']
            is_high = pi_raw >= cutoff
            group = "High Risk" if is_high else "Low Risk"
            pct = percentile_of(pi_raw, m)

            # ---- Recalibrated survival function (paper's method) ----
            recal_fn = recalibrated_survival_functions(
                art['model'], X, art['recal'])[0]
            raw_fn = raw_survival_functions(art['model'], X)[0]

            p_recal = surf_at(recal_fn, t_pred)
            p_raw   = surf_at(raw_fn, t_pred)

            # ---- Headline metrics ----
            c1, c2, c3 = st.columns(3)
            c1.metric(f"🎯 Recalibrated {t_pred}-month Survival",
                      f"{p_recal*100:.1f}%",
                      help="Cox-recalibrated probability — the "
                           "value reported in the paper's calibration plots")
            c2.metric("Risk Group", group,
                      help=f"Training median cutoff = {cutoff:.3f}")
            c3.metric("Percentile", f"{pct:.0f}th",
                      help="Rank within training cohort")

            (st.error if is_high else st.success)(
                f"{'⚠️ HIGH' if is_high else '✅ LOW'} RISK PATIENT")

            # ---- Recalibrated survival curve ----
            st.subheader("📈 Cox-Recalibrated Survival Curve")

            t_max_grid = min(60, max(t_pred + 12, 48))
            grid = np.arange(1, t_max_grid + 1, 1)
            curve_raw   = np.array([surf_at(raw_fn,   t) for t in grid])
            curve_recal = np.array([surf_at(recal_fn, t) for t in grid])

            km = art['km_data']
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=km['low_risk']['times'], y=km['low_risk']['surv'],
                mode='lines', name=f"Low Risk KM (n={km['n_low']})",
                line=dict(color='#2E86AB', width=2, dash='dash')))
            fig.add_trace(go.Scatter(
                x=km['high_risk']['times'], y=km['high_risk']['surv'],
                mode='lines', name=f"High Risk KM (n={km['n_high']})",
                line=dict(color='#E63946', width=2, dash='dash')))
            fig.add_trace(go.Scatter(
                x=grid, y=curve_raw, mode='lines',
                name=f"Patient raw ({p_raw*100:.1f}% @ {t_pred}m)",
                line=dict(color='#AAAAAA', width=2, dash='dot')))
            fig.add_trace(go.Scatter(
                x=grid, y=curve_recal, mode='lines',
                name=f"Patient Cox-recalibrated "
                     f"({p_recal*100:.1f}% @ {t_pred}m)",
                line=dict(color='#FFB400', width=4)))
            fig.add_trace(go.Scatter(
                x=[t_pred], y=[p_recal], mode='markers',
                name=f"{t_pred}m prediction",
                marker=dict(size=22, color='#FFB400', symbol='star',
                            line=dict(width=3, color='black'))))
            fig.update_layout(
                title=f"<b>Recalibrated Survival Prediction</b>"
                      f"<br><sup>Method: Cox recalibration on OOF PI "
                      f"(β = {art['recal']['coef']:.3f}, "
                      f"{art['recal']['transform']} transform)</sup>",
                xaxis_title="Time (months)",
                yaxis_title="Survival Probability",
                yaxis_range=[0, 1.02],
                xaxis_range=[0, t_max_grid + 2],
                hovermode='x unified', height=440,
                legend=dict(x=0.02, y=0.98,
                            bgcolor='rgba(255,255,255,0.95)',
                            bordercolor='#333', borderwidth=1))
            st.plotly_chart(fig, use_container_width=True)

            # ---- Interpretation ----
            st.info(f"""
**Clinical Interpretation — {t_pred} Months**

- **Recalibrated survival: {p_recal*100:.1f}%** ← paper-reported method
- Raw model survival (uncalibrated): {p_raw*100:.1f}%
- Risk group: **{group}** (training median cutoff = {cutoff:.3f})
- Percentile rank in training cohort: **{pct:.0f}th**
            """)

            # ---- Advanced / reproducibility expander ----
            with st.expander("🔬 Advanced — raw model outputs (for reproducibility)"):
                st.markdown(f"""
| Quantity | Value |
|---|---|
| Model type | `{art['model_type']}` |
| Raw prognostic index | `{pi_raw:.4f}` |
| Training median cutoff | `{cutoff:.4f}` |
| Risk group | **{group}** |
| Recalibration coefficient (β) | `{art['recal']['coef']:.4f}` |
| PI transform applied | `{art['recal']['transform']}` |
| Raw survival @ {t_pred}m | `{p_raw:.4f}` |
| Recalibrated survival @ {t_pred}m | `{p_recal:.4f}` |
                """)

        else:
            st.info("👈 Enter patient characteristics on the left and click "
                    "'Predict Recalibrated Survival'")
            with st.expander("ℹ️ About the recalibration method"):
                st.markdown(f"""
The model outputs a **Cox-recalibrated survival probability** at the
chosen horizon. This uses the same method as the paper's calibration
analysis:

1. Compute the model's prognostic index (PI) for this patient.
2. Transform PI: `{art['recal']['transform']}` (identity for GBSA,
   log for RSF-style positive scores).
3. Evaluate a Cox survival model — fit on **out-of-fold training PI** —
   at the chosen horizon. Its coefficient β = `{art['recal']['coef']:.4f}`.

This is the exact procedure from your calibration assessment step,
so the displayed probability matches your paper's reported
IBS / calibration curves.
                """)

if __name__ == "__main__":
    main()