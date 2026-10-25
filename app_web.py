# -*- coding: utf-8 -*-
"""
نسخه وب برنامه تحلیل مقاومت ویژه خاک (IEEE Std 80)
ساخته شده با Streamlit - سازگار با موبایل و کامپیوتر
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# ============================================================
#  تنظیمات صفحه (باید اولین دستور باشد)
# ============================================================
st.set_page_config(
    page_title="Soil Resistivity Analysis",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# استایل‌های سفارشی برای زیبایی بیشتر
st.markdown("""
<style>
    .main { background-color: #f3f6fc; }
    .stButton>button { background-color: #0078D7; color: white; border-radius: 8px; font-weight: bold; }
    .stButton>button:hover { background-color: #005A9E; color: white; }
    .kpi-card { background-color: white; padding: 20px; border-radius: 10px; border: 1px solid #e0e0e0; text-align: center; }
    .kpi-title { font-size: 14px; color: #666; font-weight: bold; }
    .kpi-value { font-size: 28px; color: #0078D7; font-weight: bold; margin-top: 5px; }
</style>
""", unsafe_allow_html=True)

# ============================================================
#  توابع ریاضی
# ============================================================
def rho_apparent_two_layer(a, rho1, rho2, h, tol=1e-8, max_terms=2000):
    a = np.atleast_1d(np.asarray(a, dtype=float))
    k = (rho2 - rho1) / (rho2 + rho1) if (rho2 + rho1) != 0 else 0.0
    n = np.arange(1, max_terms + 1, dtype=float)[:, None]
    A = a[None, :]
    ratio = (2.0 * n * h) / A
    term1 = 1.0 / np.sqrt(1.0 + ratio**2)
    term2 = 1.0 / np.sqrt(4.0 + ratio**2)
    terms = (k ** n) * (term1 - term2)
    rho_a = np.empty_like(a)
    for j in range(a.size):
        conv_idx = np.where(np.abs(terms[:, j]) < tol)[0]
        idx = conv_idx[0] + 1 if conv_idx.size else max_terms
        s = np.sum(terms[:idx, j])
        rho_a[j] = rho1 * (1.0 + 4.0 * s)
    return rho_a if rho_a.size > 1 else float(rho_a[0])

@st.cache_data
def fit_two_layer(a_obs_tuple, rho_obs_tuple, n_starts=40):
    a_obs = np.array(a_obs_tuple)
    rho_obs = np.array(rho_obs_tuple)
    rng = np.random.default_rng(42)

    def residual(p):
        rho1, rho2, h = p
        return np.log10(rho_apparent_two_layer(a_obs, rho1, rho2, h)) - np.log10(rho_obs)

    best = None
    rho_med = np.median(rho_obs)
    starts = [np.array([rho_med, rho_med, 1.0]), np.array([rho_obs[0], rho_obs[-1], a_obs[len(a_obs)//2]])]
    for _ in range(n_starts - len(starts)):
        starts.append(np.array([10**rng.uniform(0, 5), 10**rng.uniform(0, 5), rng.uniform(0.05, 50.0)]))

    bounds = ([1e-2, 1e-2, 1e-3], [1e6, 1e6, 1e3])
    for s0 in starts:
        s0 = np.clip(s0, bounds[0], bounds[1])
        try:
            res = least_squares(residual, s0, bounds=bounds, method='trf', x_scale='jac', max_nfev=2000)
            if best is None or res.cost < best.cost:
                best = res
        except Exception:
            continue

    rho1, rho2, h = best.x
    pred = rho_apparent_two_layer(a_obs, rho1, rho2, h)
    rel_err = np.abs(pred - rho_obs) / rho_obs * 100.0
    rmse_pct = float(np.sqrt(np.mean(rel_err**2)))
    k = (rho2 - rho1) / (rho2 + rho1)

    return {
        'rho1': float(rho1), 'rho2': float(rho2), 'h': float(h), 
        'k': float(k),
        'rmse_pct': rmse_pct, 
        'max_err_pct': float(np.max(rel_err)),
        'pred': pred
    }

# ============================================================
#  رابط کاربری
# ============================================================
st.title("⚡ Soil Resistivity Analysis — IEEE Std 80")
st.markdown("تحلیل داده‌های ونر و مدل‌سازی دولایه خاک برای طراحی سیستم زمین")

# --- سایدبار (ورودی‌ها) ---
with st.sidebar:
    st.header("📊 Input Data")
    
    input_mode = st.radio("روش ورود داده:", ["دستی (متن)", "آپلود فایل CSV"])
    
    a_vals, rho_vals = None, None
    
    if input_mode == "دستی (متن)":
        a_text = st.text_input("فاصله الکترود a (m) - با کاما جدا کنید:", "1, 2, 3, 4, 6, 8, 10, 15, 20")
        rho_text = st.text_input("مقاومت ویژه ظاهری (Ohm.m):", "120, 95, 80, 70, 58, 52, 48, 44, 42")
        try:
            a_vals = np.array([float(x) for x in a_text.replace('،', ',').split(',') if x.strip()])
            rho_vals = np.array([float(x) for x in rho_text.replace('،', ',').split(',') if x.strip()])
        except:
            st.error("لطفاً اعداد را با کاما جدا کنید.")
    else:
        uploaded_file = st.file_uploader("فایل CSV (ستون ۱: a، ستون ۲: rho)", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            a_vals = df.iloc[:, 0].values.astype(float)
            rho_vals = df.iloc[:, 1].values.astype(float)
            st.success("فایل با موفقیت بارگذاری شد.")
    
    st.markdown("---")
    st.header("⚙️ Settings")
    n_starts = st.slider("تعداد نقاط شروع برازش (Multi-start):", 10, 100, 40)
    
    run_btn = st.button("🚀 اجرای برازش و محاسبات", use_container_width=True)

# --- بدنه اصلی ---
if run_btn and a_vals is not None and rho_vals is not None:
    if len(a_vals) != len(rho_vals) or len(a_vals) < 3:
        st.error("❌ طول بردارها باید یکسان و حداقل ۳ نقطه باشد.")
    else:
        with st.spinner('در حال محاسبات… لطفاً صبر کنید...'):
            res = fit_two_layer(tuple(a_vals), tuple(rho_vals), n_starts=n_starts)
        
        # --- KPI Cards ---
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="kpi-card"><div class="kpi-title">لایه اول (ρ₁)</div><div class="kpi-value">{res["rho1"]:.1f} Ω.m</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="kpi-card"><div class="kpi-title">لایه دوم (ρ₂)</div><div class="kpi-value">{res["rho2"]:.1f} Ω.m</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="kpi-card"><div class="kpi-title">خطا (RMSE)</div><div class="kpi-value">{res["rmse_pct"]:.2f}%</div></div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # --- نمودار و تفسیر ---
        col_plot, col_text = st.columns([2, 1])
        
        with col_plot:
            fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
            a_fine = np.logspace(np.log10(a_vals.min()), np.log10(a_vals.max()), 200)
            rho_fine = rho_apparent_two_layer(a_fine, res['rho1'], res['rho2'], res['h'])
            
            ax.semilogx(a_vals, rho_vals, 'o', ms=8, mfc='red', mec='red', label='Measured Data')
            ax.semilogx(a_fine, rho_fine, '-', lw=2, color='#0078D7', label='Two-Layer Model')
            ax.set_xlabel("Electrode Spacing a (m)")
            ax.set_ylabel("Apparent Resistivity (Ohm.m)")
            ax.set_title("Measured vs. Fitted (Two-Layer Model)", fontweight='bold')
            ax.grid(True, which='both', ls='--', alpha=0.5)
            ax.legend()
            st.pyplot(fig)
            
        with col_text:
            st.subheader("📝 تفسیر مهندسی")
            quality = "عالی" if res['rmse_pct'] < 5 else ("قابل قبول" if res['rmse_pct'] < 10 else "نیاز به بازبینی")
            lay_type = "سخت‌تر" if res['rho2'] > res['rho1'] else "نرم‌تر"
            st.info(f"""
            **پارامترهای استخراج‌شده:**
            * **ρ₁:** {res['rho1']:.1f} Ω·m
            * **ρ₂:** {res['rho2']:.1f} Ω·m
            * **ضخامت لایه اول (h):** {res['h']:.2f} متر
            * **ضریب بازتاب (K):** {res['k']:.3f}
            
            **تحلیل:**
            لایه دوم {lay_type} از لایه اول است.
            کیفیت برازش: **{quality}** (خطای {res['rmse_pct']:.2f}%)
            """)

elif not run_btn:
    st.info("👈 برای شروع، داده‌ها را در پنل سمت چپ وارد کرده و روی دکمه «اجرای برازش» کلیک کنید.")