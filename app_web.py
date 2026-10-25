# -*- coding: utf-8 -*-
"""
نسخه وب پیشرفته برنامه تحلیل مقاومت ویژه خاک و طراحی سیستم زمین
مطابق با استاندارد IEEE Std 80
+ قابلیت شبیه‌سازی تصویری لایه‌های خاک با هوش مصنوعی (Pollinations)
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import requests
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="Soil & Grounding Designer", page_icon="⚡", layout="wide")

# استایل‌های سفارشی
st.markdown("""
<style>
    .kpi-card { background-color: white; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    .kpi-title { font-size: 14px; color: #666; font-weight: bold; margin-bottom: 5px; }
    .kpi-value { font-size: 24px; color: #0078D7; font-weight: bold; }
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
    a_obs, rho_obs = np.array(a_obs_tuple), np.array(rho_obs_tuple)
    rng = np.random.default_rng(42)

    def residual(p):
        return np.log10(rho_apparent_two_layer(a_obs, *p)) - np.log10(rho_obs)

    best = None
    rho_med = np.median(rho_obs)
    starts = [np.array([rho_med, rho_med, 1.0]), np.array([rho_obs[0], rho_obs[-1], a_obs[len(a_obs)//2]])]
    for _ in range(n_starts - len(starts)):
        starts.append(np.array([10**rng.uniform(0, 5), 10**rng.uniform(0, 5), rng.uniform(0.05, 50.0)]))

    bounds = ([1e-2, 1e-2, 1e-3], [1e6, 1e6, 1e3])
    for s0 in starts:
        s0 = np.clip(s0, bounds[0], bounds[1])
        try:
            res = least_squares(residual, s0, bounds=bounds, method='trf', max_nfev=2000)
            if best is None or res.cost < best.cost:
                best = res
        except:
            continue

    rho1, rho2, h = best.x
    pred = rho_apparent_two_layer(a_obs, rho1, rho2, h)
    rel_err = np.abs(pred - rho_obs) / rho_obs * 100.0
    return {
        'rho1': float(rho1), 'rho2': float(rho2), 'h': float(h), 'k': float((rho2 - rho1) / (rho2 + rho1)),
        'rmse_pct': float(np.sqrt(np.mean(rel_err**2))), 'max_err_pct': float(np.max(rel_err)), 'pred': pred
    }

def calculate_grid_resistance(rho_eq, area, length=None):
    rg55 = (rho_eq / 4.0) * np.sqrt(np.pi / area) if area > 0 else 0
    rg56 = rg55 + (rho_eq / length) if length and length > 0 else None
    return rg55, rg56

def calculate_safety_voltages(rho_s, c_s, weight, t_fault):
    i_body = (0.116 if weight == 50 else 0.157) / np.sqrt(t_fault)
    e_step = (1000 + 6 * c_s * rho_s) * i_body
    e_touch = (1000 + 1.5 * c_s * rho_s) * i_body
    return i_body, e_step, e_touch

# ============================================================
#  تابع تولید تصویر با هوش مصنوعی
# ============================================================
def generate_soil_image(rho1, rho2, h):
    # ساخت پرامپت مهندسی برای هوش مصنوعی
    prompt = (
        f"Scientific cross-section diagram of a two-layer soil model for electrical grounding. "
        f"Top layer: brown soil, thickness {h:.1f} meters, resistivity {rho1:.0f} ohm-meters. "
        f"Bottom layer: rocky/clay soil, resistivity {rho2:.0f} ohm-meters. "
        f"Clear separation line between layers, engineering visualization, clean vector style, "
        f"white background, labels showing depths and resistivities."
    )
    
    # آدرس‌دهی به سرویس رایگان Pollinations
    encoded_prompt = requests.utils.quote(prompt)
    url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=800&height=600&nologo=true"
    
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        return None

# ============================================================
#  رابط کاربری اصلی
# ============================================================
st.title("⚡ Soil & Grounding Designer (IEEE Std 80)")
st.markdown("ابزار جامع تحلیل خاک، طراحی شبکه زمین و بررسی ایمنی")

with st.sidebar:
    st.header("📊 داده‌های اندازه‌گیری (Wenner)")
    a_text = st.text_input("فاصله الکترودها a (m):", "1, 2, 3, 4, 6, 8, 10, 15, 20")
    rho_text = st.text_input("مقاومت ظاهری ρa (Ohm.m):", "120, 95, 80, 70, 58, 52, 48, 44, 42")
    
    st.divider()
    st.header("🏭 پارامترهای شبکه زمین")
    area = st.number_input("مساحت زمین (m²):", min_value=1.0, value=1000.0)
    length = st.number_input("طول کل هادی‌ها (m):", min_value=1.0, value=500.0)
    t_fault = st.number_input("زمان قطعی خطا (ثانیه):", min_value=0.1, value=0.5)
    
    run_btn = st.button("🚀 اجرای کامل محاسبات", use_container_width=True)

if run_btn:
    try:
        a_vals = np.array([float(x) for x in a_text.replace('،', ',').split(',') if x.strip()])
        rho_vals = np.array([float(x) for x in rho_text.replace('،', ',').split(',') if x.strip()])
        if len(a_vals) != len(rho_vals) or len(a_vals) < 3:
            raise ValueError("طول داده‌ها باید یکسان و حداقل ۳ باشد.")
    except ValueError as e:
        st.error(f"خطا در ورودی: {e}")
        st.stop()

    with st.spinner("در حال تحلیل مدل خاک و محاسبات شبکه..."):
        res = fit_two_layer(tuple(a_vals), tuple(rho_vals))
        rho_eq = (res['rho1'] + res['rho2']) / 2.0
        rg55, rg56 = calculate_grid_resistance(rho_eq, area, length)
        i_body, e_step, e_touch = calculate_safety_voltages(3000, 0.8, 70, t_fault)

    # --- کارت‌های خلاصه (KPI) ---
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="kpi-card"><div class="kpi-title">لایه اول (ρ₁)</div><div class="kpi-value">{res["rho1"]:.1f}</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="kpi-card"><div class="kpi-title">لایه دوم (ρ₂)</div><div class="kpi-value">{res["rho2"]:.1f}</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="kpi-card"><div class="kpi-title">Rg (فرمول ۵۶)</div><div class="kpi-value">{rg56:.2f}</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="kpi-card"><div class="kpi-title">ولتاژ گام مجاز</div><div class="kpi-value">{e_step:.1f} V</div></div>', unsafe_allow_html=True)

    st.divider()

    # --- تب‌های خروجی (شامل تب جدید AI) ---
    tab1, tab2, tab3, tab4, tab_ai = st.tabs([
        "📈 نمودار اصلی", "📉 خطاها (Residuals)", "📋 جدول نتایج", "📄 گزارش", "🎨 شبیه‌سازی هوش مصنوعی"
    ])

    with tab1:
        fig, ax = plt.subplots(figsize=(10, 5))
        a_fine = np.logspace(np.log10(a_vals.min()), np.log10(a_vals.max()), 200)
        ax.semilogx(a_vals, rho_vals, 'o', color='red', label='داده واقعی')
        ax.semilogx(a_fine, rho_apparent_two_layer(a_fine, res['rho1'], res['rho2'], res['h']), '-', color='#0078D7', lw=2, label='مدل دولایه')
        ax.set_xlabel("فاصله الکترود a (m)")
        ax.set_ylabel("مقاومت ظاهری (Ohm.m)")
        ax.grid(True, which='both', ls='--', alpha=0.5)
        ax.legend()
        st.pyplot(fig)

    with tab2:
        fig2, ax2 = plt.subplots(figsize=(10, 3))
        residuals = (res['pred'] - rho_vals) / rho_vals * 100
        ax2.semilogx(a_vals, residuals, 'o-', color='#e74c3c')
        ax2.axhline(0, color='black', ls='--')
        ax2.set_xlabel("فاصله الکترود a (m)")
        ax2.set_ylabel("خطای نسبی (%)")
        ax2.grid(True, which='both', ls='--', alpha=0.5)
        st.pyplot(fig2)

    with tab3:
        st.subheader("تحلیل حساسیت پارامترها")
        sens_data = {
            "پارامتر": ["ρ₁ (لایه اول)", "ρ₂ (لایه دوم)", "h (ضخامت)"],
            "مقدار بهینه": [f"{res['rho1']:.1f}", f"{res['rho2']:.1f}", f"{res['h']:.2f}"],
            "واحد": ["Ω.m", "Ω.m", "m"]
        }
        st.table(pd.DataFrame(sens_data))
        
        st.subheader("خروجی داده‌ها")
        df_out = pd.DataFrame({"a (m)": a_vals, "Measured": rho_vals, "Fitted": res['pred']})
        st.dataframe(df_out, use_container_width=True)

    with tab4:
        st.subheader("📄 گزارش جامع مهندسی")
        report = f"""
        ### ۱. تحلیل خاک
        * مقاومت لایه اول: {res['rho1']:.1f} Ω.m
        * مقاومت لایه دوم: {res['rho2']:.1f} Ω.m
        * ضخامت لایه اول: {res['h']:.2f} متر
        * ضریب بازتاب (K): {res['k']:.3f}
        * کیفیت برازش (RMSE): {res['rmse_pct']:.2f}%

        ### ۲. طراحی شبکه زمین
        * مقاومت معادل خاک: {rho_eq:.1f} Ω.m
        * مقاومت شبکه (فرمول ۵۵): {rg55:.2f} اهم
        * مقاومت شبکه (فرمول ۵۶): {rg56:.2f} اهم

        ### ۳. تحلیل ایمنی (IEEE 80)
        * جریان مجاز بدن: {i_body:.3f} آمپر
        * ولتاژ گام مجاز: {e_step:.1f} ولت
        * ولتاژ تماس مجاز: {e_touch:.1f} ولت
        """
        st.markdown(report)
        
        st.download_button("📥 دانلود گزارش (CSV)", data=df_out.to_csv(index=False).encode('utf-8'), file_name='soil_analysis.csv')

    # --- تب شبیه‌سازی هوش مصنوعی ---
    with tab_ai:
        st.subheader("🎨 شبیه‌سازی تصویری لایه‌های خاک با هوش مصنوعی")
        st.info("با کلیک روی دکمه زیر، هوش مصنوعی بر اساس نتایج تحلیل (ρ₁، ρ₂، h) یک تصویر مهندسی از لایه‌های زمین تولید می‌کند. (این فرآیند ممکن است ۱۰ تا ۳۰ ثانیه طول بکشد).")
        
        if st.button("✨ تولید تصویر با هوش مصنوعی"):
            with st.spinner("هوش مصنوعی در حال ترسیم مقطع زمین است..."):
                img = generate_soil_image(res['rho1'], res['rho2'], res['h'])
                if img:
                    st.success("تصویر با موفقیت تولید شد!")
                    st.image(img, caption=f"مقطع شبیه‌سازی‌شده: لایه اول ({res['rho1']:.0f} Ω.m، ضخامت {res['h']:.1f}m) روی لایه دوم ({res['rho2']:.0f} Ω.m)", use_container_width=True)
                else:
                    st.error("متأسفانه در تولید تصویر خطایی رخ داد. لطفاً دوباره تلاش کنید.")
