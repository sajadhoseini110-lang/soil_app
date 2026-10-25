# -*- coding: utf-8 -*-
"""
Soil & Grounding Designer — تحلیل مقاومت ویژه خاک (Wenner) و طراحی اولیه شبکه زمین
مرجع: IEEE Std 80
"""
from __future__ import annotations

import re
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import least_squares

# ============================================================
#  ثابت‌ها
# ============================================================
MAX_TERMS = 500
RHO_MIN, RHO_MAX = 0.1, 1e5
H_MIN, H_MAX = 0.05, 200.0

_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩٫٬،؛",
                        "01234567890123456789.,,;")


# ============================================================
#  هسته محاسباتی
# ============================================================
def parse_series(text: str, name: str) -> np.ndarray:
    """تبدیل متن (با ارقام فارسی/انگلیسی و جداکننده‌های مختلف) به آرایه عددی."""
    tokens = [t for t in re.split(r"[,;\s]+", text.translate(_DIGITS)) if t]
    try:
        return np.array([float(t) for t in tokens])
    except ValueError as exc:
        raise ValueError(f"مقدار نامعتبر در «{name}»: {exc}") from None


def validate_data(a: np.ndarray, rho: np.ndarray):
    if a.size != rho.size:
        raise ValueError(f"تعداد فاصله‌ها ({a.size}) با تعداد مقاومت‌ها ({rho.size}) برابر نیست.")
    if a.size < 4:
        raise ValueError("برای برازش مدل دولایه (۳ پارامتر) حداقل ۴ نقطه اندازه‌گیری لازم است.")
    if np.any(a <= 0) or np.any(rho <= 0):
        raise ValueError("تمام فاصله‌ها و مقاومت‌ها باید مثبت باشند.")
    order = np.argsort(a)
    a, rho = a[order], rho[order]
    if np.any(np.diff(a) == 0):
        raise ValueError("فاصله‌های تکراری در داده‌ها وجود دارد.")
    return a, rho


def rho_apparent_two_layer(a, rho1: float, rho2: float, h: float) -> np.ndarray:
    """مقاومت ظاهری Wenner برای خاک دولایه (برداری‌شده)."""
    a = np.atleast_1d(np.asarray(a, dtype=float))
    k = (rho2 - rho1) / (rho2 + rho1)
    n = np.arange(1, MAX_TERMS + 1, dtype=float)[:, None]
    ratio = 2.0 * n * h / a[None, :]
    terms = k ** n * (1.0 / np.sqrt(1.0 + ratio**2) - 1.0 / np.sqrt(4.0 + ratio**2))
    return rho1 * (1.0 + 4.0 * terms.sum(axis=0))


@st.cache_data(show_spinner=False)
def fit_two_layer(a_obs: tuple, rho_obs: tuple, n_starts: int = 30) -> dict:
    a, rho = np.asarray(a_obs), np.asarray(rho_obs)
    log_rho = np.log10(rho)
    rng = np.random.default_rng(42)

    def residual(p):
        return np.log10(rho_apparent_two_layer(a, *p)) - log_rho

    lo = np.array([RHO_MIN, RHO_MIN, H_MIN])
    hi = np.array([RHO_MAX, RHO_MAX, H_MAX])
    med = np.median(rho)
    starts = [np.array([rho[0], rho[-1], np.sqrt(a[0] * a[-1])]),
              np.array([med, med, a.mean()])]
    for _ in range(n_starts):
        starts.append(np.array([
            10 ** rng.uniform(np.log10(rho.min()) - 0.5, np.log10(rho.max()) + 0.5),
            10 ** rng.uniform(np.log10(rho.min()) - 0.5, np.log10(rho.max()) + 0.5),
            10 ** rng.uniform(np.log10(a.min()) - 0.5, np.log10(a.max()) + 0.5),
        ]))

    best = None
    for s0 in starts:
        try:
            r = least_squares(residual, np.clip(s0, lo, hi), bounds=(lo, hi),
                              method="trf", max_nfev=500)
        except (ValueError, FloatingPointError):
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        raise RuntimeError("برازش مدل دولایه همگرا نشد؛ داده‌ها را بررسی کنید.")

    rho1, rho2, h = map(float, best.x)
    pred = rho_apparent_two_layer(a, rho1, rho2, h)
    rel_err = (pred - rho) / rho * 100.0

    # مدل مرجع: خاک یکنواخت (میانگین هندسی)
    rho_u = float(np.exp(np.mean(np.log(rho))))
    rmse_uniform = float(np.sqrt(np.mean(((rho_u - rho) / rho * 100.0) ** 2)))

    return {
        "rho1": rho1, "rho2": rho2, "h": h,
        "k": (rho2 - rho1) / (rho2 + rho1),
        "pred": pred, "rel_err": rel_err,
        "rmse_pct": float(np.sqrt(np.mean(rel_err**2))),
        "max_err_pct": float(np.max(np.abs(rel_err))),
        "rmse_uniform_pct": rmse_uniform, "rho_uniform": rho_u,
        "h_at_bound": bool(h <= H_MIN * 1.05 or h >= H_MAX * 0.95),
        "rho_at_bound": bool(min(rho1, rho2) <= RHO_MIN * 1.05 or max(rho1, rho2) >= RHO_MAX * 0.95),
    }


def equivalent_resistivity(method: str, fit: dict, area: float) -> float:
    if method == "apparent":
        r_eq = np.sqrt(area / np.pi)
        return float(rho_apparent_two_layer(r_eq, fit["rho1"], fit["rho2"], fit["h"])[0])
    if method == "rho1":
        return fit["rho1"]
    return (fit["rho1"] + fit["rho2"]) / 2.0


def grid_resistance(rho: float, area: float, length: float, depth: float) -> dict:
    r = np.sqrt(area / np.pi)
    simple = rho / (4.0 * r)
    with_length = simple + rho / length
    sverak = rho * (1.0 / length + 1.0 / np.sqrt(20.0 * area)
                    * (1.0 + 1.0 / (1.0 + depth * np.sqrt(20.0 / area))))
    return {"simple": simple, "with_length": with_length, "sverak": sverak}


def surface_factor(rho: float, rho_s: float, hs: float) -> float:
    if hs <= 0 or abs(rho_s - rho) < 1e-9:
        return 1.0
    return 1.0 - 0.09 * (1.0 - rho / rho_s) / (2.0 * hs + 0.09)


def tolerable_voltages(rho: float, rho_s: float, hs: float, weight: int, ts: float) -> dict:
    i_body = (0.116 if weight == 50 else 0.157) / np.sqrt(ts)
    cs = surface_factor(rho, rho_s, hs)
    rho_surface = rho_s if hs > 0 else rho
    return {
        "i_body": i_body, "cs": cs,
        "e_step": (1000.0 + 6.0 * cs * rho_surface) * i_body,
        "e_touch": (1000.0 + 1.5 * cs * rho_surface) * i_body,
    }


def build_report(inp: dict, fit: dict, rho_eq: float, rg: dict, rg_design: float,
                 gpr: float, tv: dict) -> str:
    verdict = ("✅ GPR کمتر از ولتاژ تماس مجاز است؛ تحلیل ولتاژهای mesh و step الزامی نیست."
               if gpr <= tv["e_touch"] else
               "⚠️ GPR بیشتر از ولتاژ تماس مجاز است؛ محاسبه Em و Es و اصلاح طرح لازم است.")
    return f"""## گزارش تحلیل خاک و شبکه زمین — {date.today().isoformat()}

### ۱. مدل خاک (دولایه)
- مقاومت ویژه لایه اول ρ₁ = {fit['rho1']:.1f} Ω·m
- مقاومت ویژه لایه دوم ρ₂ = {fit['rho2']:.1f} Ω·m
- ضخامت لایه اول h = {fit['h']:.2f} m
- ضریب بازتاب K = {fit['k']:.3f}
- RMSE برازش = {fit['rmse_pct']:.2f}٪ ، بیشینه خطا = {fit['max_err_pct']:.2f}٪
- RMSE مدل یکنواخت (مرجع) = {fit['rmse_uniform_pct']:.2f}٪

### ۲. شبکه زمین
- مقاومت ویژه معادل ρ_eq = {rho_eq:.1f} Ω·m
- مساحت = {inp['area']:.0f} m² ، طول کل هادی = {inp['length']:.0f} m ، عمق دفن = {inp['depth']:.2f} m
- Rg (ρ/4r) = {rg['simple']:.3f} Ω
- Rg (ρ/4r + ρ/L) = {rg['with_length']:.3f} Ω
- Rg (Sverak — مقدار طراحی) = {rg_design:.3f} Ω
- جریان شبکه Ig = {inp['ig']:.0f} A  ←  GPR = {gpr:.0f} V

### ۳. ایمنی (وزن {inp['weight']} kg ، زمان خطا {inp['ts']} s)
- جریان مجاز بدن = {tv['i_body']:.3f} A
- ضریب Cs = {tv['cs']:.3f}
- ولتاژ گام مجاز = {tv['e_step']:.0f} V
- ولتاژ تماس مجاز = {tv['e_touch']:.0f} V

**نتیجه:** {verdict}
"""


# ----- UI -----
# ============================================================
#  رابط کاربری
# ============================================================
st.set_page_config(page_title="Soil & Grounding Designer", page_icon="⚡", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;700&display=swap');
html, body, .stMarkdown, .stAlert, label, h1, h2, h3 { font-family: 'Vazirmatn', sans-serif; }
.stMarkdown, .stAlert, h1, h2, h3, label { direction: rtl; text-align: right; }
input, textarea, .stDataFrame, .stTable { direction: ltr; text-align: left; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Soil & Grounding Designer")
st.caption("تحلیل خاک دولایه (Wenner)، مقاومت شبکه زمین و بررسی ایمنی بر پایه IEEE Std 80")

with st.sidebar:
    with st.form("inputs"):
        st.subheader("📊 داده‌های Wenner")
        mode = st.radio("نوع داده اندازه‌گیری", ["مقاومت ظاهری ρa (Ω·m)", "مقاومت قرائت‌شده R (Ω)"])
        a_text = st.text_input("فاصله الکترودها a (m)", "1, 2, 3, 4, 6, 8, 10, 15, 20")
        v_text = st.text_input("مقادیر اندازه‌گیری", "120, 95, 80, 70, 58, 52, 48, 44, 42")

        st.subheader("🏭 شبکه زمین")
        area = st.number_input("مساحت (m²)", min_value=1.0, value=1000.0)
        length = st.number_input("طول کل هادی‌ها (m)", min_value=1.0, value=500.0)
        depth = st.number_input("عمق دفن شبکه (m)", min_value=0.0, value=0.5)
        method = st.selectbox(
            "روش ρ معادل",
            ["apparent", "rho1", "mean"],
            format_func={"apparent": "ρa مدل در شعاع معادل شبکه (پیشنهادی)",
                         "rho1": "فقط ρ₁ (خاک یکنواخت)",
                         "mean": "میانگین ρ₁ و ρ₂ (غیرمهندسی)"}.get)

        st.subheader("⚡ خطا و ایمنی")
        ig = st.number_input("جریان شبکه Ig (A)", min_value=1.0, value=5000.0)
        ts = st.number_input("زمان قطع خطا (s)", min_value=0.05, value=0.5)
        weight = st.selectbox("وزن مبنا (kg)", [50, 70], index=1)
        use_surface = st.checkbox("لایه سطحی (شن/سنگ‌ریزه) دارد", value=True)
        rho_s = st.number_input("ρ لایه سطحی (Ω·m)", min_value=1.0, value=3000.0)
        hs = st.number_input("ضخامت لایه سطحی (m)", min_value=0.0, value=0.1)

        run_btn = st.form_submit_button("🚀 اجرای محاسبات", use_container_width=True)

if run_btn:
    try:
        a_raw = parse_series(a_text, "فاصله a")
        v_raw = parse_series(v_text, "مقادیر اندازه‌گیری")
        if mode.startswith("مقاومت قرائت"):
            if a_raw.size == v_raw.size:
                v_raw = 2.0 * np.pi * a_raw * v_raw   # ρa = 2π·a·R
        a_vals, rho_vals = validate_data(a_raw, v_raw)
        fit = fit_two_layer(tuple(a_vals), tuple(rho_vals))
    except (ValueError, RuntimeError) as e:
        st.error(f"خطا: {e}")
        st.stop()

    rho_eq = equivalent_resistivity(method, fit, area)
    rg = grid_resistance(rho_eq, area, length, depth)
    rg_design = rg["sverak"]
    gpr = ig * rg_design
    tv = tolerable_voltages(fit["rho1"], rho_s if use_surface else fit["rho1"],
                            hs if use_surface else 0.0, weight, ts)
    inp = dict(area=area, length=length, depth=depth, ig=ig, ts=ts, weight=weight)
    st.session_state["out"] = dict(a=a_vals, rho=rho_vals, fit=fit, rho_eq=rho_eq, rg=rg,
                                   rg_design=rg_design, gpr=gpr, tv=tv, inp=inp)

out = st.session_state.get("out")
if not out:
    st.info("داده‌ها را در پنل کناری وارد کنید و «اجرای محاسبات» را بزنید.")
    st.stop()

a_vals, rho_vals, fit = out["a"], out["rho"], out["fit"]
tv, gpr = out["tv"], out["gpr"]

# --- هشدارهای کیفیت مدل ---
if fit["rmse_pct"] > 10:
    st.warning(f"خطای برازش بالاست (RMSE = {fit['rmse_pct']:.1f}٪). داده‌ها را بازبینی کنید یا خاک ممکن است بیش از دو لایه باشد.")
if abs(fit["k"]) < 0.05 or fit["rmse_uniform_pct"] <= fit["rmse_pct"] * 1.2:
    st.info("خاک تقریباً یکنواخت است؛ ρ₂ و h در این حالت قابل اتکا نیستند.")
if fit["h_at_bound"] or fit["rho_at_bound"]:
    st.warning("یکی از پارامترها به مرز بازه جستجو رسیده است؛ نتیجه برازش یکتا نیست.")

# --- شاخص‌ها ---
cols = st.columns(4)
items = [("ρ₁ (Ω·m)", f"{fit['rho1']:.1f}"), ("ρ₂ (Ω·m)", f"{fit['rho2']:.1f}"),
         ("Rg طراحی (Ω)", f"{out['rg_design']:.3f}"), ("GPR (V)", f"{gpr:.0f}")]
for col, (title, val) in zip(cols, items):
    with col, st.container(border=True):
        st.metric(title, val)

if gpr <= tv["e_touch"]:
    st.success(f"GPR = {gpr:.0f} V کمتر از ولتاژ تماس مجاز ({tv['e_touch']:.0f} V) است؛ طرح ایمن است.")
else:
    st.error(f"GPR = {gpr:.0f} V بیشتر از ولتاژ تماس مجاز ({tv['e_touch']:.0f} V) است؛ باید Em و Es بررسی شود.")

tab1, tab2, tab3 = st.tabs(["📈 نمودارها", "📋 جدول‌ها", "📄 گزارش"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 4))
        a_fine = np.logspace(np.log10(a_vals.min()), np.log10(a_vals.max()), 200)
        ax.semilogx(a_vals, rho_vals, "o", color="red", label="Measured")
        ax.semilogx(a_fine, rho_apparent_two_layer(a_fine, fit["rho1"], fit["rho2"], fit["h"]),
                    "-", color="#0078D7", lw=2, label="Two-layer model")
        ax.set_xlabel("Probe spacing a (m)")
        ax.set_ylabel("Apparent resistivity (Ω·m)")
        ax.grid(True, which="both", ls="--", alpha=0.4)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
    with c2:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.semilogx(a_vals, fit["rel_err"], "o-", color="#e74c3c")
        ax2.axhline(0, color="black", ls="--")
        ax2.set_xlabel("Probe spacing a (m)")
        ax2.set_ylabel("Relative error (%)")
        ax2.grid(True, which="both", ls="--", alpha=0.4)
        st.pyplot(fig2)
        plt.close(fig2)

with tab2:
    st.subheader("پارامترهای مدل خاک")
    st.dataframe(pd.DataFrame({
        "پارامتر": ["ρ₁", "ρ₂", "h", "K"],
        "مقدار": [f"{fit['rho1']:.1f}", f"{fit['rho2']:.1f}", f"{fit['h']:.2f}", f"{fit['k']:.3f}"],
        "واحد": ["Ω·m", "Ω·m", "m", "—"]}), hide_index=True, use_container_width=True)
    st.subheader("مقایسه اندازه‌گیری و مدل")
    df_out = pd.DataFrame({"a (m)": a_vals, "Measured (Ω·m)": rho_vals,
                           "Fitted (Ω·m)": fit["pred"], "Error (%)": fit["rel_err"]})
    st.dataframe(df_out.round(3), hide_index=True, use_container_width=True)

with tab3:
    report = build_report(out["inp"], fit, out["rho_eq"], out["rg"], out["rg_design"], gpr, tv)
    st.markdown(report)
    d1, d2 = st.columns(2)
    d1.download_button("📥 گزارش (Markdown)", report.encode("utf-8"),
                       file_name="grounding_report.md", use_container_width=True)
    d2.download_button("📥 داده‌ها (CSV)", df_out.to_csv(index=False).encode("utf-8-sig"),
                       file_name="soil_analysis.csv", use_container_width=True)
