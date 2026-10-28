# -*- coding: utf-8 -*-
"""
Soil & Grounding Designer — تحلیل مقاومت ویژه خاک (Wenner) و طراحی اولیه شبکه زمین
مرجع: IEEE Std 80

امکانات: برازش خاک دولایه، مقطع دقیق SVG، نمای سه‌بعدی (Plotly)، مقاومت شبکه (Sverak)،
GPR و ولتاژهای مجاز، ولتاژهای mesh و step (IEEE 80)، نقشه پتانسیل و گرادیان (الهام‌گرفته از ماژول‌هایی مثل CYMGRD)، کنترل کفایت داده، بازه اطمینان پارامترها، گزارش راست‌به‌چپ و توصیه مهندسی (IEEE 80/81).
"""
from __future__ import annotations

import re
import base64
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
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

    # عدم‌قطعیت پارامترها از ژاکوبین (باقی‌مانده‌ها در مقیاس log10، پارامترها خطی‌اند)
    dof = max(a.size - 3, 1)
    sigma2 = 2.0 * best.cost / dof
    try:
        cov = sigma2 * np.linalg.pinv(best.jac.T @ best.jac)
        std = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    except np.linalg.LinAlgError:
        std = np.full(3, np.inf)
    rel_unc, ci_95 = [], []
    for x, sd in zip(best.x, std):
        if np.isfinite(sd):
            r = float(1.96 * sd / x)                     # نیم‌پهنای نسبی بازه ۹۵٪
            rel_unc.append(r)
            ci_95.append((float(x * np.exp(-r)), float(x * np.exp(r))))   # همیشه مثبت
        else:
            rel_unc.append(float("inf"))
            ci_95.append((float("nan"), float("nan")))

    return {
        "rho1": rho1, "rho2": rho2, "h": h,
        "k": (rho2 - rho1) / (rho2 + rho1),
        "ci_95": ci_95, "rel_unc": rel_unc, "dof": int(a.size - 3),
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
                 gpr: float, tv: dict, ex: dict | None = None) -> str:
    mesh_ok = bool(ex) and ex["mesh"]["Em"] <= tv["e_touch"] and ex["mesh"]["Es"] <= tv["e_step"]
    if gpr <= tv["e_touch"]:
        verdict = "✅ GPR کمتر از ولتاژ تماس مجاز است؛ تحلیل ولتاژهای mesh و step الزامی نیست."
    elif mesh_ok:
        verdict = "⚠️ GPR از ولتاژ تماس مجاز بیشتر است، اما Em و Es در حد مجازند؛ ولتاژ منتقل‌شده را کنترل کنید."
    elif ex:
        verdict = "🚨 Em یا Es از حد مجاز بیشتر است؛ هندسه شبکه، لایه سطحی، Ig یا زمان قطع باید اصلاح شود."
    else:
        verdict = "⚠️ GPR بیشتر از ولتاژ تماس مجاز است؛ محاسبه Em و Es و اصلاح طرح لازم است."
    mesh_md = ""
    if ex:
        mesh_md = (f"\n### ۴. ولتاژهای mesh و step (IEEE 80)\n- ولتاژ mesh (Em) = {ex['mesh']['Em']:,.0f} V\n"
                   f"- ولتاژ step (Es) = {ex['mesh']['Es']:,.0f} V\n"
                   f"- فاصله هادی‌ها D = {ex['geom']['D']:.1f} m ، میله‌ها = {ex['geom']['n_rods']} عدد\n")
        if ex.get("num_stats"):
            q = ex["num_stats"]
            mesh_md += (f"- تحلیل عددی: بیشینه ولتاژ تماس = {q['touch_max']:,.0f} V ، بیشینه ولتاژ گام = {q['step_max']:,.0f} V ، "
                        f"گرادیان داخل شبکه = {q['grad_in'][0]:.1f}–{q['grad_in'][1]:.1f} V/m\n")
    return f"""## گزارش تحلیل خاک و شبکه زمین — {date.today().isoformat()}
حالت استفاده: {"پروژه واقعی" if inp.get("real_mode", True) else "آموزشی"}

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

{mesh_md}
**نتیجه:** {verdict}
"""


# ----- UI -----
# ============================================================
#  استایل
# ============================================================
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;500;700;800&display=swap');
.stApp, button, input, textarea, label, .stMarkdown { font-family: 'Vazirmatn', Tahoma, sans-serif !important; }
.block-container { padding-top: 3.6rem; max-width: 1280px; direction: rtl; }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.stMarkdown, .stAlert, [data-testid="stWidgetLabel"] { direction: rtl; text-align: right; }
.js-plotly-plot, .stDataFrame, input, textarea, [data-baseweb="select"] { direction: ltr; }
section[data-testid="stSidebar"] { direction: rtl; }
section[data-testid="stSidebar"] > div { background: linear-gradient(180deg, rgba(37,99,235,.09), rgba(37,99,235,0) 40%); }

.n { direction: ltr; unicode-bidi: isolate; display: inline-block; font-weight: 700; }

/* برند سایدبار */
.brand { display:flex; align-items:center; gap:12px; margin-bottom:6px; }
.brand .logo { width:42px; height:42px; border-radius:12px; display:flex; align-items:center; justify-content:center;
  font-size:22px; background:linear-gradient(135deg,#1d4ed8,#7c3aed); box-shadow:0 6px 16px rgba(37,99,235,.35); }
.brand-t { font-weight:800; font-size:1.05rem; }
.brand-s { font-size:.75rem; opacity:.65; }
.side-t { font-weight:800; font-size:.95rem; margin:14px 0 4px; padding-bottom:6px; border-bottom:1px solid rgba(128,128,128,.25); }

/* هدر */
.hero { position:relative; overflow:hidden; border-radius:20px; padding:30px 34px; color:#fff; margin-bottom:20px;
  background: linear-gradient(135deg,#0f172a 0%,#1e3a8a 55%,#2563eb 100%); box-shadow:0 14px 34px rgba(30,58,138,.35); }
.hero::after { content:""; position:absolute; inset:auto auto -90px -70px; width:300px; height:300px; border-radius:50%;
  background:radial-gradient(circle, rgba(124,58,237,.55), rgba(124,58,237,0) 70%); }
.hero-badge { display:inline-block; font-size:.75rem; font-weight:700; padding:4px 12px; border-radius:20px;
  background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.25); margin-bottom:12px; direction:ltr; }
.hero-title { font-size:1.85rem; font-weight:800; line-height:1.5; }
.hero-sub { opacity:.85; margin-top:6px; font-size:.95rem; }
.hero-chips { margin-top:16px; display:flex; flex-wrap:wrap; gap:8px; position:relative; z-index:1; }
.hero-chips span { font-size:.78rem; padding:4px 12px; border-radius:20px; background:rgba(255,255,255,.12); direction:ltr; }

/* شاخص‌ها */
.kpi-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:14px; margin-bottom:16px; }
.kpi { border:1px solid rgba(128,128,128,.25); border-top:4px solid var(--accent); border-radius:16px; padding:16px 18px;
  background:rgba(128,128,128,.06); transition:transform .15s, box-shadow .15s; }
.kpi:hover { transform:translateY(-3px); box-shadow:0 10px 24px rgba(0,0,0,.12); }
.kpi-top { display:flex; justify-content:space-between; align-items:center; }
.kpi-label { font-size:.82rem; opacity:.7; font-weight:500; }
.kpi-ico { font-size:1.25rem; }
.kpi-value { font-size:2rem; font-weight:800; direction:ltr; text-align:right; line-height:1.3; margin-top:6px; }
.kpi-value small { font-size:.85rem; font-weight:500; opacity:.6; margin-inline-start:6px; }
.kpi-sub { font-size:.78rem; opacity:.6; margin-top:2px; }

/* برچسب‌ها و یادداشت‌ها */
.chips { display:flex; flex-wrap:wrap; gap:8px; margin:6px 0 10px; }
.chip { font-size:.8rem; font-weight:700; padding:5px 12px; border-radius:20px; border:1px solid; }
.chip.good { color:#059669; background:rgba(16,185,129,.12); border-color:rgba(16,185,129,.4); }
.chip.warn { color:#d97706; background:rgba(245,158,11,.12); border-color:rgba(245,158,11,.4); }
.chip.bad  { color:#dc2626; background:rgba(239,68,68,.12);  border-color:rgba(239,68,68,.4); }
.chip.neutral { background:rgba(128,128,128,.1); border-color:rgba(128,128,128,.3); }
.note { padding:10px 14px; border-radius:12px; margin:8px 0; font-size:.9rem; border-inline-start:4px solid; }
.note.info { background:rgba(37,99,235,.08); border-color:#2563eb; }
.note.warn { background:rgba(245,158,11,.1); border-color:#f59e0b; }

/* وضعیت ایمنی */
.status { border-radius:18px; padding:18px 24px; border:1px solid; margin:6px 0 10px; }
.status.ok  { background:rgba(16,185,129,.1); border-color:rgba(16,185,129,.5); }
.status.bad { background:rgba(239,68,68,.1);  border-color:rgba(239,68,68,.5); }
.status.warn { background:rgba(245,158,11,.1); border-color:rgba(245,158,11,.5); }
.status-t { font-size:1.15rem; font-weight:800; }
.status-d { font-size:.9rem; opacity:.85; margin-top:4px; }
.track { position:relative; height:10px; border-radius:6px; background:rgba(128,128,128,.25); margin:16px 0 4px; direction:ltr; }
.fill { height:100%; border-radius:6px; }
.limit { position:absolute; top:-5px; left:66.67%; width:2px; height:20px; background:currentColor; opacity:.7; }
.track-l { display:flex; justify-content:space-between; font-size:.72rem; opacity:.6; direction:ltr; }

/* عنوان بخش و کارت‌های کوچک */
.sec { display:flex; align-items:center; gap:10px; margin:22px 0 10px; }
.sec .bar { width:5px; height:24px; border-radius:3px; background:linear-gradient(#2563eb,#8b5cf6); }
.sec .t { font-size:1.12rem; font-weight:800; }
.sec .s { font-size:.82rem; opacity:.6; margin-inline-start:6px; }
.mini-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:12px; margin-bottom:8px; }
.mini { border:1px solid rgba(128,128,128,.25); border-radius:14px; padding:14px 16px; background:rgba(128,128,128,.06); }
.mini.hl { border-color:#2563eb; background:rgba(37,99,235,.09); box-shadow:0 0 0 1px #2563eb inset; }
.mini .l { font-size:.8rem; opacity:.7; }
.mini .v { font-size:1.45rem; font-weight:800; direction:ltr; text-align:right; }
.mini .u { font-size:.8rem; opacity:.6; margin-inline-start:4px; font-weight:500; }
.mini .tag { font-size:.7rem; font-weight:700; color:#2563eb; }

/* تب‌ها و دکمه‌ها */
.stTabs [data-baseweb="tab-list"] { gap:6px; border-bottom:1px solid rgba(128,128,128,.25); }
.stTabs [data-baseweb="tab"] { border-radius:10px 10px 0 0; padding:10px 20px; font-weight:700; }
[data-testid="stFormSubmitButton"] button, .stDownloadButton button { border-radius:12px; font-weight:700; padding:.6rem 1rem; }
[data-testid="stFormSubmitButton"] button { background:linear-gradient(135deg,#1d4ed8,#2563eb); border:0; box-shadow:0 6px 16px rgba(37,99,235,.35); }
[data-testid="stFormSubmitButton"] button p { color:#fff; }

/* گزارش */
.rep-sec { display:flex; align-items:center; gap:10px; margin:20px 0 6px; font-weight:800; font-size:1.05rem; direction:rtl; }
.rep-sec::before { content:""; width:5px; height:20px; border-radius:3px; background:linear-gradient(#2563eb,#8b5cf6); }
.rep-row { display:flex; justify-content:space-between; align-items:center; gap:14px; padding:9px 6px;
  border-bottom:1px dashed rgba(128,128,128,.3); direction:rtl; text-align:right; }
.rep-row .k { opacity:.85; }
.rep-row .v { font-weight:800; direction:ltr; unicode-bidi:isolate; text-align:left; white-space:nowrap; }
.rep-row.hl { background:rgba(37,99,235,.08); border-radius:10px; border-bottom:0; }
.rep-verdict { margin-top:18px; padding:14px 18px; border-radius:14px; font-weight:700; direction:rtl; text-align:right; }
.rep-verdict.ok  { background:rgba(16,185,129,.12); border:1px solid rgba(16,185,129,.5); }
.rep-verdict.bad { background:rgba(239,68,68,.12); border:1px solid rgba(239,68,68,.5); }
.rep-verdict.warn { background:rgba(245,158,11,.12); border:1px solid rgba(245,158,11,.5); }
.rep-title { font-size:1.2rem; font-weight:800; direction:rtl; text-align:right; }
.rep-date { font-size:.8rem; opacity:.6; direction:rtl; text-align:right; margin-bottom:4px; }

/* توصیه مهندسی */
.note.bad { background:rgba(239,68,68,.1); border-color:#ef4444; }
.adv-sum { border-radius:18px; padding:18px 22px; margin:6px 0 14px; direction:rtl; text-align:right;
  background:linear-gradient(135deg,rgba(37,99,235,.12),rgba(139,92,246,.10)); border:1px solid rgba(37,99,235,.35); }
.adv-sum-t { font-weight:800; font-size:1.1rem; margin-bottom:6px; }
.adv-prio { margin-top:10px; padding-top:10px; border-top:1px dashed rgba(128,128,128,.4); }
.adv { --c:#2563eb; border:1px solid rgba(128,128,128,.25); border-inline-start:5px solid var(--c); border-radius:14px;
  padding:14px 18px; margin:10px 0; background:rgba(128,128,128,.05); direction:rtl; text-align:right; }
.adv.bad { --c:#ef4444; } .adv.warn { --c:#f59e0b; } .adv.ok { --c:#10b981; }
.adv-h { display:flex; justify-content:space-between; align-items:center; gap:10px; font-weight:800; }
.adv-ref { font-size:.7rem; font-weight:700; padding:2px 10px; border-radius:12px; background:rgba(37,99,235,.12);
  color:#2563eb; direction:ltr; white-space:nowrap; }
.adv-b { margin-top:6px; font-size:.92rem; line-height:1.95; opacity:.95; }
.adv-li { display:flex; gap:8px; align-items:flex-start; margin-top:6px; font-size:.9rem; line-height:1.9; }
.adv-li > span:first-child { flex:0 0 auto; color:var(--c); }
.adv-foot { font-size:.78rem; opacity:.6; margin-top:14px; direction:rtl; text-align:right; }
.foot { text-align:center; font-size:.78rem; opacity:.55; margin:30px 0 6px; }
"""

C_BLUE, C_PURPLE, C_GREEN, C_AMBER, C_RED = "#2563eb", "#8b5cf6", "#10b981", "#f59e0b", "#ef4444"
FONT = "Vazirmatn, Tahoma, sans-serif"
DEFAULTS = dict(
    mode_R=False,
    a_text="1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 50, 60, 80, 100, 120, 150",
    v_text="250, 240, 230, 210, 190, 175, 160, 150, 140, 135, 130, 128, 125, 123, 122, 121",
    area=4000.0, aspect=1.0, spacing=15.0, d_mm=12.0, rod_mode="none", rod_len=3.0,
    depth=0.5, method="apparent",
    ig=5000.0, ts=0.5, weight=70, use_surface=True, rho_s=3000.0, hs=0.1, real_mode=True,
)


# ============================================================
#  توابع کمکی نمایش
# ============================================================
def H(s: str) -> str:
    """حذف تورفتگی خطوط تا Markdown قطعه HTML را به‌عنوان کد تفسیر نکند."""
    return "".join(line.strip() for line in s.strip().splitlines())


def num(x: float, fmt: str = ",.1f") -> str:
    return f'<span class="n">{x:{fmt}}</span>'


def section(title: str, sub: str = "") -> None:
    sub_html = f'<span class="s">{sub}</span>' if sub else ""
    st.markdown(H(f'<div class="sec"><div class="bar"></div><div class="t">{title}</div>{sub_html}</div>'),
                unsafe_allow_html=True)


def chip(text: str, kind: str = "neutral") -> str:
    return f'<span class="chip {kind}">{text}</span>'


def mini_card(label: str, value: str, unit: str = "", highlight: bool = False, tag: str = "") -> str:
    tag_html = f'<div class="tag">{tag}</div>' if tag else ""
    return (f'<div class="mini{" hl" if highlight else ""}">{tag_html}<div class="l">{label}</div>'
            f'<div class="v">{value}<span class="u">{unit}</span></div></div>')


def style_fig(fig: go.Figure, height: int = 380) -> go.Figure:
    grid = "rgba(128,128,128,.2)"
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=13), hoverlabel=dict(font_family=FONT),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(gridcolor=grid, zeroline=False)
    fig.update_yaxes(gridcolor=grid, zeroline=False)
    return fig


def fit_fig(a, rho, fit) -> go.Figure:
    a_fine = np.logspace(np.log10(a.min()), np.log10(a.max()), 200)
    model = rho_apparent_two_layer(a_fine, fit["rho1"], fit["rho2"], fit["h"])
    top1 = fit["rho1"] >= fit["rho2"]
    fig = go.Figure()
    fig.add_hline(y=fit["rho1"], line_dash="dot", line_color=C_BLUE, opacity=.55,
                  annotation_text="ρ₁ (a→0)", annotation_position="top left" if top1 else "bottom left")
    fig.add_hline(y=fit["rho2"], line_dash="dot", line_color=C_PURPLE, opacity=.55,
                  annotation_text="ρ₂ (a→∞)", annotation_position="bottom right" if top1 else "top right")
    fig.add_trace(go.Scatter(x=a_fine, y=model, mode="lines", name="مدل دولایه",
                             line=dict(color=C_BLUE, width=3.5)))
    fig.add_trace(go.Scatter(x=a, y=rho, mode="markers", name="اندازه‌گیری",
                             marker=dict(size=11, color=C_RED, line=dict(width=2, color="white"))))
    fig.update_xaxes(type="log", title="فاصله الکترودها a (m)")
    fig.update_yaxes(title="مقاومت ظاهری ρa (Ω·m)")
    return style_fig(fig, 400)


def residual_fig(a, err) -> go.Figure:
    colors = [C_GREEN if abs(e) <= 5 else C_AMBER if abs(e) <= 10 else C_RED for e in err]
    fig = go.Figure(go.Bar(x=[f"{v:g}" for v in a], y=err, marker_color=colors,
                           text=[f"{e:+.1f}" for e in err], textposition="outside", cliponaxis=False,
                           hovertemplate="a = %{x} m<br>خطا = %{y:.2f}٪<extra></extra>"))
    fig.add_hrect(y0=-10, y1=10, fillcolor="rgba(16,185,129,.10)", line_width=0)
    fig.update_xaxes(title="فاصله الکترودها a (m)", type="category")
    fig.update_yaxes(title="خطای نسبی (٪)")
    return style_fig(fig, 300)


# ---------- مقطع واقع‌گرایانه خاک (SVG، دقیق و بدون نیاز به اینترنت) ----------
_STOPS = [(0.0, (58, 40, 30)), (0.3, (120, 86, 56)), (0.55, (190, 160, 108)),
          (0.78, (150, 152, 155)), (1.0, (205, 208, 212))]


def _soil_rgb(rho: float):
    t = float(np.clip((np.log10(max(rho, 1.0)) - 1.0) / 3.2, 0.0, 1.0))
    for (t0, c0), (t1, c1) in zip(_STOPS, _STOPS[1:]):
        if t <= t1:
            u = (t - t0) / (t1 - t0)
            return tuple(c0[i] + (c1[i] - c0[i]) * u for i in range(3))
    return _STOPS[-1][1]


def _hex(c, k: float = 1.0) -> str:
    return "#" + "".join(f"{int(np.clip(v * k, 0, 255)):02x}" for v in c)


def _texture(rng, x0, y0, w, h, rho, base) -> str:
    """بافت بصری لایه بر اساس نوع خاک (رس، لوم، شن، سنگ)."""
    out = []
    area = w * h
    n_dots = int(min(area / 260, 800))
    for _ in range(n_dots):
        k = rng.choice([0.72, 0.82, 1.18, 1.3])
        out.append(f'<circle cx="{x0 + rng.uniform(0, w):.0f}" cy="{y0 + rng.uniform(0, h):.0f}" '
                   f'r="{rng.uniform(.6, 1.5):.1f}" fill="{_hex(base, k)}" opacity=".8"/>')
    if rho >= 60:                                           # شن و قلوه‌سنگ
        n = int(min(area / (1400 if rho < 2000 else 2600), 170))
        for _ in range(n):
            rx, ry = rng.uniform(3, 9 if rho < 2000 else 12), rng.uniform(2, 6 if rho < 2000 else 8)
            k = rng.uniform(.85, 1.25)
            out.append(f'<ellipse cx="{x0 + rng.uniform(0, w):.0f}" cy="{y0 + rng.uniform(0, h):.0f}" '
                       f'rx="{rx:.1f}" ry="{ry:.1f}" transform="rotate({rng.uniform(-40, 40):.0f})" '
                       f'fill="{_hex(base, k)}" stroke="{_hex(base, .55)}" stroke-width="1" opacity=".9"/>')
    if rho >= 2000:                                         # سنگ شکسته با ترک
        for _ in range(int(min(area / 5000, 60))):
            cx, cy = x0 + rng.uniform(0, w), y0 + rng.uniform(0, h)
            pts = []
            for ang in np.linspace(0, 2 * np.pi, 6, endpoint=False):
                r = rng.uniform(10, 24)
                pts.append(f"{cx + r * np.cos(ang):.0f},{cy + r * np.sin(ang) * .7:.0f}")
            out.append(f'<polygon points="{" ".join(pts)}" fill="{_hex(base, rng.uniform(.9, 1.15))}" '
                       f'stroke="{_hex(base, .5)}" stroke-width="1.2" opacity=".75"/>')
    for _ in range(3):                                      # لایه‌بندی طبیعی
        y = y0 + rng.uniform(.15, .9) * h
        d = f"M{x0},{y:.0f} " + " ".join(f"L{x0 + i * 40},{y + rng.uniform(-3, 3):.0f}"
                                         for i in range(1, int(w / 40) + 1))
        out.append(f'<path d="{d}" stroke="{_hex(base, .6)}" stroke-width="1" fill="none" opacity=".35"/>')
    return "".join(out)


def soil_section_svg(fit: dict, depth: float, hs: float, rho_s: float) -> str:
    W, H_, Y0, Y_END = 800, 520, 84, 504
    h = fit["h"]
    ymax = max(h * 2.0, h + 3.0)
    scale = (Y_END - Y0) / (hs + ymax)
    hs_px = max(hs * scale, 6.0) if hs > 0 else 0.0
    ys0 = Y0 + hs_px
    yb = ys0 + h * scale
    rng = np.random.default_rng(int(abs(fit["rho1"] * 1000 + fit["rho2"] * 10 + h * 100)) % (2**32))
    c1, c2 = _soil_rgb(fit["rho1"]), _soil_rgb(fit["rho2"])
    font = "Vazirmatn, Tahoma, 'Noto Sans Arabic', sans-serif"
    tx = f'font-family="{font}" style="unicode-bidi:plaintext" '
    halo = 'fill="#fff" stroke="#000" stroke-width="3.2" paint-order="stroke" stroke-linejoin="round" '

    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H_}">',
         '<defs><linearGradient id="sky" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#bfe0ff"/>'
         '<stop offset="1" stop-color="#eef7ff"/></linearGradient>'
         '<linearGradient id="shade" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#000" stop-opacity="0"/>'
         '<stop offset="1" stop-color="#000" stop-opacity=".32"/></linearGradient>'
         '<radialGradient id="cu" cx=".35" cy=".3"><stop offset="0" stop-color="#ffd2a1"/>'
         '<stop offset=".5" stop-color="#cf7f3a"/><stop offset="1" stop-color="#7a4a1e"/></radialGradient>'
         f'<clipPath id="k1"><rect x="0" y="{ys0:.0f}" width="{W}" height="{yb - ys0:.0f}"/></clipPath>'
         f'<clipPath id="k2"><rect x="0" y="{yb:.0f}" width="{W}" height="{Y_END - yb:.0f}"/></clipPath>'
         f'<clipPath id="ks"><rect x="0" y="{Y0}" width="{W}" height="{hs_px:.0f}"/></clipPath></defs>',
         f'<rect width="{W}" height="{H_}" fill="url(#sky)"/>']

    # لایه اول
    p.append(f'<rect x="0" y="{ys0:.0f}" width="{W}" height="{yb - ys0:.0f}" fill="{_hex(c1)}"/>')
    p.append(f'<g clip-path="url(#k1)">{_texture(rng, 0, ys0, W, yb - ys0, fit["rho1"], c1)}</g>')
    # لایه دوم
    p.append(f'<rect x="0" y="{yb:.0f}" width="{W}" height="{Y_END - yb:.0f}" fill="{_hex(c2)}"/>')
    p.append(f'<g clip-path="url(#k2)">{_texture(rng, 0, yb, W, Y_END - yb, fit["rho2"], c2)}</g>')
    # لایه سطحی
    if hs > 0:
        cs = _soil_rgb(rho_s)
        p.append(f'<rect x="0" y="{Y0}" width="{W}" height="{hs_px:.0f}" fill="{_hex(cs)}"/>')
        p.append(f'<g clip-path="url(#ks)">{_texture(rng, 0, Y0, W, hs_px, max(rho_s, 600), cs)}</g>')
    # سایه عمق و مرز لایه‌ها
    p.append(f'<rect x="0" y="{Y0}" width="{W}" height="{Y_END - Y0}" fill="url(#shade)"/>')
    d = f"M0,{yb:.0f} " + " ".join(f"L{i * 40},{yb + rng.uniform(-2.5, 2.5):.1f}" for i in range(1, W // 40 + 1))
    p.append(f'<path d="{d}" stroke="#1b1b1b" stroke-width="2.2" fill="none" opacity=".75"/>')
    # چمن
    blades = "".join(f'<path d="M{x:.0f},{Y0} l{rng.uniform(-3, 3):.1f},-{rng.uniform(5, 11):.0f}" '
                     f'stroke="#3f9b3a" stroke-width="2"/>' for x in rng.uniform(0, W, 200))
    p.append(f'<rect x="0" y="{Y0 - 5}" width="{W}" height="6" fill="#4fae48"/>{blades}')

    # هادی‌های شبکه زمین (مقطع)
    gy = ys0 + depth * scale
    for gx in (250, 330, 410, 490, 570, 650):
        p.append(f'<circle cx="{gx}" cy="{gy:.0f}" r="7" fill="url(#cu)" stroke="#5a3513" stroke-width="1"/>')
    p.append(f'<text x="450" y="{gy - 15:.0f}" text-anchor="middle" font-size="14" {tx}{halo}>'
             f'هادی‌های شبکه زمین · عمق دفن {depth:g} m</text>')

    # برچسب‌های لایه‌ها
    def tag(y_top, y_bot, title, value):
        cy = (y_top + y_bot) / 2
        p.append(f'<rect x="14" y="{cy - 29:.0f}" width="196" height="58" rx="12" fill="#000" opacity=".58"/>')
        p.append(f'<text x="198" y="{cy - 6:.0f}" text-anchor="end" font-size="15" font-weight="700" fill="#fff" {tx}>{title}</text>')
        p.append(f'<text x="198" y="{cy + 18:.0f}" text-anchor="end" font-size="17" fill="#ffe08a" '
                 f'font-family="{font}" style="direction:ltr;unicode-bidi:isolate">{value}</text>')

    tag(ys0, yb, "لایه اول", f"ρ₁ = {fit['rho1']:,.0f} Ω·m")
    tag(yb, Y_END, "لایه دوم (تا عمق زیاد)", f"ρ₂ = {fit['rho2']:,.0f} Ω·m")
    if hs > 0:
        p.append(f'<text x="{W - 12}" y="{Y0 - 14}" text-anchor="end" font-size="13" {tx}{halo}>'
                 f'لایه سطحی (سنگ‌ریزه) · ضخامت {hs:g} m · ρs = {rho_s:,.0f} Ω·m</text>')

    # اندازه‌گذاری h
    xd = W - 30
    p.append(f'<line x1="{xd}" y1="{ys0:.0f}" x2="{xd}" y2="{yb:.0f}" stroke="#fff" stroke-width="2.4"/>')
    for yy, sgn in ((ys0, 1), (yb, -1)):
        p.append(f'<path d="M{xd - 6},{yy + sgn * 10:.0f} L{xd},{yy:.0f} L{xd + 6},{yy + sgn * 10:.0f}" '
                 f'stroke="#fff" stroke-width="2.4" fill="none"/>')
    ym = (ys0 + yb) / 2
    p.append(f'<text transform="translate({xd - 12},{ym:.0f}) rotate(-90)" text-anchor="middle" font-size="15" '
             f'font-family="{font}" style="direction:ltr" {halo}>h = {h:.2f} m</text>')
    p.append("</svg>")
    return "".join(p)


def safety_fig(gpr: float, tv: dict, em: float, es: float) -> go.Figure:
    labels = ["GPR", "Em (mesh)", "Es (step)"]
    vals = [gpr, em, es]
    limits = [tv["e_touch"], tv["e_touch"], tv["e_step"]]
    colors = [C_GREEN if v <= l else C_RED for v, l in zip(vals, limits)]
    fig = go.Figure(go.Bar(x=vals, y=labels, orientation="h", marker_color=colors, name="مقدار محاسبه‌شده",
                           text=[f"{v:,.0f} V" for v in vals], textposition="outside", cliponaxis=False))
    fig.add_trace(go.Scatter(x=limits, y=labels, mode="markers", name="حد مجاز",
                             marker=dict(symbol="line-ns", size=26, line=dict(width=4, color=C_BLUE))))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(range=[0, max(vals + limits) * 1.25], title="ولتاژ (V)")
    return style_fig(fig, 280)


def _box(x0, x1, y0, y1, z0, z1, color, opacity, name) -> go.Mesh3d:
    """مکعب‌مستطیل توپر (۸ رأس، ۱۲ مثلث) برای نمایش لایه‌ها."""
    return go.Mesh3d(
        x=[x0, x1, x1, x0, x0, x1, x1, x0], y=[y0, y0, y1, y1, y0, y0, y1, y1],
        z=[z0, z0, z0, z0, z1, z1, z1, z1],
        i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2], j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
        color=color, opacity=opacity, name=name, showlegend=True, flatshading=True,
        hovertemplate=name + "<extra></extra>")


def fig_3d_soil(fit: dict, g: dict, depth: float, hs: float, exag: float = 3.0) -> go.Figure:
    """نمای سه‌بعدی: دو لایه خاک، لایه سطحی، هادی‌های واقعی شبکه و میله‌ها."""
    Lx, Ly = g["Lx"], g["Ly"]
    L = max(Lx, Ly)
    h = fit["h"]
    zmax = max(h * 2.0, h + 3.0, depth + g["Lr"] + 1.0)
    c1, c2 = _hex(_soil_rgb(fit["rho1"])), _hex(_soil_rgb(fit["rho2"]))
    fig = go.Figure()
    fig.add_trace(_box(0, Lx, 0, Ly, -zmax, -h, c2, 0.50, f"لایه دوم · ρ₂ = {fit['rho2']:,.0f} Ω·m"))
    fig.add_trace(_box(0, Lx, 0, Ly, -h, 0, c1, 0.55, f"لایه اول · ρ₁ = {fit['rho1']:,.0f} Ω·m"))
    if hs > 0:
        fig.add_trace(_box(0, Lx, 0, Ly, 0, hs, "#9aa0a6", 0.95, f"لایه سطحی · {hs:g} m"))

    zg = -g["h"]
    xs, ys, zs = [], [], []
    for y in g["ys"]:
        xs += [0, Lx, None]; ys += [y, y, None]; zs += [zg, zg, None]
    for x in g["xs"]:
        xs += [x, x, None]; ys += [0, Ly, None]; zs += [zg, zg, None]
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=C_AMBER, width=6),
                               name=f"هادی‌های شبکه ({g['nx']}×{g['ny']} ، D ≈ {g['D']:.1f} m)", hoverinfo="skip"))
    if g["n_rods"]:
        rx, ry, rz = [], [], []
        for x, y in g["rods"]:
            rx += [x, x, None]; ry += [y, y, None]; rz += [zg, zg - g["Lr"], None]
        fig.add_trace(go.Scatter3d(x=rx, y=ry, z=rz, mode="lines", line=dict(color="#dc2626", width=7),
                                   name=f"میله‌های زمین ({g['n_rods']} عدد × {g['Lr']:g} m)", hoverinfo="skip"))

    ticks = np.linspace(-zmax, 0, 5)
    fig.update_layout(
        height=520, margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(family=FONT, size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
        scene=dict(
            xaxis=dict(title="X (m)", backgroundcolor="rgba(0,0,0,0)"),
            yaxis=dict(title="Y (m)", backgroundcolor="rgba(0,0,0,0)"),
            zaxis=dict(title="عمق (m)", range=[-zmax, max(hs, 0.0) + 0.2], tickvals=list(ticks),
                       ticktext=[f"{-t:.1f}" for t in ticks], backgroundcolor="rgba(0,0,0,0)"),
            aspectmode="manual",
            aspectratio=dict(x=Lx / L, y=Ly / L, z=float(np.clip(zmax / L * exag, 0.12, 1.0))),
            camera=dict(eye=dict(x=1.5, y=-1.7, z=0.9)),
        ))
    return fig


def potential_map_fig(view: str, num: dict, stt: dict, g: dict, tv: dict) -> go.Figure:
    """نقشه رنگی پتانسیل / ولتاژ تماس / ولتاژ گام / گرادیان با خط حد مجاز و جانمایی هادی‌ها."""
    xs, ys = num["xs"], num["ys"]
    cfg = {
        "ولتاژ تماس": (stt["touch"], "V", tv["e_touch"], "RdYlGn_r", 0.0, 2.0 * tv["e_touch"]),
        "ولتاژ گام": (stt["step"], "V", tv["e_step"], "RdYlGn_r", 0.0, 2.0 * tv["e_step"]),
        "پتانسیل سطح": (stt["V"], "V", None, "Turbo", None, stt["gpr"]),
        "گرادیان پتانسیل": (stt["grad"], "V/m", tv["e_step"], "Turbo", 0.0, None),
    }
    Z, unit, limit, cscale, zmin, zmax = cfg[view]
    fig = go.Figure(go.Heatmap(x=xs, y=ys, z=Z, colorscale=cscale, zmin=zmin, zmax=zmax, zsmooth="best",
                               colorbar=dict(title=unit), name=view,
                               hovertemplate="x = %{x:.1f} m<br>y = %{y:.1f} m<br>%{z:,.1f} " + unit + "<extra></extra>"))
    if limit is not None and float(np.max(Z)) > limit:
        fig.add_trace(go.Contour(x=xs, y=ys, z=Z, showscale=False, showlegend=True, hoverinfo="skip",
                                 name=f"حد مجاز ({limit:,.0f} {unit})", line=dict(color="#111111", width=3),
                                 contours=dict(start=limit, end=limit, size=max(limit, 1.0), coloring="lines")))
    lx, ly = [], []
    for y in g["ys"]:
        lx += [0, g["Lx"], None]; ly += [y, y, None]
    for x in g["xs"]:
        lx += [x, x, None]; ly += [0, g["Ly"], None]
    fig.add_trace(go.Scatter(x=lx, y=ly, mode="lines", line=dict(color="rgba(0,0,0,.55)", width=1),
                             name="هادی‌های شبکه", hoverinfo="skip"))
    if g["n_rods"]:
        fig.add_trace(go.Scatter(x=g["rods"][:, 0], y=g["rods"][:, 1], mode="markers", name="میله‌های زمین",
                                 marker=dict(symbol="x", size=8, color="#111111"), hoverinfo="skip"))
    fig.update_xaxes(title="X (m)", constrain="domain")
    fig.update_yaxes(title="Y (m)", scaleanchor="x", scaleratio=1)
    return style_fig(fig, 520)


def profile_figs(num: dict, stt: dict, g: dict, tv: dict, y_frac: float):
    """نیمرخ پتانسیل و ولتاژهای تماس/گام در امتداد X در موقعیت دلخواه Y."""
    ys = num["ys"]
    y_target = g["Ly"] * y_frac
    j = int(np.argmin(np.abs(ys - y_target)))
    xs = num["xs"]
    f1 = go.Figure(go.Scatter(x=xs, y=stt["V"][j], mode="lines", name="پتانسیل سطح", line=dict(color=C_BLUE, width=3)))
    f1.add_hline(y=stt["gpr"], line_dash="dot", line_color=C_AMBER, annotation_text="GPR", annotation_position="top left")
    f1.add_vrect(x0=0, x1=g["Lx"], fillcolor="rgba(128,128,128,.10)", line_width=0)
    f1.update_xaxes(title="X (m)"); f1.update_yaxes(title="پتانسیل سطح (V)")
    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=xs, y=stt["touch"][j], mode="lines", name="ولتاژ تماس", line=dict(color=C_RED, width=3)))
    f2.add_trace(go.Scatter(x=xs, y=stt["step"][j], mode="lines", name="ولتاژ گام", line=dict(color=C_PURPLE, width=3)))
    f2.add_hline(y=tv["e_touch"], line_dash="dash", line_color=C_RED, opacity=.6,
                 annotation_text="حد تماس", annotation_position="top left")
    f2.add_hline(y=tv["e_step"], line_dash="dash", line_color=C_PURPLE, opacity=.6,
                 annotation_text="حد گام", annotation_position="top right")
    f2.add_vrect(x0=0, x1=g["Lx"], fillcolor="rgba(128,128,128,.10)", line_width=0)
    f2.update_xaxes(title="X (m)"); f2.update_yaxes(title="ولتاژ (V)")
    return style_fig(f1, 300), style_fig(f2, 300), float(ys[j])


def report_html(inp, fit, rho_eq, rg, rg_design, gpr, tv, ex=None) -> str:
    """گزارش به‌صورت HTML ثابتِ راست‌به‌چپ (بدون وابستگی به لیست‌های Markdown)."""
    def row(k, v, hl=False):
        return f'<div class="rep-row{" hl" if hl else ""}"><span class="k">{k}</span><span class="v">{v}</span></div>'

    safe_ = gpr <= tv["e_touch"]
    mesh_ok = bool(ex) and ex["mesh"]["Em"] <= tv["e_touch"] and ex["mesh"]["Es"] <= tv["e_step"]
    if safe_:
        verdict, vcls = "GPR کمتر از ولتاژ تماس مجاز است؛ تحلیل ولتاژهای mesh و step الزامی نیست.", "ok"
    elif mesh_ok:
        verdict, vcls = "GPR از ولتاژ تماس مجاز بیشتر است، اما Em و Es در حد مجازند؛ ولتاژ منتقل‌شده را کنترل کنید.", "warn"
    elif ex:
        verdict, vcls = "Em یا Es از حد مجاز بیشتر است؛ هندسه شبکه، لایه سطحی، Ig یا زمان قطع باید اصلاح شود.", "bad"
    else:
        verdict, vcls = "GPR بیشتر از ولتاژ تماس مجاز است؛ محاسبه Em و Es و اصلاح طرح لازم است.", "bad"
    parts = [
        '<div class="rep-title">گزارش تحلیل خاک و شبکه زمین</div>',
        f'<div class="rep-date">{date.today().isoformat()} · {"حالت پروژه واقعی" if inp.get("real_mode", True) else "حالت آموزشی"}</div>',
        '<div class="rep-sec">مدل خاک (دولایه)</div>',
        row("مقاومت ویژه لایه اول ρ₁", f"{fit['rho1']:.1f} Ω·m"),
        row("مقاومت ویژه لایه دوم ρ₂", f"{fit['rho2']:.1f} Ω·m"),
        row("ضخامت لایه اول h", f"{fit['h']:.2f} m"),
        row("ضریب بازتاب K", f"{fit['k']:.3f}"),
        row("RMSE برازش", f"{fit['rmse_pct']:.2f} %"),
        row("بیشینه خطای برازش", f"{fit['max_err_pct']:.2f} %"),
        '<div class="rep-sec">شبکه زمین</div>',
        row("مقاومت ویژه معادل ρ_eq", f"{rho_eq:.1f} Ω·m"),
        row("مساحت", f"{inp['area']:.0f} m²"),
        row("طول کل هادی‌ها (شامل میله‌ها)", f"{inp['length']:.0f} m"),
        row("عمق دفن", f"{inp['depth']:.2f} m"),
        row("Rg به روش ρ/4r", f"{rg['simple']:.3f} Ω"),
        row("Rg به روش ρ/4r + ρ/L", f"{rg['with_length']:.3f} Ω"),
        row("Rg به روش Sverak (مقدار طراحی)", f"{rg_design:.3f} Ω", hl=True),
        row("جریان شبکه Ig", f"{inp['ig']:,.0f} A"),
        row("افزایش پتانسیل زمین GPR", f"{gpr:,.0f} V", hl=True),
        *(([
            '<div class="rep-sec">هندسه شبکه و ولتاژهای mesh/step (IEEE 80)</div>',
            row("ابعاد زمین", f"{ex['geom']['Lx']:.1f} × {ex['geom']['Ly']:.1f} m"),
            row("تعداد هادی‌ها / فاصله D", f"{ex['geom']['nx']}×{ex['geom']['ny']} / {ex['geom']['D']:.1f} m"),
            row("میله‌های زمین", f"{ex['geom']['n_rods']} × {ex['geom']['Lr']:g} m"),
            row("ولتاژ mesh (Em)", f"{ex['mesh']['Em']:,.0f} V", hl=True),
            row("ولتاژ step (Es)", f"{ex['mesh']['Es']:,.0f} V", hl=True),
        ] + ([
            row("Rg تحلیل عددی", f"{ex['rho_eq'] * ex['num']['rg_unit']:.3f} Ω"),
            row("بیشینه ولتاژ تماس روی نقشه", f"{ex['num_stats']['touch_max']:,.0f} V"),
            row("بیشینه ولتاژ گام روی نقشه", f"{ex['num_stats']['step_max']:,.0f} V"),
            row("گرادیان پتانسیل داخل شبکه", f"{ex['num_stats']['grad_in'][0]:.1f} – {ex['num_stats']['grad_in'][1]:.1f} V/m"),
            row("گرادیان پتانسیل بیرون شبکه (بیشینه)", f"{ex['num_stats']['grad_out_max']:.1f} V/m"),
            row("مساحت ناایمن (تماس / گام)", f"{ex['num_stats']['touch_unsafe_pct']:.1f}٪ / {ex['num_stats']['step_unsafe_pct']:.1f}٪"),
        ] if ex.get("num_stats") else [])) if ex else []),
        f'<div class="rep-sec">ایمنی (وزن {inp["weight"]} kg ، زمان خطا {inp["ts"]:g} s)</div>',
        row("جریان مجاز بدن", f"{tv['i_body']:.3f} A"),
        row("ضریب لایه سطحی Cs", f"{tv['cs']:.3f}"),
        row("ولتاژ گام مجاز", f"{tv['e_step']:,.0f} V"),
        row("ولتاژ تماس مجاز", f"{tv['e_touch']:,.0f} V"),
        f'<div class="rep-verdict {vcls}">{ {"ok": "✅", "warn": "⚠️", "bad": "🚨"}[vcls] } {verdict}</div>',
    ]
    return "".join(parts)


# ============================================================
#  هندسه شبکه، ولتاژهای mesh/step (IEEE 80) و پتانسیل سطحی (تحلیل عددی)
# ============================================================
MAX_LINES = 25
ROD_MODES = {"none": "بدون میله", "corners": "فقط گوشه‌ها", "perimeter": "گوشه‌ها و پیرامون"}


def make_geometry(area, aspect, spacing, d_mm, rod_mode, rod_len, depth) -> dict:
    """هندسه شبکه مستطیلی با هادی‌های موازی و میله‌های اختیاری."""
    Lx, Ly = float(np.sqrt(area * aspect)), float(np.sqrt(area / aspect))
    want_x, want_y = int(round(Lx / spacing)) + 1, int(round(Ly / spacing)) + 1
    nx, ny = int(np.clip(want_x, 2, MAX_LINES)), int(np.clip(want_y, 2, MAX_LINES))
    xs, ys = np.linspace(0.0, Lx, nx), np.linspace(0.0, Ly, ny)
    Dx, Dy = Lx / (nx - 1), Ly / (ny - 1)
    LC = Lx * ny + Ly * nx                              # طول کل هادی‌های افقی
    if rod_mode == "corners":
        pts = [(0.0, 0.0), (Lx, 0.0), (0.0, Ly), (Lx, Ly)]
    elif rod_mode == "perimeter":
        pts = sorted({(float(x), 0.0) for x in xs} | {(float(x), Ly) for x in xs}
                     | {(0.0, float(y)) for y in ys} | {(Lx, float(y)) for y in ys})
    else:
        pts = []
    rods = np.array(pts, dtype=float).reshape(-1, 2)
    n_r = len(rods)
    return {"Lx": Lx, "Ly": Ly, "nx": nx, "ny": ny, "xs": xs, "ys": ys, "Dx": Dx, "Dy": Dy,
            "D": 0.5 * (Dx + Dy), "LC": LC, "rods": rods, "n_rods": n_r,
            "Lr": float(rod_len) if n_r else 0.0, "LR": n_r * float(rod_len),
            "LT": LC + n_r * float(rod_len), "h": max(float(depth), 0.05), "d": d_mm / 1000.0,
            "rod_mode": rod_mode, "capped": bool(want_x > MAX_LINES or want_y > MAX_LINES)}


def ieee80_mesh_step(rho: float, ig: float, g: dict) -> dict:
    """ولتاژهای mesh و step با رابطه‌های IEEE 80 (شبکه مستطیلی، خاک یکنواخت)."""
    h, d, D = g["h"], g["d"], g["D"]
    Lx, Ly, LC, LR, Lr = g["Lx"], g["Ly"], g["LC"], g["LR"], g["Lr"]
    A, Lp = Lx * Ly, 2.0 * (Lx + Ly)
    n = (2.0 * LC / Lp) * np.sqrt(Lp / (4.0 * np.sqrt(A)))      # na · nb (nc = nd = 1)
    Kh = np.sqrt(1.0 + h / 1.0)
    rods_edge = g["n_rods"] > 0 and g["rod_mode"] in ("corners", "perimeter")
    Kii = 1.0 if rods_edge else 1.0 / (2.0 * n) ** (2.0 / n)
    arg = D**2 / (16.0 * h * d) + (D + 2.0 * h) ** 2 / (8.0 * D * d) - h / (4.0 * d)
    Km = (np.log(arg) + (Kii / Kh) * np.log(8.0 / (np.pi * (2.0 * n - 1.0)))) / (2.0 * np.pi)
    Ki = 0.644 + 0.148 * n
    LM = LC + (1.55 + 1.22 * Lr / np.hypot(Lx, Ly)) * LR if rods_edge else LC + LR
    Ks = (1.0 / np.pi) * (1.0 / (2.0 * h) + 1.0 / (D + h) + (1.0 / D) * (1.0 - 0.5 ** (n - 2.0)))
    LS = 0.75 * LC + 0.85 * LR
    warns = []
    if not 0.25 <= h <= 2.5:
        warns.append(f"عمق دفن {h:.2f} m خارج از بازه معمول اعتبار رابطه‌ها (حدود ۰٫۲۵ تا ۲٫۵ متر) است.")
    if d >= 0.25 * h:
        warns.append("قطر هادی نسبت به عمق دفن بزرگ است (شرط معمول d < 0.25h برقرار نیست).")
    if D <= 2.5:
        warns.append(f"فاصله هادی‌ها ({D:.1f} m) کمتر از حدود ۲٫۵ متر است؛ رابطه‌ها در این بازه کم‌دقت می‌شوند.")
    return {"n": float(n), "Kh": float(Kh), "Kii": float(Kii), "Km": float(Km), "Ki": float(Ki), "Ks": float(Ks),
            "LM": float(LM), "LS": float(LS), "Em": float(rho * ig * Km * Ki / LM),
            "Es": float(rho * ig * Ks * Ki / LS), "warnings": warns}


def _segments(g: dict):
    """تقسیم هادی‌ها و میله‌ها به قطعات کوتاه؛ محل تقاطع‌ها همیشه مرز قطعه است."""
    h, xs, ys = g["h"], g["xs"], g["ys"]
    seg = max(2.0, g["LT"] / 700.0)
    A_, B_ = [], []

    def add(p, q):
        n = max(1, int(np.ceil(np.linalg.norm(q - p) / seg)))
        pts = p[None, :] + np.linspace(0.0, 1.0, n + 1)[:, None] * (q - p)[None, :]
        A_.append(pts[:-1]); B_.append(pts[1:])

    for y in ys:
        for x0, x1 in zip(xs[:-1], xs[1:]):
            add(np.array([x0, y, -h]), np.array([x1, y, -h]))
    for x in xs:
        for y0, y1 in zip(ys[:-1], ys[1:]):
            add(np.array([x, y0, -h]), np.array([x, y1, -h]))
    for x, y in g["rods"]:
        add(np.array([x, y, -h]), np.array([x, y, -h - g["Lr"]]))
    return np.vstack(A_), np.vstack(B_)


def _line_factor(P, A, B, a):
    """انتگرال ∫dl/r روی هر قطعه برای هر نقطه (فرم تحلیلی با asinh؛ شعاع هادی از تکینگی جلوگیری می‌کند)."""
    u = B - A
    ell = np.linalg.norm(u, axis=1)
    u = u / ell[:, None]
    w = P[:, None, :] - A[None, :, :]
    t = np.einsum("msk,sk->ms", w, u)
    r2 = np.einsum("msk,msk->ms", w, w) - t * t
    rp = np.sqrt(np.maximum(r2, a * a))
    return np.arcsinh((ell[None, :] - t) / rp) - np.arcsinh(-t / rp)


def _kernel(P, A, B, a):
    flip = np.array([1.0, 1.0, -1.0])           # تصویر نسبت به سطح زمین (z = 0)
    return _line_factor(P, A, B, a) + _line_factor(P, A * flip, B * flip, a)


@st.cache_data(show_spinner=False)
def solve_grid_potential(area, aspect, spacing, d_mm, rod_mode, rod_len, depth, resolution=240) -> dict:
    """حل نشت جریان با شرط پتانسیل ثابت روی هادی‌ها (collocation) و محاسبه پتانسیل سطح.
    ρ حذف می‌شود: Rg = ρ·rg_unit و frac کسری از GPR است."""
    g = make_geometry(area, aspect, spacing, d_mm, rod_mode, rod_len, depth)
    A, B = _segments(g)
    C = 0.5 * (A + B)
    ell = np.linalg.norm(B - A, axis=1)
    a = g["d"] / 2.0
    R = _kernel(C, A, B, a) / (4.0 * np.pi * ell[None, :])
    cur = np.linalg.solve(R, np.ones(len(ell)))
    w = cur / (4.0 * np.pi * ell)
    resolution = int(min(resolution, max(90.0, np.sqrt(4.0e7 / len(ell)))))     # سقف زمان و حافظه
    margin = max(8.0, 0.15 * max(g["Lx"], g["Ly"]))
    step = max(0.5, (max(g["Lx"], g["Ly"]) + 2 * margin) / resolution)
    xs = np.linspace(-margin, g["Lx"] + margin, int(round((g["Lx"] + 2 * margin) / step)) + 1)
    ys = np.linspace(-margin, g["Ly"] + margin, int(round((g["Ly"] + 2 * margin) / step)) + 1)
    X, Y = np.meshgrid(xs, ys)
    P = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
    frac = np.empty(len(P))
    for i in range(0, len(P), 1500):
        frac[i:i + 1500] = _kernel(P[i:i + 1500], A, B, a) @ w
    return {"rg_unit": float(1.0 / cur.sum()), "xs": xs, "ys": ys, "frac": frac.reshape(Y.shape),
            "dx": float(xs[1] - xs[0]), "dy": float(ys[1] - ys[0]), "n_seg": int(len(ell))}


def potential_stats(num: dict, g: dict, rho: float, ig: float, tv: dict) -> dict:
    """پتانسیل سطح، ولتاژ تماس، ولتاژ گام و گرادیان (V/m) از نقشه عددی."""
    gpr = ig * rho * num["rg_unit"]
    V = num["frac"] * gpr
    touch = gpr - V
    gy, gx = np.gradient(V, num["dy"], num["dx"])
    grad = np.hypot(gx, gy)                      # V/m ؛ ولتاژ گام ≈ گرادیان × ۱ متر
    X, Y = np.meshgrid(num["xs"], num["ys"])
    inside = (X >= 0) & (X <= g["Lx"]) & (Y >= 0) & (Y <= g["Ly"])
    near = (X >= -3) & (X <= g["Lx"] + 3) & (Y >= -3) & (Y <= g["Ly"] + 3)
    outside = near & ~inside

    def where_max(Z, m):
        k = np.argmax(np.where(m, Z, -np.inf))
        return float(Z.ravel()[k]), (float(X.ravel()[k]), float(Y.ravel()[k]))

    t_max, t_at = where_max(touch, inside)
    s_max, s_at = where_max(grad, near)
    return {"gpr": gpr, "V": V, "touch": touch, "step": grad, "grad": grad, "inside": inside, "near": near,
            "touch_max": t_max, "touch_at": t_at, "step_max": s_max, "step_at": s_at,
            "grad_in": (float(grad[inside].min()), float(grad[inside].max())),
            "grad_out_max": float(grad[outside].max()) if outside.any() else float("nan"),
            "v_min_pct": float(num["frac"][inside].min() * 100.0),
            "touch_unsafe_pct": float((touch[inside] > tv["e_touch"]).mean() * 100.0),
            "step_unsafe_pct": float((grad[near] > tv["e_step"]).mean() * 100.0)}


# ============================================================
#  کنترل کفایت داده و توصیه مهندسی (قاعده‌محور؛ بر پایه IEEE 80 و IEEE 81)
# ============================================================
SEV = {"bad": 0, "warn": 1, "ok": 2, "info": 3}


def data_sufficiency(a: np.ndarray, rho: np.ndarray, fit: dict, area: float) -> list:
    """کنترل کفایت داده‌های Wenner. خروجی: فهرست (سطح، متن) مرتب‌شده بر اساس شدت."""
    issues = []
    h, a_max, a_min = fit["h"], float(a.max()), float(a.min())
    r_eq = float(np.sqrt(area / np.pi))
    if a_max < 2 * h:
        issues.append(("bad", f"حداکثر فاصله اندازه‌گیری ({a_max:.1f} m) کمتر از دو برابر ضخامت لایه اول ({h:.1f} m) است؛ "
                              "لایه دوم عملاً دیده نشده و ρ₂ و h قابل اتکا نیستند."))
    elif a_max < 3 * h:
        issues.append(("warn", f"حداکثر فاصله ({a_max:.1f} m) کمتر از سه برابر ضخامت لایه اول ({h:.1f} m) است؛ "
                               "برای تخمین مطمئن ρ₂ فواصل بزرگ‌تر هم اندازه بگیرید."))
    if a_max < r_eq:
        issues.append(("bad" if a_max < 0.5 * r_eq else "warn",
                       f"ρ معادل در شعاع معادل شبکه ({r_eq:.0f} m) محاسبه می‌شود که بیرون از بازه اندازه‌گیری "
                       f"(حداکثر {a_max:.0f} m) است؛ یعنی مقدار برون‌یابی شده و Rg و GPR به آن وابسته‌اند."))
    if a.size < 6:
        issues.append(("warn", f"تعداد نقاط اندازه‌گیری ({a.size}) کم است؛ معمولاً حداقل ۶ تا ۸ فاصله، با گام تقریباً لگاریتمی، توصیه می‌شود."))
    if a_max / a_min < 10:
        issues.append(("warn", "نسبت بزرگ‌ترین به کوچک‌ترین فاصله کمتر از ۱۰ است؛ بازه فواصل باید دست‌کم یک دهه را بپوشاند."))
    if a.size >= 3:
        slope = float(np.polyfit(np.log(a[-3:]), np.log(rho[-3:]), 1)[0])
        if abs(slope) > 0.15:
            issues.append(("warn", f"منحنی در فواصل بزرگ هنوز به مقدار ثابتی نرسیده (شیب لگاریتمی {slope:+.2f})؛ "
                                   "احتمالاً لایه عمقی کامل دیده نشده است."))
    if fit["max_err_pct"] > 15 or fit["rmse_pct"] > 10:
        issues.append(("warn", f"پراکندگی یا ناسازگاری داده‌ها زیاد است (RMSE = {fit['rmse_pct']:.1f}٪ ، بیشینه خطا = {fit['max_err_pct']:.1f}٪). "
                               "اجسام فلزی مدفون یا خطای اندازه‌گیری را بررسی کنید و اندازه‌گیری را در راستای عمود تکرار کنید."))
    if fit["dof"] >= 2 and max(fit["rel_unc"]) > 0.5:
        issues.append(("warn", f"عدم‌قطعیت آماری یکی از پارامترها بیش از ±۵۰٪ است (بازه ۹۵٪: ρ₂ از {fit['ci_95'][1][0]:,.0f} تا {fit['ci_95'][1][1]:,.0f} Ω·m)."))
    elif fit["dof"] < 2:
        issues.append(("warn", "تعداد داده‌ها تقریباً برابر تعداد پارامترهاست؛ بازه اطمینان قابل محاسبه یا قابل اتکا نیست."))
    return sorted(issues, key=lambda t: SEV[t[0]])


def _scale_needed(rho_eq, area, length, depth, ig, e_touch, s_max=3.0):
    """کوچک‌ترین ضریب بزرگ‌نمایی ابعاد خطی شبکه که شرط GPR ≤ Etouch را برآورده کند (ρ ثابت)."""
    for sc in np.linspace(1.0, s_max, 121):
        if ig * grid_resistance(rho_eq, area * sc**2, length * sc**2, depth)["sverak"] <= e_touch:
            return float(sc)
    return None


def engineering_advice(out: dict) -> dict:
    fit, tv, rg, inp = out["fit"], out["tv"], out["rg"], out["inp"]
    a, rho, gpr = out["a"], out["rho"], out["gpr"]
    ig, rg_d, e_touch = inp["ig"], out["rg_design"], tv["e_touch"]
    ratio = gpr / e_touch
    real = inp.get("real_mode", True)
    k, rho1, rho2, h = fit["k"], fit["rho1"], fit["rho2"], fit["h"]
    cards = []

    def add(lvl, title, body, acts=(), ref=""):
        cards.append(dict(lvl=lvl, title=title, body=body, acts=list(acts), ref=ref))

    # --- ۱) وضعیت ایمنی (GPR، Em و Es) ---
    m, geom, stt = out["mesh"], out["geom"], out.get("num_stats")
    em, es, es_tol = m["Em"], m["Es"], tv["e_step"]
    mesh_ok = em <= e_touch and es <= es_tol
    mesh_fac = max(em / e_touch, es / es_tol)
    if ratio <= 1.0:
        acts = []
        if ratio > 0.8:
            acts.append("حاشیه ایمنی کمتر از ۲۰٪ است؛ عدم‌قطعیت ρ خاک و Ig می‌تواند آن را از بین ببرد. "
                        "نتیجه را با مقاومت ویژه فصل خشک و Ig بیشینه دوباره بسنجید.")
        add("ok" if ratio <= 0.8 else "warn", "وضعیت ایمنی: GPR در برابر ولتاژ تماس مجاز",
            f"GPR برابر {gpr:,.0f} V و ولتاژ تماس مجاز برابر {e_touch:,.0f} V است ({ratio * 100:.0f}٪ حد مجاز). "
            "طبق روند طراحی IEEE 80، وقتی GPR از ولتاژ تماس مجاز کمتر باشد، محاسبه ولتاژهای mesh و step الزامی نیست "
            "(برای اطلاع، مقادیر آن‌ها در کارت بعدی آمده است).",
            acts, "IEEE 80")
    elif mesh_ok:
        add("warn", "وضعیت ایمنی: GPR در برابر ولتاژ تماس مجاز",
            f"GPR برابر {gpr:,.0f} V و بیش از ولتاژ تماس مجاز ({e_touch:,.0f} V) است ({ratio * 100:.0f}٪)، اما ولتاژهای mesh و step "
            f"(Em = {em:,.0f} V و Es = {es:,.0f} V) در حد مجازند. طبق روند IEEE 80 طرح از نظر تماس و گام قابل قبول است؛",
            ["GPR بالا یعنی ولتاژ منتقل‌شده به حصار، لوله‌ها، ریل و کابل‌های مخابراتی باید کنترل شود.",
             "تجهیزات حفاظتی و مخابراتی متصل به مناطق دور باید برای این GPR عایق‌بندی یا ایزوله شوند."], "IEEE 80")
    else:
        add("bad", "وضعیت ایمنی: ولتاژهای mesh و step",
            f"GPR برابر {gpr:,.0f} V است و ولتاژهای محاسبه‌شده از حد مجاز عبور کرده‌اند: Em = {em:,.0f} V در برابر {e_touch:,.0f} V "
            f"({em / e_touch * 100:.0f}٪) و Es = {es:,.0f} V در برابر {es_tol:,.0f} V ({es / es_tol * 100:.0f}٪). "
            "طرح باید اصلاح شود (هندسه شبکه، میله‌ها، لایه سطحی، Ig یا زمان قطع).", [], "IEEE 80")

    # --- ۱-ب) ولتاژهای mesh و step ---
    acts = list(m["warnings"])
    if geom["capped"]:
        acts.append("تعداد هادی‌ها به سقف ۲۵ در هر جهت محدود شده است؛ فاصله واقعی بزرگ‌تر از D واردشده است.")
    if not mesh_ok or mesh_fac > 0.85:
        for label, d2, rm in [("فشرده‌تر کردن مش (D = ۰٫۷۵ برابر فعلی)", inp["spacing"] * 0.75, inp["rod_mode"]),
                              ("افزودن میله در گوشه‌ها و پیرامون", inp["spacing"], "perimeter"),
                              ("هر دو اقدام", inp["spacing"] * 0.75, "perimeter")]:
            if label.startswith("افزودن") and inp["rod_mode"] == "perimeter":
                continue
            g2 = make_geometry(inp["area"], inp["aspect"], d2, inp["d_mm"], rm, inp["rod_len"], inp["depth"])
            m2 = ieee80_mesh_step(out["rho_eq"], ig, g2)
            acts.append(f"{label}: Em = {m2['Em']:,.0f} V ، Es = {m2['Es']:,.0f} V (D ≈ {g2['D']:.1f} m ، طول کل هادی ≈ {g2['LT']:,.0f} m).")
        acts.append("لایه سطحی سنگ‌ریزه (یا ضخیم‌تر کردن آن) و کاهش Ig یا زمان قطع، حدود مجاز و ولتاژها را هم‌زمان بهبود می‌دهند.")
    add("ok" if mesh_ok and mesh_fac <= 0.85 else "warn" if mesh_ok else "bad",
        "ولتاژهای mesh و step با رابطه‌های IEEE 80",
        f"Em = {em:,.0f} V ({em / e_touch * 100:.0f}٪ ولتاژ تماس مجاز) و Es = {es:,.0f} V ({es / es_tol * 100:.0f}٪ ولتاژ گام مجاز). "
        f"هندسه: {geom['Lx']:.0f} × {geom['Ly']:.0f} m ، {geom['nx']}×{geom['ny']} هادی با D ≈ {geom['D']:.1f} m ، "
        f"{geom['n_rods']} میله ، طول کل هادی {geom['LT']:,.0f} m (n = {m['n']:.1f} ، Km = {m['Km']:.3f} ، Ki = {m['Ki']:.3f}).",
        acts, "IEEE 80")

    # --- ۱-ج) نقشه پتانسیل و گرادیان (تحلیل عددی) ---
    if stt:
        rg_num = out["rho_eq"] * out["num"]["rg_unit"]
        diff = (rg_num / rg_d - 1.0) * 100.0
        lo, hi = stt["grad_in"]
        acts = []
        tu, su = stt["touch_unsafe_pct"], stt["step_unsafe_pct"]
        if tu > 0:
            acts.append(f"{tu:.1f}٪ از مساحت شبکه ولتاژ تماس بالاتر از حد مجاز دارد؛ نقطه بیشینه حدود (x = {stt['touch_at'][0]:.1f} ، y = {stt['touch_at'][1]:.1f}) متر است.")
        if su > 0:
            acts.append(f"{su:.1f}٪ از ناحیه اطراف و داخل شبکه ولتاژ گام بالاتر از حد مجاز دارد؛ بیشینه حدود (x = {stt['step_at'][0]:.1f} ، y = {stt['step_at'][1]:.1f}) متر است.")
        if tu > 0 or su > 0:
            acts.append("گرادیان‌های زیاد معمولاً در گوشه‌ها و لبه‌های شبکه‌اند: میله در گوشه‌ها و پیرامون، هادی حلقوی اضافه در اطراف، "
                        "یا مش فشرده‌تر در محیط شبکه گرادیان را کم می‌کند.")
            acts.append("لایه سطحی سنگ‌ریزه را حدود یک متر فراتر از حصار ادامه دهید و حصار را به شبکه وصل یا هادی کنترل گرادیان بیرون حصار قرار دهید.")
        if abs(diff) > 20:
            acts.append(f"Rg عددی با Sverak بیش از ۲۰٪ اختلاف دارد ({diff:+.0f}٪)؛ برای طرح نهایی با نرم‌افزار تخصصی (مثل CYMGRD یا CDEGS) راستی‌آزمایی کنید.")
        if out["num"]["dx"] > 0.7:
            acts.append(f"گام نقشه {out['num']['dx']:.2f} m است؛ در شبکه‌های بزرگ بیشینه ولتاژ گام ممکن است کم‌برآورد شود و مقدار IEEE 80 مبناست.")
        acts.append("نقشه با خاک یکنواخت (ρ معادل) محاسبه شده و ولتاژ تماس نقطه‌ای (GPR منهای پتانسیل سطح) است؛ فاصله ۱ متری دست از ساختار لحاظ نشده است.")
        lvl = "ok" if tu == 0 and su == 0 else ("bad" if not mesh_ok else "warn")
        add(lvl, "نقشه پتانسیل سطح و محدوده گرادیان (تحلیل عددی)",
            f"Rg عددی = {rg_num:.2f} Ω ({diff:+.0f}٪ نسبت به Sverak). بیشینه ولتاژ تماس نقشه {stt['touch_max']:,.0f} V و بیشینه ولتاژ گام {stt['step_max']:,.0f} V است. "
            f"گرادیان پتانسیل داخل شبکه بین {lo:.1f} و {hi:.1f} V/m و بیرون شبکه تا {stt['grad_out_max']:.1f} V/m است؛ "
            f"کمترین پتانسیل سطح داخل شبکه {stt['v_min_pct']:.0f}٪ GPR است.",
            acts, "IEEE 80")

    # --- ۲) راهکارهای عملی ---
    if ratio > 0.8:
        acts = []
        rg_req, ig_max = e_touch / ig, e_touch / rg_d
        acts.append(f"هدف عددی: برای اینکه GPR از ولتاژ تماس مجاز بیشتر نشود، یکی از این‌ها لازم است: Rg ≤ {rg_req:.2f} Ω "
                    f"(اکنون {rg_d:.2f} Ω) یا Ig ≤ {ig_max:,.0f} A (اکنون {ig:,.0f} A).")
        sc = _scale_needed(out["rho_eq"], inp["area"], inp["length"], inp["depth"], ig, e_touch)
        if sc is not None and sc > 1.0:
            acts.append(f"افزایش ابعاد شبکه: با بزرگ‌تر شدن ابعاد خطی به حدود {sc:.2f} برابر (مساحت ≈ {inp['area'] * sc**2:,.0f} m² "
                        f"و طول هادی ≈ {inp['length'] * sc**2:,.0f} m، با فرض ثابت ماندن ρ) شرط برقرار می‌شود.")
        elif sc is None:
            acts.append("افزایش مساحت و طول هادی به‌تنهایی (تا ۳ برابر ابعاد خطی) کافی نیست؛ ترکیب چند راهکار لازم است.")
        acts.append("کاهش Ig: مسیرهای بازگشت فلزی جریان خطا (سیم محافظ خطوط انتقال، شیلد کابل‌ها، نوترال) ضریب تقسیم جریان Sf را کم "
                    "و Ig را کوچک می‌کنند.")
        t2 = max(inp["ts"] / 2.0, 0.05)
        rho_surf = inp["rho_s"] if inp["hs"] > 0 else rho1
        tv2 = tolerable_voltages(rho1, rho_surf, inp["hs"], inp["weight"], t2)
        acts.append(f"کاهش زمان قطع خطا از {inp['ts']:g} به {t2:g} ثانیه (حفاظت سریع‌تر) ولتاژ تماس مجاز را از {e_touch:,.0f} به "
                    f"{tv2['e_touch']:,.0f} V می‌رساند.")
        if inp["hs"] < 0.15:
            tv3 = tolerable_voltages(rho1, inp["rho_s"], 0.15, inp["weight"], inp["ts"])
            if tv3["e_touch"] > e_touch * 1.05:
                acts.append(f"لایه سطحی سنگ‌ریزه با ضخامت حدود ۰٫۱۵ متر (با ρs = {inp['rho_s']:,.0f} Ω·m) ولتاژ تماس مجاز را به "
                            f"{tv3['e_touch']:,.0f} V می‌رساند. ρs را برای سنگ‌ریزه خیس وارد کنید، نه خشک.")
        add("info", "راهکارهای عملی برای کاهش GPR و افزایش حاشیه ایمنی",
            "اثر هر اقدام جدا محاسبه شده است؛ در عمل معمولاً ترکیبی از آن‌ها لازم می‌شود.", acts, "IEEE 80")

    # --- ۳) ساختار خاک و الکترود ---
    if abs(k) < 0.1 or fit["rmse_uniform_pct"] <= fit["rmse_pct"] * 1.2:
        add("info", "ساختار خاک و انتخاب الکترود",
            f"خاک تقریباً یکنواخت است (K = {k:.2f}). مدل دولایه اطلاعات تازه‌ای نمی‌دهد و ρ₂ و h قابل اتکا نیستند.",
            ["برای کاهش Rg، افزایش مساحت پوشش و تراکم هادی‌ها مؤثرترین راه است؛ میله‌های عمودی اثر محدودی دارند."], "IEEE 80")
    elif k < 0:
        add("info", "ساختار خاک و انتخاب الکترود",
            f"لایه عمقی رساناتر است (ρ₂ = {rho2:,.0f} در برابر ρ₁ = {rho1:,.0f} Ω·m ، K = {k:.2f}).",
            [f"میله‌های زمین که از لایه اول عبور کنند (طول بیش از h ≈ {h:.1f} m ، مثلاً {h + 1:.1f} تا {h + 3:.1f} m) "
             "مقاومت شبکه و ولتاژهای تماس را مؤثرتر کاهش می‌دهند.",
             "اگر حفاری عمیق شدنی است، الکترودهای عمیق یا چاه زمین را بررسی کنید.",
             "لایه سطحی سنگ‌ریزه همچنان برای ولتاژ گام و تماس مفید است."], "IEEE 80")
    else:
        add("info", "ساختار خاک و انتخاب الکترود",
            f"لایه عمقی مقاوم‌تر است (ρ₂ = {rho2:,.0f} در برابر ρ₁ = {rho1:,.0f} Ω·m ، K = {k:.2f}). جریان بیشتر در لایه سطحی می‌ماند.",
            ["میله‌های بلند اثر کمی دارند؛ افزایش سطح پوشش شبکه و تراکم هادی‌ها در لایه اول مؤثرتر است.",
             "مواد بهبوددهنده خاک (مثل بنتونیت یا بتن رسانا) اطراف هادی‌ها را بررسی کنید.",
             "اتصال به الکترود دورتر در منطقه مرطوب‌تر یا کم‌مقاومت‌تر ممکن است؛ ولتاژ منتقل‌شده را کنترل کنید.",
             "ρ معادل به ابعاد شبکه وابسته است؛ برای طرح نهایی از روش Sunde یا نرم‌افزار تخصصی استفاده کنید."], "IEEE 80")

    # --- ۴) مقاومت شبکه و حساسیت به ρ معادل ---
    bounds = {}
    for name, rho_x in (("ρ معادل مدل", out["rho_eq"]), ("فقط ρ₁", rho1), ("فقط ρ₂", rho2)):
        bounds[name] = ig * grid_resistance(rho_x, inp["area"], inp["length"], inp["depth"])["sverak"]
    g_lo, g_hi = min(bounds.values()), max(bounds.values())
    spread = g_hi / g_lo
    if rg_d <= 1.0:
        rg_txt = "در محدوده‌ای است که معمولاً برای پست‌های بزرگ هدف‌گذاری می‌شود"
    elif rg_d <= 5.0:
        rg_txt = "برای پست‌های کوچک‌تر توزیع معمولاً قابل قبول است"
    else:
        rg_txt = "بالاتر از محدوده معمول است"
    add("warn" if spread > 1.3 or rg_d > 5 else "info", "مقاومت شبکه و حساسیت به ρ معادل",
        f"Rg طراحی (Sverak) برابر {rg_d:.2f} Ω است و {rg_txt} (راهنمای عملی، نه معیار پذیرش؛ معیار اصلی ایمنی ولتاژهای تماس و گام است). "
        f"اگر خاک یکنواخت با ρ₁ یا ρ₂ فرض شود، GPR بین {g_lo:,.0f} و {g_hi:,.0f} V تغییر می‌کند.",
        (["دامنه تغییر زیاد است؛ پیش از طرح نهایی ρ معادل را با روش Sunde یا نرم‌افزار تحلیل زمین دقیق‌تر کنید."]
         if spread > 1.3 else []), "IEEE 80")

    # --- ۵) کیفیت داده خاک (فقط حالت پروژه واقعی) ---
    issues = data_sufficiency(a, rho, fit, inp["area"]) if real else []
    if real:
        if issues:
            add(issues[0][0], "کیفیت و کفایت داده‌های اندازه‌گیری خاک",
                f"{len(issues)} مورد در داده‌ها یافت شد که بر اطمینان نتایج اثر می‌گذارد.",
                [t for _, t in issues], "IEEE 81")
        else:
            add("ok", "کیفیت و کفایت داده‌های اندازه‌گیری خاک",
                "پوشش فواصل، تعداد نقاط، هموار بودن منحنی و عدم‌قطعیت پارامترها قابل قبول است.", [], "IEEE 81")
        add("info", "نکات میدانی برای اندازه‌گیری مطمئن (Wenner)", "رعایت این موارد کیفیت داده را بالا می‌برد:",
            ["حداقل دو مسیر عمود بر هم اندازه‌گیری و نتایج مقایسه شود؛ اختلاف زیاد یعنی ناهمگنی جانبی خاک.",
             "مسیر آزمون با لوله، ریل، کابل یا حصار فلزی مدفون موازی نباشد و از آن‌ها دور باشد.",
             "عمق فرورفتن میله‌ها نسبت به فاصله a کوچک بماند (معمولاً کمتر از یک‌بیستم a)، تا فرمول ρ = 2πaR معتبر باشد.",
             "طراحی را بر پایه شرایط بدتر (خاک خشک و مقاومت ویژه بالاتر) انجام دهید یا اندازه‌گیری را در فصل خشک تکرار کنید.",
             "از دستگاه کالیبره با فرکانس غیر از هارمونیک‌های 50 Hz استفاده و سیم‌ها را برای حذف تزویج القایی از هم دور کنید."], "IEEE 81")
    else:
        add("info", "حالت آموزشی",
            "کنترل کفایت داده‌های اندازه‌گیری غیرفعال است. برای پروژه واقعی گزینه «حالت پروژه واقعی» را در پنل کناری فعال کنید.",
            [], "")

    # --- ۶) فرضیات ایمنی ---
    acts = [f"Ig باید با اعمال ضریب تقسیم جریان (Sf) و ضریب decrement (Df) از جریان خطای کل به دست آمده باشد؛ مقدار واردشده: {ig:,.0f} A.",
            f"ρs لایه سطحی را برای حالت خیس وارد کنید (مقدار فعلی {inp['rho_s']:,.0f} Ω·m).",
            "وزن مبنا 50 kg محافظه‌کارانه‌تر از 70 kg است؛ اگر حضور افراد سبک‌تر محتمل است آن را انتخاب کنید."]
    lvl = "info"
    if not (0.03 <= inp["ts"] <= 3.0):
        acts.insert(0, f"زمان قطع {inp['ts']:g} s خارج از بازه اعتبار فرمول جریان مجاز بدن (حدود ۰٫۰۳ تا ۳ ثانیه) است؛ نتیجه را با احتیاط ببینید.")
        lvl = "warn"
    add(lvl, "فرضیات ورودی ایمنی", "درستی این ورودی‌ها مستقیماً حدود مجاز و GPR را تعیین می‌کند:", acts, "IEEE 80")

    # --- ۷) پس از اجرا ---
    diag = float(np.sqrt(2 * inp["area"]))
    add("info", "اقدامات پس از اجرا و بهره‌برداری",
        "طراحی با محاسبه تمام نمی‌شود؛ مقدار اجراشده باید راستی‌آزمایی شود:",
        [f"پس از ساخت، Rg را با روش افت پتانسیل (fall-of-potential) اندازه بگیرید و با مقدار طراحی ({rg_d:.2f} Ω) مقایسه کنید. "
         f"پروب جریان باید چند برابر بزرگ‌ترین بُعد شبکه دور باشد (معمولاً ۵ تا ۶ برابر یا بیشتر، یعنی برای قطر ≈ {diag:.0f} m حدود {5 * diag:,.0f} m).",
         "ولتاژهای تماس و گام را در نقاط بحرانی (کنار حصار و تجهیزات) با تزریق جریان کنترل کنید.",
         "ظرفیت حرارتی هادی‌ها (رابطه Onderdonk) و استحکام مکانیکی اتصالات را برای Ig و زمان قطع بررسی کنید.",
         "ولتاژ منتقل‌شده به حصار، لوله‌ها و کابل‌های مخابراتی را کنترل کنید و اتصالات را به‌صورت دوره‌ای بازرسی کنید."],
        "IEEE 80 / IEEE 81")

    cards.sort(key=lambda c: SEV[c["lvl"]])

    # --- خلاصه ---
    if ratio <= 1:
        s1 = ("ok" if ratio <= 0.8 else "warn", f"از نظر GPR طرح قابل قبول است ({ratio * 100:.0f}٪ ولتاژ تماس مجاز).")
        priority = "اجرا و راستی‌آزمایی میدانی مقدار Rg"
    elif mesh_ok:
        s1 = ("warn", f"GPR ({gpr:,.0f} V) از ولتاژ تماس مجاز ({e_touch:,.0f} V) بیشتر است، ولی Em ({em:,.0f} V) و Es ({es:,.0f} V) در حد مجازند.")
        priority = "کنترل ولتاژ منتقل‌شده و راستی‌آزمایی میدانی"
    else:
        s1 = ("bad", f"Em ({em:,.0f} V) یا Es ({es:,.0f} V) از حد مجاز ({e_touch:,.0f} / {es_tol:,.0f} V) بیشتر است؛ طرح باید اصلاح شود.")
        priority = "اصلاح هندسه شبکه (کاهش D، میله‌ها)، لایه سطحی یا کاهش Ig و زمان قطع"
    s2 = ("info", f"خاک دولایه: ρ₁ = {rho1:,.0f} Ω·m ، ρ₂ = {rho2:,.0f} Ω·m ، h = {h:.2f} m (RMSE = {fit['rmse_pct']:.1f}٪).")
    extra = []
    if stt:
        extra.append(("ok" if stt["touch_unsafe_pct"] == 0 and stt["step_unsafe_pct"] == 0 else "warn",
                      f"نقشه عددی: ناحیه ناایمن تماس {stt['touch_unsafe_pct']:.1f}٪ و گام {stt['step_unsafe_pct']:.1f}٪ ؛ "
                      f"گرادیان داخل شبکه {stt['grad_in'][0]:.1f}–{stt['grad_in'][1]:.1f} V/m."))
    if real:
        worst = issues[0][0] if issues else "ok"
        conf = "پایین" if (worst == "bad" or len(issues) >= 3) else ("متوسط" if issues else "بالا")
        if worst == "bad":
            priority = "تکمیل اندازه‌گیری خاک (فواصل بزرگ‌تر) پیش از هر تصمیم طراحی"
        s3 = ({"پایین": "bad", "متوسط": "warn", "بالا": "ok"}[conf], f"اطمینان به داده‌های خاک: {conf}.")
    else:
        s3 = ("info", "حالت آموزشی: کنترل کفایت داده غیرفعال است.")
    return dict(summary=[s1, s2] + extra + [s3], priority=priority, cards=cards, real=real)


_ICON = {"ok": "✅", "warn": "⚠️", "bad": "🚨", "info": "ℹ️"}


def advice_html(adv: dict) -> str:
    """نمایش توصیه‌ها به‌صورت HTML راست‌به‌چپ (بدون لیست Markdown)."""
    parts = ['<div class="adv-sum"><div class="adv-sum-t">خلاصه توصیه</div>']
    for lvl, text in adv["summary"]:
        parts.append(f'<div class="adv-li"><span>{_ICON[lvl]}</span><span>{text}</span></div>')
    parts.append(f'<div class="adv-prio"><b>اولویت اول:</b> {adv["priority"]}</div></div>')
    for c in adv["cards"]:
        ref = f'<span class="adv-ref">{c["ref"]}</span>' if c["ref"] else ""
        acts = "".join(f'<div class="adv-li"><span>▸</span><span>{t}</span></div>' for t in c["acts"])
        parts.append(f'<div class="adv {c["lvl"]}"><div class="adv-h"><span>{_ICON[c["lvl"]]} {c["title"]}</span>{ref}</div>'
                     f'<div class="adv-b">{c["body"]}</div>{acts}</div>')
    parts.append('<div class="adv-foot">این توصیه‌ها قاعده‌محور و خودکارند و جایگزین محاسبات کامل IEEE 80 (Em و Es) '
                 'و بازبینی مهندس مسئول نیستند.</div>')
    return "".join(parts)


def advice_markdown(adv: dict) -> str:
    lines = ["# توصیه مهندسی", "", "## خلاصه"]
    lines += [f"- {_ICON[l]} {t}" for l, t in adv["summary"]]
    lines += [f"- **اولویت اول:** {adv['priority']}", ""]
    for c in adv["cards"]:
        lines += [f"## {_ICON[c['lvl']]} {c['title']}" + (f" ({c['ref']})" if c["ref"] else ""), c["body"], ""]
        lines += [f"- {t}" for t in c["acts"]] + [""]
    lines.append("> توصیه‌ها قاعده‌محور و خودکارند و جایگزین بازبینی مهندس مسئول نیستند.")
    return "\n".join(lines)


# ============================================================
#  اجرای تحلیل
# ============================================================
def run_analysis(mode_R, a_text, v_text, area, aspect, spacing, d_mm, rod_mode, rod_len, depth, method,
                 ig, ts, weight, use_surface, rho_s, hs, real_mode=True) -> dict:
    a_raw = parse_series(a_text, "فاصله a")
    v_raw = parse_series(v_text, "مقادیر اندازه‌گیری")
    if mode_R and a_raw.size == v_raw.size:
        v_raw = 2.0 * np.pi * a_raw * v_raw          # ρa = 2π·a·R
    a_vals, rho_vals = validate_data(a_raw, v_raw)
    fit = fit_two_layer(tuple(a_vals), tuple(rho_vals))

    geom = make_geometry(area, aspect, spacing, d_mm, rod_mode, rod_len, depth)
    rho_eq = equivalent_resistivity(method, fit, area)
    rg = grid_resistance(rho_eq, area, geom["LT"], depth)
    gpr = ig * rg["sverak"]
    mesh = ieee80_mesh_step(rho_eq, ig, geom)
    hs_eff = hs if use_surface else 0.0
    tv = tolerable_voltages(fit["rho1"], rho_s if use_surface else fit["rho1"], hs_eff, weight, ts)
    inp = dict(area=area, aspect=aspect, spacing=spacing, d_mm=d_mm, rod_mode=rod_mode, rod_len=rod_len,
               length=geom["LT"], depth=depth, ig=ig, ts=ts, weight=weight,
               hs=hs_eff, rho_s=rho_s, real_mode=real_mode)
    try:                                                   # نقشه پتانسیل (اختیاری؛ خطا کل برنامه را نمی‌شکند)
        num = solve_grid_potential(float(area), float(aspect), float(spacing), float(d_mm),
                                   rod_mode, float(rod_len), float(depth))
        num_stats = potential_stats(num, geom, rho_eq, ig, tv)
    except (np.linalg.LinAlgError, MemoryError, ValueError):
        num, num_stats = None, None
    return dict(a=a_vals, rho=rho_vals, fit=fit, rho_eq=rho_eq, rg=rg, geom=geom, mesh=mesh,
                num=num, num_stats=num_stats, rg_design=rg["sverak"], gpr=gpr, tv=tv, inp=inp)


# ============================================================
#  صفحه و پنل ورودی
# ============================================================
st.set_page_config(page_title="Grounding Designer", page_icon="⚡", layout="wide")
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown(H("""<div class="brand"><div class="logo">⚡</div><div>
        <div class="brand-t">Grounding Designer</div><div class="brand-s">IEEE Std 80</div></div></div>"""),
                unsafe_allow_html=True)
    with st.form("inputs"):
        st.markdown('<div class="side-t">📊 داده‌های Wenner</div>', unsafe_allow_html=True)
        mode = st.radio("نوع داده اندازه‌گیری", ["مقاومت ظاهری ρa (Ω·m)", "مقاومت قرائت‌شده R (Ω)"])
        a_text = st.text_input("فاصله الکترودها a (m)", DEFAULTS["a_text"])
        v_text = st.text_input("مقادیر اندازه‌گیری", DEFAULTS["v_text"])

        st.markdown('<div class="side-t">🏭 شبکه زمین</div>', unsafe_allow_html=True)
        area = st.number_input("مساحت (m²)", min_value=1.0, value=DEFAULTS["area"])
        aspect = st.number_input("نسبت طول به عرض زمین (Lx/Ly)", min_value=0.2, max_value=5.0,
                                 value=DEFAULTS["aspect"], step=0.1)
        spacing = st.number_input("فاصله هادی‌ها D (m)", min_value=1.0, value=DEFAULTS["spacing"])
        d_mm = st.number_input("قطر هادی (mm)", min_value=4.0, value=DEFAULTS["d_mm"])
        depth = st.number_input("عمق دفن شبکه (m)", min_value=0.0, value=DEFAULTS["depth"])
        rod_mode = st.selectbox("میله‌های زمین", list(ROD_MODES), format_func=ROD_MODES.get)
        rod_len = st.number_input("طول هر میله (m)", min_value=0.5, value=DEFAULTS["rod_len"])
        method = st.selectbox(
            "روش ρ معادل", ["apparent", "rho1", "mean"],
            format_func={"apparent": "ρa مدل در شعاع معادل (پیشنهادی)",
                         "rho1": "فقط ρ₁ (خاک یکنواخت)",
                         "mean": "میانگین ρ₁ و ρ₂ (غیرمهندسی)"}.get)

        st.markdown('<div class="side-t">⚡ خطا و ایمنی</div>', unsafe_allow_html=True)
        ig = st.number_input("جریان شبکه Ig (A)", min_value=1.0, value=DEFAULTS["ig"])
        ts = st.number_input("زمان قطع خطا (s)", min_value=0.05, value=DEFAULTS["ts"])
        weight = st.selectbox("وزن مبنا (kg)", [50, 70], index=1)
        use_surface = st.checkbox("لایه سطحی (شن/سنگ‌ریزه) دارد", value=DEFAULTS["use_surface"])
        rho_s = st.number_input("ρ لایه سطحی (Ω·m)", min_value=1.0, value=DEFAULTS["rho_s"])
        hs = st.number_input("ضخامت لایه سطحی (m)", min_value=0.0, value=DEFAULTS["hs"])

        st.markdown('<div class="side-t">⚙️ حالت استفاده</div>', unsafe_allow_html=True)
        real_mode = st.checkbox("حالت پروژه واقعی (کنترل کفایت داده)", value=DEFAULTS["real_mode"],
                                help="فعال: هشدارهای عمق نفوذ و کفایت داده نمایش داده می‌شود. "
                                     "غیرفعال: حالت آموزشی، بدون این کنترل‌ها.")

        run_btn = st.form_submit_button("🚀 اجرای محاسبات", type="primary", use_container_width=True)

if run_btn:
    try:
        st.session_state["out"] = run_analysis(
            mode.startswith("مقاومت قرائت"), a_text, v_text, area, aspect, spacing, d_mm, rod_mode, rod_len,
            depth, method,
            ig, ts, weight, use_surface, rho_s, hs, real_mode)
    except (ValueError, RuntimeError) as e:
        st.error(f"خطا: {e}")
        st.stop()
elif "out" not in st.session_state:
    st.session_state["out"] = run_analysis(**DEFAULTS)     # نمایش اولیه با داده نمونه

out = st.session_state["out"]
a_vals, rho_vals, fit = out["a"], out["rho"], out["fit"]
tv, gpr, rg, inp = out["tv"], out["gpr"], out["rg"], out["inp"]
ratio = gpr / tv["e_touch"]
safe = ratio <= 1.0
mesh_, geom_ = out["mesh"], out["geom"]
mesh_ok_ = mesh_["Em"] <= tv["e_touch"] and mesh_["Es"] <= tv["e_step"]

# ============================================================
#  نمایش نتایج
# ============================================================
st.markdown(H("""
<div class="hero">
  <div class="hero-badge">IEEE Std 80</div>
  <div class="hero-title">تحلیل مقاومت ویژه خاک و طراحی شبکه زمین</div>
  <div class="hero-sub">برازش مدل دولایه Wenner، مقاومت شبکه، ولتاژهای mesh و step و نقشه گرادیان پتانسیل</div>
  <div class="hero-chips"><span>Wenner</span><span>Two-layer model</span><span>Sverak</span><span>Mesh / Step</span><span>Potential map</span></div>
</div>"""), unsafe_allow_html=True)

mode_chip = chip("حالت پروژه واقعی", "good") if inp["real_mode"] else chip("حالت آموزشی", "neutral")
st.markdown(f'<div class="chips">{mode_chip}</div>', unsafe_allow_html=True)
if inp["real_mode"]:
    _issues = data_sufficiency(a_vals, rho_vals, fit, inp["area"])
    for _lvl, _txt in _issues[:2]:
        st.markdown(f'<div class="note {_lvl}">{_ICON[_lvl]} <b>هشدار داده‌ها:</b> {_txt}</div>', unsafe_allow_html=True)
    if len(_issues) > 2:
        st.markdown(f'<div class="note info">{len(_issues) - 2} مورد دیگر در تب «توصیه مهندسی» آمده است.</div>',
                    unsafe_allow_html=True)

gpr_color = C_GREEN if safe else (C_AMBER if mesh_ok_ else C_RED)
em_color = C_GREEN if mesh_["Em"] <= tv["e_touch"] else C_RED
es_color = C_GREEN if mesh_["Es"] <= tv["e_step"] else C_RED
cards = [
    ("لایه اول", "🟦", f"{fit['rho1']:,.1f}", "Ω·m", "مقاومت ویژه ρ₁", C_BLUE),
    ("لایه دوم", "🟪", f"{fit['rho2']:,.1f}", "Ω·m", f"ρ₂ · ضخامت لایه اول {fit['h']:.2f} m", C_PURPLE),
    ("مقاومت شبکه Rg", "🔌", f"{out['rg_design']:.3f}", "Ω", "روش Sverak", C_AMBER),
    ("افزایش پتانسیل زمین", "⚡", f"{gpr:,.0f}", "V", f"Ig = {inp['ig']:,.0f} A", gpr_color),
    ("ولتاژ mesh (Em)", "🔲", f"{mesh_['Em']:,.0f}", "V", f"حد تماس {tv['e_touch']:,.0f} V", em_color),
    ("ولتاژ step (Es)", "👣", f"{mesh_['Es']:,.0f}", "V", f"حد گام {tv['e_step']:,.0f} V", es_color),
]
st.markdown(H('<div class="kpi-grid">' + "".join(
    f'<div class="kpi" style="--accent:{c}"><div class="kpi-top"><div class="kpi-label">{t}</div>'
    f'<div class="kpi-ico">{i}</div></div><div class="kpi-value">{v}<small>{u}</small></div>'
    f'<div class="kpi-sub">{s}</div></div>' for t, i, v, u, s, c in cards) + '</div>'),
    unsafe_allow_html=True)

fill_pct = min(ratio, 1.5) / 1.5 * 100
fill_col = C_GREEN if ratio <= 0.8 else C_AMBER if ratio <= 1 else C_RED
if safe:
    state, title = "ok", "✅ طرح از نظر GPR ایمن است"
    desc = "طبق IEEE 80 در این حالت تحلیل ولتاژهای mesh و step الزامی نیست."
elif mesh_ok_:
    state, title = "warn", "⚠️ GPR بالاست، اما Em و Es مجازند"
    desc = "ولتاژهای تماس و گام در حد مجازند؛ ولتاژ منتقل‌شده به حصار، لوله و کابل‌ها را کنترل کنید."
else:
    state, title = "bad", "🚨 Em یا Es از حد مجاز بیشتر است"
    desc = "هندسه شبکه، میله‌ها، لایه سطحی، Ig یا زمان قطع را اصلاح کنید."
st.markdown(H(f"""
<div class="status {state}">
  <div class="status-t">{title}</div>
  <div class="status-d">GPR = {num(gpr, ',.0f')} V در برابر ولتاژ تماس مجاز {num(tv['e_touch'], ',.0f')} V ({num(ratio * 100, '.0f')}٪) ·
  Em = {num(mesh_['Em'], ',.0f')} V ({num(mesh_['Em'] / tv['e_touch'] * 100, '.0f')}٪) ·
  Es = {num(mesh_['Es'], ',.0f')} V ({num(mesh_['Es'] / tv['e_step'] * 100, '.0f')}٪) — {desc}</div>
  <div class="track"><div class="fill" style="width:{fill_pct:.1f}%;background:{fill_col}"></div><div class="limit"></div></div>
  <div class="track-l"><span>0</span><span>حد مجاز GPR</span><span>150٪</span></div>
</div>"""), unsafe_allow_html=True)

tab1, tab_3d, tab_map, tab2, tab3, tab4, tab5 = st.tabs(
    ["🌍 مدل خاک", "🌐 نمای سه‌بعدی", "🗺️ پتانسیل و گرادیان", "🛡️ شبکه و ایمنی", "📋 جدول داده‌ها", "📄 گزارش",
     "🧭 توصیه مهندسی"])

with tab1:
    q_kind = "good" if fit["rmse_pct"] <= 5 else "warn" if fit["rmse_pct"] <= 10 else "bad"
    chips_html = "".join([
        chip(f"RMSE برازش: {fit['rmse_pct']:.1f}٪", q_kind),
        chip(f"بیشینه خطا: {fit['max_err_pct']:.1f}٪", "neutral"),
        chip(f"ضریب بازتاب K = {fit['k']:.3f}", "neutral"),
        chip(f"RMSE مدل یکنواخت: {fit['rmse_uniform_pct']:.1f}٪", "neutral"),
    ])
    st.markdown(f'<div class="chips">{chips_html}</div>', unsafe_allow_html=True)
    if fit["rmse_pct"] > 10:
        st.markdown('<div class="note warn">خطای برازش بالاست؛ داده‌ها را بازبینی کنید یا خاک ممکن است بیش از دو لایه باشد.</div>',
                    unsafe_allow_html=True)
    if abs(fit["k"]) < 0.05 or fit["rmse_uniform_pct"] <= fit["rmse_pct"] * 1.2:
        st.markdown('<div class="note info">خاک تقریباً یکنواخت است؛ ρ₂ و h در این حالت قابل اتکا نیستند.</div>',
                    unsafe_allow_html=True)
    if fit["h_at_bound"] or fit["rho_at_bound"]:
        st.markdown('<div class="note warn">یکی از پارامترها به مرز بازه جستجو رسیده است؛ نتیجه برازش یکتا نیست.</div>',
                    unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        section("منحنی برازش", "اندازه‌گیری در برابر مدل")
        st.plotly_chart(fit_fig(a_vals, rho_vals, fit), use_container_width=True)
    with c2:
        section("مقطع خاک", "رسم‌شده از اعداد تحلیل · مقیاس عمق واقعی")
        svg = soil_section_svg(fit, inp["depth"], inp["hs"], inp["rho_s"])
        b64 = base64.b64encode(svg.encode("utf-8")).decode()
        st.markdown(f'<img src="data:image/svg+xml;base64,{b64}" style="width:100%;border-radius:16px;'
                    f'box-shadow:0 8px 22px rgba(0,0,0,.18)"/>', unsafe_allow_html=True)
        st.download_button("📥 دانلود مقطع (SVG)", svg.encode("utf-8"), file_name="soil_section.svg",
                           mime="image/svg+xml", use_container_width=True)
    section("خطای برازش", "ناحیه سبز: ±۱۰٪")
    st.plotly_chart(residual_fig(a_vals, fit["rel_err"]), use_container_width=True)

with tab_3d:
    section("نمای سه‌بعدی لایه‌ها و شبکه زمین", "با ماوس یا انگشت بچرخانید")
    exag = st.slider("اغراق عمق (×)", 1, 10, 3,
                     help="عمق لایه‌ها نسبت به ابعاد زمین کم است؛ این مقدار فقط ظاهر نمودار را بلندتر می‌کند.")
    st.plotly_chart(fig_3d_soil(fit, geom_, inp["depth"], inp["hs"], float(exag)), use_container_width=True)
    st.markdown('<div class="note info">هادی‌ها و میله‌ها دقیقاً با ابعاد، فاصله D و قطعات واردشده در پنل کناری رسم شده‌اند. '
                'نمودار برای درک فضایی است، نه نقشه اجرایی.</div>', unsafe_allow_html=True)

with tab_map:
    section("نقشه پتانسیل سطح و محدوده گرادیان", "تحلیل عددی · خاک یکنواخت با ρ معادل")
    num_, stt_ = out.get("num"), out.get("num_stats")
    if not stt_:
        st.markdown('<div class="note warn">تحلیل عددی پتانسیل انجام نشد؛ هندسه شبکه را بررسی کنید.</div>', unsafe_allow_html=True)
    else:
        tu, su = stt_["touch_unsafe_pct"], stt_["step_unsafe_pct"]
        st.markdown('<div class="chips">' + "".join([
            chip(f"ناحیه ناایمن تماس: {tu:.1f}٪", "good" if tu == 0 else "bad"),
            chip(f"ناحیه ناایمن گام: {su:.1f}٪", "good" if su == 0 else "bad"),
            chip(f"Rg عددی: {out['rho_eq'] * num_['rg_unit']:.2f} Ω", "neutral"),
            chip(f"{num_['n_seg']} قطعه · گام نقشه {num_['dx']:.2f} m", "neutral"),
        ]) + '</div>', unsafe_allow_html=True)

        view = st.radio("نمایش", ["ولتاژ تماس", "ولتاژ گام", "پتانسیل سطح", "گرادیان پتانسیل"], horizontal=True)
        st.plotly_chart(potential_map_fig(view, num_, stt_, geom_, tv), use_container_width=True)
        st.markdown('<div class="note info">خط مشکی حد مجاز است؛ ناحیه داخل آن از حد عبور کرده است. '
                    'رنگ سبز یعنی ایمن، زرد نزدیک حد و قرمز بالاتر از حد.</div>', unsafe_allow_html=True)

        section("محدوده گرادیان پتانسیل", "بیشینه و کمینه روی نقشه")
        lo_, hi_ = stt_["grad_in"]
        st.markdown(H('<div class="mini-grid">' + "".join([
            mini_card("گرادیان داخل شبکه (کمینه)", f"{lo_:.2f}", "V/m"),
            mini_card("گرادیان داخل شبکه (بیشینه)", f"{hi_:.2f}", "V/m", highlight=True),
            mini_card("گرادیان بیرون شبکه (بیشینه)", f"{stt_['grad_out_max']:.2f}", "V/m"),
            mini_card("بیشینه ولتاژ تماس", f"{stt_['touch_max']:,.0f}", "V"),
            mini_card("بیشینه ولتاژ گام", f"{stt_['step_max']:,.0f}", "V"),
            mini_card("کمترین پتانسیل سطح داخل شبکه", f"{stt_['v_min_pct']:.0f}", "٪ GPR"),
        ]) + '</div>'), unsafe_allow_html=True)

        section("نیمرخ در امتداد X", "موقعیت Y را انتخاب کنید")
        y_pct = st.slider("موقعیت Y (٪ عرض زمین)", 0, 100, 50)
        f_pot, f_ts, y_used = profile_figs(num_, stt_, geom_, tv, y_pct / 100.0)
        st.caption(f"نیمرخ در Y = {y_used:.1f} m ؛ ناحیه خاکستری محدوده شبکه است.")
        st.plotly_chart(f_pot, use_container_width=True)
        st.plotly_chart(f_ts, use_container_width=True)
        st.markdown('<div class="note warn">فرض‌ها: خاک یکنواخت با ρ معادل، شبکه مستطیلی، ولتاژ تماس نقطه‌ای (GPR منهای پتانسیل سطح) '
                    'و ولتاژ گام برابر گرادیان × ۱ متر. برای طرح نهایی نتیجه را با نرم‌افزار تخصصی (مثل CYMGRD یا CDEGS) '
                    'راستی‌آزمایی کنید.</div>', unsafe_allow_html=True)

with tab2:
    section("مقاومت شبکه زمین", f"ρ معادل = {out['rho_eq']:,.1f} Ω·m")
    st.markdown(H('<div class="mini-grid">' + "".join([
        mini_card("ρ / 4r", f"{rg['simple']:.3f}", "Ω"),
        mini_card("ρ / 4r + ρ / L", f"{rg['with_length']:.3f}", "Ω"),
        mini_card("Sverak", f"{rg['sverak']:.3f}", "Ω", highlight=True, tag="مقدار طراحی"),
    ]) + '</div>'), unsafe_allow_html=True)

    section("هندسه شبکه و ولتاژهای mesh/step", "رابطه‌های IEEE 80")
    st.markdown(H('<div class="mini-grid">' + "".join([
        mini_card("ابعاد زمین", f"{geom_['Lx']:.1f} × {geom_['Ly']:.1f}", "m"),
        mini_card("هادی‌ها", f"{geom_['nx']} × {geom_['ny']}", f"D ≈ {geom_['D']:.1f} m"),
        mini_card("طول کل هادی (با میله)", f"{geom_['LT']:,.0f}", "m"),
        mini_card("میله‌های زمین", f"{geom_['n_rods']}", f"× {geom_['Lr']:g} m"),
        mini_card("Em (mesh)", f"{mesh_['Em']:,.0f}", "V", highlight=True, tag=f"حد {tv['e_touch']:,.0f} V"),
        mini_card("Es (step)", f"{mesh_['Es']:,.0f}", "V", highlight=True, tag=f"حد {tv['e_step']:,.0f} V"),
        mini_card("Km · Ki · Ks", f"{mesh_['Km']:.3f} · {mesh_['Ki']:.2f} · {mesh_['Ks']:.3f}", ""),
        mini_card("n (ضریب هندسی)", f"{mesh_['n']:.1f}", ""),
    ]) + '</div>'), unsafe_allow_html=True)
    for _w in mesh_["warnings"]:
        st.markdown(f'<div class="note warn">{_w}</div>', unsafe_allow_html=True)
    if geom_["capped"]:
        st.markdown('<div class="note warn">تعداد هادی‌ها به سقف ۲۵ در هر جهت محدود شده؛ فاصله واقعی بزرگ‌تر از D واردشده است.</div>',
                    unsafe_allow_html=True)

    section("حدود مجاز ایمنی", f"وزن مبنا {inp['weight']} kg · زمان خطا {inp['ts']} s")
    st.markdown(H('<div class="mini-grid">' + "".join([
        mini_card("جریان مجاز بدن", f"{tv['i_body']:.3f}", "A"),
        mini_card("ضریب لایه سطحی Cs", f"{tv['cs']:.3f}", ""),
        mini_card("ولتاژ گام مجاز", f"{tv['e_step']:,.0f}", "V"),
        mini_card("ولتاژ تماس مجاز", f"{tv['e_touch']:,.0f}", "V"),
    ]) + '</div>'), unsafe_allow_html=True)

    section("مقایسه GPR، Em و Es با حدود مجاز", "خط آبی: حد مجاز")
    st.plotly_chart(safety_fig(gpr, tv, mesh_["Em"], mesh_["Es"]), use_container_width=True)

with tab3:
    section("پارامترهای مدل خاک")
    _fmt = lambda lo, hi: f"{lo:,.1f} – {hi:,.1f}" if np.isfinite(lo) and np.isfinite(hi) else "—"
    st.dataframe(pd.DataFrame({
        "پارامتر": ["ρ₁", "ρ₂", "h", "K"],
        "مقدار": [fit["rho1"], fit["rho2"], fit["h"], fit["k"]],
        "بازه اطمینان ۹۵٪": [_fmt(*fit["ci_95"][0]), _fmt(*fit["ci_95"][1]), _fmt(*fit["ci_95"][2]), "—"],
        "واحد": ["Ω·m", "Ω·m", "m", "—"]}),
        hide_index=True, use_container_width=True,
        column_config={"مقدار": st.column_config.NumberColumn(format="%.3f")})
    if fit["dof"] < 2:
        st.caption("تعداد داده‌ها نزدیک به تعداد پارامترهاست؛ بازه‌های اطمینان قابل اتکا نیستند.")
    section("مقایسه اندازه‌گیری و مدل")
    df_out = pd.DataFrame({"a (m)": a_vals, "Measured (Ω·m)": rho_vals,
                           "Fitted (Ω·m)": fit["pred"], "Error (%)": fit["rel_err"]})
    st.dataframe(df_out, hide_index=True, use_container_width=True,
                 column_config={c: st.column_config.NumberColumn(format="%.2f") for c in df_out.columns})

with tab4:
    report = build_report(inp, fit, out["rho_eq"], rg, out["rg_design"], gpr, tv, out)
    with st.container(border=True):
        st.markdown(report_html(inp, fit, out["rho_eq"], rg, out["rg_design"], gpr, tv, out),
                    unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    d1.download_button("📥 دانلود گزارش (Markdown)", report.encode("utf-8"),
                       file_name="grounding_report.md", use_container_width=True)
    d2.download_button("📥 دانلود داده‌ها (CSV)", df_out.to_csv(index=False).encode("utf-8-sig"),
                       file_name="soil_analysis.csv", use_container_width=True)
    with st.expander("فرضیات و محدودیت‌ها"):
        st.markdown(
            "- مدل خاک دولایه است و برای خاک چندلایه یا ناهمگن دقت کمتری دارد.\n"
            "- ρ معادل تقریبی است؛ برای طرح نهایی از روش Sunde یا نرم‌افزار تخصصی استفاده کنید.\n"
            "- Em و Es با رابطه‌های IEEE 80 (شبکه مستطیلی، خاک یکنواخت) و نقشه پتانسیل با روش عددی ساده محاسبه شده‌اند؛ "
            "برای طرح نهایی با نرم‌افزار تخصصی راستی‌آزمایی کنید.\n"
            "- خروجی برای طراحی اولیه است و طرح نهایی باید مهندس مسئول بازبینی کند.")

with tab5:
    adv = engineering_advice(out)
    section("خلاصه و توصیه مهندسی", "بر پایه IEEE 80 و IEEE 81 · قاعده‌محور")
    st.markdown(advice_html(adv), unsafe_allow_html=True)
    st.download_button("📥 دانلود توصیه‌ها (Markdown)", advice_markdown(adv).encode("utf-8"),
                       file_name="engineering_advice.md", use_container_width=True)

st.markdown('<div class="foot">Grounding Designer · بر پایه IEEE Std 80</div>', unsafe_allow_html=True)
