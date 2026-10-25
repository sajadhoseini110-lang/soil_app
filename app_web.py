# -*- coding: utf-8 -*-
"""
Soil & Grounding Designer — تحلیل مقاومت ویژه خاک (Wenner) و طراحی اولیه شبکه زمین
مرجع: IEEE Std 80
"""
from __future__ import annotations

import re
import hashlib
from datetime import date
from io import BytesIO
from urllib.parse import quote

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from PIL import Image
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
#  استایل
# ============================================================
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;500;700;800&display=swap');
.stApp, button, input, textarea, label, .stMarkdown { font-family: 'Vazirmatn', Tahoma, sans-serif !important; }
.block-container { padding-top: 1.4rem; max-width: 1280px; direction: rtl; }
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
.kpi-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-bottom:16px; }
@media (max-width: 900px) { .kpi-grid { grid-template-columns:repeat(2,1fr); } }
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
.foot { text-align:center; font-size:.78rem; opacity:.55; margin:30px 0 6px; }
"""

C_BLUE, C_PURPLE, C_GREEN, C_AMBER, C_RED = "#2563eb", "#8b5cf6", "#10b981", "#f59e0b", "#ef4444"
FONT = "Vazirmatn, Tahoma, sans-serif"
DEFAULTS = dict(
    mode_R=False, a_text="1, 2, 3, 4, 6, 8, 10, 15, 20", v_text="120, 95, 80, 70, 58, 52, 48, 44, 42",
    area=1000.0, length=500.0, depth=0.5, method="apparent",
    ig=5000.0, ts=0.5, weight=70, use_surface=True, rho_s=3000.0, hs=0.1,
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


def soil_profile_fig(fit, depth: float, hs: float, rho_s: float) -> go.Figure:
    h = fit["h"]
    ymax = max(h * 2.0, h + 3.0)
    ymin = -(hs + 0.12 * ymax)
    fig = go.Figure(go.Scatter(x=[0, 1], y=[ymin, ymax], mode="markers", marker_opacity=0,
                               showlegend=False, hoverinfo="skip"))

    def rect(y0, y1, fill, line):
        fig.add_shape(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=y0, y1=y1,
                      fillcolor=fill, line=dict(color=line, width=1.5), layer="below")

    def label(y, text, **kw):
        fig.add_annotation(xref="paper", yref="y", x=0.5, y=y, text=text, showarrow=False, **kw)

    if hs > 0:
        rect(-hs, 0, "rgba(148,163,184,.85)", "rgba(100,116,139,1)")
        label(-hs, f"لایه سطحی · ρs = {rho_s:,.0f} Ω·m", yanchor="bottom", font=dict(size=11))
    rect(0, h, "rgba(37,99,235,.30)", C_BLUE)
    rect(h, ymax, "rgba(139,92,246,.30)", C_PURPLE)
    label(h / 2, f"<b>لایه اول</b><br>ρ₁ = {fit['rho1']:,.1f} Ω·m")
    label((h + ymax) / 2, f"<b>لایه دوم (تا بی‌نهایت)</b><br>ρ₂ = {fit['rho2']:,.1f} Ω·m")
    fig.add_shape(type="line", xref="paper", yref="y", x0=0.94, x1=0.94, y0=0, y1=h,
                  line=dict(color="gray", width=2))
    fig.add_annotation(xref="paper", yref="y", x=0.92, y=h / 2, text=f"h = {h:.2f} m",
                       showarrow=False, xanchor="right", font=dict(size=11))
    if depth > 0:
        fig.add_shape(type="line", xref="paper", yref="y", x0=0.06, x1=0.80, y0=depth, y1=depth,
                      line=dict(color=C_AMBER, width=3, dash="dash"))
        fig.add_annotation(xref="paper", yref="y", x=0.06, y=depth, text=f"شبکه زمین ({depth:g} m)",
                           showarrow=False, xanchor="left", yanchor="bottom", font=dict(size=11, color=C_AMBER))
    fig.update_xaxes(range=[0, 1], visible=False, fixedrange=True)
    fig.update_yaxes(range=[ymax, ymin], title="عمق (m)", fixedrange=True)
    return style_fig(fig, 400)


def safety_fig(gpr: float, tv: dict) -> go.Figure:
    labels = ["GPR", "ولتاژ تماس مجاز", "ولتاژ گام مجاز"]
    vals = [gpr, tv["e_touch"], tv["e_step"]]
    colors = [C_GREEN if gpr <= tv["e_touch"] else C_RED, C_BLUE, C_PURPLE]
    fig = go.Figure(go.Bar(x=vals, y=labels, orientation="h", marker_color=colors,
                           text=[f"{v:,.0f} V" for v in vals], textposition="outside", cliponaxis=False))
    fig.add_vline(x=tv["e_touch"], line_dash="dash", line_color=C_BLUE, opacity=.6)
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(range=[0, max(vals) * 1.25], title="ولتاژ (V)")
    return style_fig(fig, 260)


# ============================================================
#  تصویرسازی هنری با هوش مصنوعی (اختیاری)
# ============================================================
AI_ENDPOINTS = [
    "https://gen.pollinations.ai/image/{prompt}",       # API جدید (ممکن است کلید بخواهد)
    "https://image.pollinations.ai/prompt/{prompt}",    # API قدیمی (legacy)
]


def describe_soil(rho: float) -> str:
    """تبدیل مقاومت ویژه به توصیف بصری خاک برای پرامپت."""
    if rho < 20:
        return "very wet dark marshy clay"
    if rho < 60:
        return "moist dark brown loam"
    if rho < 150:
        return "damp reddish-brown clay-loam"
    if rho < 500:
        return "dry light tan sand and gravel"
    if rho < 2000:
        return "dry grey gravelly rocky ground"
    return "solid grey granite bedrock"


def build_soil_prompt(rho1: float, rho2: float, h: float) -> str:
    ymax = max(h * 2.0, h + 3.0)
    pct = int(min(70, max(15, round(h / ymax * 100 / 5) * 5)))
    return (
        "Side-view cross-section illustration of two horizontal soil layers for an electrical grounding study, "
        "flat clean technical style, soft colors. "
        f"Top layer: {describe_soil(rho1)}, about {pct} percent of the image height. "
        f"Below a sharp straight boundary: {describe_soil(rho2)}, reaching the bottom edge. "
        "A thin grass strip on top and a few short copper grounding conductors buried near the surface. "
        "No text, no letters, no numbers, no labels, no people."
    )


@st.cache_data(show_spinner=False, ttl=3600, max_entries=16)
def generate_soil_image(prompt: str, seed: int, api_key: str = "") -> bytes:
    """دریافت تصویر از سرویس خارجی؛ در صورت شکست خطا می‌دهد (و نتیجه ناموفق کش نمی‌شود)."""
    params = {"width": 1024, "height": 768, "seed": seed, "nologo": "true", "model": "flux"}
    if api_key:
        params["key"] = api_key
    errors = []
    for template in AI_ENDPOINTS:
        host = template.split("/")[2]
        try:
            r = requests.get(template.format(prompt=quote(prompt, safe="")), params=params, timeout=(10, 90))
        except requests.RequestException as exc:
            errors.append(f"{host}: {type(exc).__name__}")
            continue
        if r.ok and r.headers.get("content-type", "").startswith("image"):
            try:
                Image.open(BytesIO(r.content)).verify()
                return r.content
            except OSError:
                errors.append(f"{host}: تصویر نامعتبر")
                continue
        errors.append(f"{host}: HTTP {r.status_code}")
    raise RuntimeError("تولید تصویر ناموفق بود (" + " | ".join(errors) +
                       "). اتصال اینترنت را بررسی کنید یا کلید API بگذارید.")


# ============================================================
#  اجرای تحلیل
# ============================================================
def run_analysis(mode_R, a_text, v_text, area, length, depth, method,
                 ig, ts, weight, use_surface, rho_s, hs) -> dict:
    a_raw = parse_series(a_text, "فاصله a")
    v_raw = parse_series(v_text, "مقادیر اندازه‌گیری")
    if mode_R and a_raw.size == v_raw.size:
        v_raw = 2.0 * np.pi * a_raw * v_raw          # ρa = 2π·a·R
    a_vals, rho_vals = validate_data(a_raw, v_raw)
    fit = fit_two_layer(tuple(a_vals), tuple(rho_vals))

    rho_eq = equivalent_resistivity(method, fit, area)
    rg = grid_resistance(rho_eq, area, length, depth)
    gpr = ig * rg["sverak"]
    hs_eff = hs if use_surface else 0.0
    tv = tolerable_voltages(fit["rho1"], rho_s if use_surface else fit["rho1"], hs_eff, weight, ts)
    inp = dict(area=area, length=length, depth=depth, ig=ig, ts=ts, weight=weight,
               hs=hs_eff, rho_s=rho_s)
    return dict(a=a_vals, rho=rho_vals, fit=fit, rho_eq=rho_eq, rg=rg,
                rg_design=rg["sverak"], gpr=gpr, tv=tv, inp=inp)


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
        length = st.number_input("طول کل هادی‌ها (m)", min_value=1.0, value=DEFAULTS["length"])
        depth = st.number_input("عمق دفن شبکه (m)", min_value=0.0, value=DEFAULTS["depth"])
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

        run_btn = st.form_submit_button("🚀 اجرای محاسبات", type="primary", use_container_width=True)

if run_btn:
    try:
        st.session_state["out"] = run_analysis(
            mode.startswith("مقاومت قرائت"), a_text, v_text, area, length, depth, method,
            ig, ts, weight, use_surface, rho_s, hs)
        st.session_state.pop("ai_img", None)
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

# ============================================================
#  نمایش نتایج
# ============================================================
st.markdown(H("""
<div class="hero">
  <div class="hero-badge">IEEE Std 80</div>
  <div class="hero-title">تحلیل مقاومت ویژه خاک و طراحی شبکه زمین</div>
  <div class="hero-sub">برازش مدل دولایه Wenner، محاسبه مقاومت شبکه و بررسی ولتاژهای مجاز</div>
  <div class="hero-chips"><span>Wenner</span><span>Two-layer model</span><span>Sverak</span><span>GPR check</span></div>
</div>"""), unsafe_allow_html=True)

gpr_color = C_GREEN if safe else C_RED
cards = [
    ("لایه اول", "🟦", f"{fit['rho1']:,.1f}", "Ω·m", "مقاومت ویژه ρ₁", C_BLUE),
    ("لایه دوم", "🟪", f"{fit['rho2']:,.1f}", "Ω·m", f"ρ₂ · ضخامت لایه اول {fit['h']:.2f} m", C_PURPLE),
    ("مقاومت شبکه Rg", "🔌", f"{out['rg_design']:.3f}", "Ω", "روش Sverak", C_AMBER),
    ("افزایش پتانسیل زمین", "⚡", f"{gpr:,.0f}", "V", f"Ig = {inp['ig']:,.0f} A", gpr_color),
]
st.markdown(H('<div class="kpi-grid">' + "".join(
    f'<div class="kpi" style="--accent:{c}"><div class="kpi-top"><div class="kpi-label">{t}</div>'
    f'<div class="kpi-ico">{i}</div></div><div class="kpi-value">{v}<small>{u}</small></div>'
    f'<div class="kpi-sub">{s}</div></div>' for t, i, v, u, s, c in cards) + '</div>'),
    unsafe_allow_html=True)

fill_pct = min(ratio, 1.5) / 1.5 * 100
fill_col = C_GREEN if ratio <= 0.8 else C_AMBER if ratio <= 1 else C_RED
title = "✅ طرح از نظر GPR ایمن است" if safe else "⚠️ طرح نیاز به بازنگری دارد"
desc = ("طبق IEEE 80 در این حالت تحلیل ولتاژهای mesh و step الزامی نیست." if safe else
        "GPR از ولتاژ تماس مجاز بیشتر است؛ محاسبه Em و Es و اصلاح طرح لازم است.")
st.markdown(H(f"""
<div class="status {'ok' if safe else 'bad'}">
  <div class="status-t">{title}</div>
  <div class="status-d">GPR = {num(gpr, ',.0f')} V در برابر ولتاژ تماس مجاز {num(tv['e_touch'], ',.0f')} V
  ({num(ratio * 100, '.0f')}٪ حد مجاز) — {desc}</div>
  <div class="track"><div class="fill" style="width:{fill_pct:.1f}%;background:{fill_col}"></div><div class="limit"></div></div>
  <div class="track-l"><span>0</span><span>حد مجاز</span><span>150٪</span></div>
</div>"""), unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🌍 مدل خاک", "🛡️ شبکه و ایمنی", "📋 جدول داده‌ها", "📄 گزارش", "🎨 تصویرسازی AI"])

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

    c1, c2 = st.columns([3, 2])
    with c1:
        section("منحنی برازش", "اندازه‌گیری در برابر مدل")
        st.plotly_chart(fit_fig(a_vals, rho_vals, fit), use_container_width=True)
    with c2:
        section("مقطع خاک", "نمایش لایه‌ها")
        st.plotly_chart(soil_profile_fig(fit, inp["depth"], inp["hs"], inp["rho_s"]),
                        use_container_width=True)
    section("خطای برازش", "ناحیه سبز: ±۱۰٪")
    st.plotly_chart(residual_fig(a_vals, fit["rel_err"]), use_container_width=True)

with tab2:
    section("مقاومت شبکه زمین", f"ρ معادل = {out['rho_eq']:,.1f} Ω·m")
    st.markdown(H('<div class="mini-grid">' + "".join([
        mini_card("ρ / 4r", f"{rg['simple']:.3f}", "Ω"),
        mini_card("ρ / 4r + ρ / L", f"{rg['with_length']:.3f}", "Ω"),
        mini_card("Sverak", f"{rg['sverak']:.3f}", "Ω", highlight=True, tag="مقدار طراحی"),
    ]) + '</div>'), unsafe_allow_html=True)

    section("حدود مجاز ایمنی", f"وزن مبنا {inp['weight']} kg · زمان خطا {inp['ts']} s")
    st.markdown(H('<div class="mini-grid">' + "".join([
        mini_card("جریان مجاز بدن", f"{tv['i_body']:.3f}", "A"),
        mini_card("ضریب لایه سطحی Cs", f"{tv['cs']:.3f}", ""),
        mini_card("ولتاژ گام مجاز", f"{tv['e_step']:,.0f}", "V"),
        mini_card("ولتاژ تماس مجاز", f"{tv['e_touch']:,.0f}", "V"),
    ]) + '</div>'), unsafe_allow_html=True)

    section("مقایسه GPR با حدود مجاز")
    st.plotly_chart(safety_fig(gpr, tv), use_container_width=True)

with tab3:
    section("پارامترهای مدل خاک")
    st.dataframe(pd.DataFrame({
        "پارامتر": ["ρ₁", "ρ₂", "h", "K"],
        "مقدار": [fit["rho1"], fit["rho2"], fit["h"], fit["k"]],
        "واحد": ["Ω·m", "Ω·m", "m", "—"]}),
        hide_index=True, use_container_width=True,
        column_config={"مقدار": st.column_config.NumberColumn(format="%.3f")})
    section("مقایسه اندازه‌گیری و مدل")
    df_out = pd.DataFrame({"a (m)": a_vals, "Measured (Ω·m)": rho_vals,
                           "Fitted (Ω·m)": fit["pred"], "Error (%)": fit["rel_err"]})
    st.dataframe(df_out, hide_index=True, use_container_width=True,
                 column_config={c: st.column_config.NumberColumn(format="%.2f") for c in df_out.columns})

with tab4:
    report = build_report(inp, fit, out["rho_eq"], rg, out["rg_design"], gpr, tv)
    with st.container(border=True):
        st.markdown(report)
    d1, d2 = st.columns(2)
    d1.download_button("📥 دانلود گزارش (Markdown)", report.encode("utf-8"),
                       file_name="grounding_report.md", use_container_width=True)
    d2.download_button("📥 دانلود داده‌ها (CSV)", df_out.to_csv(index=False).encode("utf-8-sig"),
                       file_name="soil_analysis.csv", use_container_width=True)
    with st.expander("فرضیات و محدودیت‌ها"):
        st.markdown(
            "- مدل خاک دولایه است و برای خاک چندلایه یا ناهمگن دقت کمتری دارد.\n"
            "- ρ معادل تقریبی است؛ برای طرح نهایی از روش Sunde یا نرم‌افزار تخصصی استفاده کنید.\n"
            "- ولتاژهای mesh و step محاسبه نشده‌اند و فقط GPR با ولتاژ تماس مجاز مقایسه شده است.\n"
            "- خروجی برای طراحی اولیه است و طرح نهایی باید مهندس مسئول بازبینی کند.")

with tab5:
    section("تصویرسازی هنری با هوش مصنوعی", "اختیاری")
    st.markdown('<div class="note warn">این تصویر فقط یک نمای هنری است و مقیاس یا اعداد دقیق ندارد؛ '
                'مقطع مهندسی دقیق در تب «مدل خاک» قرار دارد.</div>', unsafe_allow_html=True)
    default_prompt = build_soil_prompt(fit["rho1"], fit["rho2"], fit["h"])
    pkey = hashlib.md5(default_prompt.encode()).hexdigest()[:8]
    prompt_text = st.text_area("متن توصیف تصویر (قابل ویرایش)", default_prompt, height=140, key=f"prompt_{pkey}")
    st.caption("فقط همین متن به سرویس خارجی ارسال می‌شود و مقادیر عددی پروژه در آن نیست.")
    api_key = st.text_input("کلید API سرویس (اختیاری)", type="password",
                            help="اگر سرویس بدون کلید پاسخ نداد، کلید را از enter.pollinations.ai بگیرید.")
    b1, b2 = st.columns(2)
    gen = b1.button("✨ تولید تصویر", use_container_width=True)
    regen = b2.button("🔄 تصویر جدید", use_container_width=True)
    if regen:
        st.session_state["ai_var"] = st.session_state.get("ai_var", 0) + 1
    if gen or regen:
        seed = int(hashlib.md5(prompt_text.encode()).hexdigest()[:6], 16) + st.session_state.get("ai_var", 0)
        try:
            with st.spinner("در حال ترسیم مقطع زمین (۱۰ تا ۶۰ ثانیه)..."):
                data = generate_soil_image(prompt_text, seed, api_key.strip())
            st.session_state["ai_img"] = dict(
                data=data,
                caption=(f"نمای هنری · ρ₁ = {fit['rho1']:.0f} Ω·m ، h = {fit['h']:.1f} m ، "
                         f"ρ₂ = {fit['rho2']:.0f} Ω·m (اعداد از تحلیل‌اند، نه از تصویر)"))
        except RuntimeError as e:
            st.error(str(e))
    img = st.session_state.get("ai_img")
    if img:
        st.image(img["data"], caption=img["caption"], use_container_width=True)
        st.download_button("📥 دانلود تصویر", img["data"], file_name="soil_section.jpg", mime="image/jpeg")

st.markdown('<div class="foot">Grounding Designer · بر پایه IEEE Std 80</div>', unsafe_allow_html=True)
