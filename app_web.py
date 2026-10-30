# -*- coding: utf-8 -*-
"""
Soil & Grounding Designer — ایستگاه کاری تحلیل مقاومت ویژه خاک (Wenner) و طراحی اولیه شبکه زمین
مرجع: IEEE Std 80 / IEEE Std 81

امکانات: برازش خاک دولایه، مقطع خاک، نمای سه‌بعدی شبکه با اندازه‌گذاری هندسی (Plotly)،
مقاومت شبکه (Sverak)، GPR و ولتاژهای مجاز، ولتاژهای mesh و step (IEEE 80)،
نقشه پتانسیل و گرادیان (تحلیل عددی ساده)، کنترل کفایت داده، بازه اطمینان پارامترها،
گزارش راست‌به‌چپ و توصیه مهندسی.

نکته: توابع محاسباتی این فایل عیناً از نسخه قبلی منتقل شده‌اند؛ فقط لایه نمایش بازطراحی شده است.
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
#  ثابت‌ها (بدون تغییر نسبت به نسخه قبل)
# ============================================================

MAX_TERMS = 500


RHO_MIN, RHO_MAX = 0.1, 1e5


H_MIN, H_MAX = 0.05, 200.0


_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩٫٬،؛",
                        "01234567890123456789.,,;")


# ============================================================
#  هسته محاسباتی (بدون تغییر نسبت به نسخه قبل)
# ============================================================

def parse_series(text: str, name: str) -> np.ndarray:
    """تبدیل متن (با ارقام فارسی/انگلیسی و جداکننده‌های مختلف) به آرایه عددی."""
    tokens = [t for t in re.split(r"[,;\s]+", text.translate(_DIGITS)) if t]
    try:
        return np.array([float(t) for t in tokens])
    except ValueError as exc:
        raise ValueError(f"مقدار نامعتبر در «{name}»: {exc}") from None


def validate_data(a: np.ndarray, rho: np.ndarray):
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(rho)):
        raise ValueError("داده‌ها باید عددی و متناهی باشند؛ مقدار NaN یا Infinity مجاز نیست.")
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


# ============================================================
#  هندسه شبکه، ولتاژهای mesh/step (IEEE 80) و پتانسیل سطحی (بدون تغییر)
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
#  کنترل کفایت داده و توصیه مهندسی (بدون تغییر)
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


# ============================================================
#  اجرای تحلیل و مقادیر پیش‌فرض (بدون تغییر)
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
    # Use the same effective depth everywhere; never calculate with one depth and draw another.
    depth_eff = geom["h"]
    rho_eq = equivalent_resistivity(method, fit, area)
    rg = grid_resistance(rho_eq, area, geom["LT"], depth_eff)
    gpr = ig * rg["sverak"]
    mesh = ieee80_mesh_step(rho_eq, ig, geom)
    hs_eff = hs if use_surface else 0.0
    tv = tolerable_voltages(fit["rho1"], rho_s if use_surface else fit["rho1"], hs_eff, weight, ts)
    inp = dict(area=area, aspect=aspect, spacing=spacing, d_mm=d_mm, rod_mode=rod_mode, rod_len=rod_len,
               length=geom["LT"], depth=depth_eff, ig=ig, ts=ts, weight=weight,
               hs=hs_eff, rho_s=rho_s, real_mode=real_mode)
    try:                                                   # نقشه پتانسیل (اختیاری؛ خطا کل برنامه را نمی‌شکند)
        num = solve_grid_potential(float(area), float(aspect), float(spacing), float(d_mm),
                                   rod_mode, float(rod_len), float(depth_eff))
        num_stats = potential_stats(num, geom, rho_eq, ig, tv)
    except (np.linalg.LinAlgError, MemoryError, ValueError):
        num, num_stats = None, None
    return dict(a=a_vals, rho=rho_vals, fit=fit, rho_eq=rho_eq, rg=rg, geom=geom, mesh=mesh,
                num=num, num_stats=num_stats, rg_design=rg["sverak"], gpr=gpr, tv=tv, inp=inp)


DEFAULTS = dict(
    mode_R=False,
    a_text="1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 50, 60, 80, 100, 120, 150",
    v_text="250, 240, 230, 210, 190, 175, 160, 150, 140, 135, 130, 128, 125, 123, 122, 121",
    area=4000.0, aspect=1.0, spacing=15.0, d_mm=12.0, rod_mode="none", rod_len=3.0,
    depth=0.5, method="apparent",
    ig=5000.0, ts=0.5, weight=70, use_surface=True, rho_s=3000.0, hs=0.1, real_mode=True,
)


# ============================================================
#  هویت بصری: رنگ‌ها و CSS
# ============================================================
# معنای رنگ‌ها (در کل برنامه یکسان است):
#   فیروزه‌ای = ورودی/هندسه‌ی واردشده · سفید-خاکستری = مقدار محاسبه‌شده · خاکستری خط‌چین = حد مرجع
#   کهربایی = هشدار · قرمز = وضعیت بحرانی · سبز = در حد مجاز · آبی ملایم = اطلاعات
C_CYAN, C_AMBER, C_GREEN, C_RED, C_BLUE = "#22d3ee", "#f5a524", "#34d399", "#f87171", "#7aa2f7"
C_VIOLET, C_DIM = "#a78bfa", "#facc15"
C_TEXT, C_MUTE, C_LINE = "#e7edf7", "#93a4c0", "#2a3956"
C_PANEL = "#0f1829"
C_GRID = "rgba(147,164,192,.16)"
FONT = "Vazirmatn, Tahoma, sans-serif"

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;500;700;800&family=JetBrains+Mono:wght@500;700&display=swap');
:root {
  --gs-bg:#0a101c; --gs-bg2:#0d1525; --gs-panel:#111a2d; --gs-panel2:#16223a; --gs-line:#24324d; --gs-line2:#33456a;
  --gs-text:#e7edf7; --gs-mute:#93a4c0; --gs-dim:#6c7d9a;
  --gs-cyan:#22d3ee; --gs-amber:#f5a524; --gs-green:#34d399; --gs-red:#f87171; --gs-blue:#7aa2f7;
  --gs-mono:'JetBrains Mono','IBM Plex Mono',ui-monospace,Menlo,Consolas,monospace;
}
.stApp, [data-testid="stAppViewContainer"] { background: var(--gs-bg); color: var(--gs-text); }
.stApp, .stApp button, .stApp input, .stApp textarea, .stApp label, .stMarkdown { font-family: 'Vazirmatn', Tahoma, sans-serif !important; }
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 3rem; padding-bottom: 2rem; max-width: 1360px; direction: rtl; }
.stApp p, .stApp li, [data-testid="stWidgetLabel"] p, [data-testid="stMarkdownContainer"] { color: var(--gs-text); }
.stMarkdown, .stAlert, [data-testid="stWidgetLabel"] { direction: rtl; text-align: right; }
.stMarkdown p, .stMarkdown li, .stAlert p, [data-testid="stWidgetLabel"] *, .gs-note, .adv-b, .adv-li { direction: rtl; unicode-bidi: plaintext; text-align: right; }
.js-plotly-plot, .stDataFrame, input, textarea, [data-baseweb="select"] { direction: ltr; }
.stMarkdown bdi, .gs-note bdi, .gs-chip bdi, .adv bdi, .rep-row bdi { direction: ltr; unicode-bidi: isolate; display: inline-block; }
.stMarkdown code, .stMarkdown pre { direction: ltr; unicode-bidi: isolate; text-align: left; }
.n { direction: ltr; unicode-bidi: isolate; display: inline-block; font-family: var(--gs-mono); font-weight: 700; }

/* ---- نوار کناری و ورودی‌ها ---- */
[data-testid="stSidebar"] { background: var(--gs-bg2); border-inline-end: 1px solid var(--gs-line); direction: rtl; }
[data-testid="stSidebar"] [data-testid="stForm"] { border: 0; padding: 0; background: transparent; }
[data-testid="stExpander"] { background: var(--gs-panel); border: 1px solid var(--gs-line); border-radius: 6px; }
[data-testid="stExpander"] summary { font-weight: 700; color: var(--gs-text); }
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="textarea"] { background: var(--gs-panel2) !important; border-color: var(--gs-line2) !important; border-radius: 4px !important; }
[data-baseweb="input"] input, [data-baseweb="select"] input, textarea { color: var(--gs-text) !important; -webkit-text-fill-color: var(--gs-text); font-family: var(--gs-mono) !important; }
[data-baseweb="input"]:focus-within, [data-baseweb="select"]:focus-within > div { border-color: var(--gs-cyan) !important; box-shadow: 0 0 0 1px var(--gs-cyan); }
[data-testid="stFormSubmitButton"] button { background: var(--gs-cyan); border: 0; border-radius: 6px; font-weight: 800; color: #04202a; }
[data-testid="stFormSubmitButton"] button p { color: #04202a; }
[data-testid="stFormSubmitButton"] button:hover { filter: brightness(1.1); }
.stDownloadButton button { background: transparent; border: 1px solid var(--gs-line2); border-radius: 6px; font-weight: 700; color: var(--gs-text); }
.stDownloadButton button:hover { border-color: var(--gs-cyan); color: var(--gs-cyan); }
.stDownloadButton button p { color: inherit; }

/* ---- تب‌ها ---- */
.stTabs [data-baseweb="tab-list"] { gap: 2px; border-bottom: 1px solid var(--gs-line); overflow-x: auto; }
.stTabs [data-baseweb="tab"] { padding: 10px 16px; font-weight: 700; color: var(--gs-mute); border-radius: 4px 4px 0 0; white-space: nowrap; }
.stTabs [aria-selected="true"] { color: var(--gs-cyan); }
.stTabs [data-baseweb="tab-highlight"] { background: var(--gs-cyan); }

/* ---- سربرگ ---- */
.gs-head { display:flex; flex-wrap:wrap; justify-content:space-between; align-items:flex-end; gap:12px;
  padding: 10px 0 12px; border-bottom: 1px solid var(--gs-line); margin-bottom: 8px; }
.gs-title { direction:ltr; text-align:left; font-weight:800; font-size:1.35rem; letter-spacing:.02em; color:var(--gs-text); line-height:1.3; }
.gs-title i { font-style:normal; color:var(--gs-cyan); }
.gs-sub { font-size:.82rem; color:var(--gs-mute); margin-top:2px; }
.gs-status { display:flex; flex-wrap:wrap; gap:6px; align-items:center; }
.gs-legend { display:flex; flex-wrap:wrap; gap:14px; font-size:.74rem; color:var(--gs-mute); margin: 4px 0 10px; }
.gs-legend span::before { content:""; display:inline-block; width:9px; height:9px; margin-inline-end:6px; border-radius:2px; background:var(--c); vertical-align:baseline; }

/* ---- برچسب‌ها و یادداشت‌ها ---- */
.gs-chips { display:flex; flex-wrap:wrap; gap:6px; margin: 4px 0 10px; }
.gs-chip { font-size:.76rem; font-weight:700; padding:3px 10px; border-radius:4px; border:1px solid var(--gs-line2); background:var(--gs-panel);
  color:var(--gs-text); direction:rtl; unicode-bidi:plaintext; }
.gs-chip.ok { color:var(--gs-green); border-color:rgba(52,211,153,.5); background:rgba(52,211,153,.08); }
.gs-chip.warn { color:var(--gs-amber); border-color:rgba(245,165,36,.5); background:rgba(245,165,36,.08); }
.gs-chip.bad { color:var(--gs-red); border-color:rgba(248,113,113,.55); background:rgba(248,113,113,.09); }
.gs-chip.input { color:var(--gs-cyan); border-color:rgba(34,211,238,.45); background:rgba(34,211,238,.07); }
.gs-chip.info { color:var(--gs-blue); border-color:rgba(122,162,247,.45); background:rgba(122,162,247,.07); }
.gs-note { padding:9px 14px; border-radius:4px; margin:8px 0; font-size:.88rem; line-height:1.9; border-inline-start:3px solid var(--c, var(--gs-blue)); background:var(--gs-panel); color:var(--gs-text); }
.gs-note.info { --c:var(--gs-blue); } .gs-note.warn { --c:var(--gs-amber); } .gs-note.bad { --c:var(--gs-red); } .gs-note.ok { --c:var(--gs-green); }

/* ---- عنوان بخش ---- */
.gs-sec { display:flex; align-items:baseline; gap:10px; margin: 20px 0 8px; padding-bottom:6px; border-bottom:1px solid var(--gs-line); }
.gs-sec::before { content:""; width:8px; height:8px; background:var(--gs-cyan); align-self:center; }
.gs-sec .t { font-size:1.02rem; font-weight:800; }
.gs-sec .s { font-size:.78rem; color:var(--gs-mute); unicode-bidi:plaintext; }

/* ---- کارت‌های شاخص ---- */
.gs-kpis { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:10px; margin: 6px 0 12px; }
.gs-kpi { background:var(--gs-panel); border:1px solid var(--gs-line); border-top:2px solid var(--c); border-radius:4px; padding:10px 12px; }
.gs-kpi .top { display:flex; justify-content:space-between; align-items:center; gap:8px; }
.gs-kpi .l { font-size:.78rem; color:var(--gs-mute); }
.gs-kpi .tag { font-size:.64rem; font-weight:700; color:var(--c); border:1px solid var(--c); border-radius:3px; padding:0 5px; opacity:.9; white-space:nowrap; }
.gs-kpi .v { font-family:var(--gs-mono); font-weight:700; font-size:1.55rem; direction:ltr; text-align:right; line-height:1.35; margin-top:4px; color:var(--v, var(--gs-text)); }
.gs-kpi .v small { font-family:'Vazirmatn',sans-serif; font-size:.78rem; font-weight:500; color:var(--gs-mute); margin-inline-start:6px; }
.gs-kpi .sub { font-size:.74rem; color:var(--gs-dim); margin-top:1px; unicode-bidi:plaintext; }
.k-input { --c:var(--gs-cyan); } .k-calc { --c:#9fb3d1; } .k-limit { --c:var(--gs-dim); border-style:dashed; border-top-style:solid; }
.s-ok { --v:var(--gs-green); } .s-warn { --v:var(--gs-amber); } .s-bad { --v:var(--gs-red); }

/* ---- کارت‌های کوچک ---- */
.mini-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:8px; margin-bottom:8px; }
.mini { border:1px solid var(--gs-line); border-inline-start:3px solid var(--c, #9fb3d1); border-radius:4px; padding:8px 12px; background:var(--gs-panel); }
.mini.k-input { --c:var(--gs-cyan); } .mini.k-limit { --c:var(--gs-dim); border-style:dashed; border-inline-start-style:solid; }
.mini.hl { background:var(--gs-panel2); }
.mini .l { font-size:.76rem; color:var(--gs-mute); }
.mini .v { font-family:var(--gs-mono); font-size:1.12rem; font-weight:700; direction:ltr; text-align:right; }
.mini .u { font-family:'Vazirmatn',sans-serif; font-size:.76rem; color:var(--gs-mute); margin-inline-start:5px; font-weight:500; }
.mini .tag { font-size:.66rem; font-weight:700; color:var(--gs-cyan); }

/* ---- بنر نتیجه ---- */
.gs-verdict { border:1px solid var(--c); border-inline-start-width:5px; border-radius:4px; padding:12px 16px; margin: 6px 0 10px; background:var(--gs-panel); }
.gs-verdict.ok { --c:var(--gs-green); } .gs-verdict.warn { --c:var(--gs-amber); } .gs-verdict.bad { --c:var(--gs-red); }
.gs-verdict .t { font-weight:800; font-size:1.05rem; color:var(--c); }
.gs-verdict .d { font-size:.86rem; color:var(--gs-text); margin-top:3px; line-height:1.9; }
.track { position:relative; height:8px; border-radius:2px; background:var(--gs-line); margin:14px 0 4px; direction:ltr; }
.fill { height:100%; border-radius:2px; }
.limit { position:absolute; top:-5px; left:66.67%; width:2px; height:18px; background:var(--gs-text); opacity:.8; }
.track-l { display:flex; justify-content:space-between; font-size:.7rem; color:var(--gs-dim); direction:ltr; }

/* ---- جدول مقایسه ---- */
.gs-tw { overflow-x:auto; border:1px solid var(--gs-line); border-radius:4px; }
.gs-table { width:100%; border-collapse:collapse; font-size:.84rem; min-width:620px; }
.gs-table th { background:var(--gs-panel2); color:var(--gs-mute); font-weight:700; text-align:right; padding:8px 10px; border-bottom:1px solid var(--gs-line2); white-space:nowrap; }
.gs-table td { padding:8px 10px; border-bottom:1px solid var(--gs-line); text-align:right; background:var(--gs-panel); }
.gs-table td.num { font-family:var(--gs-mono); direction:ltr; text-align:right; white-space:nowrap; }
.gs-table tr:last-child td { border-bottom:0; }
.gs-pill { font-size:.72rem; font-weight:800; padding:2px 8px; border-radius:3px; border:1px solid var(--c); color:var(--c); white-space:nowrap; }
.gs-pill.ok { --c:var(--gs-green); } .gs-pill.warn { --c:var(--gs-amber); } .gs-pill.bad { --c:var(--gs-red); } .gs-pill.info { --c:var(--gs-blue); }

/* ---- گزارش ---- */
.rep-title { font-size:1.15rem; font-weight:800; direction:rtl; text-align:right; }
.rep-date { font-size:.78rem; color:var(--gs-mute); direction:rtl; text-align:right; margin-bottom:4px; }
.rep-sec { display:flex; align-items:center; gap:10px; margin:18px 0 6px; font-weight:800; font-size:1rem; direction:rtl; }
.rep-sec::before { content:""; width:8px; height:8px; background:var(--gs-cyan); }
.rep-row { display:flex; justify-content:space-between; align-items:center; gap:14px; flex-wrap:wrap; padding:7px 6px; border-bottom:1px dashed var(--gs-line2); direction:rtl; text-align:right; }
.rep-row .k { color:var(--gs-text); direction:rtl; unicode-bidi:plaintext; text-align:right; }
.rep-row .v { font-family:var(--gs-mono); font-weight:700; direction:ltr; unicode-bidi:isolate; text-align:left; }
.rep-row.hl { background:var(--gs-panel2); border-bottom:0; border-radius:4px; }
.rep-verdict { margin-top:16px; padding:12px 16px; border-radius:4px; font-weight:700; direction:rtl; text-align:right; border:1px solid var(--c); border-inline-start-width:5px; background:var(--gs-panel); }
.rep-verdict.ok { --c:var(--gs-green); } .rep-verdict.warn { --c:var(--gs-amber); } .rep-verdict.bad { --c:var(--gs-red); }

/* ---- توصیه مهندسی ---- */
.adv-sum { border-radius:4px; padding:14px 18px; margin:6px 0 12px; direction:rtl; text-align:right; background:var(--gs-panel); border:1px solid var(--gs-cyan); }
.adv-sum-t { font-weight:800; font-size:1.05rem; margin-bottom:6px; }
.adv-prio { margin-top:10px; padding-top:10px; border-top:1px dashed var(--gs-line2); }
.adv { --c:var(--gs-blue); border:1px solid var(--gs-line); border-inline-start:4px solid var(--c); border-radius:4px; padding:12px 16px; margin:8px 0; background:var(--gs-panel); direction:rtl; text-align:right; }
.adv.bad { --c:var(--gs-red); } .adv.warn { --c:var(--gs-amber); } .adv.ok { --c:var(--gs-green); }
.adv-h { display:flex; justify-content:space-between; align-items:center; gap:10px; font-weight:800; }
.adv-ref { font-size:.68rem; font-weight:700; padding:1px 8px; border-radius:3px; border:1px solid var(--gs-line2); color:var(--gs-mute); direction:ltr; white-space:nowrap; }
.adv-b { margin-top:6px; font-size:.9rem; line-height:1.95; }
.adv-li { display:flex; gap:8px; align-items:flex-start; margin-top:6px; font-size:.88rem; line-height:1.9; }
.adv-li > span:first-child { flex:0 0 auto; color:var(--c, var(--gs-cyan)); }
.adv-foot { font-size:.76rem; color:var(--gs-dim); margin-top:12px; direction:rtl; text-align:right; }

/* ---- برند نوار کناری و پاورقی ---- */
.brand { display:flex; align-items:center; gap:10px; margin-bottom:8px; direction:ltr; }
.brand .logo { width:34px; height:34px; border:1.5px solid var(--gs-cyan); border-radius:4px; display:flex; align-items:center; justify-content:center; color:var(--gs-cyan); font-family:var(--gs-mono); font-weight:700; font-size:.8rem; }
.brand-t { font-weight:800; font-size:.95rem; line-height:1.2; } .brand-s { font-size:.7rem; color:var(--gs-mute); }
.gs-foot { text-align:center; font-size:.74rem; color:var(--gs-dim); margin:26px 0 6px; }

/* ---- نمایشگر کوچک ---- */
@media (max-width: 640px) {
  .block-container { padding: 2.6rem .7rem 1.5rem; }
  .gs-title { font-size:1.1rem; }
  .gs-kpis { grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; }
  .gs-kpi .v { font-size:1.25rem; }
  .mini-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
  .stTabs [data-baseweb="tab"] { padding: 8px 10px; font-size:.84rem; }
  .rep-row { flex-direction:column; align-items:flex-start; gap:2px; }
  .rep-row .v { text-align:right; }
}
"""
# ============================================================
#  مقطع خاک (SVG) و مکعب سه‌بعدی (منتقل‌شده از نسخه قبل)
# ============================================================

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


def _box(x0, x1, y0, y1, z0, z1, color, opacity, name) -> go.Mesh3d:
    """مکعب‌مستطیل توپر (۸ رأس، ۱۲ مثلث) برای نمایش لایه‌ها."""
    return go.Mesh3d(
        x=[x0, x1, x1, x0, x0, x1, x1, x0], y=[y0, y0, y1, y1, y0, y0, y1, y1],
        z=[z0, z0, z0, z0, z1, z1, z1, z1],
        i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2], j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
        color=color, opacity=opacity, name=name, showlegend=True, flatshading=True,
        hovertemplate=name + "<extra></extra>")


# ============================================================
#  توابع کمکی نمایش
# ============================================================
PLOT_CFG = {"displaylogo": False, "responsive": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            "toImageButtonOptions": {"format": "png", "scale": 2}}


def H(s: str) -> str:
    """حذف تورفتگی خطوط تا Markdown قطعه HTML را به‌عنوان کد تفسیر نکند."""
    return " ".join(line.strip() for line in s.strip().splitlines())


def num(x: float, fmt: str = ",.1f") -> str:
    return f'<span class="n">{x:{fmt}}</span>'


def section(title: str, sub: str = "") -> None:
    sub_html = f'<span class="s">{sub}</span>' if sub else ""
    st.markdown(H(f'<div class="gs-sec"><span class="t">{title}</span>{sub_html}</div>'), unsafe_allow_html=True)


def note(text: str, kind: str = "info") -> None:
    st.markdown(f'<div class="gs-note {kind}">{text}</div>', unsafe_allow_html=True)


def chip(text: str, kind: str = "neutral") -> str:
    # اعداد، درصدها و واحدها در متن راست‌به‌چپ جدا (isolate) می‌شوند تا ترتیبشان به‌هم نریزد.
    text = re.sub(r'(?<![\w.])([+−-]?\d[\d,]*(?:\.\d+)?(?:\s*[٪%])?)', r'<bdi dir="ltr">\1</bdi>', text)
    return f'<span class="gs-chip {kind}" dir="rtl">{text}</span>'


def chips(items) -> None:
    st.markdown('<div class="gs-chips">' + "".join(items) + "</div>", unsafe_allow_html=True)


_KIND_TAG = {"input": "ورودی", "calc": "محاسبه", "limit": "حد مرجع"}


def kpi(label: str, value: str, unit: str = "", sub: str = "", kind: str = "calc", state: str = "") -> str:
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    return (f'<div class="gs-kpi k-{kind} s-{state}"><div class="top"><span class="l">{label}</span>'
            f'<span class="tag">{_KIND_TAG[kind]}</span></div>'
            f'<div class="v">{value}<small>{unit}</small></div>{sub_html}</div>')


def kpi_grid(cards) -> None:
    st.markdown(H('<div class="gs-kpis">' + "".join(cards) + "</div>"), unsafe_allow_html=True)


def mini_card(label: str, value: str, unit: str = "", highlight: bool = False, tag: str = "", kind: str = "calc") -> str:
    tag_html = f'<div class="tag">{tag}</div>' if tag else ""
    return (f'<div class="mini k-{kind}{" hl" if highlight else ""}">{tag_html}<div class="l">{label}</div>'
            f'<div class="v">{value}<span class="u">{unit}</span></div></div>')


def mini_grid(cards) -> None:
    st.markdown(H('<div class="mini-grid">' + "".join(cards) + "</div>"), unsafe_allow_html=True)


def show_fig(fig: go.Figure) -> None:
    st.plotly_chart(fig, use_container_width=True, theme=None, config=PLOT_CFG)


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def style_fig(fig: go.Figure, height: int = 380) -> go.Figure:
    """سبک یکسان تمام نمودارهای دوبعدی: پس‌زمینه، فونت، شبکه، راهنما و hover."""
    fig.update_layout(
        template="plotly_dark", height=height, margin=dict(l=12, r=12, t=40, b=12),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=C_PANEL,
        font=dict(family=FONT, size=12, color=C_TEXT),
        hoverlabel=dict(font_family=FONT, bgcolor="#16223a", bordercolor=C_LINE),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
    )
    ax = dict(gridcolor=C_GRID, zeroline=False, linecolor=C_LINE, showline=True, mirror=True,
              ticks="outside", tickcolor=C_LINE, tickfont=dict(size=11, color=C_MUTE))
    fig.update_xaxes(**ax)
    fig.update_yaxes(**ax)
    return fig


# ---------- برازش خاک ----------
def fit_fig(a, rho, fit) -> go.Figure:
    a_fine = np.logspace(np.log10(a.min()), np.log10(a.max()), 200)
    model = rho_apparent_two_layer(a_fine, fit["rho1"], fit["rho2"], fit["h"])
    top1 = fit["rho1"] >= fit["rho2"]
    fig = go.Figure()
    fig.add_hline(y=fit["rho1"], line_dash="dot", line_color=C_CYAN, opacity=.55, annotation_text="ρ₁ (a→0)",
                  annotation_font_color=C_CYAN, annotation_position="top left" if top1 else "bottom left")
    fig.add_hline(y=fit["rho2"], line_dash="dot", line_color=C_VIOLET, opacity=.55, annotation_text="ρ₂ (a→∞)",
                  annotation_font_color=C_VIOLET, annotation_position="bottom right" if top1 else "top right")
    fig.add_trace(go.Scatter(x=a_fine, y=model, mode="lines", name="مدل دولایه (برازش)",
                             line=dict(color=C_CYAN, width=3),
                             hovertemplate="a = %{x:.2f} m<br>ρa مدل = %{y:,.1f} Ω·m<extra></extra>"))
    fig.add_trace(go.Scatter(x=a, y=rho, mode="markers", name="اندازه‌گیری",
                             marker=dict(size=9, color="#f8fafc", line=dict(width=2, color="#0a101c")),
                             hovertemplate="a = %{x:.2f} m<br>ρa اندازه‌گیری = %{y:,.1f} Ω·m<extra></extra>"))
    fig.update_xaxes(type="log", title_text="فاصله الکترودها a (m)")
    fig.update_yaxes(title_text="مقاومت ظاهری ρa (Ω·m)")
    return style_fig(fig, 400)


def residual_fig(a, err) -> go.Figure:
    colors = [C_GREEN if abs(e) <= 5 else C_AMBER if abs(e) <= 10 else C_RED for e in err]
    fig = go.Figure(go.Bar(x=[f"{v:g}" for v in a], y=err, marker_color=colors,
                           text=[f"{e:+.1f}" for e in err], textposition="outside", cliponaxis=False,
                           hovertemplate="a = %{x} m<br>خطای نسبی = %{y:.2f} %<extra></extra>"))
    fig.add_hrect(y0=-10, y1=10, fillcolor="rgba(52,211,153,.08)", line_width=0)
    fig.update_xaxes(title_text="فاصله الکترودها a (m)", type="category")
    fig.update_yaxes(title_text="خطای نسبی مدل نسبت به اندازه‌گیری (٪)")
    return style_fig(fig, 300)


# ---------- مقطع واقع‌گرایانه خاک (SVG، دقیق و بدون نیاز به اینترنت) ----------
#  (توابع _STOPS / _soil_rgb / _hex / _texture / soil_section_svg عیناً از نسخه قبل منتقل شده‌اند.)


def safety_fig(gpr: float, tv: dict, em: float, es: float) -> go.Figure:
    labels = ["GPR", "Em (mesh)", "Es (step)"]
    vals = [gpr, em, es]
    limits = [tv["e_touch"], tv["e_touch"], tv["e_step"]]
    mesh_ok = em <= tv["e_touch"] and es <= tv["e_step"]
    gpr_col = C_GREEN if gpr <= tv["e_touch"] else (C_AMBER if mesh_ok else C_RED)
    colors = [gpr_col, C_GREEN if em <= limits[1] else C_RED, C_GREEN if es <= limits[2] else C_RED]
    fig = go.Figure(go.Bar(x=vals, y=labels, orientation="h", marker_color=colors, name="مقدار محاسبه‌شده",
                           text=[f"{v:,.0f} V" for v in vals], textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}: %{x:,.1f} V<extra></extra>"))
    fig.add_trace(go.Scatter(x=limits, y=labels, mode="markers", name="حد مجاز",
                             marker=dict(symbol="line-ns", size=28, line=dict(width=4, color="#f8fafc")),
                             hovertemplate="حد مجاز: %{x:,.0f} V<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(range=[0, max(vals + limits) * 1.28], title_text="ولتاژ (V)")
    return style_fig(fig, 270)


# ---------- نمای سه‌بعدی هندسه شبکه ----------
SCENE_BG = "#0d1626"
VIEWS = {
    "ایزومتریک": dict(eye=dict(x=1.35, y=-1.55, z=0.85), up=dict(x=0, y=0, z=1), projection=dict(type="perspective")),
    "بالا (Top)": dict(eye=dict(x=0.0, y=0.0, z=2.4), up=dict(x=0, y=1, z=0), projection=dict(type="orthographic")),
    "روبه‌رو (Front)": dict(eye=dict(x=0.0, y=-2.4, z=0.0), up=dict(x=0, y=0, z=1), projection=dict(type="orthographic")),
    "جانبی (Side)": dict(eye=dict(x=2.4, y=0.0, z=0.0), up=dict(x=0, y=0, z=1), projection=dict(type="orthographic")),
}


def _scene_axis(title: str, **kw) -> dict:
    return dict(title=dict(text=title, font=dict(size=12, color=C_MUTE)), showbackground=True, backgroundcolor=SCENE_BG,
                gridcolor="rgba(147,164,192,.22)", zerolinecolor="rgba(147,164,192,.35)", linecolor=C_LINE,
                tickfont=dict(size=10, color=C_MUTE), showspikes=False, **kw)


def _dim3d(p0, p1, text, tick_dir, tick_len, ext=(), color=C_DIM):
    """خط اندازه‌گذاری سه‌بعدی: خط اصلی + تیک انتها + خطوط امتداد + برچسب عددی (مقدار واقعی در متن)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    td = np.asarray(tick_dir, float) * tick_len
    xs, ys, zs = [], [], []

    def seg(a, b):
        xs.extend([a[0], b[0], None]); ys.extend([a[1], b[1], None]); zs.extend([a[2], b[2], None])

    seg(p0, p1); seg(p0 - td, p0 + td); seg(p1 - td, p1 + td)
    for a, b in ext:
        seg(np.asarray(a, float), np.asarray(b, float))
    mid = (p0 + p1) / 2.0
    return [go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=color, width=3), hoverinfo="skip", showlegend=False),
            go.Scatter3d(x=[mid[0]], y=[mid[1]], z=[mid[2]], mode="text", text=[text], textposition="top center",
                         textfont=dict(color=color, size=12, family=FONT), hoverinfo="skip", showlegend=False)]


def fig_3d_grid(fit: dict, g: dict, hs: float, stt: dict | None, exag: float = 1.0, view: str = "ایزومتریک",
                show_dims: bool = True, show_soil: bool = True, show_crit: bool = True, height: int = 560) -> go.Figure:
    """نمای سه‌بعدی مهندسی: هندسه واقعی شبکه، سطح زمین Z=0، اندازه‌گذاری و نماهای از پیش تنظیم‌شده.
    فقط مقادیر موجود در مدل رسم می‌شوند. اگر exag≠1 باشد فقط محور Z برای نمایش کشیده می‌شود؛
    برچسب‌های عددی و تیک‌های محور Z همیشه مقدار واقعی را نشان می‌دهند."""
    E = float(exag)
    Lx, Ly = g["Lx"], g["Ly"]
    L = max(Lx, Ly)
    off = 0.07 * L
    hl = fit["h"]                                    # ضخامت لایه اول خاک
    zg = -g["h"]                                     # عمق دفن هادی‌ها (منفی = زیر سطح زمین)
    Lr = g["Lr"] if g["n_rods"] else 0.0
    zmax = max(hl * 2.0, hl + 3.0, g["h"] + g["Lr"] + 1.0)
    zp = lambda z: z * E                             # تبدیل مختصات واقعی به مختصات رسم
    fig = go.Figure()

    # لایه‌های خاک (ضخامت‌های واقعی مدل)
    if show_soil:
        c1, c2 = _hex(_soil_rgb(fit["rho1"])), _hex(_soil_rgb(fit["rho2"]))
        fig.add_trace(_box(0, Lx, 0, Ly, zp(-zmax), zp(-hl), c2, 0.16, f"لایه دوم · ρ₂ = {fit['rho2']:,.0f} Ω·m"))
        fig.add_trace(_box(0, Lx, 0, Ly, zp(-hl), 0, c1, 0.20, f"لایه اول · ρ₁ = {fit['rho1']:,.0f} Ω·m · ضخامت {hl:.2f} m"))
        if hs > 0:
            fig.add_trace(_box(0, Lx, 0, Ly, 0, zp(hs), "#9aa0a6", 0.55, f"لایه سطحی · ضخامت {hs:g} m"))

    # صفحه مرجع سطح زمین (Z = 0)
    fig.add_trace(go.Mesh3d(x=[0, Lx, Lx, 0], y=[0, 0, Ly, Ly], z=[0, 0, 0, 0], i=[0, 0], j=[1, 2], k=[2, 3],
                            color="#38bdf8", opacity=0.12, name="سطح زمین (Z = 0)", showlegend=True, hoverinfo="skip"))
    fig.add_trace(go.Scatter3d(x=[0, Lx, Lx, 0, 0], y=[0, 0, Ly, Ly, 0], z=[0] * 5, mode="lines",
                               line=dict(color="#38bdf8", width=3), hoverinfo="skip", showlegend=False))

    # هادی‌های افقی در عمق دفن
    xs, ys, zs = [], [], []
    for y in g["ys"]:
        xs += [0, Lx, None]; ys += [y, y, None]; zs += [zp(zg), zp(zg), None]
    for x in g["xs"]:
        xs += [x, x, None]; ys += [0, Ly, None]; zs += [zp(zg), zp(zg), None]
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=C_CYAN, width=5),
                               name=f"هادی‌های افقی ({g['nx']}×{g['ny']} ، Dx = {g['Dx']:.2f} m ، Dy = {g['Dy']:.2f} m ، عمق {g['h']:g} m)",
                               hoverinfo="skip"))
    # میله‌های عمودی
    if g["n_rods"]:
        rx, ry, rz = [], [], []
        for x, y in g["rods"]:
            rx += [x, x, None]; ry += [y, y, None]; rz += [zp(zg), zp(zg - Lr), None]
        fig.add_trace(go.Scatter3d(x=rx, y=ry, z=rz, mode="lines", line=dict(color="#f1f5f9", width=7),
                                   name=f"میله‌های زمین ({g['n_rods']} عدد × {Lr:g} m)", hoverinfo="skip"))
        fig.add_trace(go.Scatter3d(x=g["rods"][:, 0], y=g["rods"][:, 1], z=[zp(zg)] * g["n_rods"], mode="markers",
                                   marker=dict(size=3, color="#f1f5f9"), showlegend=False,
                                   hovertemplate="میله زمین<br>x = %{x:.2f} m<br>y = %{y:.2f} m"
                                                 f"<br>طول = {Lr:g} m<extra></extra>"))

    # نقاط بحرانی (فقط اگر تحلیل عددی انجام شده باشد)
    if show_crit and stt:
        fig.add_trace(go.Scatter3d(x=[stt["touch_at"][0]], y=[stt["touch_at"][1]], z=[0.0], mode="markers",
                                   marker=dict(size=6, color=C_RED, symbol="diamond", line=dict(width=1, color="#fff")),
                                   name="بیشینه ولتاژ تماس (تحلیل عددی)", customdata=[stt["touch_max"]],
                                   hovertemplate="بیشینه ولتاژ تماس = %{customdata:,.0f} V<br>x = %{x:.1f} m ، y = %{y:.1f} m<extra></extra>"))
        fig.add_trace(go.Scatter3d(x=[stt["step_at"][0]], y=[stt["step_at"][1]], z=[0.0], mode="markers",
                                   marker=dict(size=6, color=C_AMBER, symbol="diamond", line=dict(width=1, color="#fff")),
                                   name="بیشینه ولتاژ گام (تحلیل عددی)", customdata=[stt["step_max"]],
                                   hovertemplate="بیشینه ولتاژ گام = %{customdata:,.0f} V<br>x = %{x:.1f} m ، y = %{y:.1f} m<extra></extra>"))

    # اندازه‌گذاری
    tl = 0.30 * off
    if show_dims:
        yd, xo, xd = -off, Lx + off, Lx + 2.3 * off
        for tr in _dim3d([0, yd, 0], [Lx, yd, 0], f"طول Lx = {Lx:,.1f} m", (0, 1, 0), tl,
                         ext=[([0, 0, 0], [0, yd, 0]), ([Lx, 0, 0], [Lx, yd, 0])]):
            fig.add_trace(tr)
        for tr in _dim3d([xd, 0, 0], [xd, Ly, 0], f"عرض Ly = {Ly:,.1f} m", (1, 0, 0), tl,
                         ext=[([Lx, 0, 0], [xd, 0, 0]), ([Lx, Ly, 0], [xd, Ly, 0])]):
            fig.add_trace(tr)
        for tr in _dim3d([xo, 0, 0], [xo, 0, zp(zg)], f"عمق دفن = {g['h']:g} m", (1, 0, 0), tl,
                         ext=[([Lx, 0, zp(zg)], [xo, 0, zp(zg)])]):
            fig.add_trace(tr)
        if Lr > 0:
            for tr in _dim3d([xo, 0, zp(zg)], [xo, 0, zp(zg - Lr)], f"طول میله = {Lr:g} m", (1, 0, 0), tl,
                             ext=[([Lx, 0, zp(zg - Lr)], [xo, 0, zp(zg - Lr)])]):
                fig.add_trace(tr)
        yx, xy = -0.45 * off, -0.45 * off
        for tr in _dim3d([g["xs"][0], yx, zp(zg)], [g["xs"][1], yx, zp(zg)], f"Dx = {g['Dx']:.2f} m", (0, 1, 0), 0.5 * tl,
                         ext=[([g["xs"][0], 0, zp(zg)], [g["xs"][0], yx, zp(zg)]), ([g["xs"][1], 0, zp(zg)], [g["xs"][1], yx, zp(zg)])]):
            fig.add_trace(tr)
        for tr in _dim3d([xy, g["ys"][0], zp(zg)], [xy, g["ys"][1], zp(zg)], f"Dy = {g['Dy']:.2f} m", (1, 0, 0), 0.5 * tl,
                         ext=[([0, g["ys"][0], zp(zg)], [xy, g["ys"][0], zp(zg)]), ([0, g["ys"][1], zp(zg)], [xy, g["ys"][1], zp(zg)])]):
            fig.add_trace(tr)

    # نشانگر جهت مثبت محورها (سه‌پایه مرجع؛ طول فلش‌ها فقط نمادین است)
    o, ln, lz = np.array([-2.6 * off, -2.6 * off, 0.0]), 1.5 * off, 0.03 * L
    tri = [("+X", (ln, 0, 0)), ("+Y", (0, ln, 0)), ("+Z", (0, 0, lz))]
    for lab, d in tri:
        tip = o + np.array(d)
        fig.add_trace(go.Scatter3d(x=[o[0], tip[0]], y=[o[1], tip[1]], z=[o[2], tip[2]], mode="lines+text",
                                   text=["", lab], textposition="top center", line=dict(color="#cbd5e1", width=4),
                                   textfont=dict(color="#cbd5e1", size=12), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter3d(x=[0], y=[Ly + 0.3 * off], z=[0], mode="text", text=["سطح زمین (Z = 0)"],
                               textposition="top right", textfont=dict(color="#38bdf8", size=11, family=FONT),
                               hoverinfo="skip", showlegend=False))

    # ticks محور Z با مقادیر واقعی (سطح زمین، عمق دفن، نوک میله، کف دامنه)
    cand = [0.0, -zmax] + ([zg - Lr] if Lr > 0 else []) + [zg, -hl]
    picked: list = []
    for t in cand:
        if all(abs(t - p) >= 0.12 * zmax for p in picked):
            picked.append(t)
    picked.sort()
    ztop = max(hs, 0.0) * E + 0.05 * L
    cam = VIEWS.get(view, VIEWS["ایزومتریک"])
    fig.update_layout(
        template="plotly_dark", height=height, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12, color=C_TEXT), hoverlabel=dict(font_family=FONT, bgcolor="#16223a"),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, font=dict(size=11)),
        uirevision=view,
        scene=dict(
            xaxis=_scene_axis("X (m) → طول", range=[-3.4 * off, Lx + 3.6 * off]),
            yaxis=_scene_axis("Y (m) → عرض", range=[-3.4 * off, Ly + 2.2 * off]),
            zaxis=_scene_axis("Z (m) ↑ (عمق: منفی)", range=[-zmax * E, ztop],
                              tickvals=[zp(t) for t in picked], ticktext=[f"{t:g}" for t in picked]),
            aspectmode="data", camera=cam))
    if E != 1.0:
        fig.add_annotation(text=f"مقیاس محور Z برای خوانایی ×{E:g} اغراق شده است؛ اعداد و برچسب‌ها مقدار واقعی‌اند.",
                           xref="paper", yref="paper", x=0.0, y=0.0, xanchor="left", yanchor="bottom", showarrow=False,
                           font=dict(size=11, color=C_AMBER), bgcolor="rgba(10,16,28,.8)")
    return fig


def _dim2d(fig, x0, y0, x1, y1, text, color=C_DIM, xanchor="center", yanchor="bottom", tx=None, ty=None):
    fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y", text="", showarrow=True,
                       arrowhead=2, arrowsize=1, arrowwidth=1.5, arrowcolor=color, arrowside="end+start")
    fig.add_annotation(x=(x0 + x1) / 2.0 if tx is None else tx, y=(y0 + y1) / 2.0 if ty is None else ty, text=text,
                       showarrow=False, font=dict(color=color, size=12), xanchor=xanchor, yanchor=yanchor,
                       bgcolor="rgba(10,16,28,.75)")


def fig_elevation(fit: dict, g: dict, hs: float, exag: float = 1.0) -> go.Figure:
    """نمای برش (نمای روبه‌رو) در صفحه Y = 0: سطح زمین، عمق دفن، میله‌ها و مرز لایه‌ها با اندازه‌گذاری."""
    E = float(exag)
    Lx, hl, zg = g["Lx"], fit["h"], -g["h"]
    Lr = g["Lr"] if g["n_rods"] else 0.0
    zmax = max(hl * 2.0, hl + 3.0, g["h"] + g["Lr"] + 1.0)
    c1, c2 = _hex(_soil_rgb(fit["rho1"])), _hex(_soil_rgb(fit["rho2"]))
    xl, xr = -0.06 * Lx, 1.42 * Lx
    ydim = max(hs, 0.0) + 0.05 * Lx / E
    fig = go.Figure()
    fig.add_shape(type="rect", x0=xl, x1=xr, y0=-zmax, y1=-hl, fillcolor=_rgba(c2, .35), line_width=0, layer="below")
    fig.add_shape(type="rect", x0=xl, x1=xr, y0=-hl, y1=0, fillcolor=_rgba(c1, .38), line_width=0, layer="below")
    if hs > 0:
        fig.add_shape(type="rect", x0=xl, x1=xr, y0=0, y1=hs, fillcolor="rgba(154,160,166,.55)", line_width=0, layer="below")
    fig.add_trace(go.Scatter(x=[xl, xr], y=[0, 0], mode="lines", line=dict(color="#38bdf8", width=2.5),
                             name="سطح زمین (Z = 0)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[xl, xr], y=[-hl, -hl], mode="lines", line=dict(color=C_MUTE, width=1.5, dash="dash"),
                             name=f"مرز دو لایه خاک (ضخامت لایه اول {hl:.2f} m)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[0, Lx], y=[zg, zg], mode="lines", line=dict(color=C_CYAN, width=4),
                             name=f"هادی افقی (عمق دفن {g['h']:g} m)",
                             hovertemplate="هادی افقی<br>عمق = " + f"{g['h']:g}" + " m<extra></extra>"))
    rods0 = g["rods"][np.isclose(g["rods"][:, 1], 0.0)] if g["n_rods"] else np.empty((0, 2))
    if len(rods0):
        rx, rz = [], []
        for x in rods0[:, 0]:
            rx += [x, x, None]; rz += [zg, zg - Lr, None]
        fig.add_trace(go.Scatter(x=rx, y=rz, mode="lines", line=dict(color="#f1f5f9", width=5),
                                 name=f"میله‌های واقع در Y = 0 ({len(rods0)} عدد × {Lr:g} m)",
                                 hovertemplate="میله زمین<extra></extra>"))
    # اندازه‌گذاری
    _dim2d(fig, 0, ydim, Lx, ydim, f"Lx = {Lx:,.1f} m")
    xa, xb = 1.05 * Lx, 1.17 * Lx
    _dim2d(fig, xa, 0, xa, zg, f"عمق دفن = {g['h']:g} m", xanchor="left", yanchor="middle", tx=xa + 0.012 * Lx)
    if Lr > 0:
        _dim2d(fig, xa, zg, xa, zg - Lr, f"طول میله = {Lr:g} m", xanchor="left", yanchor="middle", tx=xa + 0.012 * Lx)
    _dim2d(fig, xb, 0, xb, -hl, f"ضخامت لایه اول = {hl:.2f} m", xanchor="left", yanchor="middle", tx=xb + 0.012 * Lx)
    fig.update_xaxes(range=[xl, xr], title_text="X (m) → طول زمین")
    fig.update_yaxes(range=[-zmax - 0.02 * Lx / E, ydim + 0.04 * Lx / E], scaleanchor="x", scaleratio=E,
                     title_text="Z (m) — سطح زمین = 0")
    style_fig(fig, 340)
    fig.update_layout(margin=dict(l=12, r=12, t=60, b=12))
    return fig


# ---------- نقشه‌های پتانسیل، ولتاژ تماس و گام ----------
def _field_cfg(view: str, stt: dict, tv: dict):
    """(داده، واحد، حد مجاز، طیف رنگ، zmin، zmax) برای هر نمای نقشه."""
    return {
        "ولتاژ تماس": (stt["touch"], "V", tv["e_touch"], "RdYlGn_r", 0.0, 2.0 * tv["e_touch"]),
        "ولتاژ گام": (stt["step"], "V", tv["e_step"], "RdYlGn_r", 0.0, 2.0 * tv["e_step"]),
        "پتانسیل سطح": (stt["V"], "V", None, "Turbo", 0.0, stt["gpr"]),
        "گرادیان پتانسیل": (stt["grad"], "V/m", tv["e_step"], "Turbo", 0.0, None),
    }[view]


def _grid_lines(g: dict):
    lx, ly = [], []
    for y in g["ys"]:
        lx += [0, g["Lx"], None]; ly += [y, y, None]
    for x in g["xs"]:
        lx += [x, x, None]; ly += [0, g["Ly"], None]
    return lx, ly


def potential_surface_fig(view: str, num: dict, stt: dict, g: dict, tv: dict) -> go.Figure:
    """نمای سه‌بعدی (سطح) پتانسیل / ولتاژ تماس / گام / گرادیان. محور Z کمیت تحلیلی است، نه ارتفاع هندسی."""
    Z, unit, limit, cscale, zmin, zmax = _field_cfg(view, stt, tv)
    k = max(1, int(np.ceil(max(Z.shape) / 110)))              # سبک‌سازی برای موبایل
    xs, ys, Zs = num["xs"][::k], num["ys"][::k], Z[::k, ::k]
    top = float(max(Zs.max(), limit or 0.0)) * 1.08
    kw = dict(cmin=zmin) if zmin is not None else {}
    if zmax is not None:
        kw["cmax"] = zmax
    fig = go.Figure(go.Surface(
        x=xs, y=ys, z=Zs, colorscale=cscale, colorbar=dict(title=dict(text=unit), tickfont=dict(color=C_MUTE)), name=view, **kw,
        contours=dict(z=dict(show=True, usecolormap=True, project=dict(z=True))),
        hovertemplate="x = %{x:.1f} m<br>y = %{y:.1f} m<br>%{z:,.1f} " + unit + "<extra></extra>"))
    if limit is not None:
        fig.add_trace(go.Surface(x=[xs[0], xs[-1]], y=[ys[0], ys[-1]], z=[[limit, limit], [limit, limit]],
                                 opacity=0.35, showscale=False, colorscale=[[0, C_RED], [1, C_RED]],
                                 name=f"حد مجاز ({limit:,.0f} {unit})", showlegend=True, hoverinfo="skip"))
    lx, ly = _grid_lines(g)
    fig.add_trace(go.Scatter3d(x=lx, y=ly, z=[0.0] * len(lx), mode="lines", name="هادی‌های شبکه (روی کف)",
                               line=dict(color="#f8fafc", width=3), hoverinfo="skip"))
    ex, ey = float(xs[-1] - xs[0]), float(ys[-1] - ys[0])
    m = max(ex, ey)
    fig.update_layout(
        template="plotly_dark", height=560, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12, color=C_TEXT),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, font=dict(size=11)),
        scene=dict(xaxis=_scene_axis("X (m)"), yaxis=_scene_axis("Y (m)"),
                   zaxis=_scene_axis(f"{view} ({unit}) — کمیت تحلیلی", range=[0, top]),
                   aspectmode="manual", aspectratio=dict(x=ex / m, y=ey / m, z=0.5),
                   camera=dict(eye=dict(x=1.6, y=-1.6, z=1.0))))
    return fig


def potential_map_fig(view: str, num: dict, stt: dict, g: dict, tv: dict) -> go.Figure:
    """نقشه رنگی با مقیاس ثابت، خط حد مجاز، جانمایی هادی‌ها و نقطه بحرانی محاسبه‌شده."""
    xs, ys = num["xs"], num["ys"]
    Z, unit, limit, cscale, zmin, zmax = _field_cfg(view, stt, tv)
    fig = go.Figure(go.Heatmap(x=xs, y=ys, z=Z, colorscale=cscale, zmin=zmin, zmax=zmax, zsmooth="best",
                               colorbar=dict(title=dict(text=unit), tickfont=dict(color=C_MUTE)), name=view,
                               hovertemplate="x = %{x:.1f} m<br>y = %{y:.1f} m<br>%{z:,.1f} " + unit + "<extra></extra>"))
    if limit is not None and float(np.max(Z)) > limit:
        fig.add_trace(go.Contour(x=xs, y=ys, z=Z, showscale=False, showlegend=True, hoverinfo="skip",
                                 name=f"حد مجاز ({limit:,.0f} {unit})", line=dict(color="#050a14", width=3),
                                 contours=dict(start=limit, end=limit, size=max(limit, 1.0), coloring="lines")))
    lx, ly = _grid_lines(g)
    fig.add_trace(go.Scatter(x=lx, y=ly, mode="lines", line=dict(color="rgba(5,10,20,.55)", width=1),
                             name="هادی‌های شبکه", hoverinfo="skip"))
    if g["n_rods"]:
        fig.add_trace(go.Scatter(x=g["rods"][:, 0], y=g["rods"][:, 1], mode="markers", name="میله‌های زمین",
                                 marker=dict(symbol="x", size=8, color="#050a14"), hoverinfo="skip"))
    mark = {"ولتاژ تماس": (stt["touch_at"], stt["touch_max"], "V", "بیشینه ولتاژ تماس (داخل شبکه)"),
            "ولتاژ گام": (stt["step_at"], stt["step_max"], "V", "بیشینه ولتاژ گام (ناحیه نزدیک)"),
            "گرادیان پتانسیل": (stt["step_at"], stt["step_max"], "V/m", "بیشینه گرادیان (ناحیه نزدیک)")}.get(view)
    if mark:
        (mx, my), mv, mu, mn = mark
        fig.add_trace(go.Scatter(x=[mx], y=[my], mode="markers", name=f"{mn}: {mv:,.0f} {mu}",
                                 marker=dict(symbol="star", size=15, color="#f8fafc", line=dict(width=2, color="#050a14")),
                                 hovertemplate=f"{mn}<br>{mv:,.1f} {mu}<br>x = {mx:.1f} m ، y = {my:.1f} m<extra></extra>"))
    fig.update_xaxes(title_text="X (m)", constrain="domain")
    fig.update_yaxes(title_text="Y (m)", scaleanchor="x", scaleratio=1)
    return style_fig(fig, 520)


def profile_figs(num: dict, stt: dict, g: dict, tv: dict, y_frac: float):
    """نیمرخ پتانسیل و ولتاژهای تماس/گام در امتداد X در موقعیت دلخواه Y."""
    ys = num["ys"]
    y_target = g["Ly"] * y_frac
    j = int(np.argmin(np.abs(ys - y_target)))
    xs = num["xs"]
    f1 = go.Figure(go.Scatter(x=xs, y=stt["V"][j], mode="lines", name="پتانسیل سطح", line=dict(color=C_CYAN, width=3),
                              hovertemplate="x = %{x:.1f} m<br>%{y:,.1f} V<extra></extra>"))
    f1.add_hline(y=stt["gpr"], line_dash="dot", line_color=C_AMBER, annotation_text="GPR (تحلیل عددی)",
                 annotation_font_color=C_AMBER, annotation_position="top left")
    f1.add_vrect(x0=0, x1=g["Lx"], fillcolor="rgba(147,164,192,.10)", line_width=0)
    f1.update_xaxes(title_text="X (m)"); f1.update_yaxes(title_text="پتانسیل سطح (V)")
    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=xs, y=stt["touch"][j], mode="lines", name="ولتاژ تماس", line=dict(color=C_RED, width=3),
                            hovertemplate="x = %{x:.1f} m<br>تماس = %{y:,.1f} V<extra></extra>"))
    f2.add_trace(go.Scatter(x=xs, y=stt["step"][j], mode="lines", name="ولتاژ گام (گرادیان × ۱ m)", line=dict(color=C_VIOLET, width=3),
                            hovertemplate="x = %{x:.1f} m<br>گام = %{y:,.1f} V<extra></extra>"))
    f2.add_hline(y=tv["e_touch"], line_dash="dash", line_color=C_RED, opacity=.65,
                 annotation_text="حد تماس", annotation_font_color=C_RED, annotation_position="top left")
    f2.add_hline(y=tv["e_step"], line_dash="dash", line_color=C_VIOLET, opacity=.65,
                 annotation_text="حد گام", annotation_font_color=C_VIOLET, annotation_position="top right")
    f2.add_vrect(x0=0, x1=g["Lx"], fillcolor="rgba(147,164,192,.10)", line_width=0)
    f2.update_xaxes(title_text="X (m)"); f2.update_yaxes(title_text="ولتاژ (V)")
    return style_fig(f1, 300), style_fig(f2, 300), float(ys[j])

# ============================================================
#  گزارش و توصیه (متن؛ فقط نشانگرهای ایموجی به نمادهای ساده تبدیل شده‌اند)
# ============================================================
def build_report(inp: dict, fit: dict, rho_eq: float, rg: dict, rg_design: float,
                 gpr: float, tv: dict, ex: dict | None = None) -> str:
    mesh_ok = bool(ex) and ex["mesh"]["Em"] <= tv["e_touch"] and ex["mesh"]["Es"] <= tv["e_step"]
    if gpr <= tv["e_touch"]:
        verdict = "✔ GPR کمتر از ولتاژ تماس مجاز است؛ تحلیل ولتاژهای mesh و step الزامی نیست."
    elif mesh_ok:
        verdict = "▲ GPR از ولتاژ تماس مجاز بیشتر است، اما Em و Es در حد مجازند؛ ولتاژ منتقل‌شده را کنترل کنید."
    elif ex:
        verdict = "✖ Em یا Es از حد مجاز بیشتر است؛ هندسه شبکه، لایه سطحی، Ig یا زمان قطع باید اصلاح شود."
    else:
        verdict = "▲ GPR بیشتر از ولتاژ تماس مجاز است؛ محاسبه Em و Es و اصلاح طرح لازم است."
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
        f'<div class="rep-verdict {vcls}">{ {"ok": "✔", "warn": "▲", "bad": "✖"}[vcls] } {verdict}</div>',
    ]
    return "".join(parts)


_ICON = {"ok": "✔", "warn": "▲", "bad": "✖", "info": "●"}


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
#  اجزای نمای کلی: سربرگ، نتیجه‌گیری، جدول مقایسه با حدود مجاز
# ============================================================
METHOD_LABEL = {"apparent": "ρ معادل: ρa مدل در شعاع معادل", "rho1": "ρ معادل: فقط ρ₁", "mean": "ρ معادل: میانگین ρ₁ و ρ₂"}


def verdict_state(gpr: float, tv: dict, mesh: dict):
    """همان منطق نتیجه‌گیری نسخه قبل: GPR ≤ ولتاژ تماس مجاز ← ایمن؛ در غیر این صورت بررسی Em و Es."""
    ratio = gpr / tv["e_touch"]
    mesh_ok = mesh["Em"] <= tv["e_touch"] and mesh["Es"] <= tv["e_step"]
    if ratio <= 1.0:
        return "ok", "طرح از نظر GPR ایمن است", "طبق IEEE 80 در این حالت تحلیل ولتاژهای mesh و step الزامی نیست."
    if mesh_ok:
        return "warn", "GPR بالاست، اما Em و Es مجازند", \
            "ولتاژهای تماس و گام در حد مجازند؛ ولتاژ منتقل‌شده به حصار، لوله و کابل‌ها را کنترل کنید."
    return "bad", "Em یا Es از حد مجاز بیشتر است", "هندسه شبکه، میله‌ها، لایه سطحی، Ig یا زمان قطع را اصلاح کنید."


def header_html(out: dict, is_sample: bool, vstate: str, vtitle: str) -> str:
    inp, fit, stt = out["inp"], out["fit"], out["num_stats"]
    q_kind = "ok" if fit["rmse_pct"] <= 5 else "warn" if fit["rmse_pct"] <= 10 else "bad"
    items = [
        chip("حالت پروژه واقعی" if inp["real_mode"] else "حالت آموزشی", "info"),
        chip("داده نمونه پیش‌فرض (هنوز اجرا نشده)" if is_sample else "داده واردشده توسط کاربر", "warn" if is_sample else "input"),
        chip("تحلیل عددی پتانسیل: انجام شد" if stt else "تحلیل عددی پتانسیل: انجام نشد", "ok" if stt else "warn"),
        chip(f"RMSE برازش خاک: {fit['rmse_pct']:.1f}٪", q_kind),
        chip(vtitle, vstate),
    ]
    return H(f"""
    <div class="gs-head">
      <div>
        <div class="gs-title">Soil &amp; Grounding <i>Designer</i></div>
        <div class="gs-sub">تحلیل مقاومت ویژه خاک (Wenner)، مقاومت شبکه زمین، ولتاژهای تماس و گام — مرجع IEEE Std 80</div>
      </div>
      <div class="gs-status">{''.join(items)}</div>
    </div>
    <div class="gs-legend">
      <span style="--c:var(--gs-cyan)">ورودی / هندسه واردشده</span>
      <span style="--c:#9fb3d1">مقدار محاسبه‌شده</span>
      <span style="--c:var(--gs-dim)">حد مرجع</span>
      <span style="--c:var(--gs-amber)">هشدار</span>
      <span style="--c:var(--gs-red)">وضعیت بحرانی</span>
      <span style="--c:var(--gs-green)">در حد مجاز</span>
    </div>""")


def verdict_html(state: str, title: str, desc: str, ratio: float) -> str:
    fill_pct = min(ratio, 1.5) / 1.5 * 100
    if ratio <= 0.8:
        fill_col = C_GREEN
    elif ratio <= 1.0:
        fill_col = C_AMBER
    else:
        fill_col = C_AMBER if state == "warn" else C_RED
    return H(f"""
    <div class="gs-verdict {state}">
      <div class="t">{_ICON[state]} {title}</div>
      <div class="d">{desc}</div>
      <div class="track"><div class="fill" style="width:{fill_pct:.1f}%;background:{fill_col}"></div><div class="limit"></div></div>
      <div class="track-l"><span>0</span><span>GPR نسبت به ولتاژ تماس مجاز: {num(ratio * 100, '.0f')}٪ (خط عمودی = ۱۰۰٪)</span><span>150٪</span></div>
    </div>""")


def compliance_html(out: dict) -> str:
    """جدول مقایسه مقادیر محاسبه‌شده با حدود مرجع. فقط مقایسه‌هایی که از داده‌های موجود برمی‌آیند آورده می‌شوند."""
    tv, gpr, mesh, stt = out["tv"], out["gpr"], out["mesh"], out["num_stats"]
    mesh_ok = mesh["Em"] <= tv["e_touch"] and mesh["Es"] <= tv["e_step"]

    def row(label, val, unit, lim, lim_name, state, src, state_txt):
        r = val / lim * 100.0
        return (f'<tr><td>{label}</td><td class="num">{val:,.0f} {unit}</td>'
                f'<td class="num">{lim:,.0f} {unit} <small style="font-family:Vazirmatn">({lim_name})</small></td>'
                f'<td class="num">{r:,.0f}٪</td><td><span class="gs-pill {state}">{state_txt}</span></td><td>{src}</td></tr>')

    g_state = "ok" if gpr <= tv["e_touch"] else ("warn" if mesh_ok else "bad")
    g_txt = {"ok": "کمتر از حد", "warn": "بیش از حد · Em و Es مجاز", "bad": "بیش از حد"}[g_state]
    rows = [
        row("افزایش پتانسیل زمین (GPR)", gpr, "V", tv["e_touch"], "ولتاژ تماس مجاز", g_state, "Ig × Rg (Sverak)", g_txt),
        row("ولتاژ مش (Em)", mesh["Em"], "V", tv["e_touch"], "ولتاژ تماس مجاز",
            "ok" if mesh["Em"] <= tv["e_touch"] else "bad", "رابطه‌های IEEE 80",
            "در حد مجاز" if mesh["Em"] <= tv["e_touch"] else "بیش از حد مجاز"),
        row("ولتاژ گام (Es)", mesh["Es"], "V", tv["e_step"], "ولتاژ گام مجاز",
            "ok" if mesh["Es"] <= tv["e_step"] else "bad", "رابطه‌های IEEE 80",
            "در حد مجاز" if mesh["Es"] <= tv["e_step"] else "بیش از حد مجاز"),
    ]
    if stt:
        rows.append(row("بیشینه ولتاژ تماس روی نقشه", stt["touch_max"], "V", tv["e_touch"], "ولتاژ تماس مجاز",
                        "ok" if stt["touch_max"] <= tv["e_touch"] else "bad", "تحلیل عددی · خاک یکنواخت",
                        "در حد مجاز" if stt["touch_max"] <= tv["e_touch"] else "بیش از حد مجاز"))
        rows.append(row("بیشینه ولتاژ گام روی نقشه (گرادیان × ۱ m)", stt["step_max"], "V", tv["e_step"], "ولتاژ گام مجاز",
                        "ok" if stt["step_max"] <= tv["e_step"] else "bad", "تقریب عددی · خاک یکنواخت",
                        "در حد مجاز" if stt["step_max"] <= tv["e_step"] else "بیش از حد مجاز"))
    head = "<tr><th>کمیت</th><th>مقدار محاسبه‌شده</th><th>حد مرجع</th><th>نسبت به حد</th><th>وضعیت</th><th>منبع محاسبه</th></tr>"
    return f'<div class="gs-tw"><table class="gs-table"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table></div>'


# ============================================================
#  صفحه و پنل ورودی
# ============================================================
st.set_page_config(page_title="Soil & Grounding Designer", page_icon="⚡", layout="wide")
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown(H("""<div class="brand"><div class="logo">GND</div><div>
        <div class="brand-t">Grounding Designer</div><div class="brand-s">IEEE Std 80 · IEEE Std 81</div></div></div>"""),
                unsafe_allow_html=True)
    st.caption("پنل ورودی‌ها؛ پس از تغییر مقادیر، دکمه «اجرای محاسبات» را بزنید.")
    with st.form("inputs"):
        with st.expander("۱ · داده‌های اندازه‌گیری خاک (Wenner)", expanded=True):
            mode = st.radio("نوع داده اندازه‌گیری", ["مقاومت ظاهری ρa (Ω·m)", "مقاومت قرائت‌شده R (Ω)"],
                            help="اگر R قرائت‌شده را وارد می‌کنید، برنامه ρa = 2π·a·R را حساب می‌کند.")
            a_text = st.text_input("فاصله الکترودها a (m)", DEFAULTS["a_text"],
                                   help="ارقام فارسی یا انگلیسی؛ جداکننده: کاما، فاصله یا نقطه‌ویرگول. دست‌کم ۴ نقطه لازم است.")
            v_text = st.text_input("مقادیر اندازه‌گیری", DEFAULTS["v_text"],
                                   help="به تعداد فاصله‌ها؛ همه مقادیر مثبت باشند.")

        with st.expander("۲ · هندسه شبکه زمین", expanded=True):
            area = st.number_input("مساحت (m²)", min_value=1.0, value=DEFAULTS["area"])
            aspect = st.number_input("نسبت طول به عرض زمین (Lx/Ly)", min_value=0.2, max_value=5.0,
                                     value=DEFAULTS["aspect"], step=0.1)
            spacing = st.number_input("فاصله هادی‌ها D (m)", min_value=1.0, value=DEFAULTS["spacing"],
                                      help="تعداد خطوط هادی در هر جهت حداکثر ۲۵ است؛ در صورت نیاز فاصله واقعی بزرگ‌تر از D می‌شود.")
            depth = st.number_input("عمق دفن شبکه (m)", min_value=0.05, value=DEFAULTS["depth"], step=0.05,
                                    help="برای سازگاری محاسبات، حداقل عمق مدل 0.05 m است.")

        with st.expander("۳ · هادی و میله‌های زمین", expanded=True):
            d_mm = st.number_input("قطر هادی (mm)", min_value=4.0, value=DEFAULTS["d_mm"])
            rod_mode = st.selectbox("میله‌های زمین", list(ROD_MODES), format_func=ROD_MODES.get)
            rod_len = st.number_input("طول هر میله (m)", min_value=0.5, value=DEFAULTS["rod_len"])

        with st.expander("۴ · خطا و ایمنی", expanded=True):
            ig = st.number_input("جریان شبکه Ig (A)", min_value=1.0, value=DEFAULTS["ig"],
                                 help="جریان وارد‌شده به شبکه، پس از اعمال ضریب تقسیم جریان و ضریب decrement.")
            ts = st.number_input("زمان قطع خطا (s)", min_value=0.05, value=DEFAULTS["ts"],
                                 help="فرمول جریان مجاز بدن برای حدود ۰٫۰۳ تا ۳ ثانیه معتبر است.")
            weight = st.selectbox("وزن مبنا (kg)", [50, 70], index=1)
            use_surface = st.checkbox("لایه سطحی (شن/سنگ‌ریزه) دارد", value=DEFAULTS["use_surface"])
            rho_s = st.number_input("ρ لایه سطحی (Ω·m)", min_value=1.0, value=DEFAULTS["rho_s"],
                                    help="برای سنگ‌ریزه خیس وارد کنید، نه خشک.")
            hs = st.number_input("ضخامت لایه سطحی (m)", min_value=0.0, value=DEFAULTS["hs"])

        with st.expander("۵ · تنظیمات پیشرفته تحلیل", expanded=False):
            method = st.selectbox(
                "روش ρ معادل", ["apparent", "rho1", "mean"],
                format_func={"apparent": "ρa مدل در شعاع معادل (پیشنهادی)",
                             "rho1": "فقط ρ₁ (خاک یکنواخت)",
                             "mean": "میانگین ρ₁ و ρ₂ (غیرمهندسی)"}.get)
            real_mode = st.checkbox("حالت پروژه واقعی (کنترل کفایت داده)", value=DEFAULTS["real_mode"],
                                    help="فعال: هشدارهای عمق نفوذ و کفایت داده نمایش داده می‌شود. "
                                         "غیرفعال: حالت آموزشی، بدون این کنترل‌ها.")

        run_btn = st.form_submit_button("اجرای محاسبات", type="primary", use_container_width=True)

if run_btn:
    try:
        st.session_state["out"] = run_analysis(
            mode.startswith("مقاومت قرائت"), a_text, v_text, area, aspect, spacing, d_mm, rod_mode, rod_len,
            depth, method,
            ig, ts, weight, use_surface, rho_s, hs, real_mode)
        st.session_state["is_sample"] = False
        st.session_state["method"] = method
    except (ValueError, RuntimeError) as e:
        st.error(f"خطا: {e}")
        st.stop()
elif "out" not in st.session_state:
    st.session_state["out"] = run_analysis(**DEFAULTS)     # نمایش اولیه با داده نمونه
    st.session_state["is_sample"] = True

out = st.session_state["out"]
is_sample = st.session_state.get("is_sample", False)
a_vals, rho_vals, fit = out["a"], out["rho"], out["fit"]
tv, gpr, rg, inp = out["tv"], out["gpr"], out["rg"], out["inp"]
ratio = gpr / tv["e_touch"]
mesh_, geom_ = out["mesh"], out["geom"]
num_, stt_ = out.get("num"), out.get("num_stats")
vstate, vtitle, vdesc = verdict_state(gpr, tv, mesh_)
df_out = pd.DataFrame({"a (m)": a_vals, "Measured (Ω·m)": rho_vals,
                       "Fitted (Ω·m)": fit["pred"], "Error (%)": fit["rel_err"]})

# ============================================================
#  سربرگ و هشدارهای داده
# ============================================================
st.markdown(header_html(out, is_sample, vstate, vtitle), unsafe_allow_html=True)

if inp["real_mode"]:
    _issues = data_sufficiency(a_vals, rho_vals, fit, inp["area"])
    for _lvl, _txt in _issues[:2]:
        note(f'{_ICON[_lvl]} <b>هشدار داده‌ها:</b> {_txt}', _lvl)
    if len(_issues) > 2:
        note(f'{len(_issues) - 2} مورد دیگر در تب «توصیه» آمده است.', "info")

tab_ov, tab_soil, tab_3d, tab_map, tab_grid, tab_rep, tab_adv = st.tabs(
    ["نمای کلی", "تحلیل خاک", "شبکه سه‌بعدی", "پتانسیل و ولتاژ", "شبکه و ایمنی", "گزارش و خروجی", "توصیه"])

# ------------------------------------------------------------
#  نمای کلی
# ------------------------------------------------------------
with tab_ov:
    section("ورودی‌های اعمال‌شده", "پیکربندی تحلیل فعلی")
    in_chips = [chip(f"مساحت {inp['area']:,.0f} m²", "input"),
                chip(f"D = {inp['spacing']:g} m", "input"),
                chip(f"عمق دفن {inp['depth']:g} m", "input"),
                chip(f"Ig = {inp['ig']:,.0f} A", "input"),
                chip(f"زمان قطع {inp['ts']:g} s", "input"),
                chip(f"وزن مبنا {inp['weight']} kg", "input"),
                chip(f"میله‌ها: {ROD_MODES[inp['rod_mode']]}", "input"),
                chip(METHOD_LABEL[st.session_state.get("method", DEFAULTS["method"])], "input")]
    if inp["hs"] > 0:
        in_chips.append(chip(f"لایه سطحی {inp['hs']:g} m · ρs = {inp['rho_s']:,.0f} Ω·m", "input"))
    chips(in_chips)

    section("نتایج محاسبه‌شده", "مقادیر خروجی مدل")
    gpr_state = "ok" if ratio <= 1.0 else ("warn" if vstate == "warn" else "bad")
    kpi_grid([
        kpi("مقاومت ویژه لایه اول ρ₁", f"{fit['rho1']:,.1f}", "Ω·m", "برازش دولایه"),
        kpi("مقاومت ویژه لایه دوم ρ₂", f"{fit['rho2']:,.1f}", "Ω·m", f"ضخامت لایه اول h = {fit['h']:.2f} m"),
        kpi("مقاومت ویژه معادل ρ_eq", f"{out['rho_eq']:,.1f}", "Ω·m", "مبنای محاسبه Rg و ولتاژها"),
        kpi("مقاومت شبکه Rg", f"{out['rg_design']:.3f}", "Ω", "روش Sverak (مقدار طراحی)"),
        kpi("افزایش پتانسیل زمین GPR", f"{gpr:,.0f}", "V", f"Ig × Rg", state=gpr_state),
        kpi("ولتاژ مش Em", f"{mesh_['Em']:,.0f}", "V", "IEEE 80",
            state="ok" if mesh_["Em"] <= tv["e_touch"] else "bad"),
        kpi("ولتاژ گام Es", f"{mesh_['Es']:,.0f}", "V", "IEEE 80",
            state="ok" if mesh_["Es"] <= tv["e_step"] else "bad"),
    ])
    st.markdown(verdict_html(vstate, vtitle, vdesc, ratio), unsafe_allow_html=True)

    section("مقایسه با حدود مجاز", "مقدار محاسبه‌شده در برابر حد مرجع")
    st.markdown(compliance_html(out), unsafe_allow_html=True)
    if not stt_:
        note("تحلیل عددی پتانسیل انجام نشد؛ مقایسه‌ی نقشه‌ای (تماس و گام روی نقشه) در دسترس نیست.", "warn")
    else:
        note("سطرهای «تحلیل عددی» با خاک یکنواخت (ρ معادل) و ولتاژ گام ≈ گرادیان × ۱ متر محاسبه شده‌اند؛ "
             "نتیجه‌ی رابطه‌های IEEE 80 در سطرهای Em و Es است.", "info")
    section("مقایسه تصویری GPR، Em و Es", "خط سفید: حد مجاز")
    show_fig(safety_fig(gpr, tv, mesh_["Em"], mesh_["Es"]))

# ------------------------------------------------------------
#  تحلیل خاک
# ------------------------------------------------------------
with tab_soil:
    q_kind = "ok" if fit["rmse_pct"] <= 5 else "warn" if fit["rmse_pct"] <= 10 else "bad"
    chips([chip(f"RMSE برازش: {fit['rmse_pct']:.1f}٪", q_kind),
           chip(f"بیشینه خطا: {fit['max_err_pct']:.1f}٪", "neutral"),
           chip(f"ضریب بازتاب K = {fit['k']:.3f}", "neutral"),
           chip(f"RMSE مدل یکنواخت (مرجع): {fit['rmse_uniform_pct']:.1f}٪", "neutral")])
    if fit["rmse_pct"] > 10:
        note("خطای برازش بالاست؛ داده‌ها را بازبینی کنید یا خاک ممکن است بیش از دو لایه باشد.", "warn")
    if abs(fit["k"]) < 0.05 or fit["rmse_uniform_pct"] <= fit["rmse_pct"] * 1.2:
        note("خاک تقریباً یکنواخت است؛ ρ₂ و h در این حالت قابل اتکا نیستند.", "info")
    if fit["h_at_bound"] or fit["rho_at_bound"]:
        note("یکی از پارامترها به مرز بازه جستجو رسیده است؛ نتیجه برازش یکتا نیست.", "warn")

    c1, c2 = st.columns(2)
    with c1:
        section("منحنی برازش", "نقاط = اندازه‌گیری ، منحنی = مدل")
        show_fig(fit_fig(a_vals, rho_vals, fit))
    with c2:
        section("مقطع خاک", "عمق و ضخامت لایه‌ها با مقیاس واقعی")
        svg = soil_section_svg(fit, inp["depth"], inp["hs"], inp["rho_s"])
        b64 = base64.b64encode(svg.encode("utf-8")).decode()
        st.markdown(f'<img src="data:image/svg+xml;base64,{b64}" style="width:100%;border-radius:6px;'
                    f'border:1px solid #24324d"/>', unsafe_allow_html=True)
        st.caption("مقطع تصویری است: عمق دفن و ضخامت لایه‌ها با مقیاس واقعی‌اند، ولی فاصله افقی دایره‌های هادی نمادین است "
                   "(برای ابعاد افقی واقعی از تب «شبکه سه‌بعدی» استفاده کنید).")
        st.download_button("دانلود مقطع (SVG)", svg.encode("utf-8"), file_name="soil_section.svg",
                           mime="image/svg+xml", use_container_width=True)
    section("خطای برازش", "ناحیه سبز: ±۱۰٪ ؛ سبز ≤ ۵٪ ، کهربایی ≤ ۱۰٪ ، قرمز > ۱۰٪")
    show_fig(residual_fig(a_vals, fit["rel_err"]))

    section("پارامترهای مدل خاک", "با بازه اطمینان ۹۵٪ محاسبه‌شده از ژاکوبین برازش")
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
    section("مقایسه اندازه‌گیری و مدل", "خطا = (مدل − اندازه‌گیری) / اندازه‌گیری")
    st.dataframe(df_out, hide_index=True, use_container_width=True,
                 column_config={c: st.column_config.NumberColumn(format="%.2f") for c in df_out.columns})

# ------------------------------------------------------------
#  شبکه سه‌بعدی
# ------------------------------------------------------------
with tab_3d:
    section("هندسه مدل", "مقادیر واردشده و مشتق‌شده از مدل")
    mini_grid([
        mini_card("طول زمین Lx", f"{geom_['Lx']:,.1f}", "m"),
        mini_card("عرض زمین Ly", f"{geom_['Ly']:,.1f}", "m"),
        mini_card("تعداد هادی‌ها", f"{geom_['nx']} × {geom_['ny']}", "در X × در Y"),
        mini_card("فاصله هادی‌ها Dx / Dy", f"{geom_['Dx']:.2f} / {geom_['Dy']:.2f}", "m"),
        mini_card("عمق دفن هادی‌ها", f"{geom_['h']:g}", "m", kind="input"),
        mini_card("میله‌های زمین", f"{geom_['n_rods']}", f"× {geom_['Lr']:g} m" if geom_["n_rods"] else "", kind="input"),
        mini_card("طول هادی‌های افقی", f"{geom_['LC']:,.0f}", "m"),
        mini_card("طول کل هادی (با میله)", f"{geom_['LT']:,.0f}", "m"),
    ])
    section("نمای سه‌بعدی مهندسی", "سطح زمین Z = 0 ؛ عمق به‌صورت منفی")
    cc1, cc2 = st.columns([3, 2])
    view = cc1.radio("نمای دوربین", list(VIEWS), horizontal=True,
                     help="نماهای بالا، روبه‌رو و جانبی تصویر موازی (orthographic) دارند تا اندازه‌ها بدون اعوجاج پرسپکتیو خوانده شوند.")
    exag = cc2.select_slider("مقیاس عمودی Z", options=[1, 2, 3, 5, 10], value=1,
                             format_func=lambda v: "۱:۱ (واقعی)" if v == 1 else f"×{v} (اغراق‌شده)",
                             help="عمق‌ها نسبت به ابعاد زمین کوچک‌اند. افزایش این مقدار فقط محور Z را برای نمایش می‌کشد؛ اعداد همیشه واقعی‌اند.")
    tc1, tc2, tc3 = st.columns(3)
    show_dims = tc1.checkbox("اندازه‌گذاری و خطوط راهنما", value=True)
    show_soil = tc2.checkbox("لایه‌های خاک", value=True)
    show_crit = tc3.checkbox("نقاط بحرانی عددی", value=bool(stt_), disabled=not stt_,
                             help="محل بیشینه ولتاژ تماس و گام از تحلیل عددی؛ بدون تحلیل عددی در دسترس نیست.")
    show_fig(fig_3d_grid(fit, geom_, inp["hs"], stt_, float(exag), view, show_dims, show_soil, show_crit))
    if exag == 1:
        note("مقیاس عمودی واقعی (۱:۱) است. چون عمق‌ها نسبت به طول و عرض کوچک‌اند، برای دیدن عمق دفن و میله‌ها "
             "از نمای «روبه‌رو» یا «جانبی» و بزرگ‌نمایی استفاده کنید، یا مقیاس عمودی را اغراق دهید.", "info")
    else:
        note(f"محور Z برای خوانایی ×{exag} کشیده شده است؛ طول، عرض و برچسب‌های اندازه‌گذاری مقدار واقعی‌اند.", "warn")
    if geom_["capped"]:
        note("تعداد هادی‌ها به سقف ۲۵ در هر جهت محدود شده؛ فاصله واقعی (Dx و Dy) بزرگ‌تر از D واردشده است.", "warn")
    note("هادی‌ها، میله‌ها، فاصله‌ها و ابعاد دقیقاً از همین مدل ساخته می‌شوند؛ لایه‌های خاک با ضخامت‌های برازش‌شده رسم می‌شوند. "
         "نمودار برای درک فضایی و بررسی هندسه است، نه نقشه اجرایی.", "info")

    section("نمای برش (روبه‌رو، صفحه Y = 0)", "عمق دفن، طول میله و مرز لایه‌ها")
    show_fig(fig_elevation(fit, geom_, inp["hs"], float(exag)))

# ------------------------------------------------------------
#  پتانسیل و ولتاژ تماس/گام
# ------------------------------------------------------------
with tab_map:
    if not stt_:
        note("تحلیل عددی پتانسیل انجام نشد؛ هندسه شبکه را بررسی کنید.", "warn")
    else:
        tu, su = stt_["touch_unsafe_pct"], stt_["step_unsafe_pct"]
        chips([chip(f"ناحیه ناایمن تماس: {tu:.1f}٪", "ok" if tu == 0 else "bad"),
               chip(f"ناحیه ناایمن گام: {su:.1f}٪", "ok" if su == 0 else "bad"),
               chip(f"Rg عددی: {out['rho_eq'] * num_['rg_unit']:.2f} Ω", "neutral"),
               chip(f"{num_['n_seg']} قطعه · گام نقشه {num_['dx']:.2f} m", "neutral")])
        section("نقشه پتانسیل سطح و ولتاژهای تماس و گام", "تحلیل عددی · خاک یکنواخت با ρ معادل")
        vc1, vc2 = st.columns(2)
        view_q = vc1.radio("کمیت", ["ولتاژ تماس", "ولتاژ گام", "پتانسیل سطح", "گرادیان پتانسیل"], horizontal=True)
        dim = vc2.radio("بعد نمایش", ["۲بعدی (نقشه رنگی)", "۳بعدی (سطح)"], horizontal=True)
        Zv, uv = _field_cfg(view_q, stt_, tv)[:2]
        mini_grid([mini_card("کمینه داده نمایش‌داده‌شده", f"{float(np.min(Zv)):,.1f}", uv),
                   mini_card("بیشینه داده نمایش‌داده‌شده", f"{float(np.max(Zv)):,.1f}", uv, highlight=True)])
        if dim.startswith("۳"):
            show_fig(potential_surface_fig(view_q, num_, stt_, geom_, tv))
            note("صفحه قرمز نیمه‌شفاف حد مجاز است؛ هر جا سطح از آن بالاتر رفته ناایمن است. محور Z در این نما "
                 "کمیت تحلیلی است، نه ارتفاع هندسی. کانتورها روی کف و هادی‌ها با خط سفید نشان داده شده‌اند.", "info")
        else:
            show_fig(potential_map_fig(view_q, num_, stt_, geom_, tv))
            note("خط تیره، حد مجاز است؛ ناحیه داخل آن از حد عبور کرده است. ستاره، محل بیشینه محاسبه‌شده را نشان می‌دهد. "
                 "مقیاس رنگ برای تماس و گام و پتانسیل ثابت (۰ تا دو برابر حد، یا ۰ تا GPR) است؛ "
                 "مقیاس رنگ گرادیان خودکار است و فقط برای هر نقشه جداگانه معنا دارد.", "info")

        section("محدوده گرادیان پتانسیل", "بیشینه و کمینه روی نقشه")
        lo_, hi_ = stt_["grad_in"]
        mini_grid([
            mini_card("گرادیان داخل شبکه (کمینه)", f"{lo_:.2f}", "V/m"),
            mini_card("گرادیان داخل شبکه (بیشینه)", f"{hi_:.2f}", "V/m", highlight=True),
            mini_card("گرادیان بیرون شبکه (بیشینه)", f"{stt_['grad_out_max']:.2f}", "V/m"),
            mini_card("بیشینه ولتاژ تماس", f"{stt_['touch_max']:,.0f}", "V"),
            mini_card("بیشینه ولتاژ گام", f"{stt_['step_max']:,.0f}", "V"),
            mini_card("کمترین پتانسیل سطح داخل شبکه", f"{stt_['v_min_pct']:.0f}", "٪ GPR"),
        ])
        section("نیمرخ در امتداد X", "موقعیت Y را انتخاب کنید")
        y_pct = st.slider("موقعیت Y (٪ عرض زمین)", 0, 100, 50)
        f_pot, f_ts, y_used = profile_figs(num_, stt_, geom_, tv, y_pct / 100.0)
        st.caption(f"نیمرخ در Y = {y_used:.1f} m ؛ ناحیه خاکستری محدوده شبکه است.")
        show_fig(f_pot)
        show_fig(f_ts)
        note("فرض‌ها: خاک یکنواخت با ρ معادل، شبکه مستطیلی، ولتاژ تماس نقطه‌ای (GPR منهای پتانسیل سطح) "
             "و ولتاژ گام برابر گرادیان × ۱ متر (تقریب عددی، نه نتیجه‌ی رابطه‌های IEEE 80). "
             "برای طرح نهایی نتیجه را با نرم‌افزار تخصصی (مثل CYMGRD یا CDEGS) راستی‌آزمایی کنید.", "warn")

# ------------------------------------------------------------
#  شبکه و ایمنی
# ------------------------------------------------------------
with tab_grid:
    section("مقاومت شبکه زمین", f"ρ معادل = {out['rho_eq']:,.1f} Ω·m")
    mini_grid([
        mini_card("ρ / 4r", f"{rg['simple']:.3f}", "Ω"),
        mini_card("ρ / 4r + ρ / L", f"{rg['with_length']:.3f}", "Ω"),
        mini_card("Sverak", f"{rg['sverak']:.3f}", "Ω", highlight=True, tag="مقدار طراحی"),
    ])
    section("حدود مجاز ایمنی", f"وزن مبنا {inp['weight']} kg · زمان خطا {inp['ts']:g} s")
    mini_grid([
        mini_card("جریان مجاز بدن", f"{tv['i_body']:.3f}", "A", kind="limit"),
        mini_card("ضریب لایه سطحی Cs", f"{tv['cs']:.3f}", "", kind="limit"),
        mini_card("ولتاژ گام مجاز", f"{tv['e_step']:,.0f}", "V", kind="limit"),
        mini_card("ولتاژ تماس مجاز", f"{tv['e_touch']:,.0f}", "V", kind="limit"),
    ])
    for _w in mesh_["warnings"]:
        note(_w, "warn")
    with st.expander("جزئیات فنی محاسبات (IEEE 80 و تحلیل عددی)"):
        mini_grid([
            mini_card("n (ضریب هندسی)", f"{mesh_['n']:.2f}", ""),
            mini_card("Km", f"{mesh_['Km']:.4f}", ""),
            mini_card("Ki", f"{mesh_['Ki']:.4f}", ""),
            mini_card("Ks", f"{mesh_['Ks']:.4f}", ""),
            mini_card("Kh · Kii", f"{mesh_['Kh']:.3f} · {mesh_['Kii']:.3f}", ""),
            mini_card("LM (طول مؤثر mesh)", f"{mesh_['LM']:,.1f}", "m"),
            mini_card("LS (طول مؤثر step)", f"{mesh_['LS']:,.1f}", "m"),
            mini_card("درجه آزادی برازش", f"{fit['dof']}", ""),
        ])
        if num_:
            st.caption(f"تحلیل عددی: {num_['n_seg']} قطعه ، گام نقشه {num_['dx']:.2f} × {num_['dy']:.2f} m ، "
                       f"Rg = ρ_eq × {num_['rg_unit']:.5f} 1/m")

# ------------------------------------------------------------
#  گزارش و خروجی
# ------------------------------------------------------------
with tab_rep:
    report = build_report(inp, fit, out["rho_eq"], rg, out["rg_design"], gpr, tv, out)
    with st.container(border=True):
        st.markdown(report_html(inp, fit, out["rho_eq"], rg, out["rg_design"], gpr, tv, out),
                    unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    d1.download_button("دانلود گزارش (Markdown)", report.encode("utf-8"),
                       file_name="grounding_report.md", use_container_width=True)
    d2.download_button("دانلود داده‌ها (CSV)", df_out.to_csv(index=False).encode("utf-8-sig"),
                       file_name="soil_analysis.csv", use_container_width=True)
    with st.expander("فرضیات و محدودیت‌ها"):
        st.markdown(
            "- مدل خاک دولایه است و برای خاک چندلایه یا ناهمگن دقت کمتری دارد.\n"
            "- ρ معادل تقریبی است؛ برای طرح نهایی از روش Sunde یا نرم‌افزار تخصصی استفاده کنید.\n"
            "- Em و Es با رابطه‌های IEEE 80 (شبکه مستطیلی، خاک یکنواخت) و نقشه پتانسیل با روش عددی ساده محاسبه شده‌اند؛ "
            "برای طرح نهایی با نرم‌افزار تخصصی راستی‌آزمایی کنید.\n"
            "- خروجی برای طراحی اولیه است و طرح نهایی باید مهندس مسئول بازبینی کند.")

# ------------------------------------------------------------
#  توصیه مهندسی
# ------------------------------------------------------------
with tab_adv:
    adv = engineering_advice(out)
    section("خلاصه و توصیه مهندسی", "بر پایه IEEE 80 و IEEE 81 · قاعده‌محور")
    st.markdown(advice_html(adv), unsafe_allow_html=True)
    st.download_button("دانلود توصیه‌ها (Markdown)", advice_markdown(adv).encode("utf-8"),
                       file_name="engineering_advice.md", use_container_width=True)

st.markdown('<div class="gs-foot">Soil &amp; Grounding Designer · بر پایه IEEE Std 80</div>', unsafe_allow_html=True)
