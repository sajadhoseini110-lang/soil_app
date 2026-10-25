# ⚡ Soil & Grounding Designer (IEEE Std 80)

**A Comprehensive Web Application for Soil Resistivity Analysis and Grounding System Design.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://soilapp-k3nsalb3wsraerxy9fwfx6.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg being)

 used---

## 📖 Overview (معرفی)

This application is a professional engineering tool designed to interpret Wenner four-electrode soil resistivity measurements and perform initial grounding grid design based on **IEEE Std 80**. 

این نرم‌افزار یک ابزار مهندسی حرفه‌ای برای تفسیر داده‌های اندازه‌گیری مقاومت ویژه خاک با روش ونر و انجام محاسبات اولیه طراحی سیستم زمین (ارتینگ) بر اساس استاندارد **IEEE Std 80** است.

### ✨ Key Features (ویژگی‌های کلیدی)
- **Two-Layer Soil Modeling:** Advanced inverse modeling using `scipy.optimize.least_squares` to extract Layer 1 (\(\rho_1\)), Layer 2 (\(\rho_2\)), and thickness (\(h\)).
- **Safety Analysis:** Calculation of Ground Potential Rise (GPR) and comparison with tolerable touch and step voltages.
- **Surface Material Support:** Accounting for the surface layer (e.g., crushed rock) to calculate the derating factor (\(C_s\)).
- **Interactive Visualizations:** High-quality Plotly charts for soil profile, curve fitting, and safety limits.
- **AI Image Generation (Optional):** Artistic visualization of the soil cross-section using Pollinations AI.
- **Bilingual & RTL Support:** Fully optimized UI for Persian (Farsi) with English technical terms.

---

## 🚀 Live Demo (نسخه آنلاین)

You can access the live web application directly from your browser or mobile phone:
👉 **[Launch Soil & Grounding Designer](https://soilapp-k3nsalb3wsraerxy9fwfx6.streamlit.app/)**

---

## 🛠️ Technical Stack (تکنولوژی‌های استفاده شده)

- **Language:** Python 3.9+
- **Web Framework:** Streamlit
- **Mathematical Modeling:** NumPy, SciPy
- **Data Handling:** Pandas
- **Visualizations:** Plotly
- **AI Integration:** Pollinations API, Pillow (PIL)

---

## 📊 Methodology (روش کار)

1.  **Data Input:** The user inputs Wenner array spacings (\(a\)) and measured apparent resistivities (\(\rho_a\)) or resistances (\(R\)).
2.  **Inverse Modeling:** The app fits a two-layer earth model using non-linear least squares optimization.
3.  **Grid Resistance:** Calculates grid resistance using Sverak's method and simple analytical formulas.
4.  **Safety Check:** Evaluates the Ground Potential Rise (GPR) against tolerable touch voltages based on body weight and fault clearing time.

---

## 👨‍💻 Author (نویسنده)

**Sajad Hosseini**
- GitHub: [@sajadhoseini110-lang](https://github.com/sajadhoseini110-lang)

---

## ⚠️ Disclaimer (سلب مسئولیت)
This application is intended for preliminary engineering analysis and educational purposes. The results should always be verified by a licensed professional engineer before for actual grounding system construction.

*این برنامه برای تحلیل‌های اولیه مهندسی طراحی شده است. نتایج آن باید توسط مهندس مسئول تأیید شود.*