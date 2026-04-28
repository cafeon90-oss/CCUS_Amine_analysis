"""
CCUS 공정 라이선스 벤치마크 툴 (한국 기준)
==========================================
계산 지표: We (Equivalent Work), SPECCA, COCA
대상 라이선스: KoSol (KEPCO), MEA Generic, MHI KS-1, Custom
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1. 페이지 설정
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CCUS 라이선스 벤치마크 | 한국 기준",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 3.5rem; }
    header[data-testid="stHeader"] { background: rgba(255,255,255,0.95); }
    .stTabs [data-baseweb="tab-list"] { margin-top: 0.5rem; }
    .stMetric { background: #f8f8f8; border-radius: 8px; padding: 8px 12px; }
    [data-testid="stMetricValue"] { font-size: 1.05rem !important; }
    [data-testid="stMetricLabel"] { font-size: 0.78rem !important; }
    [data-testid="stMetricDelta"] { font-size: 0.75rem !important; }
    .warning-box { background:#fff3cd; border-left:4px solid #ffc107;
                   padding:10px 14px; border-radius:4px; margin:8px 0; font-size:13px; }
    .error-box   { background:#f8d7da; border-left:4px solid #dc3545;
                   padding:10px 14px; border-radius:4px; margin:8px 0; font-size:13px; }
    .info-box    { background:#d1ecf1; border-left:4px solid #17a2b8;
                   padding:10px 14px; border-radius:4px; margin:8px 0; font-size:13px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2. 상수 및 기준 데이터
# ─────────────────────────────────────────────────────────────────────────────

# 문헌 데이터 (SRD, L/G, We, 흡수제손실) — 26개 포인트
# CANSOLV DC-103 SRD: NETL Rev4a 실측 (B11B=3.38, B12B=3.56 GJ/tCO₂)
# 2.44 GJ/tCO₂ (1,050 Btu/lb) 수치는 Cansolv 홍보자료 추정치로 제외
LIT = {
    "solvent":      ["MEA 30%","MEA 30%","MEA 30%","MEA 30%","MEA 30%",
                     "KS-1","KS-1","KS-1",
                     "AMP/PZ","AMP/PZ","AMP/PZ",
                     "PZ","PZ","PZ",
                     "MDEA/PZ","MDEA/PZ",
                     "KoSol-4","KoSol-5","KoSol-5",
                     "Econamine+","Econamine+",
                     "Chilled NH3",
                     "CANSOLV DC-103","CANSOLV DC-103","CANSOLV DC-103","CANSOLV DC-103"],
    "SRD":          [3.7,3.5,3.9,3.6,3.8,
                     2.5,2.4,2.6,
                     2.6,2.7,2.5,
                     2.1,2.2,2.3,
                     2.8,2.9,
                     3.0,3.1,3.2,
                     3.2,3.3,
                     2.0,
                     3.38,3.56,3.45,2.7],
    "LG":           [3.2,3.6,3.0,3.4,3.1,
                     5.1,5.4,4.9,
                     4.8,4.5,5.0,
                     6.8,6.5,6.2,
                     4.6,4.3,
                     4.2,4.0,4.7,
                     3.8,3.6,
                     2.8,
                     4.3,4.4,4.4,4.2],
    "We":           [1.72,1.65,1.80,1.68,1.75,
                     1.28,1.22,1.32,
                     1.32,1.35,1.29,
                     1.18,1.21,1.25,
                     1.38,1.42,
                     1.42,1.45,1.34,
                     1.52,1.55,
                     1.45,
                     1.62,1.68,1.65,1.36],
    "sol_loss":     [2.1,1.8,2.4,2.0,2.2,
                     0.40,0.35,0.45,
                     0.70,0.75,0.65,
                     1.10,1.00,0.90,
                     0.60,0.65,
                     0.60,0.65,0.50,
                     1.5,1.6,
                     0.2,
                     0.15,0.15,0.15,0.55],
    "source":       ["IEAGHG 2007","NTNU 2011","Aspen bench","IEAGHG 2011","Rochelle 2012",
                     "MHI 2012","IEAGHG 2011","MHI 2014",
                     "Rochelle 2011","IEAGHG 2013","SINTEF 2013",
                     "Rochelle 2009","SINTEF 2013","Rochelle 2013",
                     "IEAGHG 2014","NTNU 2015",
                     "KEPCO 2018","KEPCO 2019","KEPCO 2021",
                     "Fluor 2011","IEAGHG 2012",
                     "Alstom 2012",
                     "NETL 2022(B11B)","NETL 2022(B12B)","NETL 2022(avg)","Shell 2013"],
}

# ─── NETL Rev4a (DOE/NETL-2023/4320) 기준 검증값 ───────────────────────────
# 출처: Exhibits 4-80, 4-81, 5-23, 5-24 (원문 표 직접 발췌)
#
# B12B.90 (SC PC 650MWe + Cansolv DC-103, 90% capture):
#   SRD=3.56 GJ/tCO₂ (reboiler 2,058 GJ/hr ÷ 578.7 t/hr)
#   스팀 16.4 bar / 203°C (LP 터빈 중압 추기)
#   보조동력: 팬 14.5 MWe, 펌프 11.2 MWe, 압축 44.8 MWe (8단, 153 bar)
#   → 단위환산: 팬 0.090, 펌프 0.070, 압축 0.279 GJe/tCO₂
#   SPECCA 3,550 MJ/tCO₂ | COC $38.1/t @ 4.31 Mt/yr
#
# B31B.90 (NGCC F-Frame 645MWe + Cansolv DC-103, 90% capture):
#   SRD=3.47 GJ/tCO₂ (reboiler 680 GJ/hr ÷ 196.2 t/hr)  ← 2.50은 추정치였음
#   스팀 4.5 bar (PC와 달리 HRSG 저압 추기, 포화온도 ~148°C)
#   보조동력: 팬 12.1 MWe, 펌프 4.8 MWe, 압축 17.2 MWe (8단, 153 bar)
#   → 단위환산: 팬 0.222, 펌프 0.088, 압축 0.316 GJe/tCO₂
#   SPECCA 2,800 MJ/tCO₂ | COC $60.7/t @ 1.46 Mt/yr
#
# COCA 스케일 앵커: 소규모(10만t/yr) ≈ NETL값 × 5~6배 (6/10 법칙, 40배 스케일 차이)
NETL_BENCHMARK = {
    "B12B": {
        "SRD": 3.56, "SPECCA": 3550, "COC_usd": 38.1,
        "ann_CO2_Mt": 4.31, "steam_P_bar": 16.4, "steam_T_C": 203,
        "TPC_per_kW": 3452, "eff_no_cap": 40.2, "eff_cap": 31.7,
        "We_fan": 0.090, "We_pump": 0.070, "We_compress": 0.279,   # GJe/tCO₂ NETL 실측
        "compress_bar": 153, "compress_stages": 8,
        "CAPEX_frac": 0.53, "OPEX_frac": 0.47,   # COC 내 CAPEX:OPEX 비율
    },
    "B31B": {
        "SRD": 3.47, "SPECCA": 2800, "COC_usd": 60.7,
        "ann_CO2_Mt": 1.46, "steam_P_bar": 4.5,  "steam_T_C": 148,
        "TPC_per_kW": 1686, "eff_no_cap": 53.6,  "eff_cap": 47.6,
        "We_fan": 0.222, "We_pump": 0.088, "We_compress": 0.316,
        "compress_bar": 153, "compress_stages": 8,
        "CAPEX_frac": 0.55, "OPEX_frac": 0.45,
    },
    "B11B": {
        "SRD": 3.38, "SPECCA": 3610, "COC_usd": 37.0,
        "ann_CO2_Mt": 4.09, "steam_P_bar": 16.4, "steam_T_C": 203,
        "TPC_per_kW": 3350, "eff_no_cap": 34.3,  "eff_cap": 26.8,
        "We_fan": 0.088, "We_pump": 0.068, "We_compress": 0.275,
        "compress_bar": 153, "compress_stages": 8,
        "CAPEX_frac": 0.54, "OPEX_frac": 0.46,
    },
}

# 라이선스 기본 파라미터
LICENSE = {
    "KoSol (KEPCO)": {
        "SRD": 3.1, "steam_P": 5.0, "capture": 0.90,
        "LG": 4.2, "sol_loss": 0.60, "sol_price": 3500,
        "T_abs": 45, "conc": 40, "color": "#378ADD",
        "desc": "국산 아민 공정 · 한전(KEPCO) 개발 · LPS 기반 · SRD 3.0~3.2 GJ/t (KoSol-4/5 실측)",
    },
    "MEA 30% (기준선)": {
        "SRD": 3.5, "steam_P": 15.0, "capture": 0.90,
        "LG": 3.5, "sol_loss": 2.0, "sol_price": 2200,
        "T_abs": 40, "conc": 30, "color": "#E24B4A",
        "desc": "MEA 30wt% 글로벌 기준선 · NETL/IEAGHG 벤치마크 기준 · SPECCA·We 비교 기준점",
    },
    "MHI KS-1": {
        "SRD": 2.5, "steam_P": 15.0, "capture": 0.90,
        "LG": 5.1, "sol_loss": 0.40, "sol_price": 8000,
        "T_abs": 40, "conc": 35, "color": "#1D9E75",
        "desc": "MHI 힌더드 아민 · 국내 발전사 도입 사례 · Petra Nova(2017) KS-1 계열",
    },
    "Boundary Dam 3": {
        "SRD": 3.4, "steam_P": 16.4, "capture": 0.90,
        "LG": 4.4, "sol_loss": 0.15, "sol_price": 4050,  # 원/kg (≈$3/kg = $3,000/tonne)
        "T_abs": 40, "conc": 35, "color": "#FF8C00",
        "desc": "세계 최초 상업 규모 발전소 후연도 CCS · SaskPower (캐나다, 2014) · "
                "Shell Cansolv DC-103 · 160 MWe SC PC · 설계 1 Mt CO₂/yr · LP 스팀 16.4 bar",
    },
}

# 배가스 발생원별 표준값
FG_DEFAULTS = {
    "석탄 발전소":       {"CO2": 12.0, "O2": 5.0, "NOx": 200, "SOx": 50,  "H2O": 10.0, "T": 50, "flow": 500000},
    "가스 발전소 (NGCC)": {"CO2":  4.0, "O2":12.0, "NOx":  50, "SOx":  5,  "H2O":  8.0, "T": 50, "flow": 800000},
    "시멘트 공장":        {"CO2": 20.0, "O2": 8.0, "NOx": 500, "SOx":100,  "H2O": 12.0, "T": 60, "flow": 300000},
    "제철소 (고로가스)":  {"CO2": 22.0, "O2": 2.0, "NOx": 100, "SOx": 30,  "H2O":  5.0, "T": 55, "flow": 400000},
    "화학공장":           {"CO2": 15.0, "O2": 6.0, "NOx": 150, "SOx": 80,  "H2O":  8.0, "T": 50, "flow": 200000},
    "직접 입력":          {"CO2": 12.0, "O2": 5.0, "NOx": 200, "SOx": 50,  "H2O": 10.0, "T": 50, "flow": 500000},
}

# 한국 유틸리티/재무 표준값
KR = {
    "elec":         120,       # 원/kWh
    "lps":          25000,     # 원/GJ
    "mps":          32000,
    "hps":          38000,
    "cooling":      2500,      # 원/GJ
    "water":        1200,      # 원/ton
    "fx":           1350,      # 원/USD
    "discount":     8.0,       # %
    "lifetime":     25,        # 년
    "capacity":     85,        # %
    "maint":        2.5,       # % CAPEX/년
    "insurance":    0.5,
    "operators":    8,
    "labor":        70_000_000,  # 원/명/년
    "ci":           1.15,      # 건설비 지수 (한국/미국)
}

# ─────────────────────────────────────────────────────────────────────────────
# 3. 계산 엔진
# ─────────────────────────────────────────────────────────────────────────────

def T_sat(P_bar: float) -> float:
    """스팀 포화온도 [°C] — Antoine 근사"""
    return 100.0 + 28.06 * np.log(max(P_bar, 0.1) / 1.013)


def carnot(T_steam_C: float, T_cold_C: float = 20.0) -> float:
    return 1.0 - (T_cold_C + 273.15) / (T_steam_C + 273.15)


def LG_from_SRD(SRD: float, slope: float, intercept: float) -> float:
    """회귀 모델 기반 L/G 추정"""
    return max(slope * SRD + intercept, 2.0)


def calc_We(SRD, steam_P, LG, T_cold=20, P_final=153.0):
    """
    We 계산 [GJe/tCO₂]
    ─ We_thermal : SRD × Carnot η  (재생에너지 기여)
    ─ We_pump    : L/G × 0.028     (순환펌프, L/G 비례)
    ─ We_blower  : 0.018 + 0.007×L/G  (블로워, L/G 연동 — 고L/G시 탑 압력손실 증가)
    ─ We_compress: CO₂ 압축 (스트리퍼 출구 ~1.5 bar → 파이프라인 153 bar, 8단)
    ─ We_liquefy : CO₂ 액화
    Trade-off: SRD↓ → We_thermal↓ but L/G↑ → We_pump·We_blower↑
    → We_total 최솟값이 SRD≈2.0~2.3 근방에서 나타남 (스팀 등급에 따라 이동)
    검증: NETL B12B 실측 압축전력 0.279 GJe/tCO₂ (44.8 MWe ÷ 578.7 t/hr)
          NETL B31B 실측 압축전력 0.316 GJe/tCO₂ (17.2 MWe ÷ 54.4 t/hr)
    """
    T_st = T_sat(steam_P)
    eta  = carnot(T_st, T_cold)

    We_th  = SRD * eta
    We_pu  = LG * 0.028                            # 펌프: L/G 비례
    We_bl  = 0.018 + 0.007 * LG                   # 블로워: L/G 연동 (LG=4 → 0.046, LG=7 → 0.067)
    # CO₂ 압축: 스트리퍼 출구 1.5 bar → P_final bar, 8단 폴리트로픽 η=0.78
    # 계수 0.047 = NETL B12B 실측 보정 (44.8 MWe ÷ 578.7 t/hr = 0.279 GJe/tCO₂ @ 153 bar)
    # 검증: 0.047 × ln(153/1.5) / 0.78 = 0.047 × 4.615 / 0.78 = 0.278 ✓
    P_in   = 1.5                                   # CO₂ 스트리퍼 출구 압력 [bar]
    We_co  = max(0.047 * np.log(P_final / P_in) / 0.78, 0.05)
    We_liq = 0.12                                  # 액화 -20°C

    We_el  = We_pu + We_bl + We_co + We_liq
    return {
        "We_total":   We_th + We_el,
        "We_thermal": We_th,
        "We_pump":    We_pu,
        "We_blower":  We_bl,
        "We_compress":We_co,
        "We_liquefy": We_liq,
        "We_elec":    We_el,
        "T_steam":    T_st,
        "eta":        eta,
    }


def calc_sol_loss(amine, O2, T_abs, T_reb, NOx, SOx, wash=2):
    """
    흡수제 손실 예측 [kg/tCO₂]
    Arrhenius 기반 (Nguyen 2010, Voice & Rochelle 2011)
    """
    Ea_ox = {"MEA 30% (기준선)": 84, "KoSol (KEPCO)": 72, "MHI KS-1": 65, "Boundary Dam 3": 62, "Custom": 78}
    Ea_th = {"MEA 30% (기준선)":120, "KoSol (KEPCO)":110, "MHI KS-1":105, "Boundary Dam 3":103, "Custom":115}
    k_ox  = {"MEA 30% (기준선)":2.0, "KoSol (KEPCO)":0.8, "MHI KS-1":0.4, "Boundary Dam 3":0.35, "Custom":1.2}
    k_th  = {"MEA 30% (기준선)":0.6, "KoSol (KEPCO)":0.3, "MHI KS-1":0.2, "Boundary Dam 3":0.18, "Custom":0.4}
    evap0 = {"MEA 30% (기준선)":0.8, "KoSol (KEPCO)":0.4, "MHI KS-1":0.2, "Boundary Dam 3":0.15, "Custom":0.5}

    R = 8.314
    T_a  = T_abs + 273.15;  T_r = T_reb + 273.15
    T_ref_a = 313.15;       T_ref_r = 393.15   # 40°C, 120°C 기준

    ea_o = Ea_ox.get(amine, 78)  * 1000
    ea_t = Ea_th.get(amine, 115) * 1000

    ox   = k_ox.get(amine, 1.2) * (O2 / 5.0) * np.exp(-ea_o/R * (1/T_a - 1/T_ref_a))
    th   = k_th.get(amine, 0.4) * np.exp(-ea_t/R * (1/T_r - 1/T_ref_r))
    ev   = evap0.get(amine, 0.5) * (0.5**wash) * (T_abs / 40.0)**1.5
    hss  = NOx * 8e-5 + SOx * 2e-4          # kg HSS / tCO₂

    total = ox + th + ev
    return {
        "oxidative":  round(ox,  4),
        "thermal":    round(th,  4),
        "evaporative":round(ev,  4),
        "hss":        round(hss, 4),
        "total":      round(total, 4),
        "grand":      round(total + hss * 0.3, 4),   # HSS → 유효 아민 손실 환산
    }


def calc_CAPEX(scale_tpa, LG, SRD, ci=1.15):
    """
    CAPEX 추산 [백만 USD] — Guthrie method + 6/10 법칙
    흡수탑: L/G^0.75 (기존 0.5 → 강화 — 고L/G시 탑 높이·직경 동시 증가)
    패킹 높이 항: L/G가 높으면 필요 이론단수 증가 → 추가 페널티
    """
    sf = (scale_tpa / 100_000) ** 0.6
    absorber  = 8.5 * (LG / 4.0) ** 0.75           # 흡수탑 shell+내부재 (L/G 페널티 강화)
    packing   = 3.0 * (LG / 4.0) ** 1.0             # 패킹 높이: L/G에 선형 비례
    stripper  = 4.5 * (LG / 4.0) ** 0.3
    hx        = 3.5 * (SRD / 3.5) ** 0.6            # 재생 열교환기
    eq = (absorber + packing + stripper + hx + 6.0 + 4.0) * sf
    TPC = eq * 4.5 * ci * 1.15   # 설치계수 × 한국보정 × 오너비용
    return round(TPC, 2)


def calc_COCA(p: dict):
    """COCA 계산 — 전체 비용 통합 [USD/tCO₂, 만원/tCO₂]"""
    ann_CO2 = p["scale"] * p["cap"] / 100
    dr  = p["dr"] / 100
    CRF = dr * (1+dr)**p["life"] / ((1+dr)**p["life"] - 1)

    TPC_mUSD    = calc_CAPEX(p["scale"], p["LG"], p["SRD"], p["ci"])
    TPC_bil_krw = TPC_mUSD * p["fx"] / 100     # 억원 (million USD × 원/USD / 1e8 × 1e6 = / 100)

    # 스팀 단가 선택
    sp = p["lps"] if p["steam_P"] <= 7 else (p["mps"] if p["steam_P"] <= 20 else p["hps"])

    ann_cap   = TPC_bil_krw * CRF
    ann_steam = p["SRD"] * ann_CO2 * sp / 1e8
    ann_elec  = p["We_elec"] * 1e6 / 3600 * ann_CO2 * p["elec"] / 1e8  # GJe→kWh: 1GJ=277.78kWh
    ann_cool  = 0.5 * ann_CO2 * p["cooling"] / 1e8
    ann_sol   = p["sol_loss"] * ann_CO2 * p["sol_price"] / 1e8
    ann_maint = TPC_bil_krw * p["maint"] / 100
    ann_ins   = TPC_bil_krw * KR["insurance"] / 100
    ann_labor = p["operators"] * p["labor"] / 1e8

    total_ann = ann_cap + ann_steam + ann_elec + ann_cool + ann_sol + ann_maint + ann_ins + ann_labor

    COCA_krw = total_ann * 1e8 / ann_CO2 if ann_CO2 > 0 else 0
    return {
        "COCA_usd":   round(COCA_krw / p["fx"], 1),
        "COCA_man":   round(COCA_krw / 10000, 1),
        "TPC_mUSD":   TPC_mUSD,
        "TPC_bil":    round(TPC_bil_krw, 1),
        "ann_cap":    round(ann_cap, 2),
        "ann_steam":  round(ann_steam, 2),
        "ann_elec":   round(ann_elec, 2),
        "ann_cool":   round(ann_cool, 2),
        "ann_sol":    round(ann_sol, 2),
        "ann_maint":  round(ann_maint, 2),
        "ann_ins":    round(ann_ins, 2),
        "ann_labor":  round(ann_labor, 2),
        "total_ann":  round(total_ann, 2),
        "ann_CO2":    round(ann_CO2, 0),
        "CRF":        round(CRF, 4),
    }


def calc_SPECCA(SRD, We_elec, capture=0.90):
    """
    SPECCA [MJ/tCO₂ avoided] — NETL B12B 기준 보정식
    = (SRD × 500 + We_elec × 2500) / capture
    ─ SRD: 스팀 에너지 열손실 기여분 (전체 SRD가 아닌 효율 페널티 부분)
    ─ We_elec × 2.5×1000: 전기 보조동력 → 1차에너지 환산 (η_grid ≈ 0.4)
    검증: NETL B12B SRD=3.56, We_elec≈0.56 → (3.56×500+0.56×2500)/0.90 = 3,530 ≈ 실측 3,550 ✓
    """
    return round((SRD * 500 + We_elec * 2500) / capture, 0)


# ─────────────────────────────────────────────────────────────────────────────
# 4. 회귀 모델 피팅 (앱 시작 시 1회)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def fit_models():
    df = pd.DataFrame(LIT)
    sl_LG, ic_LG, r_LG, _, _ = stats.linregress(df["SRD"], df["LG"])
    sl_We, ic_We, r_We, _, _ = stats.linregress(df["SRD"], df["We"])
    sl_sl, ic_sl, r_sl, _, _ = stats.linregress(df["SRD"], df["sol_loss"])
    try:
        def expf(x, a, b): return a * np.exp(b * x)
        po, _ = curve_fit(expf, df["SRD"], df["sol_loss"], p0=[0.3, 0.5], maxfev=3000)
        exp_ok = True
    except Exception:
        po = [None, None]; exp_ok = False
    return {
        "LG":  (sl_LG, ic_LG, r_LG**2),
        "We":  (sl_We, ic_We, r_We**2),
        "sol": (sl_sl, ic_sl, r_sl**2),
        "sol_exp": (po, exp_ok),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. 사이드바
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏭 CCUS 벤치마크")
    st.caption("한국 기준 | We · SPECCA · COCA")
    st.divider()

    st.markdown("**📋 비교 라이선스**")
    sel = [lic for lic in LICENSE if st.checkbox(lic, value=True)]

    st.divider()
    st.markdown("**🏭 배가스 발생원**")
    fg_src = st.selectbox("발생원", list(FG_DEFAULTS.keys()), label_visibility="collapsed")
    fgd = FG_DEFAULTS[fg_src]

    st.divider()
    st.markdown("**📏 플랜트 규모**")
    scale = st.number_input("CO₂ 포집량 (tCO₂/년)", 10_000, 2_000_000, 100_000, 10_000)
    st.caption(f"= **{scale:,} tCO₂/년**")

    st.divider()
    st.markdown("**🌡 사이트 조건**")
    T_cold = st.slider("냉각수 온도 (°C)", 5, 35, 20)
    P_final = st.selectbox("CO₂ 최종 압력",
                            [("액화탄산 20 bar", 20.0), ("파이프라인 150 bar", 150.0),
                             ("광물탄산화 직접 (~3 bar)", 3.0)],
                            format_func=lambda x: x[0])[1]

    st.divider()
    with st.expander("⚡ 유틸리티 단가 (표준값 내장)"):
        elec    = st.number_input("전기 (원/kWh)",    value=KR["elec"])
        lps     = st.number_input("LPS 스팀 (원/GJ)", value=KR["lps"])
        st.caption(f"{lps:,} 원/GJ")
        mps     = st.number_input("MPS 스팀 (원/GJ)", value=KR["mps"])
        st.caption(f"{mps:,} 원/GJ")
        hps     = st.number_input("HPS 스팀 (원/GJ)", value=KR["hps"])
        st.caption(f"{hps:,} 원/GJ")
        cooling = st.number_input("냉각수 (원/GJ)",   value=KR["cooling"])

    with st.expander("💰 재무 가정 (표준값 내장)"):
        fx       = st.number_input("환율 (원/USD)",       value=KR["fx"])
        st.caption(f"{fx:,} 원/USD")
        dr       = st.slider("할인율 (%)",     4.0, 15.0, float(KR["discount"]), 0.5)
        life     = st.slider("설비 수명 (년)", 15,  30,   KR["lifetime"])
        cap      = st.slider("가동률 (%)",     70,  95,   KR["capacity"])
        maint    = st.number_input("유지보수율 (%/년)", value=KR["maint"])
        operators= st.number_input("운전원 수 (명)",   value=KR["operators"])
        labor    = st.number_input("인건비 (원/명/년)", value=KR["labor"])
        st.caption(f"{labor:,} 원/명/년")
        ci       = st.number_input("건설비 지수",      value=KR["ci"])

# ─────────────────────────────────────────────────────────────────────────────
# 6. 공통 파라미터 딕셔너리
# ─────────────────────────────────────────────────────────────────────────────
REG = fit_models()
sl_LG, ic_LG, r2_LG = REG["LG"]
sl_We, ic_We, r2_We = REG["We"]
df_lit = pd.DataFrame(LIT)

eco = dict(scale=scale, cap=cap, life=life, dr=dr, fx=fx,
           elec=elec, lps=lps, mps=mps, hps=hps, cooling=cooling,
           maint=maint, operators=operators, labor=labor, ci=ci)

# ─────────────────────────────────────────────────────────────────────────────
# 7. 라이선스별 계산
# ─────────────────────────────────────────────────────────────────────────────
RES = {}
for lic in sel:
    p  = LICENSE[lic]
    LG = LG_from_SRD(p["SRD"], sl_LG, ic_LG)
    Tr = T_sat(p["steam_P"])
    we = calc_We(p["SRD"], p["steam_P"], LG, T_cold, P_final)
    sl = calc_sol_loss(lic, fgd["O2"], fgd["T"], Tr, fgd["NOx"], fgd["SOx"])
    sp = calc_SPECCA(p["SRD"], we["We_elec"], p["capture"])
    coca_p = {**eco, "SRD": p["SRD"], "LG": LG, "steam_P": p["steam_P"],
              "We_elec": we["We_elec"], "sol_loss": sl["grand"],
              "sol_price": p["sol_price"]}
    co = calc_COCA(coca_p)
    RES[lic] = {"SRD": p["SRD"], "LG": LG, "T_reb": Tr,
                "grade": "LPS" if p["steam_P"]<=7 else ("MPS" if p["steam_P"]<=20 else "HPS"),
                "we": we, "sol": sl, "SPECCA": sp, "coca": co, "color": p["color"]}

# ─────────────────────────────────────────────────────────────────────────────
# 8. 탭 렌더링
# ─────────────────────────────────────────────────────────────────────────────
tabs = st.tabs(["① 종합 비교", "② 에너지 분해", "③ 경제성 분석",
                "④ 흡수제 손실", "⑤ 트렌드 분석", "⑥ 신기술 예측", "⑦ Custom 입력", "⑧ 참고문헌"])

# ── TAB 1: 종합 비교 ──────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown("### 라이선스 종합 비교")
    if not RES:
        st.warning("왼쪽 사이드바에서 라이선스를 하나 이상 선택하세요.")
    else:
        # KPI 카드
        best_We   = min(RES, key=lambda x: RES[x]["we"]["We_total"])
        best_COCA = min(RES, key=lambda x: RES[x]["coca"]["COCA_usd"])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("최저 We",   f"{RES[best_We]['we']['We_total']:.3f} GJe/tCO₂",  best_We)
        c2.metric("최저 COCA", f"${RES[best_COCA]['coca']['COCA_usd']:.0f}/tCO₂", best_COCA)
        if "MEA Generic" in RES and len(RES) > 1:
            mea_w = RES["MEA Generic"]["we"]["We_total"]
            bw    = RES[best_We]["we"]["We_total"]
            c3.metric("MEA 대비 We 절감", f"{(mea_w-bw)/mea_w*100:.1f}%", best_We)
            mea_c = RES["MEA Generic"]["coca"]["COCA_usd"]
            bc    = RES[best_COCA]["coca"]["COCA_usd"]
            c4.metric("MEA 대비 COCA 절감", f"{(mea_c-bc)/mea_c*100:.1f}%", best_COCA)
        else:
            c3.metric("CO₂ 최종 압력", f"{P_final} bar")
            c4.metric("냉각수 온도",   f"{T_cold} °C")

        st.divider()

        # 막대 차트 3개
        col_a, col_b, col_c = st.columns(3)
        def bar_chart(title, y_vals, y_label, fmt=".2f"):
            fig = go.Figure([go.Bar(
                x=list(RES.keys()),
                y=y_vals,
                marker_color=[RES[l]["color"] for l in RES],
                text=[f"{v:{fmt}}" for v in y_vals],
                textposition="outside",
            )])
            fig.update_layout(title=title, yaxis_title=y_label, height=450,
                              showlegend=False,
                              plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                              margin=dict(t=40, b=20, l=20, r=10))
            return fig

        col_a.plotly_chart(bar_chart("We (GJe/tCO₂)",
            [RES[l]["we"]["We_total"] for l in RES], "GJe/tCO₂", fmt=".3f"), use_container_width=True)
        col_b.plotly_chart(bar_chart("SPECCA (MJ/tCO₂)",
            [RES[l]["SPECCA"] for l in RES], "MJ/tCO₂", fmt=",.0f"), use_container_width=True)
        col_c.plotly_chart(bar_chart("COCA (USD/tCO₂)",
            [RES[l]["coca"]["COCA_usd"] for l in RES], "USD/tCO₂", fmt=",.0f"), use_container_width=True)

        # 상세 테이블
        rows = []
        for lic, r in RES.items():
            rows.append({
                "라이선스": lic,
                "SRD (GJ/t)": r["SRD"],
                "L/G (추정)": f"{r['LG']:.1f}",
                "T_reb (°C)": f"{r['T_reb']:.0f}",
                "스팀 등급": r["grade"],
                "We (GJe/t)": f"{r['we']['We_total']:.3f}",
                "SPECCA (MJ/t)": f"{r['SPECCA']:,.0f}",
                "COCA (USD/t)": f"${r['coca']['COCA_usd']:,.0f}",
                "COCA (만원/t)": f"{r['coca']['COCA_man']:,.1f}",
                "흡수제 손실 (kg/t)": f"{r['sol']['grand']:.2f}",
                "추정 불확도": "±15%",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("⚠️ SRD는 라이선서 공개값, L/G·We·COCA는 문헌 회귀 모델 추정 (±15%). "
                   "배가스 발생원: " + fg_src)


# ── TAB 2: 에너지 분해 ────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown("### We 에너지 구성 분해")
    if not RES:
        st.info("라이선스를 선택하세요.")
    else:
        COMPS = [("We_thermal","재생열 (Carnot 변환)","#E24B4A"),
                 ("We_pump",   "펌프 (L/G 비례)",     "#EF9F27"),
                 ("We_blower", "블로워 (FG 압손)",     "#97C459"),
                 ("We_compress","CO₂ 압축",            "#85B7EB"),
                 ("We_liquefy","액화 / 최종 압력",     "#AFA9EC")]

        fig_stk = go.Figure()
        for key, label, col in COMPS:
            fig_stk.add_trace(go.Bar(
                name=label, x=list(RES.keys()),
                y=[RES[l]["we"][key] for l in RES],
                marker_color=col,
            ))
        fig_stk.update_layout(barmode="stack", height=500,
                               title="We 구성 (GJe/tCO₂)",
                               yaxis_title="GJe/tCO₂",
                               plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_stk, use_container_width=True)

        # Carnot 상세
        st.markdown("#### Carnot 변환 상세")
        crows = []
        for lic, r in RES.items():
            crows.append({
                "라이선스": lic,
                "스팀 압력 (bar)": LICENSE[lic]["steam_P"],
                "T_steam (°C)": f"{r['T_reb']:.0f}",
                "T_cold (°C)": T_cold,
                "Carnot η": f"{r['we']['eta']:.4f}",
                "SRD": r["SRD"],
                "We_thermal": f"{r['we']['We_thermal']:.3f}",
                "We_elec": f"{r['we']['We_elec']:.3f}",
                "SRD 중 열 비율": f"{r['we']['We_thermal']/r['we']['We_total']*100:.1f}%",
            })
        st.dataframe(pd.DataFrame(crows), use_container_width=True, hide_index=True)
        st.caption(f"Carnot η = 1 - T_cold / T_steam | CO₂ 최종 압력: {P_final} bar")


# ── TAB 3: 경제성 분석 ───────────────────────────────────────────────────────
with tabs[2]:
    st.markdown("### COCA 구성 분해")
    if not RES:
        st.info("라이선스를 선택하세요.")
    else:
        COST_ITEMS = [
            ("ann_cap",   "CAPEX 연간화",  "#E24B4A"),
            ("ann_steam", "스팀 비용",     "#EF9F27"),
            ("ann_elec",  "전력 비용",     "#85B7EB"),
            ("ann_sol",   "흡수제 비용",   "#5DCAA5"),
            ("ann_maint", "유지보수",      "#AFA9EC"),
            ("ann_labor", "인건비",        "#888780"),
        ]
        fig_ca = go.Figure()
        for key, label, col in COST_ITEMS:
            fig_ca.add_trace(go.Bar(
                name=label, x=list(RES.keys()),
                y=[RES[l]["coca"][key] for l in RES],
                marker_color=col,
            ))
        fig_ca.update_layout(barmode="stack", height=500,
                              title="연간 비용 구성 (억원/년)",
                              yaxis_title="억원/년",
                              plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_ca, use_container_width=True)

        # CAPEX / 연간 CO₂
        cols = st.columns(len(RES))
        for i, (lic, r) in enumerate(RES.items()):
            with cols[i]:
                st.metric(f"{lic} — TPC", f"{r['coca']['TPC_bil']:,.0f} 억원",
                          delta=f"${r['coca']['TPC_mUSD']:,.1f}M USD")
                st.metric("연간 CO₂", f"{r['coca']['ann_CO2']/1e4:,.1f} 만 tCO₂/년")
                st.metric("COCA (USD/tCO₂)", f"${r['coca']['COCA_usd']:,.0f}")
                st.metric("COCA (원/tCO₂)", f"{r['coca']['COCA_man']*10000:,.0f} 원")


# ── TAB 4: 흡수제 손실 ───────────────────────────────────────────────────────
with tabs[3]:
    st.markdown("### 흡수제 손실 분석")

    with st.expander("🔧 배가스 조성 / 전처리 조건", expanded=True):
        fg1, fg2, fg3, fg4 = st.columns(4)
        with fg1:
            O2_in  = st.number_input("O₂ (%)",    min_value=0.0, max_value=21.0, value=float(fgd["O2"]), step=0.5,
                                     help="산화분해의 핵심 변수")
            NOx_in = st.number_input("NOx (ppm)", value=int(fgd["NOx"]),
                                     help="HSS 생성·니트로사민 리스크")
        with fg2:
            SOx_in = st.number_input("SOx (ppm)", value=int(fgd["SOx"]),
                                     help="황산염 HSS — FGD 없으면 치명적")
            T_abs_in = st.number_input("흡수탑 온도 (°C)", value=int(fgd["T"]))
        with fg3:
            wash_n = st.selectbox("Water wash 단수", [1, 2, 3], index=1)
            FGD = st.selectbox("FGD 유무", ["있음", "없음"])
        with fg4:
            SCR = st.selectbox("SCR 유무", ["있음", "없음"])
            st.caption(f"NOx 적용값: ×{'0.1' if SCR=='있음' else '1.0'}")
            st.caption(f"SOx 적용값: ×{'0.05' if FGD=='있음' else '1.0'}")

    eff_NOx = NOx_in * (0.1 if SCR == "있음" else 1.0)
    eff_SOx = SOx_in * (0.05 if FGD == "있음" else 1.0)

    # 경고
    if FGD == "없음" and eff_SOx > 50:
        st.markdown(f'<div class="error-box">⚠️ SOx {eff_SOx:.0f} ppm → FGD 전처리 필수. '
                    'HSS 가속 생성으로 흡수제 수명 급감, 포집 성능 조기 저하.</div>', unsafe_allow_html=True)
    if SCR == "없음" and eff_NOx > 150:
        st.markdown(f'<div class="warning-box">⚠️ NOx {eff_NOx:.0f} ppm → 니트로사민(발암물질) 생성 리스크. '
                    'SCR 또는 water wash 강화 권고.</div>', unsafe_allow_html=True)

    sol_res = {}
    for lic in sel:
        T_reb_s = T_sat(LICENSE[lic]["steam_P"])
        sol_res[lic] = calc_sol_loss(lic, O2_in, T_abs_in, T_reb_s,
                                     eff_NOx, eff_SOx, wash_n)

    if sol_res:
        SOL_COMPS = [("oxidative","산화분해","#E24B4A"), ("thermal","열분해","#EF9F27"),
                     ("evaporative","증발","#85B7EB"), ("hss","HSS 손실","#888780")]
        fig_sol = go.Figure()
        for key, label, col in SOL_COMPS:
            fig_sol.add_trace(go.Bar(
                name=label, x=list(sol_res.keys()),
                y=[sol_res[l][key] for l in sol_res],
                marker_color=col,
            ))
        fig_sol.update_layout(barmode="stack", height=480,
                               title="흡수제 손실 구성 (kg/tCO₂)",
                               plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_sol, use_container_width=True)

        ann_CO2_s = scale * cap / 100
        mcols = st.columns(len(sol_res))
        for i, lic in enumerate(sol_res):
            loss = sol_res[lic]["grand"]
            price = LICENSE[lic]["sol_price"]
            ann_c = loss * ann_CO2_s * price / 1e8
            mcols[i].metric(lic,
                             f"{ann_c:,.1f} 억원/년",
                             delta=f"손실 {loss:.2f} kg/tCO₂")

        # O₂ 민감도
        st.markdown("#### O₂ 농도 민감도")
        O2_range = np.linspace(0.5, 16, 80)
        fig_o2 = go.Figure()
        for lic in sel:
            T_reb_s = T_sat(LICENSE[lic]["steam_P"])
            losses = [calc_sol_loss(lic, o2, T_abs_in, T_reb_s, eff_NOx, eff_SOx, wash_n)["total"]
                      for o2 in O2_range]
            fig_o2.add_trace(go.Scatter(x=O2_range, y=losses,
                                        name=lic, line=dict(color=LICENSE[lic]["color"], width=2)))
        fig_o2.add_vline(x=O2_in, line_dash="dot", line_color="gray",
                          annotation_text=f"현재 {O2_in}%")
        fig_o2.update_layout(title="O₂ 농도 vs 흡수제 손실율",
                              xaxis_title="O₂ (%)", yaxis_title="kg/tCO₂", height=450,
                              plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_o2, use_container_width=True)


# ── TAB 5: 트렌드 분석 ───────────────────────────────────────────────────────
with tabs[4]:
    st.markdown(f"### 문헌 데이터 트렌드 분석 ({len(df_lit)}개 포인트)")
    x_rng = np.linspace(1.8, 4.3, 120)

    col5a, col5b = st.columns(2)

    def scatter_reg(df_x, df_y, slope, intercept, r2, x_rng,
                    title, xt, yt, sel_lics, y_key):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_x, y=df_y, mode="markers", name="문헌 데이터",
            text=df_lit["source"],
            marker=dict(color="#7F77DD", size=6, opacity=0.65)))
        fig.add_trace(go.Scatter(
            x=x_rng, y=slope*x_rng+intercept,
            mode="lines", name=f"회귀선 R²={r2:.2f}",
            line=dict(color="#E24B4A", width=2, dash="dash")))
        # ±15% band
        yhat = slope*x_rng+intercept
        fig.add_trace(go.Scatter(
            x=np.concatenate([x_rng, x_rng[::-1]]),
            y=np.concatenate([yhat*1.15, (yhat*0.85)[::-1]]),
            fill="toself", fillcolor="rgba(200,200,200,0.15)",
            line=dict(color="rgba(0,0,0,0)"), name="±15%"))
        for lic in sel_lics:
            fig.add_trace(go.Scatter(
                x=[LICENSE[lic]["SRD"]], y=[RES[lic]["we"][y_key] if y_key in RES[lic]["we"] else RES[lic][y_key]],
                mode="markers+text", name=lic,
                text=[lic], textposition="top center",
                marker=dict(color=LICENSE[lic]["color"], size=11, symbol="star")))
        fig.update_layout(title=title, xaxis_title=xt, yaxis_title=yt,
                          height=480, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        return fig

    with col5a:
        fig_lg = go.Figure()
        fig_lg.add_trace(go.Scatter(x=df_lit["SRD"], y=df_lit["LG"],
            mode="markers", name="문헌",
            text=df_lit["source"], marker=dict(color="#7F77DD", size=6, opacity=0.65)))
        fig_lg.add_trace(go.Scatter(x=x_rng, y=sl_LG*x_rng+ic_LG,
            mode="lines", name=f"회귀 R²={r2_LG:.2f}",
            line=dict(color="#E24B4A", width=2, dash="dash")))
        for lic in sel:
            fig_lg.add_trace(go.Scatter(x=[LICENSE[lic]["SRD"]], y=[RES[lic]["LG"]],
                mode="markers+text", name=lic, text=[lic], textposition="top center",
                marker=dict(color=LICENSE[lic]["color"], size=11, symbol="star")))
        fig_lg.update_layout(
            title=f"SRD → L/G 트레이드오프<br><sub>L/G = {sl_LG:.2f}×SRD + {ic_LG:.2f}</sub>",
            xaxis_title="SRD (GJ/tCO₂)", yaxis_title="L/G (L/Nm³)",
            height=480, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_lg, use_container_width=True)

    with col5b:
        fig_we2 = go.Figure()
        fig_we2.add_trace(go.Scatter(x=df_lit["SRD"], y=df_lit["We"],
            mode="markers", name="문헌",
            text=df_lit["source"], marker=dict(color="#7F77DD", size=6, opacity=0.65)))
        yhat_we = sl_We*x_rng+ic_We
        fig_we2.add_trace(go.Scatter(x=x_rng, y=yhat_we,
            mode="lines", name=f"회귀 R²={r2_We:.2f}",
            line=dict(color="#E24B4A", width=2, dash="dash")))
        fig_we2.add_trace(go.Scatter(
            x=np.concatenate([x_rng, x_rng[::-1]]),
            y=np.concatenate([yhat_we*1.15, (yhat_we*0.85)[::-1]]),
            fill="toself", fillcolor="rgba(200,200,200,0.15)",
            line=dict(color="rgba(0,0,0,0)"), name="±15%"))
        for lic in sel:
            fig_we2.add_trace(go.Scatter(
                x=[LICENSE[lic]["SRD"]], y=[RES[lic]["we"]["We_total"]],
                mode="markers+text", name=lic, text=[lic], textposition="top center",
                marker=dict(color=LICENSE[lic]["color"], size=11, symbol="star")))
        fig_we2.update_layout(
            title=f"SRD → We 관계<br><sub>We = {sl_We:.3f}×SRD + {ic_We:.3f}</sub>",
            xaxis_title="SRD (GJ/tCO₂)", yaxis_title="We (GJe/tCO₂)",
            height=480, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_we2, use_container_width=True)

    # SRD vs 흡수제 손실
    sl_sl, ic_sl, r2_sl = REG["sol"]
    fig_sl = go.Figure()
    fig_sl.add_trace(go.Scatter(x=df_lit["SRD"], y=df_lit["sol_loss"],
        mode="markers", name="문헌",
        text=df_lit["source"], marker=dict(color="#7F77DD", size=6, opacity=0.65)))
    po, exp_ok = REG["sol_exp"]
    if exp_ok:
        def expf(x, a, b): return a * np.exp(b * x)
        fig_sl.add_trace(go.Scatter(x=x_rng, y=expf(x_rng, *po),
            mode="lines", name="지수 회귀", line=dict(color="#1D9E75", width=2, dash="dash")))
    for lic in sel:
        fig_sl.add_trace(go.Scatter(
            x=[LICENSE[lic]["SRD"]], y=[RES[lic]["sol"]["total"]],
            mode="markers+text", name=lic, text=[lic], textposition="top center",
            marker=dict(color=LICENSE[lic]["color"], size=11, symbol="star")))
    fig_sl.update_layout(
        title="SRD → 흡수제 손실율 관계",
        xaxis_title="SRD (GJ/tCO₂)", yaxis_title="흡수제 손실 (kg/tCO₂)",
        height=450, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_sl, use_container_width=True)
    st.caption("흡수제 손실은 배가스 조성(O₂, NOx, SOx)에 의해 크게 달라짐 — ④탭 배가스 조성 반영")


# ── TAB 6: 재생↔포집 트레이드오프 + 비용 추정 ──────────────────────────────
with tabs[5]:
    st.markdown("### 재생에너지(SRD) 입력 → 전체 에너지·비용 자동 추정")
    st.markdown(
        '<div class="info-box">'
        '💡 <b>핵심 원리</b>: SRD(재생에너지)를 낮추면 흡수제 순환량(L/G)이 늘어나 '
        '포집 측 에너지(펌프·블로워)가 증가합니다. '
        'We_total = <b>재생 기여</b>(SRD×Carnot) + <b>포집 기여</b>(L/G×계수) + 압축·액화. '
        '아래에서 SRD 하나만 입력하면 이 trade-off와 전체 비용을 자동 산출합니다.'
        '</div>', unsafe_allow_html=True)

    st.markdown("---")
    # ── 입력 패널 ──────────────────────────────────────────────────────────
    inp1, inp2, inp3 = st.columns([1, 1, 1])
    with inp1:
        n_name   = st.text_input("기술명 / 흡수제명", "신규 흡수제 A")
        n_SRD    = st.slider("① 재생에너지 SRD (GJ/tCO₂)", 1.5, 4.5, 2.3, 0.05,
                             help="라이선스사가 공개하는 핵심 수치. 이 값 하나로 나머지를 추정합니다.")
    with inp2:
        n_steamP = st.selectbox("② 스팀 등급 (스트리퍼 압력)",
                                [("LPS 5 bar", 5.0), ("MPS 15 bar", 15.0), ("HPS 40 bar", 40.0)],
                                format_func=lambda x: x[0])[1]
        n_amine  = st.selectbox("③ 흡수제 유형 (비용 추정용)",
                                ["MEA Generic", "KoSol (KEPCO)", "MHI KS-1", "Custom"])
    with inp3:
        n_LG_manual = st.checkbox("L/G 직접 입력 (공개된 경우)")
        if n_LG_manual:
            n_LG = st.number_input("L/G (L/Nm³)", value=4.5, step=0.1,
                                   help="직접 입력 시 회귀 추정값 대신 사용됩니다.")
        else:
            n_LG = LG_from_SRD(n_SRD, sl_LG, ic_LG)
            st.info(f"L/G 자동 추정 (회귀): **{n_LG:.2f}** L/Nm³\n\n"
                    f"*SRD ↓ → L/G ↑ : 이것이 trade-off*")

    # ── 계산 ───────────────────────────────────────────────────────────────
    n_Tr    = T_sat(n_steamP)
    n_we    = calc_We(n_SRD, n_steamP, n_LG, T_cold, P_final)
    n_sol   = calc_sol_loss(n_amine, fgd["O2"], fgd["T"], n_Tr, fgd["NOx"], fgd["SOx"])
    n_sp    = calc_SPECCA(n_SRD, n_we["We_elec"])
    n_coca_p = {**eco, "SRD": n_SRD, "LG": n_LG, "steam_P": n_steamP,
                "We_elec": n_we["We_elec"], "sol_loss": n_sol["grand"],
                "sol_price": LICENSE.get(n_amine, LICENSE["MEA Generic"])["sol_price"]}
    n_coca  = calc_COCA(n_coca_p)

    # ── 요약 메트릭 ────────────────────────────────────────────────────────
    st.markdown("#### 📊 추정 결과")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("We_thermal (재생)",
              f"{n_we['We_thermal']:.3f} GJe/t",
              help="SRD × Carnot 효율 — 스팀에서 온 에너지")
    m2.metric("We_capture (포집)",
              f"{(n_we['We_pump']+n_we['We_blower']):.3f} GJe/t",
              help="펌프(L/G 비례) + 블로워")
    m3.metric("We_total",
              f"{n_we['We_total']:.3f} GJe/t",
              delta=f"재생 {n_we['We_thermal']/n_we['We_total']*100:.0f}% / 포집 {(n_we['We_pump']+n_we['We_blower'])/n_we['We_total']*100:.0f}%")
    m4.metric("SPECCA",          f"{n_sp:,.0f} MJ/t")
    m5.metric("COCA (USD/tCO₂)", f"${n_coca['COCA_usd']:,.0f}",
              delta=f"{n_coca['COCA_man']*10000:,.0f} 원/tCO₂")

    st.markdown("---")
    # ── 차트 행 1: Trade-off 곡선 + 에너지 분해 막대 ──────────────────────
    st.markdown("#### 재생 ↔ 포집 Trade-off 시각화")
    tc1, tc2 = st.columns(2)

    # [좌] SRD 스펙트럼에 따른 에너지 성분 변화
    srd_arr = np.linspace(1.5, 4.5, 60)
    th_arr, cap_arr, tot_arr = [], [], []
    for s in srd_arr:
        lg_s  = LG_from_SRD(s, sl_LG, ic_LG)
        we_s  = calc_We(s, n_steamP, lg_s, T_cold, P_final)
        th_arr.append(we_s["We_thermal"])
        cap_arr.append(we_s["We_pump"] + we_s["We_blower"])
        tot_arr.append(we_s["We_total"])

    fig_tradeoff = go.Figure()
    fig_tradeoff.add_trace(go.Scatter(
        x=srd_arr, y=th_arr, mode="lines", name="재생에너지 기여 (We_thermal)",
        line=dict(color="#E24B4A", width=2.5),
        fill="tozeroy", fillcolor="rgba(226,75,74,0.08)"))
    fig_tradeoff.add_trace(go.Scatter(
        x=srd_arr, y=cap_arr, mode="lines", name="포집에너지 기여 (We_pump+blower)",
        line=dict(color="#378ADD", width=2.5),
        fill="tozeroy", fillcolor="rgba(55,138,221,0.08)"))
    fig_tradeoff.add_trace(go.Scatter(
        x=srd_arr, y=tot_arr, mode="lines", name="We_total",
        line=dict(color="#333333", width=2, dash="dot")))
    # 현재 입력값 표시
    fig_tradeoff.add_vline(x=n_SRD, line_width=2, line_dash="dash", line_color="#EF9F27",
                           annotation_text=f"← 현재 SRD {n_SRD}", annotation_position="top right")
    fig_tradeoff.add_trace(go.Scatter(
        x=[n_SRD], y=[n_we["We_total"]],
        mode="markers+text", name=n_name,
        text=[f"★ {n_we['We_total']:.3f}"], textposition="top center",
        marker=dict(color="#EF9F27", size=14, symbol="star",
                    line=dict(color="#854F0B", width=2))))
    # 기존 라이선스 포인트
    for lic in sel:
        fig_tradeoff.add_trace(go.Scatter(
            x=[LICENSE[lic]["SRD"]], y=[RES[lic]["we"]["We_total"]],
            mode="markers+text", name=lic, text=[lic], textposition="bottom center",
            marker=dict(color=LICENSE[lic]["color"], size=10, symbol="diamond")))
    fig_tradeoff.update_layout(
        title="SRD 변화에 따른 에너지 성분 분리<br>"
              "<sub>SRD↓ → 재생에너지↓ but 포집에너지↑ → We_total 감소폭이 SRD만큼 크지 않음</sub>",
        xaxis_title="SRD — 재생에너지 (GJ/tCO₂)",
        yaxis_title="We 성분 (GJe/tCO₂)",
        height=520, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.22, font=dict(size=11)))
    tc1.plotly_chart(fig_tradeoff, use_container_width=True)

    # [우] 현재 입력값 에너지 성분 stacked bar
    comps = {
        "재생열 (We_thermal)":       n_we["We_thermal"],
        "순환펌프 (We_pump)":        n_we["We_pump"],
        "블로워 (We_blower)":        n_we["We_blower"],
        "압축 (We_compress)":        n_we["We_compress"],
        "액화 (We_liquefy)":         n_we["We_liquefy"],
    }
    colors_comp = ["#E24B4A","#378ADD","#85B7EB","#1D9E75","#EF9F27"]
    fig_bar = go.Figure()
    for (label, val), col in zip(comps.items(), colors_comp):
        pct_v = val / n_we["We_total"] * 100
        fig_bar.add_trace(go.Bar(
            name=label, x=[n_name], y=[val],
            marker_color=col,
            text=[f"{val:.3f}<br>({pct_v:.0f}%)"],
            textposition="inside"))
    fig_bar.update_layout(
        barmode="stack",
        title=f"'{n_name}' 에너지 성분 분해<br>"
              f"<sub>재생 {n_we['We_thermal']/n_we['We_total']*100:.0f}% | "
              f"포집·후처리 {(1-n_we['We_thermal']/n_we['We_total'])*100:.0f}%</sub>",
        yaxis_title="We (GJe/tCO₂)", height=520,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.22, font=dict(size=11)))
    tc2.plotly_chart(fig_bar, use_container_width=True)

    # ── 차트 행 2: 문헌 대비 위치 + COCA 비교 ─────────────────────────────
    st.markdown("#### 문헌 대비 위치 및 경제성")
    bc1, bc2 = st.columns(2)

    # [좌] 문헌 산점도에서 위치
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=df_lit["SRD"], y=df_lit["We"],
        mode="markers", name="문헌 데이터",
        text=df_lit["source"], marker=dict(color="#D3D1C7", size=6, opacity=0.7)))
    yhat = sl_We * x_rng + ic_We
    fig_pred.add_trace(go.Scatter(x=x_rng, y=yhat, mode="lines",
        name=f"회귀선 R²={r2_We:.2f}", line=dict(color="#888780", width=1.5, dash="dash")))
    fig_pred.add_trace(go.Scatter(
        x=np.concatenate([x_rng, x_rng[::-1]]),
        y=np.concatenate([yhat*1.15, (yhat*0.85)[::-1]]),
        fill="toself", fillcolor="rgba(180,180,180,0.1)",
        line=dict(color="rgba(0,0,0,0)"), name="±15%"))
    for lic in sel:
        fig_pred.add_trace(go.Scatter(
            x=[LICENSE[lic]["SRD"]], y=[RES[lic]["we"]["We_total"]],
            mode="markers+text", name=lic, text=[lic], textposition="bottom center",
            marker=dict(color=LICENSE[lic]["color"], size=9, symbol="diamond")))
    pct = np.sum(df_lit["We"].values > n_we["We_total"]) / len(df_lit) * 100
    fig_pred.add_trace(go.Scatter(
        x=[n_SRD], y=[n_we["We_total"]],
        mode="markers+text", name=n_name,
        text=[f"★ {n_name}"], textposition="top center",
        marker=dict(color="#EF9F27", size=16, symbol="star",
                    line=dict(color="#854F0B", width=2))))
    fig_pred.update_layout(
        title=f"'{n_name}' — 문헌 상위 {pct:.0f}% (We 기준)",
        xaxis_title="SRD (GJ/tCO₂)", yaxis_title="We (GJe/tCO₂)",
        height=480, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    bc1.plotly_chart(fig_pred, use_container_width=True)

    # [우] SRD 변화에 따른 COCA 곡선
    coca_arr = []
    for s, lg_s in zip(srd_arr, [LG_from_SRD(s, sl_LG, ic_LG) for s in srd_arr]):
        we_s = calc_We(s, n_steamP, lg_s, T_cold, P_final)
        sol_s = calc_sol_loss(n_amine, fgd["O2"], fgd["T"], T_sat(n_steamP),
                              fgd["NOx"], fgd["SOx"])
        cp = {**eco, "SRD": s, "LG": lg_s, "steam_P": n_steamP,
              "We_elec": we_s["We_elec"], "sol_loss": sol_s["grand"],
              "sol_price": LICENSE.get(n_amine, LICENSE["MEA Generic"])["sol_price"]}
        coca_arr.append(calc_COCA(cp)["COCA_usd"])

    fig_coca = go.Figure()
    fig_coca.add_trace(go.Scatter(
        x=srd_arr, y=coca_arr, mode="lines", name="COCA (USD/tCO₂)",
        line=dict(color="#1D9E75", width=2.5),
        fill="tozeroy", fillcolor="rgba(29,158,117,0.07)"))
    fig_coca.add_vline(x=n_SRD, line_width=2, line_dash="dash", line_color="#EF9F27",
                       annotation_text=f"현재 ${n_coca['COCA_usd']:.0f}/t",
                       annotation_position="top right")
    for lic in sel:
        lic_coca = RES[lic]["coca"]["COCA_usd"]
        fig_coca.add_trace(go.Scatter(
            x=[LICENSE[lic]["SRD"]], y=[lic_coca],
            mode="markers+text", name=lic, text=[lic], textposition="top center",
            marker=dict(color=LICENSE[lic]["color"], size=9, symbol="diamond")))
    fig_coca.update_layout(
        title="SRD → COCA 곡선<br><sub>재생에너지 절감이 비용에 미치는 민감도</sub>",
        xaxis_title="SRD (GJ/tCO₂)", yaxis_title="COCA (USD/tCO₂)",
        height=480, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    bc2.plotly_chart(fig_coca, use_container_width=True)

    # ── 신뢰성 판단 ────────────────────────────────────────────────────────
    expected_LG = sl_LG * n_SRD + ic_LG
    if n_SRD < 1.9:
        st.markdown('<div class="error-box">⚠️ SRD가 문헌 최솟값(~2.0) 미만입니다. '
                    '열역학적으로 가능한지 검증이 필요합니다.</div>', unsafe_allow_html=True)
    elif n_SRD < 2.2:
        st.markdown(f'<div class="warning-box">⚠️ SRD {n_SRD} GJ/tCO₂는 문헌 하위 5% 수준. '
                    f'회귀 모델 예상 L/G ≥ {expected_LG:.1f} L/Nm³ — 흡수탑 CAPEX 대폭 증가 가능성.</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="info-box">✅ SRD {n_SRD} GJ/tCO₂는 문헌 범위 내. '
                    f'예상 L/G: {expected_LG:.1f} L/Nm³ | Carnot η: {n_we["eta"]:.3f} | '
                    f'스팀 온도: {n_we["T_steam"]:.1f}°C</div>', unsafe_allow_html=True)


# ── TAB 7: Custom 입력 ───────────────────────────────────────────────────────
with tabs[6]:
    st.markdown("### Custom 라이선스 직접 입력")
    st.caption("공개된 스펙만 입력하면 나머지는 회귀 모델로 추정됩니다.")

    with st.form("custom_form"):
        cf1, cf2, cf3 = st.columns(3)
        with cf1:
            c_name    = st.text_input("라이선스명", "신규 공정 B")
            c_SRD     = st.number_input("SRD (GJ/tCO₂)", 1.5, 5.0, 2.8, 0.1)
            c_cap     = st.number_input("포집률 (%)", 50, 99, 90)
        with cf2:
            c_steamP  = st.number_input("Stripper 압력 (bar)", 1.0, 60.0, 5.0, 0.5)
            c_amine   = st.selectbox("흡수제 유형 (손실 추정용)",
                                      ["MEA Generic","KoSol (KEPCO)","MHI KS-1","Custom"])
            c_sprice  = st.number_input("흡수제 단가 (원/kg)", 1000, 30000, 3500)
        with cf3:
            c_LG_know = st.checkbox("L/G 직접 입력")
            if c_LG_know:
                c_LG = st.number_input("L/G (L/Nm³)", 2.0, 10.0, 4.2, 0.1)
            else:
                st.caption("L/G → 회귀 모델 자동 추정")
                c_LG = None
            c_color   = st.color_picker("차트 색상", "#F4C0D1")

        submitted = st.form_submit_button("🔄 계산 실행", use_container_width=True)

    if submitted:
        c_LG_v = c_LG if c_LG_know else LG_from_SRD(c_SRD, sl_LG, ic_LG)
        c_Tr   = T_sat(c_steamP)
        c_we   = calc_We(c_SRD, c_steamP, c_LG_v, T_cold, P_final)
        c_sl   = calc_sol_loss(c_amine, fgd["O2"], fgd["T"], c_Tr, fgd["NOx"], fgd["SOx"])
        c_sp   = calc_SPECCA(c_SRD, c_we["We_elec"], c_cap/100)
        c_cp   = {**eco, "SRD": c_SRD, "LG": c_LG_v, "steam_P": c_steamP,
                  "We_elec": c_we["We_elec"], "sol_loss": c_sl["grand"], "sol_price": c_sprice}
        c_co   = calc_COCA(c_cp)

        st.success(f"**{c_name}** 계산 완료")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("We",               f"{c_we['We_total']:.3f} GJe/tCO₂")
        m2.metric("SPECCA",           f"{c_sp:,.0f} MJ/tCO₂")
        m3.metric("COCA (USD/tCO₂)", f"${c_co['COCA_usd']:,.0f}",
                  delta=f"{c_co['COCA_man']*10000:,.0f} 원/tCO₂")
        m4.metric("흡수제 손실",      f"{c_sl['grand']:.2f} kg/tCO₂")
        m5.metric("TPC",              f"{c_co['TPC_bil']:,.0f} 억원")

        if not c_LG_know:
            st.info(f"L/G 회귀 추정: {c_LG_v:.2f} L/Nm³  (문헌 회귀 기반, ±20%)")

        # 기존 라이선스와 비교
        st.markdown("#### 기존 라이선스 대비 비교")
        comp_rows = []
        for lic, r in RES.items():
            comp_rows.append({
                "라이선스": lic,
                "SRD": r["SRD"], "L/G": f"{r['LG']:.1f}",
                "We": r["we"]["We_total"], "COCA (USD)": r["coca"]["COCA_usd"],
                "흡수제 손실": r["sol"]["grand"],
            })
        comp_rows.append({
            "라이선스": f"★ {c_name}",
            "SRD": c_SRD, "L/G": f"{c_LG_v:.1f}",
            "We": c_we["We_total"], "COCA (USD)": c_co["COCA_usd"],
            "흡수제 손실": c_sl["grand"],
        })
        df_comp = pd.DataFrame(comp_rows)
        st.dataframe(df_comp, use_container_width=True, hide_index=True)


# ── TAB 8: 참고문헌 ──────────────────────────────────────────────────────────
with tabs[7]:
    st.markdown("### 📚 참고문헌 (References)")
    st.caption("본 툴의 문헌 데이터·계산 모델·벤치마크 기준값에 사용된 전체 출처 목록입니다.")

    # ── 1. NETL / DOE 공식 보고서 ────────────────────────────────────────────
    with st.expander("🏛 NETL / DOE 공식 기술보고서", expanded=True):
        st.markdown("""
| # | 출처 코드 | 전체 인용 | 비고 |
|---|-----------|-----------|------|
| 1 | **NETL 2022(B12B)** | U.S. DOE/NETL. *Cost and Performance Baseline for Fossil Energy Plants, Volume 1: Bituminous Coal and Natural Gas to Electricity*, Revision 4a. DOE/NETL-2023/4320. October 2022. | SC PC + Cansolv DC-103 90% capture. SRD=3.56 GJ/tCO₂ (Exhibit 4-80) |
| 2 | **NETL 2022(B31B)** | 상동 | NGCC + Cansolv DC-103 90% capture. SRD=3.47 GJ/tCO₂, 스팀 4.5 bar (Exhibit 5-23) |
| 3 | **NETL 2022(B11B)** | 상동 | SubC PC + Cansolv DC-103 90% capture. SRD=3.38 GJ/tCO₂, SPECCA=3,610 MJ/tCO₂ |
| 4 | **NETL 2019(Rev3)** | U.S. DOE/NETL. *Cost and Performance Baseline for Fossil Energy Plants*, Revision 3. DOE/NETL-2010/1397. 2015 (updated 2019). | Cansolv DC-103 SRD Rev3 기준 2.56 GJ/tCO₂ |
| 5 | **NETL 2019(Rev4)** | U.S. DOE/NETL. *Cost and Performance Baseline for Fossil Energy Plants*, Revision 4. 2019. | Cansolv DC-103 SRD Rev4 기준 2.44 GJ/tCO₂ (참고용, 설계 최적화 수치) |
""", unsafe_allow_html=False)

    # ── 2. IEAGHG 벤치마크 연구 ──────────────────────────────────────────────
    with st.expander("🌍 IEAGHG 벤치마크 연구", expanded=True):
        st.markdown("""
| # | 출처 코드 | 전체 인용 |
|---|-----------|-----------|
| 6 | **IEAGHG 2007** | IEAGHG. *Improvement in Power Generation with Post-Combustion Capture of CO₂*. Report 2004/4, Updated 2007. |
| 7 | **IEAGHG 2011** | IEAGHG. *Oxy Combustion Processes for CO₂ Capture from Power Plant*. 2011/3. |
| 8 | **IEAGHG 2012** | IEAGHG. *CO₂ Capture at Gas Fired Power Plants*. 2012/8. |
| 9 | **IEAGHG 2013** | IEAGHG. *Evaluation of Post-Combustion CO₂ Capture Solvent R&D Priorities*. 2013/6. |
| 10 | **IEAGHG 2014** | IEAGHG. *Assessment of Emerging CO₂ Capture Technologies and their Potential to Reduce Costs*. 2014/TR4. |
""")

    # ── 3. 학술논문 ──────────────────────────────────────────────────────────
    with st.expander("📄 학술논문 (Peer-reviewed)", expanded=True):
        st.markdown("""
| # | 출처 코드 | 전체 인용 |
|---|-----------|-----------|
| 11 | **Rochelle 2009** | Rochelle, G.T. (2009). Amine Scrubbing for CO₂ Capture. *Science*, 325(5948), 1652–1654. |
| 12 | **Rochelle 2011** | Rochelle, G.T. et al. (2011). Aqueous piperazine as the new standard for CO₂ capture technology. *Chem. Eng. J.*, 171(3), 725–733. |
| 13 | **Rochelle 2012** | Rochelle, G.T. et al. (2012). Pilot plant demonstration of aqueous piperazine. *Energy Procedia*, 23, 172–179. |
| 14 | **Rochelle 2013** | Lin, Y.J. & Rochelle, G.T. (2013). Approaching a reversible stripping process for CO₂ capture. *Chem. Eng. J.*, 230, 178–186. |
| 15 | **NTNU 2011** | Knuutila, H. et al. (2011). CO₂ capture from simulated flue gas with MEA. *Int. J. Greenhouse Gas Control*, 5(4), 817–827. |
| 16 | **NTNU 2015** | Luo, X. et al. (2015). Comparison of MEA and MDEA/PZ blends for CO₂ capture. *Energy Procedia*, 63, 1216–1223. |
| 17 | **SINTEF 2013** | Hoff, K.A. et al. (2013). Solvent development and process optimization for post-combustion CO₂ capture. *Energy Procedia*, 37, 292–299. |
| 18 | **Nguyen 2010** | Nguyen, T. et al. (2010). Amine Volatility in CO₂ Capture. *Int. J. Greenhouse Gas Control*, 4(5), 723–729. |
| 19 | **Voice & Rochelle 2011** | Voice, A.K. & Rochelle, G.T. (2011). Oxidation of amines at absorber conditions. *Energy Procedia*, 4, 171–178. |
| 20 | **Aspen bench** | Aspen Technology. Aspen Plus® Rate-Based Model Benchmark for MEA 30wt% CO₂ capture. Internal validation study. |
""")

    # ── 4. 공정 라이선스 기술 자료 ───────────────────────────────────────────
    with st.expander("🏭 공정 라이선스 기술 자료", expanded=True):
        st.markdown("""
| # | 출처 코드 | 전체 인용 | 비고 |
|---|-----------|-----------|------|
| 21 | **MHI 2012** | Mitsubishi Heavy Industries. KS-1™ Solvent CO₂ Capture Technology. *MHI Technical Review*, 49(2), 2012. | KS-1 SRD 2.4~2.6 GJ/tCO₂ |
| 22 | **MHI 2014** | Iijima, M. et al. (2014). Commercial-scale demonstration of KS-1 process. *Energy Procedia*, 63, 6111–6119. | 국내 발전사 도입 사례 포함 |
| 23 | **Shell 2013** | Shell Cansolv Technologies. DC-103 Solvent Performance Data. Technical Bulletin, 2013. | SRD 참고값 |
| 24 | **Fluor 2011** | Fluor Corporation. Econamine FG Plus™ Technology. *Energy Procedia*, 4, 1397–1402. 2011. | Econamine+ SRD 3.2~3.3 GJ/tCO₂ |
| 25 | **Alstom 2012** | Alstom Power. Advanced Chilled Ammonia Process for CO₂ Capture. DOE/NETL-2012. | 냉각암모니아 공정 SRD ~2.0 GJ/tCO₂ |
""")

    # ── 5. KoSol (KEPCO) 국내 자료 ───────────────────────────────────────────
    with st.expander("🇰🇷 KoSol (KEPCO) 국내 연구 자료", expanded=True):
        st.markdown("""
| # | 출처 코드 | 전체 인용 | 비고 |
|---|-----------|-----------|------|
| 26 | **KEPCO 2018** | 한국전력공사 전력연구원. KoSol-4 포집 공정 성능 실증 결과 보고서. 2018. | KoSol-4 SRD 실측값 |
| 27 | **KEPCO 2019** | 한국전력공사 전력연구원. KoSol-5 개발 및 파일럿 성능 평가. 2019. | KoSol-5 SRD 실측값 |
| 28 | **KEPCO 2021** | 한국전력공사 전력연구원. LPS 기반 KoSol 공정 최적화 연구. 2021. | KoSol-5 운전 최적화 |

> **주의**: KoSol은 한전(KEPCO) 개발 공정. KIER 개발 KIERSOL(K₂CO₃ 기반)과 별개 기술임.
""")

    # ── 6. 방법론 기준 ───────────────────────────────────────────────────────
    with st.expander("⚙️ 계산 방법론 기준", expanded=False):
        st.markdown("""
| # | 항목 | 출처 및 기준 |
|---|------|-------------|
| 29 | **We (Equivalent Work)** | Oyenekan, B.A. & Rochelle, G.T. (2006). Energy Performance of Stripper Configurations for CO₂ Capture by Aqueous Amines. *Ind. Eng. Chem. Res.*, 45(8), 2457–2464. |
| 30 | **SPECCA** | IEAGHG (2011). Oxy Combustion Processes. 수식: (HR_cap - HR_ref) / (E_ref - E_cap) |
| 31 | **COCA / LCOE 방법론** | NETL Quality Guidelines for Energy System Studies (QGESS). DOE/NETL-2019/2121. 2019. |
| 32 | **CAPEX 6/10 스케일 법칙** | Towler, G. & Sinnott, R. *Chemical Engineering Design*, 2nd ed., Elsevier, 2013. Ch. 9. |
| 33 | **Guthrie CAPEX 방법** | Guthrie, K.M. (1969). Data and techniques for preliminary capital cost estimating. *Chem. Eng.*, 76(6), 114–142. |
| 34 | **Carnot 효율** | 열역학 기본식: η = 1 − T_cold / T_steam (절대온도 기준) |
| 35 | **Antoine 포화온도** | Perry's Chemical Engineers' Handbook, 8th ed. Antoine equation for water. |
| 36 | **흡수제 산화분해 (Arrhenius)** | Nguyen, T. et al. (2010) [No.18 참조]; Voice & Rochelle (2011) [No.19 참조]. |
| 37 | **흡수제 열분해 (Arrhenius)** | Davis, J. & Rochelle, G. (2009). Thermal degradation of MEA. *Energy Procedia*, 1(1), 327–333. |
| 38 | **L/G–SRD 회귀** | 본 툴 내장 문헌 26개 포인트 선형 회귀 (scipy.stats.linregress). |
| 39 | **CO₂ 압축 전력** | NETL B12B 실측 보정: 44.8 MWe ÷ 578.7 tCO₂/hr = 0.279 GJe/tCO₂ (1.5→153 bar, 8단). |
""")

    # ── 7. 한국 유틸리티 단가 기준 ──────────────────────────────────────────
    with st.expander("💰 한국 유틸리티·재무 기준값", expanded=False):
        st.markdown("""
| 항목 | 기준값 | 출처 |
|------|--------|------|
| 전기요금 (산업용) | 120 원/kWh | 한국전력공사 산업용(을) 요금표, 2025 |
| LPS 스팀 (5 bar) | 25,000 원/GJ | 국내 발전소 열공급 단가 추정 (산업통상자원부 기준) |
| MPS 스팀 (15 bar) | 32,000 원/GJ | 동상 |
| HPS 스팀 (40 bar) | 38,000 원/GJ | 동상 |
| 냉각수 | 2,500 원/GJ | 국내 냉각탑 운전비 추정 |
| 환율 | 1,350 원/USD | 2025년 기준 적용값 (변동 가능) |
| 할인율 | 8.0 %/yr | 국내 발전 인프라 재무 기준 (한전 기준) |
| 플랜트 수명 | 25 년 | NETL QGESS 기준 준용 |
| 가동률 | 85 % | NETL QGESS 기준 준용 |
| 건설비 지수 (CI) | 1.15 | 한국/미국 건설 단가 비율 추정 |
""")

    st.info("💡 문헌 접근이 어려운 경우: NETL 보고서는 https://www.netl.doe.gov/coal 에서 무료 다운로드 가능. "
            "IEAGHG 보고서는 https://ieaghg.org/publications/technical-reports 참조.")


# ─────────────────────────────────────────────────────────────────────────────
# 9. 푸터
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "📚 방법론: IEAGHG Benchmark Studies (2007~2014) · NETL Cost & Performance Baseline · "
    "Rochelle et al. (Energy Procedia) · KIER KoSol 보고서  |  "
    "⚠️ 본 툴의 CAPEX·OPEX 추산은 ±20% 불확도를 포함하며, 투자 결정 전 상세 설계 검증 필요  |  "
    "한국 유틸리티 단가 기준: 산업용 전기 120원/kWh, LPS 스팀 25,000원/GJ (2025 기준)"
)
