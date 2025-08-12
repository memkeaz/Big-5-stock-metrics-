import streamlit as st
import pandas as pd
import numpy as np
import requests

# =================== Page ===================
st.set_page_config(page_title="Phil Town Big 5 Screener", layout="wide")
st.title("Phil Town Big 5 — 10-Year Screener")
st.caption("Big 5 (Sales, EPS, Equity, FCF CAGR + 10-yr Avg ROIC), 10/5/3/1 breakdowns, Rule #1 EPS & FCF DCF valuations with live MOS sliders, and a Value-Investor summary.")

# =================== Top controls ===================
c1, c2 = st.columns([3,1])
with c1:
    st.info("If results don't show, click **Clear Cache** then search again.")
with c2:
    if st.button("Clear Cache"):
        st.cache_data.clear()
        st.success("Cache cleared. Run a new search.")

# =================== Sidebar ===================
st.sidebar.header("Settings")
provider = st.sidebar.radio("Data Provider", ["Alpha Vantage", "FMP"])
ticker = st.text_input("Enter ticker (e.g., AAPL, MSFT, ADBE):", value="MSFT").strip().upper()
run = st.button("Search")

AV_KEY = st.secrets.get("ALPHAVANTAGE_API_KEY", "").strip()
FMP_KEY = st.secrets.get("FMP_API_KEY", "").strip()
OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", "").strip()

st.sidebar.markdown("### Valuation Assumptions")
# Rule #1 EPS — app uses LOWER of (your estimate, 10y EPS CAGR, analyst 5y EPS growth), capped at 15%
years_eps = 10  # Rule #1 standard
growth_eps_user = st.sidebar.number_input("Your EPS growth estimate (annual, %)", 0.0, 50.0, 12.0, step=0.5) / 100.0
auto_pe = st.sidebar.checkbox("Terminal P/E ≈ lower of (2× growth, current P/E), capped at 50", value=True)
terminal_pe_manual = st.sidebar.number_input("Terminal P/E (manual if Auto off)", 5.0, 60.0, 20.0, step=0.5)

# FCF DCF
years_dcf = st.sidebar.slider("Years (FCF DCF)", 5, 15, 10)
growth_fcf = st.sidebar.number_input("FCF growth (annual, %)", 0.0, 50.0, 10.0, step=0.5) / 100.0
terminal_g = st.sidebar.number_input("Terminal growth (FCF, %)", 0.0, 6.0, 3.0, step=0.25) / 100.0

# Discount (MARR)
discount = st.sidebar.number_input("MARR / Discount rate (%, both models)", 4.0, 20.0, 10.0, step=0.5) / 100.0

# Default MOS in sidebar (used to prefill live sliders)
st.sidebar.markdown("### Default MOS (prefill for live sliders)")
mos_eps_pct_default = st.sidebar.slider("Default MOS for Rule #1 EPS (%)", 0, 90, 50, step=5) / 100.0
mos_dcf_pct_default = st.sidebar.slider("Default MOS for FCF DCF (%)", 0, 90, 50, step=5) / 100.0

# =================== Demo ===================
def demo_msft_df():
    years = list(range(2015, 2025))
    shares = [7900, 7800, 7750, 7720, 7700, 7680, 7650, 7450, 7420, 7400]  # millions (demo)
    df = pd.DataFrame({
        "Revenue": [93580, 85320, 89950, 110360, 125843, 143015, 168088, 198270, 211915, 245000],
        "EPS":     [2.48, 2.79, 3.31, 2.13, 5.76, 6.20, 8.05, 9.21, 9.68, 11.00],
        "Equity":  [72163, 82718, 82572, 82572, 118304, 118304, 166542, 166542, 194000, 210000],
        "FCF":     [23969, 31378, 31922, 32694, 45230, 45300, 56300, 65700, 67800, 78000],
        "ROIC":    [0.12, 0.13, 0.15, 0.10, 0.18, 0.19, 0.21, 0.22, 0.20, 0.21],
        "SharesDiluted": shares
    }, index=years).astype(float)
    return df.iloc[-10:], years[-10:]

# =================== Helpers ===================
def cagr_over_years(first_val, last_val, first_year, last_year):
    try:
        years = last_year - first_year
        if years <= 0 or first_val <= 0 or last_val <= 0 or pd.isna(first_val) or pd.isna(last_val):
            return np.nan
        return (last_val / first_val) ** (1 / years) - 1
    except Exception:
        return np.nan

def series_cagr_gap(s: pd.Series) -> float:
    y = s.dropna()
    if len(y) < 2: return np.nan
    return cagr_over_years(y.iloc[0], y.iloc[-1], int(y.index[0]), int(y.index[-1]))

def yoy(series: pd.Series) -> float:
    y = series.dropna()
    if len(y) < 2: return np.nan
    prev, last = y.iloc[-2], y.iloc[-1]
    if prev <= 0 or pd.isna(prev) or pd.isna(last): return np.nan
    return (last / prev) - 1

def pct(x: float) -> str:
    return "—" if (x is None or pd.isna(x)) else f"{x*100:.1f}%"

def safe_mean(s: pd.Series) -> float:
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    return s.mean() if len(s) else np.nan

# -------------------- Alpha Vantage fetch (patched for accuracy) --------------------
AV_BASE = "https://www.alphavantage.co/query"

def av_get(fn: str, symbol: str, apikey: str):
    params = {"function": fn, "symbol": symbol, "apikey": apikey}
    r = requests.get(AV_BASE, params=params, timeout=30)
    r.raise_for_status()
    j = r.json()
    return j.get("annualReports", [])

def av_series(reports, field) -> pd.Series:
    """Return a numeric series by year for a single AV field (if present)."""
    if not reports: return pd.Series(dtype="float64")
    rows = []
    for rep in reports:
        y = pd.to_datetime(rep.get("fiscalDateEnding", ""), errors="coerce").year
        if pd.isna(y): continue
        raw = rep.get(field, None)
        val = pd.to_numeric(raw, errors="coerce")
        rows.append((int(y), val))
    if not rows: return pd.Series(dtype="float64")
    s = pd.Series(dict(rows)).sort_index()
    return s.iloc[-11:].astype("float64")

def av_series_sum(reports, fields) -> pd.Series:
    """Sum multiple AV fields (useful for total debt = short + long)."""
    total = None
    for f in fields:
        s = av_series(reports, f)
        if total is None:
            total = s
        else:
            total = total.add(s, fill_value=0)
    return total if total is not None else pd.Series(dtype="float64")

@st.cache_data(show_spinner=False, ttl=1800)
def fetch_alpha_vantage(symbol: str, apikey: str):
    inc = av_get("INCOME_STATEMENT", symbol, apikey)
    bal = av_get("BALANCE_SHEET",  symbol, apikey)
    cfs = av_get("CASH_FLOW",      symbol, apikey)

    # Income statement
    revenue        = av_series(inc, "totalRevenue")
    net_income     = av_series(inc, "netIncome")
    diluted_eps    = av_series(inc, "dilutedEPS")
    ebit           = av_series(inc, "ebit")
    tax_expense    = av_series(inc, "incomeTaxExpense")
    pretax_income  = av_series(inc, "incomeBeforeTax")

    # Shares: AV often lacks weighted-average diluted shares.
    # Use balance-sheet end-of-period shares as a robust proxy.
    shares_diluted = av_series(bal, "commonStockSharesOutstanding")

    # Balance sheet
    equity = av_series(bal, "totalShareholderEquity")

    # Debt: build from components to avoid missing 'totalDebt'
    # Try (shortTermDebt + longTermDebt), else (currentLongTermDebt + longTermDebtNoncurrent), else fallbacks.
    debt_primary   = av_series_sum(bal, ["shortTermDebt", "longTermDebt"])
    debt_alt       = av_series_sum(bal, ["currentLongTermDebt", "longTermDebtNoncurrent"])
    debt_fallback1 = av_series(bal, "totalDebt")
    debt_fallback2 = av_series(bal, "shortLongTermDebtTotal")
    total_debt = debt_primary
    if total_debt is None or total_debt.empty:
        total_debt = debt_alt
    if total_debt is None or total_debt.empty:
        total_debt = debt_fallback1
    if total_debt is None or total_debt.empty:
        total_debt = debt_fallback2

    # Cash: handle multiple AV variants
    cash = av_series(bal, "cashAndCashEquivalentsAtCarryingValue")
    if cash.empty:
        cash = av_series(bal, "cashAndCashEquivalents")
    if cash.empty:
        cash = av_series(bal, "cashAndShortTermInvestments")

    # Cash flow
    cfo   = av_series(cfs, "operatingCashflow")
    capex = av_series(cfs, "capitalExpenditures")

    # Align years
    years = sorted(
        set(revenue.index) | set(net_income.index) | set(diluted_eps.index) |
        set(shares_diluted.index) | set(ebit.index) | set(tax_expense.index) |
        set(pretax_income.index) | set(equity.index) | set(total_debt.index) |
        set(cash.index) | set(cfo.index) | set(capex.index)
    )[-11:]

    def A(s): return s.reindex(years).astype("float64") if years else pd.Series(dtype="float64")

    revenue, net_income, diluted_eps, shares_diluted, ebit, tax_expense, pretax_income, equity, total_debt, cash, cfo, capex = [
        A(x) for x in [revenue, net_income, diluted_eps, shares_diluted, ebit, tax_expense, pretax_income, equity, total_debt, cash, cfo, capex]
    ]

    # EPS fallback if dilutedEPS missing
    eps = diluted_eps.copy()
    if eps.isna().all() and not net_income.isna().all() and not shares_diluted.isna().all():
        with np.errstate(invalid="ignore", divide="ignore"):
            eps = net_income / shares_diluted.replace({0: np.nan})

    # FCF = CFO − CapEx  (CapEx on AV is usually negative; subtracting handles sign)
    if not cfo.isna().all() and not capex.isna().all():
        fcf = (cfo - capex)
    else:
        fcf = pd.Series([np.nan]*len(years), index=years)

    # ROIC = NOPAT / (Debt + Equity − Cash)
    with np.errstate(divide="ignore", invalid="ignore"):
        tax_rate = (tax_expense / pretax_income).clip(0, 1)
    nopat = (ebit * (1 - tax_rate.fillna(0.21))) if not ebit.isna().all() else net_income
    invested_capital = (total_debt.fillna(0) + equity.fillna(0) - cash.fillna(0)).replace({0: np.nan})
    with np.errstate(divide="ignore", invalid="ignore"):
        roic = (nopat / invested_capital).replace([np.inf, -np.inf], np.nan)

    df = pd.DataFrame({
        "Revenue": revenue,
        "EPS": eps,
        "Equity": equity,
        "FCF": fcf,
        "ROIC": roic,
        "SharesDiluted": shares_diluted  # used for FCF per-share
    }).sort_index().tail(11)

    return df, years, "Alpha Vantage (patched)"



