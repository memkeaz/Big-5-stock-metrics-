import streamlit as st
import pandas as pd
import numpy as np
import requests
from typing import Tuple, List

# -------------------- Page Config --------------------
st.set_page_config(page_title="Phil Town Big 5 Screener", layout="wide")
st.title("Phil Town Big 5 — 10-Year Screener")
st.caption("Big 5 (Sales, EPS, Equity, FCF CAGR + 10-yr Avg ROIC), full 10/5/3/1 breakdowns, Intrinsic Value (EPS & FCF DCF), and a Value-Investor summary.")

# -------------------- Cache Control --------------------
scol1, scol2 = st.columns([3, 1])
with scol1:
    st.info("If results don't show, click **Clear Cache** then search again.")
with scol2:
    if st.button("Clear Cache"):
        st.cache_data.clear()
        st.success("Cache cleared. Run a new search.")

# -------------------- Sidebar --------------------
st.sidebar.header("Settings")
provider = st.sidebar.radio("Data Provider", ["Alpha Vantage", "FMP"])
ticker = st.text_input("Enter ticker (e.g., AAPL, MSFT, ADBE):", value="MSFT").strip().upper()
run = st.button("Search")

AV_KEY = st.secrets.get("ALPHAVANTAGE_API_KEY", "").strip()
FMP_KEY = st.secrets.get("FMP_API_KEY", "").strip()
OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", "").strip()

# ---------- Valuation assumptions (with your defaults) ----------
st.sidebar.markdown("### Valuation Assumptions")
# EPS (P/E) model
years_eps = st.sidebar.slider("Years (EPS model)", 5, 15, 10)
growth_eps = st.sidebar.number_input("EPS growth (annual, %)", 0.0, 50.0, 12.0, step=0.5) / 100.0
auto_pe = st.sidebar.checkbox("Auto terminal P/E ≈ 2× growth (cap 50, floor 8)", value=True)
terminal_pe_input = st.sidebar.number_input("Terminal P/E (override when Auto off)", 5.0, 60.0, 20.0, step=0.5)
# FCF DCF model
years_dcf = st.sidebar.slider("Years (FCF DCF)", 5, 15, 10)
growth_fcf = st.sidebar.number_input("FCF growth (annual, %)", 0.0, 50.0, 10.0, step=0.5) / 100.0
terminal_g = st.sidebar.number_input("Terminal growth (FCF, %)", 0.0, 6.0, 3.0, step=0.25) / 100.0
# Discount rate (default 10%)
discount = st.sidebar.number_input("Discount rate (both models, %)", 4.0, 20.0, 10.0, step=0.5) / 100.0

def terminal_pe_from_growth(g):
    return max(8.0, min(50.0, (g * 100.0) * 2.0))
terminal_pe = terminal_pe_from_growth(growth_eps) if auto_pe else terminal_pe_input

# -------------------- Demo Mode --------------------
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

# -------------------- Helpers --------------------
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

# -------------------- Alpha Vantage fetch --------------------
AV_BASE = "https://www.alphavantage.co/query"

def av_get(fn: str, symbol: str, apikey: str):
    params = {"function": fn, "symbol": symbol, "apikey": apikey}
    r = requests.get(AV_BASE, params=params, timeout=30)
    r.raise_for_status()
    j = r.json()
    return j.get("annualReports", [])

def av_series(reports, field) -> pd.Series:
    if not reports: return pd.Series(dtype="float64")
    rows = []
    for rep in reports:
        y = pd.to_datetime(rep.get("fiscalDateEnding", ""), errors="coerce").year
        if pd.isna(y): continue
        val = pd.to_numeric(rep.get(field, None), errors="coerce")
        rows.append((int(y), val))
    if not rows: return pd.Series(dtype="float64")
    s = pd.Series(dict(rows)).sort_index()
    return s.iloc[-11:].astype("float64")

@st.cache_data(show_spinner=False, ttl=1800)
def fetch_alpha_vantage(symbol: str, apikey: str):
    inc = av_get("INCOME_STATEMENT", symbol, apikey)
    bal = av_get("BALANCE_SHEET",  symbol, apikey)
    cfs = av_get("CASH_FLOW",      symbol, apikey)

    revenue        = av_series(inc, "totalRevenue")
    net_income     = av_series(inc, "netIncome")
    diluted_eps    = av_series(inc, "dilutedEPS")
    diluted_shares = av_series(inc, "weightedAverageShsOutDil")
    ebit           = av_series(inc, "ebit")
    tax_expense    = av_series(inc, "incomeTaxExpense")
    pretax_income  = av_series(inc, "incomeBeforeTax")

    equity     = av_series(bal, "totalShareholderEquity")
    total_debt = av_series(bal, "totalDebt") if not av_series(bal, "totalDebt").empty else av_series(bal, "shortLongTermDebtTotal")
    cash       = av_series(bal, "cashAndCashEquivalentsAtCarryingValue")
    if cash.empty: cash = av_series(bal, "cashAndShortTermInvestments")

    cfo   = av_series(cfs, "operatingCashflow")
    capex = av_series(cfs, "capitalExpenditures")

    years = sorted(set(revenue.index) | set(net_income.index) | set(diluted_eps.index) |
                   set(diluted_shares.index) | set(ebit.index) | set(tax_expense.index) |
                   set(pretax_income.index) | set(equity.index) | set(total_debt.index) |
                   set(cash.index) | set(cfo.index) | set(capex.index))[-11:]

    def A(s): return s.reindex(years).astype("float64") if years else pd.Series(dtype="float64")
    revenue, net_income, diluted_eps, diluted_shares, ebit, tax_expense, pretax_income, equity, total_debt, cash, cfo, capex = [
        A(x) for x in [revenue, net_income, diluted_eps, diluted_shares, ebit, tax_expense, pretax_income, equity, total_debt, cash, cfo, capex]
    ]

    # EPS fallback (NI / diluted shares)
    eps = diluted_eps.copy()
    if eps.isna().all() and not net_income.isna().all() and not diluted_shares.isna().all():
        with np.errstate(invalid="ignore", divide="ignore"):
            eps = net_income / diluted_shares.replace({0: np.nan})

    # FCF = CFO − CapEx
    fcf = (cfo - capex) if (not cfo.isna().all() and not capex.isna().all()) else pd.Series([np.nan]*len(years), index=years)

    # ROIC = NOPAT / Invested Capital
    with np.errstate(divide="ignore", invalid="ignore"):
        tax_rate = (tax_expense / pretax_income).clip(0, 1)
    nopat = (ebit * (1 - tax_rate.fillna(0.21))) if not ebit.isna().all() else net_income
    invested_capital = (total_debt.fillna(0) + equity.fillna(0) - cash.fillna(0)).replace({0: np.nan})
    with np.errstate(divide="ignore", invalid="ignore"):
        roic = (nopat / invested_capital).replace([np.inf, -np.inf], np.nan)

    df = pd.DataFrame({
        "Revenue": revenue, "EPS": eps, "Equity": equity, "FCF": fcf, "ROIC": roic, "SharesDiluted": diluted_shares
    }).sort_index().tail(11)
    return df, years, "Alpha Vantage"

# -------------------- FMP fetch --------------------
FMP_BASE = "https://financialmodelingprep.com/api/v3"

def fmp_get(path: str, apikey: str, params=None):
    if params is None: params = {}
    params["apikey"] = apikey
    r = requests.get(f"{FMP_BASE}/{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def fmp_series(reports, field) -> pd.Series:
    if not reports: return pd.Series(dtype="float64")
    rows = []
    for rep in reports:
        y = pd.to_datetime(rep.get("date") or rep.get("calendarYear"), errors="coerce").year
        if pd.isna(y):
            try: y = int(rep.get("calendarYear"))
            except: continue
        val = pd.to_numeric(rep.get(field, None), errors="coerce")
        rows.append((int(y), val))
    if not rows: return pd.Series(dtype="float64")
    s = pd.Series(dict(rows)).sort_index()
    return s.iloc[-11:].astype("float64")

@st.cache_data(show_spinner=False, ttl=1800)
def fetch_fmp(symbol: str, apikey: str):
    inc = fmp_get(f"income-statement/{symbol}", apikey, {"period":"annual", "limit": 40})
    bal = fmp_get(f"balance-sheet-statement/{symbol}", apikey, {"period":"annual", "limit": 40})
    cfs = fmp_get(f"cash-flow-statement/{symbol}", apikey, {"period":"annual", "limit": 40})

    revenue        = fmp_series(inc, "revenue")
    net_income     = fmp_series(inc, "netIncome")
    diluted_eps    = fmp_series(inc, "epsdiluted")
    diluted_shares = fmp_series(inc, "weightedAverageShsOutDil")
    ebit           = fmp_series(inc, "ebit")
    tax_expense    = fmp_series(inc, "incomeTaxExpense")
    pretax_income  = fmp_series(inc, "incomeBeforeTax")

    equity     = fmp_series(bal, "totalStockholdersEquity")
    total_debt = fmp_series(bal, "totalDebt")
    cash       = fmp_series(bal, "cashAndCashEquivalents")

    cfo   = fmp_series(cfs, "netCashProvidedByOperatingActivities")
    capex = fmp_series(cfs, "capitalExpenditure")

    years = sorted(set(revenue.index) | set(net_income.index) | set(diluted_eps.index) |
                   set(diluted_shares.index) | set(ebit.index) | set(tax_expense.index) |
                   set(pretax_income.index) | set(equity.index) | set(total_debt.index) |
                   set(cash.index) | set(cfo.index) | set(capex.index))[-11:]

    def A(s): return s.reindex(years).astype("float64") if years else pd.Series(dtype="float64")
    revenue, net_income, diluted_eps, diluted_shares, ebit, tax_expense, pretax_income, equity, total_debt, cash, cfo, capex = [
        A(x) for x in [revenue, net_income, diluted_eps, diluted_shares, ebit, tax_expense, pretax_income, equity, total_debt, cash, cfo, capex]
    ]

    eps = diluted_eps.copy()
    if eps.isna().all() and not net_income.isna().all() and not diluted_shares.isna().all():
        with np.errstate(invalid="ignore", divide="ignore"):
            eps = net_income / diluted_shares.replace({0: np.nan})

    fcf = (cfo - capex) if (not cfo.isna().all() and not capex.isna().all()) else pd.Series([np.nan]*len(years), index=years)

    with np.errstate(divide="ignore", invalid="ignore"):
        tax_rate = (tax_expense / pretax_income).clip(0, 1)
    nopat = (ebit * (1 - tax_rate.fillna(0.21))) if not ebit.isna().all() else net_income
    invested_capital = (total_debt.fillna(0) + equity.fillna(0) - cash.fillna(0)).replace({0: np.nan})
    with np.errstate(divide="ignore", invalid="ignore"):
        roic = (nopat / invested_capital).replace([np.inf, -np.inf], np.nan)

    df = pd.DataFrame({
        "Revenue": revenue, "EPS": eps, "Equity": equity, "FCF": fcf, "ROIC": roic, "SharesDiluted": diluted_shares
    }).sort_index().tail(11)
    return df, years, "FMP"

# -------------------- Current Price --------------------
def get_price_alpha_vantage(symbol: str, apikey: str) -> float:
    try:
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={apikey}"
        j = requests.get(url, timeout=30).json()
        p = float(j.get("Global Quote", {}).get("05. price", "nan"))
        return p
    except Exception:
        return np.nan

def get_price_fmp(symbol: str, apikey: str) -> float:
    try:
        url = f"https://financialmodelingprep.com/api/v3/quote-short/{symbol}?apikey={apikey}"
        j = requests.get(url, timeout=30).json()
        if isinstance(j, list) and j:
            return float(j[0].get("price", "nan"))
        return np.nan
    except Exception:
        return np.nan

# -------------------- Intrinsic Value Models --------------------
def intrinsic_eps_model(eps_last: float, growth: float, years: int, terminal_pe: float, discount: float) -> float:
    if pd.isna(eps_last) or eps_last <= 0: return np.nan
    eps_future = eps_last * ((1 + growth) ** years)
    future_price = eps_future * terminal_pe
    intrinsic_now = future_price / ((1 + discount) ** years)
    return intrinsic_now

def intrinsic_dcf_fcf(fcf_per_share_last: float, growth: float, years: int, terminal_g: float, discount: float) -> float:
    if pd.isna(fcf_per_share_last) or fcf_per_share_last <= 0: return np.nan
    if discount <= terminal_g: return np.nan
    pv = 0.0
    f = fcf_per_share_last
    for t in range(1, years + 1):
        f *= (1 + growth)
        pv += f / ((1 + discount) ** t)
    f_next = f * (1 + terminal_g)
    tv = f_next / (discount - terminal_g)
    pv += tv / ((1 + discount) ** years)
    return pv

# -------------------- Run Search --------------------
if run:
    using_av = (provider == "Alpha Vantage")
    key_ok = (AV_KEY if using_av else FMP_KEY)
    st.info(f"Provider: **{provider}** · API key set: **{'Yes' if key_ok else 'No'}** · Ticker: **{ticker}**")

    if not key_ok:
        st.error("Missing API key in Streamlit Secrets.")
        df, years = demo_msft_df()
        source = "Demo (sample)"
        current_price = np.nan
    else:
        try:
            if using_av:
                df, years, source = fetch_alpha_vantage(ticker, AV_KEY)
                current_price = get_price_alpha_vantage(ticker, AV_KEY)
            else:
                df, years, source = fetch_fmp(ticker, FMP_KEY)
                current_price = get_price_fmp(ticker, FMP_KEY)
            if df.empty:
                st.warning("No data returned — switching to Demo Mode.")
                df, years = demo_msft_df()
                source = "Demo (sample)"
                current_price = np.nan
        except Exception as e:
            st.error(f"Error: {e}")
            df, years = demo_msft_df()
            source = "Demo (sample)"
            current_price = np.nan

    # -------------------- Big 5 (10-year) --------------------
    def series_cagr_gap(s: pd.Series) -> float:
        y = s.dropna()
        if len(y) < 2: return np.nan
        return cagr_over_years(y.iloc[0], y.iloc[-1], int(y.index[0]), int(y.index[-1]))

    sales_cagr_10 = series_cagr_gap(df["Revenue"])
    eps_cagr_10   = series_cagr_gap(df["EPS"])
    eqty_cagr_10  = series_cagr_gap(df["Equity"])
    fcf_cagr_10   = series_cagr_gap(df["FCF"])
    roic_avg_10   = safe_mean(df["ROIC"])

    def pf(v): return "PASS ✅" if not pd.isna(v) and v >= 0.10 else ("—" if pd.isna(v) else "FAIL ❌")
    def fmt(x): return "—" if pd.isna(x) else f"{x*100:.1f}%"

    big5 = pd.DataFrame({
        "Metric": ["Sales (Revenue) CAGR","EPS CAGR","Equity CAGR","FCF CAGR","ROIC (10-yr Avg)"],
        "Value (10y)":  [fmt(sales_cagr_10), fmt(eps_cagr_10), fmt(eqty_cagr_10), fmt(fcf_cagr_10), fmt(roic_avg_10)],
        "Pass ≥10%?": [pf(sales_cagr_10), pf(eps_cagr_10), pf(eqty_cagr_10), pf(fcf_cagr_10), pf(roic_avg_10)]
    })

    st.subheader(f"Big 5 — 10-Year Check · {ticker}  ·  Source: {source}")
    st.dataframe(big5, use_container_width=True)

    # -------------------- Metric Breakdown (10 / First-5 / Last-3 / Last-1) --------------------
    # For growth metrics (Sales/EPS/Equity/FCF): use CAGR windows; Last-1 is YoY
    def breakdown_growth(s: pd.Series):
        s = s.dropna()
        if len(s) < 2:
            return np.nan, np.nan, np.nan, np.nan
        ten = series_cagr_gap(s)
        first5 = np.nan
        if len(s) >= 5:
            window = s.iloc[:5]
            first5 = cagr_over_years(window.iloc[0], window.iloc[-1], int(window.index[0]), int(window.index[-1]))
        last3 = np.nan
        if len(s) >= 4:
            w = s.iloc[-4:]
            last3 = cagr_over_years(w.iloc[0], w.iloc[-1], int(w.index[0]), int(w.index[-1]))
        last1 = yoy(s)
        return ten, first5, last3, last1

    # For ROIC: use averages (Last-1 is the last ROIC)
    def breakdown_roic(s: pd.Series):
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        if len(s) == 0:
            return np.nan, np.nan, np.nan, np.nan
        ten = safe_mean(s)
        first5 = safe_mean(s.iloc[:5]) if len(s) >= 5 else np.nan
        last3 = safe_mean(s.iloc[-3:]) if len(s) >= 1 else np.nan
        last1 = s.iloc[-1]
        return ten, first5, last3, last1

    # Compute all rows
    sales_b = breakdown_growth(df["Revenue"])
    eps_b   = breakdown_growth(df["EPS"])
    eqty_b  = breakdown_growth(df["Equity"])
    fcf_b   = breakdown_growth(df["FCF"])
    roic_b  = breakdown_roic(df["ROIC"])

    breakdown_df = pd.DataFrame({
        "Metric": ["Sales CAGR","EPS CAGR","Equity CAGR","FCF CAGR","ROIC"],
        "10yr":   [sales_b[0], eps_b[0], eqty_b[0], fcf_b[0], roic_b[0]],
        "First 5yr": [sales_b[1], eps_b[1], eqty_b[1], fcf_b[1], roic_b[1]],
        "Last 3yr":  [sales_b[2], eps_b[2], eqty_b[2], fcf_b[2], roic_b[2]],
        "Last 1yr":  [sales_b[3], eps_b[3], eqty_b[3], fcf_b[3], roic_b[3]],
    })

    breakdown_fmt = breakdown_df.copy()
    for col in ["10yr","First 5yr","Last 3yr","Last 1yr"]:
        breakdown_fmt[col] = breakdown_fmt[col].apply(lambda v: "—" if pd.isna(v) else f"{v*100:.1f}%")

    st.markdown("### Metric Breakdown (10 / First‑5 / Last‑3 / Last‑1)")
    st.dataframe(breakdown_fmt, use_container_width=True)

    # -------------------- Data Coverage --------------------
    st.markdown("#### Data Coverage (non-missing values used)")
    coverage = df.notna().sum().rename("Valid Years").to_frame()
    coverage["Out of"] = len(df.index)
    st.dataframe(coverage.T, use_container_width=True)

    # -------------------- Intrinsic Value --------------------
    st.markdown("### Intrinsic Value")
    last_eps = df["EPS"].dropna().iloc[-1] if df["EPS"].notna().any() else np.nan

    shares_last = df["SharesDiluted"].dropna().iloc[-1] if "SharesDiluted" in df and df["SharesDiluted"].notna().any() else np.nan
    fcf_last = df["FCF"].dropna().iloc[-1] if df["FCF"].notna().any() else np.nan
    fcf_per_share_last = (fcf_last / shares_last) if (not pd.isna(fcf_last) and not pd.isna(shares_last) and shares_last > 0) else np.nan

    iv_eps = intrinsic_eps_model(last_eps, growth_eps, years_eps, terminal_pe, discount)
    iv_dcf = intrinsic_dcf_fcf(fcf_per_share_last, growth_fcf, years_dcf, terminal_g, discount)

    if using_av:
        current_price = get_price_alpha_vantage(ticker, AV_KEY) if key_ok else np.nan
    else:
        current_price = get_price_fmp(ticker, FMP_KEY) if key_ok else np.nan

    colv1, colv2, colv3, colv4 = st.columns(4)
    colv1.metric("Intrinsic (EPS / P‑E)", f"${iv_eps:,.2f}" if not pd.isna(iv_eps) else "—")
    colv2.metric("Intrinsic (FCF DCF / sh.)", f"${iv_dcf:,.2f}" if not pd.isna(iv_dcf) else "—")
    colv3.metric("Current Price", f"${current_price:,.2f}" if not pd.isna(current_price) else "—")
    colv4.metric("Terminal P/E used", f"{terminal_pe:.1f}")

    if not pd.isna(current_price):
        if not pd.isna(iv_eps):
            mos_eps = (iv_eps - current_price) / current_price
            st.write(f"**Margin of Safety (EPS model):** {pct(mos_eps)}")
        if not pd.isna(iv_dcf):
            mos_dcf = (iv_dcf - current_price) / current_price
            st.write(f"**Margin of Safety (FCF DCF):** {pct(mos_dcf)}")

    with st.expander("Valuation Inputs Recap"):
        rec = {
            "EPS last": last_eps,
            "FCF last": fcf_last,
            "Shares diluted last": shares_last,
            "FCF per share last": fcf_per_share_last,
            "EPS growth": growth_eps, "Years (EPS)": years_eps, "Terminal P/E": terminal_pe,
            "FCF growth": growth_fcf, "Years (DCF)": years_dcf, "Terminal g": terminal_g,
            "Discount rate": discount
        }
        st.json({k: (None if pd.isna(v) else (float(v) if isinstance(v, (int, float, np.floating)) else v)) for k, v in rec.items()})

    # -------------------- Value-Investor Summary (OpenAI) --------------------
    st.markdown("### Value-Investor Summary (OpenAI)")
    if not OPENAI_KEY:
        st.info("Add **OPENAI_API_KEY** in Secrets to enable the summary.")
    else:
        want = st.button("Generate Summary with OpenAI")
        if want:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=OPENAI_KEY)
                context = {
                    "ticker": ticker,
                    "years": list(map(int, df.index.tolist())),
                    "big5_10y": {
                        "sales_cagr": float(sales_cagr_10) if not pd.isna(sales_cagr_10) else None,
                        "eps_cagr": float(eps_cagr_10) if not pd.isna(eps_cagr_10) else None,
                        "equity_cagr": float(eqty_cagr_10) if not pd.isna(eqty_cagr_10) else None,
                        "fcf_cagr": float(fcf_cagr_10) if not pd.isna(fcf_cagr_10) else None,
                        "roic_avg": float(roic_avg_10) if not pd.isna(roic_avg_10) else None
                    },
                    "breakdowns": {
                        "Sales": sales_b, "EPS": eps_b, "Equity": eqty_b, "FCF": fcf_b, "ROIC": roic_b
                    },
                    "intrinsic": {
                        "iv_eps": float(iv_eps) if not pd.isna(iv_eps) else None,
                        "iv_dcf": float(iv_dcf) if not pd.isna(iv_dcf) else None,
                        "current_price": float(current_price) if not pd.isna(current_price) else None
                    }
                }
                prompt = (
                    "You are a disciplined value investor (Phil Town style). "
                    "Using the structured data below, evaluate whether the business clears the 10% rule on Sales, EPS, Equity, FCF CAGRs and 10-yr Avg ROIC. "
                    "Comment on the 10/5/3/1 breakdowns (e.g., improving or deteriorating). "
                    "Interpret intrinsic values vs current price and note any risks (declining ROIC, leverage, inconsistent FCF). "
                    "Keep it under 180 words, neutral tone.\n\n"
                    f"{context}"
                )
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role":"user","content": prompt}],
                    temperature=0.3,
                    max_tokens=350,
                )
                st.write(resp.choices[0].message.content)
            except Exception as e:
                st.error(f"OpenAI error: {e}")

    with st.expander("Raw series used"):
        st.dataframe(df)

    with st.expander("Mini charts"):
        c1, c2, c3 = st.columns(3)
        c1.line_chart(df[["Revenue","FCF"]].dropna(), height=220)
        c2.line_chart(df[["EPS"]].dropna(), height=220)
        c3.line_chart(df[["ROIC"]].dropna(), height=220)

else:
    st.info("Enter a ticker and click **Search**. Choose provider in the sidebar and confirm your API key in Secrets.")
