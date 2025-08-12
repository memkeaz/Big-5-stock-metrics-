import streamlit as st
import pandas as pd
import numpy as np
import requests

# -------------------- Page Config --------------------
st.set_page_config(page_title="Phil Town Big 5 Screener", layout="wide")
st.title("Phil Town Big 5 — 10-Year Screener")
st.caption("Checks 10-year CAGRs for Sales, EPS, Equity, FCF, and the 10-yr Avg ROIC (NOPAT / Invested Capital). Pass rule: ≥ 10%.")

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

# -------------------- Demo Mode --------------------
def demo_msft_df():
    years = list(range(2015, 2025))
    df = pd.DataFrame({
        "Revenue": [93580, 85320, 89950, 110360, 125843, 143015, 168088, 198270, 211915, 245000],
        "EPS":     [2.48, 2.79, 3.31, 2.13, 5.76, 6.20, 8.05, 9.21, 9.68, 11.00],
        "Equity":  [72163, 82718, 82572, 82572, 118304, 118304, 166542, 166542, 194000, 210000],
        "FCF":     [23969, 31378, 31922, 32694, 45230, 45300, 56300, 65700, 67800, 78000],
        "ROIC":    [0.12, 0.13, 0.15, 0.10, 0.18, 0.19, 0.21, 0.22, 0.20, 0.21],
    }, index=years).astype(float)
    return df.iloc[-10:], years[-10:]

# -------------------- Helpers --------------------
def cagr(series: pd.Series, years: int) -> float:
    try:
        if years <= 0 or series.empty: return np.nan
        first, last = series.iloc[0], series.iloc[-1]
        if pd.isna(first) or pd.isna(last) or first <= 0 or last <= 0: return np.nan
        return (last / first) ** (1 / years) - 1
    except Exception:
        return np.nan

def pct(x: float) -> str:
    return "—" if (x is None or pd.isna(x)) else f"{x*100:.1f}%"

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

    df = pd.DataFrame({"Revenue": revenue, "EPS": eps, "Equity": equity, "FCF": fcf, "ROIC": roic}).sort_index().tail(11)
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

    df = pd.DataFrame({"Revenue": revenue, "EPS": eps, "Equity": equity, "FCF": fcf, "ROIC": roic}).sort_index().tail(11)
    return df, years, "FMP"

# -------------------- Run Search --------------------
if run:
    using_av = (provider == "Alpha Vantage")
    key_ok = (AV_KEY if using_av else FMP_KEY)
    st.info(f"Provider: **{provider}** · API key set: **{'Yes' if key_ok else 'No'}** · Ticker: **{ticker}**")

    if not key_ok:
        st.error("Missing API key in Streamlit Secrets.")
        df, years = demo_msft_df()
        source = "Demo (sample)"
    else:
        try:
            if using_av:
                df, years, source = fetch_alpha_vantage(ticker, AV_KEY)
            else:
                df, years, source = fetch_fmp(ticker, FMP_KEY)
            if df.empty:
                st.warning("No data returned — switching to Demo Mode.")
                df, years = demo_msft_df()
                source = "Demo (sample)"
        except Exception as e:
            st.error(f"Error: {e}")
            df, years = demo_msft_df()
            source = "Demo (sample)"

    # -------------------- Big 5 Metrics --------------------
    n_years = max(len(df.index) - 1, 0)
    sales_cagr = cagr(df["Revenue"], n_years)
    eps_cagr   = cagr(df["EPS"], n_years)
    eqty_cagr  = cagr(df["Equity"], n_years)
    fcf_cagr   = cagr(df["FCF"], n_years)
    roic_avg   = df["ROIC"].replace([np.inf, -np.inf], np.nan).dropna().mean()

    def pf(v): return "PASS ✅" if not pd.isna(v) and v >= 0.10 else ("—" if pd.isna(v) else "FAIL ❌")
    def fmt(x): return "—" if pd.isna(x) else f"{x*100:.1f}%"

    # KPI cards
    colA, colB, colC, colD, colE = st.columns(5)
    colA.metric("Sales CAGR", fmt(sales_cagr))
    colB.metric("EPS CAGR",   fmt(eps_cagr))
    colC.metric("Equity CAGR",fmt(eqty_cagr))
    colD.metric("FCF CAGR",   fmt(fcf_cagr))
    colE.metric("Avg ROIC",   fmt(roic_avg))

    results = pd.DataFrame({
        "Metric": ["Sales (Revenue) CAGR","EPS CAGR","Equity CAGR","FCF CAGR","ROIC (10-yr Avg)"],
        "Value":  [fmt(sales_cagr), fmt(eps_cagr), fmt(eqty_cagr), fmt(fcf_cagr), fmt(roic_avg)],
        "Pass ≥10%?": [pf(sales_cagr), pf(eps_cagr), pf(eqty_cagr), pf(fcf_cagr), pf(roic_avg)]
    })

    st.subheader(f"Big 5 Results — {ticker}  ·  Source: {source}")
    st.dataframe(results, use_container_width=True)

    # ===== ROIC Breakdown =====
    st.markdown("#### ROIC Breakdown (from the 10-year window)")
    roic_series = df.get("ROIC", pd.Series(dtype=float)).astype(float).replace([np.inf, -np.inf], np.nan)
    roic_valid = roic_series.dropna()

    roic_10yr_avg = roic_valid.mean() if len(roic_valid) >= 1 else np.nan
    roic_first5   = roic_valid.iloc[:5].mean() if len(roic_valid) >= 5 else np.nan
    roic_last3    = roic_valid.iloc[-3:].mean() if len(roic_valid) >= 1 else np.nan
    roic_last1    = roic_valid.iloc[-1] if len(roic_valid) >= 1 else np.nan

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROIC — 10-yr Avg", fmt(roic_10yr_avg))
    c2.metric("ROIC — First 5-yr Avg", fmt(roic_first5))
    c3.metric("ROIC — Last 3-yr Avg", fmt(roic_last3))
    c4.metric("ROIC — Last 1-yr", fmt(roic_last1))
    st.caption("ROIC = NOPAT / (Debt + Equity − Cash). First 5-yr uses the oldest five years in the displayed 10-year window.")

    with st.expander("Raw series used"):
        st.dataframe(df)

    with st.expander("Mini charts"):
        c1, c2, c3 = st.columns(3)
        c1.line_chart(df[["Revenue","FCF"]].dropna(), height=220)
        c2.line_chart(df[["EPS"]].dropna(), height=220)
        c3.line_chart(df[["ROIC"]].dropna(), height=220)

else:
    st.info("Enter a ticker and click **Search**. Choose provider in the sidebar and confirm your API key in Secrets.")
