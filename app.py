import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from typing import Tuple, Dict, List

# -------------------- Page setup --------------------
st.set_page_config(page_title="Phil Town Big 5 Screener", page_icon="📈", layout="wide")

# Header
st.markdown("""
<h1 style="margin-bottom:0">📈 Phil Town Big 5 Screener</h1>
<p style="color:#666;margin-top:4px">
Check 10-year CAGRs for <b>Sales, EPS, Equity, FCF</b> and the <b>10-yr Avg ROIC</b> against the 10% rule.
</p>
""", unsafe_allow_html=True)

# -------------------- Sidebar --------------------
st.sidebar.header("Settings")
provider = st.sidebar.selectbox("Data provider", ["Alpha Vantage", "Financial Modeling Prep (FMP)"])
st.sidebar.caption("Tip: FMP is often the smoothest on Streamlit Cloud.")

# Discover API keys from Streamlit secrets
AV_KEY = st.secrets.get("ALPHAVANTAGE_API_KEY", "").strip()
FMP_KEY = st.secrets.get("FMP_API_KEY", "").strip()

if provider == "Alpha Vantage":
    if not AV_KEY:
        st.sidebar.error("Missing Alpha Vantage key.\nAdd in: ⋯ → Manage app → Settings → Secrets\n\nALPHAVANTAGE_API_KEY = your_key")
else:
    if not FMP_KEY:
        st.sidebar.error("Missing FMP key.\nAdd in: ⋯ → Manage app → Settings → Secrets\n\nFMP_API_KEY = your_key")

st.sidebar.markdown("---")
st.sidebar.write("Examples: `AAPL`, `MSFT`, `ADBE`, `AMZN`, `NVDA`")

# -------------------- Input form with dedicated Search button --------------------
with st.form(key="search_form"):
    c1, c2 = st.columns([2,1])
    with c1:
        ticker = st.text_input("Ticker", value="MSFT").strip().upper()
    with c2:
        run = st.form_submit_button("🔎 Search")

# -------------------- Helpers --------------------
def cagr(series: pd.Series, years: int) -> float:
    try:
        if years <= 0 or series.empty:
            return np.nan
        first, last = series.iloc[0], series.iloc[-1]
        if pd.isna(first) or pd.isna(last) or first <= 0 or last <= 0:
            return np.nan
        return (last / first) ** (1 / years) - 1
    except Exception:
        return np.nan

def pct(x: float) -> str:
    return "—" if (x is None or pd.isna(x)) else f"{x*100:.1f}%"

# -------------------- Data layer: Alpha Vantage --------------------
AV_BASE = "https://www.alphavantage.co/query"

def av_get(fn: str, symbol: str, apikey: str):
    params = {"function": fn, "symbol": symbol, "apikey": apikey}
    r = requests.get(AV_BASE, params=params, timeout=30)
    r.raise_for_status()
    j = r.json()
    if "annualReports" not in j:
        return []
    return j["annualReports"]  # newest first

def av_series(reports: List[Dict], field: str) -> pd.Series:
    if not reports:
        return pd.Series(dtype="float64")
    rows = []
    for rep in reports:
        y = pd.to_datetime(rep.get("fiscalDateEnding", ""), errors="coerce").year
        if pd.isna(y):
            continue
        val = pd.to_numeric(rep.get(field, None), errors="coerce")
        rows.append((int(y), val))
    if not rows:
        return pd.Series(dtype="float64")
    s = pd.Series(dict(rows)).sort_index()
    return s.iloc[-11:].astype("float64")

@st.cache_data(show_spinner=False, ttl=1800)
def fetch_alpha_vantage(symbol: str, apikey: str) -> Tuple[pd.DataFrame, List[int], str]:
    # Respect 5 req/min (3 calls → small spacing)
    inc = av_get("INCOME_STATEMENT", symbol, apikey); time.sleep(0.6)
    bal = av_get("BALANCE_SHEET",  symbol, apikey); time.sleep(0.6)
    cfs = av_get("CASH_FLOW",      symbol, apikey); time.sleep(0.6)

    revenue        = av_series(inc, "totalRevenue")
    net_income     = av_series(inc, "netIncome")
    diluted_eps    = av_series(inc, "dilutedEPS")
    diluted_shares = av_series(inc, "weightedAverageShsOutDil")
    ebit           = av_series(inc, "ebit")
    tax_expense    = av_series(inc, "incomeTaxExpense")
    pretax_income  = av_series(inc, "incomeBeforeTax")

    equity     = av_series(bal, "totalShareholderEquity")
    total_debt = av_series(bal, "totalDebt")
    if total_debt.empty:
        total_debt = av_series(bal, "shortLongTermDebtTotal")
    cash       = av_series(bal, "cashAndCashEquivalentsAtCarryingValue")
    if cash.empty:
        cash = av_series(bal, "cashAndShortTermInvestments")

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

    # EPS fallback if AV didn't provide it
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
    return df, years, "Alpha Vantage"

# -------------------- Data layer: FMP --------------------
FMP_BASE = "https://financialmodelingprep.com/api/v3"

def fmp_get(path: str, apikey: str, params=None):
    if params is None: params = {}
    params["apikey"] = apikey
    r = requests.get(f"{FMP_BASE}/{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def fmp_series(reports: List[Dict], field: str) -> pd.Series:
    if not reports: return pd.Series(dtype="float64")
    rows = []
    for rep in reports:
        # FMP has date, fillingDate, calendarYear. Prefer date/calendarYear.
        y = pd.to_datetime(rep.get("date") or rep.get("fillingDate") or rep.get("calendarYear"), errors="coerce").year
        if pd.isna(y):
            try: y = int(rep.get("calendarYear"))
            except: continue
        val = pd.to_numeric(rep.get(field, None), errors="coerce")
        rows.append((int(y), val))
    if not rows: return pd.Series(dtype="float64")
    s = pd.Series(dict(rows)).sort_index()
    return s.iloc[-11:].astype("float64")

@st.cache_data(show_spinner=False, ttl=1800)
def fetch_fmp(symbol: str, apikey: str) -> Tuple[pd.DataFrame, List[int], str]:
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

# -------------------- Run search --------------------
if run:
    if provider == "Alpha Vantage" and not AV_KEY:
        st.error("Please add ALPHAVANTAGE_API_KEY in Streamlit Secrets (⋯ → Manage app → Settings → Secrets).")
    elif provider != "Alpha Vantage" and not FMP_KEY:
        st.error("Please add FMP_API_KEY in Streamlit Secrets (⋯ → Manage app → Settings → Secrets).")
    else:
        try:
            with st.spinner(f"Fetching {ticker} fundamentals from {provider}…"):
                if provider == "Alpha Vantage":
                    df, years, source = fetch_alpha_vantage(ticker, AV_KEY)
                else:
                    df, years, source = fetch_fmp(ticker, FMP_KEY)

            if df.empty:
                st.warning("No data returned. Try a major US ticker (AAPL/MSFT). If you just added your API key, wait a minute and try again.")
            else:
                n_years = max(len(years) - 1, 0)

                # KPI cards
                colA, colB, colC, colD, colE = st.columns(5)
                sales_cagr = cagr(df["Revenue"].dropna(), n_years) if "Revenue" in df else np.nan
                eps_cagr   = cagr(df["EPS"].dropna(),     n_years) if "EPS"     in df else np.nan
                eqty_cagr  = cagr(df["Equity"].dropna(),  n_years) if "Equity"  in df else np.nan
                fcf_cagr   = cagr(df["FCF"].dropna(),     n_years) if "FCF"     in df else np.nan
                roic_avg   = df["ROIC"].replace([np.inf, -np.inf], np.nan).dropna().mean() if "ROIC" in df else np.nan

                colA.metric("Sales CAGR", pct(sales_cagr))
                colB.metric("EPS CAGR",   pct(eps_cagr))
                colC.metric("Equity CAGR",pct(eqty_cagr))
                colD.metric("FCF CAGR",   pct(fcf_cagr))
                colE.metric("Avg ROIC",   pct(roic_avg))

                # Pass/Fail table
                def pf(v): return "PASS ✅" if (not pd.isna(v) and v >= 0.10) else ("—" if pd.isna(v) else "FAIL ❌")
                results = pd.DataFrame({
                    "Metric": ["Sales (Revenue) CAGR","EPS CAGR","Equity CAGR","FCF CAGR","ROIC (10-yr Avg)"],
                    "Value":  [pct(sales_cagr), pct(eps_cagr), pct(eqty_cagr), pct(fcf_cagr), pct(roic_avg)],
                    "Pass ≥10%?": [pf(sales_cagr), pf(eps_cagr), pf(eqty_cagr), pf(fcf_cagr), pf(roic_avg)]
                })

                st.markdown(f"##### Results — {ticker}  ·  Source: {source}")
                st.dataframe(results, use_container_width=True)

                with st.expander("Show raw series used"):
                    st.dataframe(df)

                with st.expander("Mini charts"):
                    cc1, cc2, cc3 = st.columns(3)
                    with cc1:
                        st.line_chart(df[["Revenue","FCF"]].dropna(), height=220)
                    with cc2:
                        st.line_chart(df[["EPS"]].dropna(), height=220)
                    with cc3:
                        st.line_chart(df[["ROIC"]].dropna(), height=220)

                st.caption("Notes: EPS prefers reported diluted EPS; else NI ÷ diluted shares. FCF = CFO − CapEx. ROIC proxy = NOPAT / (Debt + Equity − Cash).")

        except requests.HTTPError as http_err:
            st.error(f"HTTP error from {provider}: {http_err}")
        except Exception as e:
            st.exception(e)
else:
    st.info("Enter a ticker and click **Search**. Use the sidebar to choose the data provider and confirm your API key is set in Secrets.")
