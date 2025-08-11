import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import time
import re
from typing import Iterable, Tuple

st.set_page_config(page_title="Phil Town Big 5 Screener", layout="wide")
st.title("Phil Town Big 5 — 10-Year Screener")
st.caption("Fetches ~10 years of fundamentals via Yahoo Finance and checks Sales, EPS, Equity, FCF CAGRs ≥ 10%, and 10-yr Avg ROIC ≥ 10%.")

# ---------- Helpers ----------
def _to_year_list(cols: Iterable):
    years = []
    for c in cols:
        y = None
        try:
            if hasattr(c, "year"):              # pandas Period/Timestamp
                y = int(c.year)
            elif isinstance(c, str):            # '2024-09-30' or '2024'
                m = re.search(r"(\d{4})", c)
                if m:
                    y = int(m.group(1))
            elif isinstance(c, (int, np.integer)) and 1900 <= int(c) <= 2100:
                y = int(c)
        except Exception:
            pass
        years.append(y)
    return years

def _tidy(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    years = _to_year_list(df.columns)
    df.columns = years
    keep_idx = [i for i, y in enumerate(years) if isinstance(y, int)]
    if not keep_idx:
        return pd.DataFrame()
    df = df.iloc[:, keep_idx]
    df = df.reindex(sorted(df.columns), axis=1).iloc[:, -11:]     # most recent ~10 intervals
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.loc[:, ~df.columns.duplicated(keep="last")]
    return df

def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

def pick(df: pd.DataFrame, names) -> Tuple[pd.Series, str]:
    """Pick a row by name (exact-insensitive then substring). Returns (series, matched_label_or_None)."""
    if df is None or df.empty:
        return pd.Series(dtype="float64"), None
    idx_map = {_normalize(i): i for i in df.index}
    for n in names:
        key = _normalize(n)
        if key in idx_map:
            return df.loc[idx_map[key]], idx_map[key]
    # substring fallback
    for idx in df.index:
        s = _normalize(idx)
        for n in names:
            if _normalize(n) in s:
                return df.loc[idx], idx
    return pd.Series(dtype="float64"), None

# ---------- Data fetch ----------
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_statements(ticker: str):
    tk = yf.Ticker(ticker)
    inc = bal = cfs = pd.DataFrame()

    # Try modern API with light retry (handles occasional cold starts)
    for _ in range(2):
        try:
            inc = tk.get_income_stmt(freq="annual")
            bal = tk.get_balance_sheet(freq="annual")
            cfs = tk.get_cashflow(freq="annual")
            break
        except Exception:
            time.sleep(1)

    # Fallback to legacy attributes if needed
    if inc is None or inc.empty:
        try: inc = tk.financials
        except Exception: inc = pd.DataFrame()
    if bal is None or bal.empty:
        try: bal = tk.balance_sheet
        except Exception: bal = pd.DataFrame()
    if cfs is None or cfs.empty:
        try: cfs = tk.cashflow
        except Exception: cfs = pd.DataFrame()

    return _tidy(inc), _tidy(bal), _tidy(cfs)

@st.cache_data(show_spinner=False, ttl=3600)
def compute_series(ticker: str):
    inc, bal, cfs = fetch_statements(ticker)

    # Broad label sets (cover common variations & financials)
    revenue_names        = ["Total Revenue","Revenue","Total Operating Revenues","Operating Revenue","Revenues","Total Net Revenue"]
    net_income_names     = ["Net Income","NetIncome","Net Income Applicable To Common Shares","Net Income Common Stockholders"]
    diluted_shares_names = ["Diluted Average Shares","Weighted Average Diluted Shares Outstanding","Diluted Shares","Diluted Average Shares Outstanding","Weighted Average Shares Diluted"]
    diluted_eps_names    = ["Diluted EPS","DilutedEPS","EPS Diluted"]
    ebit_names           = ["EBIT","Operating Income","Earnings Before Interest and Taxes","Operating Profit"]

    equity_names     = ["Total Stockholder Equity","Total Stockholders' Equity","TotalStockholderEquity","Shareholders' Equity","Total Equity","Total Shareholder Equity","Common Stock Equity","Total Common Equity"]
    total_debt_names = ["Total Debt","Short Long Term Debt","ShortLongTermDebtTotal","Total Liabilities & Debt","Total Interest-Bearing Debt"]
    cash_names       = ["Cash And Cash Equivalents","Cash And Cash Equivalents At Carrying Value","Cash","CashAndCashEquivalents"]

    cfo_names        = ["Operating Cash Flow","Total Cash From Operating Activities","NetCashProvidedByUsedInOperatingActivities"]
    capex_names      = ["Capital Expenditure","CapitalExpenditures","Investments In Property Plant And Equipment","Purchase Of Property Plant And Equipment"]
    tax_expense_names   = ["Income Tax Expense","IncomeTaxExpense","Provision For Income Taxes"]
    pretax_income_names = ["Income Before Tax","IncomeBeforeTax","Earnings Before Tax","Pretax Income"]

    # Extract rows
    revenue, rev_row           = pick(inc, revenue_names)
    net_income, ni_row         = pick(inc, net_income_names)
    diluted_shares, sh_row     = pick(inc, diluted_shares_names)
    diluted_eps, eps_row       = pick(inc, diluted_eps_names)
    ebit, ebit_row             = pick(inc, ebit_names)

    equity, eq_row             = pick(bal, equity_names)
    total_debt, debt_row       = pick(bal, total_debt_names)
    cash, cash_row             = pick(bal, cash_names)

    cfo, cfo_row               = pick(cfs, cfo_names)
    capex, capex_row           = pick(cfs, capex_names)

    years = sorted(set(revenue.index) | set(net_income.index) | set(diluted_shares.index) |
                   set(diluted_eps.index) | set(ebit.index) | set(equity.index) |
                   set(total_debt.index) | set(cash.index) | set(cfo.index) | set(capex.index))[-11:]

    def align(s):
        return s.reindex(years).astype("float64") if len(years) else pd.Series(dtype="float64")

    revenue, net_income, diluted_shares, diluted_eps, ebit, equity, total_debt, cash, cfo, capex = [
        align(x) for x in [revenue, net_income, diluted_shares, diluted_eps, ebit, equity, total_debt, cash, cfo, capex]
    ]

    # EPS: prefer reported; else NI / diluted shares
    eps = diluted_eps.copy()
    if eps.isna().all() and not net_income.isna().all() and not diluted_shares.isna().all():
        with np.errstate(invalid="ignore", divide="ignore"):
            eps = net_income / diluted_shares.replace({0: np.nan})

    # FCF = CFO − CapEx
    fcf = (cfo - capex) if (not cfo.isna().all() and not capex.isna().all()) else pd.Series([np.nan]*len(years), index=years)

    # ROIC proxy: NOPAT / (Debt + Equity − Cash)
    tax_expense, _  = pick(inc, tax_expense_names);  tax_expense  = align(tax_expense)
    pretax_income, _= pick(inc, pretax_income_names);pretax_income= align(pretax_income)
    with np.errstate(divide="ignore", invalid="ignore"):
        tax_rate = (tax_expense / pretax_income).clip(0, 1)

    if ebit.isna().all():
        ebit, _ = pick(inc, ["Operating Income","OperatingIncome"]); ebit = align(ebit)

    invested_capital = (total_debt.fillna(0) + equity.fillna(0) - cash.fillna(0)).replace({0: np.nan})
    nopat = (ebit * (1 - tax_rate.fillna(0.21))) if not ebit.isna().all() else net_income
    with np.errstate(divide="ignore", invalid="ignore"):
        roic_series = (nopat / invested_capital).replace([np.inf, -np.inf], np.nan)

    df = pd.DataFrame({
        "Revenue": revenue, "EPS": eps, "Equity": equity, "FCF": fcf, "ROIC": roic_series
    })

    # Diagnostics
    found = {
        "Revenue row": rev_row, "Net income row": ni_row, "Diluted shares row": sh_row, "Diluted EPS row": eps_row, "EBIT row": ebit_row,
        "Equity row": eq_row, "Total debt row": debt_row, "Cash row": cash_row, "CFO row": cfo_row, "CapEx row": capex_row
    }

    return df, years, inc, bal, cfs, found

def CAGR(series: pd.Series, years_count: int):
    try:
        first, last = series.iloc[0], series.iloc[-1]
        if pd.isna(first) or pd.isna(last) or first <= 0 or last <= 0 or years_count <= 0:
            return np.nan
        return (last/first)**(1/years_count) - 1
    except Exception:
        return np.nan

def pct(x): return "—" if (x is None or pd.isna(x)) else f"{x*100:.1f}%"

# ---------- UI ----------
ticker = st.text_input("Enter ticker (e.g., AAPL, MSFT, ADBE):", value="AAPL").strip().upper()

try:
    df, years, inc, bal, cfs, found = compute_series(ticker)
    n_years = max(len(years) - 1, 0)

    col1, col2 = st.columns([1,2])
    with col1:
        st.subheader("Years used")
        st.write(years if years else "—")
        st.caption(f"Span: {n_years} year(s) of growth.")

    # Metrics
    sales_cagr = CAGR(df["Revenue"].dropna(), n_years) if "Revenue" in df else np.nan
    eps_cagr   = CAGR(df["EPS"].dropna(),     n_years) if "EPS"     in df else np.nan
    eqty_cagr  = CAGR(df["Equity"].dropna(),  n_years) if "Equity"  in df else np.nan
    fcf_cagr   = CAGR(df["FCF"].dropna(),     n_years) if "FCF"     in df else np.nan
    roic_avg   = df["ROIC"].replace([np.inf, -np.inf], np.nan).dropna().mean() if "ROIC" in df else np.nan

    def passfail(v): return "—" if pd.isna(v) else ("PASS ✅" if v >= 0.10 else "FAIL ❌")

    results = pd.DataFrame({
        "Metric": ["Sales (Revenue) CAGR","EPS CAGR","Equity CAGR","FCF CAGR","ROIC (10-yr Avg)"],
        "Value":  [pct(sales_cagr), pct(eps_cagr), pct(eqty_cagr), pct(fcf_cagr), pct(roic_avg)],
        "Pass ≥10%?": [passfail(sales_cagr), passfail(eps_cagr), passfail(eqty_cagr), passfail(fcf_cagr), passfail(roic_avg)]
    })

    with col2:
        st.subheader(f"Big 5 Results — {ticker}")
        st.dataframe(results, use_container_width=True)

    # Raw statements + quick row label debug
    with st.expander("Raw statements pulled"):
        st.write("Income Statement (annual):"); st.dataframe(inc)
        st.write("Balance Sheet (annual):");   st.dataframe(bal)
        st.write("Cash Flow (annual):");       st.dataframe(cfs)

    with st.expander("Debug: first 20 row labels from each statement"):
        st.write("Income rows:", [str(i) for i in list(inc.index)[:20]])
        st.write("Balance rows:", [str(i) for i in list(bal.index)[:20]])
        st.write("Cashflow rows:", [str(i) for i in list(cfs.index)[:20]])

    with st.expander("Which rows were found?"):
        st.json(found)

    with st.expander("Final aligned series used in math"):
        st.dataframe(df)

    st.caption("If a series is empty, check which row names were found above. Different industries (esp. banks/insurers) use different labels.")

except Exception as e:
    st.exception(e)
    st.stop()
