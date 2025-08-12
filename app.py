# app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests

# =================== Page & Style ===================
st.set_page_config(page_title="Phil Town Big 5 Screener", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    """
    <style>
      .block-container { padding-top: 1rem; padding-bottom: 3rem; max-width: 1100px; }
      @media (max-width: 640px) {
        .block-container { padding-left: 0.8rem; padding-right: 0.8rem; }
      }
      .fine { color: #777; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True
)

# =================== Header ===================
st.title("Phil Town Big 5 Screener")
st.caption("Big 5 (Sales, EPS, Equity, FCF CAGRs + 10-yr Avg ROIC), 10/5/3/1 breakdowns, and two valuations: Rule #1 EPS & FCF-DCF. One MOS slider applies to both valuations.")

# =================== Secrets ===================
AV_KEY  = st.secrets.get("ALPHAVANTAGE_API_KEY", "").strip()
FMP_KEY = st.secrets.get("FMP_API_KEY", "").strip()

# =================== Session State ===================
if "data" not in st.session_state:
    st.session_state.data = None  # (df, years, source, diag, current_price, provider_label)
if "last_query" not in st.session_state:
    st.session_state.last_query = {"ticker": None, "provider": None}

# =================== Controls (fetch only on Search) ===================
with st.form("search"):
    c1, c2 = st.columns([2, 1])
    with c1:
        ticker = st.text_input("Ticker", value=st.session_state.last_query.get("ticker") or "ADBE").strip().upper()
    with c2:
        provider = st.selectbox(
            "Data source",
            ["Auto (FMP→Yahoo→AV)", "Yahoo Finance", "FMP", "Alpha Vantage"],
            index=0 if st.session_state.last_query.get("provider") is None else
                ["Auto (FMP→Yahoo→AV)", "Yahoo Finance", "FMP", "Alpha Vantage"]
                .index(st.session_state.last_query.get("provider", "Auto (FMP→Yahoo→AV)"))
        )

    with st.expander("Valuation Settings", expanded=False):
        st.markdown("**Discount / MARR & Growth Assumptions**")
        discount   = st.number_input("MARR / Discount rate (%, both models)", 4.0, 20.0, 10.0, step=0.5, key="discount") / 100.0

        st.markdown("**Rule #1 EPS (Phil Town)**")
        years_eps        = st.slider("Years (EPS projection)", 5, 15, 10, key="years_eps")
        growth_eps_user  = st.number_input("Your EPS growth estimate (annual, %)", 0.0, 50.0, 12.0, step=0.5, key="growth_eps_user") / 100.0
        auto_pe          = st.checkbox("Terminal P/E ≈ lower of (2× growth, current P/E), capped at 50", value=True, key="auto_pe")
        terminal_pe_man  = st.number_input("Terminal P/E (manual if Auto off)", 5.0, 60.0, 20.0, step=0.5, key="terminal_pe_man")

        st.markdown("---")
        st.markdown("**FCF DCF**")
        years_dcf  = st.slider("Years (DCF)", 5, 15, 10, key="years_dcf")
        growth_fcf = st.number_input("FCF growth (annual, %)", 0.0, 50.0, 10.0, step=0.5, key="growth_fcf") / 100.0
        terminal_g = st.number_input("Terminal growth (FCF, %)", 0.0, 6.0, 3.0, step=0.25, key="terminal_g") / 100.0

        st.markdown("---")
        st.markdown("**Margin of Safety (applies to BOTH valuations)**")
        mos_pct = st.slider("MOS % of fair value (1–100%)", 1, 100, 50, step=1, key="mos_pct")

    go = st.form_submit_button("Search")

# =================== Helpers ===================
def cagr_over_years(first_val, last_val, first_year, last_year):
    try:
        years = last_year - first_year
        if years <= 0 or first_val <= 0 or last_val <= 0 or pd.isna(first_val) or pd.isna(last_val):
            return np.nan
        return (last_val / first_val) ** (1 / years) - 1
    except Exception:
        return np.nan

def series_cagr_gap(s):
    y = s.dropna()
    if len(y) < 2: return np.nan
    return cagr_over_years(y.iloc[0], y.iloc[-1], int(y.index[0]), int(y.index[-1]))

def yoy(series):
    y = series.dropna()
    if len(y) < 2: return np.nan
    prev, last = y.iloc[-2], y.iloc[-1]
    if prev <= 0 or pd.isna(prev) or pd.isna(last): return np.nan
    return (last / prev) - 1

def safe_mean(s):
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    return s.mean() if len(s) else np.nan

def pct(x): return "—" if pd.isna(x) else f"{x*100:.1f}%"

def normalize_capex(capex):
    if capex is None or capex.empty: return capex
    return capex.apply(lambda v: abs(v) if not pd.isna(v) else v)

def roic_series_from(nopat, debt, equity, cash):
    ic = (debt.fillna(0) + equity.fillna(0) - cash.fillna(0))
    ic = ic.replace({0: np.nan})
    ic_avg = (ic + ic.shift(1)) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        roic = (nopat / ic_avg).replace([np.inf, -np.inf], np.nan)
    return roic

def diag_counts(df): return {k:int(v) for k,v in df.notna().sum().to_dict().items()}

def latest_positive(series, lookback=None):
    """Return the most recent positive non-NaN value (optionally within last N points)."""
    s = series.dropna()
    if lookback is not None:
        s = s.iloc[-lookback:]
    s = s[s > 0]
    if s.empty: return np.nan
    return s.iloc[-1]

# =================== Providers ===================
AV_BASE  = "https://www.alphavantage.co/query"
FMP_BASE = "https://financialmodelingprep.com/api/v3"

# ---- Yahoo (yfinance) ----
@st.cache_data(show_spinner=False, ttl=1800)
def fetch_yahoo(symbol):
    try:
        import yfinance as yf
    except Exception as e:
        raise RuntimeError("Missing yfinance. Add to requirements.txt") from e

    tk = yf.Ticker(symbol)

    def tidy(df):
        if df is None or df.empty: return pd.DataFrame()
        df = df.copy()
        years = [int(pd.to_datetime(c).year) for c in df.columns]
        df.columns = years
        df = df.reindex(sorted(df.columns), axis=1).iloc[:, -11:]
        return df.apply(pd.to_numeric, errors="coerce")

    inc = tidy(tk.get_income_stmt(freq="annual") if hasattr(tk, "get_income_stmt") else tk.financials)
    bal = tidy(tk.get_balance_sheet(freq="annual") if hasattr(tk, "get_balance_sheet") else tk.balance_sheet)
    cfs = tidy(tk.get_cashflow(freq="annual") if hasattr(tk, "get_cashflow") else tk.cashflow)

    def pick(df, names):
        if df is None or df.empty: return pd.Series(dtype="float64")
        for n in names:
            if n in df.index: return df.loc[n]
        idx = {i.lower().replace(" ", ""): i for i in df.index}
        for n in names:
            key = n.lower().replace(" ", "")
            for k, orig in idx.items():
                if key in k: return df.loc[orig]
        return pd.Series(dtype="float64")

    revenue     = pick(inc, ["Total Revenue","Revenue","TotalOperatingRevenues"])
    net_income  = pick(inc, ["Net Income","NetIncome","Net Income Applicable To Common Shares"])
    diluted_eps = pick(inc, ["Diluted EPS","DilutedEPS","EPS Diluted"])
    ebit        = pick(inc, ["EBIT","Operating Income","Earnings Before Interest and Taxes","Operating Income"])

    tax_expense = pick(inc, ["Income Tax Expense","Provision For Income Taxes"])
    pretax      = pick(inc, ["Income Before Tax","Earnings Before Tax","Pretax Income"])
    shares      = pick(inc, ["Weighted Average Shares Diluted","Weighted Average Diluted Shares Outstanding","Diluted Average Shares"])
    equity      = pick(bal, ["Total Stockholder Equity","TotalStockholderEquity","Common Stock Equity"])
    total_debt  = pick(bal, ["Total Debt","Short Long Term Debt","ShortLongTermDebtTotal"])
    if total_debt.isna().all():
        total_debt = pick(bal, ["Current Debt","Long Term Debt","LongTermDebt","ShortTermDebt"]).fillna(0)
    cash        = pick(bal, ["Cash And Cash Equivalents","CashAndCashEquivalents","Cash"])

    cfo   = pick(cfs, ["Operating Cash Flow","Total Cash From Operating Activities","NetCashProvidedByUsedInOperatingActivities"])
    capex = normalize_capex(pick(cfs, ["Capital Expenditure","CapitalExpenditures","Investments In Property Plant And Equipment","Purchase Of Property Plant And Equipment"]))

    years = sorted(set(revenue.index) | set(net_income.index) | set(diluted_eps.index) | set(shares.index) |
                   set(ebit.index) | set(tax_expense.index) | set(pretax.index) | set(equity.index) |
                   set(total_debt.index) | set(cash.index) | set(cfo.index) | set(capex.index))[-11:]

    def A(s): return s.reindex(years).astype("float64") if years else pd.Series(dtype="float64")
    revenue, net_income, eps, shares, ebit, tax_expense, pretax, equity, total_debt, cash, cfo, capex = [
        A(x) for x in [revenue, net_income, diluted_eps, shares, ebit, tax_expense, pretax, equity, total_debt, cash, cfo, capex]
    ]

    if eps.isna().all() and not net_income.isna().all() and not shares.isna().all():
        with np.errstate(invalid="ignore", divide="ignore"):
            eps = net_income / shares.replace({0: np.nan})

    fcf = (cfo - capex) if (not cfo.isna().all() and not capex.isna().all()) else pd.Series([np.nan]*len(years), index=years)
    with np.errstate(divide="ignore", invalid="ignore"):
        tax_rate = (tax_expense / pretax).clip(0, 1)
    nopat = (ebit * (1 - tax_rate.fillna(0.21))) if not ebit.isna().all() else net_income
    roic = roic_series_from(nopat, total_debt, equity, cash)

    df = pd.DataFrame({"Revenue": revenue, "EPS": eps, "Equity": equity, "FCF": fcf, "ROIC": roic, "SharesDiluted": shares}).sort_index().tail(11)
    diag = diag_counts(df)
    return df, years, "Yahoo Finance (yfinance)", diag

def get_price_yahoo(symbol):
    try:
        import yfinance as yf
        info = yf.Ticker(symbol).fast_info
        return float(info.get("last_price", np.nan))
    except Exception:
        return np.nan

# ---- FMP (robust, 10+ years with as-reported fallback) ----
FMP_BASE = "https://financialmodelingprep.com/api/v3"

def fmp_get(path, apikey, params=None):
    if params is None: params = {}
    params["apikey"] = apikey
    r = requests.get(f"{FMP_BASE}/{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def _fmp_pick_year(rep):
    for key in ("date", "calendarYear", "fillingDate", "filingDate"):
        v = rep.get(key)
        if v:
            y = pd.to_datetime(v, errors="coerce").year
            if not pd.isna(y): return int(y)
            try:
                return int(v)
            except Exception:
                pass
    return None

def fmp_series_any(reports, fields):
    if not reports:
        return pd.Series(dtype="float64")
    rows = []
    for rep in reports:
        y = _fmp_pick_year(rep)
        if y is None: 
            continue
        val = None
        for f in fields:
            if f in rep and rep[f] not in (None, "", "None"):
                val = pd.to_numeric(rep[f], errors="coerce")
                break
        rows.append((y, val))
    if not rows:
        return pd.Series(dtype="float64")
    s = pd.Series(dict(rows)).sort_index()
    return s.iloc[-120:].astype("float64")

def _fmp_build_df(inc, bal, cfs):
    revenue        = fmp_series_any(inc, ["revenue","Revenue"])
    net_income     = fmp_series_any(inc, ["netIncome","netIncomeLoss","NetIncomeLoss"])
    diluted_eps    = fmp_series_any(inc, ["epsdiluted","epsDiluted","EPS Diluted"])
    diluted_shares = fmp_series_any(inc, ["weightedAverageShsOutDil","weightedAverageDilutedSharesOutstanding"])
    ebit           = fmp_series_any(inc, ["ebit","operatingIncome","OperatingIncomeLoss"])
    tax_expense    = fmp_series_any(inc, ["incomeTaxExpense","incomeTaxExpenseBenefit"])
    pretax_income  = fmp_series_any(inc, ["incomeBeforeTax","incomeBeforeTaxIncomeTaxExpense"])

    equity     = fmp_series_any(bal, ["totalStockholdersEquity","totalShareholdersEquity"])
    total_debt = fmp_series_any(bal, ["totalDebt","totalDebtAndLeaseObligation"])
    cash       = fmp_series_any(bal, ["cashAndCashEquivalents","cashAndCashEquivalentsAtCarryingValue"])

    # CFO & CapEx — expanded aliases (as-reported varies a lot)
    cfo = fmp_series_any(cfs, [
        "netCashProvidedByOperatingActivities",
        "netCashProvidedByUsedInOperatingActivities",
        "netCashProvidedByUsedInOperatingActivitiesContinuingOperations",
        "netCashProvidedByUsedInOperatingActivitiesAndDiscontinuedOperations"
    ])
    capex = fmp_series_any(cfs, [
        "capitalExpenditure",
        "paymentsForProceedsFromPropertyPlantAndEquipment",
        "paymentsToAcquirePropertyPlantAndEquipment",
        "purchaseOfPropertyAndEquipment",
        "purchaseOfPropertyPlantAndEquipment",
        "capitalExpenditures",
    ])

    years = sorted(set(revenue.index) | set(net_income.index) | set(diluted_eps.index) |
                   set(diluted_shares.index) | set(ebit.index) | set(tax_expense.index) |
                   set(pretax_income.index) | set(equity.index) | set(total_debt.index) |
                   set(cash.index) | set(cfo.index) | set(capex.index))

    if not years:
        return pd.DataFrame(), [], {}

    years = years[-11:]

    def A(s): 
        return s.reindex(years).astype("float64") if years else pd.Series(dtype="float64")

    revenue, net_income, diluted_eps, diluted_shares, ebit, tax_expense, pretax_income, equity, total_debt, cash, cfo, capex = [
        A(x) for x in [revenue, net_income, diluted_eps, diluted_shares, ebit, tax_expense, pretax_income, equity, total_debt, cash, cfo, capex]
    ]

    # EPS fallback if diluted missing
    eps = diluted_eps.copy()
    if eps.isna().all() and not net_income.isna().all() and not diluted_shares.isna().all():
        with np.errstate(invalid="ignore", divide="ignore"):
            eps = net_income / diluted_shares.replace({0: np.nan})

    # Normalize CapEx (some feeds positive)
    capex = capex.apply(lambda v: abs(v) if not pd.isna(v) else v)

    # FCF and ROIC
    fcf = (cfo - capex) if (not cfo.isna().all() and not capex.isna().all()) else pd.Series([np.nan]*len(years), index=years)
    with np.errstate(divide="ignore", invalid="ignore"):
        tax_rate = (tax_expense / pretax_income).clip(0, 1)
    nopat = (ebit * (1 - tax_rate.fillna(0.21))) if not ebit.isna().all() else net_income

    ic = (total_debt.fillna(0) + equity.fillna(0) - cash.fillna(0)).replace({0: np.nan})
    ic_avg = (ic + ic.shift(1)) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        roic = (nopat / ic_avg).replace([np.inf, -np.inf], np.nan)

    df = pd.DataFrame({
        "Revenue": revenue, "EPS": eps, "Equity": equity, "FCF": fcf, "ROIC": roic, "SharesDiluted": diluted_shares
    }).sort_index().tail(11)

    diag = diag_counts(df)
    return df, years, diag

@st.cache_data(show_spinner=False, ttl=1800)
def fetch_fmp(symbol, apikey):
    # 1) Normal, period=annual, large limit
    try:
        inc = fmp_get(f"income-statement/{symbol}", apikey, {"period":"annual", "limit":120})
        bal = fmp_get(f"balance-sheet-statement/{symbol}", apikey, {"period":"annual", "limit":120})
        cfs = fmp_get(f"cash-flow-statement/{symbol}", apikey, {"period":"annual", "limit":120})
        df, years, diag = _fmp_build_df(inc, bal, cfs)
        if len(years) >= 10:
            return df, years, "FMP (annual)", diag
    except Exception:
        pass
    # 2) Some accounts want timescale=yearly
    try:
        inc = fmp_get(f"income-statement/{symbol}", apikey, {"timescale":"yearly", "limit":120})
        bal = fmp_get(f"balance-sheet-statement/{symbol}", apikey, {"timescale":"yearly", "limit":120})
        cfs = fmp_get(f"cash-flow-statement/{symbol}", apikey, {"timescale":"yearly", "limit":120})
        df, years, diag = _fmp_build_df(inc, bal, cfs)
        if len(years) >= 10:
            return df, years, "FMP (yearly)", diag
    except Exception:
        pass
    # 3) Fallback: as-reported (often deeper)
    inc = fmp_get(f"income-statement-as-reported/{symbol}", apikey, {"limit":120})
    bal = fmp_get(f"balance-sheet-statement-as-reported/{symbol}", apikey, {"limit":120})
    cfs = fmp_get(f"cash-flow-statement-as-reported/{symbol}", apikey, {"limit":120})
    df, years, diag = _fmp_build_df(inc, bal, cfs)
    return df, years, "FMP (as-reported)", diag

def get_price_fmp(symbol, apikey):
    try:
        j = requests.get(f"{FMP_BASE}/quote-short/{symbol}?apikey={apikey}", timeout=30).json()
        if isinstance(j, list) and j: return float(j[0].get("price", "nan"))
        return np.nan
    except Exception:
        return np.nan

# ---- Alpha Vantage ----
def av_get(fn, symbol, apikey):
    params = {"function": fn, "symbol": symbol, "apikey": apikey}
    r = requests.get(AV_BASE, params=params, timeout=30)
    r.raise_for_status()
    j = r.json()
    if isinstance(j, dict):
        if j.get("Note"):         raise RuntimeError("Alpha Vantage rate limit hit. Wait a minute or switch provider.")
        if j.get("Information"):  raise RuntimeError(f"Alpha Vantage error: {j['Information']}")
        if j.get("Error Message"):raise RuntimeError(f"Alpha Vantage error: {j['Error Message']}")
    return j.get("annualReports", []), j.get("quarterlyReports", [])

def av_series(reports, field):
    if not reports: return pd.Series(dtype="float64")
    rows = []
    for rep in reports:
        y = pd.to_datetime(rep.get("fiscalDateEnding", ""), errors="coerce").year
        if pd.isna(y): continue
        rows.append((int(y), pd.to_numeric(rep.get(field), errors="coerce")))
    if not rows: return pd.Series(dtype="float64")
    s = pd.Series(dict(rows)).sort_index()
    return s.iloc[-11:].astype("float64")

def av_series_sum(reports, fields):
    total = None
    for f in fields:
        s = av_series(reports, f)
        total = s if total is None else total.add(s, fill_value=0)
    return total if total is not None else pd.Series(dtype="float64")

@st.cache_data(show_spinner=False, ttl=1800)
def fetch_alpha_vantage(symbol, apikey):
    inc_a, _  = av_get("INCOME_STATEMENT", symbol, apikey)
    bal_a, _  = av_get("BALANCE_SHEET",  symbol, apikey)
    cfs_a, _  = av_get("CASH_FLOW",      symbol, apikey)

    if not inc_a and not bal_a and not cfs_a:
        raise RuntimeError("Alpha Vantage returned no annual data. Try Yahoo or FMP, or wait due to rate limits.")

    revenue        = av_series(inc_a, "totalRevenue")
    net_income     = av_series(inc_a, "netIncome")
    diluted_eps    = av_series(inc_a, "dilutedEPS")
    ebit           = av_series(inc_a, "ebit")
    tax_expense    = av_series(inc_a, "incomeTaxExpense")
    pretax_income  = av_series(inc_a, "incomeBeforeTax")

    shares_diluted = av_series(bal_a, "commonStockSharesOutstanding")
    equity         = av_series(bal_a, "totalShareholderEquity")
    debt_primary   = av_series_sum(bal_a, ["shortTermDebt", "longTermDebt"])
    debt_alt       = av_series_sum(bal_a, ["currentLongTermDebt", "longTermDebtNoncurrent"])
    total_debt     = debt_primary if (debt_primary is not None and not debt_primary.empty) else debt_alt
    if total_debt is None or total_debt.empty: total_debt = av_series(bal_a, "totalDebt")
    if total_debt is None or total_debt.empty: total_debt = av_series(bal_a, "shortLongTermDebtTotal")
    cash = av_series(bal_a, "cashAndCashEquivalentsAtCarryingValue")
    if cash.empty: cash = av_series(bal_a, "cashAndCashEquivalents")
    if cash.empty: cash = av_series(bal_a, "cashAndShortTermInvestments")

    cfo   = av_series(cfs_a, "operatingCashflow")
    capex = normalize_capex(av_series(cfs_a, "capitalExpenditures"))

    years = sorted(set(revenue.index) | set(net_income.index) | set(diluted_eps.index) |
                   set(shares_diluted.index) | set(ebit.index) | set(tax_expense.index) |
                   set(pretax_income.index) | set(equity.index) | set(total_debt.index) |
                   set(cash.index) | set(cfo.index) | set(capex.index))[-11:]

    if not years:
        raise RuntimeError("Alpha Vantage fundamentals contain no annual years. Try Yahoo/FMP or wait due to rate limits.")

    def A(s): return s.reindex(years).astype("float64")
    revenue, net_income, diluted_eps, shares_diluted, ebit, tax_expense, pretax_income, equity, total_debt, cash, cfo, capex = [
        A(x) for x in [revenue, net_income, diluted_eps, shares_diluted, ebit, tax_expense, pretax_income, equity, total_debt, cash, cfo, capex]
    ]

    eps = diluted_eps.copy()
    if eps.isna().all() and not net_income.isna().all() and not shares_diluted.isna().all():
        with np.errstate(invalid="ignore", divide="ignore"):
            eps = net_income / shares_diluted.replace({0: np.nan})

    fcf = (cfo - capex) if (not cfo.isna().all() and not capex.isna().all()) else pd.Series([np.nan]*len(years), index=years)
    with np.errstate(divide="ignore", invalid="ignore"):
        tax_rate = (tax_expense / pretax_income).clip(0, 1)
    nopat = (ebit * (1 - tax_rate.fillna(0.21))) if not ebit.isna().all() else net_income
    invested_capital = (total_debt.fillna(0) + equity.fillna(0) - cash.fillna(0)).replace({0: np.nan})
    with np.errstate(divide="ignore", invalid="ignore"):
        roic = (nopat / invested_capital).replace([np.inf, -np.inf], np.nan)

    df = pd.DataFrame({"Revenue": revenue, "EPS": eps, "Equity": equity, "FCF": fcf, "ROIC": roic, "SharesDiluted": shares_diluted}).sort_index().tail(11)
    diag = {"annual_reports_counts": {"income": len(inc_a), "balance": len(bal_a), "cashflow": len(cfs_a)}, "series_non_missing": diag_counts(df)}
    return df, years, "Alpha Vantage (patched)", diag

def get_price_alpha_vantage(symbol, apikey):
    try:
        j = requests.get(f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={apikey}", timeout=30).json()
        return float(j.get("Global Quote", {}).get("05. price", "nan"))
    except Exception:
        return np.nan

# ---------- Price wrappers ----------
def get_price_yahoo_safe(symbol): return get_price_yahoo(symbol)
def get_price_fmp_safe(symbol):   return get_price_fmp(symbol, FMP_KEY) if FMP_KEY else np.nan
def get_price_av_safe(symbol):    return get_price_alpha_vantage(symbol, AV_KEY) if AV_KEY else np.nan

# =================== Auto chooser (FMP → Yahoo → AV) ===================
def try_provider(fn_fetch, fn_price):
    try:
        df, years, source, diag = fn_fetch()
        if df is not None and not df.empty and len(df.index) >= 3:
            price = fn_price() if fn_price else np.nan
            return True, df, years, source, diag, price
    except Exception:
        pass
    return False, None, None, None, None, None

def choose_provider_auto(symbol):
    tried = []
    if FMP_KEY:
        ok, df, years, source, diag, price = try_provider(lambda: fetch_fmp(symbol, FMP_KEY),
                                                          lambda: get_price_fmp(symbol, FMP_KEY))
        tried.append(("FMP", ok))
        if ok: return df, years, source, diag, price, "FMP"
    ok, df, years, source, diag, price = try_provider(lambda: fetch_yahoo(symbol),
                                                      lambda: get_price_yahoo_safe(symbol))
    tried.append(("Yahoo", ok))
    if ok: return df, years, source, diag, price, "Yahoo Finance"
    if AV_KEY:
        ok, df, years, source, diag, price = try_provider(lambda: fetch_alpha_vantage(symbol, AV_KEY),
                                                          lambda: get_price_av_safe(symbol))
        tried.append(("Alpha Vantage", ok))
        if ok: return df, years, source, diag, price, "Alpha Vantage"
    detail = ", ".join([f"{n}:{'OK' if ok else 'fail'}" for n, ok in tried]) or "no providers attempted"
    raise RuntimeError(f"No provider returned usable data ({detail}).")

# =================== Fetch only when Search is pressed ===================
if go:
    try:
        if provider.startswith("Auto"):
            df, years, source, diag, current_price, used = choose_provider_auto(ticker)
            st.success(f"Auto selected: {used}")
        elif provider == "Yahoo Finance":
            df, years, source, diag = fetch_yahoo(ticker)
            current_price = get_price_yahoo_safe(ticker)
        elif provider == "FMP":
            if not FMP_KEY:
                st.error("Missing FMP_API_KEY in Secrets.")
                st.stop()
            df, years, source, diag = fetch_fmp(ticker, FMP_KEY)
            current_price = get_price_fmp_safe(ticker)
        else:  # Alpha Vantage
            if not AV_KEY:
                st.error("Missing ALPHAVANTAGE_API_KEY in Secrets.")
                st.stop()
            df, years, source, diag = fetch_alpha_vantage(ticker, AV_KEY)
            current_price = get_price_av_safe(ticker)
    except Exception as e:
        st.error(f"Fetch error: {e}")
        st.stop()

    if df.empty or len(df.index) < 3:
        st.warning("Not enough data returned by selected provider. Try Auto mode.")
        st.json(diag)
        st.stop()

    st.session_state.data = (df, years, source, diag, current_price, provider)
    st.session_state.last_query = {"ticker": ticker, "provider": provider}

# =================== Render (if we have stored data) ===================
if st.session_state.data is None:
    st.info("Enter a ticker and click **Search** to load data. (Auto mode will pick a working provider.)")
    st.stop()

df, years, source, diag, current_price, chosen = st.session_state.data
st.info(
    f"Provider: {source} · Years: {len(df.index)} · "
    f"Has EPS: {df['EPS'].notna().sum()} · Has FCF: {df['FCF'].notna().sum()} · Has ROIC: {df['ROIC'].notna().sum()}"
)

# =================== Tabs ===================
tabs = st.tabs(["Overview", "Big 5", "Breakdowns", "Valuation", "Diagnostics"])

# -------- Overview --------
with tabs[0]:
    st.subheader(f"{st.session_state.last_query['ticker']} · Source: {source}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Years loaded", f"{len(df.index)}")
    c2.metric("Current Price", "—" if pd.isna(current_price) else f"${current_price:,.2f}")
    c3.metric("Latest EPS", "—" if df['EPS'].dropna().empty else f"{df['EPS'].dropna().iloc[-1]:.2f}")
    c4.metric("Latest ROIC", "—" if df['ROIC'].dropna().empty else pct(df['ROIC'].dropna().iloc[-1]))
    with st.expander("Quick charts"):
        cc1, cc2, cc3 = st.columns(3)
        cc1.line_chart(df[["Revenue","FCF"]].dropna(), height=200, use_container_width=True)
        cc2.line_chart(df[["EPS"]].dropna(), height=200, use_container_width=True)
        cc3.line_chart(df[["ROIC"]].dropna(), height=200, use_container_width=True)

# -------- Big 5 --------
with tabs[1]:
    st.subheader("Big 5 — 10-Year Check")
    sales_cagr_10 = series_cagr_gap(df["Revenue"])
    eps_cagr_10   = series_cagr_gap(df["EPS"])
    eqty_cagr_10  = series_cagr_gap(df["Equity"])
    fcf_cagr_10   = series_cagr_gap(df["FCF"])
    roic_avg_10   = safe_mean(df["ROIC"])
    def pf(v): return "PASS ✅" if not pd.isna(v) and v >= 0.10 else ("—" if pd.isna(v) else "FAIL ❌")
    big5 = pd.DataFrame({
        "Metric": ["Sales (Revenue) CAGR","EPS CAGR","Equity CAGR","FCF CAGR","ROIC (10-yr Avg)"],
        "Value (10y)":  [pct(sales_cagr_10), pct(eps_cagr_10), pct(eqty_cagr_10), pct(fcf_cagr_10), pct(roic_avg_10)],
        "Pass ≥10%?": [pf(sales_cagr_10), pf(eps_cagr_10), pf(eqty_cagr_10), pf(fcf_cagr_10), pf(roic_avg_10)]
    })
    st.dataframe(big5, use_container_width=True, height=240)

# -------- Breakdowns --------
with tabs[2]:
    st.subheader("Trends: 10 / First-5 / Last-3 / Last-1")

    def breakdown_growth(s):
        s = s.dropna()
        if len(s) < 2:
            return np.nan, np.nan, np.nan, np.nan
        ten = series_cagr_gap(s)
        if len(s) >= 5:
            first5 = cagr_over_years(
                s.iloc[0], s.iloc[min(4, len(s) - 1)],
                int(s.index[0]), int(s.index[min(4, len(s) - 1)])
            )
        else:
            first5 = np.nan
        if len(s) >= 4:
            last3 = cagr_over_years(
                s.iloc[-4], s.iloc[-1],
                int(s.index[-4]), int(s.index[-1])
            )
        else:
            last3 = np.nan
        last1 = yoy(s)
        return ten, first5, last3, last1

    def breakdown_roic(s):
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        if len(s)==0: return np.nan, np.nan, np.nan, np.nan
        ten    = safe_mean(s)
        first5 = safe_mean(s.iloc[:5]) if len(s)>=5 else np.nan
        last3  = safe_mean(s.iloc[-3:]) if len(s)>=3 else np.nan
        last1  = s.iloc[-1]
        return ten, first5, last3, last1

    metrics = {
        "Sales CAGR": breakdown_growth(df["Revenue"]),
        "EPS CAGR":   breakdown_growth(df["EPS"]),
        "Equity CAGR":breakdown_growth(df["Equity"]),
        "FCF CAGR":   breakdown_growth(df["FCF"]),
        "ROIC":       breakdown_roic(df["ROIC"]),
    }
    bdf = pd.DataFrame([(k,)+v for k,v in metrics.items()], columns=["Metric","10yr","First 5yr","Last 3yr","Last 1yr"])
    for col in ["10yr","First 5yr","Last 3yr","Last 1yr"]:
        bdf[col] = bdf[col].apply(lambda x: "—" if pd.isna(x) else f"{x*100:.1f}%")
    st.dataframe(bdf, use_container_width=True, height=260)

# -------- Valuation (two models + one MOS slider) --------
with tabs[3]:
    st.subheader("Intrinsic Value (Two Models)")

    mos_frac = st.session_state["mos_pct"] / 100.0

    # --- Rule #1 EPS ---
    eps_series = df["EPS"]
    last_eps = latest_positive(eps_series, lookback=5)  # use latest positive EPS in last 5 data points
    eps_hist_cagr = series_cagr_gap(eps_series)

    @st.cache_data(show_spinner=False, ttl=3600)
    def get_analyst_eps_growth_5y(symbol):
        try:
            url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{symbol}?modules=earningsTrend"
            headers = {"User-Agent": "Mozilla/5.0"}
            j = requests.get(url, headers=headers, timeout=30).json()
            trend = j["quoteSummary"]["result"][0]["earningsTrend"]["trend"]
            for t in trend:
                if t.get("period") in ("+5y", "5y"):
                    raw = t.get("growth", {}).get("raw")
                    if raw is not None: return float(raw)
            lt = j["quoteSummary"]["result"][0]["earningsTrend"].get("longTermEpsGrowthRate", {})
            if "raw" in lt and lt["raw"] is not None: return float(lt["raw"])
        except Exception:
            pass
        return np.nan

    analyst_growth = get_analyst_eps_growth_5y(st.session_state.last_query["ticker"])

    # Growth used = lower of (user, hist, analyst) capped at 15%
    candidates = [g for g in [st.session_state["growth_eps_user"], eps_hist_cagr, analyst_growth] if not pd.isna(g) and g >= 0]
    rule1_growth = min(candidates) if candidates else np.nan
    if not pd.isna(rule1_growth): rule1_growth = min(rule1_growth, 0.15)

    # Current price already fetched
    current_pe = np.nan
    if not pd.isna(current_price) and not pd.isna(last_eps) and last_eps > 0:
        current_pe = current_price / last_eps

    if st.session_state["auto_pe"] and not pd.isna(rule1_growth):
        pe_from_growth = (rule1_growth * 100) * 2
        choices = [pe_from_growth]
        if not pd.isna(current_pe) and current_pe > 0: choices.append(current_pe)
        term_pe = min(min(choices), 50.0)
    else:
        term_pe = st.session_state["terminal_pe_man"]

    def rule1_eps_prices(eps_now, growth, years, terminal_pe, marr):
        if pd.isna(eps_now) or eps_now <= 0: return np.nan, np.nan, np.nan
        if pd.isna(growth) or pd.isna(terminal_pe) or terminal_pe <= 0: return np.nan, np.nan, np.nan
        fut_eps = eps_now * ((1 + growth) ** years)
        sticker = fut_eps * terminal_pe
        fair    = sticker / ((1 + marr) ** years)
        return fut_eps, sticker, fair

    fut_eps, sticker, fair_rule1 = rule1_eps_prices(last_eps, rule1_growth, st.session_state["years_eps"], term_pe, st.session_state["discount"])
    mos_price_rule1 = fair_rule1 * (1 - mos_frac) if not pd.isna(fair_rule1) else np.nan

    # --- FCF-DCF per share ---
    shares_series = df["SharesDiluted"]
    fcf_series    = df["FCF"]
    shares_last   = latest_positive(shares_series, lookback=5)
    fcf_last      = latest_positive(fcf_series, lookback=5)
    fcf_ps_last   = (fcf_last / shares_last) if (not pd.isna(fcf_last) and not pd.isna(shares_last) and shares_last > 0) else np.nan

    def intrinsic_dcf_fcf_per_share(fps_last, growth, years, terminal_g, discount):
        if pd.isna(fps_last) or fps_last <= 0: return np.nan
        if discount <= terminal_g: return np.nan
        pv = 0.0
        f = fps_last
        for t in range(1, years + 1):
            f *= (1 + growth)                 # grow next year's FCF/share
            pv += f / ((1 + discount) ** t)   # discount to present
        f_next = f * (1 + terminal_g)         # FCF in year N+1
        tv = f_next / (discount - terminal_g) # Gordon terminal value at year N
        pv += tv / ((1 + discount) ** years)  # discount terminal back to today
        return pv

    iv_dcf  = intrinsic_dcf_fcf_per_share(fcf_ps_last, st.session_state["growth_fcf"],
                                          st.session_state["years_dcf"], st.session_state["terminal_g"], st.session_state["discount"])
    mos_price_dcf = iv_dcf * (1 - mos_frac) if not pd.isna(iv_dcf) else np.nan

    # --- Display ---
    k1, k2, k3 = st.columns(3)
    k1.metric("Rule #1 Fair Value / share", "—" if pd.isna(fair_rule1) else f"${fair_rule1:,.2f}")
    k2.metric("DCF Fair Value / share",     "—" if pd.isna(iv_dcf)     else f"${iv_dcf:,.2f}")
    k3.metric("Current Price",              "—" if pd.isna(current_price) else f"${current_price:,.2f}")

    m1, m2 = st.columns(2)
    m1.metric(f"MOS Price (Rule #1, {int(st.session_state['mos_pct'])}%)", "—" if pd.isna(mos_price_rule1) else f"${mos_price_rule1:,.2f}")
    m2.metric(f"MOS Price (DCF, {int(st.session_state['mos_pct'])}%)",     "—" if pd.isna(mos_price_dcf)   else f"${mos_price_dcf:,.2f}")

    # Friendly hints if an input is missing
    if pd.isna(fair_rule1):
        st.warning("Rule #1 fair value unavailable — check EPS last (must be > 0) and growth inputs. Try switching provider to FMP or Yahoo.")
    if pd.isna(iv_dcf):
        st.warning("DCF fair value unavailable — need positive FCF/share. Try switching provider or check CFO/CapEx fields.")

    with st.expander("Valuation inputs used"):
        st.write({
            "EPS last (latest positive in last 5)": None if pd.isna(last_eps) else round(float(last_eps), 4),
            "Rule #1 growth (used)": None if pd.isna(rule1_growth) else round(float(rule1_growth*100), 2),
            "Terminal P/E": None if pd.isna(term_pe) else round(float(term_pe), 2),
            "Future EPS (year N)": None if pd.isna(fut_eps) else round(float(fut_eps), 4),
            "Sticker (EPS × P/E at year N)": None if pd.isna(sticker) else round(float(sticker), 2),
            "Fair (sticker discounted @ MARR)": None if pd.isna(fair_rule1) else round(float(fair_rule1), 2),
            "FCF/share last (latest positive in last 5)": None if pd.isna(fcf_ps_last) else round(float(fcf_ps_last), 4),
            "DCF years": st.session_state["years_dcf"],
            "DCF growth %": round(float(st.session_state["growth_fcf"]*100), 2),
            "Terminal growth %": round(float(st.session_state["terminal_g"]*100), 2),
            "Discount (MARR) %": round(float(st.session_state["discount"]*100), 2),
        })
    st.caption("MOS slider applies equally to both fair values. With reasonable ADBE settings (10% MARR, growth ≤15%, Auto P/E), Rule #1 fair value often lands around the mid-$500s.")

# -------- Diagnostics --------
with tabs[4]:
    st.subheader("Data Diagnostics")
    st.write("Non-missing values per series (higher is better):")
    st.json(diag)
    with st.expander("Raw data"):
        st.dataframe(df, use_container_width=True)
    st.caption("If EPS/FCF are missing on one provider, try Auto mode or switch providers. Fiscal calendars and labels vary.")
