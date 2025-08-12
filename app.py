import streamlit as st
import pandas as pd
import numpy as np
import requests

# =================== Page ===================
st.set_page_config(page_title="Phil Town Big 5 Screener", layout="wide")
st.title("Phil Town Big 5 — 10-Year Screener")
st.caption("Big 5 (Sales, EPS, Equity, FCF CAGR + 10-yr Avg ROIC), 10/5/3/1 breakdowns, Rule #1 EPS & FCF DCF valuations with custom MOS sliders, and a Value-Investor summary.")

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
# Rule #1 EPS — you provide an estimate; app uses LOWER of (this, 10y EPS CAGR, analyst 5y EPS growth), capped at 15%
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

# Per-valuation Margin of Safety sliders (as % off fair value)
st.sidebar.markdown("### Margin of Safety sliders")
mos_eps_pct = st.sidebar.slider("MOS for Rule #1 EPS (%)", 0, 90, 50, step=5) / 100.0
mos_dcf_pct = st.sidebar.slider("MOS for FCF DCF (%)", 0, 90, 40, step=5) / 100.0

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

# =================== Data fetchers ===================
AV_BASE = "https://www.alphavantage.co/query"
FMP_BASE = "https://financialmodelingprep.com/api/v3"

def av_get(fn: str, symbol: str, apikey: str):
    params = {"function": fn, "symbol": symbol, "apikey": apikey}
    r = requests.get(AV_BASE, params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("annualReports", [])

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

    # ROIC
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

# Current price
def get_price_alpha_vantage(symbol: str, apikey: str) -> float:
    try:
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={apikey}"
        j = requests.get(url, timeout=30).json()
        return float(j.get("Global Quote", {}).get("05. price", "nan"))
    except Exception:
        return np.nan

def get_price_fmp(symbol: str, apikey: str) -> float:
    try:
        url = f"{FMP_BASE}/quote-short/{symbol}?apikey={apikey}"
        j = requests.get(url, timeout=30).json()
        if isinstance(j, list) and j:
            return float(j[0].get("price", "nan"))
        return np.nan
    except Exception:
        return np.nan

# Analyst 5y EPS growth (Yahoo Finance earningsTrend)
@st.cache_data(show_spinner=False, ttl=3600)
def get_analyst_eps_growth_5y(symbol: str) -> float:
    try:
        url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{symbol}?modules=earningsTrend"
        headers = {"User-Agent": "Mozilla/5.0"}
        j = requests.get(url, headers=headers, timeout=30).json()
        trend = j["quoteSummary"]["result"][0]["earningsTrend"]["trend"]
        # Look for "+5y" period or "longTermEpsGrowthRate"
        for t in trend:
            if t.get("period") in ("+5y", "5y"):
                val = t.get("growth", {}).get("raw")
                if val is not None:
                    return float(val)  # already decimal (e.g., 0.12)
        # fallback: sometimes stored as longTermEpsGrowthRate
        lt = j["quoteSummary"]["result"][0]["earningsTrend"].get("longTermEpsGrowthRate", {})
        if "raw" in lt and lt["raw"] is not None:
            return float(lt["raw"])
    except Exception:
        pass
    return np.nan

# =================== Intrinsic value (FCF DCF) ===================
def intrinsic_dcf_fcf(fps_last: float, growth: float, years: int, terminal_g: float, discount: float) -> float:
    if pd.isna(fps_last) or fps_last <= 0: return np.nan
    if discount <= terminal_g: return np.nan
    pv = 0.0
    f = fps_last
    for t in range(1, years + 1):
        f *= (1 + growth)
        pv += f / ((1 + discount) ** t)
    f_next = f * (1 + terminal_g)
    tv = f_next / (discount - terminal_g)
    pv += tv / ((1 + discount) ** years)
    return pv

# =================== Run ===================
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

    # ---------- Big 5 (10-year) ----------
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

    # ---------- Metric Breakdown (10 / First-5 / Last-3 / Last-1) ----------
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

    def breakdown_roic(s: pd.Series):
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        if len(s) == 0:
            return np.nan, np.nan, np.nan, np.nan
        ten = safe_mean(s)
        first5 = safe_mean(s.iloc[:5]) if len(s) >= 5 else np.nan
        last3 = safe_mean(s.iloc[-3:]) if len(s) >= 1 else np.nan
        last1 = s.iloc[-1]
        return ten, first5, last3, last1

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

    st.markdown("### Metric Breakdown (10 / First-5 / Last-3 / Last-1)")
    st.dataframe(breakdown_fmt, use_container_width=True)

    # ===== Footer shows Rule #1 inputs actually used =====
    # We'll compute them below first, then display a caption afterward.

    # ---------- Data Coverage ----------
    st.markdown("#### Data Coverage (non-missing values used)")
    coverage = df.notna().sum().rename("Valid Years").to_frame()
    coverage["Out of"] = len(df.index)
    st.dataframe(coverage.T, use_container_width=True)

    # ---------- Rule #1 EPS valuation ----------
    st.markdown("### Intrinsic Value")

    # Current EPS (last annual)
    last_eps = df["EPS"].dropna().iloc[-1] if df["EPS"].notna().any() else np.nan

    # Analyst 5y EPS growth (Yahoo)
    analyst_growth = get_analyst_eps_growth_5y(ticker)  # decimal (e.g., 0.12)

    # Historical EPS CAGR (10y)
    eps_hist_cagr = eps_cagr_10

    # Rule #1 growth used = lower of (user, hist, analyst), cap at 15%
    candidates = [g for g in [growth_eps_user, eps_hist_cagr, analyst_growth] if not pd.isna(g) and g >= 0]
    rule1_growth = min(candidates) if candidates else np.nan
    if not pd.isna(rule1_growth):
        rule1_growth = min(rule1_growth, 0.15)  # cap 15%

    # Current price
    if using_av:
        current_price = get_price_alpha_vantage(ticker, AV_KEY) if key_ok else np.nan
    else:
        current_price = get_price_fmp(ticker, FMP_KEY) if key_ok else np.nan

    # Current P/E (fallback)
    current_pe = np.nan
    if not pd.isna(current_price) and not pd.isna(last_eps) and last_eps > 0:
        current_pe = current_price / last_eps

    # Terminal P/E used
    if auto_pe and not pd.isna(rule1_growth):
        pe_from_growth = (rule1_growth * 100.0) * 2.0
        choices = [pe_from_growth]
        if not pd.isna(current_pe) and current_pe > 0:
            choices.append(current_pe)
        term_pe = min(min(choices), 50.0) if choices else np.nan
    else:
        term_pe = terminal_pe_manual

    def rule1_eps_to_prices(eps_now, growth, years, terminal_pe, marr):
        if pd.isna(eps_now) or eps_now <= 0: return np.nan, np.nan, np.nan
        if pd.isna(growth) or growth < 0: return np.nan, np.nan, np.nan
        if pd.isna(terminal_pe) or terminal_pe <= 0: return np.nan, np.nan, np.nan
        fut_eps = eps_now * ((1 + growth) ** years)
        sticker = fut_eps * terminal_pe
        fair = sticker / ((1 + marr) ** years)
        mos = fair * (1.0 - mos_eps_pct)  # custom MOS slider for EPS
        return sticker, fair, mos

    sticker_price, fair_value, mos_price_rule1 = rule1_eps_to_prices(last_eps, rule1_growth, years_eps, term_pe, discount)

    # ---------- FCF DCF ----------
    shares_last = df["SharesDiluted"].dropna().iloc[-1] if "SharesDiluted" in df and df["SharesDiluted"].notna().any() else np.nan
    fcf_last = df["FCF"].dropna().iloc[-1] if df["FCF"].notna().any() else np.nan
    fcf_per_share_last = (fcf_last / shares_last) if (not pd.isna(fcf_last) and not pd.isna(shares_last) and shares_last > 0) else np.nan

    iv_dcf = intrinsic_dcf_fcf(fcf_per_share_last, growth_fcf, years_dcf, terminal_g, discount)
    mos_price_dcf = iv_dcf * (1.0 - mos_dcf_pct) if not pd.isna(iv_dcf) else np.nan

    # ---------- Display valuations ----------
    colv1, colv2, colv3, colv4 = st.columns(4)
    colv1.metric("Sticker Price (Rule #1 EPS)", f"${sticker_price:,.2f}" if not pd.isna(sticker_price) else "—")
    colv2.metric("Fair Value (PV @ MARR)", f"${fair_value:,.2f}" if not pd.isna(fair_value) else "—")
    colv3.metric(f"MOS Price (EPS, {int(mos_eps_pct*100)}%)", f"${mos_price_rule1:,.2f}" if not pd.isna(mos_price_rule1) else "—")
    colv4.metric("Current Price", f"${current_price:,.2f}" if not pd.isna(current_price) else "—")

    colv5, colv6, colv7, colv8 = st.columns(4)
    colv5.metric("Rule #1 Growth Used", "—" if pd.isna(rule1_growth) else f"{rule1_growth*100:.1f}%")
    colv6.metric("Terminal P/E Used", "—" if pd.isna(term_pe) else f"{term_pe:.1f}")
    colv7.metric("DCF Intrinsic / sh.", f"${iv_dcf:,.2f}" if not pd.isna(iv_dcf) else "—")
    colv8.metric(f"MOS Price (DCF, {int(mos_dcf_pct*100)}%)", f"${mos_price_dcf:,.2f}" if not pd.isna(mos_price_dcf) else "—")

    # Add caption under Metric Breakdown now that we know the chosen Rule #1 inputs
    rule1_growth_text = "—" if pd.isna(rule1_growth) else f"{rule1_growth*100:.1f}%"
    term_pe_text = "—" if pd.isna(term_pe) else f"{term_pe:.1f}"
    st.caption(
        "Growth rows show **CAGR**; 'Last 1yr' is **YoY**. ROIC rows show **averages**; 'Last 1yr' is the most recent ROIC.  \n"
        f"**Rule #1 EPS settings used:** Growth = {rule1_growth_text} · Terminal P/E = {term_pe_text}."
    )

    # ---------- Optional: Value-Investor Summary ----------
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
                    "rule1": {
                        "eps_last": float(last_eps) if not pd.isna(last_eps) else None,
                        "growth_used": float(rule1_growth) if not pd.isna(rule1_growth) else None,
                        "terminal_pe": float(term_pe) if not pd.isna(term_pe) else None,
                        "sticker": float(sticker_price) if not pd.isna(sticker_price) else None,
                        "fair_value": float(fair_value) if not pd.isna(fair_value) else None,
                        "mos_price": float(mos_price_rule1) if not pd.isna(mos_price_rule1) else None,
                    },
                    "dcf": {
                        "fcf_per_share_last": float(fcf_per_share_last) if not pd.isna(fcf_per_share_last) else None,
                        "iv_dcf": float(iv_dcf) if not pd.isna(iv_dcf) else None,
                        "discount": float(discount)
                    },
                    "current_price": float(current_price) if not pd.isna(current_price) else None
                }
                prompt = (
                    "You are a disciplined value investor (Phil Town style). "
                    "Use the structured data to judge 10% rule compliance (Sales/EPS/Equity/FCF CAGRs and 10-yr Avg ROIC), "
                    "comment on the 10/5/3/1 trends, and compare the Rule #1 EPS MOS & DCF intrinsic values to the current price. "
                    "Be concise (<180 words), neutral, and focus on business quality and price."
                )
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role":"user","content": prompt + "\n\n" + str(context)}],
                    temperature=0.3,
                    max_tokens=350,
                )
                st.write(resp.choices[0].message.content)
            except Exception as e:
                st.error(f"OpenAI error: {e}")

    # ---------- Raw + charts ----------
    with st.expander("Raw series used"):
        st.dataframe(df)

    with st.expander("Mini charts"):
        cc1, cc2, cc3 = st.columns(3)
        cc1.line_chart(df[["Revenue","FCF"]].dropna(), height=220)
        cc2.line_chart(df[["EPS"]].dropna(), height=220)
        cc3.line_chart(df[["ROIC"]].dropna(), height=220)

else:
    st.info("Enter a ticker and click **Search**. Choose provider in the sidebar and confirm your API key in Secrets.")
