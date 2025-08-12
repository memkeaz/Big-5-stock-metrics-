import streamlit as st
import pandas as pd
import numpy as np
import requests

# =================== Page ===================
st.set_page_config(
    page_title="Phil Town Big 5 Screener",
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.markdown(
    """
    <style>
      /* Mobile-friendly paddings & tighter layout */
      .block-container { padding-top: 1rem; padding-bottom: 3rem; max-width: 1100px; }
      @media (max-width: 640px) {
        .block-container { padding-left: 0.8rem; padding-right: 0.8rem; }
        .stMetric { text-align: left; }
      }
      .fine { color: #777; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True
)

# =================== Header ===================
st.title("Phil Town Big 5 Screener")
st.caption("Check the Big 5 (Sales, EPS, Equity, FCF CAGRs + 10‑yr Avg ROIC), view 10/5/3/1 trend breakdowns, and run Rule #1 EPS & FCF‑DCF valuations with adjustable MOS.")

# =================== Secrets ===================
OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", "").strip()
AV_KEY     = st.secrets.get("ALPHAVANTAGE_API_KEY", "").strip()
FMP_KEY    = st.secrets.get("FMP_API_KEY", "").strip()

# =================== Controls (single Search button) ===================
with st.form("search"):
    top1, top2 = st.columns([2, 1])
    with top1:
        ticker = st.text_input("Ticker", value="ADBE").strip().upper()
    with top2:
        provider = st.selectbox("Data source", ["FMP", "Alpha Vantage"], index=0)
    adv = st.expander("Valuation Settings", expanded=False)
    with adv:
        st.markdown("**Rule #1 EPS (Phil Town)**")
        years_eps        = st.slider("Years (EPS projection)", 5, 15, 10)
        growth_eps_user  = st.number_input("Your EPS growth estimate (annual, %)", 0.0, 50.0, 12.0, step=0.5) / 100.0
        auto_pe          = st.checkbox("Terminal P/E ≈ lower of (2× growth, current P/E), capped at 50", value=True)
        terminal_pe_man  = st.number_input("Terminal P/E (manual if Auto off)", 5.0, 60.0, 20.0, step=0.5)
        st.markdown("---")
        st.markdown("**FCF DCF**")
        years_dcf  = st.slider("Years (DCF)", 5, 15, 10)
        growth_fcf = st.number_input("FCF growth (annual, %)", 0.0, 50.0, 10.0, step=0.5) / 100.0
        terminal_g = st.number_input("Terminal growth (FCF, %)", 0.0, 6.0, 3.0, step=0.25) / 100.0
        discount   = st.number_input("MARR / Discount rate (%, both models)", 4.0, 20.0, 10.0, step=0.5) / 100.0
        st.markdown("---")
        st.markdown("**Default Margin of Safety (%)**")
        mos_eps_default = st.slider("Rule #1 EPS MOS", 0, 90, 50, step=5) / 100.0
        mos_dcf_default = st.slider("FCF DCF MOS", 0, 90, 50, step=5) / 100.0

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

def safe_mean(s: pd.Series) -> float:
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    return s.mean() if len(s) else np.nan

def pct(x): return "—" if pd.isna(x) else f"{x*100:.1f}%"

# =================== Fetchers ===================
AV_BASE  = "https://www.alphavantage.co/query"
FMP_BASE = "https://financialmodelingprep.com/api/v3"

def _diag(df: pd.DataFrame):
    return {k:int(v) for k,v in df.notna().sum().to_dict().items()}

# ----- Alpha Vantage (patched) -----
def av_get(fn: str, symbol: str, apikey: str):
    p = {"function": fn, "symbol": symbol, "apikey": apikey}
    r = requests.get(AV_BASE, params=p, timeout=30)
    r.raise_for_status()
    return r.json().get("annualReports", [])

def av_series(reports, field) -> pd.Series:
    if not reports: return pd.Series(dtype="float64")
    rows = []
    for rep in reports:
        y = pd.to_datetime(rep.get("fiscalDateEnding", ""), errors="coerce").year
        if pd.isna(y): continue
        rows.append((int(y), pd.to_numeric(rep.get(field), errors="coerce")))
    if not rows: return pd.Series(dtype="float64")
    s = pd.Series(dict(rows)).sort_index()
    return s.iloc[-11:].astype("float64")

def av_series_sum(reports, fields) -> pd.Series:
    total = None
    for f in fields:
        s = av_series(reports, f)
        total = s if total is None else total.add(s, fill_value=0)
    return total if total is not None else pd.Series(dtype="float64")

@st.cache_data(show_spinner=False, ttl=1800)
def fetch_alpha_vantage(symbol: str, apikey: str):
    inc = av_get("INCOME_STATEMENT", symbol, apikey)
    bal = av_get("BALANCE_SHEET",  symbol, apikey)
    cfs = av_get("CASH_FLOW",      symbol, apikey)

    revenue        = av_series(inc, "totalRevenue")
    net_income     = av_series(inc, "netIncome")
    diluted_eps    = av_series(inc, "dilutedEPS")
    ebit           = av_series(inc, "ebit")
    tax_expense    = av_series(inc, "incomeTaxExpense")
    pretax_income  = av_series(inc, "incomeBeforeTax")
    shares_diluted = av_series(bal, "commonStockSharesOutstanding")  # proxy
    equity         = av_series(bal, "totalShareholderEquity")

    debt_primary   = av_series_sum(bal, ["shortTermDebt", "longTermDebt"])
    debt_alt       = av_series_sum(bal, ["currentLongTermDebt", "longTermDebtNoncurrent"])
    debt_fb1       = av_series(bal, "totalDebt")
    debt_fb2       = av_series(bal, "shortLongTermDebtTotal")
    total_debt     = debt_primary if (not debt_primary.empty) else (debt_alt if not debt_alt.empty else (debt_fb1 if not debt_fb1.empty else debt_fb2))

    cash = av_series(bal, "cashAndCashEquivalentsAtCarryingValue")
    if cash.empty: cash = av_series(bal, "cashAndCashEquivalents")
    if cash.empty: cash = av_series(bal, "cashAndShortTermInvestments")

    cfo   = av_series(cfs, "operatingCashflow")
    capex = av_series(cfs, "capitalExpenditures")

    years = sorted(set(revenue.index) | set(net_income.index) | set(diluted_eps.index) |
                   set(shares_diluted.index) | set(ebit.index) | set(tax_expense.index) |
                   set(pretax_income.index) | set(equity.index) | set(total_debt.index) |
                   set(cash.index) | set(cfo.index) | set(capex.index))[-11:]

    def A(s): return s.reindex(years).astype("float64") if years else pd.Series(dtype="float64")
    revenue, net_income, diluted_eps, shares_diluted, ebit, tax_expense, pretax_income, equity, total_debt, cash, cfo, capex = [
        A(x) for x in [revenue, net_income, diluted_eps, shares_diluted, ebit, tax_expense, pretax_income, equity, total_debt, cash, cfo, capex]
    ]

    eps = diluted_eps.copy()
    if eps.isna().all() and not net_income.isna().all() and not shares_diluted.isna().all():
        with np.errstate(invalid="ignore", divide="ignore"):
            eps = net_income / shares_diluted.replace({0: np.nan})  # fallback, not perfect

    fcf = (cfo - capex) if (not cfo.isna().all() and not capex.isna().all()) else pd.Series([np.nan]*len(years), index=years)

    with np.errstate(divide="ignore", invalid="ignore"):
        tax_rate = (tax_expense / pretax_income).clip(0, 1)
    nopat = (ebit * (1 - tax_rate.fillna(0.21))) if not ebit.isna().all() else net_income
    invested_capital = (total_debt.fillna(0) + equity.fillna(0) - cash.fillna(0))
    invested_capital = invested_capital.replace({0: np.nan})  # avoid division blowups
    with np.errstate(divide="ignore", invalid="ignore"):
        roic = (nopat / invested_capital).replace([np.inf, -np.inf], np.nan)

    df = pd.DataFrame({
        "Revenue": revenue, "EPS": eps, "Equity": equity, "FCF": fcf, "ROIC": roic, "SharesDiluted": shares_diluted
    }).sort_index().tail(11)

    diag = _diag(df)
    return df, years, "Alpha Vantage (patched)", diag

# ----- FMP (recommended for EPS accuracy) -----
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
            try: y = int(rep.get("calendarYear")); 
            except: continue
        rows.append((int(y), pd.to_numeric(rep.get(field), errors="coerce")))
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

    diag = _diag(df)
    return df, years, "FMP", diag

# =================== Price & Analyst growth ===================
def get_price_alpha_vantage(symbol: str, apikey: str) -> float:
    try:
        j = requests.get(
            f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={apikey}",
            timeout=30
        ).json()
        return float(j.get("Global Quote", {}).get("05. price", "nan"))
    except Exception:
        return np.nan

def get_price_fmp(symbol: str, apikey: str) -> float:
    try:
        j = requests.get(f"{FMP_BASE}/quote-short/{symbol}?apikey={apikey}", timeout=30).json()
        if isinstance(j, list) and j:
            return float(j[0].get("price", "nan"))
        return np.nan
    except Exception:
        return np.nan

@st.cache_data(show_spinner=False, ttl=3600)
def get_analyst_eps_growth_5y(symbol: str) -> float:
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
        if "raw" in lt and lt["raw"] is not None:
            return float(lt["raw"])
    except Exception:
        pass
    return np.nan

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
if go:
    using_av  = (provider == "Alpha Vantage")
    have_key  = (AV_KEY if using_av else FMP_KEY)
    if not have_key:
        st.error("Missing API key in Streamlit Secrets for the selected provider.")
        st.stop()

    # Fetch
    try:
        if using_av:
            df, years, source, diag = fetch_alpha_vantage(ticker, AV_KEY)
            current_price = get_price_alpha_vantage(ticker, AV_KEY)
            st.caption("Using Alpha Vantage — if EPS/ROIC looks off for ADBE, try **FMP**.")
        else:
            df, years, source, diag = fetch_fmp(ticker, FMP_KEY)
            current_price = get_price_fmp(ticker, FMP_KEY)
    except Exception as e:
        st.error(f"Fetch error: {e}")
        st.stop()

    if df.empty or len(df.index) < 3:
        st.warning("Not enough data returned. Try switching the provider.")
        st.json(diag)
        st.stop()

    # =================== SECTIONS ===================
    tabs = st.tabs(["Overview", "Big 5", "Breakdowns", "Valuation", "Summary (AI)", "Diagnostics"])

    # -------- Overview --------
    with tabs[0]:
        st.subheader(f"{ticker} · Source: {source}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Years loaded", f"{len(df.index)}")
        c2.metric("Current Price", "—" if pd.isna(current_price) else f"${current_price:,.2f}")
        c3.metric("Latest EPS", "—" if df['EPS'].dropna().empty else f"{df['EPS'].dropna().iloc[-1]:.2f}")
        c4.metric("Latest ROIC", "—" if df['ROIC'].dropna().empty else pct(df['ROIC'].dropna().iloc[-1]))
        st.markdown('<span class="fine">Tip: If ADBE looks wrong on Alpha Vantage, switch to FMP (more reliable EPS).</span>', unsafe_allow_html=True)
        with st.expander("Quick charts"):
            cc1, cc2, cc3 = st.columns(3)
            cc1.line_chart(df[["Revenue","FCF"]].dropna(), height=200, use_container_width=True)
            cc2.line_chart(df[["EPS"]].dropna(), height=200, use_container_width=True)
            cc3.line_chart(df[["ROIC"]].dropna(), height=200, use_container_width=True)

    # -------- Big 5 --------
    with tabs[1]:
        st.subheader("Big 5 — 10‑Year Check")
        sales_cagr_10 = series_cagr_gap(df["Revenue"])
        eps_cagr_10   = series_cagr_gap(df["EPS"])
        eqty_cagr_10  = series_cagr_gap(df["Equity"])
        fcf_cagr_10   = series_cagr_gap(df["FCF"])
        roic_avg_10   = safe_mean(df["ROIC"])
        def pf(v): return "PASS ✅" if not pd.isna(v) and v >= 0.10 else ("—" if pd.isna(v) else "FAIL ❌")
        big5 = pd.DataFrame({
            "Metric": ["Sales (Revenue) CAGR","EPS CAGR","Equity CAGR","FCF CAGR","ROIC (10‑yr Avg)"],
            "Value (10y)":  [pct(sales_cagr_10), pct(eps_cagr_10), pct(eqty_cagr_10), pct(fcf_cagr_10), pct(roic_avg_10)],
            "Pass ≥10%?": [pf(sales_cagr_10), pf(eps_cagr_10), pf(eqty_cagr_10), pf(fcf_cagr_10), pf(roic_avg_10)]
        })
        st.dataframe(big5, use_container_width=True, height=240)

    # -------- Breakdowns --------
    with tabs[2]:
        st.subheader("Trends: 10 / First‑5 / Last‑3 / Last‑1")
        def breakdown_growth(s: pd.Series):
            s = s.dropna()
            if len(s) < 2: return np.nan, np.nan, np.nan, np.nan
            ten = series_cagr_gap(s)
            first5 = cagr_over_years(s.iloc[0], s.iloc[min(4, len(s)-1)], int(s.index[0]), int(s.index[min(4, len(s)-1]))) if len(s) >= 5 else np.nan
            last3  = cagr_over_years(s.iloc[-4], s.iloc[-1], int(s.index[-4]), int(s.index[-1])) if len(s) >= 4 else np.nan
            last1  = yoy(s)
            return ten, first5, last3, last1

        def breakdown_roic(s: pd.Series):
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
        bdf = pd.DataFrame(
            [(k,)+v for k,v in metrics.items()],
            columns=["Metric","10yr","First 5yr","Last 3yr","Last 1yr"]
        )
        for col in ["10yr","First 5yr","Last 3yr","Last 1yr"]:
            bdf[col] = bdf[col].apply(lambda x: "—" if pd.isna(x) else f"{x*100:.1f}%")
        st.dataframe(bdf, use_container_width=True, height=260)

    # -------- Valuation --------
    with tabs[3]:
        st.subheader("Intrinsic Value")
        # Live MOS sliders (mobile-friendly in-row)
        m1, m2 = st.columns(2)
        with m1: mos_eps_live = st.slider("MOS — Rule #1 EPS", 0, 90, int(mos_eps_default*100), step=5) / 100.0
        with m2: mos_dcf_live = st.slider("MOS — FCF DCF",   0, 90, int(mos_dcf_default*100), step=5) / 100.0

        last_eps = df["EPS"].dropna().iloc[-1] if df["EPS"].notna().any() else np.nan
        analyst_growth = get_analyst_eps_growth_5y(ticker)  # decimal, may be NaN
        eps_hist_cagr  = series_cagr_gap(df["EPS"])

        # Rule #1 growth used = lower of (user, hist, analyst), capped at 15%
        candidates = [g for g in [growth_fcf*0 + growth_eps_user, eps_hist_cagr, analyst_growth] if not pd.isna(g) and g >= 0]
        rule1_growth = min(candidates) if candidates else np.nan
        if not pd.isna(rule1_growth): rule1_growth = min(rule1_growth, 0.15)

        # Current price
        current_price = (get_price_alpha_vantage(ticker, AV_KEY) if using_av else get_price_fmp(ticker, FMP_KEY))

        current_pe = np.nan
        if not pd.isna(current_price) and not pd.isna(last_eps) and last_eps > 0:
            current_pe = current_price / last_eps

        # Terminal P/E
        if auto_pe and not pd.isna(rule1_growth):
            pe_from_growth = (rule1_growth * 100) * 2
            choices = [pe_from_growth]
            if not pd.isna(current_pe) and current_pe > 0: choices.append(current_pe)
            term_pe = min(min(choices), 50.0)
        else:
            term_pe = terminal_pe_man

        def rule1_eps_prices(eps_now, growth, years, terminal_pe, marr):
            if pd.isna(eps_now) or eps_now <= 0: return np.nan, np.nan
            if pd.isna(growth) or pd.isna(terminal_pe) or terminal_pe <= 0: return np.nan, np.nan
            fut_eps = eps_now * ((1 + growth) ** years)
            sticker = fut_eps * terminal_pe
            fair    = sticker / ((1 + marr) ** years)
            return sticker, fair

        sticker, fair = rule1_eps_prices(last_eps, rule1_growth, years_eps, term_pe, discount)
        mos_rule1 = fair * (1 - mos_eps_live) if not pd.isna(fair) else np.nan

        # FCF DCF (per share)
        shares_last = df["SharesDiluted"].dropna().iloc[-1] if df["SharesDiluted"].notna().any() else np.nan
        fcf_last    = df["FCF"].dropna().iloc[-1] if df["FCF"].notna().any() else np.nan
        fcf_ps_last = fcf_last / shares_last if (not pd.isna(fcf_last) and not pd.isna(shares_last) and shares_last > 0) else np.nan
        iv_dcf      = intrinsic_dcf_fcf(fcf_ps_last, growth_fcf, years_dcf, terminal_g, discount)
        mos_dcf     = iv_dcf * (1 - mos_dcf_live) if not pd.isna(iv_dcf) else np.nan

        # Metrics
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Sticker (Rule #1 EPS)", "—" if pd.isna(sticker) else f"${sticker:,.2f}")
        k2.metric("Fair Value (PV @ MARR)", "—" if pd.isna(fair) else f"${fair:,.2f}")
        k3.metric(f"MOS Price (EPS, {int(mos_eps_live*100)}%)", "—" if pd.isna(mos_rule1) else f"${mos_rule1:,.2f}")
        k4.metric("Current Price", "—" if pd.isna(current_price) else f"${current_price:,.2f}")

        j1, j2 = st.columns(2)
        j1.metric("DCF Intrinsic / share", "—" if pd.isna(iv_dcf) else f"${iv_dcf:,.2f}")
        j2.metric(f"MOS Price (DCF, {int(mos_dcf_live*100)}%)", "—" if pd.isna(mos_dcf) else f"${mos_dcf:,.2f}")

        st.caption(f"Rule #1 growth used: {'—' if pd.isna(rule1_growth) else f'{rule1_growth*100:.1f}%'} · Terminal P/E: {'—' if pd.isna(term_pe) else f'{term_pe:.1f}'}")

    # -------- Summary (AI) --------
    with tabs[4]:
        st.subheader("Value‑Investor Summary (OpenAI)")
        if not OPENAI_KEY:
            st.info("Add **OPENAI_API_KEY** in Streamlit Secrets to enable this summary.")
        else:
            if st.button("Generate Summary"):
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=OPENAI_KEY)
                    sales_cagr_10 = series_cagr_gap(df["Revenue"])
                    eps_cagr_10   = series_cagr_gap(df["EPS"])
                    eqty_cagr_10  = series_cagr_gap(df["Equity"])
                    fcf_cagr_10   = series_cagr_gap(df["FCF"])
                    roic_avg_10   = safe_mean(df["ROIC"])
                    context = {
                        "ticker": ticker,
                        "years": list(map(int, df.index.tolist())),
                        "big5_10y": {
                            "sales_cagr": float(sales_cagr_10) if not pd.isna(sales_cagr_10) else None,
                            "eps_cagr": float(eps_cagr_10) if not pd.isna(eps_cagr_10) else None,
                            "equity_cagr": float(eqty_cagr_10) if not pd.isna(eqty_cagr_10) else None,
                            "fcf_cagr": float(fcf_cagr_10) if not pd.isna(fcf_cagr_10) else None,
                            "roic_avg": float(roic_avg_10) if not pd.isna(roic_avg_10) else None
                        }
                    }
                    prompt = (
                        "You are a disciplined value investor (Phil Town style). "
                        "Use the structured data to judge 10% rule compliance, mention notable 10/5/3/1 trends, "
                        "and comment on whether valuations appear conservative relative to a quality business."
                        "Be concise (<160 words) and neutral."
                    )
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role":"user","content": prompt + "\n\n" + str(context)}],
                        temperature=0.3,
                        max_tokens=300,
                    )
                    st.write(resp.choices[0].message.content)
                except Exception as e:
                    st.error(f"OpenAI error: {e}")

    # -------- Diagnostics --------
    with tabs[5]:
        st.subheader("Data Diagnostics")
        st.write("Non‑missing values per series (higher is better):")
        st.json(diag)
        with st.expander("Raw data"):
            st.dataframe(df, use_container_width=True)
        st.caption("If EPS is missing or odd on Alpha Vantage, switch to FMP. Financials for some tickers vary by provider, fiscal year alignment, or field naming.")
