import streamlit as st
import openai
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yfinance as yf
import io
import contextlib
import re
import sys
from io import StringIO

# --- App Config ---
st.set_page_config(page_title="QuantGPT", layout="wide", initial_sidebar_state="expanded")
st.title("🤖 QuantGPT - Your AI Quant Assistant")

# --- Strategy Presets ---
presets = {
    "Select a preset...": "",
    "MA Crossover (50/200) on TSLA": "Backtest a 50/200-day moving average crossover strategy on TSLA from 2010-01-01 to 2022-12-31 using yfinance, pandas, numpy, and matplotlib.",
    "RSI (14) Mean Reversion on AAPL": "Backtest a strategy that buys when the 14-day RSI goes below 30 and sells when it goes above 70 on AAPL from 2015-01-01 to 2022-12-31 using yfinance, pandas, numpy, and matplotlib.",
    "Bollinger Bands (20,2) on MSFT": "Backtest a mean reversion strategy using Bollinger Bands (20, 2) on MSFT from 2012-01-01 to 2022-12-31 using yfinance, pandas, numpy, and matplotlib.",
    "Buy & Hold on NVDA": "Backtest a buy-and-hold strategy on NVDA from 2010-01-01 to 2022-12-31 using yfinance, pandas, numpy, and matplotlib."
}

# --- Sidebar / Onboarding ---
st.sidebar.header("Welcome to QuantGPT!")
st.sidebar.write("Type a trading strategy in plain English or choose a preset below.")
choice = st.sidebar.selectbox("Try a preset strategy:", list(presets.keys()))

# API key input
api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if not api_key:
    st.sidebar.warning("Enter your OpenAI API key to generate code.")

# User input area
user_input = st.text_area("Ask for a trading strategy:", value=presets.get(choice, ""), height=120)

# Validate input
if not user_input.strip():
    st.info("🔍 Please enter a strategy or select a preset to continue.")
    st.stop()

# Initialize OpenAI client - FIXED: Use the user-provided API key
client = None
if api_key:
    try:
        client = openai.OpenAI(api_key=api_key)
    except Exception as e:
        st.sidebar.error(f"❌ Invalid API key: {e}")
        st.stop()
else:
    st.stop()

# Generate code
if st.button("Generate Code"):
    with st.spinner("Generating strategy code..."):
        try:
            prompt_system = """You are a quant expert. Return only Python code using yfinance, pandas, numpy, and matplotlib to backtest the user's request.

IMPORTANT REQUIREMENTS:
1. Always create a pandas DataFrame named 'data' with stock price data
2. Always include a 'positions' column indicating buy/sell signals (1 for long, 0 for flat, -1 for short)
3. Use yfinance to fetch data: data = yf.download('SYMBOL', start='YYYY-MM-DD', end='YYYY-MM-DD', auto_adjust=False)
4. Do not use print() statements, use comments instead
5. Make sure the 'positions' column is properly shifted to avoid look-ahead bias
6. AVOID creating columns with DataFrame assignment errors - use .loc or assign one column at a time
7. Do NOT create columns like 'strategy_returns' - stick to 'positions' and basic indicators

Example structure:
```python
# Fetch data
data = yf.download('TSLA', start='2010-01-01', end='2022-12-31', auto_adjust=False)

# Calculate indicators (e.g., moving averages)
data['SMA_50'] = data['Close'].rolling(50).mean()
data['SMA_200'] = data['Close'].rolling(200).mean()

# Generate signals (1 for buy, 0 for sell)
data['signal'] = np.where(data['SMA_50'] > data['SMA_200'], 1, 0)
data['positions'] = data['signal'].shift(1).fillna(0)
```"""

            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": prompt_system},
                    {"role": "user", "content": user_input}
                ]
            )
            st.session_state["code"] = response.choices[0].message.content
        except Exception as e:
            st.error(f"❌ Error generating code: {e}")

# Display and run code
if "code" in st.session_state:
    st.subheader("📄 Generated Code")
    st.code(st.session_state["code"], language="python")

    if st.button("Run Backtest"):
        raw = st.session_state["code"]

        # Clean code blocks - IMPROVED: Handle multiple formats
        code_patterns = [
            r"```python\n(.*?)```",
            r"```\n(.*?)```",
            r"```python(.*?)```",
            r"```(.*?)```"
        ]

        clean_code = raw
        for pattern in code_patterns:
            m = re.search(pattern, raw, flags=re.S)
            if m:
                clean_code = m.group(1)
                break

        clean_code = clean_code.strip("`\n ")

        # Add some preprocessing to fix common issues
        clean_code = clean_code.replace("auto_adjust=True", "auto_adjust=False")

        st.text_area("Cleaned Code", clean_code, height=200)

        # Execute code - IMPROVED: Better error handling
        exec_globals = {"yf": yf, "pd": pd, "np": np, "plt": plt}
        exec_locals = {}
        out, err = StringIO(), StringIO()

        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                exec(clean_code, exec_globals, exec_locals)
            except Exception as e:
                st.error(f"❌ Execution error: {e}")
                error_details = err.getvalue()
                if error_details:
                    st.error(f"Error details: {error_details}")

                # Show helpful debugging info
                st.subheader("🔍 Debug Information")
                st.text("Generated code that caused the error:")
                st.code(clean_code, language="python")

                # Provide specific error guidance
                error_str = str(e)
                if "Cannot set a DataFrame with multiple columns" in error_str:
                    st.info(
                        "💡 **Common Fix**: The code is trying to assign multiple columns to a single column. This usually happens with pandas operations.")
                elif "strategy_returns" in error_str:
                    st.info(
                        "💡 **Common Fix**: Issue with 'strategy_returns' column assignment. Try regenerating the code.")
                elif "yf.download" in error_str:
                    st.info("💡 **Common Fix**: Data download issue. Check ticker symbol and date range.")

                st.stop()

        # Show any output from the code
        output = out.getvalue()
        if output:
            st.text("Code Output:")
            st.text(output)

        # Validate data DataFrame
        data = exec_locals.get("data")
        if data is None:
            st.error("❌ No DataFrame named 'data' was created by your code.")
            st.info("Available variables: " + ", ".join(exec_locals.keys()))
            st.stop()

        if not isinstance(data, pd.DataFrame):
            st.error(f"❌ Variable 'data' is not a DataFrame. It's a {type(data)}")
            st.stop()

        if data.empty:
            st.error("❌ The 'data' DataFrame is empty.")
            st.stop()

        st.success(f"✅ Data loaded successfully! Shape: {data.shape}")
        st.write("Data columns:", list(data.columns))

        # Data validation warnings
        if len(data) < 252:
            st.warning("⚠️ Less than 1 year of data — results may be unreliable.")

        if data.shape[1] < 3:
            st.warning("⚠️ Fewer than 3 columns in data — verify your data download.")

        # Ensure price column exists
        if "Close" not in data.columns:
            if "Adj Close" in data.columns:
                data["Close"] = data["Adj Close"]
                st.info("ℹ️ Using 'Adj Close' as 'Close' price.")
            else:
                data["Close"] = data.iloc[:, 0]
                st.info("ℹ️ Using first column as 'Close' price.")

        # Ensure positions column - IMPROVED: Better derivation logic
        if "positions" not in data.columns:
            st.info("📋 'positions' column not found. Attempting to derive from signals...")

            if "Signal" in data.columns:
                data["positions"] = data["Signal"].shift(1).fillna(0)
                st.info("✅ Derived positions from 'Signal' column")
            elif "signal" in data.columns:
                data["positions"] = data["signal"].shift(1).fillna(0)
                st.info("✅ Derived positions from 'signal' column")
            elif "Position" in data.columns:
                data["positions"] = data["Position"].shift(1).fillna(0)
                st.info("✅ Derived positions from 'Position' column")
            elif "SMA_50" in data.columns and "SMA_200" in data.columns:
                data["positions"] = (data["SMA_50"] > data["SMA_200"]).astype(float).shift(1).fillna(0)
                st.info("✅ Derived positions from SMA crossover")
            else:
                st.warning("⚠️ Could not derive 'positions'. Creating buy-and-hold strategy.")
                data["positions"] = 1.0  # Buy and hold
                st.info("Using buy-and-hold strategy (positions = 1)")

        # Calculate returns
        data["market_ret"] = data["Close"].pct_change()
        data["strat_ret"] = data["positions"].shift(1) * data["market_ret"]
        valid = data.dropna()

        if len(valid) == 0:
            st.error("❌ No valid data points after removing NaN values.")
            st.stop()

        # Performance metrics
        total_ret = (1 + valid["strat_ret"]).cumprod().iloc[-1] - 1
        ann_ret = valid["strat_ret"].mean() * 252
        ann_vol = valid["strat_ret"].std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol != 0 else float("nan")
        cum = (1 + valid["strat_ret"]).cumprod()
        max_dd = (cum / cum.cummax() - 1).min()

        # Market benchmark
        market_total_ret = (1 + valid["market_ret"]).cumprod().iloc[-1] - 1
        market_ann_ret = valid["market_ret"].mean() * 252

        # Display metrics
        st.subheader("📈 Performance Metrics")
        cols = st.columns(3)
        cols[0].metric("Total Return", f"{total_ret:.2%}")
        cols[1].metric("Annualized Return", f"{ann_ret:.2%}")
        cols[2].metric("Sharpe Ratio", f"{sharpe:.2f}")

        # Additional metrics row
        cols2 = st.columns(3)
        cols2[0].metric("Market Return", f"{market_total_ret:.2%}")
        cols2[1].metric("Annualized Volatility", f"{ann_vol:.2%}")
        cols2[2].metric("Max Drawdown", f"{max_dd:.2%}")

        # Equity curve
        st.subheader("📊 Equity Curve vs Market")
        fig, ax = plt.subplots(figsize=(12, 6))

        strategy_cum = (1 + valid["strat_ret"]).cumprod()
        market_cum = (1 + valid["market_ret"]).cumprod()

        strategy_cum.plot(ax=ax, label="Strategy", linewidth=2)
        market_cum.plot(ax=ax, label="Market (Buy & Hold)", linewidth=2, alpha=0.7)

        ax.set_title("Strategy vs Market Performance", fontsize=14, fontweight='bold')
        ax.set_ylabel("Cumulative Returns")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        # Additional plots if signals exist
        if "Signal" in data.columns or "signal" in data.columns:
            st.subheader("📊 Price and Signals")
            signal_col = "Signal" if "Signal" in data.columns else "signal"

            fig2, ax2 = plt.subplots(figsize=(12, 6))

            # Plot price
            ax2.plot(data.index, data["Close"], label="Close Price", linewidth=1)

            # Plot buy/sell signals
            buy_signals = data[data[signal_col] == 1]
            sell_signals = data[data[signal_col] == 0]

            if not buy_signals.empty:
                ax2.scatter(buy_signals.index, buy_signals["Close"],
                            color='green', marker='^', s=100, label='Buy Signal', alpha=0.7)

            if not sell_signals.empty:
                ax2.scatter(sell_signals.index, sell_signals["Close"],
                            color='red', marker='v', s=100, label='Sell Signal', alpha=0.7)

            ax2.set_title("Price Chart with Trading Signals", fontsize=14, fontweight='bold')
            ax2.set_ylabel("Price")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            st.pyplot(fig2)

        # Download CSV
        csv = data.to_csv()
        st.download_button("📥 Download Results CSV", data=csv, file_name="backtest_results.csv", mime="text/csv")

        # Sample data
        st.subheader("📋 Sample Data")
        st.dataframe(valid.tail(10))
