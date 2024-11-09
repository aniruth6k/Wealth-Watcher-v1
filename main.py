import os
import streamlit as st
import json
import openai
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
import newsapi
from newsapi import NewsApiClient
import re
import pandas as pd

load_dotenv()

st.set_page_config(page_title="Wealth Watcher", page_icon=":chart_with_upwards_trend:")
st.title("Wealth Watcher")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a financial advisor. How can I help you?")
    ]


def get_nse_ticker(ticker):
    return ticker + ".NS"


def get_stock_price(ticker):
    data = yf.Ticker(ticker).history(period="1y")
    if not data.empty:
        return str(data["Close"].iloc[-1])
    else:
        return "Unable to fetch stock price data for the given ticker symbol."


def calculate_SMA(ticker, window):
    data = yf.Ticker(ticker).history(period="1y")
    return str(data["Close"].ewm(span=window, adjust=False).mean().iloc[-1])


def calculate_EMA(ticker, window):
    data = yf.Ticker(ticker).history(period="1y")
    return str(data["Close"].ewm(span=window, adjust=False).mean().iloc[-1])


def get_stock_open_price(ticker):
    today = yf.Ticker(ticker).history(period="1d")
    if not today.empty:
        open_price = str(today["Open"].iloc[-1])
        return f"The opening price of {ticker} today was {open_price}."
    else:
        return "Unable to fetch stock price data for the given ticker symbol."


def get_stock_price_movement(ticker):
    today = yf.Ticker(ticker).history(period="1d")
    if not today.empty:
        open_price = today["Open"].iloc[-1]
        close_price = today["Close"].iloc[-1]
        if close_price > open_price:
            direction = "up"
        elif close_price < open_price:
            direction = "down"
        else:
            direction = "unchanged"
        return f"{ticker} went {direction} today."
    else:
        return "Unable to fetch stock price data for the given ticker symbol."


def get_stock_trading_volume(ticker):
    data = yf.Ticker(ticker).history(period="1d")
    if not data.empty:
        volume = str(data["Volume"].iloc[-1])
        return f"The trading volume of {ticker} today was {volume}."
    else:
        return "Unable to fetch stock trading volume data for the given ticker symbol."


def get_52_week_high(ticker):
    data = yf.Ticker(ticker).info
    if "fiftyTwoWeekHigh" in data:
        high_price = data["fiftyTwoWeekHigh"]
        return f"The 52-week high price for {ticker} is {high_price}."
    else:
        return "Unable to fetch 52-week high data for the given ticker symbol."


def get_52_week_low(ticker):
    data = yf.Ticker(ticker).info
    if "fiftyTwoWeekLow" in data:
        low_price = data["fiftyTwoWeekLow"]
        return f"The 52-week low price for {ticker} is {low_price}."
    else:
        return "Unable to fetch 52-week low data for the given ticker symbol."


def plot_stock_price(ticker, period=None, year=None, start_year=None, end_year=None):
    try:
        if start_year and end_year:
            start_date = f"{start_year}-01-01"
            end_date = f"{end_year}-12-31"
            data = yf.Ticker(ticker).history(start=start_date, end=end_date)
            title = f"{ticker} Stock Price Chart from {start_year} to {end_year}"
        elif year:
            start_date = f"{year}-01-01"
            end_date = f"{year}-12-31"
            data = yf.Ticker(ticker).history(start=start_date, end=end_date)
            title = f"{ticker} Stock Price Chart for {year}"
        else:
            if period:
                period_str = f"{period}y"
            else:
                period_str = "1y"
            data = yf.Ticker(ticker).history(period=period_str)
            title = f"{ticker} {period}-Year Stock Price Chart"

        if not data.empty:
            plt.figure(figsize=(10, 6))
            plt.plot(data.index, data["Close"])
            plt.title(title)
            plt.xlabel("Date")
            plt.ylabel("Price (USD)")
            plt.savefig("stock.png", bbox_inches="tight")
            return "The stock price chart has been generated."
        else:
            return "Unable to fetch stock price data for the given ticker symbol."
    except Exception as e:
        return f"An error occurred: {str(e)}"


def compare_stock_sp500(ticker):
    try:
        stock_data = yf.Ticker(ticker).history(period="1y")
        sp500_data = yf.Ticker("^GSPC").history(period="1y")

        if not stock_data.empty and not sp500_data.empty:
            stock_return = (
                (stock_data["Close"][-1] - stock_data["Close"][0])
                / stock_data["Close"][0]
                * 100
            )
            sp500_return = (
                (sp500_data["Close"][-1] - sp500_data["Close"][0])
                / sp500_data["Close"][0]
                * 100
            )

            if stock_return > sp500_return:
                return f"{ticker} outperformed the S&P 500 over the last year, with a return of {stock_return:.2f}% compared to {sp500_return:.2f}% for the S&P 500."
            elif stock_return < sp500_return:
                return f"{ticker} underperformed the S&P 500 over the last year, with a return of {stock_return:.2f}% compared to {sp500_return:.2f}% for the S&P 500."
            else:
                return f"{ticker} performed on par with the S&P 500 over the last year, both with a return of {stock_return:.2f}%."
        else:
            return "Unable to fetch stock price data for the given ticker symbol or the S&P 500 index."
    except Exception as e:
        return f"An error occurred: {str(e)}"


def get_pe_ratio(ticker):
    data = yf.Ticker(ticker).info
    if "trailingPE" in data:
        pe_ratio = data["trailingPE"]
        return f"The P/E ratio for {ticker} is {pe_ratio}."
    else:
        return f"Unable to fetch P/E ratio data for {ticker}."


def get_dividend_info(ticker):
    data = yf.Ticker(ticker).info
    if "dividendYield" in data and data["dividendYield"] is not None:
        dividend_yield = data["dividendYield"] * 100
        return f"{ticker} pays a dividend with a yield of {dividend_yield:.2f}%."
    elif "dividendYield" in data and data["dividendYield"] is None:
        return f"{ticker} does not pay a dividend."
    else:
        return f"Unable to fetch dividend information for {ticker}."


def get_market_cap(ticker):
    data = yf.Ticker(ticker).info
    if "marketCap" in data:
        market_cap = data["marketCap"]
        return f"The market capitalization of {ticker} is ${market_cap:,.2f}."
    else:
        return f"Unable to fetch market capitalization data for {ticker}."


def get_stock_type(ticker):
    data = yf.Ticker(ticker).info
    if (
        "trailingPE" in data
        and "forwardPE" in data
        and "priceToBook" in data
        and "industry" in data
    ):
        trailing_pe = data["trailingPE"]
        forward_pe = data["forwardPE"]
        price_to_book = data["priceToBook"]
        industry = data["industry"]

        if (
            trailing_pe > 20
            and forward_pe > 20
            and industry.lower()
            in ["technology", "software", "internet", "biotechnology"]
        ):
            return f"{ticker} is considered a growth stock based on its high P/E ratios and industry ({industry})."
        elif price_to_book < 1.5 and industry.lower() in [
            "banking",
            "insurance",
            "energy",
            "utilities",
        ]:
            return f"{ticker} is considered a value stock based on its low price-to-book ratio and industry ({industry})."
        else:
            return f"{ticker} does not clearly fit the criteria for a growth or value stock based on the available information."
    else:
        return f"Unable to determine the stock type for {ticker} due to missing data."


def get_top_company_risks(ticker):
    try:
        company_info = yf.Ticker(ticker).info
        if "longBusinessSummary" in company_info:
            business_summary = company_info["longBusinessSummary"]
            risk_factors = []
            if "riskFactors" in company_info and company_info["riskFactors"]:
                risk_factors = company_info["riskFactors"][
                    :3
                ]  # Get the top 3 risk factors

            if risk_factors:
                risk_summary = "\n".join(
                    [f"{idx + 1}. {risk}" for idx, risk in enumerate(risk_factors)]
                )
                return f"According to the available information, the top 3 risks facing {ticker} are:\n\n{risk_summary}"
            else:
                return f"Unfortunately, I could not find specific risk factor information for {ticker}. However, here is a summary of the business:\n\n{business_summary}"
        else:
            return f"Unable to retrieve business summary or risk factor information for {ticker}."
    except Exception as e:
        return f"An error occurred: {str(e)}"


def get_company_competitors(ticker):
    try:
        company_info = yf.Ticker(ticker).info
        if "competitors" in company_info and company_info["competitors"]:
            competitors = company_info["competitors"]
            return f"According to the available information, the main competitors of {ticker} are:\n\n{', '.join(competitors)}"
        else:
            return f"Unfortunately, I could not find specific information about the main competitors of {ticker}."
    except Exception as e:
        return f"An error occurred: {str(e)}"


def get_next_earnings_date(ticker):
    try:
        company_info = yf.Ticker(ticker).calendar
        if company_info.empty:
            return f"Unable to find the next earnings date for {ticker}."
        else:
            earnings_dates = company_info[company_info["Event"] == "earnings"]
            if earnings_dates.empty:
                return f"No upcoming earnings dates found for {ticker}."
            else:
                next_earnings_date = earnings_dates.iloc[0]["Earnings Date"]
                formatted_date = datetime.utcfromtimestamp(next_earnings_date).strftime(
                    "%Y-%m-%d"
                )
                return f"The next earnings report for {ticker} is expected on {formatted_date}."
    except Exception as e:
        return f"An error occurred: {str(e)}"


def check_50day_ma(ticker):
    try:
        data = yf.Ticker(ticker).history(period="60d")
        if data.empty:
            return f"Unable to fetch stock price data for {ticker}."
        else:
            current_price = data["Close"].iloc[-1]
            ma_50 = data["Close"].rolling(window=50).mean().iloc[-1]
            if current_price > ma_50:
                status = "above"
            elif current_price < ma_50:
                status = "below"
            else:
                status = "equal to"
            return f"The current stock price of {ticker} is {status} its 50-day moving average."
    except Exception as e:
        return f"An error occurred: {str(e)}"


def calculate_RSI(ticker, window=14):
    try:
        data = yf.Ticker(ticker).history(period="60d")
        if data.empty:
            return f"Unable to fetch stock price data for {ticker}."
        else:
            delta = data["Close"].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window).mean()
            avg_loss = loss.rolling(window).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            return f"The current {window}-day RSI for {ticker} is {current_rsi:.2f}."
    except Exception as e:
        return f"An error occurred: {str(e)}"


def get_stock_trend(ticker, period="60d"):
    try:
        data = yf.Ticker(ticker).history(period=period)
        if data.empty:
            return f"Unable to fetch stock price data for {ticker}."
        else:
            start_price = data["Close"].iloc[0]
            end_price = data["Close"].iloc[-1]
            if end_price > start_price:
                trend = "uptrend"
            elif end_price < start_price:
                trend = "downtrend"
            else:
                trend = "sideways"
            return f"Based on the {period} price data, the stock {ticker} appears to be in a {trend}."
    except Exception as e:
        return f"An error occurred: {str(e)}"


def check_head_shoulders_pattern(ticker, period="1y"):
    try:
        data = yf.Ticker(ticker).history(period=period)
        if data.empty:
            return f"Unable to fetch stock price data for {ticker}."
        else:
            close_prices = data["Close"].values
            highs = data["High"].values
            lows = data["Low"].values

            # Find potential head and shoulders pattern
            head_found = False
            left_shoulder_found = False
            right_shoulder_found = False
            for i in range(len(close_prices) - 2):
                # Left shoulder
                if (
                    not left_shoulder_found
                    and close_prices[i] < close_prices[i + 1] > close_prices[i + 2]
                ):
                    left_shoulder_idx = i + 1
                    left_shoulder_found = True
                    continue

                # Head
                if (
                    left_shoulder_found
                    and not head_found
                    and close_prices[i] < close_prices[i + 1] > close_prices[i + 2]
                ):
                    head_idx = i + 1
                    if highs[head_idx] > highs[left_shoulder_idx]:
                        head_found = True
                    continue

                # Right shoulder
                if (
                    head_found
                    and not right_shoulder_found
                    and close_prices[i] < close_prices[i + 1] > close_prices[i + 2]
                ):
                    right_shoulder_idx = i + 1
                    if (
                        highs[right_shoulder_idx] < highs[head_idx]
                        and lows[right_shoulder_idx] > lows[left_shoulder_idx]
                    ):
                        right_shoulder_found = True
                        break

            if left_shoulder_found and head_found and right_shoulder_found:
                return f"Based on the {period} price data, the stock {ticker} has formed a head and shoulders pattern, which is a potential bearish reversal signal."
            else:
                return f"Based on the {period} price data, the stock {ticker} has not formed a head and shoulders pattern."
    except Exception as e:
        return f"An error occurred: {str(e)}"


def plot_stock_with_macd(ticker, period="1y"):
    try:
        data = yf.Ticker(ticker).history(period=period)
        if data.empty:
            return f"Unable to fetch stock price data for {ticker}."
        else:
            # Calculate MACD
            short_window = 12
            long_window = 26
            signal_window = 9

            data["EMA12"] = data["Close"].ewm(span=short_window, adjust=False).mean()
            data["EMA26"] = data["Close"].ewm(span=long_window, adjust=False).mean()
            data["MACD"] = data["EMA12"] - data["EMA26"]
            data["Signal"] = data["MACD"].ewm(span=signal_window, adjust=False).mean()
            data["Histogram"] = data["MACD"] - data["Signal"]

            # Create subplots
            fig, (ax1, ax2) = plt.subplots(
                2,
                1,
                figsize=(12, 8),
                sharex=True,
                gridspec_kw={"height_ratios": [3, 1]},
            )

            # Plot stock price
            ax1.plot(data.index, data["Close"], label="Close Price")
            ax1.set_title(f"{ticker} Stock Price with MACD")
            ax1.set_ylabel("Price (USD)")
            ax1.legend()
            ax1.grid()

            # Plot MACD
            ax2.plot(data.index, data["MACD"], label="MACD")
            ax2.plot(data.index, data["Signal"], label="Signal")
            ax2.bar(
                data.index,
                data["Histogram"],
                label="Histogram",
                alpha=0.3,
                color="gray",
            )
            ax2.axhline(y=0, color="r", linestyle="--", alpha=0.5)
            ax2.set_xlabel("Date")
            ax2.set_ylabel("MACD")
            ax2.legend()
            ax2.grid()

            plt.tight_layout()
            plt.savefig("stock_with_macd.png", bbox_inches="tight")
            plt.close()

            return f"The stock price chart with MACD for {ticker} has been generated."
    except Exception as e:
        return f"An error occurred: {str(e)}"


def compare_valuations(ticker1, ticker2):
    try:
        # Fetch data for both stocks
        stock1 = yf.Ticker(ticker1)
        stock2 = yf.Ticker(ticker2)

        # Get financial metrics
        info1 = stock1.info
        info2 = stock2.info

        # Key market valuation metrics
        metrics = [
            (
                "Market Cap (B)",
                "marketCap",
                lambda x: f"${x / 1e9:.2f}B" if x else "N/A",
            ),
            (
                "Enterprise Value (B)",
                "enterpriseValue",
                lambda x: f"${x / 1e9:.2f}B" if x else "N/A",
            ),
            ("Stock Price", "currentPrice", lambda x: f"${x:.2f}" if x else "N/A"),
            ("52-Week High", "fiftyTwoWeekHigh", lambda x: f"${x:.2f}" if x else "N/A"),
            ("52-Week Low", "fiftyTwoWeekLow", lambda x: f"${x:.2f}" if x else "N/A"),
        ]

        response = f"Market Valuation: {ticker1} vs {ticker2}\n\n"
        response += f"{'Metric':<20} {ticker1:<15} {ticker2:<15}\n"
        response += "-" * 50 + "\n"

        for display_name, key, transform in metrics:
            value1 = transform(info1.get(key, "N/A"))
            value2 = transform(info2.get(key, "N/A"))
            response += f"{display_name:<20} {value1:<15} {value2:<15}\n"

        # Add a brief explanation of each metric
        response += "\nMetric Explanations:\n"
        response += "- Market Cap: Total value of all outstanding shares. Bigger = larger company.\n"
        response += "- Enterprise Value: Market Cap + Debt - Cash. Often seen as a company's 'takeover price'.\n"
        response += "- Stock Price: Current trading price per share.\n"
        response += "- 52-Week Range: Lowest and highest prices in the past year, showing volatility and trends.\n"

        return response
    except Exception as e:
        return f"An error occurred: {str(e)}"


def compare_stocks(ticker1, ticker2):
    try:
        # Fetch data for both stocks
        stock1 = yf.Ticker(ticker1)
        stock2 = yf.Ticker(ticker2)

        # Get historical data for the past year
        data1 = stock1.history(period="1y")
        data2 = stock2.history(period="1y")

        # Calculate 1-year returns
        return1 = (data1["Close"][-1] - data1["Close"][0]) / data1["Close"][0] * 100
        return2 = (data2["Close"][-1] - data2["Close"][0]) / data2["Close"][0] * 100

        # Get financial metrics
        info1 = stock1.info
        info2 = stock2.info

        pe1 = info1.get("trailingPE", "N/A")
        pe2 = info2.get("trailingPE", "N/A")

        pb1 = info1.get("priceToBook", "N/A")
        pb2 = info2.get("priceToBook", "N/A")

        eps_growth1 = info1.get("earningsGrowth", "N/A")
        eps_growth2 = info2.get("earningsGrowth", "N/A")

        div_yield1 = info1.get("dividendYield", "N/A")
        div_yield2 = info2.get("dividendYield", "N/A")

        # Format the response
        response = f"Comparing {ticker1} and {ticker2}:\n\n"
        response += (
            f"1-Year Return:\n{ticker1}: {return1:.2f}% | {ticker2}: {return2:.2f}%\n\n"
        )
        response += f"P/E Ratio:\n{ticker1}: {pe1} | {ticker2}: {pe2}\n\n"
        response += f"Price-to-Book Ratio:\n{ticker1}: {pb1} | {ticker2}: {pb2}\n\n"
        response += (
            f"EPS Growth:\n{ticker1}: {eps_growth1} | {ticker2}: {eps_growth2}\n\n"
        )
        response += (
            f"Dividend Yield:\n{ticker1}: {div_yield1} | {ticker2}: {div_yield2}\n\n"
        )

        # Make a simple recommendation based on 1-year return
        better_buy = ticker1 if return1 > return2 else ticker2
        response += f"Based on 1-year return, {better_buy} has performed better. However, this is just one metric, and you should consider all factors, including your investment goals, risk tolerance, and a deeper analysis of each company's financials and future prospects before making a decision."

        return response
    except Exception as e:
        return f"An error occurred: {str(e)}"


def get_sector_stocks(sector):
    try:
        # Use yfinance's built-in list or use a more comprehensive list from a file
        tickers = yf.Tickers(" ".join(yf.Ticker(f"^{sector}").components.keys()))
        sector_stocks = [
            ticker
            for ticker, info in tickers.info.items()
            if info.get("sector") == sector
        ]
        return sector_stocks[:50]  # Limit to top 50 to avoid rate limits
    except Exception as e:
        return []


def get_top_sector_stocks(sector, period="1mo", top_n=10):
    try:
        # Map common sector names to their yfinance symbols (using broader market ETFs)
        sector_map = {
            "Technology": {"etf": "QQQ", "name": "NASDAQ-100 (Tech-heavy)"},
            "Healthcare": {"etf": "XLV", "name": "Health Care Select"},
            "Financials": {"etf": "XLF", "name": "Financial Select"},
            "Consumer Discretionary": {
                "etf": "XLY",
                "name": "Consumer Discretionary Select",
            },
            "Industrials": {"etf": "XLI", "name": "Industrial Select"},
            "Energy": {"etf": "XLE", "name": "Energy Select"},
            "Consumer Staples": {"etf": "XLP", "name": "Consumer Staples Select"},
            "Materials": {"etf": "XLB", "name": "Materials Select"},
            "Communication Services": {
                "etf": "XLC",
                "name": "Communication Services Select",
            },
            "Utilities": {"etf": "XLU", "name": "Utilities Select"},
            "Real Estate": {"etf": "XLRE", "name": "Real Estate Select"},
        }

        if sector not in sector_map:
            return f"I apologize, but I don't have specific data for the '{sector}' sector. I can provide insights on these sectors: {', '.join(sector_map.keys())}."

        etf_info = sector_map[sector]
        etf_ticker = etf_info["etf"]
        etf_name = etf_info["name"]

        # For Tech sector, use the top components of NASDAQ-100
        if sector == "Technology":
            sector_stocks = [
                "AAPL",
                "MSFT",
                "AMZN",
                "NVDA",
                "META",
                "GOOGL",
                "GOOG",
                "TSLA",
                "AVGO",
                "CSCO",
                "ADBE",
                "INTC",
                "NFLX",
                "PYPL",
                "CMCSA",
                "AMD",
                "QCOM",
                "INTU",
                "TXN",
                "AMAT",
            ]
        else:
            # Fetch top holdings from the sector ETF
            sector_etf = yf.Ticker(etf_ticker)
            sector_stocks = [
                stock[0] for stock in sector_etf.info.get("holdings", [])[:20]
            ]

        if not sector_stocks:
            return f"I'm having trouble fetching data for stocks in the {sector} sector. This might be due to temporary API limitations. Please try again later or consider researching top companies in this sector through other financial websites."

        # Fetch and calculate returns
        returns = {}
        for ticker in sector_stocks:
            try:
                data = yf.Ticker(ticker).history(period=period)
                if not data.empty and len(data) > 1:
                    start_price = data["Close"].iloc[0]
                    end_price = data["Close"].iloc[-1]
                    returns[ticker] = (end_price - start_price) / start_price * 100
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")

        if not returns:
            return f"I tried to fetch data for top companies in the {sector} sector, but I encountered issues. This is likely due to temporary API limitations. Please try again later or consider using other financial research tools."

        # Sort and select top performers
        top_stocks = sorted(returns.items(), key=lambda x: x[1], reverse=True)[:top_n]

        # Prepare the response
        time_descriptions = {
            "1mo": "month",
            "3mo": "3 months",
            "6mo": "6 months",
            "1y": "year",
        }
        period_desc = time_descriptions.get(period, period)

        response = f"Top {len(top_stocks)} Performing Stocks in the {sector} Sector (Last {period_desc}):\n\n"
        for i, (ticker, ret) in enumerate(top_stocks, 1):
            try:
                info = yf.Ticker(ticker).info
                name = info.get("longName", ticker)
                response += f"{i}. {name} ({ticker})\n   - Return: {ret:.2f}%\n"
            except Exception as e:
                print(f"Error fetching info for {ticker}: {e}")
                response += f"{i}. {ticker}\n   - Return: {ret:.2f}%\n"

        response += f"\nThese stocks have outperformed within the {etf_name} ETF, which is a good proxy for the {sector} sector. However, remember:\n\n"
        response += "1. Past performance doesn't guarantee future results.\n"
        response += "2. High returns often come with high risk.\n"
        response += "3. A well-diversified portfolio is key to managing risk.\n"
        response += "4. This list is based on recent performance. For a complete picture, also consider:\n"
        response += "   a) Long-term performance\n"
        response += "   b) Company financials\n"
        response += "   c) Market position\n"
        response += "   d) Growth prospects\n"
        response += "   e) Economic trends\n\n"
        response += "Always do thorough research or consult a financial advisor before investing."

        return response
    except Exception as e:
        return f"I encountered an error while trying to fetch stock data: {str(e)}. This is likely due to temporary API limitations or network issues. Please try again later or consider using other financial research tools for now."


def get_potential_growth_stocks(industry, top_n=5):
    """
    Identify potential growth stocks in a specified industry.

    :param industry: The target industry (e.g., 'Technology', 'Biotechnology', 'Software')
    :param top_n: Number of top stocks to return (default is 5)
    :return: A list of potential growth stocks with their metrics
    """
    # List of major stock indices to search through
    indices = [
        "^GSPC",
        "^IXIC",
        "^DJI",
    ]  # S&P 500, Nasdaq Composite, Dow Jones Industrial Average

    potential_stocks = []

    for index in indices:
        tickers = yf.Ticker(index).info["components"]
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info

                # Check if the stock belongs to the target industry
                if "industry" in info and info["industry"] == industry:
                    # Criteria for growth stocks
                    trailing_pe = info.get("trailingPE", float("inf"))
                    forward_pe = info.get("forwardPE", float("inf"))
                    peg_ratio = info.get("pegRatio", float("inf"))
                    earnings_growth = info.get("earningsGrowth", 0)
                    revenue_growth = info.get("revenueGrowth", 0)
                    beta = info.get("beta", 0)

                    # High P/E ratios, high growth rates, and higher beta are typical of growth stocks
                    if (
                        (trailing_pe > 25 or forward_pe > 25)
                        and (earnings_growth > 0.15 or revenue_growth > 0.15)
                        and beta > 1
                    ):
                        potential_stocks.append(
                            {
                                "Ticker": ticker,
                                "Company": info.get("longName", "N/A"),
                                "Industry": industry,
                                "Trailing P/E": trailing_pe,
                                "Forward P/E": forward_pe,
                                "PEG Ratio": peg_ratio,
                                "Earnings Growth": earnings_growth,
                                "Revenue Growth": revenue_growth,
                                "Beta": beta,
                            }
                        )
            except Exception as e:
                pass  # Skip any stocks that cause errors

    # Sort stocks by a composite growth score (you can adjust the weights)
    for stock in potential_stocks:
        stock["Growth Score"] = (
            0.3
            * (
                stock["Earnings Growth"]
                if stock["Earnings Growth"] != float("inf")
                else 0
            )
            + 0.3
            * (
                stock["Revenue Growth"]
                if stock["Revenue Growth"] != float("inf")
                else 0
            )
            + 0.2
            * (
                1 / stock["PEG Ratio"]
                if stock["PEG Ratio"] not in [0, float("inf")]
                else 0
            )
            + 0.2 * (stock["Beta"] if stock["Beta"] != float("inf") else 1)
        )

    # Sort by the growth score in descending order and select top N
    potential_stocks.sort(key=lambda x: x["Growth Score"], reverse=True)
    top_stocks = potential_stocks[:top_n]

    return top_stocks


# def manage_portfolio(action, ticker=None, shares=1):
#    if 'portfolio' not in st.session_state:
#        st.session_state.portfolio = {}
#
#    if action == 'add' and ticker:
#        if ticker in st.session_state.portfolio:
#            st.session_state.portfolio[ticker] += shares
#        else:
#            st.session_state.portfolio[ticker] = shares
#        return f"{shares} share(s) of {ticker} have been added to your portfolio."
#    elif action == 'remove' and ticker:
#        if ticker in st.session_state.portfolio:
#            if shares >= st.session_state.portfolio[ticker]:
#                del st.session_state.portfolio[ticker]
#                return f"All shares of {ticker} have been removed from your portfolio."
#            else:
#                st.session_state.portfolio[ticker] -= shares
#                return f"{shares} share(s) of {ticker} have been removed from your portfolio."
#        else:
#            return f"{ticker} is not in your portfolio."
#    elif action == 'list':
#        if not st.session_state.portfolio:
#            return "Your portfolio is empty."
#        else:
#            stocks = [f"{ticker} ({shares} share{'s' if shares > 1 else ''})" for ticker, shares in st.session_state.portfolio.items()]
#            return "Your portfolio contains: " + ", ".join(stocks)
#    else:
#        return "Invalid action. Use 'add', 'remove', or 'list'."
#
# def display_portfolio():
#    if 'portfolio' not in st.session_state or not st.session_state.portfolio:
#        return "Your portfolio is empty. Please add some stocks first."
#
#    total_value = 0
#    portfolio_summary = []
#
#    for ticker, shares in st.session_state.portfolio.items():
#        try:
#            stock = yf.Ticker(ticker)
#            data = stock.history(period='1d')
#            if not data.empty:
#                current_price = data['Close'].iloc[-1]
#                stock_value = current_price * shares
#                total_value += stock_value
#
#                # Use the full company name if available, otherwise use the ticker
#                info = stock.info
#                company_name = info.get('longName', ticker)
#
#                # Create a summary string for each stock
#                summary = f"{company_name} ({ticker}): {shares} share{'s' if shares > 1 else ''}"
#                portfolio_summary.append(summary)
#
#        except Exception as e:
#            st.error(f"Error fetching data for {ticker}: {e}")
#
#    if portfolio_summary:
#        # Join all stock summaries with newlines for better readability
#        portfolio_list = "\n".join(portfolio_summary)
#        response = f"Your portfolio contains:\n\n{portfolio_list}\n\nTotal Portfolio Value: ${total_value:.2f}"
#        return response
#    else:
#        return "No data available for your portfolio stocks."
#
#    if portfolio_data:
#        df = pd.DataFrame(portfolio_data)
#        df.set_index('Ticker', inplace=True)
#        st.table(df)
#        st.write(f"Total Portfolio Value: ${total_value:.2f}")
#
#        # Create a pie chart for sector allocation
#        plt.figure(figsize=(8, 8))
#        labels = []
#        sizes = []
#        for sector, value in sector_values.items():
#            labels.append(f"{sector} (${value:.2f})")
#            sizes.append(value)
#        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
#        plt.title('Portfolio Sector Allocation')
#        plt.axis('equal')
#        plt.tight_layout()
#        plt.savefig('portfolio_allocation.png')
#        plt.close()
#
#        st.image('portfolio_allocation.png')
#        return "Here's your current portfolio."
#    else:
#        return "No data available for your portfolio stocks."
#
# def get_portfolio_performance():
#    if 'portfolio' not in st.session_state or not st.session_state.portfolio:
#        return "Your portfolio is empty. Please add some stocks first."
#
#    total_return = 0
#    for ticker in st.session_state.portfolio:
#        data = yf.Ticker(ticker).history(period='1y')
#        if not data.empty:
#            start_price = data['Close'].iloc[0]
#            end_price = data['Close'].iloc[-1]
#            return_pct = (end_price - start_price) / start_price * 100
#            total_return += return_pct
#
#    avg_return = total_return / len(st.session_state.portfolio)
#    return f"The overall performance of your portfolio is {avg_return:.2f}% over the past year."
#
# def get_best_performing_stock():
#    if 'portfolio' not in st.session_state or not st.session_state.portfolio:
#        return "Your portfolio is empty. Please add some stocks first."
#
#    best_return = float('-inf')
#    best_stock = None
#
#    for ticker in st.session_state.portfolio:
#        data = yf.Ticker(ticker).history(period='1y')
#        if not data.empty:
#            start_price = data['Close'].iloc[0]
#            end_price = data['Close'].iloc[-1]
#            return_pct = (end_price - start_price) / start_price * 100
#            if return_pct > best_return:
#                best_return = return_pct
#                best_stock = ticker
#
#    if best_stock:
#        return f"Your best-performing stock is {best_stock} with a return of {best_return:.2f}% over the past year."
#    else:
#        return "Unable to determine the best-performing stock."
#
# def analyze_portfolio_performance():
#    if 'portfolio' not in st.session_state or not st.session_state.portfolio:
#        return "Your portfolio is empty. Please add some stocks first."
#
#    total_current_value = 0
#    total_initial_value = 0
#    portfolio_returns = []
#    sp500_data = yf.Ticker('^GSPC').history(period='1y')
#    sp500_return = (sp500_data['Close'].iloc[-1] - sp500_data['Close'].iloc[0]) / sp500_data['Close'].iloc[0] * 100
#
#    outperforming_stocks = []
#    underperforming_stocks = []
#
#    for ticker, shares in st.session_state.portfolio.items():
#        try:
#            stock = yf.Ticker(ticker)
#            data = stock.history(period='1y')
#            if not data.empty:
#                current_price = data['Close'].iloc[-1]
#                initial_price = data['Close'].iloc[0]
#                stock_return = (current_price - initial_price) / initial_price * 100
#
#                current_value = current_price * shares
#                initial_value = initial_price * shares
#
#                total_current_value += current_value
#                total_initial_value += initial_value
#
#                portfolio_returns.append(stock_return * (initial_value / total_initial_value))
#
#                # Compare each stock's performance to S&P 500
#                if stock_return > sp500_return:
#                    outperforming_stocks.append((ticker, stock_return))
#                else:
#                    underperforming_stocks.append((ticker, stock_return))
#
#            else:
#                st.error(f"No data available for {ticker}")
#        except Exception as e:
#            st.error(f"Error fetching data for {ticker}: {e}")
#
#    if total_initial_value > 0:
#        portfolio_return = (total_current_value - total_initial_value) / total_initial_value * 100
#        weighted_return = sum(portfolio_returns)
#
#        outperforming_stocks.sort(key=lambda x: x[1], reverse=True)
#        underperforming_stocks.sort(key=lambda x: x[1])
#
#        response = f"Your portfolio's overall performance:\n\n"
#        response += f"1. Total Return: {portfolio_return:.2f}%\n"
#        response += f"   - Started with: ${total_initial_value:.2f}\n"
#        response += f"   - Currently worth: ${total_current_value:.2f}\n\n"
#
#        response += f"2. Benchmark Comparison:\n"
#        response += f"   - Your portfolio: {portfolio_return:.2f}%\n"
#        response += f"   - S&P 500: {sp500_return:.2f}%\n"
#        response += f"   - You are {['underperforming', 'outperforming'][portfolio_return > sp500_return]} the market.\n\n"
#
#        response += "3. Stock-by-Stock Performance:\n"
#        if outperforming_stocks:
#            response += "   Outperforming the S&P 500:\n"
#            for ticker, return_pct in outperforming_stocks[:3]:  # Show top 3
#                response += f"   - {ticker}: {return_pct:.2f}%\n"
#        if underperforming_stocks:
#            response += "   Underperforming the S&P 500:\n"
#            for ticker, return_pct in underperforming_stocks[:3]:  # Show bottom 3
#                response += f"   - {ticker}: {return_pct:.2f}%\n"
#
#        response += "\nNote: Performance is calculated over the past year, assuming you held each stock for the entire period."
#
#        return response
#    else:
#        return "Unable to calculate portfolio performance. This could be due to insufficient historical data."


def initialize_portfolio():
    if "portfolio" not in st.session_state:
        st.session_state.portfolio = {}


def manage_portfolio(action, ticker=None, shares=1):
    initialize_portfolio()

    if action == "add" and ticker:
        ticker = ticker.upper()  # Convert to uppercase for consistency
        if ticker in st.session_state.portfolio:
            st.session_state.portfolio[ticker] += shares
        else:
            st.session_state.portfolio[ticker] = shares
        return f"{shares} share(s) of {ticker} have been added to your portfolio."
    elif action == "remove" and ticker:
        ticker = ticker.upper()
        if ticker in st.session_state.portfolio:
            if shares >= st.session_state.portfolio[ticker]:
                del st.session_state.portfolio[ticker]
                return f"All shares of {ticker} have been removed from your portfolio."
            else:
                st.session_state.portfolio[ticker] -= shares
                return f"{shares} share(s) of {ticker} have been removed from your portfolio."
        else:
            return f"{ticker} is not in your portfolio."
    elif action == "list":
        if not st.session_state.portfolio:
            return "Your portfolio is empty."
        else:
            stocks = [
                f"{ticker} ({shares} share{'s' if shares > 1 else ''})"
                for ticker, shares in st.session_state.portfolio.items()
            ]
            return "Your portfolio contains: " + ", ".join(stocks)
    else:
        return "Invalid action. Use 'add', 'remove', or 'list'."


def display_portfolio():
    initialize_portfolio()
    if not st.session_state.portfolio:
        return "Your portfolio is empty. Please add some stocks first."

    total_value = 0
    portfolio_data = []

    for ticker, shares in st.session_state.portfolio.items():
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period="1d")
            if not data.empty:
                current_price = data["Close"].iloc[-1]
                stock_value = current_price * shares
                total_value += stock_value

                info = stock.info
                company_name = info.get("longName", ticker)

                portfolio_data.append(
                    {
                        "Ticker": ticker,
                        "Company": company_name,
                        "Shares": shares,
                        "Price": current_price,
                        "Value": stock_value,
                    }
                )
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {e}")

    if portfolio_data:
        df = pd.DataFrame(portfolio_data)
        df.set_index("Ticker", inplace=True)
        st.table(df)
        st.write(f"Total Portfolio Value: ${total_value:.2f}")
        return "Here's your current portfolio."
    else:
        return "Unable to fetch data for your portfolio stocks."


def calculate_portfolio_performance(period="1y"):
    initialize_portfolio()
    if not st.session_state.portfolio:
        return "Your portfolio is empty. Please add some stocks first."

    total_current_value = 0
    total_initial_value = 0
    stock_performances = []

    for ticker, shares in st.session_state.portfolio.items():
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            if not data.empty and len(data) > 1:
                current_price = data["Close"].iloc[-1]
                initial_price = data["Close"].iloc[0]

                current_value = current_price * shares
                initial_value = initial_price * shares

                total_current_value += current_value
                total_initial_value += initial_value

                stock_return = (current_price - initial_price) / initial_price * 100
                stock_performances.append((ticker, stock_return, initial_value))
            else:
                st.warning(
                    f"Insufficient data for {ticker}. Skipping from calculation."
                )
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {e}")

    if total_initial_value > 0:
        portfolio_return = (
            (total_current_value - total_initial_value) / total_initial_value * 100
        )

        # Calculate weighted returns
        weighted_returns = []
        for ticker, return_pct, initial_value in stock_performances:
            weight = initial_value / total_initial_value
            weighted_returns.append((ticker, return_pct, weight))

        # Sort by contribution to highlight top contributors/detractors
        weighted_returns.sort(key=lambda x: x[1] * x[2], reverse=True)

        response = f"Your portfolio's performance over the past {period}:\n\n"
        response += f"1. Total Return: {portfolio_return:.2f}%\n"
        response += f"   - Started with: ${total_initial_value:.2f}\n"
        response += f"   - Currently worth: ${total_current_value:.2f}\n\n"

        response += "2. Stock-by-Stock Performance:\n"
        for ticker, return_pct, weight in weighted_returns[:3]:  # Top 3 contributors
            contribution = return_pct * weight
            response += f"   - {ticker}: {return_pct:.2f}% | Weight: {weight:.1%} | Contribution: {contribution:.2f}%\n"

        if len(weighted_returns) > 6:
            response += "   ...\n"
            for ticker, return_pct, weight in weighted_returns[
                -3:
            ]:  # Bottom 3 detractors
                contribution = return_pct * weight
                response += f"   - {ticker}: {return_pct:.2f}% | Weight: {weight:.1%} | Contribution: {contribution:.2f}%\n"

        # Compare to S&P 500
        try:
            sp500_data = yf.Ticker("^GSPC").history(period=period)
            sp500_return = (
                (sp500_data["Close"].iloc[-1] - sp500_data["Close"].iloc[0])
                / sp500_data["Close"].iloc[0]
                * 100
            )
            response += f"\n3. Benchmark Comparison:\n"
            response += f"   - Your portfolio: {portfolio_return:.2f}%\n"
            response += f"   - S&P 500: {sp500_return:.2f}%\n"
            response += f"   - You are {['underperforming', 'outperforming'][portfolio_return > sp500_return]} the market.\n"
        except Exception as e:
            st.error(f"Error fetching S&P 500 data: {e}")

        return response
    else:
        return "Unable to calculate portfolio performance. This could be due to insufficient historical data."


def get_best_performing_stock():
    initialize_portfolio()
    if not st.session_state.portfolio:
        return "Your portfolio is empty. Please add some stocks first."

    best_return = float("-inf")
    best_stock = None
    best_stock_name = None

    for ticker, shares in st.session_state.portfolio.items():
        try:
            data = yf.Ticker(ticker).history(period="1y")
            if not data.empty and len(data) > 1:
                start_price = data["Close"].iloc[0]
                end_price = data["Close"].iloc[-1]
                return_pct = (end_price - start_price) / start_price * 100
                if return_pct > best_return:
                    best_return = return_pct
                    best_stock = ticker
                    best_stock_name = yf.Ticker(ticker).info.get("longName", ticker)
            else:
                st.warning(f"No yearly data available for {ticker}.")
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {e}")

    if best_stock:
        return f"Your best-performing stock is {best_stock_name} ({best_stock}) with a return of {best_return:.2f}% over the past year."
    else:
        return "Unable to determine the best-performing stock due to insufficient data."


def get_most_volatile_stocks(top_n=3):
    initialize_portfolio()
    if not st.session_state.portfolio:
        return "Your portfolio is empty. Please add some stocks first."

    stock_volatilities = []

    for ticker in st.session_state.portfolio:
        try:
            data = yf.Ticker(ticker).history(period="1y")
            if not data.empty and len(data) > 1:
                daily_returns = data["Close"].pct_change().dropna()
                volatility = daily_returns.std() * (252**0.5)  # Annualized volatility
                stock_volatilities.append((ticker, volatility))
            else:
                st.warning(f"No yearly data available for {ticker}.")
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {e}")

    if not stock_volatilities:
        return "Unable to calculate volatility for any of your stocks."

    stock_volatilities.sort(key=lambda x: x[1], reverse=True)
    top_volatile = stock_volatilities[:top_n]

    response = f"Your {top_n} most volatile stock{'s' if top_n > 1 else ''}:\n\n"
    for ticker, volatility in top_volatile:
        stock_name = yf.Ticker(ticker).info.get("longName", ticker)
        response += f"1. {stock_name} ({ticker})\n"
        response += f"   - Annualized Volatility: {volatility * 100:.2f}%\n\n"

    response += "Note: Volatility is measured by the standard deviation of daily returns, annualized. Higher volatility means larger price swings."
    return response


def analyze_sector_allocation():
    initialize_portfolio()
    if not st.session_state.portfolio:
        return "Your portfolio is empty. Please add some stocks first."

    sector_values = {}
    total_value = 0
    unknown_value = 0

    for ticker, shares in st.session_state.portfolio.items():
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period="1d")
            info = stock.info

            if not data.empty:
                current_price = data["Close"].iloc[-1]
                stock_value = current_price * shares
                total_value += stock_value

                sector = info.get("sector")
                if sector:
                    if sector in sector_values:
                        sector_values[sector] += stock_value
                    else:
                        sector_values[sector] = stock_value
                else:
                    unknown_value += stock_value
                    st.warning(
                        f"Unable to determine sector for {ticker}. It will be categorized as 'Unknown'."
                    )
            else:
                st.warning(f"No price data available for {ticker}.")
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {e}")

    if not sector_values and unknown_value == 0:
        return "Unable to fetch sector data for any of your stocks."

    if unknown_value > 0:
        sector_values["Unknown"] = unknown_value

    # Sort sectors by value for better readability
    sorted_sectors = sorted(sector_values.items(), key=lambda x: x[1], reverse=True)

    response = "Your portfolio's sector allocation:\n\n"
    for sector, value in sorted_sectors:
        percentage = (value / total_value) * 100
        response += f"1. {sector}: ${value:.2f} ({percentage:.2f}%)\n"

    # Generate a pie chart for sector allocation
    plt.figure(figsize=(10, 6))
    labels = []
    sizes = []
    for sector, value in sorted_sectors:
        labels.append(f"{sector} ({value/total_value:.1%})")
        sizes.append(value)
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90, pctdistance=0.85)
    plt.title("Portfolio Sector Allocation", fontsize=16)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig("sector_allocation.png")
    plt.close()

    st.image("sector_allocation.png")
    return response


def should_rebalance_portfolio():
    initialize_portfolio()
    if not st.session_state.portfolio:
        return "Your portfolio is empty. Please add some stocks first."

    total_value = 0
    stock_values = {}

    for ticker, shares in st.session_state.portfolio.items():
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period="1d")
            if not data.empty:
                current_price = data["Close"].iloc[-1]
                stock_value = current_price * shares
                total_value += stock_value
                stock_values[ticker] = stock_value
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {e}")

    if not stock_values:
        return "Unable to fetch data for your portfolio stocks."

    # Calculate current allocation
    allocations = {
        ticker: value / total_value for ticker, value in stock_values.items()
    }

    # Define threshold for rebalancing (e.g., 5% deviation from equal weight)
    threshold = 0.05
    equal_weight = 1 / len(st.session_state.portfolio)

    for ticker, allocation in allocations.items():
        if abs(allocation - equal_weight) > threshold:
            return (
                "Yes, you should consider rebalancing your portfolio. Some stocks have deviated "
                "significantly from an equal-weight allocation."
            )

    return "No need to rebalance at this time. Your portfolio is reasonably balanced."


def get_latest_market_news():
    try:
        news_api_key = os.environ.get("NEWS_API_KEY")
        if not news_api_key:
            return "Please set the NEWS_API_KEY environment variable."

        newsapi = NewsApiClient(api_key=news_api_key)
        market_news = newsapi.get_top_headlines(
            q="stock market", language="en", country="us"
        )
        if market_news["totalResults"] > 0:
            latest_article = market_news["articles"][0]
            title = latest_article["title"]
            description = latest_article["description"]
            return (
                f"The latest news affecting the stock market is: {title}\n{description}"
            )
        else:
            return "No recent news found related to the stock market."
    except Exception as e:
        return f"An error occurred: {str(e)}"


def get_top_trending_stocks():
    try:
        news_api_key = os.environ.get("NEWS_API_KEY")
        if not news_api_key:
            return "Please set the NEWS_API_KEY environment variable."

        newsapi = NewsApiClient(api_key=news_api_key)
        trending_stocks = newsapi.get_everything(
            q="trending stocks", language="en", sort_by="relevancy"
        )
        if trending_stocks["totalResults"] > 0:
            trending_articles = trending_stocks["articles"][:5]
            trending_stocks_list = [
                article["title"].replace("\n", " ") for article in trending_articles
            ]
            return f"The top trending stocks right now are:\n\n{''.join(trending_stocks_list)}"
        else:
            return "No news found for trending stocks."
    except Exception as e:
        return f"An error occurred: {str(e)}"


def get_market_valuation():
    try:
        news_api_key = os.environ.get("NEWS_API_KEY")
        if not news_api_key:
            return "Please set the NEWS_API_KEY environment variable."

        newsapi = NewsApiClient(api_key=news_api_key)
        market_valuation = newsapi.get_everything(
            q="stock market valuation", language="en", sort_by="relevancy"
        )
        if market_valuation["totalResults"] > 0:
            valuation_article = market_valuation["articles"][0]
            title = valuation_article["title"]
            description = valuation_article["description"]
            return f"According to the latest news, the market valuation is: {title}\n{description}"
        else:
            return "No news found for market valuation."
    except Exception as e:
        return f"An error occurred: {str(e)}"


def get_economic_events_impacting_market():
    try:
        news_api_key = os.environ.get("NEWS_API_KEY")
        if not news_api_key:
            return "Please set the NEWS_API_KEY environment variable."

        newsapi = NewsApiClient(api_key=news_api_key)
        economic_events = newsapi.get_everything(
            q="economic events stock market", language="en", sort_by="relevancy"
        )
        if economic_events["totalResults"] > 0:
            event_articles = economic_events["articles"][:3]
            event_list = [f"{article['title']}" for article in event_articles]
            return f"Economic events that could impact the market this week:\n\n{' '.join(event_list)}"
        else:
            return "No news found for economic events impacting the market."
    except Exception as e:
        return f"An error occurred: {str(e)}"


def get_analyst_opinions(ticker):
    try:
        news_api_key = os.environ.get("NEWS_API_KEY")
        if not news_api_key:
            return "Please set the NEWS_API_KEY environment variable."

        newsapi = NewsApiClient(api_key=news_api_key)
        analyst_opinions = newsapi.get_everything(
            q=f"{ticker} analyst opinions", language="en", sort_by="relevancy"
        )
        if analyst_opinions["totalResults"] > 0:
            opinion_articles = analyst_opinions["articles"][:3]
            opinion_list = [f"{article['title']}" for article in opinion_articles]
            return f"Here's what analysts are saying about {ticker}:\n\n{' '.join(opinion_list)}"
        else:
            return f"No analyst opinions found for {ticker}."
    except Exception as e:
        return f"An error occurred: {str(e)}"


def get_company_news(ticker):
    try:
        ticker_info = yf.Ticker(ticker)
        news = ticker_info.news
        if news:
            news_list = []
            for article in news[:5]:  # Get the latest 5 news articles
                title = article["title"]
                link = article["link"]
                news_item = f"Title: {title}\nLink: {link}\n"
                news_list.append(news_item)
            return f"Here are the latest news about {ticker}:\n\n" + "\n".join(
                news_list
            )
        else:
            return f"No recent news found about {ticker}."
    except Exception as e:
        return f"An error occurred: {str(e)}"


def get_income_statement(ticker):
    try:
        data = yf.Ticker(ticker).financials
        if data.empty:
            return "Unable to fetch income statement data for the given ticker symbol."

        # Summarize the income statement data
        summary = data.head(10)  # Limiting to the first 5 rows
        return summary.to_json()
    except Exception as e:
        return f"An error occurred: {str(e)}"


def get_balance_sheet(ticker):
    try:
        data = yf.Ticker(ticker).balance_sheet
        if data.empty:
            return "Unable to fetch balance sheet data for the given ticker symbol."

        # Summarize the balance sheet data
        summary = data.head(10)  # Limiting to the first 10 rows
        return summary.to_json()
    except Exception as e:
        return f"An error occurred: {str(e)}"


# def get_cash_flow(ticker):
# data = yf.Ticker(ticker).cashflow
# return (
# data.to_json()
# if not data.empty
# else "Unable to fetch cash flow data for the given ticker symbol."
# )


def get_cash_flow(ticker):
    try:
        data = yf.Ticker(ticker).cashflow
        if data.empty:
            return "Unable to fetch cash flow data for the given ticker symbol."

        # Summarize the cash flow data
        summary = data.head(10)  # Limiting to the first 10 rows
        return summary.to_json()
    except Exception as e:
        return f"An error occurred: {str(e)}"


def get_major_holders(ticker):
    data = yf.Ticker(ticker).major_holders
    return (
        data.to_json()
        if not data.empty
        else "Unable to fetch major holders data for the given ticker symbol."
    )


def get_institutional_holders(ticker):
    data = yf.Ticker(ticker).institutional_holders
    return (
        data.to_json()
        if not data.empty
        else "Unable to fetch institutional holders data for the given ticker symbol."
    )


def get_mutualfund_holders(ticker):
    data = yf.Ticker(ticker).mutualfund_holders
    return (
        data.to_json()
        if not data.empty
        else "Unable to fetch mutual fund holders data for the given ticker symbol."
    )


def get_insider_transactions(ticker):
    data = yf.Ticker(ticker).insider_transactions
    return (
        data.to_json()
        if not data.empty
        else "Unable to fetch insider transactions data for the given ticker symbol."
    )


def get_insider_purchases(ticker):
    data = yf.Ticker(ticker).insider_purchases
    return (
        data.to_json()
        if not data.empty
        else "Unable to fetch insider purchases data for the given ticker symbol."
    )


def get_recommendations(ticker):
    data = yf.Ticker(ticker).recommendations
    return (
        data.to_json()
        if not data.empty
        else "Unable to fetch recommendations data for the given ticker symbol."
    )


def get_recommendations_summary(ticker):
    data = yf.Ticker(ticker).recommendations_summary
    return (
        data.to_json()
        if not data.empty
        else "Unable to fetch recommendations summary data for the given ticker symbol."
    )


def get_future_earnings(ticker):
    data = yf.Ticker(ticker).earnings_dates
    return (
        data.to_json()
        if not data.empty
        else "Unable to fetch future earnings data for the given ticker symbol."
    )


def get_news(ticker):
    data = yf.Ticker(ticker).news
    return (
        json.dumps(data)
        if data
        else "Unable to fetch news data for the given ticker symbol."
    )


def get_recent_actions(ticker):
    data = yf.Ticker(ticker).actions
    return (
        data.tail(5).to_json()
        if not data.empty
        else "Unable to fetch recent actions data for the given ticker symbol."
    )


def get_stock_split(ticker):
    data = yf.Ticker(ticker).splits
    return (
        data.to_json()
        if not data.empty
        else "Unable to fetch stock split data for the given ticker symbol."
    )


def plot_bar_chart(ticker, period=None, year=None, start_year=None, end_year=None):
    try:
        if start_year and end_year:
            start_date = f"{start_year}-01-01"
            end_date = f"{end_year}-12-31"
            data = yf.Ticker(ticker).history(start=start_date, end=end_date)
            title = f"{ticker} Bar Chart from {start_year} to {end_year}"
        elif year:
            start_date = f"{year}-01-01"
            end_date = f"{year}-12-31"
            data = yf.Ticker(ticker).history(start=start_date, end=end_date)
            title = f"{ticker} Bar Chart for {year}"
        else:
            period_str = f"{period}y" if period else "1y"
            data = yf.Ticker(ticker).history(period=period_str)
            title = f"{ticker} {period}-Year Bar Chart"

        if not data.empty:
            plt.figure(figsize=(10, 6))
            plt.bar(data.index, data["Close"])
            plt.title(title)
            plt.xlabel("Date")
            plt.ylabel("Price (USD)")
            plt.savefig("bar_chart.png", bbox_inches="tight")
            return "The bar chart has been generated."
        else:
            return "Unable to fetch bar chart data for the given ticker symbol."
    except Exception as e:
        return f"An error occurred: {str(e)}"


def plot_ordinary_shares_number(ticker):
    try:
        data = yf.Ticker(ticker).balance_sheet
        ordinary_shares_number = data.loc["Ordinary Shares Number"]
        if not ordinary_shares_number.empty:
            plt.figure(figsize=(10, 6))
            ordinary_shares_number.plot(kind="bar")
            plt.title(f"{ticker} Ordinary Shares Number")
            plt.xlabel("Date")
            plt.ylabel("Number of Shares")
            plt.savefig("ordinary_shares_number.png", bbox_inches="tight")
            plt.close()
        return "The ordinary shares number chart has been generated."
    except Exception as e:
        return f"An error occurred: {str(e)}"


def plot_share_issued(ticker):
    try:
        data = yf.Ticker(ticker).balance_sheet
        share_issued = data.loc["Share Issued"]
        if not share_issued.empty:
            plt.figure(figsize=(10, 6))
            share_issued.plot(kind="bar")
            plt.title(f"{ticker} Share Issued")
            plt.xlabel("Date")
            plt.ylabel("Number of Shares Issued")
            plt.savefig("share_issued.png", bbox_inches="tight")
        return "The share issued chart has been generated."
    except Exception as e:
        return f"An error occurred: {str(e)}"


def plot_net_debt(ticker):
    try:
        data = yf.Ticker(ticker).balance_sheet
        net_debt = data.loc["Net Debt"]
        if not net_debt.empty:
            plt.figure(figsize=(10, 6))
            net_debt.plot(kind="bar")
            plt.title(f"{ticker} Net Debt")
            plt.xlabel("Date")
            plt.ylabel("Net Debt (in USD)")
            plt.savefig("net_debt.png", bbox_inches="tight")
        return "The net debt chart has been generated."
    except Exception as e:
        return f"An error occurred: {str(e)}"


def plot_total_debt(ticker):
    try:
        data = yf.Ticker(ticker).balance_sheet
        total_debt = data.loc["Total Debt"]
        if not total_debt.empty:
            plt.figure(figsize=(10, 6))
            total_debt.plot(kind="bar")
            plt.title(f"{ticker} Total Debt")
            plt.xlabel("Date")
            plt.ylabel("Total Debt (in USD)")
            plt.savefig("total_debt.png", bbox_inches="tight")
        return "The total debt chart has been generated."
    except Exception as e:
        return f"An error occurred: {str(e)}"


def plot_tangible_book_value(ticker):
    try:
        data = yf.Ticker(ticker).balance_sheet
        tangible_book_value = data.loc["Tangible Book Value"]
        if not tangible_book_value.empty:
            plt.figure(figsize=(10, 6))
            tangible_book_value.plot(kind="bar")
            plt.title(f"{ticker} Tangible Book Value")
            plt.xlabel("Date")
            plt.ylabel("Tangible Book Value (in USD)")
            plt.savefig("tangible_book_value.png", bbox_inches="tight")
        return "The tangible book value chart has been generated."
    except Exception as e:
        return f"An error occurred: {str(e)}"


def plot_invested_capital(ticker):
    try:
        data = yf.Ticker(ticker).balance_sheet
        invested_capital = data.loc["Invested Capital"]
        if not invested_capital.empty:
            plt.figure(figsize=(10, 6))
            invested_capital.plot(kind="bar")
            plt.title(f"{ticker} Invested Capital")
            plt.xlabel("Date")
            plt.ylabel("Invested Capital (in USD)")
            plt.savefig("invested_capital.png", bbox_inches="tight")
        return "The invested capital chart has been generated."
    except Exception as e:
        return f"An error occurred: {str(e)}"


def plot_working_capital(ticker):
    try:
        data = yf.Ticker(ticker).balance_sheet
        working_capital = data.loc["Working Capital"]
        if not working_capital.empty:
            plt.figure(figsize=(10, 6))
            working_capital.plot(kind="bar")
            plt.title(f"{ticker} Working Capital")
            plt.xlabel("Date")
            plt.ylabel("Working Capital (in USD)")
            plt.savefig("working_capital.png", bbox_inches="tight")
        return "The working capital chart has been generated."
    except Exception as e:
        return f"An error occurred: {str(e)}"


def plot_net_tangible_assets(ticker):
    try:
        data = yf.Ticker(ticker).balance_sheet
        net_tangible_assets = data.loc["Net Tangible Assets"]
        if not net_tangible_assets.empty:
            plt.figure(figsize=(10, 6))
            net_tangible_assets.plot(kind="bar")
            plt.title(f"{ticker} Net Tangible Assets")
            plt.xlabel("Date")
            plt.ylabel("Net Tangible Assets (in USD)")
            plt.savefig("net_tangible_assets.png", bbox_inches="tight")
        return "The net tangible assets chart has been generated."
    except Exception as e:
        return f"An error occurred: {str(e)}"


def plot_capital_lease_obligations(ticker):
    try:
        data = yf.Ticker(ticker).balance_sheet
        capital_lease_obligations = data.loc["Capital Lease Obligations"]
        if not capital_lease_obligations.empty:
            plt.figure(figsize=(10, 6))
            capital_lease_obligations.plot(kind="bar")
            plt.title(f"{ticker} Capital Lease Obligations")
            plt.xlabel("Date")
            plt.ylabel("Capital Lease Obligations (in USD)")
            plt.savefig("capital_lease_obligations.png", bbox_inches="tight")
        return "The capital lease obligations chart has been generated."
    except Exception as e:
        return f"An error occurred: {str(e)}"


def plot_free_cash_flow(ticker, period="annual"):
    try:
        data = yf.Ticker(ticker).cashflow.T
        if data.empty or "Free Cash Flow" not in data.columns:
            return "Unable to fetch Free Cash Flow data for the given ticker symbol."

        free_cash_flow_data = data["Free Cash Flow"]
        if period == "annual":
            summary = free_cash_flow_data.head(10)
        else:
            summary = free_cash_flow_data.resample("QE").sum().head(10)

        plt.figure(figsize=(10, 6))
        plt.plot(summary.index, summary.values, marker="o")
        plt.title(f"Free Cash Flow for {ticker}")
        plt.xlabel("Date")
        plt.ylabel("Free Cash Flow")
        plt.grid(True)
        plt.savefig("free_cash_flow.png")
        return "The Free Cash Flow chart has been generated."
    except Exception as e:
        return f"An error occurred: {str(e)}"


def plot_repurchase_of_capital_stock(ticker, period="annual"):
    try:
        data = yf.Ticker(ticker).cashflow.T
        if data.empty or "Repurchase of Capital Stock" not in data.columns:
            return "Unable to fetch Repurchase of Capital Stock data for the given ticker symbol."

        repurchase_data = data["Repurchase of Capital Stock"]
        if period == "annual":
            summary = repurchase_data.head(10)
        else:
            summary = repurchase_data.resample("Q").sum().head(10)

        plt.figure(figsize=(10, 6))
        plt.plot(summary.index, summary.values, marker="o")
        plt.title(f"Repurchase of Capital Stock for {ticker}")
        plt.xlabel("Date")
        plt.ylabel("Repurchase of Capital Stock")
        plt.grid(True)
        plt.savefig("repurchase_of_capital_stock.png")
        return "The Repurchase of Capital Stock chart has been generated."
    except Exception as e:
        return f"An error occurred: {str(e)}"


def plot_repayment_of_debt(ticker, period="annual"):
    try:
        data = yf.Ticker(ticker).cashflow.T
        if data.empty or "Repayment of Debt" not in data.columns:
            return "Unable to fetch Repayment of Debt data for the given ticker symbol."

        repayment_data = data["Repayment of Debt"]
        if period == "annual":
            summary = repayment_data.head(10)
        else:
            summary = repayment_data.resample("Q").sum().head(10)

        plt.figure(figsize=(10, 6))
        plt.plot(summary.index, summary.values, marker="o")
        plt.title(f"Repayment of Debt for {ticker}")
        plt.xlabel("Date")
        plt.ylabel("Repayment of Debt")
        plt.grid(True)
        plt.savefig("repayment_of_debt.png")
        return "The Repayment of Debt chart has been generated."
    except Exception as e:
        return f"An error occurred: {str(e)}"


def plot_issuance_of_debt(ticker, period="annual"):
    try:
        data = yf.Ticker(ticker).cashflow.T
        if data.empty or "Issuance of Debt" not in data.columns:
            return "Unable to fetch Issuance of Debt data for the given ticker symbol."

        issuance_data = data["Issuance of Debt"]
        if period == "annual":
            summary = issuance_data.head(10)
        else:
            summary = issuance_data.resample("Q").sum().head(10)

        plt.figure(figsize=(10, 6))
        plt.plot(summary.index, summary.values, marker="o")
        plt.title(f"Issuance of Debt for {ticker}")
        plt.xlabel("Date")
        plt.ylabel("Issuance of Debt")
        plt.grid(True)
        plt.savefig("issuance_of_debt.png")
        return "The Issuance of Debt chart has been generated."
    except Exception as e:
        return f"An error occurred: {str(e)}"


def plot_capital_expenditure(ticker, period="annual"):
    try:
        data = yf.Ticker(ticker).cashflow.T
        if data.empty or "Capital Expenditure" not in data.columns:
            return (
                "Unable to fetch Capital Expenditure data for the given ticker symbol."
            )

        capex_data = data["Capital Expenditure"]
        if period == "annual":
            summary = capex_data.head(10)
        else:
            summary = capex_data.resample("Q").sum().head(10)

        plt.figure(figsize=(10, 6))
        plt.plot(summary.index, summary.values, marker="o")
        plt.title(f"Capital Expenditure for {ticker}")
        plt.xlabel("Date")
        plt.ylabel("Capital Expenditure")
        plt.grid(True)
        plt.savefig("capital_expenditure.png")
        return "The Capital Expenditure chart has been generated."
    except Exception as e:
        return f"An error occurred: {str(e)}"


def plot_interest_paid(ticker, period="annual"):
    try:
        data = yf.Ticker(ticker).cashflow.T
        if data.empty or "Interest Paid" not in data.columns:
            return "Unable to fetch Interest Paid data for the given ticker symbol."

        interest_paid_data = data["Interest Paid"]
        if period == "annual":
            summary = interest_paid_data.head(10)
        else:
            summary = interest_paid_data.resample("Q").sum().head(10)

        plt.figure(figsize=(10, 6))
        plt.plot(summary.index, summary.values, marker="o")
        plt.title(f"Interest Paid for {ticker}")
        plt.xlabel("Date")
        plt.ylabel("Interest Paid")
        plt.grid(True)
        plt.savefig("interest_paid.png")
        return "The Interest Paid chart has been generated."
    except Exception as e:
        return f"An error occurred: {str(e)}"


def plot_income_tax_paid(ticker, period="annual"):
    try:
        data = yf.Ticker(ticker).cashflow.T
        if data.empty or "Income Tax Paid" not in data.columns:
            return "Unable to fetch Income Tax Paid data for the given ticker symbol."

        income_tax_paid_data = data["Income Tax Paid"]
        if period == "annual":
            summary = income_tax_paid_data.head(10)
        else:
            summary = income_tax_paid_data.resample("Q").sum().head(10)

        plt.figure(figsize=(10, 6))
        plt.plot(summary.index, summary.values, marker="o")
        plt.title(f"Income Tax Paid for {ticker}")
        plt.xlabel("Date")
        plt.ylabel("Income Tax Paid")
        plt.grid(True)
        plt.savefig("income_tax_paid.png")
        return "The Income Tax Paid chart has been generated."
    except Exception as e:
        return f"An error occurred: {str(e)}"


def plot_ending_cash_position(ticker, period="annual"):
    try:
        data = yf.Ticker(ticker).cashflow.T
        if data.empty or "Cash Position" not in data.columns:
            return (
                "Unable to fetch Ending Cash Position data for the given ticker symbol."
            )

        ending_cash_position_data = data["Cash Position"]
        if period == "annual":
            summary = ending_cash_position_data.head(10)
        else:
            summary = ending_cash_position_data.resample("Q").sum().head(10)

        plt.figure(figsize=(10, 6))
        plt.plot(summary.index, summary.values, marker="o")
        plt.title(f"Ending Cash Position for {ticker}")
        plt.xlabel("Date")
        plt.ylabel("Ending Cash Position")
        plt.grid(True)
        plt.savefig("ending_cash_position.png")
        return "The Ending Cash Position chart has been generated."
    except Exception as e:
        return f"An error occurred: {str(e)}"


def plot_beginning_cash_position(ticker, period="annual"):
    try:
        data = yf.Ticker(ticker).cashflow.T
        if data.empty or "Beginning Cash Position" not in data.columns:
            return "Unable to fetch Beginning Cash Position data for the given ticker symbol."

        beginning_cash_position_data = data["Beginning Cash Position"]
        if period == "annual":
            summary = beginning_cash_position_data.head(10)
        else:
            summary = beginning_cash_position_data.resample("Q").sum().head(10)

        plt.figure(figsize=(10, 6))
        plt.plot(summary.index, summary.values, marker="o")
        plt.title(f"Beginning Cash Position for {ticker}")
        plt.xlabel("Date")
        plt.ylabel("Beginning Cash Position")
        plt.grid(True)
        plt.savefig("beginning_cash_position.png")
        return "The Beginning Cash Position chart has been generated."
    except Exception as e:
        return f"An error occurred: {str(e)}"


def plot_change_in_cash(ticker, period="annual"):
    try:
        data = yf.Ticker(ticker).cashflow.T
        if data.empty or "Change in Cash" not in data.columns:
            return "Unable to fetch Change in Cash data for the given ticker symbol."

        change_in_cash_data = data["Change in Cash"]
        if period == "annual":
            summary = change_in_cash_data.head(10)
        else:
            summary = change_in_cash_data.resample("Q").sum().head(10)

        plt.figure(figsize=(10, 6))
        plt.plot(summary.index, summary.values, marker="o")
        plt.title(f"Change in Cash for {ticker}")
        plt.xlabel("Date")
        plt.ylabel("Change in Cash")
        plt.grid(True)
        plt.savefig("change_in_cash.png")
        return "The Change in Cash chart has been generated."
    except Exception as e:
        return f"An error occurred: {str(e)}"


def plot_ebitda(ticker, period="5y"):
    try:
        data = yf.Ticker(ticker).financials.loc["EBITDA"]
        data = data[data.index.str.contains(period)]

        if data.empty:
            return "No data available for the given period."

        plt.figure(figsize=(10, 6))
        data.plot(kind="bar")
        plt.title(f"EBITDA for {ticker}")
        plt.xlabel("Date")
        plt.ylabel("EBITDA")
        plt.savefig("ebitda.png", bbox_inches="tight")
        return "The EBITDA chart has been generated."
    except Exception as e:
        return f"An error occurred: {str(e)}"


def plot_ebit(ticker, period="5y"):
    try:
        data = yf.Ticker(ticker).financials.loc["EBIT"]
        data = data[data.index.str.contains(period)]

        if data.empty:
            return "No data available for the given period."

        plt.figure(figsize=(10, 6))
        data.plot(kind="bar")
        plt.title(f"EBIT for {ticker}")
        plt.xlabel("Date")
        plt.ylabel("EBIT")
        plt.savefig("ebit.png", bbox_inches="tight")
        return "The EBIT chart has been generated."
    except Exception as e:
        return f"An error occurred: {str(e)}"


def plot_net_income_from_continuing_operations_net_minority_interest(
    ticker, period="5y"
):
    try:
        data = yf.Ticker(ticker).financials.loc[
            "Net Income from Continuing Operations Net Minority Interest"
        ]
        data = data[data.index.str.contains(period)]

        if data.empty:
            return "No data available for the given period."

        plt.figure(figsize=(10, 6))
        data.plot(kind="bar")
        plt.title(
            f"Net Income from Continuing Operations Net Minority Interest for {ticker}"
        )
        plt.xlabel("Date")
        plt.ylabel("Net Income")
        plt.savefig(
            "net_income_from_continuing_operations_net_minority_interest.png",
            bbox_inches="tight",
        )
        return "The Net Income from Continuing Operations Net Minority Interest chart has been generated."
    except Exception as e:
        return f"An error occurred: {str(e)}"


def plot_depreciation(ticker, period="5y"):
    try:
        data = yf.Ticker(ticker).financials.loc["Depreciation"]
        data = data[data.index.str.contains(period)]

        if data.empty:
            return "No data available for the given period."

        plt.figure(figsize=(10, 6))
        data.plot(kind="bar")
        plt.title(f"Depreciation for {ticker}")
        plt.xlabel("Date")
        plt.ylabel("Depreciation")
        plt.savefig("depreciation.png", bbox_inches="tight")
        return "The Depreciation chart has been generated."
    except Exception as e:
        return f"An error occurred: {str(e)}"


def plot_cost_of_revenue(ticker, period="5y"):
    try:
        data = yf.Ticker(ticker).financials.loc["Cost Of Revenue"]
        data = data[data.index.str.contains(period)]

        if data.empty:
            return "No data available for the given period."

        plt.figure(figsize=(10, 6))
        data.plot(kind="bar")
        plt.title(f"Cost Of Revenue for {ticker}")
        plt.xlabel("Date")
        plt.ylabel("Cost Of Revenue")
        plt.savefig("cost_of_revenue.png", bbox_inches="tight")
        plt.close()
        return "The Cost Of Revenue chart has been generated."
    except Exception as e:
        return f"An error occurred: {str(e)}"


def plot_tax_effect_of_unusual_items(ticker, period="5y"):
    try:
        data = yf.Ticker(ticker).financials.loc["Tax Effect of Unusual Items"]
        data = data[data.index.str.contains(period)]

        if data.empty:
            return "No data available for the given period."

        plt.figure(figsize=(10, 6))
        data.plot(kind="bar")
        plt.title(f"Tax Effect of Unusual Items for {ticker}")
        plt.xlabel("Date")
        plt.ylabel("Tax Effect of Unusual Items")
        plt.savefig("tax_effect_of_unusual_items.png", bbox_inches="tight")
        return "The Tax Effect of Unusual Items chart has been generated."
    except Exception as e:
        return f"An error occurred: {str(e)}"


def plot_total_unusual_items(ticker, period="5y"):
    try:
        data = yf.Ticker(ticker).financials.loc["Total Unusual Items"]
        data = data[data.index.str.contains(period)]

        if data.empty:
            return "No data available for the given period."

        plt.figure(figsize=(10, 6))
        data.plot(kind="bar")
        plt.title(f"Total Unusual Items for {ticker}")
        plt.xlabel("Date")
        plt.ylabel("Total Unusual Items")
        plt.savefig("total_unusual_items.png", bbox_inches="tight")
        return "The Total Unusual Items chart has been generated."
    except Exception as e:
        return f"An error occurred: {str(e)}"


def plot_two_shares_side_by_side(ticker1, ticker2, start_date, end_date, interval="1d"):
    try:
        # Download stock data
        data1 = yf.download(ticker1, start=start_date, end=end_date, interval=interval)
        data2 = yf.download(ticker2, start=start_date, end=end_date, interval=interval)

        # Check if data is available
        if data1.empty or data2.empty:
            return "No data available for the given tickers and period."

        # Align the data by date
        data1 = data1["Close"]
        data2 = data2["Close"]
        combined_data = data1.to_frame(name=ticker1).join(
            data2.to_frame(name=ticker2), how="inner"
        )

        # Handle missing data by forward filling
        combined_data = combined_data.ffill()

        # Normalize the prices to a common starting point for comparison
        combined_data[ticker1] = (
            combined_data[ticker1] / combined_data[ticker1].iloc[0] * 100
        )
        combined_data[ticker2] = (
            combined_data[ticker2] / combined_data[ticker2].iloc[0] * 100
        )

        # Plotting
        plt.figure(figsize=(14, 7))
        plt.plot(combined_data[ticker1], label=ticker1, color="blue")
        plt.plot(combined_data[ticker2], label=ticker2, color="red")

        plt.title(f"Stock Prices: {ticker1} vs {ticker2}")
        plt.xlabel("Date")
        plt.ylabel("Normalized Stock Price")
        plt.legend()
        plt.grid(True)
        plt.savefig("comparison_chart.png", bbox_inches="tight")
        return "The comparison chart has been generated."
    except Exception as e:
        return f"An error occurred: {str(e)}"


sector_to_companies = {
    "it": ["AAPL", "MSFT", "GOOGL"],
    "medical": ["JNJ", "PFE", "MRK"],
    "finance": ["JPM", "BAC", "C"],
    "consumer_discretionary": ["AMZN", "TSLA", "HD"],
    "utilities": ["DUK", "SO", "NEE"],
    "energy": ["XOM", "CVX", "SLB"],
    "consumer_staples": ["PG", "KO", "PEP"],
    "industrials": ["BA", "CAT", "GE"],
    "telecommunication": ["VZ", "T", "TMUS"],
    "real_estate": ["O", "SPG", "PLD"],
}


def plot_companies_in_sector(sector, start_date, end_date, interval="1d"):
    try:
        # Check if the sector exists in the mapping
        if sector.lower() not in sector_to_companies:
            return f"Sector '{sector}' not found in the available sectors."

        companies = sector_to_companies[sector.lower()]

        plt.figure(figsize=(14, 7))

        for company in companies:
            # Download stock data
            data = yf.download(
                company, start=start_date, end=end_date, interval=interval
            )

            if data.empty:
                continue

            # Align data by date and forward fill missing data
            data = data["Close"].ffill()

            # Normalize the prices to a common starting point for comparison
            normalized_data = data / data.iloc[0] * 100

            # Plot the data
            plt.plot(normalized_data, label=company)

        plt.title(f"Stock Prices for Companies in the {sector.capitalize()} Sector")
        plt.xlabel("Date")
        plt.ylabel("Normalized Stock Price")
        plt.legend()
        plt.grid(True)
        plt.savefig("sector_comparison_chart.png", bbox_inches="tight")
        return "The sector comparison chart has been generated."
    except Exception as e:
        return f"An error occurred: {str(e)}"


functions = [
    {
        "name": "get_stock_price",
        "description": "Gets the latest stock price given the ticker symbol of a company.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (for example AAPL for Apple).",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "calculate_SMA",
        "description": "Calculate the simple moving average for a given stock ticker and a window.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (for example AAPL for Apple)",
                },
                "window": {
                    "type": "integer",
                    "description": "The timeframe to consider when calculating the SMA.",
                },
            },
            "required": ["ticker", "window"],
        },
    },
    {
        "name": "calculate_EMA",
        "description": "Calculate the exponential moving average for a given stock ticker and a window.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)",
                },
                "window": {
                    "type": "integer",
                    "description": "The timeframe to consider when calculating the EMA.",
                },
            },
            "required": ["ticker", "window"],
        },
    },
    {
        "name": "get_stock_open_price",
        "description": "Get the opening price of a stock for today.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_stock_price_movement",
        "description": "Get the price movement direction (up, down, or unchanged) of a stock for today.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_stock_trading_volume",
        "description": "Get the trading volume of a stock for today.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_52_week_high",
        "description": "Get the 52-week high price for a given stock ticker.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_52_week_low",
        "description": "Get the 52-week low price for a given stock ticker.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "compare_stock_sp500",
        "description": "Compare a stock's performance with the S&P 500 index over the last year.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_pe_ratio",
        "description": "Get the P/E (price-to-earnings) ratio for a given stock ticker.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_dividend_info",
        "description": "Get information about whether a company pays a dividend and the dividend yield.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_market_cap",
        "description": "Get the market capitalization of a company based on its stock ticker.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_stock_type",
        "description": "Determine whether a company is considered a growth stock or a value stock based on its financial metrics and industry.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_top_company_risks",
        "description": "Get the top 3 risks facing a company based on its ticker symbol.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_company_competitors",
        "description": "Get the main competitors of a company based on its ticker symbol.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_next_earnings_date",
        "description": "Get the next earnings report date for a company based on its ticker symbol.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "check_50day_ma",
        "description": "Check if a company's stock price is above or below its 50-day moving average.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "calculate_RSI",
        "description": "Calculate the Relative Strength Index (RSI) for a given stock ticker and window.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)",
                },
                "window": {
                    "type": "integer",
                    "description": "The number of days to use for the RSI calculation (default is 14).",
                    "default": 14,
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_stock_trend",
        "description": "Determine if a company's stock is in an uptrend or downtrend based on recent price movement.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)",
                },
                "period": {
                    "type": "string",
                    "description": 'The time period to consider for the trend analysis (e.g., "60d" for 60 days, "1y" for 1 year). Default is "60d".',
                    "default": "60d",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "check_head_shoulders_pattern",
        "description": "Check if a company's stock has formed a head and shoulders pattern based on recent price data.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)",
                },
                "period": {
                    "type": "string",
                    "description": 'The time period to consider for the pattern detection (e.g., "1y" for 1 year, "6m" for 6 months). Default is "1y".',
                    "default": "1y",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "plot_stock_with_macd",
        "description": "Generate a stock price chart with MACD indicator for a given ticker symbol.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)",
                },
                "period": {
                    "type": "string",
                    "description": 'The time period to consider (e.g., "1y" for 1 year, "6m" for 6 months). Default is "1y".',
                    "default": "1y",
                },
            },
            "required": ["ticker"],
        },
    },
    #    {
    #        'name': 'get_latest_market_news',
    #        'description': "Get a summary of the latest news articles affecting the stock market.",
    #        'parameters': {}
    #    },
    {
        "name": "compare_valuations",
        "description": "Compare the current market valuations of two companies.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker1": {
                    "type": "string",
                    "description": "The stock ticker symbol for the first company (e.g., AAPL for Apple)",
                },
                "ticker2": {
                    "type": "string",
                    "description": "The stock ticker symbol for the second company (e.g., MSFT for Microsoft)",
                },
            },
            "required": ["ticker1", "ticker2"],
        },
    },
    {
        "name": "compare_stocks",
        "description": "Compare two stocks to determine which might be a better investment.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker1": {
                    "type": "string",
                    "description": "The stock ticker symbol for the first company (e.g., AAPL for Apple)",
                },
                "ticker2": {
                    "type": "string",
                    "description": "The stock ticker symbol for the second company (e.g., MSFT for Microsoft)",
                },
            },
            "required": ["ticker1", "ticker2"],
        },
    },
    {
        "name": "get_top_sector_stocks",
        "description": "Get the top-performing stocks in a specific sector.",
        "parameters": {
            "type": "object",
            "properties": {
                "sector": {
                    "type": "string",
                    "description": "The sector to analyze (e.g., Technology, Healthcare, Financials, etc.)",
                },
                "period": {
                    "type": "string",
                    "description": 'The time period to consider (e.g., "1mo" for 1 month, "3mo" for 3 months, "1y" for 1 year). Default is "1mo".',
                    "default": "1mo",
                },
                "top_n": {
                    "type": "integer",
                    "description": "The number of top stocks to return. Default is 10.",
                    "default": 10,
                },
            },
            "required": ["sector"],
        },
    },
    {
        "name": "get_potential_growth_stocks",
        "description": "Identify potential growth stocks in a specified industry.",
        "parameters": {
            "type": "object",
            "properties": {
                "industry": {
                    "type": "string",
                    "description": 'The target industry (e.g., "Technology", "Biotechnology", "Software")',
                },
                "top_n": {
                    "type": "integer",
                    "description": "Number of top stocks to return (default is 5)",
                    "default": 5,
                },
            },
            "required": ["industry"],
        },
    },
    #    {
    #        'name': 'manage_portfolio',
    #        'description': "Manage your stock portfolio by adding or removing stocks.",
    #        'parameters': {
    #            'type': 'object',
    #            'properties': {
    #                'action': {
    #                    'type': 'string',
    #                    'description': "The action to perform: 'add', 'remove', or 'list'.",
    #                    'enum': ['add', 'remove', 'list']
    #                },
    #                'ticker': {
    #                    'type': 'string',
    #                    'description': 'The stock ticker symbol to add or remove (e.g., AAPL for Apple). Not needed for "list" action.',
    #                },
    #                'shares': {
    #                    'type': 'integer',
    #                    'description': 'The number of shares to add or remove. Default is 1.',
    #                    'default': 1
    #                }
    #            },
    #            'required': ['action'],
    #        }
    #    },
    #    {
    #        'name': 'display_portfolio',
    #        'description': "Display the list of company along with the number of stocks in your current stock portfolio.",
    #        'parameters': {}
    #    },
    #    {
    #        'name': 'get_portfolio_performance',
    #        'description': "Get the overall performance of your stock portfolio.",
    #        'parameters': {}
    #    },
    #    {
    #        'name': 'get_best_performing_stock',
    #        'description': "Identify your best-performing stock in the portfolio.",
    #        'parameters': {}
    #    },
    #    {
    #        'name': 'analyze_portfolio_performance',
    #        'description': "Provide a comprehensive analysis of your portfolio's performance over the past year.",
    #        'parameters': {}
    #    },
    {
        "name": "manage_portfolio",
        "description": "Manage your stock portfolio by adding or removing stocks.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "The action to perform: 'add', 'remove', or 'list'.",
                    "enum": ["add", "remove", "list"],
                },
                "ticker": {
                    "type": "string",
                    "description": 'The stock ticker symbol to add or remove (e.g., AAPL for Apple). Not needed for "list" action.',
                },
                "shares": {
                    "type": "integer",
                    "description": "The number of shares to add or remove. Default is 1.",
                    "default": 1,
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "display_portfolio",
        "description": "Display the list of companies along with the number of stocks in your current stock portfolio.",
        "parameters": {},
    },
    {
        "name": "calculate_portfolio_performance",
        "description": "Calculate and report the overall performance of your stock portfolio.",
        "parameters": {
            "type": "object",
            "properties": {
                "period": {
                    "type": "string",
                    "description": 'The time period to consider (e.g., "1y" for 1 year, "6m" for 6 months, "1mo" for 1 month). Default is "1y".',
                    "default": "1y",
                }
            },
        },
    },
    {
        "name": "get_best_performing_stock",
        "description": "Identify your best-performing stock in the portfolio.",
        "parameters": {},
    },
    {
        "name": "get_most_volatile_stocks",
        "description": "Identify the most volatile stocks in your portfolio.",
        "parameters": {
            "type": "object",
            "properties": {
                "top_n": {
                    "type": "integer",
                    "description": "The number of most volatile stocks to return. Default is 3.",
                    "default": 3,
                }
            },
        },
    },
    {
        "name": "analyze_sector_allocation",
        "description": "Analyze how much of your portfolio is invested in each sector.",
        "parameters": {},
    },
    {
        "name": "should_rebalance_portfolio",
        "description": "Determine if you should rebalance your portfolio based on current allocations.",
        "parameters": {},
    },
    {
        "name": "get_latest_market_news",
        "description": "Get a summary of the latest news articles affecting the stock market.",
        "parameters": {},
    },
    {
        "name": "get_top_trending_stocks",
        "description": "Get a list of the top trending stocks based on news articles.",
        "parameters": {},
    },
    {
        "name": "get_market_valuation",
        "description": "Get the latest news on whether the market is overvalued or undervalued.",
        "parameters": {},
    },
    {
        "name": "get_economic_events_impacting_market",
        "description": "Get a list of economic events that could impact the stock market this week.",
        "parameters": {},
    },
    {
        "name": "get_analyst_opinions",
        "description": "Get the latest analyst opinions and news for a given stock ticker.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (e.g., AAPL for Apple).",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_company_news",
        "description": "Get the latest news articles about a specific company.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (e.g., MSFT for Microsoft).",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_income_statement",
        "description": "Gets the income statement given the ticker symbol of a company.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (e.g., AAPL for Apple).",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_balance_sheet",
        "description": "Gets the balance sheet given the ticker symbol of a company.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (for example AAPL for Apple).",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_cash_flow",
        "description": "Gets the cash flow statement given the ticker symbol of a company.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (for example AAPL for Apple).",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_major_holders",
        "description": "Gets the major holders given the ticker symbol of a company.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (for example AAPL for Apple).",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_institutional_holders",
        "description": "Gets the institutional holders given the ticker symbol of a company.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (for example AAPL for Apple).",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_mutualfund_holders",
        "description": "Gets the mutual fund holders given the ticker symbol of a company.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (for example AAPL for Apple).",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_insider_transactions",
        "description": "Gets the insider transactions given the ticker symbol of a company.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (for example AAPL for Apple).",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_insider_purchases",
        "description": "Gets the insider purchases given the ticker symbol of a company.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (for example AAPL for Apple).",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_recommendations",
        "description": "Gets the recommendations given the ticker symbol of a company.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (for example AAPL for Apple).",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_recommendations_summary",
        "description": "Gets the recommendations summary given the ticker symbol of a company.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (for example AAPL for Apple).",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_future_earnings",
        "description": "Gets the future earnings dates given the ticker symbol of a company.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (for example AAPL for Apple).",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_news",
        "description": "Gets the news given the ticker symbol of a company.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (for example AAPL for Apple).",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_recent_actions",
        "description": "Gets the recent actions of a company given its ticker symbol.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (for example AAPL for Apple).",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_stock_split",
        "description": "Gets the stock split data of a company given its ticker symbol.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (for example AAPL for Apple).",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "plot_stock_price",
        "description": "Generate a stock price chart for a given ticker symbol.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)",
                },
                "period": {
                    "type": "integer",
                    "description": "The number of years to plot (e.g., 3 for 3 years)",
                },
                "year": {
                    "type": "integer",
                    "description": "The specific year to plot (e.g., 2022 for the year 2022)",
                },
                "start_year": {
                    "type": "integer",
                    "description": "The start year to plot (e.g., 2020)",
                },
                "end_year": {
                    "type": "integer",
                    "description": "The end year to plot (e.g., 2023)",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "plot_ordinary_shares_number",
        "description": "Plot the ordinary shares number for a given ticker symbol.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "plot_share_issued",
        "description": "Generate a chart of the shares issued for a given ticker symbol.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "plot_net_debt",
        "description": "Generate a chart of the net debt for a given ticker symbol.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "plot_total_debt",
        "description": "Generate a chart of the total debt for a given ticker symbol.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "plot_tangible_book_value",
        "description": "Generate a chart of the tangible book value for a given ticker symbol.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "plot_invested_capital",
        "description": "Generate a chart of the invested capital for a given ticker symbol.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "plot_working_capital",
        "description": "Generate a chart of the working capital for a given ticker symbol.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "plot_net_tangible_assets",
        "description": "Generate a chart of the net tangible assets for a given ticker symbol.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "plot_capital_lease_obligations",
        "description": "Generate a chart of the capital lease obligations for a given ticker symbol.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "plot_free_cash_flow",
        "description": "Generate a Free Cash Flow chart for a given ticker symbol.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)",
                },
                "period": {
                    "type": "string",
                    "description": "The period for the data (e.g., 'annual' or 'quarterly')",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "plot_repurchase_of_capital_stock",
        "description": "Generate a Repurchase of Capital Stock chart for a given ticker symbol.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)",
                },
                "period": {
                    "type": "string",
                    "description": "The period for the data (e.g., 'annual' or 'quarterly')",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "plot_repayment_of_debt",
        "description": "Generate a Repayment of Debt chart for a given ticker symbol.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)",
                },
                "period": {
                    "type": "string",
                    "description": "The period for the data (e.g., 'annual' or 'quarterly')",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "plot_issuance_of_debt",
        "description": "Generate an Issuance of Debt chart for a given ticker symbol.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)",
                },
                "period": {
                    "type": "string",
                    "description": "The period for the data (e.g., 'annual' or 'quarterly')",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "plot_capital_expenditure",
        "description": "Generate a Capital Expenditure chart for a given ticker symbol.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)",
                },
                "period": {
                    "type": "string",
                    "description": "The period for the data (e.g., 'annual' or 'quarterly')",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "plot_interest_paid",
        "description": "Generate an Interest Paid chart for a given ticker symbol.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)",
                },
                "period": {
                    "type": "string",
                    "description": "The period for the data (e.g., 'annual' or 'quarterly')",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "plot_income_tax_paid",
        "description": "Generate an Income Tax Paid chart for a given ticker symbol.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)",
                },
                "period": {
                    "type": "string",
                    "description": "The period for the data (e.g., 'annual' or 'quarterly')",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "plot_ending_cash_position",
        "description": "Generate an Ending Cash Position chart for a given ticker symbol.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)",
                },
                "period": {
                    "type": "string",
                    "description": "The period for the data (e.g., 'annual' or 'quarterly')",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "plot_beginning_cash_position",
        "description": "Generate a Beginning Cash Position chart for a given ticker symbol.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)",
                },
                "period": {
                    "type": "string",
                    "description": "The period for the data (e.g., 'annual' or 'quarterly')",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "plot_change_in_cash",
        "description": "Generate a Change in Cash chart for a given ticker symbol.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)",
                },
                "period": {
                    "type": "string",
                    "description": "The period for the data (e.g., 'annual' or 'quarterly')",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "plot_ebitda",
        "description": "Generate a chart for EBITDA for a given ticker symbol.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)",
                },
                "period": {
                    "type": "string",
                    "description": "The period to plot (e.g., '5y' for 5 years)",
                },
            },
            "required": ["ticker", "period"],
        },
    },
    {
        "name": "plot_ebit",
        "description": "Generate a chart for EBIT for a given ticker symbol.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)",
                },
                "period": {
                    "type": "string",
                    "description": "The period to plot (e.g., '5y' for 5 years)",
                },
            },
            "required": ["ticker", "period"],
        },
    },
    {
        "name": "plot_net_income_from_continuing_operations_net_minority_interest",
        "description": "Generate a chart for Net Income from Continuing Operations Net Minority Interest for a given ticker symbol.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)",
                },
                "period": {
                    "type": "string",
                    "description": "The period to plot (e.g., '5y' for 5 years)",
                },
            },
            "required": ["ticker", "period"],
        },
    },
    {
        "name": "plot_depreciation",
        "description": "Generate a chart for Depreciation for a given ticker symbol.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)",
                },
                "period": {
                    "type": "string",
                    "description": "The period to plot (e.g., '5y' for 5 years)",
                },
            },
            "required": ["ticker", "period"],
        },
    },
    {
        "name": "plot_cost_of_revenue",
        "description": "Generate a chart for Cost Of Revenue for a given ticker symbol.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)",
                },
                "period": {
                    "type": "string",
                    "description": "The period to plot (e.g., '5y' for 5 years)",
                },
            },
            "required": ["ticker", "period"],
        },
    },
    {
        "name": "plot_tax_effect_of_unusual_items",
        "description": "Generate a chart for Tax Effect of Unusual Items for a given ticker symbol.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)",
                },
                "period": {
                    "type": "string",
                    "description": "The period to plot (e.g., '5y' for 5 years)",
                },
            },
            "required": ["ticker", "period"],
        },
    },
    {
        "name": "plot_total_unusual_items",
        "description": "Generate a chart for Total Unusual Items for a given ticker symbol.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)",
                },
                "period": {
                    "type": "string",
                    "description": "The period to plot (e.g., '5y' for 5 years)",
                },
            },
            "required": ["ticker", "period"],
        },
    },
    {
        "name": "plot_two_shares_side_by_side",
        "description": "Generate a line chart for two given ticker symbols to compare their stock prices over various timelines.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker1": {
                    "type": "string",
                    "description": "The stock ticker symbol for the first company (e.g., AAPL for Apple)",
                },
                "ticker2": {
                    "type": "string",
                    "description": "The stock ticker symbol for the second company (e.g., MSFT for Microsoft)",
                },
                "start_date": {
                    "type": "string",
                    "description": "The start date for the data (e.g., '2020-01-01')",
                },
                "end_date": {
                    "type": "string",
                    "description": "The end date for the data (e.g., '2023-12-31')",
                },
                "interval": {
                    "type": "string",
                    "description": "The interval for the data (e.g., '1d', '1wk', '1mo')",
                },
            },
            "required": ["ticker1", "ticker2", "start_date", "end_date", "interval"],
        },
    },
    {
        "name": "plot_companies_in_sector",
        "description": "Generate a comparison chart for the stock prices of companies within a specified sector.",
        "parameters": {
            "type": "object",
            "properties": {
                "sector": {
                    "type": "string",
                    "description": "The sector name (e.g., 'it', 'medical')",
                },
                "start_date": {
                    "type": "string",
                    "description": "The start date for the data (e.g., '2020-01-01')",
                },
                "end_date": {
                    "type": "string",
                    "description": "The end date for the data (e.g., '2023-12-31')",
                },
                "interval": {
                    "type": "string",
                    "description": "The data interval (e.g., '1d' for daily, '1wk' for weekly, '1mo' for monthly). Default is '1d'.",
                    "default": "1d",
                },
            },
            "required": ["sector", "start_date", "end_date"],
        },
    },
]

available_functions = {
    "get_nse_ticker": get_nse_ticker,
    "get_stock_price": get_stock_price,
    "calculate_SMA": calculate_SMA,
    "calculate_EMA": calculate_EMA,
    "get_stock_open_price": get_stock_open_price,
    "get_stock_price_movement": get_stock_price_movement,
    "get_stock_trading_volume": get_stock_trading_volume,
    "get_52_week_high": get_52_week_high,
    "get_52_week_low": get_52_week_low,
    "plot_stock_price": plot_stock_price,
    "compare_stock_sp500": compare_stock_sp500,
    "get_pe_ratio": get_pe_ratio,
    "get_dividend_info": get_dividend_info,
    "get_market_cap": get_market_cap,
    "get_stock_type": get_stock_type,
    "get_top_company_risks": get_top_company_risks,
    "get_company_competitors": get_company_competitors,
    "get_next_earnings_date": get_next_earnings_date,
    "check_50day_ma": check_50day_ma,
    "calculate_RSI": calculate_RSI,
    "get_stock_trend": get_stock_trend,
    "check_head_shoulders_pattern": check_head_shoulders_pattern,
    "plot_stock_with_macd": plot_stock_with_macd,
    "compare_valuations": compare_valuations,
    "compare_stocks": compare_stocks,
    "get_top_sector_stocks": get_top_sector_stocks,
    "get_potential_growth_stocks": get_potential_growth_stocks,
    #    'manage_portfolio': manage_portfolio,
    #    'display_portfolio': display_portfolio,
    #    'get_portfolio_performance': get_portfolio_performance,
    #    'get_best_performing_stock': get_best_performing_stock,
    #    'analyze_portfolio_performance': analyze_portfolio_performance,
    "manage_portfolio": manage_portfolio,
    "display_portfolio": display_portfolio,
    "calculate_portfolio_performance": calculate_portfolio_performance,
    "get_best_performing_stock": get_best_performing_stock,
    "get_most_volatile_stocks": get_most_volatile_stocks,
    "analyze_sector_allocation": analyze_sector_allocation,
    "should_rebalance_portfolio": should_rebalance_portfolio,
    "get_latest_market_news": get_latest_market_news,
    "get_top_trending_stocks": get_top_trending_stocks,
    "get_market_valuation": get_market_valuation,
    "get_economic_events_impacting_market": get_economic_events_impacting_market,
    "get_analyst_opinions": get_analyst_opinions,
    "get_company_news": get_company_news,
    "get_income_statement": get_income_statement,
    "get_balance_sheet": get_balance_sheet,
    "get_cash_flow": get_cash_flow,
    "get_major_holders": get_major_holders,
    "get_institutional_holders": get_institutional_holders,
    "get_mutualfund_holders": get_mutualfund_holders,
    "get_insider_transactions": get_insider_transactions,
    "get_insider_purchases": get_insider_purchases,
    "get_recommendations": get_recommendations,
    "get_recommendations_summary": get_recommendations_summary,
    "get_future_earnings": get_future_earnings,
    "get_news": get_news,
    "get_recent_actions": get_recent_actions,
    "get_stock_split": get_stock_split,
    "plot_bar_chart": plot_bar_chart,
    "plot_ordinary_shares_number": plot_ordinary_shares_number,
    "plot_share_issued": plot_share_issued,
    "plot_net_debt": plot_net_debt,
    "plot_total_debt": plot_total_debt,
    "plot_tangible_book_value": plot_tangible_book_value,
    "plot_invested_capital": plot_invested_capital,
    "plot_working_capital": plot_working_capital,
    "plot_net_tangible_assets": plot_net_tangible_assets,
    "plot_capital_lease_obligations": plot_capital_lease_obligations,
    "plot_free_cash_flow": plot_free_cash_flow,
    "plot_repayment_of_debt": plot_repayment_of_debt,
    "plot_issuance_of_debt": plot_issuance_of_debt,
    "plot_capital_expenditure": plot_capital_expenditure,
    "plot_interest_paid": plot_interest_paid,
    "plot_income_tax_paid": plot_income_tax_paid,
    "plot_ending_cash_position": plot_ending_cash_position,
    "plot_beginning_cash_position": plot_beginning_cash_position,
    "plot_change_in_cash": plot_change_in_cash,
    "plot_ebitda": plot_ebitda,
    "plot_ebit": plot_ebit,
    "plot_net_income_from_continuing_operations_net_minority_interest": plot_net_income_from_continuing_operations_net_minority_interest,
    "plot_depreciation": plot_depreciation,
    "plot_cost_of_revenue": plot_cost_of_revenue,
    "plot_tax_effect_of_unusual_items": plot_tax_effect_of_unusual_items,
    "plot_total_unusual_items": plot_total_unusual_items,
    "plot_two_shares_side_by_side": plot_two_shares_side_by_side,
    "plot_companies_in_sector": plot_companies_in_sector,
}


def get_response(query, chat_history):
    st.session_state["messages"] = [
        {
            "role": "system",
            "content": "You are a helpful financial advisor named prabot, that explains concepts in an easy-to-understand way. You can also help in executing python programs related to finance calculations and also show the steps inside proper expression box.",
        }
    ]
    st.session_state["messages"].extend(
        [
            {
                "role": "user" if isinstance(message, HumanMessage) else "assistant",
                "content": message.content,
            }
            for message in chat_history
        ]
    )

    user_input = query

    if user_input:
        try:
            st.session_state["messages"].append(
                {"role": "user", "content": f"{user_input}"}
            )

            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=st.session_state["messages"],
                functions=functions,
                function_call="auto",
            )

            response_message = response.choices[0].message

            if response_message.function_call:
                function_name = response_message.function_call.name
                function_args = json.loads(response_message.function_call.arguments)
                args_dict = {}
                function_to_call = available_functions[
                    function_name
                ]  # Move this line to the top

                if function_name == "plot_stock_with_macd":
                    args_dict = {"ticker": function_args.get("ticker")}
                    function_response = function_to_call(**args_dict)
                    st.image("stock_with_macd.png")
                elif function_name == "plot_stock_price":
                    args_dict = {
                        "ticker": function_args.get("ticker"),
                        "period": function_args.get("period"),
                        "year": function_args.get("year"),
                        "start_year": function_args.get("start_year"),
                        "end_year": function_args.get("end_year"),
                    }
                    function_response = function_to_call(**args_dict)
                    st.image("stock.png")
                elif function_name == "plot_ordinary_shares_number":
                    args_dict = {"ticker": function_args.get("ticker")}
                    function_response = plot_ordinary_shares_number(**args_dict)
                    st.image("ordinary_shares_number.png")
                elif function_name == "plot_share_issued":
                    args_dict = {"ticker": function_args.get("ticker")}
                    function_response = plot_share_issued(**args_dict)
                    st.image("share_issued.png")
                elif function_name == "plot_net_debt":
                    args_dict = {"ticker": function_args.get("ticker")}
                    function_response = plot_net_debt(**args_dict)
                    st.image("net_debt.png")
                elif function_name == "plot_total_debt":
                    args_dict = {"ticker": function_args.get("ticker")}
                    function_response = plot_total_debt(**args_dict)
                    st.image("total_debt.png")
                elif function_name == "plot_tangible_book_value":
                    args_dict = {"ticker": function_args.get("ticker")}
                    function_response = plot_tangible_book_value(**args_dict)
                    st.image("tangible_book_value.png")
                elif function_name == "plot_invested_capital":
                    args_dict = {"ticker": function_args.get("ticker")}
                    function_response = plot_invested_capital(**args_dict)
                    st.image("invested_capital.png")
                elif function_name == "plot_working_capital":
                    args_dict = {"ticker": function_args.get("ticker")}
                    function_response = plot_working_capital(**args_dict)
                    st.image("working_capital.png")
                elif function_name == "plot_net_tangible_assets":
                    args_dict = {"ticker": function_args.get("ticker")}
                    function_response = plot_net_tangible_assets(**args_dict)
                    st.image("net_tangible_assets.png")
                elif function_name == "plot_capital_lease_obligations":
                    args_dict = {"ticker": function_args.get("ticker")}
                    function_response = plot_capital_lease_obligations(**args_dict)
                    st.image("capital_lease_obligations.png")
                elif function_name == "plot_free_cash_flow":
                    args_dict = {
                        "ticker": function_args.get("ticker"),
                        "period": function_args.get("period"),
                    }
                    function_response = function_to_call(**args_dict)
                    st.image("free_cash_flow.png")
                elif function_name == "plot_repurchase_of_capital_stock":
                    args_dict = {
                        "ticker": function_args.get("ticker"),
                        "period": function_args.get("period"),
                    }
                    function_response = function_to_call(**args_dict)
                    st.image("repurchase_of_capital_stock.png")
                elif function_name == "plot_repayment_of_debt":
                    args_dict = {
                        "ticker": function_args.get("ticker"),
                        "period": function_args.get("period"),
                    }
                    function_response = function_to_call(**args_dict)
                    st.image("repayment_of_debt.png")
                elif function_name == "plot_issuance_of_debt":
                    args_dict = {
                        "ticker": function_args.get("ticker"),
                        "period": function_args.get("period"),
                    }
                    function_response = function_to_call(**args_dict)
                    st.image("issuance_of_debt.png")
                elif function_name == "plot_capital_expenditure":
                    args_dict = {
                        "ticker": function_args.get("ticker"),
                        "period": function_args.get("period"),
                    }
                    function_response = function_to_call(**args_dict)
                    st.image("capital_expenditure.png")
                elif function_name == "plot_interest_paid":
                    args_dict = {
                        "ticker": function_args.get("ticker"),
                        "period": function_args.get("period"),
                    }
                    function_response = function_to_call(**args_dict)
                    st.image("interest_paid.png")
                elif function_name == "plot_income_tax_paid":
                    args_dict = {
                        "ticker": function_args.get("ticker"),
                        "period": function_args.get("period"),
                    }
                    function_response = function_to_call(**args_dict)
                    st.image("income_tax_paid.png")
                elif function_name == "plot_ending_cash_position":
                    args_dict = {
                        "ticker": function_args.get("ticker"),
                        "period": function_args.get("period"),
                    }
                    function_response = function_to_call(**args_dict)
                    st.image("ending_cash_position.png")
                elif function_name == "plot_beginning_cash_position":
                    args_dict = {
                        "ticker": function_args.get("ticker"),
                        "period": function_args.get("period"),
                    }
                    function_response = function_to_call(**args_dict)
                    st.image("beginning_cash_position.png")
                elif function_name == "plot_change_in_cash":
                    args_dict = {
                        "ticker": function_args.get("ticker"),
                        "period": function_args.get("period"),
                    }
                    function_response = function_to_call(**args_dict)
                    st.image("change_in_cash.png")
                elif function_name == "plot_ebitda":
                    args_dict = {
                        "ticker": function_args.get("ticker"),
                        "period": function_args.get("period"),
                    }
                    function_response = function_to_call(**args_dict)
                    st.image("ebitda.png")
                elif function_name == "plot_ebit":
                    args_dict = {
                        "ticker": function_args.get("ticker"),
                        "period": function_args.get("period"),
                    }
                    function_response = function_to_call(**args_dict)
                    st.image("ebit.png")
                elif (
                    function_name
                    == "plot_net_income_from_continuing_operations_net_minority_interest"
                ):
                    args_dict = {
                        "ticker": function_args.get("ticker"),
                        "period": function_args.get("period"),
                    }
                    function_response = function_to_call(**args_dict)
                    st.image(
                        "net_income_from_continuing_operations_net_minority_interest.png"
                    )
                elif function_name == "plot_depreciation":
                    args_dict = {
                        "ticker": function_args.get("ticker"),
                        "period": function_args.get("period"),
                    }
                    function_response = function_to_call(**args_dict)
                    st.image("depreciation.png")
                # elif function_name == "plot_cost_of_revenue":
                # args_dict = {
                # "ticker": function_args.get("ticker"),
                # "period": function_args.get("period"),
                # }
                # function_response = plot_cost_of_revenue(**args_dict)
                # st.image("cost_of_revenue.png")
                elif function_name == "plot_cost_of_revenue":
                    args_dict = {
                        "ticker": function_args.get("ticker"),
                        "period": function_args.get("period"),
                    }
                    function_response = function_to_call(**args_dict)
                    if os.path.exists("cost_of_revenue.png"):
                        st.image("cost_of_revenue.png")
                    else:
                        st.error("The cost_of_revenue.png file was not found.")
                elif function_name == "plot_tax_effect_of_unusual_items":
                    args_dict = {
                        "ticker": function_args.get("ticker"),
                        "period": function_args.get("period"),
                    }
                    function_response = function_to_call(**args_dict)
                    st.image("tax_effect_of_unusual_items.png")
                elif function_name == "plot_total_unusual_items":
                    args_dict = {
                        "ticker": function_args.get("ticker"),
                        "period": function_args.get("period"),
                    }
                    function_response = function_to_call(**args_dict)
                    st.image("total_unusual_items.png")
                elif function_name == "plot_two_shares_side_by_side":
                    args_dict = {
                        "ticker1": function_args.get("ticker1"),
                        "ticker2": function_args.get("ticker2"),
                        "start_date": function_args.get("start_date"),
                        "end_date": function_args.get("end_date"),
                        "interval": function_args.get("interval"),
                    }
                    function_response = function_to_call(**args_dict)
                    st.image("two_shares_side_by_side.png")
                elif function_name == "plot_companies_in_sector":
                    args_dict = {
                        "sector": function_args.get("sector"),
                        "start_date": function_args.get("start_date"),
                        "end_date": function_args.get("end_date"),
                        "interval": function_args.get("interval", "1d"),
                    }
                    function_response = function_to_call(**args_dict)
                    st.image("sector_comparison_chart.png")
                else:
                    if function_name in [
                        "get_stock_price",
                        "calculate_RSI",
                        "calculate_MACD",
                    ]:
                        args_dict = {"ticker": function_args.get("ticker")}
                    elif function_name in ["calculate_SMA", "calculate_EMA"]:
                        args_dict = {
                            "ticker": function_args.get("ticker"),
                            "window": function_args.get("window"),
                        }
                    elif function_name == "get_stock_price_movement":
                        args_dict = {"ticker": function_args.get("ticker")}
                    elif (
                        function_name == "get_stock_open_price"
                    ):  # Fix for get_stock_open_price
                        args_dict = {"ticker": function_args.get("ticker")}
                    elif function_name == "get_stock_trading_volume":
                        args_dict = {"ticker": function_args.get("ticker")}
                    elif function_name == "get_52_week_high":
                        args_dict = {"ticker": function_args.get("ticker")}
                    elif function_name == "get_52_week_low":
                        args_dict = {"ticker": function_args.get("ticker")}
                    elif function_name == "compare_stock_sp500":
                        args_dict = {"ticker": function_args.get("ticker")}
                    elif function_name == "get_pe_ratio":
                        args_dict = {"ticker": function_args.get("ticker")}
                    elif function_name == "get_dividend_info":
                        args_dict = {"ticker": function_args.get("ticker")}
                    elif function_name == "get_market_cap":
                        args_dict = {"ticker": function_args.get("ticker")}
                    elif function_name == "get_stock_type":
                        args_dict = {"ticker": function_args.get("ticker")}
                    elif function_name == "get_top_company_risks":
                        args_dict = {"ticker": function_args.get("ticker")}
                    elif function_name == "get_company_competitors":
                        args_dict = {"ticker": function_args.get("ticker")}
                    elif function_name == "get_next_earnings_date":
                        args_dict = {"ticker": function_args.get("ticker")}
                    elif function_name == "check_50day_ma":
                        args_dict = {"ticker": function_args.get("ticker")}
                    elif function_name == "calculate_RSI":
                        args_dict = {"ticker": function_args.get("ticker")}
                    elif function_name == "get_stock_trend":
                        args_dict = {"ticker": function_args.get("ticker")}
                    elif function_name == "check_head_shoulders_pattern":
                        args_dict = {"ticker": function_args.get("ticker")}
                        args_dict = {
                            "ticker1": function_args.get("ticker1"),
                            "ticker2": function_args.get("ticker2"),
                        }
                        function_response = function_to_call(**args_dict)
                    elif function_name == "compare_stocks":
                        args_dict = {
                            "ticker1": function_args.get("ticker1"),
                            "ticker2": function_args.get("ticker2"),
                        }
                        function_response = function_to_call(**args_dict)
                    elif (
                        function_name == "get_top_sector_stocks"
                    ):  # New condition for your function
                        args_dict = {
                            "sector": function_args.get("sector"),
                            "period": function_args.get("period", "1mo"),
                            "top_n": function_args.get("top_n", 10),
                        }
                        function_response = function_to_call(**args_dict)
                    elif (
                        function_name == "get_potential_growth_stocks"
                    ):  # New condition for your function
                        args_dict = {
                            "industry": function_args.get("industry"),
                            "top_n": function_args.get("top_n", 5),
                        }
                        function_response = function_to_call(**args_dict)
                    elif function_name == "manage_portfolio":
                        args_dict = {
                            "action": function_args.get("action"),
                            "ticker": function_args.get("ticker"),
                            "shares": function_args.get("shares", 1),
                        }
                    elif function_name == "display_portfolio":
                        args_dict = {}
                    elif function_name == "calculate_portfolio_performance":
                        args_dict = {"period": function_args.get("period", "1y")}
                    elif function_name == "get_most_volatile_stocks":
                        args_dict = {"top_n": function_args.get("top_n", 3)}
                    elif function_name in [
                        "get_best_performing_stock",
                        "analyze_sector_allocation",
                    ]:
                        args_dict = {}
                    elif function_name == "analyze_sector_allocation":
                        st.image("sector_allocation.png")
                    elif function_name == "should_rebalance_portfolio":
                        args_dict = {}
                    elif function_name == "get_latest_market_news":
                        function_response = function_to_call()
                        st.write(function_response)
                    elif function_name == "get_top_trending_stocks":
                        function_response = function_to_call()
                        st.write(function_response)
                    elif function_name == "get_market_valuation":
                        function_response = function_to_call()
                        st.write(function_response)
                    elif function_name == "get_economic_events_impacting_market":
                        function_response = function_to_call()
                        st.write(function_response)
                    elif function_name == "get_analyst_opinions":
                        args_dict = {"ticker": function_args.get("ticker")}
                        function_response = function_to_call(**args_dict)
                        st.write(function_response)
                    elif function_name == "get_company_news":
                        args_dict = {"ticker": function_args.get("ticker")}
                        function_response = function_to_call(**args_dict)
                    elif function_name in [
                        "get_income_statement",
                        "get_balance_sheet",
                        "get_cash_flow",
                    ]:
                        args_dict = {"ticker": function_args.get("ticker")}
                        function_response = function_to_call(**args_dict)
                    elif function_name in [
                        "get_major_holders",
                        "get_institutional_holders",
                        "get_mutualfund_holders",
                    ]:
                        args_dict = {"ticker": function_args.get("ticker")}
                    elif function_name in [
                        "get_insider_transactions",
                        "get_insider_purchases",
                    ]:
                        args_dict = {"ticker": function_args.get("ticker")}
                    elif function_name in [
                        "get_recommendations",
                        "get_recommendations_summary",
                    ]:
                        args_dict = {"ticker": function_args.get("ticker")}
                    elif function_name == "get_future_earnings":
                        args_dict = {"ticker": function_args.get("ticker")}
                    elif function_name == "get_news":
                        args_dict = {"ticker": function_args.get("ticker")}
                    elif function_name == "get_recent_actions":
                        args_dict = {"ticker": function_args.get("ticker")}
                    elif function_name == "get_stock_split":
                        args_dict = {"ticker": function_args.get("ticker")}
                    else:
                        args_dict = function_args

                    function_response = function_to_call(**args_dict)
                    st.session_state["messages"].append(response_message)
                    st.session_state["messages"].append(
                        {
                            "role": "function",
                            "name": function_name,
                            "content": function_response,
                        }
                    )
                    second_response = openai.chat.completions.create(
                        model="gpt-4o",
                        messages=st.session_state["messages"],
                    )
                    response_text = second_response.choices[0].message.content
                    st.write(response_text)
                    st.session_state["messages"].append(
                        {"role": "assistant", "content": response_text}
                    )
            else:
                response_text = response_message.content
                st.write(response_text)
                st.session_state["messages"].append(
                    {"role": "assistant", "content": response_text}
                )

        except Exception as e:
            raise e

    return st.session_state["messages"]


def strip_formatting(text):
    return re.sub(r"\033\[[0-9;]*m", "", text)


for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)

user_query = st.chat_input("Your message")

if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        ai_response = get_response(user_query, st.session_state.chat_history)
        st.session_state.chat_history.append(AIMessage(ai_response[-1]["content"]))
