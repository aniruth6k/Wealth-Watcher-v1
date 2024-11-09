# Wealth Watcher

Wealth Watcher is a Streamlit-based application that helps users monitor and analyze stock prices using advanced analytics, including AI-powered conversational functionality for easy access to financial information. Users can query stock prices, calculate indicators like SMA and EMA, and visualize stock trends over the past year.

## Features

- **Real-Time Stock Price Retrieval**: Get the latest stock price for a specified ticker symbol.
- **Historical Stock Price Plotting**: View a one-year historical plot for any stock ticker.
- **Conversational Interface**: AI-powered bot to answer finance-related queries in an easy-to-understand way.

## Getting Started

### Prerequisites

- **Python 3.10+**
- **Streamlit**: For interactive web application development.
- **LangChain**: To handle conversations with AI capabilities.
- **yfinance**: For stock price data retrieval.
- **dotenv**: To manage environment variables for API keys.
- **matplotlib**: For plotting stock price data.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/wealth-watcher.git
   cd wealth-watcher
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment variables in a `.env` file. You will need an OpenAI API key for the chatbot functionality:
   ```plaintext
   OPENAI_API_KEY=your_openai_api_key
   ```

### Running the Application

1. Start the application with Streamlit:
   ```bash
   streamlit run main.py
   ```

2. Open the displayed URL (usually `http://localhost:8501`) to interact with Wealth Watcher.

## Usage

- **Get Stock Price**: Type queries like, "What is the stock price of AAPL?"
- **Calculate SMA or EMA**: Type queries like, "Calculate SMA for AAPL with window 20."
- **Plot Stock Price**: Type, "Plot the stock price for AAPL," and the application will generate a one-year historical plot.

## File Structure

- **main.py**: The main script that handles user input, stock data retrieval, and interaction with the AI-powered bot.
- **requirements.txt**: Contains the dependencies required to run the application.

## Future Features

- An Android/IOS app is being made using flutter along with a finance tracker feature in it.
