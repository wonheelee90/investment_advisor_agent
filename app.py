from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
from langchain.agents import Tool, create_openai_functions_agent, AgentExecutor
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import requests
from bs4 import BeautifulSoup

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# --- NEW TEST ROUTE ---
# This is a simple route to check if the server is running correctly.
@app.route('/', methods=['GET'])
def index():
    return "Server is running!"
# --- END NEW TEST ROUTE ---


# 1. Define Tools
def get_stock_price_target(ticker: str) -> str:
    """
    Scrapes MarketWatch for the analyst price target of a given stock ticker.
    The input should be a stock ticker symbol, e.g., 'AAPL', 'TSLA'.
    """
    try:
        url = f"https://www.marketwatch.com/investing/stock/{ticker}/analystestimates"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the element containing the price target
        # Note: This is brittle and may break if the website structure changes.
        target_element = soup.find('td', class_='table__cell positive', string='High')
        if target_element and target_element.find_next_sibling('td', class_='table__cell'):
            price_target = target_element.find_next_sibling('td', class_='table__cell').text.strip()
            return f"The high analyst price target for {ticker} is {price_target}."
        else:
            return f"Could not find the price target for {ticker} on MarketWatch."
    except Exception as e:
        return f"An error occurred while trying to fetch the stock price target for {ticker}: {str(e)}"

search = GoogleSearchAPIWrapper()

stock_price_tool = Tool(
    name="getStockPriceTarget",
    func=get_stock_price_target,
    description="Use this tool to get the analyst price target for a specific stock ticker. The input should be a stock ticker symbol, like 'AAPL' or 'TSLA'."
)

google_search_tool = Tool(
    name="GoogleSearch",
    func=search.run,
    description="Use this tool to find real-time information, news, and answer questions about the latest market trends, risers, and droppers. You can ask it things like 'Top rising stocks today' or 'Latest news on NVDA'."
)

tools = [stock_price_tool, google_search_tool]


# 2. Create the Agent
llm = ChatOpenAI(model="gpt-4", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful financial advisor. Your goal is to provide accurate and up-to-date information on the stock market."),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# 3. Define API Endpoints
@app.route('/query', methods=['GET', 'POST']) # Temporarily added 'GET' for debugging
def handle_query():
    # --- DEBUGGING BLOCK ---
    if request.method == 'GET':
        return "The /query endpoint is working! It's ready for POST requests."
    # --- END DEBUGGING BLOCK ---

    if not request.json or 'query' not in request.json:
        return jsonify({"error": "Missing query in request body"}), 400

    user_query = request.json['query']

    try:
        response = agent_executor.invoke({"input": user_query})
        return jsonify({"answer": response.get('output', 'No output from agent.')})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)

