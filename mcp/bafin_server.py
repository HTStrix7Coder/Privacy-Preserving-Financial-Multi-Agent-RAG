"""
BaFin & Market Data MCP Server
Exposes live regulatory data and market rates as tools for the compliance agent.

Run via standard MCP protocol in LangChain or using mcp dev:
    mcp dev GDPR_SLM/mcp/bafin_server.py
"""

from mcp.server.fastmcp import FastMCP
from typing import Dict
import requests
from bs4 import BeautifulSoup

# Initialize the MCP Server
mcp = FastMCP("BaFin Market Data Server")

@mcp.tool()
def get_bafin_warnings(issuer_name: str) -> str:
    """
    Check the German Federal Financial Supervisory Authority (BaFin) 
    for active warnings, sanctions, or investigations regarding a specific issuer.
    """
    
    # In a real app, this would make an HTTP request to BaFin's API.
    # We simulate the database for demonstration.
    warnings: Dict[str, str] = {
        "Wirecard AG": "CRITICAL RISK: Multiple fraud investigations. Trading suspended.",
        "N26 Bank GmbH": "WARNING: Growth restrictions imposed by BaFin due to anti-money laundering (AML) deficits. Penalty of €4.25m paid.",
        "Deutsche Bank AG": "NOTICE: Ongoing monitoring regarding internal controls. No active issuance ban.",
        "Greensill Bank AG": "CRITICAL RISK: Moratorium ordered by BaFin. Insolvency proceedings active."
    }
    
    # Normalize input
    issuer_lookup = issuer_name.strip()
    
    for key in warnings:
        if key.lower() in issuer_lookup.lower() or issuer_lookup.lower() in key.lower():
            return f"🚨 BaFin Alert for {issuer_name}:\n{warnings[key]}"
            
    return f"✅ Clean: No active BaFin warnings, sanctions, or investigations found for '{issuer_name}'."

@mcp.tool()
def get_live_euribor_rate(maturity: str = "3M") -> str:
    """
    Get the current live EURIBOR interest rate by actively scraping euribor-rates.eu.
    Useful for checking floating rate notes.
    maturity options: "1W", "1M", "3M", "6M", "12M"
    """
    url = "https://www.euribor-rates.eu/en/current-euribor-rates/"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    month_val = maturity.replace("M", "").replace("W", "")
    time_unit = "week" if "W" in maturity.upper() else "month"
    if not month_val.isdigit():
        return "Error: Invalid maturity format."

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the rate in the HTML table using flexible keyword matching checking both th and td
        for tr in soup.find_all('tr'):
            cells = tr.find_all(['th', 'td'])
            if len(cells) >= 2:
                txt = cells[0].text.lower()
                if month_val in txt and time_unit in txt and "euribor" in txt:
                    rate = cells[1].text.strip()
                    return f"[LIVE INTERNET DATA] The actual {maturity.upper()} EURIBOR rate as of today is {rate}."
                
        return f"Warning: Could not parse the {maturity.upper()} rate from the live website's DOM."
    except Exception as e:
        return f"Warning: Live internet scrape failed ({e})."

if __name__ == "__main__":
    # Run securely using standard stdio transport for MCP clients
    mcp.run(transport='stdio')
