import requests
import json
from typing import List, Dict, Any
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
import os

#loading env variables
load_dotenv()
SERPER_API_KEY = os.getenv("SERPER_API_KEY")


def fetch_google_news_rss(num: int = 10) -> List[Dict[str, Any]]:
    """Fetch the latest news articles from Google News RSS feed."""
    try:
        url = "https://news.google.com/rss"
        r = requests.get(url, timeout = 30)
        r.raise_for_status()

        root = ET.fromstring(r.content)
        items = root.findall(".//item")

        results = []
        for item in items[:num]:
            title = item.find("title")
            link = item.find("link")
            pub_date = item.find('pubDate')
            source = item.find('source')

            results.append({
                "title": title.text if title is not None else "No Title",
                "link": link.text if link is not None else "",
                "pub_date": pub_date.text if pub_date is not None else "No date",
                "source": source.text if source is not None else "Google News",
            })
        return results
    except Exception as e:
        return {"ok": False, "error": repr(e)}

def serper_news_search(query: str, num: int = 5) -> List[Dict[str, Any]]:
    """Search for news articles using Serper API."""
    url = "https://google.serper.dev/news"
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    payload = {
        "q": query,
        "gl": "us",
        "hl": "en",
        "tbs": "qdr:d",
    }
    try:
        r = requests.post(url, headers = headers, data = payload, timeout = 30)
        r.raise_for_status()
        data = r.json()
        results = []
        for item in data.get("news", [])[:num]:
            results.append({
                "title": item.get("title", "No Title"),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "date": item.get("date", "No date"),
                "source": item.get("source", "Unknown"),
            })
        return results
    except Exception as e:
        return {"ok": False, "error": repr(e)}
    
def serper_site_search(query: str, site: str, num: int = 5) -> List[Dict[str, Any]]:
    """Site restricted web search."""
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    payload = {"q": f"site:{site} {query}", "gl": "us", "hl": "en"}
    r = requests.post(url, headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    results = []
    for item in data.get("organic", [])[:num]:
        results.append({
            "title": item.get("title"),
            "link": item.get("link"),
            "snippet": item.get("snippet"),
            "favicons": item.get("favicons", {})
        })
    return results

# def fetch_article(url: str, max_chars: int = 12000) -> Dict[str, Any]:
#     """Fetch and extract clean article text with trafilatura."""
#     try:
#         downloaded = trafilatura.fetch_url(url, timeout=30)
#         text = trafilatura.extract(downloaded, include_comments=False) if downloaded else None
#         if not text:
#             return {"ok": False, "error": "could_not_extract"}
#         text = text.strip()
#         if len(text) > max_chars:
#             text = text[:max_chars] + " ..."
#         return {"ok": True, "text": text}
#     except Exception as e:
#         return {"ok": False, "error": repr(e)}

# OpenAI-style tool specs for function calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_google_news_rss",
            "description": "Fetch general top headlines from Google News RSS feed. Use this when you want to see what's happening in the world today without a specific topic focus.",
            "parameters": {
                "type": "object",
                "properties": {
                    "num": {"type": "integer", "minimum": 1, "maximum": 20, "description": "Number of news items to fetch"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "serper_news_search",
            "description": "Search Google News for articles about a specific topic or query. Use this when you need news about particular subjects, companies, or events.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "num": {"type": "integer", "minimum": 1, "maximum": 20}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "serper_site_search",
            "description": "Search a specific news domain for relevant articles.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "site": {"type": "string", "description": "Domain like ft.com or nytimes.com"},
                    "num": {"type": "integer", "minimum": 1, "maximum": 10}
                },
                "required": ["query", "site"]
            }
        }
    },
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "fetch_article",
    #         "description": "Download and extract the main text of an article from a URL. ONLY use this when the user asks specific questions about article content, details, or wants to analyze/quote from particular articles. Do NOT use this for general news summaries or overviews.",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "url": {"type": "string"},
    #                 "max_chars": {"type": "integer", "minimum": 1000, "maximum": 60000}
    #             },
    #             "required": ["url"]
    #         }
    #     }
    # }
]

FUNCTION_MAP = {
    "fetch_google_news_rss": fetch_google_news_rss,
    "serper_news_search": serper_news_search,
    "serper_site_search": serper_site_search,
    # "fetch_article": fetch_article,
}