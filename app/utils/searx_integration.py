
import requests

class SearxSearch:
    """A utility for performing searches using the SEARX API."""

    def __init__(self, searx_url: str):
        self.searx_url = searx_url

    def search(self, query: str, categories: list = None, max_results: int = 10) -> list:
        """Perform a search query and return results."""
        try:
            params = {
                "q": query,
                "format": "json",
                "categories": ",".join(categories) if categories else "general",
                "count": max_results,
            }
            response = requests.get(self.searx_url, params=params, timeout=10)
            response.raise_for_status()
            return response.json().get("results", [])
        except requests.RequestException as e:
            return [f"Error performing search: {e}"]

if __name__ == "__main__":
    # Example usage
    searx = SearxSearch(searx_url="https://searx.example.com")
    results = searx.search(query="test query", categories=["science"])
    for result in results:
        print(result["title"], "-", result["url"])
