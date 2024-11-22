
import requests
from bs4 import BeautifulSoup

class WebScraper:
    """A simple web scraper for extracting text content."""

    @staticmethod
    def fetch_url(url: str) -> str:
        """Fetch HTML content from a URL."""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            return f"Error fetching URL: {e}"

    @staticmethod
    def parse_html(html: str) -> str:
        """Extract and clean text content from HTML."""
        try:
            soup = BeautifulSoup(html, "html.parser")
            return soup.get_text(separator=" ", strip=True)
        except Exception as e:
            return f"Error parsing HTML: {e}"

if __name__ == "__main__":
    # Example usage
    url = "https://example.com"
    html_content = WebScraper.fetch_url(url)
    text_content = WebScraper.parse_html(html_content)
    print(text_content)
