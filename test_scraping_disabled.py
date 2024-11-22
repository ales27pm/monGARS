
import sys
sys.path.insert(0, '/mnt/data/AutonomousAssistantProject/project')  # Add project path to sys.path

from app.utils.scraping_web import WebScraper
from app.utils.playwright_scraping import PlaywrightScraper
from app.utils.searx_integration import SearxSearch

# Test WebScraper
def test_web_scraper():
    html = WebScraper.fetch_url("https://example.com")
    assert "Example Domain" in WebScraper.parse_html(html)

# Test PlaywrightScraper
def test_playwright_scraper():
    html = PlaywrightScraper.fetch_dynamic_content("https://example.com")
    assert "Example Domain" in html

# Test SearxSearch
@pytest.mark.skip(reason="Requires a working SEARX instance.")
def test_searx_search():
    searx = SearxSearch(searx_url="https://searx.example.com")
    results = searx.search(query="test query", categories=["science"])
    assert len(results) > 0
