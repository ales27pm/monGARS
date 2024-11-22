
from playwright.sync_api import sync_playwright

class PlaywrightScraper:
    """A robust scraper using Playwright for dynamic content."""

    @staticmethod
    def fetch_dynamic_content(url: str, timeout: int = 30) -> str:
        """Fetch and render dynamic content from a URL."""
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context()
                page = context.new_page()
                page.goto(url, timeout=timeout * 1000)
                content = page.content()
                browser.close()
                return content
            except Exception as e:
                return f"Error fetching dynamic content: {e}"

if __name__ == "__main__":
    # Example usage
    url = "https://example.com"
    html_content = PlaywrightScraper.fetch_dynamic_content(url)
    print(html_content)
