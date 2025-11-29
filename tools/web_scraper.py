from langchain_core.tools import tool
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from urllib.parse import urljoin

@tool
def get_rendered_html(target_url: str) -> dict:
    """
    Fetch and return the fully rendered HTML of a webpage.
    """
    print("\nFetching and rendering:", target_url)
    try:
        with sync_playwright() as playwright_instance:
            browser_instance = playwright_instance.chromium.launch(headless=True)
            page_instance = browser_instance.new_page()

            page_instance.goto(target_url, wait_until="networkidle")
            html_content = page_instance.content()

            browser_instance.close()

            # Parse images
            html_parser = BeautifulSoup(html_content, "html.parser")
            image_urls = [urljoin(target_url, img["src"]) for img in html_parser.find_all("img", src=True)]
            if len(html_content) > 300000:
                    print("Warning: HTML too large, truncating...")
                    html_content = html_content[:300000] + "... [TRUNCATED DUE TO SIZE]"
            return {
                "html": html_content,
                "images": image_urls,
                "url": target_url
            }

    except Exception as error:
        return {"error": f"Error fetching/rendering page: {str(error)}"}


