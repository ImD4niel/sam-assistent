import requests
from bs4 import BeautifulSoup
import markdownify
import logging

logger = logging.getLogger(__name__)

class BrowserTool:
    """
    A lightweight autonomous browser that can visit URLs and extract markdown content.
    """
    name = "browser"
    description = "Visits a website and reads its content. Input: 'url' (str)."

    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    async def execute(self, url: str) -> str:
        """Visits the URL and returns the content as Markdown."""
        try:
            logger.info(f"Browser visiting: {url}")
            if not url.startswith("http"):
                url = "https://" + url
                
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove junk elements
            for script in soup(["script", "style", "nav", "footer", "iframe", "svg"]):
                script.decompose()
                
            # Convert to Markdown
            clean_html = str(soup.body or soup)
            md_content = markdownify.markdownify(clean_html, heading_style="ATX")
            
            # Cleanup excessive newlines
            import re
            md_content = re.sub(r'\n{3,}', '\n\n', md_content).strip()
            
            # Truncate if too long (to fit context)
            if len(md_content) > 8000:
                md_content = md_content[:8000] + "\n\n[...Content Truncated...]"
                
            return md_content
        except Exception as e:
            logger.error(f"Browser error: {e}")
            return f"Failed to load page: {str(e)}"
