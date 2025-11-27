from playwright.async_api import async_playwright
import asyncio

async def render_page(url: str, timeout_ms: int = 60000):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()

        await page.goto(url, wait_until="networkidle", timeout=timeout_ms)
        await asyncio.sleep(1)

        html = await page.content()
        await browser.close()
        return html
