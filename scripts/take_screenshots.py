"""Capture README screenshots of every UI tab from a running app.

Usage (API and UI already running):
    uv run python -m scripts.take_screenshots --query-image path/to/photo.jpg

Drives the locally installed Google Chrome, so no Playwright browser download is needed.
"""

import argparse
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

# Streamlit scrolls an inner container, so full_page can't extend the capture: the viewport
# itself must be tall enough to show a whole result grid.
VIEWPORT = {"width": 1440, "height": 1500}
# Upper bound for API calls behind a click (CLIP inference, LLM round-trips).
RESULT_TIMEOUT_MS = 120_000
RESULTS_PER_GRID = 10

# Streamlit re-renders asynchronously, so "no spinner" can be true before a rerun even starts.
# Waiting for the expected number of visible, decoded photos is the only reliable signal.
_VISIBLE_PHOTOS_LOADED = """minimum => {
    const photos = [...document.querySelectorAll('[data-testid="stImage"] img')]
        .filter(img => img.offsetParent !== null);
    return photos.length >= minimum && photos.every(img => img.complete && img.naturalWidth > 0);
}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ui-url", default="http://127.0.0.1:8501")
    parser.add_argument("--out", type=Path, default=Path("docs/screenshots"))
    parser.add_argument("--query-image", type=Path, required=True, help="Photo for image search.")
    parser.add_argument("--text-query", default="red floral summer dress")
    parser.add_argument(
        "--chat-message", default="Comfortable running shoes under $40 with good ratings, please"
    )
    return parser.parse_args()


def wait_for_photos(page: Page, minimum: int) -> None:
    page.wait_for_function(_VISIBLE_PHOTOS_LOADED, arg=minimum, timeout=RESULT_TIMEOUT_MS)


def capture(page: Page, out: Path, name: str) -> None:
    # JPEG: photo grids compress ~5x better than PNG, which keeps re-captures cheap in git.
    path = out / f"{name}.jpg"
    page.screenshot(path=path, quality=85)
    print(f"saved {path}")


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome")
        page = browser.new_page(viewport=VIEWPORT)
        page.goto(args.ui_url)

        wait_for_photos(page, RESULTS_PER_GRID)
        capture(page, args.out, "recommendations-content")

        page.get_by_text("Collaborative (similar shoppers)").click()
        page.get_by_text("Most similar shoppers").wait_for(timeout=RESULT_TIMEOUT_MS)
        wait_for_photos(page, RESULTS_PER_GRID)
        capture(page, args.out, "recommendations-collaborative")

        page.get_by_role("tab", name="Text → image search", exact=True).click()
        page.get_by_label("Describe what you are looking for").fill(args.text_query)
        page.keyboard.press("Enter")
        wait_for_photos(page, RESULTS_PER_GRID)
        capture(page, args.out, "text-search")

        page.get_by_role("tab", name="Image search", exact=True).click()
        page.locator('input[type="file"]').set_input_files(args.query_image)
        wait_for_photos(page, RESULTS_PER_GRID + 1)  # results plus the uploaded photo
        capture(page, args.out, "image-search")

        page.get_by_role("tab", name="Shopping assistant", exact=True).click()
        page.get_by_placeholder("Ask Clothy", exact=False).fill(args.chat_message)
        page.keyboard.press("Enter")
        reply = page.get_by_test_id("stChatMessage").nth(1).get_by_test_id("stMarkdownContainer")
        reply.first.wait_for(timeout=RESULT_TIMEOUT_MS)
        wait_for_photos(page, 1)
        capture(page, args.out, "shopping-assistant")

        browser.close()


if __name__ == "__main__":
    main()
