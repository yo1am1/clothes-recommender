"""Capture README screenshots of every UI tab from a running app.

Usage (API and UI already running):
    uv run python -m scripts.take_screenshots [--query-image path/to/photo.jpg] [--skip-chat]

Drives the locally installed Google Chrome by default, so no Playwright browser download is
needed; pass `--channel ""` to use Playwright's own Chromium instead. Without
--query-image, image search uses the first example photo.
"""

import argparse
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

# Streamlit scrolls an inner container, so full_page can't extend the capture: the viewport
# itself must be tall enough to show a whole result grid.
VIEWPORT = {"width": 1440, "height": 1500}
# Upper bound for API calls behind a click (CLIP inference, LLM round-trips).
RESULT_TIMEOUT_MS = 120_000
# Photos to wait for: two rows of three cards fill the viewport. A few products have no
# photo, so waiting for a whole grid could hang; wait_for_rerun ensures they are fresh.
PHOTOS_IN_VIEW = 6

# Streamlit re-renders asynchronously, so "no spinner" can be true before a rerun even starts.
# The script waits for the expected number of visible, decoded photos instead.
_VISIBLE_PHOTOS_LOADED = """minimum => {
    const photos = [...document.querySelectorAll('[data-testid="stImage"] img')]
        .filter(img => img.offsetParent !== null);
    return photos.length >= minimum && photos.every(img => img.complete && img.naturalWidth > 0);
}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ui-url", default="http://127.0.0.1:8501")
    parser.add_argument("--out", type=Path, default=Path("docs/screenshots"))
    parser.add_argument("--channel", default="chrome", help='Browser channel; "" for Chromium.')
    parser.add_argument("--executable-path", help="Browser binary, overriding --channel.")
    parser.add_argument("--query-image", type=Path, help="Photo for image search.")
    parser.add_argument("--text-query", default="striped summer dress")
    parser.add_argument(
        "--chat-message", default="Black wide-leg trousers for women under €40, please"
    )
    parser.add_argument("--skip-chat", action="store_true", help="No LLM is configured.")
    parser.add_argument(
        "--example", type=int, default=0, help="Curated customer example to show (0-based)."
    )
    return parser.parse_args()


_RERUN_FINISHED = """() =>
    !document.querySelector('[data-stale="true"]') &&
    !document.querySelector('[data-testid="stStatusWidgetRunningIcon"]')"""


def wait_for_rerun(page: Page) -> None:
    """Wait until Streamlit has replaced the previous run's (stale) elements."""
    page.wait_for_timeout(500)  # the rerun starts asynchronously after an interaction
    page.wait_for_function(_RERUN_FINISHED, timeout=RESULT_TIMEOUT_MS)


def choose_example(page: Page, index: int) -> None:
    """Pick a curated customer in the visible tab's first select box."""
    page.locator('[data-testid="stSelectbox"]:visible').first.click()
    page.get_by_role("option").nth(index).click()
    wait_for_rerun(page)


def wait_for_photos(page: Page, minimum: int) -> None:
    page.wait_for_function(_VISIBLE_PHOTOS_LOADED, arg=minimum, timeout=RESULT_TIMEOUT_MS)


def capture(page: Page, out: Path, name: str) -> None:
    # JPEG: photo grids come out about 3x smaller than PNG, which keeps re-captures cheap in git.
    path = out / f"{name}.jpg"
    page.screenshot(path=path, quality=85)
    print(f"saved {path}")


def open_tab(page: Page, name: str) -> None:
    page.get_by_role("tab", name=name, exact=True).click()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            channel=args.channel or None, executable_path=args.executable_path
        )
        page = browser.new_page(viewport=VIEWPORT)
        page.goto(args.ui_url)

        wait_for_photos(page, PHOTOS_IN_VIEW)
        if args.example:
            choose_example(page, args.example)
            wait_for_photos(page, PHOTOS_IN_VIEW)
        capture(page, args.out, "recommendations-graph")

        page.get_by_text("Content-based (CLIP taste vector)").click()
        wait_for_rerun(page)
        wait_for_photos(page, PHOTOS_IN_VIEW)
        capture(page, args.out, "recommendations-content")

        page.get_by_text("Collaborative (similar shoppers)").click()
        wait_for_rerun(page)
        page.get_by_text("Most similar shoppers").wait_for(timeout=RESULT_TIMEOUT_MS)
        wait_for_photos(page, PHOTOS_IN_VIEW)
        capture(page, args.out, "recommendations-collaborative")

        page.get_by_text("Hybrid (collaborative + content-based)").click()
        wait_for_rerun(page)
        wait_for_photos(page, PHOTOS_IN_VIEW)
        capture(page, args.out, "recommendations-hybrid")

        open_tab(page, "Knowledge graph")
        if args.example:
            choose_example(page, args.example)
        page.get_by_text("Looks similar").click()
        wait_for_rerun(page)
        wait_for_photos(page, 3)
        capture(page, args.out, "knowledge-graph-similar")

        open_tab(page, "Text → image search")
        page.get_by_label("Describe what you are looking for").fill(args.text_query)
        page.keyboard.press("Enter")
        wait_for_rerun(page)
        wait_for_photos(page, PHOTOS_IN_VIEW)
        capture(page, args.out, "text-search")

        open_tab(page, "Image search")
        if args.query_image:
            page.locator('input[type="file"]').set_input_files(args.query_image)
        else:
            page.get_by_text("pick an example photo").wait_for(timeout=RESULT_TIMEOUT_MS)
            page.locator('[class*="st-key-query_photo_"] button').first.click()
        page.get_by_text("Query photo").wait_for(timeout=RESULT_TIMEOUT_MS)
        wait_for_rerun(page)
        wait_for_photos(page, PHOTOS_IN_VIEW + 1)  # results plus the query photo
        capture(page, args.out, "image-search")

        open_tab(page, "Evaluation")
        page.get_by_text("MAP@12 is mean average precision").wait_for(timeout=RESULT_TIMEOUT_MS)
        page.wait_for_timeout(1500)  # the bar chart renders after the text
        capture(page, args.out, "evaluation")

        if not args.skip_chat:
            open_tab(page, "Shopping assistant")
            page.get_by_placeholder("Ask Clothy", exact=False).fill(args.chat_message)
            page.keyboard.press("Enter")
            reply = page.get_by_test_id("stChatMessage").nth(1)
            reply.get_by_test_id("stMarkdownContainer").first.wait_for(timeout=RESULT_TIMEOUT_MS)
            wait_for_photos(page, 1)
            capture(page, args.out, "shopping-assistant")

        browser.close()


if __name__ == "__main__":
    main()
