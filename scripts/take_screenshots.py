"""Capture README screenshots of the MVP pages from a running app.

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
    """Pick a shopping history, opening the feed's profile popover when present."""
    popover = page.get_by_role("button", name="Shopping history", exact=True)
    if popover.count():
        popover.click()
    controls = page.get_by_test_id("stPopoverBody") if popover.count() else page
    controls.locator('[data-testid="stSelectbox"]:visible').first.click()
    page.get_by_role("option").nth(index).click()
    wait_for_rerun(page)
    if popover.count():
        page.keyboard.press("Escape")


def wait_for_photos(page: Page, minimum: int) -> None:
    page.wait_for_function(_VISIBLE_PHOTOS_LOADED, arg=minimum, timeout=RESULT_TIMEOUT_MS)


def assert_card_alignment(page: Page) -> None:
    """Cards in one row must share a bottom edge and action baseline."""
    cards = page.locator('[class*="st-key-product_card_"]').evaluate_all(
        """elements => elements.map(element => {
            const card = element.getBoundingClientRect();
            const action = element.querySelector('[data-testid="stButton"] button')
                .getBoundingClientRect();
            return {top: card.top, bottom: card.bottom, actionBottom: action.bottom};
        })"""
    )
    previous = None
    for card in cards:
        if previous is not None and abs(card["top"] - previous["top"]) < 1:
            assert abs(card["bottom"] - previous["bottom"]) < 1, "Uneven product card heights"
            assert abs(card["actionBottom"] - previous["actionBottom"]) < 1, (
                "Uneven product actions"
            )
        previous = card


def assert_chat_layout(page: Page) -> None:
    """The open panel and its input must fit on screen, even with a long transcript."""
    panel = page.get_by_test_id("stPopoverBody").bounding_box()
    field = page.get_by_test_id("stChatInputTextArea").bounding_box()
    viewport = page.viewport_size
    assert panel is not None and field is not None and viewport is not None
    assert panel["x"] >= 0 and panel["x"] + panel["width"] <= viewport["width"] + 1
    assert panel["y"] >= 0 and panel["y"] + panel["height"] <= viewport["height"] + 1
    assert panel["y"] <= field["y"]
    assert field["y"] + field["height"] <= panel["y"] + panel["height"] - 16, (
        "Chat input is clipped by its panel"
    )
    assert page.get_by_test_id("stChatInputSubmitButton").is_visible()


def capture(page: Page, out: Path, name: str) -> None:
    wait_for_rerun(page)
    assert_card_alignment(page)
    # JPEG: photo grids come out about 3x smaller than PNG, which keeps re-captures cheap in git.
    path = out / f"{name}.jpg"
    page.screenshot(path=path, quality=85)
    print(f"saved {path}")


def open_page(page: Page, name: str) -> None:
    page.get_by_role("link", name=name, exact=True).click()
    wait_for_rerun(page)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            channel=args.channel or None, executable_path=args.executable_path
        )
        page = browser.new_page(viewport=VIEWPORT)
        page.goto(args.ui_url)

        open_page(page, "Fashion")
        wait_for_photos(page, PHOTOS_IN_VIEW)
        if args.example:
            choose_example(page, args.example)
            wait_for_photos(page, PHOTOS_IN_VIEW)
        capture(page, args.out, "mvp-for-you")

        for strategy, name in [
            ("Similar shoppers", "recommendations-collaborative"),
            ("Purchase connections", "recommendations-graph"),
            ("Visual style", "recommendations-content"),
        ]:
            page.get_by_text("How picks are chosen", exact=True).click()
            page.get_by_text(strategy, exact=True).click()
            wait_for_rerun(page)
            wait_for_photos(page, PHOTOS_IN_VIEW)
            page.get_by_text("How picks are chosen", exact=True).click()
            capture(page, args.out, name)

        open_page(page, "Product connections")
        if args.example:
            choose_example(page, args.example)
        page.get_by_text("Looks similar").click()
        wait_for_rerun(page)
        wait_for_photos(page, 3)
        capture(page, args.out, "knowledge-graph-similar")

        open_page(page, "Discover")
        page.get_by_label("Describe what you are looking for").fill(args.text_query)
        page.keyboard.press("Enter")
        wait_for_rerun(page)
        wait_for_photos(page, PHOTOS_IN_VIEW)
        capture(page, args.out, "text-search")

        page.get_by_text("Photo", exact=True).click()
        wait_for_rerun(page)
        if args.query_image:
            page.locator('input[type="file"]').set_input_files(args.query_image)
        else:
            page.get_by_text("pick an example photo", exact=False).click(timeout=RESULT_TIMEOUT_MS)
            page.locator('[class*="st-key-query_photo_"] button').first.click()
        page.get_by_text("Query photo").wait_for(timeout=RESULT_TIMEOUT_MS)
        wait_for_rerun(page)
        wait_for_photos(page, PHOTOS_IN_VIEW + 1)  # results plus the query photo
        capture(page, args.out, "image-search")

        page.goto(args.ui_url.rstrip("/") + "/quality")
        page.get_by_text("MAP@12 measures ranking quality", exact=False).wait_for(
            timeout=RESULT_TIMEOUT_MS
        )
        page.wait_for_timeout(1500)  # the bar chart renders after the text
        capture(page, args.out, "evaluation")

        if not args.skip_chat:
            page.get_by_role("button", name="Ask Curio", exact=False).click()
            page.get_by_placeholder("Describe a piece or ask a follow-up…").fill(args.chat_message)
            page.keyboard.press("Enter")
            page.locator('[class*="st-key-chat_product_popup_"]').first.wait_for(
                timeout=RESULT_TIMEOUT_MS
            )
            wait_for_photos(page, 1)
            assert_chat_layout(page)
            capture(page, args.out, "shopping-assistant")
            for viewport in [{"width": 390, "height": 844}, {"width": 320, "height": 568}]:
                page.set_viewport_size(viewport)
                page.wait_for_timeout(250)
                assert_chat_layout(page)
            capture(page, args.out, "shopping-assistant-mobile")

        browser.close()


if __name__ == "__main__":
    main()
