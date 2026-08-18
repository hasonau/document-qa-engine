"""One-off: drive the local Document Assistant and save a LinkedIn screenshot."""

from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "month1_rag_engine" / "data" / "building" / "Muhammad-pages.pdf"
OUT = Path(__file__).resolve().parent / "taif-and-france.png"

Q_IN_DOC = "What happened with Holy prophet when he went to Tai'if?"
Q_NOT_IN_DOC = "What is the capital of France?"


def last_assistant_text(page) -> str:
    bubbles = page.locator(".bubble.assistant p")
    n = bubbles.count()
    if n == 0:
        return ""
    return bubbles.nth(n - 1).inner_text().strip()


def ask_and_wait(page, question: str) -> str:
    before = page.locator(".bubble.assistant p").count()
    page.get_by_label("Message").fill(question)
    page.get_by_role("button", name="Send").click()
    page.wait_for_function(
        """(before) => {
          const bubbles = [...document.querySelectorAll('.bubble.assistant p')];
          if (bubbles.length <= before) return false;
          const t = bubbles.at(-1).innerText.trim();
          const btn = document.querySelector('.composer button[type="submit"]');
          const idle = btn && btn.textContent.trim() === 'Send';
          return idle && t && t !== 'Thinking…';
        }""",
        arg=before,
        timeout=180_000,
    )
    return last_assistant_text(page)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not PDF.is_file():
        raise SystemExit(f"PDF not found: {PDF}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 780, "height": 1100})
        page.goto("http://localhost:5173/", wait_until="networkidle")

        page.set_input_files('input[type="file"]', str(PDF))
        page.get_by_role("button", name="Upload").click()
        page.wait_for_selector(".upload-status:has-text('Uploaded:')", timeout=180_000)

        taif = ask_and_wait(page, Q_IN_DOC)
        print("TAIF_ANSWER:", taif[:400])
        bad = (
            not taif
            or taif.lower() == "not in documents"
            or "could not reach" in taif.lower()
            or "try again" in taif.lower()
        )
        if bad:
            raise SystemExit("Ta'if question did not return a grounded answer")

        france = ask_and_wait(page, Q_NOT_IN_DOC)
        print("FRANCE_ANSWER:", france)
        if france.lower() != "not in documents":
            raise SystemExit(f"France question did not refuse, got: {france!r}")

        # Hide file picker / local path / filename before capture
        page.evaluate(
            """() => {
              const upload = document.querySelector('.upload');
              if (upload) upload.style.display = 'none';
              const app = document.querySelector('.app');
              const messages = document.querySelector('.messages');
              if (messages) {
                messages.style.overflow = 'visible';
                messages.style.flex = 'none';
                messages.style.height = 'auto';
              }
              if (app) {
                app.style.height = 'auto';
                app.style.maxWidth = '720px';
              }
            }"""
        )
        page.locator(".app").screenshot(path=str(OUT))
        print(f"SAVED: {OUT}")
        browser.close()


if __name__ == "__main__":
    main()
