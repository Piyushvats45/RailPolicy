import requests
from bs4 import BeautifulSoup
import json
import time

HEADERS = {
    "User-Agent": "Mozilla/5.0 (personal research project; contact: your_email@example.com)"
}


def scrape_page(url: str) -> list[dict]:
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    faqs = []
    # EXAMPLE selectors -- replace with the real target site's class names.
    for item in soup.select(".faq-item"):
        question_el = item.select_one(".faq-question")
        answer_el = item.select_one(".faq-answer")
        if question_el and answer_el:
            faqs.append({
                "question": question_el.get_text(strip=True),
                "answer": answer_el.get_text(strip=True),
                "source_url": url,
            })
    return faqs


def scrape_multiple(urls: list[str], output_path: str):
    all_faqs = []
    for url in urls:
        print(f"Scraping {url} ...")
        try:
            all_faqs.extend(scrape_page(url))
        except Exception as e:
            print(f"  Failed: {e}")
        time.sleep(1)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_faqs, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(all_faqs)} FAQ entries to {output_path}")


if __name__ == "__main__":
    target_urls = ["https://example.com/faq"]
    scrape_multiple(target_urls, "faqs_raw.json")
