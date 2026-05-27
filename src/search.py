"""
search.py - optional if you want to search by partial title in OpenAlex.
"""

import requests

def search_papers_by_title(query: str, max_results: int = 10) -> list:
    from config import OPENALEX_BASE_URL
    params = {"search": query, "per-page": max_results}
    r = requests.get(OPENALEX_BASE_URL, params=params)
    r.raise_for_status()
    data = r.json()
    return data.get("results", [])

def prompt_user_to_choose_paper(papers: list) -> dict:
    if not papers:
        print("No papers found.")
        return None
    print("\n=== Search Results ===")
    for i, p in enumerate(papers):
        title = p.get("display_name", "Unknown Title")
        pub_year = p.get("publication_year", "N/A")
        auths = p.get("authorships", [])
        first_author = auths[0]["author"]["display_name"] if auths else "N/A"
        print(f"{i}. {title} (Author: {first_author}, Year: {pub_year})")

    choice = input("\nEnter index: ").strip()
    if not choice.isdigit():
        return None
    idx = int(choice)
    if idx < 0 or idx >= len(papers):
        return None
    return papers[idx]
