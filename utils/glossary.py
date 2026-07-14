from __future__ import annotations

import html
import re


GLOSSARY_PAGE_PATH = "./terms_glossary"


def slugify_term(term: str) -> str:
    slug = term.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    return slug.strip("-")


def glossary_href(term: str) -> str:
    slug = slugify_term(term)
    return f"{GLOSSARY_PAGE_PATH}?term={slug}#{slug}"


def glossary_link(label: str, term: str | None = None) -> str:
    target = term or label
    href = glossary_href(target)
    return f'<a href="{html.escape(href, quote=True)}">{html.escape(label)}</a>'
