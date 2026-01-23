# ingestion/cleaners.py
import re
import pycountry

def remove_junk_sections(text, section_markers=None):
    if section_markers is None:
        section_markers = [
            "references",
            "acknowledgment", "acknowledgement", "acknowledgments", "acknowledgements",
            "author information", "author contribution", "author contributions",
            "associated content",
        ]
    pattern = re.compile(
        rf"^\s*[\.\-\u25A0\u2022]*\s*({'|'.join(section_markers)})",
        re.IGNORECASE | re.MULTILINE
    )
    match = pattern.search(text)
    return text[:match.start()] if match else text


def remove_junk_lines(text, junk_patterns=None):
    if junk_patterns is None:
        junk_patterns = [
            "doi:", "et al.", "https://", "http://", ".org", ".com", "conflict of interest",
            "bio:", "funding", "journal", "citation", "cc-by", "preprint", "arxiv",
            "license", "open access", "submitted to", "peer review", "double-blind",
            "published", "copyright", "all rights reserved",
            "correspondence should be addressed to",
            "authors contributed equally",
            "this manuscript has been authored by",
            "contract no", "supporting information",
            "read online", "received:", "revised:", "accepted:", "cite this:", "©",
            "download", "public access plan",
            "department of", "university",
            "national laboratory", "laboratory", "government",
        ]

    countries = [country.name.lower() for country in pycountry.countries]

    lines = text.splitlines()
    cleaned = []
    for line in lines:
        lower = line.lower()
        if any(pat in lower for pat in junk_patterns):
            continue
        if "*" in line or "†" in line or "∇" in line:
            continue
        if any(re.search(rf"\b{re.escape(c)}\b", lower) for c in countries):
            continue
        cleaned.append(line)

    return "\n".join(cleaned)
