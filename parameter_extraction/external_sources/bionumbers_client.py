"""
Helpers for querying the BioNumbers web catalog when a parameter lookup
misses the local merged database. This module focuses on three tasks:

1. Searching BioNumbers with a free-text query.
2. Fetching and parsing a specific BNID page.
3. Normalizing the scraped content to the schema used in merged_parameters.json.

Networking is performed with `requests` and HTML parsing with BeautifulSoup.
All file paths are constrained to the parameter_extraction workspace.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import logging

import requests
from bs4 import BeautifulSoup


BIO_NUMBERS_BASE = "https://bionumbers.hms.harvard.edu"
SEARCH_URL = f"{BIO_NUMBERS_BASE}/search.aspx"
ENTRY_URL = f"{BIO_NUMBERS_BASE}/bionumber.aspx"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)

logger = logging.getLogger(__name__)


@dataclass
class BioNumberEntry:
    bnid: str
    name: str
    value: str
    units: Optional[str]
    context: Optional[str]
    organism: Optional[str]
    citation: Optional[str]

    def to_parameter_doc(self) -> Dict[str, Any]:
        """Return a dictionary compatible with merged_parameters.json schema."""
        source_name = f"bionumbers_{self.bnid}"
        instance = {
            "value": self.value,
            "units": self.units,
            "context": build_context(self),
            "source": source_name,
            "section": "external",
        }
        return {"parameter": self.name, "instances": [instance]}


def build_context(entry: BioNumberEntry) -> str:
    """Compose a descriptive context string for downstream storage."""
    bits = []
    if entry.context:
        bits.append(entry.context.strip())
    if entry.organism:
        bits.append(f"organism: {entry.organism.strip()}")
    bits.append(f"BNID {entry.bnid}")
    if entry.citation:
        bits.append(f"ref: {entry.citation.strip()}")
    return "; ".join(bits)


class BioNumbersCache:
    """Simple JSON-based cache for BNID responses."""

    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        if not cache_path.exists():
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text("{}", encoding="utf-8")
        self._data = json.loads(self.cache_path.read_text(encoding="utf-8"))

    def get(self, bnid: str) -> Optional[BioNumberEntry]:
        data = self._data.get(bnid)
        return BioNumberEntry(**data) if data else None

    def set(self, entry: BioNumberEntry) -> None:
        self._data[entry.bnid] = asdict(entry)
        self.cache_path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")


class BioNumbersClient:
    """HTTP facade for BioNumbers search + detail pages."""

    def __init__(self, session: Optional[requests.Session] = None, cache_path: Optional[Path] = None):
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        cache_path = cache_path or Path("parameter_extraction/external_sources/bionumbers_cache.json")
        self.cache = BioNumbersCache(cache_path)

    def search(self, query: str, max_results: int = 5) -> List[str]:
        """
        Return a list of BNIDs that match the query.
        The new BioNumbers website uses 'trm' parameter and links to bionumber.aspx?id=X
        """
        if not query.strip():
            return []
        params = {"trm": query}
        logger.debug("Searching BioNumbers for %s", query)
        resp = self.session.get(SEARCH_URL, params=params, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Find all links to bionumber.aspx pages
        bnumber_links = soup.find_all("a", href=lambda x: x and "bionumber.aspx" in x.lower())
        if not bnumber_links:
            return []

        bnids: List[str] = []
        for link in bnumber_links:
            href = link.get("href", "")
            # Extract ID from URL like /bionumber.aspx?id=106703&ver=6...
            if "id=" in href:
                try:
                    bnid = href.split("id=")[1].split("&")[0]
                    if bnid and bnid not in bnids:
                        bnids.append(bnid)
                        if len(bnids) >= max_results:
                            break
                except (IndexError, ValueError):
                    continue
        return bnids

    def fetch_entry(self, bnid: str) -> Optional[BioNumberEntry]:
        """Scrape the BioNumbers detail page for a given BNID (now uses 'id' parameter)."""
        bnid = bnid.replace("BNID", "").strip()
        if not bnid:
            return None
        cached = self.cache.get(bnid)
        if cached:
            return cached
        # New website uses 'id' parameter instead of 'v'
        params = {"id": bnid}
        logger.debug("Fetching BNID %s", bnid)
        resp = self.session.get(ENTRY_URL, params=params, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        data = parse_entry_page(soup)
        if not data:
            return None
        entry = BioNumberEntry(bnid=bnid, **data)
        self.cache.set(entry)
        return entry


def parse_entry_page(soup: BeautifulSoup) -> Optional[Dict[str, Any]]:
    """Extract the structured fields from a BNID page soup (updated for new website structure)."""
    # Get parameter name from H1 tag
    h1 = soup.find("h1")
    name = h1.get_text(strip=True) if h1 else None

    # Get other fields from table rows where labels are in <th> and values in <td>
    def grab(label: str) -> Optional[str]:
        # Find table rows with <th> (label) and <td> (value)
        for row in soup.find_all("tr"):
            th = row.find("th")
            td = row.find("td")
            if th and td:
                label_text = th.get_text(strip=True)
                if label.lower() in label_text.lower():
                    return td.get_text(strip=True)
        return None

    value = grab("Value")
    units = None  # Units are often included with value in new format
    organism = grab("Organism")
    context = grab("Comments")
    citation = grab("Reference")

    if not (name and value):
        return None

    # Try to separate units from value if they're together
    # The value often has format like "2300\r\n µm^2/sec"
    if value and ("\n" in value or "\r" in value):
        # Split on any whitespace/newline
        import re
        parts = re.split(r'[\r\n]+', value, 1)
        value = parts[0].strip()
        units = parts[1].strip() if len(parts) > 1 and parts[1].strip() else None

    return {
        "name": name,
        "value": value,
        "units": units,
        "context": context,
        "organism": organism,
        "citation": citation,
    }
