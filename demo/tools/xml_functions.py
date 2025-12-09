from io import StringIO
import re
from typing import Dict, List, Optional, Set
import pandas as pd
from lxml import etree


def extract_paragraphs(tree, paragraph):
    for sec in tree.getroot().xpath(".//sec"):
        title_elem = sec.find("title")
        if title_elem is not None and title_elem.text and paragraph in title_elem.text.lower():
            return [
                " ".join(p.itertext()).strip()
                for p in sec.xpath(".//p")
            ]
    return []

def extract_all_body_paragraphs(tree):
    paras = []
    for p in tree.getroot().xpath(".//*[local-name()='body']//*[local-name()='p']"):
        text = " ".join(p.itertext()).strip()
        if text:
            paras.append(text)
    return paras


def extract_full_text(tree):
    abstract_elem = tree.find(".//abstract")
    abstract = " ".join(abstract_elem.itertext()).strip().lower() if abstract_elem is not None else None
    conclusion_paragraphs = extract_paragraphs(tree, "conclusions")
    result_paragraphs = extract_paragraphs(tree, "results")
    
    return abstract, result_paragraphs, conclusion_paragraphs

def extract_tables(tree):
    tables = []
    captions = []
    for wrap in tree.findall(".//table-wrap"):
        # caption
        cap_elem = wrap.find("./caption")
        
        if cap_elem is None:
            continue
        
        caption = " ".join(cap_elem.itertext()).strip().lower() if cap_elem is not None else ""
        
        captions.append(caption)
        # table → HTML → DataFrame → CSV
        tbl_elem = wrap.find(".//table")
        if tbl_elem is None:
            captions.pop()
            continue
        html_str = etree.tostring(tbl_elem, encoding="unicode", method="html")
        df = pd.read_html(StringIO(html_str))[0]                   # first (or only) table
        csv_str = df.to_csv(index=False, sep=" ").lower() # space-separated
        tables.append((caption, csv_str))
    return captions, tables

def extract_table_captions(tree):
    tables = []
    for wrap in tree.findall(".//table-wrap"):
        # caption
        cap_elem = wrap.find("./caption")
        caption = " ".join(cap_elem.itertext()).strip().lower() if cap_elem is not None else ""
    return tables

def extract_text_from_pmc(tree):
    root = tree.getroot()
    keep = {"p", "sec", "title", "table-wrap", "fig", "caption", "abstract"}

    pieces = []
    for elem in root.iter():
        if elem.tag in keep:
            # all descendant text, including inline tags
            txt = " ".join(t.strip() for t in elem.itertext() if t.strip())
            if txt:
                pieces.append(txt)

    return " ".join(pieces)

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _text(elem) -> str:
    """Full text under an element (joins nested tags) without splitting sentences."""
    return _norm(elem.xpath("string()"))

# -------------------- section canonicalization --------------------

def _canon_from_title(title: str) -> Optional[str]:
    """Map a free-text section title into a canonical key."""
    t = (title or "").lower()
    rules = [
        (r"\b(results?\s+and\s+discussion)\b", "results_discussion"),
        (r"\b(materials?\s+and\s+methods?)\b", "methods"),
        (r"\b(patients?\s+and\s+methods?)\b", "methods"),
        (r"\b(methods?|methodology|experimental|experimental\s+procedures)\b", "methods"),
        (r"\b(results?)\b", "results"),
        (r"\b(discussion|discussion\s+and\s+conclusions?)\b", "discussion"),
        (r"\b(conclusions?|concluding\s+remarks)\b", "conclusion"),
        (r"\b(introduction|background)\b", "introduction"),
        (r"\b(abstract)\b", "abstract"),
    ]
    for pat, key in rules:
        if re.search(pat, t):
            return key
    return None

def _canon_from_sectype(sec) -> Optional[str]:
    """Prefer JATS sec-type when present; fall back to title heuristics."""
    st = (sec.get("sec-type") or "").lower()
    if not st:
        return None

    # common JATS values grouped by semantic bucket
    mapping = {
        "materials|materials and methods|methods|methodology|experimental": "methods",
        "results|observations": "results",
        "discussion|results and discussion": "discussion",
        "conclusion|conclusions": "conclusion",
        "introduction|background": "introduction",
        "abstract": "abstract",
    }
    for pat, key in mapping.items():
        if re.fullmatch(pat.replace(" ", r"\s+"), st):
            return key
    return _canon_from_title(st)

# -------------------- paragraph harvesting --------------------

def _paras_in(node) -> List[str]:
    """
    Return CLEAN paragraphs under `node`, excluding captions, tables, figures,
    references, footnotes, author notes, supplementary materials.
    """
    xpath = (
        ".//*[local-name()='p' and "
        "not(ancestor::*[local-name()='caption' or "
        "               local-name()='table-wrap' or "
        "               local-name()='fig' or "
        "               local-name()='ref-list' or "
        "               local-name()='fn' or "
        "               local-name()='author-notes' or "
        "               local-name()='supplementary-material'])]"
    )
    out: List[str] = []
    for p in node.xpath(xpath):
        t = _text(p)
        if t:
            out.append(t)
    return out

def _merge_tiny_paragraphs(paras: List[str], min_chars: int = 200) -> List[str]:
    """
    Optional: some journals put one sentence per <p>. This merges very short
    paragraphs into their predecessor until min_chars is reached.
    """
    if not paras:
        return paras
    merged: List[str] = []
    buf = ""
    for p in paras:
        candidate = (buf + " " + p).strip() if buf else p
        if len(candidate) < min_chars:
            buf = candidate
        else:
            if buf:
                merged.append(candidate)
                buf = ""
            else:
                merged.append(p)
    if buf:
        merged.append(buf)
    return merged

# -------------------- main extractor --------------------

def extract_paragraphs_by_sections(
    tree: etree._ElementTree,
    wanted: Optional[Set[str]] = None,
    *,
    merge_tiny: bool = False,
    min_chars: int = 200,
) -> Dict[str, List[str]]:
    """
    Returns {section_key: [paragraphs...]} with keys like:
      'abstract', 'methods', 'results', 'discussion', 'conclusion',
      'introduction', 'results_discussion'

    Parameters
    ----------
    tree : lxml.etree._ElementTree
        Parsed JATS/PMC XML
    wanted : optional set of section keys to include
    merge_tiny : if True, merges sentence-sized <p> shards
    min_chars : minimum chars for merged paragraphs (only if merge_tiny=True)
    """
    root = tree.getroot()
    out: Dict[str, List[str]] = {}

    def add(sec_key: str, paras: List[str]):
        if not paras:
            return
        if wanted and sec_key not in wanted:
            return
        if merge_tiny:
            paras = _merge_tiny_paragraphs(paras, min_chars=min_chars)
        # Keep only non-empty strings
        cleaned = [p for p in paras if p]
        if cleaned:
            out.setdefault(sec_key, []).extend(cleaned)

    # 1) Abstracts (handles simple and structured abstracts)
    for abs_el in root.xpath(".//*[local-name()='abstract']"):
        # structured abstracts may have nested <sec><title>...</title><p>...</p></sec>
        sec_nodes = abs_el.xpath(".//*[local-name()='sec']")
        if sec_nodes:
            for s in sec_nodes:
                title_nodes = s.xpath("./*[local-name()='title']")
                skey = _canon_from_title(_text(title_nodes[0]) if title_nodes else "abstract") or "abstract"
                paras = _paras_in(s)
                add(skey, paras)
        else:
            paras = _paras_in(abs_el)
            add("abstract", paras)

    # 2) Body sections by <sec>, using sec-type or title
    for sec in root.xpath(".//*[local-name()='body']//*[local-name()='sec']"):
        # prefer sec-type when present; else use title text
        skey = _canon_from_sectype(sec)
        if not skey:
            titles = sec.xpath("./*[local-name()='title']")
            skey = _canon_from_title(_text(titles[0]) if titles else "") or None
        if not skey:
            continue
        paras = _paras_in(sec)
        add(skey, paras)

    # 3) Some articles have conclusions outside <body> (rare but seen)
    for sec in root.xpath(
        ".//*[local-name()='sec' and "
        "(contains(translate(./*[local-name()='title']/text(),'CONCLUSIONS','conclusions'),'conclusion'))]"
    ):
        paras = _paras_in(sec)
        add("conclusion", paras)

    return out