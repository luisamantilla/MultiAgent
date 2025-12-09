from io import BytesIO
import requests
from lxml import etree


def get_pmids(query, MAX_PMIDS):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": MAX_PMIDS
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    return r.json()["esearchresult"]["idlist"]


def get_pmid_to_pmcid(pmid):
    url = "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles"
    params = {
        "email": "kl441@duke.edu",
        "tool": "multiagent",
        "ids": str(pmid)
    }
    headers = {
        "User-Agent": "multiagent-pipeline/0.1 (kl441@duke.edu)"
    }

    response = requests.get(url, params=params, headers=headers)

    if response.status_code != 200:
        print(f"[ERROR] HTTP {response.status_code}: {response.text}")
        return None

    try:
        root = etree.fromstring(response.content)
    except etree.XMLSyntaxError as e:
        print(f"[ERROR] XML parsing failed for PMID {pmid}: {e}")
        print(response.text[:200])
        return None

    record = root.find(".//record")
    if record is not None and "pmcid" in record.attrib:
        return record.attrib["pmcid"]

    return None


def fetch_pmc_xml(pmcid):
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pmc",
        "id": pmcid,
        "retmode": "xml"
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    
    tree = etree.parse(BytesIO(r.content))
    
    doi = None
    for article_id in tree.xpath("//article-id"):
        if article_id.get("pub-id-type") == "doi":
            doi = article_id.text
            break
    return tree, doi