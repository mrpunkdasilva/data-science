"""Ingestion of the TST Livro de Jurisprudência PDF.

Parses Súmulas, Orientação Jurisprudencial (OJ) and Precedentes Normativos (PN)
verbete by verbete, keeping only the current redaction for each verbete (the
book also reproduces the full historical sequence of each verbete).
"""

# mypy: ignore-errors

import re
from collections import Counter
from pathlib import Path

import pandas as pd
import pdfplumber
from config import settings

HEADING_SUMULA = re.compile(
    r"^(?:S[úu]mula[\sº.]+|[Nn][º°o.:]?)\s*(\d{1,4})(?:\s+(.*))?$", re.IGNORECASE
)
HEADING_OJ = re.compile(r"^OJ-([a-z0-9/]+)-(\d{1,4})[\s.\-]*\s*(.*)$", re.IGNORECASE)
HEADING_PN = re.compile(
    r"^(?:Precedente\s+Normativo|PN)[\s-]*(?:[nºo.]?\s*)?(\d{1,4})[\s-]*(.+)$",
    re.IGNORECASE,
)

BANNER_SUMULAS = re.compile(r"^S[ÚU]MULAS\b")
BANNER_OJ = re.compile(r"^ORIENTA[ÇC][ÕO]ES\s+JURISPRUDENCIAIS")
BANNER_PN = re.compile(r"^PRECEDENTES\s+NORMATIVOS")
BANNER_INDEX = re.compile(r"^[IÍ]NDICE\s+REMISSIVO\s*$", re.IGNORECASE)

FURNITURE = re.compile(
    r"^(?:S[úu]mulas?|Orienta[çc][ãõ]es?\s+Jurisprudenciais|"
    r"Orienta[çc][ãõ]o\s+Jurisprudencial(?: da .*)?|"
    r"Precedentes?\s+Normativos?|[A-Z]-?\d+|\d+|"
    r"A-?\d+|C-?\d+|G-?\d+|"
    r"OVISSIMER|ECIDN[ÍI]|SETNEDECERP\s+SOVITAMRON|SALUM[ÚU]S.*)$",
    re.IGNORECASE,
)

HISTORY_BOUNDARY = re.compile(
    r"^(Hist[óo]rico:|Reda[çc][ãõ]o\s+(original|anterior|nova)|"
    r"S[úu]mula\s+(alterada|mantida|cancelada|convertida|revisada|superada)|"
    r"(nova|nova\s+reda[çc][ãõ]o) em decorr)",
    re.IGNORECASE,
)

ORG_MAP = {
    "SDI1": "SBDI-1",
    "SBDI1": "SBDI-1",
    "SBDI-I": "SBDI-1",
    "SDI2": "SBDI-2",
    "SBDI2": "SBDI-2",
    "SBDI-II": "SBDI-2",
    "SDC": "SDC",
    "TP": "TP/OE",
    "OE": "TP/OE",
    "TP/OE": "TP/OE",
    "SDI1T": "SBDI-1T",
    "SBDI1T": "SBDI-1T",
}


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation/spaces."""
    return re.sub(r"[^a-z0-9çãõáéíóúâêôàü]+", "", text.lower())


def _join_lines(lines: list[str]) -> list[str]:
    """Merge hyphenated line breaks (text is justified) and strip blank lines."""
    out: list[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # de-hyphenate: "pe-" + "soal" -> "pessoal"
        if out and out[-1].endswith("-") and out[-1][:-1].strip():
            out[-1] = out[-1][:-1].rstrip() + (
                line if line[0] in ".,;:()" else line[0].lower() + line[1:]
            )
            continue
        out.append(line)
    return out


def _iter_clean_lines(pdf) -> list[str]:
    """Extract text from all pages, returning cleaned line strings."""
    lines: list[str] = []
    for page in pdf.pages:
        text = page.extract_text() or ""
        lines.extend(text.splitlines())
        if re.search(BANNER_INDEX, text):
            break
    return _join_lines(lines)


def _segment_verbetes(lines: list[str]) -> list[dict]:
    """Split cleaned lines into verbete dicts by heading."""
    verbetes: list[dict] = []
    cur: dict | None = None
    done: bool = False

    def flush() -> None:
        nonlocal cur
        if cur and cur["body"]:
            verbetes.append(cur)
        cur = None

    for line in lines:
        if BANNER_INDEX.match(line):
            break

        if FURNITURE.match(line):
            continue

        m_oj = HEADING_OJ.match(line)
        m_pn = HEADING_PN.match(line)
        m_sum = HEADING_SUMULA.match(line)

        if m_oj:
            orgao_raw = m_oj.group(1).upper()
            num = int(m_oj.group(2))
            tema = m_oj.group(3).strip() or m_oj.group(2)
            if cur and cur["tipo"] == "oj" and cur["codigo"] == num:
                continue  # historical repeat
            flush()
            cur = {
                "tipo": "oj",
                "orgao": ORG_MAP.get(orgao_raw, orgao_raw),
                "codigo": num,
                "tema": tema,
                "body": [],
            }
            done = False
            continue
        if m_pn:
            num = int(m_pn.group(1))
            tema = m_pn.group(2).strip() or m_pn.group(1)
            if cur and cur["tipo"] == "precedente" and cur["codigo"] == num:
                continue
            flush()
            cur = {
                "tipo": "precedente",
                "orgao": "SDC",
                "codigo": num,
                "tema": tema,
                "body": [],
            }
            done = False
            continue
        if m_sum:
            num = int(m_sum.group(1))
            tema = (m_sum.group(2) or "").strip()
            if cur and cur["tipo"] == "sumula" and cur["codigo"] == num:
                continue  # historical repeat of the same verbete
            flush()
            cur = {
                "tipo": "sumula",
                "orgao": "TST",
                "codigo": num,
                "tema": tema,
                "body": [],
            }
            done = False
            continue

        if cur is None:
            continue  # leading content before first verbete

        if HISTORY_BOUNDARY.match(line):
            done = True
            continue
        if not done:
            cur["body"].append(line)

    flush()
    return verbetes


def _label_verbetes(verbetes: list[dict]) -> None:
    """Assign heuristic quality labels adapted to the jurisprudence domain."""
    norms = Counter(_normalize(" ".join(v["body"])) for v in verbetes if v["body"])

    for v in verbetes:
        text = " ".join(v["body"]).strip()
        header = " ".join(x for x in [v["tema"], text]).lower()
        n = len(text)
        dup = norms[_normalize(text)]
        v["duplicate_count"] = dup

        if any(
            kw in header
            for kw in ["cancelad", "(negativo)", "revogad", "cassad", "superad"]
        ):
            v["quality_label"] = "cancelado"
        elif dup > 1:
            v["quality_label"] = "duplicado"
        elif n < settings.VERBETE_RUIDO:
            v["quality_label"] = "ruido"
        elif n < settings.VERBETE_CURTO:
            v["quality_label"] = "curto"
        else:
            v["quality_label"] = "completo"


def load_verbetes(
    pdf_path: str | Path | None = None,
    min_text_len: int | None = None,
) -> pd.DataFrame:
    """
    Load verbetes (Súmulas, OJs and Precedentes Normativos) from the TST PDF.

    Args:
        pdf_path: Path to the Livro de Jurisprudência PDF
        min_text_len: Drop verbetes with body shorter than this

    Returns:
        DataFrame with one verbete per row
    """
    pdf_path = Path(pdf_path or settings.TST_PDF_PATH)
    min_len = min_text_len or settings.VERBETE_MIN_TEXT

    with pdfplumber.open(str(pdf_path)) as pdf:
        lines = _iter_clean_lines(pdf)

    console_log(f"Extracted {len(lines)} cleaned lines from {pdf_path.name}")
    verbetes = _segment_verbetes(lines)
    console_log(f"Segmented {len(verbetes)} verbetes")
    _label_verbetes(verbetes)

    rows = []
    for v in verbetes:
        text = " ".join(v["body"]).strip()
        text = re.sub(r"\s+", " ", text)
        if len(text) < min_len:
            continue
        codigo = v["codigo"]
        org = v["orgao"]
        tema = v["tema"].strip()
        if not tema and text:
            tema = text[:100]
        doc_id = (
            f"{v['tipo']}_{org.lower()}_{codigo}"
            if org not in ("TST", "SDC")
            else f"{v['tipo']}_{codigo}"
        )
        rows.append(
            {
                "doc_id": doc_id,
                "tipo": v["tipo"],
                "orgao": v["orgao"],
                "codigo": codigo,
                "tema": v["tema"].strip(),
                "text": text,
                "text_len": len(text),
                "source": pdf_path.name,
                "quality_label": v.get("quality_label", "completo"),
            }
        )

    df = pd.DataFrame(rows)
    type_counts = df["tipo"].value_counts().to_dict() if not df.empty else {}
    console_log(f"Verbetes loaded: {len(df)} -> {type_counts}")
    return df


def console_log(msg: str) -> None:
    """Simple console logging."""
    print(f"[INGEST] {msg}")
