from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
CATALOG_KIND = "sealed-evaluation-internet-candidates"
RIGHTS_TIERS = {
    "open-production",
    "restricted-research",
    "discovery-only",
}
MEDIA_TYPES = {
    "audio",
    "audio-collection",
    "audiovisual",
    "score",
    "symbolic-score",
}
USER_AGENT = (
    "GuitarTabStudio/1.0 sealed-evaluation metadata discovery "
    "(no restricted-media download)"
)
URL_PATTERN = re.compile(r"^https?://", re.IGNORECASE)
TAG_PATTERN = re.compile(r"<[^>]+>")
SPACE_PATTERN = re.compile(r"\s+")
NON_WORD_PATTERN = re.compile(r"[^\w]+", re.UNICODE)
MUTOPIA_BASE_URL = "https://www.mutopiaproject.org"
MUTOPIA_CATALOG_URL = f"{MUTOPIA_BASE_URL}/cgibin/make-table.cgi"
WIKIMEDIA_API_URL = "https://commons.wikimedia.org/w/api.php"
OPENVERSE_API_URL = "https://api.openverse.org/v1/audio/"
INTERNET_ARCHIVE_API_URL = "https://archive.org/advancedsearch.php"


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        value = ", ".join(str(item) for item in value if item is not None)
    rendered = TAG_PATTERN.sub(" ", html.unescape(str(value)))
    return SPACE_PATTERN.sub(" ", rendered).strip()


def _url(value: object) -> str | None:
    rendered = _text(value)
    return rendered if URL_PATTERN.match(rendered) else None


def _float(value: object) -> float | None:
    try:
        rendered = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return rendered if rendered >= 0 else None


def _milliseconds_to_seconds(value: object) -> float | None:
    rendered = _float(value)
    return None if rendered is None else rendered / 1000.0


def _normalized(value: object) -> str:
    rendered = _text(value).casefold()
    rendered = re.sub(r"\.[a-z0-9]{2,5}$", "", rendered)
    return NON_WORD_PATTERN.sub(" ", rendered).strip()


def _work_identity(title: str, creator: str) -> dict[str, str]:
    normalized_title = _normalized(title)
    normalized_creator = _normalized(
        re.sub(r"\([^)]*\)", "", re.sub(r"^by\s+", "", creator, flags=re.I))
    )
    joined = f"{normalized_title}\0{normalized_creator}"
    return {
        "title": title,
        "creator": creator,
        "normalized_title": normalized_title,
        "normalized_creator": normalized_creator,
        "identity_sha256": hashlib.sha256(joined.encode("utf-8")).hexdigest(),
    }


def _license_text(declared_license: object, license_url: object) -> str:
    return f"{_text(declared_license)} {_text(license_url)}".casefold()


def classify_rights(
    declared_license: object,
    license_url: object,
    *,
    provider: str,
) -> dict[str, object]:
    value = _license_text(declared_license, license_url)
    metadata_only = ["metadata-indexing", "rights-review"]
    if provider == "youtube":
        return {
            "tier": (
                "restricted-research"
                if value and "unknown" not in value
                else "discovery-only"
            ),
            "permitted_operations": metadata_only,
            "download_prohibited": True,
            "compliance_obligations": [
                "obtain-separate-permission-before-media-acquisition",
                "preserve-platform-source-url",
            ],
        }
    if not value or any(
        marker in value
        for marker in (
            "unknown",
            "unspecified",
            "all rights reserved",
            "no known license",
        )
    ):
        return {
            "tier": "discovery-only",
            "permitted_operations": metadata_only,
            "download_prohibited": True,
            "compliance_obligations": [
                "obtain-permission-or-keep-metadata-only",
            ],
        }
    if any(
        marker in value
        for marker in (
            "public domain",
            "publicdomain",
            "creativecommons.org/publicdomain",
            "cc0",
            "cc zero",
        )
    ):
        return {
            "tier": "open-production",
            "permitted_operations": [
                "controlled-evaluation",
                "download",
                "metadata-indexing",
                "production-evaluation",
                "redistribution",
                "training",
            ],
            "download_prohibited": False,
            "compliance_obligations": [
                "verify-item-level-public-domain-status",
            ],
        }
    has_by = bool(
        re.search(r"(?:\bcc\s*)?by(?:[- /]|$)", value)
        or "creativecommons.org/licenses/by/" in value
        or "creative commons attribution" in value
    )
    has_share_alike = any(
        marker in value
        for marker in ("by-sa", "sharealike", "share alike", "/sa/")
    )
    has_noncommercial = any(
        marker in value
        for marker in ("by-nc", "noncommercial", "non-commercial", "/nc")
    )
    has_no_derivatives = any(
        marker in value
        for marker in ("by-nd", "no derivatives", "noderivatives", "/nd")
    )
    if has_by and not (has_noncommercial or has_no_derivatives):
        obligations = [
            "attribution",
            "indicate-modifications",
            "license-link",
        ]
        operations = [
            "controlled-evaluation",
            "download",
            "metadata-indexing",
            "production-evaluation",
            "redistribution-with-attribution",
            "training",
        ]
        if has_share_alike:
            obligations.append("share-alike-on-distributed-adaptations")
            operations.append(
                "distributed-adaptations-under-compatible-license"
            )
        return {
            "tier": "open-production",
            "permitted_operations": sorted(operations),
            "download_prohibited": False,
            "compliance_obligations": sorted(obligations),
        }
    obligations = ["rights-review"]
    if has_by:
        obligations.extend(
            ["attribution", "indicate-modifications", "license-link"]
        )
    if has_noncommercial:
        obligations.append("noncommercial-use-only")
    if has_no_derivatives:
        obligations.append("no-distributed-adaptations")
    return {
        "tier": "restricted-research",
        "permitted_operations": [
            "metadata-indexing",
            "noncommercial-research-review",
            "rights-review",
        ],
        "download_prohibited": True,
        "compliance_obligations": sorted(set(obligations)),
    }


def _candidate(
    *,
    provider: str,
    source_id: object,
    source_url: object,
    media_type: str,
    title: object,
    creator: object,
    declared_license: object,
    license_url: object,
    rights_evidence_url: object,
    duration_seconds: object = None,
    asset_url: object = None,
    upstream_provider: object = None,
    score_formats: Iterable[str] = (),
    score_assets: Iterable[dict[str, object]] = (),
    content_evidence: dict[str, object] | None = None,
) -> dict[str, Any]:
    rendered_source_id = _text(source_id)
    rendered_title = _text(title) or rendered_source_id
    rendered_creator = _text(creator) or "unknown"
    rendered_license = _text(declared_license) or "unknown"
    rendered_license_url = _url(license_url)
    rights = classify_rights(
        rendered_license,
        rendered_license_url,
        provider=provider,
    )
    rendered_score_assets = []
    if not rights["download_prohibited"]:
        for asset in score_assets:
            asset_format = _text(asset.get("format"))
            asset_url = _url(asset.get("url"))
            if asset_format and asset_url:
                rendered_score_assets.append(
                    {"format": asset_format, "url": asset_url}
                )
    rendered_score_assets.sort(
        key=lambda value: (value["format"], value["url"])
    )
    candidate = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": f"{provider}:{rendered_source_id}",
        "provider": provider,
        "upstream_provider": _text(upstream_provider) or None,
        "source_id": rendered_source_id,
        "source_url": _url(source_url),
        "media_type": media_type,
        "title": rendered_title,
        "creator": rendered_creator,
        "work_identity": _work_identity(rendered_title, rendered_creator),
        "recording_identity": (
            f"{provider}:{rendered_source_id}"
            if media_type in {"audio", "audiovisual"}
            else None
        ),
        "duration_seconds": _float(duration_seconds),
        "declared_license": rendered_license,
        "license_url": rendered_license_url,
        "rights_evidence_url": _url(rights_evidence_url),
        "rights_tier": rights["tier"],
        "permitted_operations": sorted(rights["permitted_operations"]),
        "compliance_obligations": sorted(
            rights["compliance_obligations"]
        ),
        "download_prohibited": rights["download_prohibited"],
        "asset_url": (
            _url(asset_url) if not rights["download_prohibited"] else None
        ),
        "available_score_formats": sorted(set(score_formats)),
        "score_assets": rendered_score_assets,
        "content_evidence": content_evidence or {},
        "score_pair": {
            "status": (
                "score-source"
                if media_type in {"score", "symbolic-score"}
                else "unpaired"
            ),
            "counterpart_candidate_ids": [],
            "match_basis": (
                "source-is-score"
                if media_type == "score"
                else None
            ),
        },
        "audit_status": "unreviewed",
        "qualified_for_sealed_evaluation": False,
    }
    validate_candidate(candidate)
    return candidate


def validate_candidate(candidate: dict[str, Any]) -> None:
    required_strings = (
        "candidate_id",
        "provider",
        "source_id",
        "source_url",
        "media_type",
        "title",
        "creator",
        "declared_license",
        "rights_evidence_url",
        "rights_tier",
    )
    for field in required_strings:
        if not isinstance(candidate.get(field), str) or not candidate[field]:
            raise ValueError(f"Candidate requires {field}")
    if not URL_PATTERN.match(candidate["source_url"]):
        raise ValueError("Candidate source_url must use HTTP(S)")
    if not URL_PATTERN.match(candidate["rights_evidence_url"]):
        raise ValueError("Candidate rights_evidence_url must use HTTP(S)")
    if candidate["media_type"] not in MEDIA_TYPES:
        raise ValueError(f"Unsupported media type: {candidate['media_type']}")
    if candidate["rights_tier"] not in RIGHTS_TIERS:
        raise ValueError(f"Unsupported rights tier: {candidate['rights_tier']}")
    operations = candidate.get("permitted_operations")
    if (
        not isinstance(operations, list)
        or operations != sorted(set(operations))
        or any(not isinstance(value, str) or not value for value in operations)
    ):
        raise ValueError("Candidate permitted_operations must be sorted strings")
    obligations = candidate.get("compliance_obligations")
    if (
        not isinstance(obligations, list)
        or obligations != sorted(set(obligations))
        or any(not isinstance(value, str) or not value for value in obligations)
    ):
        raise ValueError(
            "Candidate compliance_obligations must be sorted strings"
        )
    if candidate["rights_tier"] != "open-production":
        if candidate.get("download_prohibited") is not True:
            raise ValueError("Restricted candidates must prohibit download")
        if candidate.get("asset_url") is not None:
            raise ValueError("Restricted candidates must not expose asset_url")
        if candidate.get("score_assets"):
            raise ValueError(
                "Restricted candidates must not expose score assets"
            )
    if candidate.get("qualified_for_sealed_evaluation") is not False:
        raise ValueError("Discovery candidates cannot be pre-qualified")
    score_pair = candidate.get("score_pair")
    if not isinstance(score_pair, dict):
        raise ValueError("Candidate requires score_pair metadata")


def _metadata_value(metadata: object, field: str) -> object:
    if not isinstance(metadata, dict):
        return None
    value = metadata.get(field)
    if not isinstance(value, dict):
        return None
    return value.get("value")


def normalize_wikimedia(payload: dict[str, Any]) -> list[dict[str, Any]]:
    query = payload.get("query")
    pages = query.get("pages") if isinstance(query, dict) else None
    if not isinstance(pages, list):
        return []
    records = []
    for page in sorted(pages, key=lambda value: int(value.get("pageid", 0))):
        image_info = page.get("imageinfo")
        if not isinstance(image_info, list) or not image_info:
            continue
        info = image_info[0]
        metadata = info.get("extmetadata")
        mime = _text(info.get("mime")).casefold()
        page_title = _text(page.get("title"))
        is_symbolic = (
            "midi" in mime
            or page_title.casefold().endswith((".mid", ".midi"))
        )
        media_type = (
            "symbolic-score"
            if is_symbolic
            else "audiovisual"
            if _float(info.get("width"))
            else "audio"
        )
        license_url = _metadata_value(metadata, "LicenseUrl")
        source_url = info.get("descriptionurl")
        categories = sorted(
            {
                value.strip()
                for value in _text(
                    _metadata_value(metadata, "Categories")
                ).split("|")
                if value.strip()
            }
        )
        records.append(
            _candidate(
                provider="wikimedia-commons",
                source_id=page.get("pageid"),
                source_url=source_url,
                media_type=media_type,
                title=(
                    _metadata_value(metadata, "ObjectName")
                    or page.get("title")
                ),
                creator=_metadata_value(metadata, "Artist"),
                declared_license=(
                    _metadata_value(metadata, "LicenseShortName")
                    or _metadata_value(metadata, "UsageTerms")
                ),
                license_url=license_url,
                rights_evidence_url=source_url,
                duration_seconds=info.get("duration"),
                asset_url=info.get("url"),
                score_formats=("midi",) if is_symbolic else (),
                score_assets=(
                    [{"format": "midi", "url": info.get("url")}]
                    if is_symbolic
                    else ()
                ),
                content_evidence={
                    "categories": categories,
                    "credit": _text(
                        _metadata_value(metadata, "Credit")
                    ),
                    "description": _text(
                        _metadata_value(metadata, "ImageDescription")
                    ),
                    "mime": mime,
                    "size_bytes": int(info.get("size") or 0),
                    "width": int(info.get("width") or 0),
                    "height": int(info.get("height") or 0),
                },
            )
        )
    return records


def normalize_openverse(payload: dict[str, Any]) -> list[dict[str, Any]]:
    results = payload.get("results")
    if not isinstance(results, list):
        return []
    records = []
    for result in sorted(results, key=lambda value: _text(value.get("id"))):
        source_url = (
            result.get("foreign_landing_url")
            or result.get("detail_url")
        )
        license_url = result.get("license_url")
        records.append(
            _candidate(
                provider="openverse",
                source_id=result.get("id"),
                source_url=source_url,
                media_type="audio",
                title=result.get("title"),
                creator=result.get("creator"),
                declared_license=" ".join(
                    value
                    for value in (
                        _text(result.get("license")),
                        _text(result.get("license_version")),
                    )
                    if value
                ),
                license_url=license_url,
                rights_evidence_url=license_url or source_url,
                duration_seconds=_milliseconds_to_seconds(
                    result.get("duration")
                ),
                asset_url=result.get("url"),
                upstream_provider=result.get("provider") or result.get("source"),
                content_evidence={
                    "audio_set": _text(result.get("audio_set")),
                    "category": _text(result.get("category")),
                    "filetype": _text(result.get("filetype")),
                    "genres": sorted(
                        _text(value)
                        for value in (result.get("genres") or [])
                        if _text(value)
                    ),
                },
            )
        )
    return records


def normalize_internet_archive(
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    response = payload.get("response")
    documents = response.get("docs") if isinstance(response, dict) else None
    if not isinstance(documents, list):
        return []
    records = []
    for document in sorted(
        documents,
        key=lambda value: _text(value.get("identifier")),
    ):
        identifier = _text(document.get("identifier"))
        source_url = f"https://archive.org/details/{urllib.parse.quote(identifier)}"
        license_url = document.get("licenseurl")
        records.append(
            _candidate(
                provider="internet-archive",
                source_id=identifier,
                source_url=source_url,
                media_type="audio-collection",
                title=document.get("title"),
                creator=document.get("creator"),
                declared_license=license_url or "unknown",
                license_url=license_url,
                rights_evidence_url=source_url,
                duration_seconds=document.get("duration"),
                content_evidence={
                    "collection": sorted(
                        _text(value)
                        for value in (
                            document.get("collection")
                            if isinstance(
                                document.get("collection"),
                                list,
                            )
                            else [document.get("collection")]
                        )
                        if _text(value)
                    ),
                    "date": _text(document.get("date")),
                    "description": _text(document.get("description")),
                    "subject": sorted(
                        _text(value)
                        for value in (
                            document.get("subject")
                            if isinstance(document.get("subject"), list)
                            else [document.get("subject")]
                        )
                        if _text(value)
                    ),
                },
            )
        )
    return records


ARCHIVE_AUDIO_FORMAT_PRIORITY = {
    "flac": 0,
    "24bit flac": 0,
    "wave": 1,
    "wav": 1,
    "ogg vorbis": 2,
    "vbr mp3": 3,
    "128kbps mp3": 4,
    "64kbps mp3": 5,
}


def _archive_duration(value: object) -> float | None:
    rendered = _text(value)
    if not rendered:
        return None
    if ":" not in rendered:
        return _float(rendered)
    try:
        parts = [float(part) for part in rendered.split(":")]
    except ValueError:
        return None
    duration = 0.0
    for part in parts:
        duration = duration * 60 + part
    return duration if duration >= 0 else None


def _archive_file_key(file: dict[str, Any]) -> str:
    track = _text(file.get("track"))
    title = _text(file.get("title"))
    name = _text(file.get("name"))
    if track:
        return f"track:{track}"
    original = _text(file.get("original"))
    if original:
        return original.casefold()
    if title:
        return f"title:{title.casefold()}"
    stem = re.sub(
        r"\.(?:flac|ogg|oga|wav|wave|mp3)$",
        "",
        name,
        flags=re.IGNORECASE,
    )
    return f"stem:{stem.casefold()}"


def normalize_internet_archive_files(
    payload: dict[str, Any],
    parent: dict[str, Any],
) -> list[dict[str, Any]]:
    metadata = payload.get("metadata")
    files = payload.get("files")
    if not isinstance(metadata, dict) or not isinstance(files, list):
        return []
    item_id = _text(metadata.get("identifier") or parent.get("source_id"))
    if not item_id:
        return []
    choices: dict[str, tuple[tuple[int, int], dict[str, Any]]] = {}
    for file in files:
        if not isinstance(file, dict):
            continue
        file_format = _text(file.get("format")).casefold()
        if file_format not in ARCHIVE_AUDIO_FORMAT_PRIORITY:
            continue
        duration = _archive_duration(file.get("length"))
        if duration is None or duration < 30:
            continue
        source_rank = 0 if _text(file.get("source")) == "original" else 1
        rank = (
            source_rank,
            ARCHIVE_AUDIO_FORMAT_PRIORITY[file_format],
        )
        key = _archive_file_key(file)
        existing = choices.get(key)
        if existing is None or rank < existing[0]:
            choices[key] = (rank, file)

    records = []
    item_source_url = (
        f"https://archive.org/details/{urllib.parse.quote(item_id)}"
    )
    subjects = metadata.get("subject")
    subject_values = (
        subjects if isinstance(subjects, list) else [subjects]
    )
    collections = metadata.get("collection")
    collection_values = (
        collections if isinstance(collections, list) else [collections]
    )
    for _key, (_rank, file) in sorted(choices.items()):
        name = _text(file.get("name"))
        if not name:
            continue
        asset_url = (
            "https://archive.org/download/"
            f"{urllib.parse.quote(item_id)}/"
            f"{urllib.parse.quote(name)}"
        )
        file_title = (
            _text(file.get("title"))
            or (
                _text(metadata.get("title"))
                if len(choices) == 1
                else Path(name).stem
            )
        )
        records.append(
            _candidate(
                provider="internet-archive",
                source_id=f"{item_id}/{name}",
                source_url=item_source_url,
                media_type="audio",
                title=file_title,
                creator=metadata.get("creator") or parent.get("creator"),
                declared_license=(
                    metadata.get("licenseurl")
                    or parent.get("declared_license")
                ),
                license_url=(
                    metadata.get("licenseurl")
                    or parent.get("license_url")
                ),
                rights_evidence_url=item_source_url,
                duration_seconds=_archive_duration(file.get("length")),
                asset_url=asset_url,
                content_evidence={
                    "collection": sorted(
                        _text(value)
                        for value in collection_values
                        if _text(value)
                    ),
                    "description": _text(metadata.get("description")),
                    "file_format": _text(file.get("format")),
                    "file_name": name,
                    "file_size_bytes": int(file.get("size") or 0),
                    "parent_candidate_id": parent["candidate_id"],
                    "source_kind": _text(file.get("source")),
                    "subject": sorted(
                        _text(value)
                        for value in subject_values
                        if _text(value)
                    ),
                    "track": _text(file.get("track")),
                },
            )
        )
    return records


class _MutopiaParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[dict[str, Any]]]] = []
        self._table_depth = 0
        self._is_result_table = False
        self._rows: list[list[dict[str, Any]]] = []
        self._row: list[dict[str, Any]] | None = None
        self._cell: dict[str, Any] | None = None

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        attributes = dict(attrs)
        if tag == "table":
            classes = (attributes.get("class") or "").split()
            if self._table_depth == 0 and "result-table" in classes:
                self._is_result_table = True
                self._rows = []
            if self._is_result_table:
                self._table_depth += 1
            return
        if not self._is_result_table:
            return
        if tag == "tr" and self._table_depth == 1:
            self._row = []
        elif tag == "td" and self._row is not None:
            self._cell = {"text": [], "links": []}
        elif tag == "a" and self._cell is not None:
            href = attributes.get("href")
            if href:
                self._cell["links"].append(href)

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell["text"].append(data)

    def handle_endtag(self, tag: str) -> None:
        if not self._is_result_table:
            return
        if tag == "td" and self._cell is not None and self._row is not None:
            self._cell["text"] = _text(" ".join(self._cell["text"]))
            self._row.append(self._cell)
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self._rows.append(self._row)
            self._row = None
        elif tag == "table":
            self._table_depth -= 1
            if self._table_depth == 0:
                if self._rows:
                    self.tables.append(self._rows)
                self._is_result_table = False


def normalize_mutopia(document: str) -> list[dict[str, Any]]:
    parser = _MutopiaParser()
    parser.feed(document)
    records = []
    for rows in parser.tables:
        content_rows = [
            row for row in rows
            if not (
                len(row) == 1
                and (
                    not row[0].get("links")
                    or "music preview" in _text(
                        row[0].get("text")
                    ).casefold()
                )
            )
        ]
        links = [
            href
            for row in content_rows
            for cell in row
            for href in cell.get("links", [])
        ]
        info_link = next(
            (href for href in links if "piece-info.cgi?id=" in href),
            None,
        )
        if not info_link or not content_rows or not content_rows[0]:
            continue
        query = urllib.parse.parse_qs(
            urllib.parse.urlparse(info_link).query
        )
        source_id = (query.get("id") or [""])[0]
        if not source_id:
            continue
        title = content_rows[0][0].get("text")
        creator = (
            content_rows[0][1].get("text")
            if len(content_rows[0]) > 1
            else "unknown"
        )
        license_cell = next(
            (
                cell
                for row in content_rows
                for cell in row
                if any(
                    "legal.html" in href
                    for href in cell.get("links", [])
                )
                or any(
                    marker in _text(cell.get("text")).casefold()
                    for marker in (
                        "creative commons",
                        "public domain",
                    )
                )
            ),
            {},
        )
        license_links = license_cell.get("links") or []
        license_url = (
            urllib.parse.urljoin(MUTOPIA_CATALOG_URL, license_links[0])
            if license_links
            else f"{MUTOPIA_BASE_URL}/legal.html"
        )
        formats = set()
        score_assets = []
        for href in links:
            lowered = href.casefold()
            if lowered.endswith(".ly") or lowered.endswith("-lys.zip"):
                asset_format = "lilypond"
            elif lowered.endswith(".mid") or lowered.endswith("-mids.zip"):
                asset_format = "midi"
            elif lowered.endswith(".pdf") or lowered.endswith("-pdfs.zip"):
                asset_format = "pdf"
            else:
                continue
            formats.add(asset_format)
            score_assets.append(
                {
                    "format": asset_format,
                    "url": urllib.parse.urljoin(MUTOPIA_CATALOG_URL, href),
                }
            )
        source_url = urllib.parse.urljoin(MUTOPIA_CATALOG_URL, info_link)
        instrument_row = content_rows[1] if len(content_rows) > 1 else []
        records.append(
            _candidate(
                provider="mutopia",
                source_id=source_id,
                source_url=source_url,
                media_type="score",
                title=title,
                creator=creator,
                declared_license=license_cell.get("text") or "unknown",
                license_url=license_url,
                rights_evidence_url=license_url,
                score_formats=formats,
                score_assets=score_assets,
                content_evidence={
                    "composition_date": (
                        _text(instrument_row[1].get("text"))
                        if len(instrument_row) > 1
                        else ""
                    ),
                    "instrument": (
                        _text(instrument_row[0].get("text"))
                        if instrument_row
                        else ""
                    ),
                    "style": (
                        _text(instrument_row[2].get("text"))
                        if len(instrument_row) > 2
                        else ""
                    ),
                },
            )
        )
    return sorted(records, key=lambda value: value["candidate_id"])


def normalize_youtube_metadata(
    records: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates = []
    for record in records:
        video_id = record.get("video_id") or record.get("id")
        source_url = (
            record.get("watch_url")
            or f"https://www.youtube.com/watch?v={urllib.parse.quote(_text(video_id))}"
        )
        candidates.append(
            _candidate(
                provider="youtube",
                source_id=video_id,
                source_url=source_url,
                media_type="audiovisual",
                title=record.get("title"),
                creator=record.get("channel") or record.get("creator"),
                declared_license=record.get("declared_license") or "unknown",
                license_url=record.get("license_url"),
                rights_evidence_url=source_url,
                duration_seconds=record.get("duration_seconds"),
            )
        )
    return sorted(candidates, key=lambda value: value["candidate_id"])


def pair_score_candidates(
    records: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates = [json.loads(json.dumps(record)) for record in records]
    scores_by_key: dict[tuple[str, str], list[str]] = {}
    for candidate in candidates:
        if candidate["media_type"] not in {"score", "symbolic-score"}:
            continue
        identity = candidate["work_identity"]
        key = (
            identity["normalized_title"],
            identity["normalized_creator"],
        )
        scores_by_key.setdefault(key, []).append(candidate["candidate_id"])
    for candidate in candidates:
        if candidate["media_type"] not in {"audio", "audiovisual"}:
            continue
        identity = candidate["work_identity"]
        key = (
            identity["normalized_title"],
            identity["normalized_creator"],
        )
        counterparts = sorted(scores_by_key.get(key, []))
        if counterparts:
            candidate["score_pair"] = {
                "status": "candidate-match",
                "counterpart_candidate_ids": counterparts,
                "match_basis": "normalized-title-and-creator",
            }
    return sorted(candidates, key=lambda value: value["candidate_id"])


def catalog_summary(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    candidates = list(records)
    duration_buckets: Counter[str] = Counter()
    for candidate in candidates:
        duration = candidate.get("duration_seconds")
        if duration is None:
            duration_buckets["unknown"] += 1
        elif duration < 30:
            duration_buckets["under-30-seconds"] += 1
        elif duration < 180:
            duration_buckets["30-to-179-seconds"] += 1
        else:
            duration_buckets["180-seconds-or-more"] += 1
    return {
        "candidates": len(candidates),
        "providers": dict(
            sorted(Counter(item["provider"] for item in candidates).items())
        ),
        "rights_tiers": dict(
            sorted(Counter(item["rights_tier"] for item in candidates).items())
        ),
        "media_types": dict(
            sorted(Counter(item["media_type"] for item in candidates).items())
        ),
        "duration_buckets": dict(sorted(duration_buckets.items())),
        "score_pair_status": dict(
            sorted(
                Counter(
                    item["score_pair"]["status"] for item in candidates
                ).items()
            )
        ),
        "qualified_for_sealed_evaluation": sum(
            bool(item["qualified_for_sealed_evaluation"])
            for item in candidates
        ),
    }


def build_catalog(
    records: Iterable[dict[str, Any]],
    *,
    provider_status: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    deduplicated: dict[str, dict[str, Any]] = {}
    for candidate in records:
        validate_candidate(candidate)
        candidate_id = candidate["candidate_id"]
        if (
            candidate_id in deduplicated
            and canonical_json(candidate) != canonical_json(deduplicated[candidate_id])
        ):
            raise ValueError(f"Conflicting candidate identity: {candidate_id}")
        deduplicated[candidate_id] = candidate
    candidates = pair_score_candidates(deduplicated.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": CATALOG_KIND,
        "policy": {
            "open-production": (
                "May be downloaded only after item-level rights audit."
            ),
            "restricted-research": (
                "Metadata only in automated discovery; no media download."
            ),
            "discovery-only": (
                "Metadata, source URL and fingerprints only; no media download."
            ),
        },
        "provider_status": dict(sorted((provider_status or {}).items())),
        "summary": catalog_summary(candidates),
        "records": candidates,
    }


def load_catalog(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("kind") != CATALOG_KIND
        or not isinstance(value.get("records"), list)
    ):
        raise ValueError("Unsupported internet candidate catalog")
    candidate_ids = set()
    for candidate in value["records"]:
        if not isinstance(candidate, dict):
            raise ValueError("Catalog records must be objects")
        validate_candidate(candidate)
        candidate_id = candidate["candidate_id"]
        if candidate_id in candidate_ids:
            raise ValueError(f"Duplicate candidate identity: {candidate_id}")
        candidate_ids.add(candidate_id)
    return value


def _request(url: str, *, timeout: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _request_json(url: str, *, timeout: float) -> dict[str, Any]:
    payload = json.loads(_request(url, timeout=timeout))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object from {url}")
    return payload


def _fetch_wikimedia_category(
    *,
    category: str,
    member_type: str,
    timeout: float,
) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    continuation: str | None = None
    while True:
        parameters = {
            "action": "query",
            "generator": "categorymembers",
            "gcmtitle": category,
            "gcmtype": member_type,
            "gcmlimit": "500",
            "prop": "info|imageinfo",
            "iiprop": "url|mime|size|extmetadata",
            "format": "json",
            "formatversion": "2",
        }
        if continuation:
            parameters["gcmcontinue"] = continuation
        url = f"{WIKIMEDIA_API_URL}?{urllib.parse.urlencode(parameters)}"
        payload = _request_json(url, timeout=timeout)
        query = payload.get("query")
        payload_pages = query.get("pages") if isinstance(query, dict) else None
        if isinstance(payload_pages, list):
            pages.extend(payload_pages)
        continuation_value = payload.get("continue")
        continuation = (
            _text(continuation_value.get("gcmcontinue"))
            if isinstance(continuation_value, dict)
            else ""
        )
        if not continuation:
            break
    return pages


def fetch_wikimedia(
    *,
    limit: int,
    timeout: float,
    include_subcategories: bool = True,
) -> list[dict[str, Any]]:
    root = "Category:Audio files of guitar music"
    categories = [
        root,
        "Category:Videos of guitars",
    ]
    if include_subcategories:
        subcategories = _fetch_wikimedia_category(
            category=root,
            member_type="subcat",
            timeout=timeout,
        )
        categories.extend(
            sorted(
                _text(page.get("title"))
                for page in subcategories
                if _text(page.get("title"))
                and "midi" not in _text(page.get("title")).casefold()
            )
        )
    records: dict[str, dict[str, Any]] = {}
    for category in categories:
        pages = _fetch_wikimedia_category(
            category=category,
            member_type="file",
            timeout=timeout,
        )
        payload = {"query": {"pages": pages}}
        for candidate in normalize_wikimedia(payload):
            existing = records.get(candidate["candidate_id"])
            if existing is None:
                candidate["content_evidence"]["discovery_categories"] = [
                    category
                ]
                records[candidate["candidate_id"]] = candidate
                continue
            discovery_categories = existing["content_evidence"].setdefault(
                "discovery_categories",
                [],
            )
            if category not in discovery_categories:
                discovery_categories.append(category)
                discovery_categories.sort()
        if len(records) >= limit:
            break
    return sorted(
        records.values(),
        key=lambda value: value["candidate_id"],
    )[:limit]


def fetch_openverse(
    *,
    pages: int,
    page_size: int,
    timeout: float,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for page in range(1, pages + 1):
        parameters = {
            "q": "solo guitar",
            "page": str(page),
            "page_size": str(page_size),
        }
        url = f"{OPENVERSE_API_URL}?{urllib.parse.urlencode(parameters)}"
        records.extend(
            normalize_openverse(_request_json(url, timeout=timeout))
        )
    return records


def fetch_internet_archive(
    *,
    pages: int,
    page_size: int,
    timeout: float,
    expansion_limit: int = 0,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for page in range(1, pages + 1):
        parameters = [
            (
                "q",
                'subject:"guitar" AND mediatype:audio AND licenseurl:*',
            ),
            ("fl[]", "identifier"),
            ("fl[]", "title"),
            ("fl[]", "creator"),
            ("fl[]", "licenseurl"),
            ("fl[]", "date"),
            ("fl[]", "collection"),
            ("fl[]", "description"),
            ("fl[]", "subject"),
            ("sort[]", "identifier asc"),
            ("rows", str(page_size)),
            ("page", str(page)),
            ("output", "json"),
        ]
        url = f"{INTERNET_ARCHIVE_API_URL}?{urllib.parse.urlencode(parameters)}"
        records.extend(
            normalize_internet_archive(_request_json(url, timeout=timeout))
        )
    if expansion_limit <= 0:
        return records
    parents = [
        candidate
        for candidate in sorted(
            records,
            key=lambda value: value["candidate_id"],
        )
        if candidate["rights_tier"] == "open-production"
    ][:expansion_limit]
    expanded: list[dict[str, Any]] = []

    def load(parent: dict[str, Any]) -> list[dict[str, Any]]:
        identifier = urllib.parse.quote(parent["source_id"], safe="")
        payload = _request_json(
            f"https://archive.org/metadata/{identifier}",
            timeout=timeout,
        )
        return normalize_internet_archive_files(payload, parent)

    with ThreadPoolExecutor(max_workers=min(8, len(parents) or 1)) as executor:
        futures = {
            executor.submit(load, parent): parent["candidate_id"]
            for parent in parents
        }
        for future in as_completed(futures):
            try:
                expanded.extend(future.result())
            except Exception:
                continue
    return records + sorted(
        expanded,
        key=lambda value: value["candidate_id"],
    )


def unexpanded_internet_archive_parents(
    records: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates = list(records)
    expanded_parent_ids = {
        parent
        for candidate in candidates
        if candidate["provider"] == "internet-archive"
        and candidate["media_type"] == "audio"
        and isinstance(candidate.get("content_evidence"), dict)
        and isinstance(
            parent := candidate["content_evidence"].get(
                "parent_candidate_id"
            ),
            str,
        )
        and parent
    }
    return [
        candidate
        for candidate in sorted(
            candidates,
            key=lambda value: value["candidate_id"],
        )
        if candidate["provider"] == "internet-archive"
        and candidate["media_type"] == "audio-collection"
        and candidate["rights_tier"] == "open-production"
        and candidate["candidate_id"] not in expanded_parent_ids
    ]


def expand_internet_archive_records(
    records: Iterable[dict[str, Any]],
    *,
    timeout: float,
    workers: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if workers < 1:
        raise ValueError("Archive expansion workers must be positive")
    parents = unexpanded_internet_archive_parents(records)
    expanded: list[dict[str, Any]] = []
    succeeded = 0
    failed = 0

    def load(parent: dict[str, Any]) -> list[dict[str, Any]]:
        identifier = urllib.parse.quote(parent["source_id"], safe="")
        payload = _request_json(
            f"https://archive.org/metadata/{identifier}",
            timeout=timeout,
        )
        return normalize_internet_archive_files(payload, parent)

    with ThreadPoolExecutor(
        max_workers=min(workers, len(parents) or 1)
    ) as executor:
        futures = {
            executor.submit(load, parent): parent["candidate_id"]
            for parent in parents
        }
        for future in as_completed(futures):
            try:
                expanded.extend(future.result())
                succeeded += 1
            except Exception:
                failed += 1
    return (
        sorted(expanded, key=lambda value: value["candidate_id"]),
        {
            "parents_requested": len(parents),
            "parents_succeeded": succeeded,
            "parents_failed": failed,
            "records_added": len(expanded),
        },
    )


def fetch_mutopia(
    *,
    pages: int,
    timeout: float,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for page in range(pages):
        parameters = {
            "startat": str(page * 10),
            "Instrument": "Guitar",
            "preview": "1",
        }
        url = f"{MUTOPIA_CATALOG_URL}?{urllib.parse.urlencode(parameters)}"
        document = _request(url, timeout=timeout).decode(
            "utf-8",
            errors="replace",
        )
        page_records = normalize_mutopia(document)
        if not page_records:
            break
        records.extend(page_records)
        if len(page_records) < 10:
            break
    return records


def _read_youtube_metadata(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected JSON object")
        records.append(value)
    return normalize_youtube_metadata(records)


def discover(arguments: argparse.Namespace) -> dict[str, Any]:
    providers = {
        value.strip()
        for value in arguments.providers.split(",")
        if value.strip()
    }
    supported = {
        "wikimedia",
        "openverse",
        "internet-archive",
        "mutopia",
    }
    unknown = providers - supported
    if unknown:
        raise ValueError(f"Unknown providers: {', '.join(sorted(unknown))}")
    records: list[dict[str, Any]] = []
    statuses: dict[str, dict[str, str]] = {}
    if arguments.seed_catalog:
        seed = load_catalog(Path(arguments.seed_catalog).resolve())
        records.extend(seed["records"])
        statuses.update(seed.get("provider_status") or {})
    fetchers = {
        "wikimedia": lambda: fetch_wikimedia(
            limit=arguments.wikimedia_limit,
            timeout=arguments.timeout,
        ),
        "openverse": lambda: fetch_openverse(
            pages=arguments.openverse_pages,
            page_size=arguments.page_size,
            timeout=arguments.timeout,
        ),
        "internet-archive": lambda: fetch_internet_archive(
            pages=arguments.internet_archive_pages,
            page_size=arguments.page_size,
            timeout=arguments.timeout,
            expansion_limit=arguments.internet_archive_expansion_limit,
        ),
        "mutopia": lambda: fetch_mutopia(
            pages=arguments.mutopia_pages,
            timeout=arguments.timeout,
        ),
    }
    for provider in sorted(providers):
        try:
            provider_records = fetchers[provider]()
        except Exception as error:
            statuses[provider] = {
                "status": "unavailable",
                "error_type": type(error).__name__,
            }
            continue
        records.extend(provider_records)
        statuses[provider] = {
            "status": "retrieved",
            "records": str(len(provider_records)),
        }
    if arguments.expand_seed_internet_archive:
        if not arguments.seed_catalog:
            raise ValueError(
                "--expand-seed-internet-archive requires --seed-catalog"
            )
        expanded, expansion = expand_internet_archive_records(
            records,
            timeout=arguments.timeout,
            workers=arguments.archive_expansion_workers,
        )
        records.extend(expanded)
        statuses["internet-archive"] = {
            "status": (
                "retrieved-and-expanded"
                if expansion["parents_failed"] == 0
                else "retrieved-and-partially-expanded"
            ),
            **{
                key: str(value)
                for key, value in expansion.items()
            },
        }
    if arguments.youtube_metadata:
        youtube_path = Path(arguments.youtube_metadata).resolve()
        youtube_records = _read_youtube_metadata(youtube_path)
        records.extend(youtube_records)
        statuses["youtube"] = {
            "status": "imported-metadata-only",
            "records": str(len(youtube_records)),
        }
    if not records:
        raise RuntimeError("No provider returned candidate metadata")
    catalog = build_catalog(records, provider_status=statuses)
    if arguments.expand_seed_internet_archive:
        catalog["provider_status"]["internet-archive"]["records"] = str(
            sum(
                record["provider"] == "internet-archive"
                for record in catalog["records"]
            )
        )
    output = Path(arguments.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json(catalog) + b"\n")
    if arguments.report:
        report = {
            "schema_version": SCHEMA_VERSION,
            "kind": "sealed-evaluation-internet-candidate-summary",
            "catalog_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "provider_status": catalog["provider_status"],
            "summary": catalog["summary"],
            "admission_note": (
                "Discovery records are not sealed-evaluation admissions. "
                "Item-level rights, content, identity and annotation audits "
                "remain required."
            ),
        }
        report_path = Path(arguments.report).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_bytes(canonical_json(report) + b"\n")
    return catalog


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Discover internet guitar candidates without downloading "
            "restricted or unknown media."
        ),
    )
    parser.add_argument(
        "--providers",
        default="wikimedia,openverse,internet-archive,mutopia",
    )
    parser.add_argument("--seed-catalog")
    parser.add_argument(
        "--expand-seed-internet-archive",
        action="store_true",
    )
    parser.add_argument(
        "--archive-expansion-workers",
        type=int,
        default=4,
    )
    parser.add_argument("--wikimedia-limit", type=int, default=500)
    parser.add_argument("--openverse-pages", type=int, default=1)
    parser.add_argument("--internet-archive-pages", type=int, default=5)
    parser.add_argument(
        "--internet-archive-expansion-limit",
        type=int,
        default=60,
    )
    parser.add_argument("--mutopia-pages", type=int, default=40)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--youtube-metadata")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report")
    arguments = parser.parse_args()
    if min(
        arguments.wikimedia_limit,
        arguments.openverse_pages,
        arguments.internet_archive_pages,
        arguments.mutopia_pages,
        arguments.page_size,
    ) < 1:
        raise ValueError("Discovery limits must be positive")
    if arguments.internet_archive_expansion_limit < 0:
        raise ValueError("Archive expansion limit cannot be negative")
    if not 1 <= arguments.archive_expansion_workers <= 8:
        raise ValueError("Archive expansion workers must be in 1..8")
    catalog = discover(arguments)
    print(json.dumps(catalog["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
