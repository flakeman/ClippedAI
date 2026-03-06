import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Callable

from yt_dlp import YoutubeDL


CURRENT_YEAR = datetime.now().year


@dataclass
class SearchIntent:
    raw_query: str
    query_text: str
    limit: int
    target_year: Optional[int] = None
    sort_by_popularity: bool = False


class YtdlpLogger:
    def __init__(self, logger: Optional[Callable[[str], None]] = None):
        self._logger = logger

    def debug(self, msg: str):
        if self._logger and msg and not msg.startswith("[debug] "):
            self._logger(msg)

    def warning(self, msg: str):
        if self._logger and msg:
            self._logger(f"Warning: {msg}")

    def error(self, msg: str):
        if self._logger and msg:
            self._logger(f"Error: {msg}")


def _parse_limit(query: str) -> int:
    match = re.search(r"\btop\s*(\d+)\b", query, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"\bтоп\s*(\d+)\b", query, flags=re.IGNORECASE)
    if match:
        return max(1, min(int(match.group(1)), 20))
    return 5


def _parse_target_year(query: str) -> Optional[int]:
    year_match = re.search(r"\b(19\d{2}|20\d{2})\b", query)
    if year_match:
        return int(year_match.group(1))

    rel_match = re.search(r"\b(\d+)\s+(?:years?\s+ago|лет\s+назад|года?\s+назад)\b", query, flags=re.IGNORECASE)
    if rel_match:
        return max(1970, CURRENT_YEAR - int(rel_match.group(1)))
    return None


def parse_natural_query(query: str) -> SearchIntent:
    query = (query or "").strip()
    lowered = query.lower()
    limit = _parse_limit(lowered)
    target_year = _parse_target_year(lowered)
    sort_by_popularity = any(token in lowered for token in ["popular", "viral", "most viewed", "популяр", "вирус", "топ"])

    cleaned = query
    cleaned = re.sub(r"\btop\s*\d+\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bтоп\s*\d+\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(19\d{2}|20\d{2})\b", "", cleaned)
    cleaned = re.sub(r"\b\d+\s+(?:years?\s+ago|лет\s+назад|года?\s+назад)\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.-")

    return SearchIntent(
        raw_query=query,
        query_text=cleaned or query,
        limit=limit,
        target_year=target_year,
        sort_by_popularity=sort_by_popularity,
    )


def _build_search_queries(intent: SearchIntent) -> List[str]:
    queries = []
    base = intent.query_text
    if intent.target_year:
        if intent.sort_by_popularity:
            queries.append(f"most viewed {base} {intent.target_year}")
            queries.append(f"popular {base} {intent.target_year}")
            queries.append(f"viral {base} {intent.target_year}")
            queries.append(f"популярные {base} {intent.target_year}")
        queries.append(f"{base} {intent.target_year}")
    else:
        queries.append(intent.raw_query)
        queries.append(base)
    seen = set()
    unique = []
    for item in queries:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(item.strip())
    return unique[:4]


def _normalize_result(entry: Dict) -> Optional[Dict[str, str]]:
    if not entry:
        return None
    webpage_url = entry.get("webpage_url") or ""
    if not webpage_url and entry.get("id"):
        webpage_url = f"https://www.youtube.com/watch?v={entry['id']}"
    upload_date = str(entry.get("upload_date") or "")
    year = upload_date[:4] if len(upload_date) >= 4 else ""
    return {
        "title": entry.get("title") or "Untitled",
        "url": entry.get("url") or webpage_url,
        "webpage_url": webpage_url,
        "uploader": entry.get("uploader") or "",
        "duration": str(entry.get("duration") or ""),
        "view_count": int(entry.get("view_count") or 0),
        "upload_date": upload_date,
        "year": year,
    }


def _score_result(result: Dict[str, str], intent: SearchIntent) -> tuple:
    views = int(result.get("view_count") or 0)
    year = int(result["year"]) if result.get("year", "").isdigit() else None
    year_score = 0
    if intent.target_year and year is not None:
        year_score = -abs(year - intent.target_year)
    elif intent.target_year and year is None:
        year_score = -999
    popularity_score = views if intent.sort_by_popularity else 0
    return (year_score, popularity_score, views)


def is_url(value: str) -> bool:
    return bool(re.match(r"^https?://", (value or "").strip(), flags=re.IGNORECASE))


def search_youtube(query: str, limit: int = 5, logger: Optional[Callable[[str], None]] = None) -> List[Dict[str, str]]:
    query = (query or "").strip()
    if not query:
        return []

    intent = parse_natural_query(query)
    if logger:
        logger(
            f"Parsed search intent: query='{intent.query_text}', limit={intent.limit}, "
            f"target_year={intent.target_year or '-'}, sort_by_popularity={intent.sort_by_popularity}"
        )
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "noplaylist": True,
        "ignoreerrors": True,
        "no_warnings": True,
        "logger": YtdlpLogger(logger),
    }
    raw_results = []
    with YoutubeDL(ydl_opts) as ydl:
        for search_query in _build_search_queries(intent):
            if logger:
                logger(f"Running source search query: {search_query}")
            info = ydl.extract_info(f"ytsearch{max(limit * 4, 20)}:{search_query}", download=False)
            entries = info.get("entries", []) if isinstance(info, dict) else []
            if logger:
                logger(f"Search query returned {len(entries)} raw result(s)")
            for entry in entries:
                normalized = _normalize_result(entry)
                if normalized:
                    raw_results.append(normalized)

    deduped = {}
    for item in raw_results:
        key = item.get("webpage_url") or item.get("url") or item.get("title")
        if key and key not in deduped:
            deduped[key] = item

    results = list(deduped.values())
    if logger:
        logger(f"Deduped results: {len(results)}")
    if intent.target_year:
        results = [
            item for item in results
            if item.get("year", "").isdigit() and abs(int(item["year"]) - intent.target_year) <= 1
        ] or results
        if logger:
            logger(f"After year filter around {intent.target_year}: {len(results)}")

    results.sort(key=lambda item: _score_result(item, intent), reverse=True)
    if logger and results:
        best = results[0]
        logger(
            f"Top result: {best.get('title', 'Untitled')} "
            f"(year={best.get('year') or '-'}, views={best.get('view_count') or 0})"
        )
    return results[: intent.limit]


def download_video(source: str, output_dir: str, logger: Optional[Callable[[str], None]] = None) -> Path:
    source = (source or "").strip()
    if not source:
        raise ValueError("Source is empty.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    target = source if is_url(source) else f"ytsearch1:{source}"
    ydl_opts = {
        "format": "bv*+ba/b",
        "outtmpl": str(output_path / "%(title).180B [%(id)s].%(ext)s"),
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "logger": YtdlpLogger(logger),
    }

    before = {p.resolve() for p in output_path.glob("*")}
    if logger:
        logger(f"Starting download target: {target}")
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([target])
    after = [p.resolve() for p in output_path.glob("*.mp4") if p.resolve() not in before]
    if not after:
        raise RuntimeError("Download completed but no mp4 file was produced.")
    latest = max(after, key=lambda p: p.stat().st_mtime)
    if logger:
        logger(f"Download produced file: {latest.name}")
    return latest
