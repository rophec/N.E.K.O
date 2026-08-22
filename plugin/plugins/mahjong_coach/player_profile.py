from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from .models import PlayerProfile


FOUR_PLAYER_MODES = "8,9,11,12,15,16"
DEFAULT_BASE_URL = "https://5-data.amae-koromo.com/api/v2/pl4"
CACHE_TTL_SECONDS = 24 * 60 * 60


class PlayerProfileProvider(Protocol):
    source: str

    def search(
        self,
        nickname: str,
        *,
        limit: int = 10,
        force_refresh: bool = False,
    ) -> list[dict[str, Any]]: ...

    def fetch_profile(
        self,
        account_id: str,
        *,
        nickname: str = "",
        force_refresh: bool = False,
    ) -> PlayerProfile: ...


class ProfileLookupError(RuntimeError):
    pass


class AmaeKoromoProvider:
    source = "amae_koromo"

    def __init__(
        self,
        *,
        cache_path: Path,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = 3.0,
        cache_ttl_seconds: int = CACHE_TTL_SECONDS,
    ) -> None:
        self.cache_path = cache_path.expanduser().resolve()
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = max(0.5, float(timeout_seconds))
        self.cache_ttl_seconds = max(60, int(cache_ttl_seconds))

    def search(
        self,
        nickname: str,
        *,
        limit: int = 10,
        force_refresh: bool = False,
    ) -> list[dict[str, Any]]:
        nickname = str(nickname or "").strip()
        if not nickname:
            return []
        safe_limit = max(1, min(20, int(limit)))
        cache_key = f"search:{nickname.casefold()}:{safe_limit}"
        cached = None if force_refresh else self._cache_get(cache_key)
        if isinstance(cached, list):
            return [dict(item) for item in cached if isinstance(item, dict)]
        payload = self._request_json(
            f"{self.base_url}/search_player/{quote(nickname, safe='')}?"
            + urlencode({"limit": safe_limit, "tag": "all"})
        )
        if not isinstance(payload, list):
            raise ProfileLookupError("player search returned an invalid response")
        results = [self._normalize_candidate(item) for item in payload if isinstance(item, dict)]
        results = [item for item in results if item.get("account_id")]
        self._cache_put(cache_key, results)
        return results

    def fetch_profile(
        self,
        account_id: str,
        *,
        nickname: str = "",
        force_refresh: bool = False,
    ) -> PlayerProfile:
        account_id = str(account_id or "").strip()
        if not account_id.isdigit():
            raise ProfileLookupError("invalid player account id")
        cache_key = f"profile:{account_id}"
        cached = None if force_refresh else self._cache_get(cache_key)
        if isinstance(cached, dict):
            return PlayerProfile.from_payload(cached)
        end_timestamp = int(time.time())
        query = urlencode({"mode": FOUR_PLAYER_MODES}, safe=",")
        stats = self._request_json(
            f"{self.base_url}/player_stats/{account_id}/1262304000/{end_timestamp}?{query}"
        )
        extended = self._request_json(
            f"{self.base_url}/player_extended_stats/{account_id}/1262304000/{end_timestamp}?{query}"
        )
        if not isinstance(stats, dict) or not isinstance(extended, dict):
            raise ProfileLookupError("player statistics returned an invalid response")
        level = stats.get("level") if isinstance(stats.get("level"), dict) else {}
        candidate = self._profile_from_stats(
            account_id=account_id,
            nickname=str(stats.get("nickname") or nickname or "").strip(),
            level_id=_to_int(level.get("id")),
            stats=stats,
            extended=extended,
        )
        self._cache_put(cache_key, candidate.to_dict())
        return candidate

    def confirm(self, profile: PlayerProfile) -> PlayerProfile:
        return replace(profile, confirmed=True, source=self.source, fetched_at=profile.fetched_at or time.time())

    def _request_json(self, url: str) -> Any:
        request = Request(url, headers={"User-Agent": "N.E.K.O-Mahjong-Coach/0.3"})
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise ProfileLookupError(f"Amae-Koromo unavailable: {type(exc).__name__}") from exc

    def _normalize_candidate(self, item: dict[str, Any]) -> dict[str, Any]:
        level = item.get("level") if isinstance(item.get("level"), dict) else {}
        level_id = _to_int(level.get("id"))
        return {
            "account_id": str(item.get("id") or ""),
            "nickname": str(item.get("nickname") or ""),
            "rank": rank_from_level_id(level_id),
            "level_id": level_id,
            "level_score": _to_int(level.get("score")),
            "latest_timestamp": _to_int(item.get("latest_timestamp")),
            "source": self.source,
        }

    def _profile_from_stats(
        self,
        *,
        account_id: str,
        nickname: str,
        level_id: int,
        stats: dict[str, Any],
        extended: dict[str, Any],
    ) -> PlayerProfile:
        win_rate = _rate(extended.get("和牌率"))
        deal_in_rate = _rate(extended.get("放铳率"))
        riichi_rate = _rate(extended.get("立直率"))
        call_rate = _rate(extended.get("副露率"))
        risk = suggested_risk_tolerance(deal_in_rate, riichi_rate, call_rate)
        call_bias = "open" if (call_rate or 0.0) >= 0.38 else "closed" if (call_rate or 1.0) <= 0.24 else "balanced"
        goal = "speed" if call_bias == "open" else "value" if (riichi_rate or 0.0) >= 0.22 else "balanced"
        return PlayerProfile(
            rank=rank_from_level_id(level_id),
            room="unknown",
            risk_tolerance=risk,
            goal_bias=goal,
            call_bias=call_bias,
            source=self.source,
            account_id=account_id,
            nickname=nickname,
            confirmed=False,
            fetched_at=time.time(),
            sample_count=max(_to_int(stats.get("count")), _to_int(extended.get("count"))),
            win_rate=win_rate,
            deal_in_rate=deal_in_rate,
            riichi_rate=riichi_rate,
            call_rate=call_rate,
        )

    def _load_cache(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        return payload if isinstance(payload, dict) else {}

    def _cache_get(self, key: str) -> Any:
        item = self._load_cache().get(key)
        if not isinstance(item, dict):
            return None
        if time.time() - float(item.get("cached_at") or 0.0) > self.cache_ttl_seconds:
            return None
        return item.get("value")

    def _cache_put(self, key: str, value: Any) -> None:
        payload = self._load_cache()
        payload[key] = {"cached_at": time.time(), "value": value}
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, self.cache_path)


def rank_from_level_id(level_id: int) -> str:
    # Mahjong Soul encodes the rank family in the ten-thousands place
    # (1xxxx novice through 6xxxx celestial); the remaining digits identify
    # the sub-rank and score rules.
    major = max(0, int(level_id)) // 10_000
    return {
        1: "novice",
        2: "adept",
        3: "expert",
        4: "master",
        5: "saint",
        6: "celestial",
    }.get(major, "unknown")


def suggested_risk_tolerance(
    deal_in_rate: float | None,
    riichi_rate: float | None,
    call_rate: float | None,
) -> str:
    aggression = (riichi_rate or 0.18) * 0.8 + (call_rate or 0.30) * 0.45 + (deal_in_rate or 0.13) * 0.35
    if aggression >= 0.36:
        return "aggressive"
    if aggression <= 0.28:
        return "conservative"
    return "balanced"


def _rate(value: Any) -> float | None:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
