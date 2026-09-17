import collections
import logging
import os
import requests
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv, dotenv_values

# Load environment variables
load_dotenv(override=True)

logger = logging.getLogger(__name__)


class RiotAuthError(Exception):
    """Raised when Riot API rejects the API key (HTTP 401 or 403)."""
    pass


class RiotRateLimiter:
    """
    Thread-safe sliding-window rate limiter designed specifically for Riot's rate limits:
    - 20 requests per 1 second
    - 100 requests per 120 seconds
    """

    def __init__(self, per_second: int = 18, per_2min: int = 95):
        self.per_second = per_second
        self.per_2min = per_2min
        self.lock = threading.Lock()
        self.timestamps_1s = collections.deque()
        self.timestamps_120s = collections.deque()

    def acquire(self):
        if os.environ.get("RIOT_DISABLE_RATE_LIMIT") == "1":
            return

        while True:
            wait_time = 0.0
            with self.lock:
                now = time.time()
                while self.timestamps_1s and now - self.timestamps_1s[0] >= 1.0:
                    self.timestamps_1s.popleft()
                while self.timestamps_120s and now - self.timestamps_120s[0] >= 120.0:
                    self.timestamps_120s.popleft()

                if len(self.timestamps_1s) >= self.per_second:
                    wait_time = max(wait_time, 1.0 - (now - self.timestamps_1s[0]) + 0.05)
                if len(self.timestamps_120s) >= self.per_2min:
                    wait_time = max(wait_time, 120.0 - (now - self.timestamps_120s[0]) + 0.2)

                if wait_time <= 0:
                    self.timestamps_1s.append(now)
                    self.timestamps_120s.append(now)
                    return

            if wait_time > 0:
                time.sleep(wait_time)


class RiotAPI:
    # Class-level tracking to share call counts across instances/threads
    _total_calls = 0
    _lock = threading.Lock()
    _last_limits = {}  # Tracks {region: {app_limit: x, app_count: y, methods: {}}}
    _limiter = RiotRateLimiter(per_second=18, per_2min=95)

    def __init__(self, region="KR"):
        self.api_key = os.getenv("RIOT_API_KEY")
        if not self.api_key:
            raise ValueError("RIOT_API_KEY not found in .env file")

        self.session = requests.Session()

        # Region mappings for routing values
        self.region_routes = {
            "BR": "americas",
            "BR1": "americas",
            "EUN": "europe",
            "EUN1": "europe",
            "EUW": "europe",
            "EUW1": "europe",
            "JP": "asia",
            "JP1": "asia",
            "KR": "asia",
            "LA1": "americas",
            "LA2": "americas",
            "NA": "americas",
            "NA1": "americas",
            "OC1": "sea",
            "OCE": "sea",
            "RU": "europe",
            "TR": "europe",
            "TR1": "europe",
        }

        # Platform routing values
        self.platform_routes = {
            "BR": "br1",
            "BR1": "br1",
            "EUN": "eun1",
            "EUN1": "eun1",
            "EUW": "euw1",
            "EUW1": "euw1",
            "JP": "jp1",
            "JP1": "jp1",
            "KR": "kr",
            "LA1": "la1",
            "LA2": "la2",
            "NA": "na1",
            "NA1": "na1",
            "OC1": "oc1",
            "OCE": "oc1",
            "RU": "ru",
            "TR": "tr1",
            "TR1": "tr1",
        }

        self.region = region.upper()
        if self.region not in self.region_routes:
            raise ValueError(
                f"Unsupported region: {region}. Supported: {list(self.region_routes.keys())}"
            )

        self.regional_url = (
            f"https://{self.region_routes[self.region]}.api.riotgames.com"
        )
        self.platform_url = (
            f"https://{self.platform_routes[self.region]}.api.riotgames.com"
        )

    def close(self):
        """Close the underlying requests.Session."""
        if hasattr(self, "session") and self.session:
            self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def reload_key(self) -> str | None:
        """Reload RIOT_API_KEY from .env file when developer key expires."""
        try:
            vals = dotenv_values(".env")
            new_key = vals.get("RIOT_API_KEY")
            if new_key and new_key != self.api_key:
                self.api_key = new_key
                os.environ["RIOT_API_KEY"] = new_key
                return new_key
        except Exception as e:
            logger.error("Error reading .env during key reload: %s", e)
        return None

    def _get_headers(self):
        return {"X-Riot-Token": self.api_key}

    def _make_request(self, url, max_retries=3):
        """Helper to cleanly make requests with automatic retries for rate limits (429) & server errors."""
        for attempt in range(max_retries):
            # Proactively enforce sliding-window rate limit
            RiotAPI._limiter.acquire()
            headers = self._get_headers()
            try:
                response = self.session.get(url, headers=headers, timeout=15)
                
                # Proactively track rate limit headers
                self._update_rate_limits(response.headers)
                
                if response.status_code == 200:
                    with RiotAPI._lock:
                        RiotAPI._total_calls += 1
                    return response
                elif response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 5))
                    logger.warning(
                        "Rate limited (429). Waiting for %s seconds... (Attempt %s/%s)",
                        retry_after,
                        attempt + 1,
                        max_retries,
                    )
                    time.sleep(retry_after)
                    continue
                elif response.status_code in (401, 403):
                    logger.warning(
                        "Riot API returned HTTP %s (expired or invalid key). Attempting key reload from .env...",
                        response.status_code,
                    )
                    self.reload_key()
                    if self.api_key and self.api_key != headers.get("X-Riot-Token"):
                        logger.info("New RIOT_API_KEY reloaded from .env. Retrying request...")
                        continue
                    raise RiotAuthError(
                        f"Riot API token rejected with HTTP {response.status_code}. "
                        "Please update RIOT_API_KEY in your .env file."
                    )
                elif response.status_code >= 500:
                    logger.warning(
                        "Server error %s. Retrying in 5 seconds... (Attempt %s/%s)",
                        response.status_code,
                        attempt + 1,
                        max_retries,
                    )
                    time.sleep(5)
                    continue
                else:
                    logger.warning("API Error %s for URL: %s", response.status_code, url)
                    return response
            except requests.exceptions.RequestException as e:
                logger.error(
                    f"Request exception: {e}. Retrying in 5 seconds... (Attempt {attempt+1}/{max_retries})",
                    exc_info=True,
                )
                time.sleep(5)
            except (OSError, IOError) as e:
                # Catches SSL errors, broken pipe, connection reset, etc.
                logger.error(
                    f"Network/SSL error: {e}. Retrying in 5 seconds... (Attempt {attempt+1}/{max_retries})",
                    exc_info=True,
                )
                time.sleep(5)

        logger.error("Failed to fetch %s after %s attempts.", url, max_retries)
        return None

    def _update_rate_limits(self, headers):
        """Parse Riot rate limit headers to keep track of usage."""
        app_limit = headers.get("X-App-Rate-Limit")
        app_count = headers.get("X-App-Rate-Limit-Count")
        
        if app_limit and app_count:
            # We track per region because each platform has its own bucket
            RiotAPI._last_limits[self.region] = {
                "app_limit": app_limit,
                "app_count": app_count,
                "method_limit": headers.get("X-Method-Rate-Limit"),
                "method_count": headers.get("X-Method-Rate-Limit-Count"),
                "timestamp": time.time()
            }

    @classmethod
    def get_global_stats(cls):
        """Returns total successful API calls and the current rate limit status."""
        return {
            "total_calls": cls._total_calls,
            "limits": cls._last_limits
        }

    def get_puuid(self, game_name, tag_line):
        """Account-V1: Get PUUID by Riot ID."""
        url = f"{self.regional_url}/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}"
        response = self._make_request(url)
        if response and response.status_code == 200:
            return response.json()["puuid"]
        return None

    def get_account_by_puuid(self, puuid):
        """Account-V1: Get game_name and tag_line by PUUID."""
        url = f"{self.regional_url}/riot/account/v1/accounts/by-puuid/{puuid}"
        response = self._make_request(url)
        if response and response.status_code == 200:
            data = response.json()
            return {"game_name": data.get("gameName"), "tag_line": data.get("tagLine")}
        return None

    def get_match_ids(
        self, puuid, count=20, start_time=None, end_time=None, queue=None, type=None
    ):
        """Match-V5: Get a list of match IDs for a PUUID with optional queue, type, and time filtering."""
        params = ["start=0", f"count={count}"]

        if queue is not None:
            params.append(f"queue={queue}")
        if type:
            params.append(f"type={type}")
        if start_time:
            # Riot API expects Unix timestamp in seconds
            params.append(f"startTime={int(start_time)}")
        if end_time:
            params.append(f"endTime={int(end_time)}")

        query_params = "&".join(params)
        url = f"{self.regional_url}/lol/match/v5/matches/by-puuid/{puuid}/ids?{query_params}"
        response = self._make_request(url)
        if response and response.status_code == 200:
            return response.json()
        return []

    def get_match_details(self, match_id):
        """Match-V5: Get full match details."""
        url = f"{self.regional_url}/lol/match/v5/matches/{match_id}"
        response = self._make_request(url)
        if response and response.status_code == 200:
            return response.json()
        return None

    def get_match_details_batch(self, match_ids, max_workers=5, verbose=False):
        """Fetch multiple matches in parallel using a thread pool."""
        results = {}
        total = len(match_ids)
        completed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_mid = {
                executor.submit(self.get_match_details, mid): mid for mid in match_ids
            }
            for future in as_completed(future_to_mid):
                mid = future_to_mid[future]
                completed += 1
                try:
                    data = future.result()
                    if data:
                        results[mid] = data
                        if verbose:
                            logger.info("      [%s/%s] Téléchargement : %s", completed, total, mid)
                except Exception as e:
                    logger.error(f"      Error fetching match {mid}: {e}", exc_info=True)
        return results

    def get_match_timeline(self, match_id):
        """Match-V5: Get the full timeline (frames + events) for a match.

        Endpoint: /lol/match/v5/matches/{match_id}/timeline
        Returns the timeline dict or None on failure.

        Rate limit: separate method-level bucket from get_match_details.
        Doubles the API call cost when fetching new matches.
        """
        url = f"{self.regional_url}/lol/match/v5/matches/{match_id}/timeline"
        response = self._make_request(url)
        if response and response.status_code == 200:
            return response.json()
        return None

    def get_match_timelines_batch(self, match_ids, max_workers=5, verbose=False):
        """Fetch multiple timelines in parallel using a thread pool.

        Same shape as get_match_details_batch.
        """
        results = {}
        total = len(match_ids)
        completed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_mid = {
                executor.submit(self.get_match_timeline, mid): mid for mid in match_ids
            }
            for future in as_completed(future_to_mid):
                mid = future_to_mid[future]
                completed += 1
                try:
                    data = future.result()
                    if data:
                        results[mid] = data
                        if verbose:
                            logger.info("      [%s/%s] Timeline: %s", completed, total, mid)
                except Exception as e:
                    logger.error(f"Error fetching timeline for {mid}: {e}", exc_info=True)
        return results

    def get_featured_games(self) -> list:
        """Spectator-V4: Retrieve the list of currently featured games.
        Returns a list of game dicts or an empty list on failure.
        """
        url = f"{self.platform_url}/lol/spectator/v4/featured-games"
        response = self._make_request(url)
        if response and response.status_code == 200:
            return response.json().get("gameList", [])
        return []

    def get_active_game_by_summoner(self, puuid: str) -> dict | None:
        """Spectator-V4: Get the live game data for a summoner identified by PUUID.
        Returns the game dict if the summoner is currently in a game, otherwise None.
        """
        url = f"{self.platform_url}/lol/spectator/v4/active-games/by-summoner/{puuid}"
        response = self._make_request(url)
        if response:
            if response.status_code == 200:
                return response.json()
            # 404 means no active game
            if response.status_code == 404:
                return None
        return None

    def get_top_league(self, tier="challenger", queue="RANKED_SOLO_5x5"):
        """League-V4: Get top players in a specific league (challenger, grandmaster, master)."""
        tier = tier.lower()
        if tier not in ["challenger", "grandmaster", "master"]:
            raise ValueError("Tier must be challenger, grandmaster, or master")

        url = f"{self.platform_url}/lol/league/v4/{tier}leagues/by-queue/{queue}"
        response = self._make_request(url)
        if response and response.status_code == 200:
            return response.json().get("entries", [])
        return []

    def get_puuid_by_summoner_id(self, summoner_id):
        """Summoner-V4: Get PUUID by summoner ID."""
        url = f"{self.platform_url}/lol/summoner/v4/summoners/{summoner_id}"
        response = self._make_request(url)
        if response and response.status_code == 200:
            return response.json().get("puuid")
        return None

    def get_top_players(self, total_count=1000):
        """Combines Challenger, Grandmaster, and Master to get top X players."""
        all_entries = []

        # 1. Fetch Challenger
        logger.info("Fetching Challenger players...")
        all_entries.extend(self.get_top_league("challenger"))

        # 2. Fetch Grandmaster if needed
        if len(all_entries) < total_count:
            logger.info("Fetching Grandmaster players...")
            all_entries.extend(self.get_top_league("grandmaster"))

        # 3. Fetch Master if needed
        if len(all_entries) < total_count:
            logger.info("Fetching Master players...")
            all_entries.extend(self.get_top_league("master"))

        # Sort by league points to get the absolute best
        all_entries.sort(key=lambda x: x.get("leaguePoints", 0), reverse=True)

        return all_entries[:total_count]

    def get_player_summary(self, game_name, tag_line, match_count=5):
        """Fetches recent match performance for a player."""
        puuid = self.get_puuid(game_name, tag_line)
        if not puuid:
            return {"error": "Player not found"}

        match_ids = self.get_match_ids(puuid, count=match_count)
        stats = {"name": game_name, "win_rate": 0, "avg_kda": 0, "recent_matches": []}

        wins = 0
        total_kda = 0
        valid_matches = 0

        for mid in match_ids:
            details = self.get_match_details(mid)
            if not details:
                continue

            # Find the participant in this match
            p_info = next(
                (p for p in details["info"]["participants"] if p["puuid"] == puuid),
                None,
            )
            if p_info:
                wins += 1 if p_info["win"] else 0
                kda = (p_info["kills"] + p_info["assists"]) / max(1, p_info["deaths"])
                total_kda += kda
                valid_matches += 1
                stats["recent_matches"].append(
                    {
                        "champion": p_info["championName"],
                        "win": p_info["win"],
                        "kda": round(kda, 2),
                    }
                )

        if valid_matches > 0:
            stats["win_rate"] = wins / valid_matches
            stats["avg_kda"] = total_kda / valid_matches

        return stats


def get_faker_history(count=5):
    """Utility to quickly get Faker's recent match data."""
    api = RiotAPI()
    return api.get_player_summary("Hide on bush", "KR1", match_count=count)


def get_matchup_data(team_a_players, team_b_players):
    """
    team_a_players: List of (name, tag) tuples
    team_b_players: List of (name, tag) tuples
    """
    api = RiotAPI()
    matchup = {
        "team_a": [api.get_player_summary(n, t) for n, t in team_a_players],
        "team_b": [api.get_player_summary(n, t) for n, t in team_b_players],
    }
    return matchup


if __name__ == "__main__":
    # Test fetch
    print("Connecting to Riot API for Faker (Hide on bush#KR1)...")
    stats = get_faker_history(5)

    if "error" in stats:
        print(f"Error: {stats['error']}")
    else:
        print(f"Retrieved summary for {stats['name']}.")
        print(f"Win Rate: {stats['win_rate']:.1%}, Avg KDA: {stats['avg_kda']:.2f}")

        if stats["recent_matches"]:
            first = stats["recent_matches"][0]
            print(
                f"Latest Match: Champion: {first['champion']}, Win: {first['win']}, KDA: {first['kda']}"
            )
