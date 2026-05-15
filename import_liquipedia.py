import time

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

import database
import riot_api
import data_collector


def normalise_handle(handle, default_tag="EUW"):
    """
    Cleans a Liquipedia handle and ensures it has a Riot Tag.
    Example: 'Leny' -> 'Leny#EUW', 'Caps#EUW' -> 'Caps#EUW'
    """
    handle = handle.strip().replace('\u00a0', ' ') # Clean non-breaking spaces
    if '#' in handle:
        return handle
    return f"{handle}#{default_tag}"

def extract_profile_data(html):
    """
    Parses a Liquipedia player profile HTML to find Riot ID and Alternate IDs.
    Supports both old (infobox-row) and new (infobox-cell-2) layouts.
    """
    soup = BeautifulSoup(html, 'html.parser')
    data = {"main": None, "smurfs": []}
    
    # Check for new layout (infobox-cell-2/description cells as labels)
    cells = soup.find_all('div', class_='infobox-description')
    for cell in cells:
        text = cell.get_text(strip=True).lower()
        if "riot id" in text or "alternate ids" in text:
            # Value is usually in the next sibling div
            val_cell = cell.find_next_sibling('div')
            if val_cell:
                val = val_cell.get_text(strip=True)
                if "riot id" in text:
                    data["main"] = val
                elif "alternate ids" in text:
                    data["smurfs"] = [s.strip() for s in val.split(',') if s.strip()]
    
    # Fallback to old layout (infobox-row) if nothing found yet
    if not data["main"] and not data["smurfs"]:
        rows = soup.find_all(class_="infobox-row")
        for row in rows:
            label = row.find(class_="infobox-label")
            if not label: continue
            label_text = label.get_text(strip=True).lower()
            desc = row.find(class_="infobox-description")
            if not desc: continue
            
            if "riot id" in label_text:
                data["main"] = desc.get_text(strip=True)
            elif "alternate ids" in label_text:
                data["smurfs"] = [s.strip() for s in desc.get_text(strip=True).split(',') if s.strip()]
            
    return data

def resolve_pro_name(pro_name: str) -> tuple[str, str] | None:
    pro_name = pro_name.strip().replace(' ', '_')
    url = f"https://liquipedia.net/leagueoflegends/{pro_name}"
    headers = {"User-Agent": "NNriot/1.0 (Scraper for personal LoL data tool)"}
    try:
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
    except Exception:
        return None

    data = extract_profile_data(resp.text)
    main_id = data.get("main")
    
    if main_id:
        main_h = normalise_handle(main_id)
        if '#' in main_h:
            name, tag = main_h.split('#', 1)
            return (name, tag)
            
    return None

def get_player_links(html):
    """
    Parses the tournament stats page to find links to player profiles.
    Correctly targets the player column (index 1).
    """
    soup = BeautifulSoup(html, 'html.parser')
    table = soup.find('table', class_='sortable')
    links = []
    if table:
        for row in table.find_all('tr')[1:]: # Skip header
            cells = row.find_all('td')
            # Column 0 is Team, Column 1 is Player
            if len(cells) > 1 and cells[1].find('a'):
                href = cells[1].find('a')['href']
                if '/leagueoflegends/' in href:
                    # Construct absolute URL
                    full_url = "https://liquipedia.net" + href
                    if full_url not in links:
                        links.append(full_url)
    return links

# Load for standalone use
load_dotenv()

def import_from_liquipedia(stats_url, matches_per_player=50):
    """
    Main pipeline: discovery -> extraction -> resolution -> ingestion.
    """
    database.init_db()
    headers = {"User-Agent": "NNriot/1.0 (Scraper for personal LoL data tool)"}
    
    print(f"[*] Fetching tournament stats from {stats_url}...")
    try:
        resp = requests.get(stats_url, headers=headers)
        resp.raise_for_status()
    except Exception as e:
        print(f"[!] Error fetching stats: {e}")
        return

    player_urls = get_player_links(resp.text)
    print(f"[*] Found {len(player_urls)} unique player profiles.")
    
    # Use EUW as default region for LFL
    api = riot_api.RiotAPI("EUW")
    
    for i, url in enumerate(player_urls):
        player_name_slug = url.split('/')[-1]
        print(f"[{i+1}/{len(player_urls)}] Processing {player_name_slug}...")
        
        # Respect Liquipedia rate limits
        if i > 0:
            time.sleep(3)
        
        try:
            p_resp = requests.get(url, headers=headers)
            p_resp.raise_for_status()
        except Exception as e:
            print(f"  [!] Failed to fetch profile: {e}")
            continue
            
        data = extract_profile_data(p_resp.text)
        main_handle = data["main"] or player_name_slug
        
        # Resolve Main Account
        main_h = normalise_handle(main_handle)
        name, tag = main_h.split('#')
        main_puuid = api.get_puuid(name, tag)
        
        if not main_puuid:
            print(f"  [?] Could not resolve main account handle: {main_h}")
            continue
            
        # Save canonical player
        database.save_player(main_puuid, name, tag)
        accounts_to_collect = [{"puuid": main_puuid, "server": "EUW", "gamename": name, "tagline": tag}]
        print(f"  [OK] Resolved main: {main_h}")
        
        # Resolve Smurfs (Alternate IDs)
        for smurf in data["smurfs"]:
            s_h = normalise_handle(smurf)
            s_name, s_tag = s_h.split('#')
            s_puuid = api.get_puuid(s_name, s_tag)
            
            if s_puuid:
                database.save_player(s_puuid, s_name, s_tag)
                database.link_accounts(main_puuid, s_puuid)
                accounts_to_collect.append({"puuid": s_puuid, "server": "EUW", "gamename": s_name, "tagline": s_tag})
                print(f"  [OK] Linked smurf: {s_h}")
        
        # Collect match history if accounts were found
        if accounts_to_collect:
            print(f"  [*] Collecting metadata and {matches_per_player} matches for {len(accounts_to_collect)} accounts...")
            try:
                data_collector.collect_by_puuid(accounts_to_collect, matches_per_player=matches_per_player)
            except Exception as e:
                print(f"  [!] Collection failed: {e}")

if __name__ == "__main__":
    import sys
    import io

    # Ensure stdout handles unicode characters (e.g. for Shōnen)
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

    LFL_STATS_URL = "https://liquipedia.net/leagueoflegends/LFL/2026/Spring/Player_Stats"
    # Matches per player set to 50 as requested
    import_from_liquipedia(LFL_STATS_URL, matches_per_player=50)
