"""
Social Media Scraper using Playwright
--------------------------------------
Scrapes Twitter/X search results for @DishubDKI mentions without API keys.
Uses headless Chromium to render JavaScript-heavy pages.

Runs as a background task, caching results to avoid excessive scraping.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from datetime import datetime

# Cache for scraped results
_cache_lock = threading.Lock()
_cached_mentions: list[dict] = []
_last_scrape_ts: float = 0.0
_cached_query: str = ""  # Track which query produced the cache
_CACHE_TTL_SECONDS = 120  # Re-scrape every 2 minutes max
_refresh_lock = threading.Lock()
_refresh_in_progress: bool = False


def _get_current_query() -> str:
    try:
        import app.config as _cfg_check
        return getattr(_cfg_check, 'X_SEARCH_QUERY', '')
    except Exception:
        return ''


def _refresh_cache_async(queries: list[str], max_results: int, current_query: str) -> None:
    global _refresh_in_progress
    try:
        _scrape_and_update_cache(queries=queries, max_results=max_results, current_query=current_query)
    finally:
        with _refresh_lock:
            _refresh_in_progress = False


def _schedule_background_refresh(queries: list[str], max_results: int, current_query: str) -> bool:
    global _refresh_in_progress
    with _refresh_lock:
        if _refresh_in_progress:
            return False
        _refresh_in_progress = True
    threading.Thread(
        target=_refresh_cache_async,
        args=(list(queries or []), int(max_results or 20), current_query),
        daemon=True,
    ).start()
    return True


def _scrape_and_update_cache(queries: list[str], max_results: int, current_query: str) -> list[dict]:
    global _cached_mentions, _last_scrape_ts, _cached_query

    if queries is None:
        queries = [
            "DishubDKI macet OR kemacetan OR parkir liar OR kecelakaan",
            "@DishubDKI jakarta lalu lintas",
            "tag DishubDKI lapor",
        ]

    all_mentions = []

    # Strategy 1: Scrape X/Twitter directly (public timeline) - PRIMARY
    try:
        results = _scrape_x_direct(queries)
        all_mentions.extend(results)
    except Exception as e:
        print(f"[SOCIAL-SCRAPER] X direct scrape error: {e}")
    
    # Strategy 2: Try Nitter for Twitter specifically - FALLBACK
    if len(all_mentions) < 3:
        try:
            results = _scrape_nitter(queries, max_results)
            all_mentions.extend(results)
        except Exception as e:
            print(f"[SOCIAL-SCRAPER] Nitter scrape error: {e}")
    
    # Deduplicate by title/text
    seen = set()
    unique = []
    for m in all_mentions:
        key = m.get("title", "")[:50]
        if key and key not in seen:
            seen.add(key)
            unique.append(m)
    
    # Sort by date
    unique.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
    
    # Update cache
    with _cache_lock:
        _cached_mentions = unique[:max_results]
        _last_scrape_ts = time.time()
        _cached_query = current_query

    return unique[:max_results]


def scrape_twitter_mentions(queries: list[str] = None, max_results: int = 20) -> list[dict]:
    """
    Scrape social media mentions of @DishubDKI using Playwright.
    Covers: Twitter/X, Threads, Instagram, Facebook via Google search.

    If stale cache is available for the same query, return it immediately and
    refresh in the background so UI requests do not block on browser startup.
    """
    current_query = _get_current_query()

    with _cache_lock:
        cached_mentions = _cached_mentions[:max_results]
        cache_valid = (
            (time.time() - _last_scrape_ts) < _CACHE_TTL_SECONDS
            and bool(_cached_mentions)
            and _cached_query == current_query
        )
        cache_same_query = bool(_cached_mentions) and _cached_query == current_query

    if cache_valid:
        return cached_mentions

    if cache_same_query:
        _schedule_background_refresh(queries, max_results, current_query)
        return cached_mentions

    return _scrape_and_update_cache(queries=queries, max_results=max_results, current_query=current_query)


def _scrape_google_all_platforms(queries: list[str], max_results: int) -> list[dict]:
    """Scrape Google search for @DishubDKI mentions across ALL social platforms:
    Twitter/X, Threads, Instagram, Facebook.
    Uses consent bypass and realistic browser behavior.
    """
    from playwright.sync_api import sync_playwright
    
    results = []
    
    # Platform-specific search queries - ONLY X/Twitter
    platform_searches = [
        # Twitter/X only
        {"query_suffix": "site:twitter.com OR site:x.com", "platform": "twitter"},
    ]
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=['--disable-blink-features=AutomationControlled', '--no-sandbox']
            )
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                viewport={"width": 1366, "height": 768},
                locale="id-ID",
                extra_http_headers={"Accept-Language": "id-ID,id;q=0.9,en;q=0.8"},
            )
            # Set Google consent cookie to bypass consent page
            context.add_cookies([
                {"name": "CONSENT", "value": "YES+cb.20231204-08-p0.id+FX+111", "domain": ".google.com", "path": "/"},
                {"name": "SOCS", "value": "CAISHAgDEhJnd3NfMjAyMzEyMDQtMF9SQzEaAmVuIAEaBgiA_LiqBg", "domain": ".google.com", "path": "/"},
            ])
            # Block images/css for speed
            context.route("**/*.{png,jpg,jpeg,gif,svg,css,woff,woff2}", lambda route: route.abort())
            
            page = context.new_page()
            
            for platform_info in platform_searches:
                platform = platform_info["platform"]
                site_filter = platform_info["query_suffix"]
                
                for query in queries[:1]:
                    try:
                        # Use DuckDuckGo (more permissive than Google for scraping)
                        search_q = f"{query} {site_filter}"
                        search_url = f"https://duckduckgo.com/html/?q={search_q}&t=h_&ia=web"
                        
                        page.goto(search_url, timeout=15000, wait_until="domcontentloaded")
                        page.wait_for_timeout(2000)
                        
                        # DuckDuckGo HTML version has simple structure
                        items = page.query_selector_all(".result, .web-result, .results_links")
                        
                        if not items:
                            # Try regular DuckDuckGo
                            search_url2 = f"https://duckduckgo.com/?q={search_q}&t=h_&ia=web"
                            page.goto(search_url2, timeout=12000, wait_until="domcontentloaded")
                            page.wait_for_timeout(3000)
                            items = page.query_selector_all("[data-testid='result'], article, .nrn-react-div")
                        
                        for item in items[:6]:
                            try:
                                title_el = item.query_selector("a.result__a, h2 a, a[data-testid='result-title-a']")
                                snippet_el = item.query_selector(".result__snippet, a.result__snippet, [data-testid='result-snippet']")
                                
                                title = title_el.inner_text() if title_el else ""
                                link = title_el.get_attribute("href") if title_el else ""
                                snippet = snippet_el.inner_text() if snippet_el else ""
                                
                                if not title or not link:
                                    continue
                                
                                # Verify platform domain
                                platform_domains = {
                                    "twitter": ["twitter.com", "x.com"],
                                    "threads": ["threads.net"],
                                    "instagram": ["instagram.com"],
                                    "facebook": ["facebook.com", "fb.com"],
                                }
                                domains = platform_domains.get(platform, [])
                                if not any(d in link for d in domains):
                                    continue
                                
                                username = _extract_username(link, platform)
                                text_lower = (title + " " + snippet).lower()
                                priority = "normal"
                                if any(k in text_lower for k in ("kecelakaan", "tabrakan", "darurat", "korban")):
                                    priority = "high"
                                elif any(k in text_lower for k in ("macet parah", "lumpuh", "banjir")):
                                    priority = "high"
                                
                                auto_type = _classify_mention(text_lower)
                                
                                results.append({
                                    "title": title,
                                    "description": snippet[:200],
                                    "source": username or platform.capitalize(),
                                    "link": link,
                                    "pub_date": "",
                                    "timestamp": time.time(),
                                    "priority": priority,
                                    "auto_type": auto_type,
                                    "platform": platform,
                                })
                            except Exception:
                                continue
                        
                        time.sleep(1.5)
                        
                    except Exception as e:
                        print(f"[SOCIAL-SCRAPER] {platform} query failed: {e}")
                        continue
            
            browser.close()
    except Exception as e:
        print(f"[SOCIAL-SCRAPER] Playwright error: {e}")
    
    return results


def _extract_username(url: str, platform: str) -> str:
    """Extract username from social media URL."""
    try:
        if platform == "twitter":
            m = re.search(r'(?:twitter\.com|x\.com)/(\w+)', url)
            return f"@{m.group(1)}" if m else ""
        elif platform == "threads":
            m = re.search(r'threads\.net/@?(\w+)', url)
            return f"@{m.group(1)}" if m else ""
        elif platform == "instagram":
            m = re.search(r'instagram\.com/(?:p/\w+|(\w+))', url)
            return f"@{m.group(1)}" if m and m.group(1) else ""
        elif platform == "facebook":
            m = re.search(r'facebook\.com/(\w+)', url)
            return m.group(1) if m else ""
    except Exception:
        pass
    return ""


def _scrape_x_direct(queries: list[str]) -> list[dict]:
    """Scrape X.com (Twitter) search using authenticated session cookies.
    Cookies are loaded from data/x_cookies.json (exported from browser).
    """
    from playwright.sync_api import sync_playwright
    
    results = []
    cookies_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "x_cookies.json")
    
    if not os.path.exists(cookies_path):
        print("[SOCIAL-SCRAPER] X cookies not found at data/x_cookies.json")
        return results
    
    try:
        with open(cookies_path, 'r') as f:
            raw_cookies = json.load(f)
    except Exception as e:
        print(f"[SOCIAL-SCRAPER] Failed to load X cookies: {e}")
        return results
    
    # Convert cookie format for Playwright
    pw_cookies = []
    for c in raw_cookies:
        cookie = {
            "name": c["name"],
            "value": c["value"],
            "domain": c["domain"],
            "path": c.get("path", "/"),
            "secure": c.get("secure", True),
            "httpOnly": c.get("httpOnly", False),
        }
        if c.get("expirationDate"):
            cookie["expires"] = float(c["expirationDate"])
        if c.get("sameSite"):
            ss = str(c["sameSite"]).lower()
            if ss == "no_restriction":
                cookie["sameSite"] = "None"
            elif ss == "lax":
                cookie["sameSite"] = "Lax"
            elif ss == "strict":
                cookie["sameSite"] = "Strict"
        pw_cookies.append(cookie)
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=['--disable-blink-features=AutomationControlled', '--no-sandbox']
            )
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                viewport={"width": 1366, "height": 900},
            )
            
            # Load cookies
            context.add_cookies(pw_cookies)
            
            page = context.new_page()
            
            # Search for @DishubDKI mentions (query from settings)
            import app.config as _cfg
            search_query = getattr(_cfg, 'X_SEARCH_QUERY', '@DishubDKI OR #DishubDKI OR to:DishubDKI')
            search_url = f"https://x.com/search?q={search_query}&src=typed_query&f=live"
            
            print(f"[SOCIAL-SCRAPER] Loading X.com search: {search_query}")
            page.goto(search_url, timeout=20000, wait_until="domcontentloaded")
            
            # Wait for tweets to render
            page.wait_for_timeout(4000)
            
            # Try to find tweet elements
            # X.com uses article[data-testid="tweet"] for tweets
            tweets = page.query_selector_all('article[data-testid="tweet"]')
            print(f"[SOCIAL-SCRAPER] Found {len(tweets)} tweets on X.com")
            
            for tweet in tweets[:15]:
                try:
                    # Get tweet text
                    text_el = tweet.query_selector('[data-testid="tweetText"]')
                    text = text_el.inner_text() if text_el else ""
                    if not text or len(text) < 5:
                        continue
                    
                    # Filter: must actually mention DishubDKI or be traffic-related
                    text_lower = text.lower()
                    is_relevant = (
                        "dishub" in text_lower or
                        "dki" in text_lower or
                        any(k in text_lower for k in (
                            "macet", "kemacetan", "parkir", "lalu lintas", "lalin",
                            "kecelakaan", "busway", "transjakarta", "tilang", "e-tle",
                            "jalanan", "jalan", "tol", "simpang", "lampu merah",
                            "banjir", "genangan", "longsor", "pohon tumbang",
                            "padat", "lancar", "tersendat", "stuck", "gridlock",
                            "motor", "mobil", "truk", "angkot", "ojol",
                            "pelanggaran", "lawan arah", "marka", "zebra cross",
                            "perbaikan jalan", "proyek", "galian",
                        ))
                    )
                    if not is_relevant:
                        continue
                    
                    # Get username
                    user_el = tweet.query_selector('div[data-testid="User-Name"] a[role="link"]')
                    username = ""
                    tweet_link = ""
                    if user_el:
                        href = user_el.get_attribute("href") or ""
                        if href:
                            username = f"@{href.strip('/').split('/')[-1]}"
                    
                    # Get tweet link (for timestamp)
                    time_el = tweet.query_selector('time')
                    pub_date = ""
                    if time_el:
                        pub_date = time_el.get_attribute("datetime") or ""
                    
                    link_el = tweet.query_selector('a[href*="/status/"]')
                    if link_el:
                        tweet_link = "https://x.com" + (link_el.get_attribute("href") or "")
                    
                    # Classify
                    text_lower = text.lower()
                    priority = "normal"
                    if any(k in text_lower for k in ("kecelakaan", "tabrakan", "darurat", "korban", "meninggal")):
                        priority = "high"
                    elif any(k in text_lower for k in ("macet parah", "lumpuh", "banjir")):
                        priority = "high"
                    
                    auto_type = _classify_mention(text_lower)
                    
                    results.append({
                        "title": text[:150],
                        "description": text[:300],
                        "source": username or "@unknown",
                        "link": tweet_link or "https://x.com/search?q=%40DishubDKI",
                        "pub_date": pub_date,
                        "timestamp": time.time(),
                        "priority": priority,
                        "auto_type": auto_type,
                        "platform": "twitter",
                    })
                except Exception:
                    continue
            
            browser.close()
            
    except Exception as e:
        print(f"[SOCIAL-SCRAPER] X direct error: {e}")
    
    return results


def _scrape_nitter(queries: list[str], max_results: int) -> list[dict]:
    """Scrape Nitter (Twitter mirror) for mentions."""
    from playwright.sync_api import sync_playwright
    
    # Nitter instances (some may be down)
    nitter_instances = [
        "https://nitter.privacydev.net",
        "https://nitter.poast.org",
        "https://nitter.net",
    ]
    
    results = []
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                viewport={"width": 1280, "height": 720},
            )
            page = context.new_page()
            
            for instance in nitter_instances:
                try:
                    search_url = f"{instance}/search?f=tweets&q=%40DishubDKI"
                    page.goto(search_url, timeout=10000)
                    page.wait_for_load_state("domcontentloaded", timeout=8000)
                    
                    # Check if page loaded successfully
                    if "timeline-item" not in page.content():
                        continue
                    
                    tweets = page.query_selector_all(".timeline-item")
                    
                    for tweet in tweets[:max_results]:
                        try:
                            username_el = tweet.query_selector(".username")
                            content_el = tweet.query_selector(".tweet-content")
                            time_el = tweet.query_selector(".tweet-date a")
                            
                            username = username_el.inner_text() if username_el else ""
                            content = content_el.inner_text() if content_el else ""
                            tweet_time = time_el.get_attribute("title") if time_el else ""
                            tweet_link = time_el.get_attribute("href") if time_el else ""
                            
                            if not content:
                                continue
                            
                            text_lower = content.lower()
                            priority = "normal"
                            if any(k in text_lower for k in ("kecelakaan", "tabrakan", "darurat")):
                                priority = "high"
                            
                            auto_type = _classify_mention(text_lower)
                            
                            results.append({
                                "title": content[:150],
                                "description": content,
                                "source": username or "Twitter User",
                                "link": f"https://twitter.com{tweet_link}" if tweet_link else "",
                                "pub_date": tweet_time,
                                "timestamp": time.time(),
                                "priority": priority,
                                "auto_type": auto_type,
                                "platform": "twitter",
                            })
                        except Exception:
                            continue
                    
                    if results:
                        break  # Got results from this instance, stop trying others
                        
                except Exception:
                    continue
            
            browser.close()
    except Exception as e:
        print(f"[SOCIAL-SCRAPER] Nitter error: {e}")
    
    return results


def _classify_mention(text_lower: str) -> str | None:
    """Auto-classify a social media mention into violation/issue type."""
    if any(k in text_lower for k in ("parkir liar", "parkir sembarangan", "parkir ilegal", "parkir di")):
        return "illegal_parking"
    if any(k in text_lower for k in ("busway", "transjakarta", "jalur bus", "steril")):
        return "busway_occupancy"
    if any(k in text_lower for k in ("sepeda", "jalur sepeda", "bike lane", "pesepeda")):
        return "bicycle_lane"
    if any(k in text_lower for k in ("lawan arah", "lawan arus", "wrong way", "counter flow")):
        return "wrong_way"
    if any(k in text_lower for k in ("macet", "kemacetan", "padat", "stuck", "gridlock", "tersendat")):
        return "traffic_congestion"
    if any(k in text_lower for k in ("kecelakaan", "tabrakan", "laka lantas", "tabrak", "kecelakaan lalu")):
        return "accident"
    if any(k in text_lower for k in ("banjir", "genangan", "tergenang")):
        return "flooding"
    if any(k in text_lower for k in ("lampu merah", "traffic light", "sinyal mati")):
        return "traffic_signal"
    if any(k in text_lower for k in ("jalan rusak", "berlubang", "perbaikan jalan", "galian")):
        return "road_damage"
    return None
