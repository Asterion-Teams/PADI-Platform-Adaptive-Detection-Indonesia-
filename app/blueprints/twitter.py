"""
Twitter / social media search routes.
Extracted from routes.py for modularity.
"""
import re
import json
import datetime
import urllib.request
import urllib.error
import urllib.parse
import html
from flask import Blueprint, jsonify, request
import app.config as cfg

bp = Blueprint('twitter', __name__)


@bp.route("/api/twitter/search")
def api_twitter_search():
    """Search for traffic-related news/social media posts.
    Primary: Google News RSS (free, no API key).
    Fallback: Twitter API v2 if Bearer Token has credits.
    """
    query = request.args.get("q") or cfg.TWITTER_SEARCH_QUERY
    max_results = min(int(request.args.get("max_results") or cfg.TWITTER_MAX_RESULTS), 30)

    # Primary: Google News RSS
    try:
        encoded_q = urllib.parse.quote(query)
        rss_url = f"https://news.google.com/rss/search?q={encoded_q}&hl=id&gl=ID&ceid=ID:id"
        req = urllib.request.Request(rss_url)
        req.add_header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) SmartTrafficAI/1.0")

        with urllib.request.urlopen(req, timeout=10) as resp:
            xml_data = resp.read().decode("utf-8")

        import xml.etree.ElementTree as ET
        root = ET.fromstring(xml_data)
        channel = root.find("channel")
        items = channel.findall("item") if channel is not None else []

        tweets = []
        for item in items[:max_results]:
            title = item.findtext("title") or ""
            link = item.findtext("link") or ""
            pub_date = item.findtext("pubDate") or ""
            source = item.findtext("source") or ""
            description = item.findtext("description") or ""
            description = re.sub(r"<[^>]+>", "", html.unescape(description))[:200]

            tweets.append({
                "id": link,
                "text": title,
                "description": description,
                "created_at": pub_date,
                "author_name": source,
                "author_username": "news",
                "author_avatar": "",
                "link": link,
                "metrics": {},
                "source": "google_news",
            })

        if tweets:
            from email.utils import parsedate_to_datetime
            def _parse_date(t):
                try:
                    return parsedate_to_datetime(t.get("created_at") or "")
                except Exception:
                    return datetime.datetime(2000, 1, 1)
            tweets.sort(key=_parse_date, reverse=True)
            return jsonify({"status": "success", "tweets": tweets, "query": query, "count": len(tweets), "source": "google_news"})
    except Exception as e:
        pass  # Fall through to Twitter API

    # Fallback: Twitter API v2
    if cfg.TWITTER_BEARER_TOKEN:
        try:
            url = (f"https://api.twitter.com/2/tweets/search/recent"
                   f"?query={urllib.parse.quote(cfg.TWITTER_SEARCH_QUERY)}"
                   f"&max_results={max(10, max_results)}"
                   f"&tweet.fields=created_at,author_id,public_metrics,text"
                   f"&expansions=author_id&user.fields=name,username,profile_image_url")
            req = urllib.request.Request(url)
            req.add_header("Authorization", f"Bearer {cfg.TWITTER_BEARER_TOKEN}")
            req.add_header("User-Agent", "SmartTrafficAI/1.0")
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())

            tweets = []
            users_map = {}
            if "includes" in data and "users" in data["includes"]:
                for u in data["includes"]["users"]:
                    users_map[u["id"]] = u
            for t in (data.get("data") or []):
                author = users_map.get(t.get("author_id"), {})
                tweets.append({
                    "id": t.get("id"),
                    "text": t.get("text"),
                    "created_at": t.get("created_at"),
                    "author_name": author.get("name", ""),
                    "author_username": author.get("username", ""),
                    "author_avatar": author.get("profile_image_url", ""),
                    "metrics": t.get("public_metrics", {}),
                    "source": "twitter",
                })
            return jsonify({"status": "success", "tweets": tweets, "query": cfg.TWITTER_SEARCH_QUERY, "count": len(tweets), "source": "twitter"})
        except Exception:
            pass

    return jsonify({"status": "error", "message": "Tidak dapat memuat berita. Coba lagi nanti."}), 503
