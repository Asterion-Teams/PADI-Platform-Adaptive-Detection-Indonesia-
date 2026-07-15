from flask import Flask
from app.utils import load_config, load_stats, sync_stats_with_config, warm_history_from_db, recover_downtime_gaps
from app.services.camera import start_camera_agents
from app.database import init_db
import app.globals as g
import threading


def create_app():
    # Initialize Globals
    g.CCTV_SOURCES = load_config()
    if g.CCTV_SOURCES:
        active = next((s for s in (g.CCTV_SOURCES or []) if (s or {}).get("active") is True and (s or {}).get("url")), None)
        g.VIDEO_SOURCE = (active or g.CCTV_SOURCES[0])["url"]

    g.global_stats = load_stats()

    # Initialize Database
    init_db()

    # Sync stats with config (Remove zombie entries)
    sync_stats_with_config()

    # Recover gaps if app was down (fills history up to "now" using recent pattern)
    try:
        warm_history_from_db(hours=24)
        threading.Thread(target=recover_downtime_gaps, daemon=True).start()
    except Exception as e:
        print(f"[WARN] Downtime recovery failed: {e}")

    # Auto-backfill vehicle identity for violations missing details
    try:
        from app.services.vehicle_agent import start_agent
        start_agent()
    except Exception as e:
        print(f"[WARN] Vehicle Agent init failed: {e}")

    app = Flask(__name__)

    # Session secret key
    from app.auth import SECRET_KEY
    app.secret_key = SECRET_KEY

    # ── Security: Initialize CSRF protection ─────────────────────────────────
    try:
        from app.csrf import init_csrf
        init_csrf(app)
        print("[SECURITY] CSRF protection initialized")
    except Exception as e:
        print(f"[SECURITY] CSRF init failed: {e}")

    # ── Security: Register rate limit status endpoint ────────────────────────
    try:
        from app.ratelimit import rate_limit_status
        @app.route("/api/ratelimit/status")
        def ratelimit_status_api():
            limit_name = request.args.get("limit", "default")
            status = rate_limit_status(limit_name=limit_name)
            return jsonify(status)

        # Exempt health check from rate limiting
        @app.route("/health")
        def health():
            return jsonify({"status": "ok"})

        @app.route("/api/csrf_token", methods=["GET"])
        def get_csrf_token_api():
            """Return CSRF token for AJAX requests."""
            from app.csrf import generate_csrf_token
            return jsonify({"csrf_token": generate_csrf_token()})
    except Exception as e:
        print(f"[SECURITY] Rate limit init failed: {e}")

    # ── Template helpers ────────────────────────────────────────────────────
    from flask import request, jsonify
    @app.context_processor
    def inject_csrf():
        """Make csrf_token() available in all templates automatically."""
        try:
            from app.csrf import generate_csrf_token
            return {"csrf_token": generate_csrf_token}
        except Exception:
            return {}

    # Register Blueprints
    from app.routes import bp
    app.register_blueprint(bp)

    return app
