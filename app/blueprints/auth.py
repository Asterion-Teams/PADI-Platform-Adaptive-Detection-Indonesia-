"""
Auth routes: login, logout, operator dashboard.
Separated from main routes.py for modularity.
"""
from flask import Blueprint, render_template, redirect, request, jsonify, session
from app.auth import check_login, get_current_user, is_demo_auth_bypass

bp = Blueprint('auth_routes', __name__)


@bp.route("/")
def index():
    user = get_current_user()
    if not user:
        return redirect("/login")
    role = user.get("role") if user else None
    if role == "admin":
        return redirect("/dashboard")
    return redirect("/enforcement")


@bp.route("/login", methods=["GET", "POST"])
def login_page():
    user = get_current_user()
    if user:
        role = user.get("role") if user else None
        if role == "admin":
            return redirect("/dashboard")
        return redirect("/enforcement")

    if request.method == "GET":
        return render_template("login.html")

    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")

    user = check_login(username, password)
    if user:
        session["user"] = user
        role = user.get("role") if user else None
        if role == "admin":
            return redirect("/dashboard")
        return redirect("/enforcement")

    return render_template("login.html", error="Invalid credentials. Please try again.", username=username)


@bp.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/login")


@bp.route("/operator")
def operator_dashboard():
    user = get_current_user()
    if not user or user.get("role") != "operator":
        return redirect("/login")
    return render_template("operator.html", user=user)


@bp.route("/api/auth/status")
def auth_status():
    user = get_current_user()
    if user:
        return jsonify({
            "status": "authenticated",
            "user": {"username": user.get("username"), "role": user.get("role"), "name": user.get("name")},
            "demo_mode": is_demo_auth_bypass(),
        })
    return jsonify({"status": "anonymous", "demo_mode": is_demo_auth_bypass()})


@bp.route("/api/auth/login", methods=["POST"])
def api_login():
    data = request.get_json(silent=True) or {}
    username = str(data.get("username") or "").strip()
    password = str(data.get("password") or "")
    if not username or not password:
        return jsonify({"status": "error", "message": "Username and password required"}), 400
    user = check_login(username, password)
    if user:
        session["user"] = user
        return jsonify({"status": "success", "user": {"username": user.get("username"), "role": user.get("role"), "name": user.get("name")}})
    return jsonify({"status": "error", "message": "Invalid credentials"}), 401


@bp.route("/api/auth/logout", methods=["POST"])
def api_logout():
    session.pop("user", None)
    return jsonify({"status": "success"})
