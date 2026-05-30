import os
import sys
import time
import platform

sys.dont_write_bytecode = True

os.environ.setdefault("OLLAMA_MODEL", "mistral:7b")

deps_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".deps")
if os.path.isdir(deps_dir) and deps_dir not in sys.path:
    sys.path.insert(0, deps_dir)

# ─────────────────────────────────────────────────────────────────────────────
# CLI Branding & Startup
# ─────────────────────────────────────────────────────────────────────────────

CYAN    = "\033[96m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
RESET   = "\033[0m"
MAGENTA = "\033[95m"
WHITE   = "\033[97m"

BANNER = f"""
{CYAN}{BOLD}
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                                                                      ║
    ║      █████╗ ███████╗████████╗███████╗██████╗ ██╗ ██████╗ ███╗   ██╗  ║
    ║     ██╔══██╗██╔════╝╚══██╔══╝██╔════╝██╔══██╗██║██╔═══██╗████╗  ██║  ║
    ║     ███████║███████╗   ██║   █████╗  ██████╔╝██║██║   ██║██╔██╗ ██║  ║
    ║     ██╔══██║╚════██║   ██║   ██╔══╝  ██╔══██╗██║██║   ██║██║╚██╗██║  ║
    ║     ██║  ██║███████║   ██║   ███████╗██║  ██║██║╚██████╔╝██║ ╚████║  ║
    ║     ╚═╝  ╚═╝╚══════╝   ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝  ║
    ║                                                                      ║
    ║     {WHITE}S M A R T   T R A F F I C   A I{CYAN}                              ║
    ║     {DIM}Intelligent Traffic Enforcement & Behaviour Analysis{CYAN}{BOLD}        ║
    ║                                                                      ║
    ╚══════════════════════════════════════════════════════════════════════╝
{RESET}"""


def print_section(title):
    print(f"\n  {DIM}{'─' * 60}{RESET}")
    print(f"  {BOLD}{WHITE}▸ {title}{RESET}")
    print(f"  {DIM}{'─' * 60}{RESET}")


def print_status(label, value, color=GREEN):
    print(f"    {DIM}•{RESET} {label:<28} {color}{BOLD}{value}{RESET}")


def print_ok(msg):
    print(f"    {GREEN}✓{RESET} {msg}")


def print_warn(msg):
    print(f"    {YELLOW}⚠{RESET} {msg}")


if __name__ == "__main__":
    os.system("")  # Enable ANSI on Windows

    print(BANNER)
    print(f"  {DIM}AI Open Innovation Challenge 2026 — Case 1: DISHUB DKI Jakarta{RESET}")
    print(f"  {DIM}Intelligent Traffic Enforcement & Behaviour Analysis (E-TLE){RESET}")
    print(f"  {MAGENTA}{BOLD}Team Asterion{RESET}")
    print()

    # ── System Info ──────────────────────────────────────────────────────────
    print_section("System Information")
    print_status("Platform", f"{platform.system()} {platform.release()} ({platform.machine()})")
    print_status("Python", platform.python_version())

    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
            print_status("GPU", f"{gpu_name} ({gpu_mem})", GREEN)
            print_status("CUDA", torch.version.cuda, GREEN)
        else:
            print_status("GPU", "Not available (CPU mode)", YELLOW)
    except ImportError:
        print_status("GPU", "PyTorch not installed", RED)

    # ── Initialize App ───────────────────────────────────────────────────────
    print_section("Initializing Application")

    t0 = time.time()
    from app import create_app
    from app.services.camera import start_camera_agents
    from app.config import HOST_IP, HOST_PORT, YOLO_CUSTOM_PATH, USE_CUSTOM_YOLO

    app = create_app()
    print_ok("Flask application created")
    print_ok("Database initialized")

    # ── Model Loading ────────────────────────────────────────────────────────
    print_section("AI Model")

    if USE_CUSTOM_YOLO and os.path.isfile(YOLO_CUSTOM_PATH):
        model_name = os.path.basename(YOLO_CUSTOM_PATH)
        model_size = f"{os.path.getsize(YOLO_CUSTOM_PATH) / 1024 / 1024:.1f} MB"
        print_status("Model", f"{model_name} ({model_size})", GREEN)
        print_status("Type", "Custom trained (Roboflow v3)", GREEN)
        print_status("Classes", "bus, car, microbus, motorbike, pickup-van, truck", CYAN)
    else:
        print_status("Model", "yolov8l.pt (COCO generic)", YELLOW)
        print_status("Classes", "COCO vehicle subset", YELLOW)

    # ── Start Camera Agents ──────────────────────────────────────────────────
    print_section("Camera Agents")
    start_camera_agents()

    from app.globals import CCTV_SOURCES, camera_agents
    n_cameras = len(CCTV_SOURCES) if CCTV_SOURCES else 0
    n_agents = len(camera_agents)
    print_status("Configured cameras", str(n_cameras))
    print_status("Active agents", str(n_agents), GREEN if n_agents > 0 else YELLOW)
    # ── Enforcement Engine ───────────────────────────────────────────────────
    print_section("Enforcement Engine (E-TLE)")
    from app.config import VIOLATIONS_ENABLED, ANPR_ENABLED, ILLEGAL_PARKING_MIN_SECONDS
    print_status("Violation detection", "ENABLED" if VIOLATIONS_ENABLED else "DISABLED",
                 GREEN if VIOLATIONS_ENABLED else RED)
    print_status("ANPR (plate recognition)", "ENABLED" if ANPR_ENABLED else "DISABLED",
                 GREEN if ANPR_ENABLED else YELLOW)
    print_status("Parking threshold", f"{ILLEGAL_PARKING_MIN_SECONDS}s")
    print_status("Zone types", "no_parking, busway, bicycle, bus_stop", CYAN)

    # ── Server Ready ─────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print_section("Server")

    local_url = f"http://127.0.0.1:{HOST_PORT}"
    print_status("Status", "READY", GREEN)
    print_status("Startup time", f"{elapsed:.2f}s")
    print_status("Host", f"{HOST_IP}:{HOST_PORT}")
    print()
    print(f"  {BOLD}{GREEN}  ┌──────────────────────────────────────────────────┐{RESET}")
    print(f"  {BOLD}{GREEN}  │                                                  │{RESET}")
    print(f"  {BOLD}{GREEN}  │   🌐  {WHITE}Dashboard:    {CYAN}{local_url}/dashboard{GREEN}     │{RESET}")
    print(f"  {BOLD}{GREEN}  │   🛡️   {WHITE}Enforcement: {CYAN}{local_url}/enforcement{GREEN}  │{RESET}")
    print(f"  {BOLD}{GREEN}  │   📋  {WHITE}Zone Editor:  {CYAN}{local_url}/zones{GREEN}        │{RESET}")
    print(f"  {BOLD}{GREEN}  │   📊  {WHITE}Exec Summary: {CYAN}{local_url}/executive_summary{GREEN} │{RESET}")
    print(f"  {BOLD}{GREEN}  │                                                  │{RESET}")
    print(f"  {BOLD}{GREEN}  └──────────────────────────────────────────────────┘{RESET}")
    print()
    print(f"  {DIM}Press Ctrl+C to stop the server{RESET}")
    print(f"  {DIM}{'═' * 60}{RESET}\n")

    # ── Run Flask ────────────────────────────────────────────────────────────
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.WARNING)  # Suppress per-request logs for cleaner output

    app.run(host=HOST_IP, port=HOST_PORT, debug=False, use_reloader=False, threaded=True)
