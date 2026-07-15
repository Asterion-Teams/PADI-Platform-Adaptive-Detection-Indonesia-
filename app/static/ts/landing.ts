// TypeScript script for PADI Landing Page

interface ParticleConfig {
    x: number;
    y: number;
    size: number;
    speedX: number;
    speedY: number;
    color: string;
    alpha: number;
}

class Particle {
    x: number;
    y: number;
    size: number;
    speedX: number;
    speedY: number;
    color: string;
    alpha: number;
    targetAlpha: number;

    constructor(x: number, y: number, isDark: boolean) {
        this.x = x;
        this.y = y;
        this.size = Math.random() * 5 + 2;
        this.speedX = Math.random() * 0.4 - 0.2;
        this.speedY = Math.random() * 0.4 - 0.2;
        this.alpha = Math.random() * 0.5 + 0.1;
        this.targetAlpha = this.alpha;
        
        // Use soft glowing theme-responsive colors (teal/blue shades)
        const hues = [190, 200, 210, 220];
        const hue = hues[Math.floor(Math.random() * hues.length)];
        const lightness = isDark ? 65 : 45;
        this.color = `hsla(${hue}, 85%, ${lightness}%, `;
    }

    update(mouseX: number, mouseY: number): void {
        this.x += this.speedX;
        this.y += this.speedY;

        // Mouse hover repulsion
        const dx = this.x - mouseX;
        const dy = this.y - mouseY;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 100) {
            const force = (100 - dist) / 100;
            this.x += (dx / dist) * force * 2;
            this.y += (dy / dist) * force * 2;
        }

        // Fading animations
        if (Math.abs(this.alpha - this.targetAlpha) < 0.01) {
            this.targetAlpha = Math.random() * 0.5 + 0.1;
        }
        this.alpha += (this.targetAlpha - this.alpha) * 0.02;
    }

    draw(ctx: CanvasRenderingContext2D): void {
        ctx.save();
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        ctx.fillStyle = this.color + this.alpha + ")";
        ctx.shadowBlur = 10;
        ctx.shadowColor = ctx.fillStyle;
        ctx.fill();
        ctx.restore();
    }
}

class ParticleSystem {
    private canvas: HTMLCanvasElement;
    private ctx: CanvasRenderingContext2D;
    private particles: Particle[] = [];
    private mouseX: number = -9999;
    private mouseY: number = -9999;
    private animationId: number | null = null;

    constructor(canvasId: string) {
        this.canvas = document.getElementById(canvasId) as HTMLCanvasElement;
        this.ctx = this.canvas.getContext("2d")!;
        this.init();
        this.resize();
        this.animate();

        window.addEventListener("resize", () => this.resize());
        window.addEventListener("mousemove", (e) => this.handleMouseMove(e));
        window.addEventListener("themechanged", () => this.handleThemeChange());
    }

    private init(): void {
        const isDark = document.documentElement.classList.contains("dark") || document.body.classList.contains("dark");
        const count = Math.min(100, Math.floor((window.innerWidth * window.innerHeight) / 15000));
        this.particles = [];
        for (let i = 0; i < count; i++) {
            this.particles.push(
                new Particle(Math.random() * this.canvas.width, Math.random() * this.canvas.height, isDark)
            );
        }
    }

    private resize(): void {
        this.canvas.width = this.canvas.parentElement?.clientWidth || window.innerWidth;
        this.canvas.height = this.canvas.parentElement?.clientHeight || window.innerHeight;
        this.init();
    }

    private handleMouseMove(e: MouseEvent): void {
        const rect = this.canvas.getBoundingClientRect();
        this.mouseX = e.clientX - rect.left;
        this.mouseY = e.clientY - rect.top;
    }

    private handleThemeChange(): void {
        this.init();
    }

    private animate(): void {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        const w = this.canvas.width;
        const h = this.canvas.height;

        this.particles.forEach((p) => {
            p.update(this.mouseX, this.mouseY);
            
            // Boundary wrap around
            if (p.x < 0) p.x = w;
            if (p.x > w) p.x = 0;
            if (p.y < 0) p.y = h;
            if (p.y > h) p.y = 0;

            p.draw(this.ctx);
        });

        this.animationId = requestAnimationFrame(() => this.animate());
    }
}

class StatsCounter {
    private counters: HTMLElement[] = [];
    private observer: IntersectionObserver;

    constructor() {
        this.counters = Array.from(document.querySelectorAll(".counter-value"));
        this.observer = new IntersectionObserver((entries) => {
            entries.forEach((e) => {
                if (e.isIntersecting) {
                    this.startCount(e.target as HTMLElement);
                    this.observer.unobserve(e.target);
                }
            });
        }, { threshold: 0.1 });

        this.counters.forEach((c) => this.observer.observe(c));
    }

    private startCount(el: HTMLElement): void {
        const rawTarget = el.dataset.target || "0";
        const isPercentage = rawTarget.endsWith("%");
        const isPlus = rawTarget.endsWith("+");
        const isMs = rawTarget.endsWith("ms");
        
        let target = parseFloat(rawTarget.replace(/[^\d\.]/g, ""));
        let count = 0;
        const duration = 1800; // 1.8s duration
        const startTime = performance.now();

        const update = (time: number) => {
            const progress = Math.min(1, (time - startTime) / duration);
            // Ease out quad
            const ease = progress * (2 - progress);
            count = ease * target;
            
            let displayVal = "";
            if (rawTarget.includes(".")) {
                displayVal = count.toFixed(1);
            } else {
                displayVal = Math.floor(count).toString();
            }

            if (isPercentage) displayVal += "%";
            if (isPlus) displayVal += "+";
            if (isMs) displayVal += " ms";

            el.textContent = displayVal;

            if (progress < 1) {
                requestAnimationFrame(update);
            } else {
                el.textContent = rawTarget;
            }
        };

        requestAnimationFrame(update);
    }
}

class FeatureTabs {
    private buttons: HTMLButtonElement[] = [];
    private contents: HTMLElement[] = [];

    constructor() {
        this.buttons = Array.from(document.querySelectorAll(".tab-btn"));
        this.contents = Array.from(document.querySelectorAll(".tab-content-item"));
        
        this.buttons.forEach((btn) => {
            btn.addEventListener("click", () => this.switchTab(btn));
        });
    }

    private switchTab(clickedBtn: HTMLButtonElement): void {
        const targetId = clickedBtn.dataset.tab;
        
        this.buttons.forEach((btn) => {
            btn.classList.toggle("active-tab", btn === clickedBtn);
            if (btn === clickedBtn) {
                btn.classList.add("bg-sky-600/10", "border-sky-500/30", "text-sky-400");
                btn.classList.remove("border-transparent", "text-slate-400");
            } else {
                btn.classList.remove("bg-sky-600/10", "border-sky-500/30", "text-sky-400");
                btn.classList.add("border-transparent", "text-slate-400");
            }
        });

        this.contents.forEach((content) => {
            const isTarget = content.id === targetId;
            if (isTarget) {
                content.classList.remove("hidden");
                content.classList.add("animate-fade-in");
            } else {
                content.classList.add("hidden");
                content.classList.remove("animate-fade-in");
            }
        });
    }
}

// Initialize on DOM ready
document.addEventListener("DOMContentLoaded", () => {
    if (document.getElementById("hero-particles")) {
        new ParticleSystem("hero-particles");
    }
    new StatsCounter();
    new FeatureTabs();
});
