# -*- coding: utf-8 -*-
path = r'c:\Avicena\traffict\competition\big-data-traffict-competitiom\app\templates\settings.html'

with open(path, 'a', encoding='utf-8') as f:
    f.write('''{% block content %}
<div class="flex-1 overflow-y-auto p-4 md:p-6 space-y-5">

    <!-- System Status GPU Bar -->
    <div class="gpu-bar fade-in">
        <div class="gpu-stat">
            <span class="status-dot on" id="sys-status-dot"></span>
            <span class="gpu-stat-value" id="sys-status-text" style="color:var(--text-primary);font-size:.82rem;">System Online</span>
        </div>
        <div class="gpu-bar-divider"></div>
        <div class="gpu-stat">
            <i class="fas fa-microchip" style="color:rgba(99,102,241,.7);font-size:.8rem;"></i>
            <div>
                <div class="gpu-stat-label">GPU</div>
                <div class="gpu-stat-value" id="sys-gpu">&#8212;</div>
            </div>
        </div>
        <div class="gpu-bar-divider"></div>
        <div class="gpu-stat">
            <i class="fas fa-brain" style="color:rgba(6,182,212,.7);font-size:.8rem;"></i>
            <div>
                <div class="gpu-stat-label">Model</div>
                <div class="gpu-stat-value" id="sys-model">&#8212;</div>
            </div>
        </div>
        <div class="gpu-bar-divider"></div>
        <div class="gpu-stat">
            <i class="fas fa-robot" style="color:rgba(139,92,246,.7);font-size:.8rem;"></i>
            <div>
                <div class="gpu-stat-label">Agents</div>
                <div class="gpu-stat-value" id="sys-agents">&#8212;</div>
            </div>
        </div>
    </div>

    <div class="settings-wrap">
        <aside>
            <nav class="settings-tabs" id="settings-tabs">
                <button class="settings-tab active" data-tab="tab-ai-model"><i class="fas fa-brain"></i> AI Model</button>
                <button class="settings-tab" data-tab="tab-detection"><i class="fas fa-crosshairs"></i> Detection</button>
                <button class="settings-tab" data-tab="tab-enforcement"><i class="fas fa-shield-halved"></i> Enforcement</button>
                <button class="settings-tab" data-tab="tab-social"><i class="fas fa-newspaper"></i> Social Media</button>
                <button class="settings-tab" data-tab="tab-ai-provider"><i class="fas fa-robot"></i> AI Provider</button>
                <button class="settings-tab" data-tab="tab-hikvision"><i class="fas fa-camera-cctv"></i> Hikvision</button>
                <button class="settings-tab" data-tab="tab-agent-tools"><i class="fas fa-screwdriver-wrench"></i> Agent Tools</button>
            </nav>
            <div class="sidebar-actions">
                <button id="btn-save" class="flex items-center justify-center gap-2 px-4 py-2 btn-primary rounded-md text-sm font-medium transition-all">
                    <i class="fas fa-save text-xs"></i>
                    <span>Simpan</span>
                </button>
                <button id="btn-restart" class="flex items-center justify-center gap-2 px-3 py-2 btn-secondary rounded-md text-sm font-medium transition-all">
                    <i class="fas fa-rotate text-xs"></i>
                    <span>Restart Agents</span>
                </button>
            </div>
        </aside>

        <div class="space-y-5">

            <div class="tab-panel active" id="tab-ai-model">
                <div class="setting-card">
                    <h3 class="section-title"><i class="fas fa-brain"></i> AI Model <span class="subtitle" id="current-model-label">&#8212;</span></h3>
                    <div class="model-grid" id="model-grid"></div>
                    <p class="setting-desc mt-3">Klik model untuk switch. Hot-swap tanpa restart server. Hover untuk hapus.</p>
                </div>
            </div>

            <div class="tab-panel" id="tab-detection">
                <div class="setting-card">
                    <h3 class="section-title"><i class="fas fa-crosshairs"></i> Deteksi</h3>
                    <div class="space-y-4">
                        <div>
                            <label class="setting-label">Confidence Threshold</label>
                            <input type="range" min="0.05" max="0.95" step="0.05" id="s-conf" class="w-full accent-sky-500">
                            <div class="flex justify-between text-xs" style="color: var(--text-tertiary);"><span>0.05</span><span id="v-conf">&#8212;</span><span>0.95</span></div>
                            <p class="setting-desc">Minimum confidence untuk deteksi kendaraan.</p>
                        </div>
                        <div>
                            <label class="setting-label">IoU Threshold (NMS)</label>
                            <input type="range" min="0.1" max="0.9" step="0.05" id="s-iou" class="w-full accent-sky-500">
                            <div class="flex justify-between text-xs" style="color: var(--text-tertiary);"><span>0.1</span><span id="v-iou">&#8212;</span><span>0.9</span></div>
                        </div>
                        <div>
                            <label class="setting-label">Inference Image Size</label>
                            <select id="s-imgsz" class="input-field">
                                <option value="320">320px (fastest)</option>
                                <option value="480">480px</option>
                                <option value="640">640px (recommended)</option>
                                <option value="960">960px</option>
                                <option value="1280">1280px (highest)</option>
                            </select>
                        </div>
                        <div class="grid grid-cols-2 gap-3">
                            <div>
                                <label class="setting-label">Stream FPS</label>
                                <input type="number" min="1" max="60" id="s-fps" class="input-field">
                            </div>
                            <div>
                                <label class="setting-label">JPEG Quality</label>
                                <input type="number" min="30" max="95" id="s-jpeg" class="input-field">
                            </div>
                        </div>
                        <div>
                            <label class="setting-label">Timezone</label>
                            <select id="s-timezone" class="input-field">
                                <option value="Asia/Jakarta">WIB (Asia/Jakarta)</option>
                                <option value="Asia/Makassar">WITA (Asia/Makassar)</option>
                                <option value="Asia/Jayapura">WIT (Asia/Jayapura)</option>
                                <option value="UTC">UTC</option>
                            </select>
                        </div>
                    </div>
                </div>
            </div>

            <div class="tab-panel" id="tab-enforcement">
                <div class="setting-card">
                    <h3 class="section-title"><i class="fas fa-shield-halved"></i> Enforcement (E-TLE)</h3>
                    <div class="space-y-4">
                        <div class="flex items-center justify-between">
                            <div><div class="text-sm" style="color: var(--text-primary);">Violation Detection</div><p class="setting-desc">Master switch enforcement engine</p></div>
                            <div id="t-violations" class="toggle on"></div>
                        </div>
                        <div class="flex items-center justify-between">
                            <div><div class="text-sm" style="color: var(--text-primary);">ANPR (Plate Recognition)</div><p class="setting-desc">Baca plat nomor pelanggar</p></div>
                            <div id="t-anpr" class="toggle on"></div>
                        </div>
                        <hr style="border-color: var(--border-color);">
                        <div>
                            <label class="setting-label">Parking Threshold (detik)</label>
                            <input type="number" min="10" max="300" step="5" id="s-parking-sec" class="input-field">
                            <p class="setting-desc">Kendaraan harus diam selama ini sebelum dianggap pelanggaran.</p>
                        </div>
                        <div>
                            <label class="setting-label">Dynamic Lane Threshold (detik)</label>
                            <input type="number" min="1" max="30" step="1" id="s-lane-sec" class="input-field">
                            <p class="setting-desc">Threshold untuk busway/jalur sepeda.</p>
                        </div>
                        <div>
                            <label class="setting-label">Violation Cooldown (detik)</label>
                            <input type="number" min="10" max="600" step="10" id="s-cooldown" class="input-field">
                            <p class="setting-desc">Jeda sebelum kendaraan sama bisa dicatat ulang.</p>
                        </div>
                    </div>
                </div>
            </div>

            <div class="tab-panel" id="tab-social">
                <div class="setting-card">
                    <h3 class="section-title"><i class="fas fa-newspaper"></i> Monitor Berita Lalu Lintas</h3>
                    <div class="space-y-3">
                        <div>
                            <label class="setting-label">Keyword Pencarian Berita</label>
                            <input id="s-news-query" class="input-field" placeholder="macet OR kemacetan OR lalu lintas jakarta">
                            <p class="setting-desc">Keyword untuk mencari berita terkait lalu lintas. Gunakan OR untuk multiple terms.</p>
                        </div>
                        <button id="btn-test-news" class="btn-primary px-4 py-2 text-sm font-medium rounded-md">
                            <i class="fas fa-search mr-1"></i> Test Pencarian
                        </button>
                        <div id="news-preview" class="mt-3 rounded-lg border p-3 hidden max-h-[250px] overflow-y-auto" style="border-color: var(--border-color);">
                            <p class="text-xs" style="color: var(--text-tertiary);">Loading...</p>
                        </div>
                    </div>
                </div>
                <div class="setting-card mt-5">
                    <h3 class="section-title"><i class="fab fa-x-twitter text-slate-300"></i> CRM &#8212; X.com Social Media Scraper</h3>
                    <div class="space-y-3">
                        <div>
                            <label class="setting-label">Search Query (Hashtag / Mention)</label>
                            <input id="s-x-query" class="input-field" placeholder="@DishubDKI OR #DishubDKI OR to:DishubDKI">
                            <p class="setting-desc">Query pencarian di X.com. Gunakan OR untuk multiple terms.</p>
                        </div>
                        <div>
                            <label class="setting-label">X.com Session Cookies (JSON)</label>
                            <textarea id="s-x-cookies" rows="4" class="input-field font-mono text-xs" placeholder="[{&quot;name&quot;:&quot;auth_token&quot;,&quot;value&quot;:&quot;xxx&quot;,&quot;domain&quot;:&quot;.x.com&quot;,...}]"></textarea>
                            <p class="setting-desc">Export cookies dari browser (EditThisCookie).</p>
                        </div>
                        <div class="flex items-center gap-3">
                            <span id="x-cookies-status" class="text-xs" style="color: var(--text-tertiary);">&#8212;</span>
                            <button id="btn-test-x" class="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white text-sm font-semibold rounded-lg">
                                <i class="fab fa-x-twitter mr-1"></i> Test Scrape
                            </button>
                        </div>
                        <div id="x-preview" class="mt-2 rounded-lg border p-3 hidden max-h-[200px] overflow-y-auto" style="border-color: var(--border-color);">
                            <p class="text-xs" style="color: var(--text-tertiary);">Loading...</p>
                        </div>
                    </div>
                </div>
            </div>

            <div class="tab-panel" id="tab-ai-provider">
                <div class="setting-card">
                    <h3 class="section-title"><i class="fas fa-robot"></i> AI Mode</h3>
                    <div class="flex items-center gap-3 mb-4">
                        <div class="flex items-center gap-2">
                            <div id="t-ai-mode" class="toggle on" onclick="toggleAiMode()"></div>
                            <div>
                                <div class="text-sm font-medium" style="color: var(--text-primary);" id="ai-mode-label">Online AI</div>
                                <div class="text-[11px]" style="color: var(--text-tertiary);">Aktif &#8212; klik untuk switch</div>
                            </div>
                        </div>
                        <div id="ai-mode-status" class="ml-auto text-xs px-2.5 py-1 rounded-full font-medium bg-emerald-50 dark:bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border border-emerald-200 dark:border-emerald-500/20">
                            <i class="fas fa-globe mr-1"></i>Online
                        </div>
                    </div>
                    <div id="ai-online-section">
                        <div class="space-y-3">
                            <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
                                <div>
                                    <label class="setting-label">Provider</label>
                                    <select id="s-ai-provider" class="input-field">
                                        <option value="sumopod">SumoPod AI</option>
                                        <option value="openai">OpenAI</option>
                                        <option value="custom">Custom (OpenAI-compatible)</option>
                                    </select>
                                </div>
                                <div>
                                    <label class="setting-label">Model</label>
                                    <input id="s-ai-model" class="input-field" placeholder="gpt-4o-mini">
                                </div>
                            </div>
                            <div>
                                <label class="setting-label">Base URL</label>
                                <input id="s-ai-baseurl" class="input-field" placeholder="https://ai.sumopod.com/v1">
                            </div>
                            <div>
                                <label class="setting-label">API Key</label>
                                <input id="s-ai-apikey" type="password" class="input-field" placeholder="sk-xxxxx">
                            </div>
                            <div>
                                <button id="btn-test-ai" class="btn-primary px-4 py-2 text-sm font-medium rounded-md">
                                    <i class="fas fa-plug mr-1"></i> Test Koneksi
                                </button>
                                <span id="ai-online-status" class="ml-3 text-xs" style="color: var(--text-tertiary);"></span>
                            </div>
                        </div>
                    </div>
                    <div id="ai-local-section" style="display:none;">
                        <div class="space-y-3">
                            <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
                                <div>
                                    <label class="setting-label">Base URL</label>
                                    <input id="s-ollama-url" class="input-field" placeholder="http://localhost:11434">
                                </div>
                                <div>
                                    <label class="setting-label">Model</label>
                                    <div class="flex gap-2">
                                        <select id="s-ollama-model" class="input-field flex-1">
                                            <option value="">&#8212; pilih model &#8212;</option>
                                        </select>
                                        <button onclick="refreshOllamaModels()" class="px-2.5 py-2 btn-secondary rounded-md text-xs" title="Refresh daftar model">
                                            <i class="fas fa-rotate"></i>
                                        </button>
                                    </div>
                                </div>
                            </div>
                            <div class="p-3 rounded-lg" style="background: var(--input-surface); border: 1px solid var(--border-color);">
                                <div class="text-xs font-medium mb-2" style="color: var(--text-primary);">
                                    <i class="fas fa-microchip mr-1"></i> Model yang tersedia
                                </div>
                                <div id="ollama-models-list" class="flex flex-wrap gap-1.5">
                                    <span class="text-xs" style="color: var(--text-tertiary);">Memuat...</span>
                                </div>
                            </div>
                            <div>
                                <button id="btn-test-ollama" class="btn-primary px-4 py-2 text-sm font-medium rounded-md">
                                    <i class="fas fa-plug mr-1"></i> Test Koneksi
                                </button>
                                <span id="ai-local-status" class="ml-3 text-xs" style="color: var(--text-tertiary);"></span>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="setting-card" style="margin-top:1rem;">
                    <h3 class="section-title"><i class="fas fa-sliders"></i> Fitur AI</h3>
                    <div class="space-y-3">
                        <div class="flex items-center justify-between">
                            <div>
                                <div class="text-sm font-medium" style="color: var(--text-primary);">AI untuk koreksi ANPR</div>
                                <p class="setting-desc">PaddleOCR + AI model untuk akurasi plat nomor lebih tinggi</p>
                            </div>
                            <div id="t-ai-anpr" class="toggle on"></div>
                        </div>
                        <div class="flex items-center justify-between">
                            <div>
                                <div class="text-sm font-medium" style="color: var(--text-primary);">AI untuk PADI Chat Assistant</div>
                                <p class="setting-desc">Chatbot menggunakan AI provider</p>
                            </div>
                            <div id="t-ai-chat" class="toggle on"></div>
                        </div>
                        <div>
                            <label class="setting-label">Chat Model</label>
                            <input id="s-ai-chat-model" class="input-field" placeholder="gpt-4o-mini">
                            <p class="setting-desc">Model untuk PADI Assistant. Bisa berbeda dari model ANPR.</p>
                        </div>
                        <div class="flex items-center gap-3">
                            <button id="btn-test-ai-2" class="btn-primary px-4 py-2 text-sm font-medium rounded-md">
                                <i class="fas fa-plug mr-1"></i> Test Koneksi
                            </button>
                            <span id="ai-status" class="text-xs" style="color: var(--text-tertiary);">&#8212;</span>
                        </div>
                    </div>
                </div>
            </div>

            <div class="tab-panel" id="tab-hikvision">
                <div class="setting-card">
                    <h3 class="section-title"><i class="fas fa-camera-cctv"></i> Hikvision ISAPI Integration</h3>
                    <p class="setting-desc mb-4">Koneksi langsung ke kamera Hikvision via ISAPI untuk snapshot resolusi tinggi dan informasi perangkat.</p>
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div class="space-y-3">
                            <div>
                                <label class="setting-label">IP Address / Hostname</label>
                                <input id="hik-host" class="input-field" placeholder="192.168.1.100">
                            </div>
                            <div class="grid grid-cols-2 gap-3">
                                <div>
                                    <label class="setting-label">Username</label>
                                    <input id="hik-user" class="input-field" value="admin">
                                </div>
                                <div>
                                    <label class="setting-label">Password</label>
                                    <input id="hik-pass" type="password" class="input-field" placeholder="&#8226;&#8226;&#8226;&#8226;&#8226;&#8226;">
                                </div>
                            </div>
                            <div class="grid grid-cols-2 gap-3">
                                <div>
                                    <label class="setting-label">Port</label>
                                    <input id="hik-port" type="number" class="input-field" value="80">
                                </div>
                                <div>
                                    <label class="setting-label">Channel</label>
                                    <input id="hik-channel" type="number" class="input-field" value="1" min="1" max="32">
                                </div>
                            </div>
                            <div class="flex flex-wrap gap-2 pt-2">
                                <button id="btn-hik-test" class="btn-secondary px-4 py-2 text-sm font-medium rounded-md">
                                    <i class="fas fa-plug mr-1"></i> Test Koneksi
                                </button>
                                <button id="btn-hik-snapshot" class="btn-primary px-4 py-2 text-sm font-medium rounded-md">
                                    <i class="fas fa-camera mr-1"></i> Snapshot
                                </button>
                                <button id="btn-hik-anpr" class="btn-secondary px-4 py-2 text-sm font-medium rounded-md">
                                    <i class="fas fa-id-card mr-1"></i> ANPR
                                </button>
                                <button id="btn-hik-add" class="btn-primary px-4 py-2 text-sm font-medium rounded-md">
                                    <i class="fas fa-plus mr-1"></i> Tambah
                                </button>
                            </div>
                        </div>
                        <div>
                            <div id="hik-result" class="rounded-lg border p-4 min-h-[200px] flex items-center justify-center" style="border-color: var(--border-color);">
                                <p class="text-xs text-center" style="color: var(--text-tertiary);"><i class="fas fa-camera-cctv text-2xl block mb-2" style="color: var(--text-tertiary);"></i>Klik &quot;Test Koneksi&quot; untuk memulai</p>
                            </div>
                        </div>
                    </div>
                    <div id="hik-snapshot-area" class="mt-4 hidden">
                        <h4 class="text-sm font-medium mb-2" style="color: var(--text-primary);"><i class="fas fa-image mr-1"></i> Snapshot Preview (Full Resolution)</h4>
                        <div class="rounded-lg border overflow-hidden bg-black" style="border-color: var(--border-color);">
                            <img id="hik-snapshot-img" class="w-full max-h-[400px] object-contain" />
                        </div>
                        <p class="setting-desc mt-1">Resolusi penuh dari ISAPI &#8212; lebih baik untuk ANPR dibanding RTSP stream.</p>
                    </div>
                </div>
            </div>

            <div class="tab-panel" id="tab-agent-tools">
                <div class="setting-card">
                    <h3 class="section-title"><i class="fas fa-screwdriver-wrench"></i> Agent Tools <span class="ml-auto text-xs font-normal" style="color: var(--text-tertiary);">scripts di folder <code style="color: var(--text-tertiary);">/tools</code></span></h3>
                    <p class="setting-desc mb-3">Jalankan skrip operasional &amp; maintenance langsung dari UI. Tool <span class="font-semibold" style="color: var(--text-primary);">maintenance</span> memodifikasi data &#8212; gunakan <strong>Dry Run</strong> untuk preview.</p>
                    <div class="mb-4 p-3 rounded-lg border" style="border-color: var(--border-color); background: var(--input-surface);">
                        <div class="flex items-center justify-between flex-wrap gap-3">
                            <div class="flex items-center gap-3">
                                <i class="fas fa-brain"></i>
                                <div>
                                    <div class="text-sm font-semibold" style="color: var(--text-primary);">Vehicle Identity Agent</div>
                                    <div class="text-[11px]" style="color: var(--text-secondary);">Mengisi plat, merek, warna kendaraan via AI vision. <strong style="color: var(--text-primary);">Pause untuk stop penggunaan token AI.</strong></div>
                                </div>
                            </div>
                            <div class="flex items-center gap-3">
                                <span id="via-status-badge" class="text-xs px-2 py-1 rounded-full font-medium bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20">
                                    <i class="fas fa-spinner fa-spin mr-1"></i> Loading...
                                </span>
                                <button id="btn-via-toggle" class="btn-secondary px-4 py-1.5 text-xs font-medium rounded-md transition-all flex items-center gap-1.5">
                                    <i class="fas fa-pause"></i>
                                    <span>Pause</span>
                                </button>
                            </div>
                        </div>
                    </div>
                    <div class="flex flex-wrap gap-2 mb-4" id="agent-tools-filters">
                        <button data-cat="all" class="at-filter at-filter-active px-3 py-1 rounded-full text-xs font-medium">Semua</button>
                        <button data-cat="diagnostic" class="at-filter px-3 py-1 rounded-full text-xs font-medium">Diagnostic</button>
                        <button data-cat="maintenance" class="at-filter px-3 py-1 rounded-full text-xs font-medium">Maintenance</button>
                        <button data-cat="test" class="at-filter px-3 py-1 rounded-full text-xs font-medium">Test</button>
                    </div>
                    <div id="agent-tools-grid" class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 mb-4"></div>
                    <div id="agent-tools-output-wrap" class="hidden">
                        <div class="flex items-center justify-between mb-2">
                            <h4 class="text-sm font-semibold" style="color: var(--text-primary);"><i class="fas fa-terminal text-emerald-600 dark:text-emerald-400 mr-1"></i> Output <span id="at-output-tool" class="font-normal" style="color: var(--text-secondary);"></span></h4>
                            <div class="flex items-center gap-2">
                                <span id="at-output-status" class="text-xs" style="color: var(--text-tertiary);"></span>
                                <button id="at-clear-output" class="text-xs hover:text-white" style="color: var(--text-secondary);"><i class="fas fa-xmark"></i></button>
                            </div>
                        </div>
                        <div class="rounded-lg border bg-black/80 overflow-hidden shadow-inner" style="border-color: var(--border-color);">
                            <pre id="agent-tools-output" class="p-4 text-xs font-mono max-h-[360px] overflow-auto whitespace-pre-wrap break-all text-emerald-400 font-medium tracking-wide"></pre>
                        </div>
                    </div>
                </div>
            </div>

        </div>
    </div>
</div>

<!-- Toast -->
<div class="toast-bar" id="toast-bar"></div>
<!-- Confirm Delete Modal -->
<div class="confirm-overlay" id="confirm-overlay">
    <div class="confirm-dialog">
        <h3><i class="fas fa-exclamation-triangle mr-2" style="color:#f87171"></i>Hapus Model?</h3>
        <p id="confirm-msg"></p>
        <div class="confirm-actions">
            <button onclick="closeConfirm()" class="btn-secondary px-4 py-2 text-sm font-semibold rounded-lg transition-all">Batal</button>
            <button onclick="doDelete()" id="confirm-delete-btn" class="btn-danger px-4 py-2 text-sm font-semibold rounded-lg transition-all flex items-center gap-2"><i class="fas fa-trash-alt text-[10px]"></i>Hapus</button>
        </div>
    </div>
</div>
''')
print("Part 2 (HTML content + modal) written")
