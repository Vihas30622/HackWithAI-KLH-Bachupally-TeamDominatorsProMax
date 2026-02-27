/* ================================================================
   FaceVault — app.js
   API base: http://localhost:5000
   ================================================================ */

const API = "http://localhost:5000/api";

let allMembers = [];
let cameraActive = false;

// ── Init ──────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    initDropzone();
    initRegisterForm();
    loadStatus();
    loadMembers();

    setInterval(loadStatus, 5000);
    setInterval(() => {
        if (!document.getElementById("section-verify").classList.contains("hidden")) {
            loadLogs();
        }
    }, 4000);
});

// ── Tab switching ─────────────────────────────────────────────────
function switchTab(tab) {
    ["home", "register", "verify"].forEach(t => {
        document.getElementById(`section-${t}`).classList.toggle("hidden", t !== tab);
        document.getElementById(`section-${t}`).classList.toggle("active", t === tab);
        document.getElementById(`tab-${t}`).classList.toggle("active", t === tab);
    });
    if (tab === "verify") loadLogs();
}
window.switchTab = switchTab;

// ── Status ────────────────────────────────────────────────────────
async function loadStatus() {
    try {
        const res = await fetch(`${API}/status`);
        const data = await res.json();
        document.getElementById("status-dot").className = "status-dot online";
        document.getElementById("status-text").textContent = "API Online";
        document.getElementById("member-count").textContent = data.members_count ?? "–";

        // Sync camera UI if status changed externally
        if (data.camera_active !== cameraActive) {
            cameraActive = data.camera_active;
            syncCameraUI();
        }
    } catch {
        document.getElementById("status-dot").className = "status-dot offline";
        document.getElementById("status-text").textContent = "API Offline";
    }
}

// ── Members ───────────────────────────────────────────────────────
async function loadMembers() {
    try {
        const res = await fetch(`${API}/members`);
        allMembers = await res.json();
        renderMembers(allMembers);
    } catch {
        document.getElementById("members-list").innerHTML =
            `<div class="empty-state"><div class="empty-icon">⚠️</div><p>Couldn't load members. Is the API running?</p></div>`;
    }
}
window.loadMembers = loadMembers;

function filterMembers() {
    const q = document.getElementById("member-search").value.toLowerCase();
    renderMembers(allMembers.filter(m =>
        m.name.toLowerCase().includes(q) ||
        m.member_id.toLowerCase().includes(q) ||
        (m.membership_level || "").toLowerCase().includes(q)
    ));
}
window.filterMembers = filterMembers;

function renderMembers(members) {
    const list = document.getElementById("members-list");
    if (!members || members.length === 0) {
        list.innerHTML = `<div class="empty-state"><div class="empty-icon">👥</div><p>No members found.</p></div>`;
        return;
    }
    list.innerHTML = members.map(memberRowHTML).join("");
}

function memberRowHTML(m) {
    const level = m.membership_level || "Premium";
    const cls = `level-${level.toLowerCase()}`;
    const imgEl = m.has_image
        ? `<img class="member-avatar"
            src="${API}/member_image/${m.member_id}"
            alt="${escHtml(m.name)}"
            onerror="this.style.display='none';this.nextElementSibling.style.display='flex';" />
       <div class="member-avatar-placeholder" style="display:none">👤</div>`
        : `<div class="member-avatar-placeholder">👤</div>`;

    return `
    <div class="member-row" id="member-row-${m.member_id}">
      ${imgEl}
      <div class="member-info">
        <div class="member-name">${escHtml(m.name)}</div>
        <div class="member-meta">
          <span class="member-id">${escHtml(m.member_id)}</span>
          <span class="member-level-badge ${cls}">${escHtml(level)}</span>
        </div>
      </div>
      <button class="member-delete-btn"
              onclick="deleteMember('${m.member_id}', '${escHtml(m.name).replace(/'/g, "\\'")}')"
              title="Remove ${escHtml(m.name)}">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polyline points="3 6 5 6 21 6"/>
          <path d="M19 6l-1 14H6L5 6"/>
          <path d="M10 11v6M14 11v6M9 6V4h6v2"/>
        </svg>
      </button>
    </div>`;
}

async function deleteMember(memberId, name) {
    if (!confirm(`Remove "${name}" from the database? This cannot be undone.`)) return;

    const row = document.getElementById(`member-row-${memberId}`);
    if (row) { row.style.opacity = "0.4"; row.style.pointerEvents = "none"; }

    try {
        // NOTE: Using /api/delete/<id> — avoids Flask static-file route collision
        const res = await fetch(`${API}/delete/${memberId}`, { method: "DELETE" });
        const data = await res.json();

        if (data.success) {
            await loadMembers();
            await loadStatus();
        } else {
            alert(`Delete failed: ${data.error || "Unknown error"}`);
            if (row) { row.style.opacity = ""; row.style.pointerEvents = ""; }
        }
    } catch (err) {
        alert("Could not reach API. Is the Flask server running?");
        if (row) { row.style.opacity = ""; row.style.pointerEvents = ""; }
    }
}
window.deleteMember = deleteMember;

// ── Register form ─────────────────────────────────────────────────
function initRegisterForm() {
    document.getElementById("register-form").addEventListener("submit", async (e) => {
        e.preventDefault();
        const name = document.getElementById("member-name").value.trim();
        const level = document.getElementById("member-level").value;
        const file = document.getElementById("photo-input").files[0];
        const toast = document.getElementById("register-result");
        const btn = document.getElementById("register-btn");
        const btnTxt = document.getElementById("register-btn-text");

        if (!name) { showToast(toast, "error", "⚠️  Please enter a member name."); return; }
        if (!file) { showToast(toast, "error", "⚠️  Please select a photo."); return; }

        btn.disabled = true;
        btnTxt.innerHTML = `<span class="spinner"></span>&nbsp;Processing…`;
        hideToast(toast);

        try {
            const fd = new FormData();
            fd.append("name", name);
            fd.append("membership_level", level);
            fd.append("photo", file);

            const res = await fetch(`${API}/register`, { method: "POST", body: fd });
            const data = await res.json();

            if (data.success) {
                showToast(toast, "success", `✅ ${data.message}`);
                resetRegisterForm();
                await loadMembers();
                await loadStatus();
            } else {
                showToast(toast, "error", `❌ ${data.error}`);
            }
        } catch {
            showToast(toast, "error", "❌ Failed to connect to API. Is the Flask server running?");
        } finally {
            btn.disabled = false;
            btnTxt.textContent = "Enrol Member";
        }
    });
}

function resetRegisterForm() {
    document.getElementById("member-name").value = "";
    document.getElementById("member-level").value = "Premium";
    document.getElementById("photo-input").value = "";
    document.getElementById("preview-img").classList.add("hidden");
    document.getElementById("dropzone-inner").classList.remove("hidden");
}

// ── Dropzone ──────────────────────────────────────────────────────
function initDropzone() {
    const zone = document.getElementById("dropzone");
    const input = document.getElementById("photo-input");
    const preview = document.getElementById("preview-img");
    const inner = document.getElementById("dropzone-inner");

    const showPreview = (file) => {
        if (!file || !file.type.startsWith("image/")) return;
        const reader = new FileReader();
        reader.onload = () => {
            preview.src = reader.result;
            preview.classList.remove("hidden");
            inner.classList.add("hidden");
        };
        reader.readAsDataURL(file);
    };

    zone.addEventListener("click", () => input.click());
    input.addEventListener("change", () => { if (input.files[0]) showPreview(input.files[0]); });
    zone.addEventListener("dragover", e => { e.preventDefault(); zone.classList.add("drag-over"); });
    zone.addEventListener("dragleave", () => zone.classList.remove("drag-over"));
    zone.addEventListener("drop", e => {
        e.preventDefault(); zone.classList.remove("drag-over");
        const file = e.dataTransfer.files[0];
        if (file) {
            const dt = new DataTransfer(); dt.items.add(file); input.files = dt.files;
            showPreview(file);
        }
    });
}

// ── Camera ────────────────────────────────────────────────────────
async function startCamera() {
    const idx = parseInt(document.getElementById("cam-index").value) || 0;
    const btn = document.getElementById("start-btn");
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner"></span>&nbsp;Starting…`;

    try {
        const res = await fetch(`${API}/start_camera`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ camera_index: idx }),
        });
        const data = await res.json();

        if (data.success) {
            cameraActive = true;
            // Give camera thread 1.5 s to produce its first frame
            await new Promise(r => setTimeout(r, 1500));
            syncCameraUI();
        } else {
            alert(`Could not start camera: ${data.error}`);
        }
    } catch {
        alert("Failed to connect to API. Is the Flask server running?");
    } finally {
        btn.disabled = false;
        btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polygon points="10,8 16,12 10,16"/></svg> Start Camera`;
    }
}
window.startCamera = startCamera;

async function stopCamera() {
    const btn = document.getElementById("stop-btn");
    btn.disabled = true;
    try {
        await fetch(`${API}/stop_camera`, { method: "POST" });
        cameraActive = false;
        // Clear the stream src immediately to drop the connection
        const streamEl = document.getElementById("video-stream");
        streamEl.src = "";
        syncCameraUI();
    } catch {
        alert("Failed to stop camera.");
    } finally {
        btn.disabled = false;
        btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><rect x="9" y="9" width="6" height="6"/></svg> Stop Camera`;
    }
}
window.stopCamera = stopCamera;

function syncCameraUI() {
    const streamEl = document.getElementById("video-stream");
    const offlineEl = document.getElementById("camera-offline");
    const startBtn = document.getElementById("start-btn");
    const stopBtn = document.getElementById("stop-btn");
    const liveBadge = document.getElementById("camera-live-badge");

    if (cameraActive) {
        // Fresh URL with timestamp cache-bust so the browser doesn't reuse a stale connection
        streamEl.src = `${API}/video_feed?t=${Date.now()}`;
        streamEl.classList.remove("hidden");
        offlineEl.classList.add("hidden");
        startBtn.classList.add("hidden");
        stopBtn.classList.remove("hidden");
        liveBadge.classList.remove("hidden");
    } else {
        streamEl.src = "";
        streamEl.classList.add("hidden");
        offlineEl.classList.remove("hidden");
        startBtn.classList.remove("hidden");
        stopBtn.classList.add("hidden");
        liveBadge.classList.add("hidden");
    }
}

// ── Logs ──────────────────────────────────────────────────────────
async function clearLog() {
    try {
        const res = await fetch(`${API}/logs/clear`, { method: "POST" });
        const data = await res.json();
        if (data.success) {
            await loadLogs();
        } else {
            alert(`Could not clear log: ${data.error}`);
        }
    } catch {
        alert("Could not reach API.");
    }
}
window.clearLog = clearLog;

async function loadLogs() {
    try {
        const res = await fetch(`${API}/logs`);
        const rows = await res.json();
        const list = document.getElementById("log-list");
        if (!rows || rows.length === 0) {
            list.innerHTML = `<div class="empty-state"><div class="empty-icon">📋</div><p>No entries yet.</p></div>`;
            return;
        }
        list.innerHTML = rows.slice(0, 50).map(r => {
            const granted = r.Decision === "GRANTED";
            const time = r.Timestamp ? r.Timestamp.slice(11, 19) : "";
            return `<div class="log-row">
        <span class="log-decision ${granted ? "granted" : "denied"}">${r.Decision}</span>
        <span class="log-name">${escHtml(r.Name || "Unknown")}</span>
        <span class="log-time">${time}</span>
      </div>`;
        }).join("");
    } catch { /* silent — server may be busy with camera */ }
}
window.loadLogs = loadLogs;

// ── Helpers ───────────────────────────────────────────────────────
function showToast(el, type, msg) { el.className = `result-toast ${type}`; el.textContent = msg; }
function hideToast(el) { el.className = "result-toast hidden"; }
function escHtml(str) {
    return String(str)
        .replace(/&/g, "&amp;").replace(/</g, "&lt;")
        .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
