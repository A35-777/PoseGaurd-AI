/**
 * PoseGuard AI — Application Controller
 * Handles file ingestion, hybrid defect analysis, 3D viewport playback, and reports.
 */

let viewer = null;
let currentSessionId = null;
let currentAnalysis = null;
let isPlaying = false;
let playbackTimer = null;
let playbackSpeed = 1.0;
let defectFramesSet = new Set();

document.addEventListener("DOMContentLoaded", async () => {
    viewer = new SkeletonViewer3D("viewport-canvas");
    await fetchSystemInfo();
    setupEventListeners();
    setupDropzone();
    // Auto-load clean walk demo by default
    loadDemo("walk_clean");
});

async function fetchSystemInfo() {
    try {
        const res = await fetch("/api/info");
        const data = await res.json();
        document.getElementById("badge-model").textContent = `${data.model_version} • 214K params (855 KB) • CPU`;
        document.getElementById("badge-ram").textContent = `RAM: ${data.ram_available_gb} GB Free (${data.ram_percent}%)`;
    } catch (e) {
        console.warn("Could not fetch system info:", e);
    }
}

function setupEventListeners() {
    // Playback buttons
    const playBtn = document.getElementById("btn-play");
    playBtn.addEventListener("click", togglePlay);
    
    document.getElementById("btn-prev").addEventListener("click", () => stepFrame(-1));
    document.getElementById("btn-next").addEventListener("click", () => stepFrame(1));
    
    const scrubber = document.getElementById("timeline-scrubber");
    scrubber.addEventListener("input", (e) => {
        const f = parseInt(e.target.value);
        goToFrame(f);
    });
    
    // Analyze button
    document.getElementById("btn-analyze").addEventListener("click", runAnalysis);
    
    // Speed selector
    const speedSelect = document.getElementById("select-speed");
    if (speedSelect) {
        speedSelect.addEventListener("change", (e) => {
            playbackSpeed = parseFloat(e.target.value);
            if (isPlaying) {
                stopPlayback();
                startPlayback();
            }
        });
    }
    
    // Copy findings button
    document.getElementById("btn-copy-findings").addEventListener("click", copyFindingsToClipboard);
}

function setupDropzone() {
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("file-input");
    
    dropzone.addEventListener("click", () => fileInput.click());
    
    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
    });
    
    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("dragover");
    });
    
    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });
    
    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });
}

async function handleFileUpload(file) {
    const formData = new FormData();
    formData.append("file", file);
    
    showLoading("Importing and parsing skeleton...");
    try {
        const res = await fetch("/api/upload", {
            method: "POST",
            body: formData
        });
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "Upload failed");
        }
        const data = await res.json();
        onAnimationLoaded(data);
    } catch (e) {
        alert("Import Error: " + e.message);
    } finally {
        hideLoading();
    }
}

async function loadDemo(name) {
    showLoading(`Loading demo clip: ${name}...`);
    try {
        const res = await fetch(`/api/demo/${name}`);
        if (!res.ok) throw new Error("Could not load demo animation");
        const data = await res.json();
        onAnimationLoaded(data);
    } catch (e) {
        console.error("Demo load failed:", e);
    } finally {
        hideLoading();
    }
}

function onAnimationLoaded(data) {
    currentSessionId = data.session_id;
    currentAnalysis = null;
    defectFramesSet.clear();
    
    // Update metadata preview
    document.getElementById("meta-filename").textContent = data.filename;
    document.getElementById("meta-frames").textContent = data.num_frames;
    document.getElementById("meta-fps").textContent = `${data.fps.toFixed(1)} fps`;
    document.getElementById("meta-duration").textContent = `${data.duration_seconds.toFixed(2)}s`;
    document.getElementById("meta-joints").textContent = data.num_joints;
    
    const groundInput = document.getElementById("input-ground-y");
    groundInput.value = data.estimated_ground_y.toFixed(3);
    
    // Configure Scrubber
    const scrubber = document.getElementById("timeline-scrubber");
    scrubber.min = 0;
    scrubber.max = data.num_frames - 1;
    scrubber.value = 0;
    document.getElementById("frame-total").textContent = data.num_frames - 1;
    
    // Clear defect track & findings
    document.getElementById("timeline-defects").innerHTML = "";
    document.getElementById("findings-list").innerHTML = `
        <div style="text-align: center; color: var(--text-muted); padding: 40px 10px;">
            Ready to analyze. Click <strong>"Run Quality Analysis"</strong> below.
        </div>
    `;
    document.getElementById("status-badge").className = "badge-status";
    document.getElementById("status-badge").textContent = "READY";
    document.getElementById("btn-export-json").href = "#";
    document.getElementById("btn-export-csv").href = "#";
    
    // Load into 3D Viewport
    viewer.loadSkeleton(data.joint_names, data.parents, data.joint_positions, data.fps, data.estimated_ground_y);
    goToFrame(0);
    startPlayback();
}

async function runAnalysis() {
    if (!currentSessionId) {
        alert("Please load or import an animation file first.");
        return;
    }
    
    const groundY = parseFloat(document.getElementById("input-ground-y").value);
    
    // Gather enabled types
    const enabledTypes = [];
    if (document.getElementById("chk-sliding").checked) enabledTypes.push("foot_sliding");
    if (document.getElementById("chk-penetration").checked) enabledTypes.push("ground_penetration");
    if (document.getElementById("chk-transition").checked) enabledTypes.push("unnatural_pose_transition");
    if (document.getElementById("chk-contact").checked) enabledTypes.push("contact_inconsistency");
    if (document.getElementById("chk-stretch").checked) enabledTypes.push("bone_stretch_anomaly");
    
    showLoading("Running Hybrid AI & Geometric Diagnosis...");
    try {
        const res = await fetch("/api/analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                session_id: currentSessionId,
                ground_y: groundY,
                enabled_types: enabledTypes
            })
        });
        
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "Analysis failed");
        }
        
        const result = await res.json();
        currentAnalysis = result;
        renderAnalysisResults(result);
    } catch (e) {
        alert("Analysis Error: " + e.message);
    } finally {
        hideLoading();
    }
}

function renderAnalysisResults(result) {
    const summary = result.quality_summary;
    const badge = document.getElementById("status-badge");
    badge.textContent = summary.overall_status.replace("_", " ").toUpperCase();
    badge.className = `badge-status status-${summary.overall_status}`;
    
    // Set Export Links
    document.getElementById("btn-export-json").href = `/api/export/json/${currentSessionId}`;
    document.getElementById("btn-export-csv").href = `/api/export/csv/${currentSessionId}`;
    
    // Render Timeline defect markers
    const track = document.getElementById("timeline-defects");
    track.innerHTML = "";
    defectFramesSet.clear();
    
    const totalFrames = result.num_frames;
    
    result.defects.forEach(defect => {
        const leftPct = (defect.start_frame / totalFrames) * 100;
        const widthPct = Math.max(0.6, ((defect.end_frame - defect.start_frame + 1) / totalFrames) * 100);
        
        const marker = document.createElement("div");
        marker.className = `defect-marker marker-${defect.type}`;
        marker.style.left = `${leftPct}%`;
        marker.style.width = `${widthPct}%`;
        marker.title = `${defect.type}: frames ${defect.start_frame}-${defect.end_frame}`;
        marker.addEventListener("click", () => {
            goToFrame(defect.start_frame);
            stopPlayback();
        });
        track.appendChild(marker);
        
        for (let f = defect.start_frame; f <= defect.end_frame; f++) {
            defectFramesSet.add(f);
        }
    });
    
    // Render Findings List
    const list = document.getElementById("findings-list");
    list.innerHTML = "";
    
    if (result.defects.length === 0 && result.uncertain_findings.length === 0) {
        list.innerHTML = `
            <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid #10b981; border-radius: 6px; padding: 16px; text-align: center;">
                <div style="color: #10b981; font-weight: 700; margin-bottom: 4px;">✓ PRISTINE KINEMATICS</div>
                <div style="font-size: 12px; color: var(--text-secondary);">No motion defects detected. 100% of frames conform to biomechanical and physical constraints.</div>
            </div>
        `;
        return;
    }
    
    // Summary row
    const sumRow = document.createElement("div");
    sumRow.style.cssText = "display: flex; justify-content: space-between; font-size: 11px; color: var(--text-muted); margin-bottom: 8px;";
    sumRow.innerHTML = `
        <span>Defects: <strong>${result.defects.length}</strong></span>
        <span>Clean Frames: <strong>${summary.defect_free_frame_percentage}%</strong></span>
    `;
    list.appendChild(sumRow);
    
    result.defects.forEach((d, idx) => {
        const card = document.createElement("div");
        card.className = `defect-card sev-${d.severity}`;
        
        // Geometric Evidence Formatter
        let evidenceStr = "";
        if (d.geometric_evidence.max_slide_speed_mps) {
            evidenceStr = `Speed: ${(d.geometric_evidence.max_slide_speed_mps * 100).toFixed(1)} cm/s | Dist: ${(d.geometric_evidence.total_slide_distance_m * 100).toFixed(1)} cm`;
        } else if (d.geometric_evidence.max_penetration_depth_m) {
            evidenceStr = `Depth: ${(d.geometric_evidence.max_penetration_depth_m * 100).toFixed(1)} cm below floor`;
        } else if (d.geometric_evidence.peak_acceleration_mps2) {
            evidenceStr = `Accel: ${d.geometric_evidence.peak_acceleration_mps2.toFixed(1)} m/s²`;
        } else if (d.geometric_evidence.max_stretch_percent) {
            evidenceStr = `Stretch: +${d.geometric_evidence.max_stretch_percent.toFixed(1)}%`;
        } else if (d.geometric_evidence.contact_state_toggles) {
            evidenceStr = `Chatter: ${d.geometric_evidence.contact_state_toggles} state flips`;
        }
        
        card.innerHTML = `
            <div class="card-top">
                <span class="defect-name">${d.type.replace(/_/g, " ")}</span>
                <span class="defect-frames">F${d.start_frame}–F${d.end_frame}</span>
            </div>
            <div class="defect-meta-row">
                <span>Joint: <strong>${d.joint}</strong></span>
                <span>Sev: <strong>${d.severity.toUpperCase()}</strong></span>
                <span>Conf: <strong>${(d.confidence * 100).toFixed(0)}%</strong></span>
            </div>
            ${evidenceStr ? `<div style="font-size: 11px; color: var(--accent-cyan); font-family: var(--font-mono); margin-bottom: 6px;">${evidenceStr}</div>` : ""}
            <div class="defect-suggestion">${d.suggested_correction}</div>
        `;
        
        card.addEventListener("click", () => {
            goToFrame(d.start_frame);
            stopPlayback();
        });
        
        list.appendChild(card);
    });
    
    // Uncertain / Borderline findings
    if (result.uncertain_findings.length > 0) {
        const uncertHeader = document.createElement("div");
        uncertHeader.style.cssText = "font-size: 11px; text-transform: uppercase; color: var(--text-muted); font-weight: 700; margin: 12px 0 6px 0;";
        uncertHeader.textContent = "Uncertain / Borderline Patterns";
        list.appendChild(uncertHeader);
        
        result.uncertain_findings.forEach(u => {
            const uCard = document.createElement("div");
            uCard.className = "defect-card sev-low";
            uCard.style.borderLeftColor = "#94a3b8";
            uCard.innerHTML = `
                <div class="card-top">
                    <span class="defect-name">${u.type.replace(/_/g, " ")} (Borderline)</span>
                    <span class="defect-frames">F${u.start_frame}–F${u.end_frame}</span>
                </div>
                <div class="defect-meta-row">
                    <span>Neural Conf: <strong>${(u.confidence * 100).toFixed(0)}%</strong></span>
                </div>
                <div class="defect-suggestion" style="border-left-color: #94a3b8;">${u.suggested_correction}</div>
            `;
            uCard.addEventListener("click", () => {
                goToFrame(u.start_frame);
                stopPlayback();
            });
            list.appendChild(uCard);
        });
    }
}

function goToFrame(frameIdx) {
    if (!viewer) return;
    const isDefect = defectFramesSet.has(frameIdx);
    viewer.updateFrame(frameIdx, isDefect);
    document.getElementById("timeline-scrubber").value = frameIdx;
    document.getElementById("frame-current").textContent = frameIdx;
}

function stepFrame(delta) {
    const scrubber = document.getElementById("timeline-scrubber");
    const current = parseInt(scrubber.value);
    const max = parseInt(scrubber.max);
    const next = Math.max(0, Math.min(current + delta, max));
    goToFrame(next);
}

function togglePlay() {
    if (isPlaying) {
        stopPlayback();
    } else {
        startPlayback();
    }
}

function startPlayback() {
    if (isPlaying) return;
    isPlaying = true;
    document.getElementById("btn-play").innerHTML = "❚❚";
    
    const scrubber = document.getElementById("timeline-scrubber");
    const max = parseInt(scrubber.max);
    const intervalMs = (1000 / (viewer.fps || 30)) / playbackSpeed;
    
    playbackTimer = setInterval(() => {
        let current = parseInt(scrubber.value);
        let next = current + 1;
        if (next > max) next = 0; // Loop
        goToFrame(next);
    }, intervalMs);
}

function stopPlayback() {
    isPlaying = false;
    document.getElementById("btn-play").innerHTML = "▶";
    if (playbackTimer) {
        clearInterval(playbackTimer);
        playbackTimer = null;
    }
}

function copyFindingsToClipboard() {
    if (!currentAnalysis) {
        alert("No analysis findings to copy.");
        return;
    }
    const lines = [];
    lines.push(`PoseGuard AI Analysis Report: ${currentAnalysis.file_path || 'Animation'}`);
    lines.push(`Overall Status: ${currentAnalysis.quality_summary.overall_status.toUpperCase()}`);
    lines.push(`Confirmed Defects: ${currentAnalysis.defects.length}`);
    lines.push(`Defect-Free Frames: ${currentAnalysis.quality_summary.defect_free_frame_percentage}%\n`);
    
    currentAnalysis.defects.forEach((d, i) => {
        lines.push(`[${i+1}] ${d.type.toUpperCase()} on '${d.joint}'`);
        lines.push(`    Frames: ${d.start_frame}–${d.end_frame} | Severity: ${d.severity.toUpperCase()} | Confidence: ${(d.confidence*100).toFixed(0)}%`);
        lines.push(`    Correction: ${d.suggested_correction}\n`);
    });
    
    navigator.clipboard.writeText(lines.join("\n")).then(() => {
        alert("Analysis findings copied to clipboard!");
    }).catch(e => {
        console.error("Clipboard copy failed:", e);
    });
}

function showLoading(msg) {
    const overlay = document.getElementById("loading-overlay");
    if (overlay) {
        document.getElementById("loading-msg").textContent = msg || "Processing...";
        overlay.style.display = "flex";
    }
}

function hideLoading() {
    const overlay = document.getElementById("loading-overlay");
    if (overlay) {
        overlay.style.display = "none";
    }
}

