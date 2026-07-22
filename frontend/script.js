/**
 * script.js — Content Moderation Frontend v2
 * Handles tabs, text analysis, file upload, URL analysis, and bulk results.
 */

(function () {
    'use strict';

    // ── DOM refs ──────────────────────────────────────────────────────────────

    // Tabs
    var tabs = document.querySelectorAll('.tab');
    var panels = document.querySelectorAll('.tab-panel');

    // Text tab
    var textInput = document.getElementById('text-input');
    var charCount = document.getElementById('char-count');
    var analyzeTextBtn = document.getElementById('analyze-text-btn');
    var chipButtons = document.querySelectorAll('.chip');

    // File tab
    var dropZone = document.getElementById('drop-zone');
    var fileInput = document.getElementById('file-input');
    var browseBtn = document.getElementById('browse-btn');
    var fileSelected = document.getElementById('file-selected');
    var fileName = document.getElementById('file-name');
    var fileSize = document.getElementById('file-size');
    var fileRemoveBtn = document.getElementById('file-remove-btn');
    var analyzeFileBtn = document.getElementById('analyze-file-btn');

    // URL tab
    var urlInput = document.getElementById('url-input');
    var analyzeUrlBtn = document.getElementById('analyze-url-btn');

    // Results
    var resultsSingle = document.getElementById('results-single');
    var resultsBulk = document.getElementById('results-bulk');
    var errorBox = document.getElementById('error-box');
    var errorMessage = document.getElementById('error-message');

    // Single result elements
    var resultCardSingle = document.getElementById('result-card-single');
    var gaugeFillSingle = document.getElementById('gauge-fill-single');
    var gaugeValueSingle = document.getElementById('gauge-value-single');
    var verdictBadgeSingle = document.getElementById('verdict-badge-single');
    var verdictIconSingle = document.getElementById('verdict-icon-single');
    var verdictTextSingle = document.getElementById('verdict-text-single');
    var verdictDetailSingle = document.getElementById('verdict-detail-single');
    var valSafeSingle = document.getElementById('val-safe-single');
    var valToxicSingle = document.getElementById('val-toxic-single');
    var latencySingle = document.getElementById('latency-single');

    // Bulk result elements
    var resultCardBulk = document.getElementById('result-card-bulk');
    var gaugeFillBulk = document.getElementById('gauge-fill-bulk');
    var gaugeValueBulk = document.getElementById('gauge-value-bulk');
    var verdictBadgeBulk = document.getElementById('verdict-badge-bulk');
    var verdictIconBulk = document.getElementById('verdict-icon-bulk');
    var verdictTextBulk = document.getElementById('verdict-text-bulk');
    var verdictDetailBulk = document.getElementById('verdict-detail-bulk');
    var bulkSource = document.getElementById('bulk-source');
    var bulkChunks = document.getElementById('bulk-chunks');
    var bulkToxicCount = document.getElementById('bulk-toxic-count');
    var bulkTime = document.getElementById('bulk-time');
    var chunkToggle = document.getElementById('chunk-toggle');
    var chunkList = document.getElementById('chunk-list');

    // Compare tab
    var compareInput = document.getElementById('compare-input');
    var compareCharCount = document.getElementById('compare-char-count');
    var analyzeCompareBtn = document.getElementById('analyze-compare-btn');

    // Compare results
    var resultsCompare = document.getElementById('results-compare');
    var resultCardCompareCustom = document.getElementById('result-card-compare-custom');
    var resultCardCompareBase = document.getElementById('result-card-compare-base');
    var gaugeFillCompareCustom = document.getElementById('gauge-fill-compare-custom');
    var gaugeValueCompareCustom = document.getElementById('gauge-value-compare-custom');
    var verdictBadgeCompareCustom = document.getElementById('verdict-badge-compare-custom');
    var verdictIconCompareCustom = document.getElementById('verdict-icon-compare-custom');
    var verdictTextCompareCustom = document.getElementById('verdict-text-compare-custom');
    var verdictDetailCompareCustom = document.getElementById('verdict-detail-compare-custom');
    var valSafeCompareCustom = document.getElementById('val-safe-compare-custom');
    var valToxicCompareCustom = document.getElementById('val-toxic-compare-custom');
    var latencyCompareCustom = document.getElementById('latency-compare-custom');

    var gaugeFillCompareBase = document.getElementById('gauge-fill-compare-base');
    var gaugeValueCompareBase = document.getElementById('gauge-value-compare-base');
    var verdictBadgeCompareBase = document.getElementById('verdict-badge-compare-base');
    var verdictIconCompareBase = document.getElementById('verdict-icon-compare-base');
    var verdictTextCompareBase = document.getElementById('verdict-text-compare-base');
    var verdictDetailCompareBase = document.getElementById('verdict-detail-compare-base');
    var valSafeCompareBase = document.getElementById('val-safe-compare-base');
    var valToxicCompareBase = document.getElementById('val-toxic-compare-base');
    var latencyCompareBase = document.getElementById('latency-compare-base');

    // Status
    var statusDot = document.getElementById('status-dot');
    var statusText = document.getElementById('status-text');

    // State
    var selectedFile = null;


    // ── Health check ──────────────────────────────────────────────────────────

    function checkHealth() {
        fetch('/health')
            .then(function (res) { return res.json(); })
            .then(function (data) {
                if (data.model_loaded && data.tokenizer_loaded) {
                    statusDot.className = 'status-dot is-online';
                    statusText.textContent = 'Model ready';
                    statusText.style.color = '#4ADE80';
                } else {
                    statusDot.className = 'status-dot is-offline';
                    statusText.textContent = 'Loading model…';
                    statusText.style.color = '';
                }
            })
            .catch(function () {
                statusDot.className = 'status-dot is-offline';
                statusText.textContent = 'Offline';
                statusText.style.color = '#F87171';
            });
    }
    checkHealth();


    // ── Tab switching ─────────────────────────────────────────────────────────

    tabs.forEach(function (tab) {
        tab.addEventListener('click', function () {
            var target = tab.getAttribute('data-tab');

            tabs.forEach(function (t) {
                t.classList.remove('is-active');
                t.setAttribute('aria-selected', 'false');
            });
            tab.classList.add('is-active');
            tab.setAttribute('aria-selected', 'true');

            panels.forEach(function (p) { p.classList.remove('is-active'); });
            document.getElementById('panel-' + target).classList.add('is-active');

            // Hide any previous results/errors when switching tabs
            hideAllResults();
            hideError();
        });
    });


    // ── Character counter ─────────────────────────────────────────────────────

    function updateCharCount() {
        var len = textInput.value.length;
        charCount.innerHTML = len.toLocaleString() + '<span class="char-sep">/</span>50,000';
    }
    textInput.addEventListener('input', updateCharCount);

    function updateCompareCharCount() {
        var len = compareInput.value.length;
        compareCharCount.innerHTML = len.toLocaleString() + '<span class="char-sep">/</span>50,000';
    }
    compareInput.addEventListener('input', updateCompareCharCount);


    // ── Keyboard shortcut (Ctrl/Cmd + Enter) ──────────────────────────────────

    textInput.addEventListener('keydown', function (e) {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            analyzeTextBtn.click();
        }
    });

    compareInput.addEventListener('keydown', function (e) {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            analyzeCompareBtn.click();
        }
    });


    // ── Example chips ─────────────────────────────────────────────────────────

    chipButtons.forEach(function (chip) {
        chip.addEventListener('click', function () {
            var isCompare = chip.closest('#panel-compare') !== null;
            if (isCompare) {
                compareInput.value = chip.getAttribute('data-text');
                updateCompareCharCount();
                compareInput.focus();
                setTimeout(function () { analyzeCompareBtn.click(); }, 150);
            } else {
                textInput.value = chip.getAttribute('data-text');
                updateCharCount();
                textInput.focus();
                setTimeout(function () { analyzeTextBtn.click(); }, 150);
            }
        });
    });



    // ═══════════════════════════════════════════════════════════════════════════
    //  WEBSOCKET — LIVE TYPING ANALYSIS
    // ═══════════════════════════════════════════════════════════════════════════

    var liveBar      = document.getElementById('live-bar');
    var livePulse    = document.getElementById('live-pulse');
    var liveLabel    = document.getElementById('live-label');
    var liveGaugeFill= document.getElementById('live-gauge-fill');
    var liveScore    = document.getElementById('live-score');
    var liveVerdict  = document.getElementById('live-verdict');
    var liveLatency  = document.getElementById('live-latency');

    var ws = null;
    var wsReconnectDelay = 1000;  // ms, doubles on each failure (max 16 s)
    var wsDebounceTimer  = null;
    var WS_DEBOUNCE_MS   = 300;
    var WS_MIN_CHARS     = 3;     // don't send fewer than this many chars

    function wsConnect() {
        var proto = location.protocol === 'https:' ? 'wss' : 'ws';
        var url   = proto + '://' + location.host + '/ws/predict';
        ws = new WebSocket(url);

        ws.onopen = function () {
            wsReconnectDelay = 1000;   // reset back-off
            setPulse('connecting');
            // Immediately send current text if any
            var t = textInput.value.trim();
            if (t.length >= WS_MIN_CHARS) ws.send(t);
        };

        ws.onmessage = function (evt) {
            try {
                var data = JSON.parse(evt.data);
                if (data.error) { return; }
                updateLiveBar(data);
            } catch (_) {}
        };

        ws.onerror  = function () { /* onclose will fire next */ };
        ws.onclose  = function () {
            ws = null;
            setPulse('connecting');
            // Exponential back-off reconnect
            wsReconnectDelay = Math.min(wsReconnectDelay * 2, 16000);
            setTimeout(wsConnect, wsReconnectDelay);
        };
    }

    function wsSend(text) {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(text);
        }
    }

    /* Update all live-bar elements from a prediction payload */
    function updateLiveBar(data) {
        var toxicPct = Math.round((data.confidence['toxic'] || 0) * 100);
        var isToxic  = data.toxic;
        var state    = isToxic ? (toxicPct >= 70 ? 'toxic' : 'warning') : 'safe';

        // Bar visibility + colour state
        liveBar.classList.add('is-active');
        liveBar.className = 'live-bar is-active is-' + state;

        // Pulse dot
        setPulse(state);

        // Gauge
        liveGaugeFill.style.width = toxicPct + '%';
        liveGaugeFill.className   = 'live-gauge-fill' + (state !== 'safe' ? ' is-' + state : '');

        // Score
        liveScore.textContent = toxicPct + '%';

        // Verdict badge
        liveVerdict.className   = 'live-verdict is-' + state;
        liveVerdict.textContent = isToxic ? (toxicPct >= 70 ? '⚠ Toxic' : '⚡ Borderline') : '✓ Clean';

        // Latency chip
        if (data.latency_ms > 0) {
            liveLatency.textContent = data.latency_ms + 'ms';
        }
    }

    function setPulse(state) {
        livePulse.className = 'live-pulse' + (state ? ' is-' + state : '');
    }

    function hideLiveBar() {
        liveBar.classList.remove('is-active');
        liveBar.className = 'live-bar';
        liveGaugeFill.style.width = '0%';
        liveScore.textContent = '—';
        liveVerdict.className = 'live-verdict';
        liveVerdict.textContent = '';
        liveLatency.textContent = '';
        setPulse('');
    }

    /* Debounced send on every keystroke */
    textInput.addEventListener('input', function () {
        clearTimeout(wsDebounceTimer);
        var text = textInput.value.trim();
        if (text.length < WS_MIN_CHARS) {
            hideLiveBar();
            return;
        }
        setPulse('connecting');
        wsDebounceTimer = setTimeout(function () {
            wsSend(text);
        }, WS_DEBOUNCE_MS);
    });

    /* Hide live bar when switching away from text tab */
    tabs.forEach(function (tab) {
        tab.addEventListener('click', function () {
            if (tab.getAttribute('data-tab') !== 'text') {
                hideLiveBar();
            }
        });
    });

    // Start WebSocket connection
    wsConnect();


    // ═══════════════════════════════════════════════════════════════════════════
    //  TEXT ANALYSIS
    // ═══════════════════════════════════════════════════════════════════════════

    analyzeTextBtn.addEventListener('click', async function () {
        var text = textInput.value.trim();
        if (!text) {
            showError('Type or paste some text to analyze.');
            return;
        }

        hideError();
        hideAllResults();
        setLoading(analyzeTextBtn, true);

        var startTime = performance.now();

        try {
            // Use /predict for short text, /predict/bulk for long text
            var isLong = text.length > 500;
            var endpoint = isLong ? '/predict/bulk' : '/predict';

            var res = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text }),
            });

            var elapsed = Math.round(performance.now() - startTime);

            if (!res.ok) {
                var errData = await res.json().catch(function () { return { detail: 'Server returned ' + res.status }; });
                throw new Error(errData.detail || 'Request failed');
            }

            var data = await res.json();

            if (isLong) {
                renderBulkResults(data);
            } else {
                renderSingleResult(data, elapsed);
            }
        } catch (err) {
            showError(err.message || 'Something went wrong. Is the server running?');
        } finally {
            setLoading(analyzeTextBtn, false);
        }
    });


    analyzeCompareBtn.addEventListener('click', async function () {
        var text = compareInput.value.trim();
        if (!text) {
            showError('Type or paste some text to compare models.');
            return;
        }

        hideError();
        hideAllResults();
        setLoading(analyzeCompareBtn, true);

        try {
            var res = await fetch('/predict/compare', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text }),
            });

            if (!res.ok) {
                var errData = await res.json().catch(function () { return { detail: 'Server returned ' + res.status }; });
                throw new Error(errData.detail || 'Comparison request failed');
            }

            var data = await res.json();
            renderComparisonResults(data);
        } catch (err) {
            showError(err.message || 'Something went wrong. Is the server running?');
        } finally {
            setLoading(analyzeCompareBtn, false);
        }
    });


    // ═══════════════════════════════════════════════════════════════════════════
    //  FILE UPLOAD
    // ═══════════════════════════════════════════════════════════════════════════

    // Browse button
    browseBtn.addEventListener('click', function (e) {
        e.stopPropagation();
        fileInput.click();
    });

    // Click on drop zone
    dropZone.addEventListener('click', function () {
        fileInput.click();
    });

    // Drag events
    dropZone.addEventListener('dragover', function (e) {
        e.preventDefault();
        dropZone.classList.add('is-dragover');
    });

    dropZone.addEventListener('dragleave', function () {
        dropZone.classList.remove('is-dragover');
    });

    dropZone.addEventListener('drop', function (e) {
        e.preventDefault();
        dropZone.classList.remove('is-dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    // File input change
    fileInput.addEventListener('change', function () {
        if (fileInput.files.length > 0) {
            handleFileSelect(fileInput.files[0]);
        }
    });

    function handleFileSelect(file) {
        selectedFile = file;
        fileName.textContent = file.name;
        fileSize.textContent = formatFileSize(file.size);
        dropZone.style.display = 'none';
        fileSelected.style.display = 'flex';
        hideError();
        hideAllResults();
    }

    // Remove file
    fileRemoveBtn.addEventListener('click', function () {
        selectedFile = null;
        fileInput.value = '';
        dropZone.style.display = 'flex';
        fileSelected.style.display = 'none';
        hideAllResults();
        hideError();
    });

    // Analyze file
    analyzeFileBtn.addEventListener('click', async function () {
        if (!selectedFile) {
            showError('No file selected.');
            return;
        }

        hideError();
        hideAllResults();
        setLoading(analyzeFileBtn, true);

        try {
            var formData = new FormData();
            formData.append('file', selectedFile);

            var res = await fetch('/predict/file', {
                method: 'POST',
                body: formData,
            });

            if (!res.ok) {
                var errData = await res.json().catch(function () { return { detail: 'Server returned ' + res.status }; });
                throw new Error(errData.detail || 'File analysis failed');
            }

            var data = await res.json();
            renderBulkResults(data);
        } catch (err) {
            showError(err.message || 'File analysis failed.');
        } finally {
            setLoading(analyzeFileBtn, false);
        }
    });


    // ═══════════════════════════════════════════════════════════════════════════
    //  URL ANALYSIS
    // ═══════════════════════════════════════════════════════════════════════════

    // Enter key in URL input
    urlInput.addEventListener('keydown', function (e) {
        if (e.key === 'Enter') {
            e.preventDefault();
            analyzeUrlBtn.click();
        }
    });

    analyzeUrlBtn.addEventListener('click', async function () {
        var url = urlInput.value.trim();
        if (!url) {
            showError('Enter a URL to analyze.');
            return;
        }

        hideError();
        hideAllResults();
        setLoading(analyzeUrlBtn, true);

        try {
            var res = await fetch('/predict/url', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url: url }),
            });

            if (!res.ok) {
                var errData = await res.json().catch(function () { return { detail: 'Server returned ' + res.status }; });
                throw new Error(errData.detail || 'URL analysis failed');
            }

            var data = await res.json();
            renderBulkResults(data);
        } catch (err) {
            showError(err.message || 'URL analysis failed.');
        } finally {
            setLoading(analyzeUrlBtn, false);
        }
    });


    // ═══════════════════════════════════════════════════════════════════════════
    //  RENDER RESULTS
    // ═══════════════════════════════════════════════════════════════════════════

    function renderSingleResult(data, latencyMs) {
        var isToxic = data.toxic;
        var toxicScore = data.confidence['toxic'];

        resultCardSingle.className = 'result-card ' + (isToxic ? 'is-toxic' : 'is-safe');
        animateGauge(gaugeFillSingle, gaugeValueSingle, toxicScore);

        verdictBadgeSingle.className = 'verdict-badge ' + (isToxic ? 'is-toxic' : 'is-safe');
        verdictIconSingle.textContent = isToxic ? '⚠' : '✓';
        verdictTextSingle.textContent = isToxic ? 'Toxic content detected' : 'Content looks clean';
        verdictDetailSingle.textContent = (data.score * 100).toFixed(1) + '% model confidence';

        valSafeSingle.textContent = (data.confidence['non-toxic'] * 100).toFixed(1) + '%';
        valToxicSingle.textContent = (toxicScore * 100).toFixed(1) + '%';

        latencySingle.innerHTML =
            '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg> ' +
            latencyMs + 'ms round-trip';

        resultsSingle.classList.add('is-visible');
    }


    function renderBulkResults(data) {
        var isToxic = data.overall_toxic;
        var toxicScore = data.overall_confidence['toxic'];

        resultCardBulk.className = 'result-card ' + (isToxic ? 'is-toxic' : 'is-safe');
        animateGauge(gaugeFillBulk, gaugeValueBulk, toxicScore);

        verdictBadgeBulk.className = 'verdict-badge ' + (isToxic ? 'is-toxic' : 'is-safe');
        verdictIconBulk.textContent = isToxic ? '⚠' : '✓';
        verdictTextBulk.textContent = isToxic ? 'Toxic content detected' : 'Content looks clean';
        verdictDetailBulk.textContent = data.toxic_chunks + ' of ' + data.total_chunks + ' chunks flagged as toxic';

        // Stats
        var sourceLabel = data.source === 'file' ? data.source_name : (data.source === 'url' ? truncateText(data.source_name, 30) : 'Text input');
        bulkSource.textContent = sourceLabel;
        bulkChunks.textContent = data.total_chunks;
        bulkToxicCount.textContent = data.toxic_chunks;
        bulkTime.textContent = data.processing_time_ms + 'ms';

        // Chunk breakdown
        renderChunkList(data.chunks);

        // Reset toggle
        chunkToggle.classList.remove('is-open');
        chunkList.style.display = 'none';

        resultsBulk.classList.add('is-visible');
    }


    function renderComparisonResults(data) {
        // Render Custom Model
        var custom = data.custom_model;
        var customToxic = custom.toxic;
        var customScore = custom.confidence['toxic'];

        resultCardCompareCustom.className = 'result-card ' + (customToxic ? 'is-toxic' : 'is-safe');
        animateGauge(gaugeFillCompareCustom, gaugeValueCompareCustom, customScore);

        verdictBadgeCompareCustom.className = 'verdict-badge ' + (customToxic ? 'is-toxic' : 'is-safe');
        verdictIconCompareCustom.textContent = customToxic ? '⚠' : '✓';
        verdictTextCompareCustom.textContent = customToxic ? 'Toxic detected' : 'Content clean';
        verdictDetailCompareCustom.textContent = (custom.score * 100).toFixed(1) + '% confidence';

        valSafeCompareCustom.textContent = (custom.confidence['non-toxic'] * 100).toFixed(1) + '%';
        valToxicCompareCustom.textContent = (customScore * 100).toFixed(1) + '%';

        latencyCompareCustom.innerHTML =
            '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg> ' +
            custom.latency_ms + 'ms latency';

        // Render Base Model
        var base = data.base_model;
        var baseToxic = base.toxic;
        var baseScore = base.confidence['toxic'];

        resultCardCompareBase.className = 'result-card ' + (baseToxic ? 'is-toxic' : 'is-safe');
        animateGauge(gaugeFillCompareBase, gaugeValueCompareBase, baseScore);

        verdictBadgeCompareBase.className = 'verdict-badge ' + (baseToxic ? 'is-toxic' : 'is-safe');
        verdictIconCompareBase.textContent = baseToxic ? '⚠' : '✓';
        verdictTextCompareBase.textContent = baseToxic ? 'Toxic detected' : 'Content clean';
        verdictDetailCompareBase.textContent = (base.score * 100).toFixed(1) + '% confidence';

        valSafeCompareBase.textContent = (base.confidence['non-toxic'] * 100).toFixed(1) + '%';
        valToxicCompareBase.textContent = (baseScore * 100).toFixed(1) + '%';

        latencyCompareBase.innerHTML =
            '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg> ' +
            base.latency_ms + 'ms latency';

        resultsCompare.classList.add('is-visible');
    }


    function renderChunkList(chunks) {
        var html = '';
        for (var i = 0; i < chunks.length; i++) {
            var c = chunks[i];
            var toxicPercent = (c.confidence['toxic'] * 100).toFixed(1);
            var scoreClass = c.toxic ? 'is-toxic' : (c.confidence['toxic'] > 0.3 ? 'is-warning' : 'is-safe');
            var scoreLabel = toxicPercent + '%';

            html += '<div class="chunk-item">' +
                '<span class="chunk-index">#' + (c.chunk_index + 1) + '</span>' +
                '<div class="chunk-body"><p class="chunk-preview">' + escapeHtml(c.text_preview) + '</p></div>' +
                '<span class="chunk-score ' + scoreClass + '">' + scoreLabel + '</span>' +
                '</div>';
        }
        chunkList.innerHTML = html;
    }


    // ── Gauge animation ───────────────────────────────────────────────────────

    function animateGauge(fillEl, valueEl, percent) {
        var maxOffset = 251.2;
        var offset = maxOffset - (maxOffset * percent);

        // Reset
        fillEl.style.strokeDashoffset = maxOffset;

        // Color
        if (percent > 0.7) {
            fillEl.style.stroke = '#F87171';
        } else if (percent > 0.4) {
            fillEl.style.stroke = '#FBBF24';
        } else {
            fillEl.style.stroke = '#4ADE80';
        }

        valueEl.textContent = (percent * 100).toFixed(1) + '%';

        // Animate
        requestAnimationFrame(function () {
            requestAnimationFrame(function () {
                fillEl.style.strokeDashoffset = offset;
            });
        });
    }


    // ── Chunk toggle ──────────────────────────────────────────────────────────

    chunkToggle.addEventListener('click', function () {
        var isOpen = chunkToggle.classList.toggle('is-open');
        chunkList.style.display = isOpen ? 'block' : 'none';
        chunkToggle.querySelector('span').textContent = isOpen ? 'Hide chunk-by-chunk breakdown' : 'Show chunk-by-chunk breakdown';
    });


    // ── Helpers ───────────────────────────────────────────────────────────────

    function setLoading(btn, loading) {
        if (loading) {
            btn.classList.add('is-loading');
            btn.disabled = true;
        } else {
            btn.classList.remove('is-loading');
            btn.disabled = false;
        }
    }

    function showError(msg) {
        errorMessage.textContent = msg;
        errorBox.classList.add('is-visible');
    }

    function hideError() {
        errorBox.classList.remove('is-visible');
    }

    function hideAllResults() {
        resultsSingle.classList.remove('is-visible');
        resultsBulk.classList.remove('is-visible');
        resultsCompare.classList.remove('is-visible');
    }

    function formatFileSize(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / 1024 / 1024).toFixed(1) + ' MB';
    }

    function truncateText(text, maxLen) {
        if (text.length <= maxLen) return text;
        return text.substring(0, maxLen) + '…';
    }

    function escapeHtml(str) {
        var div = document.createElement('div');
        div.appendChild(document.createTextNode(str));
        return div.innerHTML;
    }

})();
