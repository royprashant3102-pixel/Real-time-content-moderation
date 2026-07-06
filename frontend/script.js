/**
 * script.js — Handles text analysis via the /predict API and renders results.
 */

(function () {
    'use strict';

    // ── DOM refs ──────────────────────────────────────────────────────────────
    const textInput    = document.getElementById('text-input');
    const charCount    = document.getElementById('char-count');
    const analyzeBtn   = document.getElementById('analyze-btn');
    const resultsEl    = document.getElementById('results');
    const errorBox     = document.getElementById('error-box');
    const errorMessage = document.getElementById('error-message');

    // Verdict
    const verdictEl    = document.getElementById('result-verdict');
    const verdictIcon  = document.getElementById('verdict-icon');
    const verdictLabel = document.getElementById('verdict-label');
    const verdictScore = document.getElementById('verdict-score');

    // Bars
    const barFillSafe  = document.getElementById('bar-fill-safe');
    const barFillToxic = document.getElementById('bar-fill-toxic');
    const barValueSafe = document.getElementById('bar-value-safe');
    const barValueToxic = document.getElementById('bar-value-toxic');

    // Meta
    const latencyDisplay = document.getElementById('latency-display');

    // ── Character counter ─────────────────────────────────────────────────────
    function updateCharCount() {
        const len = textInput.value.length;
        charCount.textContent = len.toLocaleString() + ' / 5,000';
    }
    textInput.addEventListener('input', updateCharCount);

    // ── Keyboard shortcut (Ctrl/Cmd + Enter) ──────────────────────────────────
    textInput.addEventListener('keydown', function (e) {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            analyzeBtn.click();
        }
    });

    // ── Analyze ───────────────────────────────────────────────────────────────
    analyzeBtn.addEventListener('click', async function () {
        const text = textInput.value.trim();

        if (!text) {
            showError('Please enter some text first.');
            return;
        }

        // UI: loading state
        hideError();
        hideResults();
        analyzeBtn.classList.add('is-loading');
        analyzeBtn.disabled = true;

        const startTime = performance.now();

        try {
            const res = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text }),
            });

            const elapsed = Math.round(performance.now() - startTime);

            if (!res.ok) {
                const errData = await res.json().catch(function () {
                    return { detail: 'Server returned ' + res.status };
                });
                throw new Error(errData.detail || 'Request failed');
            }

            const data = await res.json();
            renderResults(data, elapsed);
        } catch (err) {
            showError(err.message || 'Something went wrong. Is the server running?');
        } finally {
            analyzeBtn.classList.remove('is-loading');
            analyzeBtn.disabled = false;
        }
    });

    // ── Render results ────────────────────────────────────────────────────────
    function renderResults(data, latencyMs) {
        var isToxic = data.toxic;

        // Verdict card
        verdictEl.className = 'result-verdict ' + (isToxic ? 'is-toxic' : 'is-safe');
        verdictIcon.textContent = isToxic ? '⚠️' : '✓';
        verdictLabel.textContent = isToxic ? 'Likely toxic' : 'Looks clean';
        verdictScore.textContent = (data.score * 100).toFixed(1) + '% confidence';

        // Confidence bars — reset first, then animate
        var safePercent  = (data.confidence['non-toxic'] * 100).toFixed(1);
        var toxicPercent = (data.confidence['toxic'] * 100).toFixed(1);

        barFillSafe.style.width  = '0%';
        barFillToxic.style.width = '0%';

        barValueSafe.textContent  = safePercent + '%';
        barValueToxic.textContent = toxicPercent + '%';

        // Delay to trigger CSS transition
        requestAnimationFrame(function () {
            requestAnimationFrame(function () {
                barFillSafe.style.width  = safePercent + '%';
                barFillToxic.style.width = toxicPercent + '%';
            });
        });

        // Latency
        latencyDisplay.textContent = 'Response time: ' + latencyMs + 'ms (round-trip)';

        // Show
        resultsEl.classList.add('is-visible');
    }

    // ── Error handling ────────────────────────────────────────────────────────
    function showError(msg) {
        errorMessage.textContent = msg;
        errorBox.classList.add('is-visible');
    }

    function hideError() {
        errorBox.classList.remove('is-visible');
    }

    function hideResults() {
        resultsEl.classList.remove('is-visible');
    }
})();
