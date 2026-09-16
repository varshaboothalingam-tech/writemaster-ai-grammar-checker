document.addEventListener('DOMContentLoaded', function() {
    const editor = document.getElementById('editor');
    const btnCheck = document.getElementById('btnCheck');
    const btnFixAll = document.getElementById('btnFixAll');
    const btnUndo = document.getElementById('btnUndo');
    const btnClear = document.getElementById('btnClear');
    const btnPaste = document.getElementById('btnPaste');
    const btnRephrase = document.getElementById('btnRephrase');
    const btnExport = document.getElementById('btnExport');
    const btnSynonym = document.getElementById('btnSynonym');
    const btnExportPdf = document.getElementById('btnExportPdf');
    const dialectSelect = document.getElementById('dialectSelect');
    const genreSelect = document.getElementById('genreSelect');
    const useAiCheck = document.getElementById('useAiCheck');
    const aiBadge = document.getElementById('aiBadge');
    const synonymInput = document.getElementById('synonymInput');
    const highlightedText = document.getElementById('highlightedText');
    const issuesList = document.getElementById('issuesList');
    const issueCountBadge = document.getElementById('issueCountBadge');
    const charCount = document.getElementById('charCount');
    const wordCount = document.getElementById('wordCount');
    const sentenceCount = document.getElementById('sentenceCount');
    const scoreValue = document.getElementById('scoreValue');
    const scoreLevel = document.getElementById('scoreLevel');
    const scoreCircle = document.getElementById('scoreCircle');
    const toneDominant = document.getElementById('toneDominant');
    const toneConfidence = document.getElementById('toneConfidence');
    const toneBreakdown = document.getElementById('toneBreakdown');
    const readabilityValue = document.getElementById('readabilityValue');
    const readabilityLevel = document.getElementById('readabilityLevel');
    const readabilityDetails = document.getElementById('readabilityDetails');
    const rephraseResult = document.getElementById('rephraseResult');
    const rephraseOutput = document.getElementById('rephraseOutput');
    const synonymResults = document.getElementById('synonymResults');

    let currentIssues = [];
    let previousText = '';
    let scoreChart = null;
    let selectedTone = 'professional';
    let debounceTimer = null;
    let autoCheckEnabled = true;

    editor.addEventListener('input', function() {
        updateCounts();
        scheduleAutoCheck();
    });
    editor.addEventListener('keydown', function(e) {
        if (e.ctrlKey && e.key === 'Enter') {
            e.preventDefault();
            checkGrammar();
        }
    });

    function scheduleAutoCheck() {
        if (!autoCheckEnabled) return;
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => {
            const text = editor.value.trim();
            if (text.length > 0) {
                checkGrammar();
            } else {
                clearResults();
            }
        }, 700);
    }

    function clearResults() {
        currentIssues = [];
        updateAiBadge(false);
        highlightedText.innerHTML = '<p class="text-muted mb-0">Start typing to check your writing.</p>';
        issuesList.innerHTML = '<p class="text-muted mb-0">No issues found yet.</p>';
        issueCountBadge.textContent = '0 issues';
        scoreValue.textContent = '--';
        scoreLevel.textContent = 'Not analyzed';
        scoreCircle.className = 'score-circle mx-auto';
    }

    btnCheck.addEventListener('click', () => checkGrammar(true));
    btnFixAll.addEventListener('click', fixAll);
    btnUndo.addEventListener('click', undo);
    btnClear.addEventListener('click', clearEditor);
    btnPaste.addEventListener('click', pasteFromClipboard);
    btnRephrase.addEventListener('click', rephraseText);
    btnExport.addEventListener('click', exportReport);
    btnSynonym.addEventListener('click', lookupSynonym);
    if (btnExportPdf) btnExportPdf.addEventListener('click', exportPdf);
    synonymInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter') lookupSynonym();
    });

    document.querySelectorAll('.tone-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            document.querySelectorAll('.tone-btn').forEach(b => b.classList.remove('active'));
            this.classList.add('active');
            selectedTone = this.dataset.tone;
        });
    });

    function updateCounts() {
        const text = editor.value;
        charCount.textContent = text.length;
        const words = text.trim().split(/\s+/).filter(w => w.length > 0);
        wordCount.textContent = words.length;
        const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
        sentenceCount.textContent = sentences.length;
    }

    async function checkGrammar(isManual = false) {
        const text = editor.value.trim();
        if (!text) {
            if (isManual) alert('Please enter some text to check.');
            return;
        }

        if (isManual) {
            btnCheck.disabled = true;
            btnCheck.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Checking...';
        }

        try {
            const response = await fetch('/api/check-v4', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text,
                    dialect: dialectSelect ? dialectSelect.value : 'us',
                    genre: genreSelect ? genreSelect.value : 'general',
                    use_ai: useAiCheck ? useAiCheck.checked : false,
                })
            });
            const data = await response.json();
            currentIssues = data.issues;
            updateAiBadge(data.meta && data.meta.ai_used);
            displayHighlightedText(text, data.issues);
            displayIssues(data.issues);
            displayReadability(data.readability);
            if (data.scores) {
                displayScores(data.scores);
                if (data.scores.overall) updateScore(data.scores.overall.score);
            } else {
                updateScore(100);
            }
            if (isManual) await detectTone(text);
        } catch (error) {
            console.error('Check failed:', error);
            if (isManual) alert('Error checking grammar. Please try again.');
        }

        if (isManual) {
            btnCheck.disabled = false;
            btnCheck.innerHTML = '<i class="fas fa-spell-check"></i> Check Grammar';
        }
    }

    function updateAiBadge(aiUsed) {
        if (!aiBadge) return;
        if (aiUsed) {
            aiBadge.classList.remove('d-none');
            aiBadge.textContent = '\u2713 AI-validated';
        } else {
            aiBadge.classList.add('d-none');
        }
    }

    function displayHighlightedText(text, issues) {
        if (!issues.length) {
            highlightedText.innerHTML = `<div class="highlighted-text"><p class="text-success mb-0"><i class="fas fa-check-circle"></i> No issues found. Great writing!</p></div>`;
            return;
        }

        const sorted = [...issues].sort((a, b) => (b.position || 0) - (a.position || 0));
        let result = text;

        for (const issue of sorted) {
            const word = issue.word || '';
            if (!word) continue;
            const pos = result.toLowerCase().indexOf(word.toLowerCase());
            if (pos === -1) continue;

            const cls = issue.severity === 'error' ? 'highlight-error' :
                        issue.severity === 'warning' ? 'highlight-warning' :
                        issue.type === 'contextual_word_usage' ? 'highlight-contextual' :
                        'highlight-info';
            const issueIdx = currentIssues.indexOf(issue);
            const before = result.substring(0, pos);
            const after = result.substring(pos + word.length);
            result = `${before}<span class="${cls} clickable-highlight" data-issue-index="${issueIdx}" title="Click for suggestion">${escapeHtml(word)}</span>${after}`;
        }

        highlightedText.innerHTML = `<div class="highlighted-text">${result}</div>`;

        highlightedText.querySelectorAll('.clickable-highlight').forEach(el => {
            el.addEventListener('click', function(e) {
                e.stopPropagation();
                const idx = parseInt(this.dataset.issueIndex);
                showIssuePopover(idx, this);
            });
        });
    }

    function showIssuePopover(issueIdx, anchorEl) {
        removeIssuePopover();
        const issue = currentIssues[issueIdx];
        if (!issue) return;

        const popover = document.createElement('div');
        popover.className = 'issue-popover';
        popover.id = 'activePopover';

        const confPct = Math.round((issue.confidence || 0.8) * 100);
        const suggestions = (issue.suggestions || []).filter(s => s).map(s =>
            `<span class="badge bg-light text-dark suggestion-chip" style="cursor:pointer; margin:2px;" onclick="applySuggestion(${issueIdx}, '${escapeHtml(s).replace(/'/g, "\\'")}')">${escapeHtml(s)}</span>`
        ).join(' ');

        popover.innerHTML = `
            <div class="popover-header">
                <strong>${escapeHtml(issue.word || 'Issue')}</strong>
                <span class="badge bg-${issue.severity === 'error' ? 'danger' : issue.severity === 'warning' ? 'warning' : 'info'}" style="font-size:10px;">${issue.severity} ${confPct}%</span>
                <button class="popover-close" onclick="removeIssuePopover()">&times;</button>
            </div>
            <div class="popover-body">
                <div style="font-size:13px; margin-bottom:6px;">${escapeHtml(issue.message)}</div>
                ${issue.explanation ? `<div style="font-size:11px; color:var(--text-muted); font-style:italic; margin-bottom:6px;">${escapeHtml(issue.explanation)}</div>` : ''}
                <div style="margin-bottom:6px;">${suggestions}</div>
                <div class="popover-actions">
                    <button class="btn-apply" style="font-size:11px; padding:2px 8px;" onclick="applyIssueFix(${issueIdx}); removeIssuePopover();"><i class="fas fa-check"></i> Accept</button>
                    <button class="btn-ignore" style="font-size:11px; padding:2px 8px;" onclick="ignoreIssue(${issueIdx}); removeIssuePopover();"><i class="fas fa-times"></i> Ignore</button>
                </div>
            </div>
        `;

        document.body.appendChild(popover);

        const rect = anchorEl.getBoundingClientRect();
        popover.style.position = 'fixed';
        popover.style.left = rect.left + 'px';
        popover.style.top = (rect.bottom + 6) + 'px';
        popover.style.zIndex = '10000';
        popover.style.maxWidth = '320px';

        document.addEventListener('click', function closePopover(e) {
            if (!popover.contains(e.target) && e.target !== anchorEl) {
                removeIssuePopover();
                document.removeEventListener('click', closePopover);
            }
        });
    }

    window.removeIssuePopover = function() {
        const existing = document.getElementById('activePopover');
        if (existing) existing.remove();
    };

    function displayIssues(issues) {
        const count = issues.length;
        issueCountBadge.textContent = `${count} issue${count !== 1 ? 's' : ''}`;

        const spelling = issues.filter(i => i.type === 'spelling').length;
        const grammar = issues.filter(i => i.type === 'grammar').length;
        const style = issues.filter(i => i.type === 'style').length;
        const punctuation = issues.filter(i => i.type === 'punctuation').length;
        const context = issues.filter(i => i.type === 'context' || i.type === 'word_usage' || i.type === 'contextual_word_usage').length;

        document.getElementById('spellingCount').textContent = spelling;
        document.getElementById('grammarCount').textContent = grammar;
        document.getElementById('styleCount').textContent = style;
        document.getElementById('punctuationCount').textContent = punctuation;
        document.getElementById('contextCount').textContent = context;

        if (!count) {
            issuesList.innerHTML = '<p class="text-success mb-0"><i class="fas fa-check-circle"></i> No issues found!</p>';
            return;
        }

        issuesList.innerHTML = issues.map((issue, idx) => {
            const sevClass = `severity-${issue.severity}`;
            const confPct = Math.round((issue.confidence || 0.8) * 100);
            const suggestions = (issue.suggestions || []).filter(s => s).map(s =>
                `<span class="badge bg-light text-dark suggestion-chip" style="cursor:pointer;" onclick="applySuggestion(${idx}, '${escapeHtml(s).replace(/'/g, "\\'")}')">${escapeHtml(s)}</span>`
            ).join('');
            const explanation = issue.explanation ? `<div class="issue-explanation" style="font-size:12px;color:var(--text-muted);margin-top:4px;font-style:italic;">${escapeHtml(issue.explanation)}</div>` : '';
            const dictBtn = issue.can_add_to_dict
                ? `<button class="btn-dict" onclick="addToDictionary(${idx})" title="Add to dictionary"><i class="fas fa-book"></i> Dict</button>` : '';

            return `
                <div class="issue-card" data-index="${idx}">
                    <div class="issue-header">
                        <span class="issue-title">${escapeHtml(issue.word || 'Issue')}</span>
                        <span class="issue-severity ${sevClass}">${issue.severity}</span>
                        <span class="badge bg-light text-muted">${escapeHtml(issue.rule || issue.type)}</span>
                        <span class="badge bg-light text-muted" style="font-size:10px;">${confPct}%</span>
                    </div>
                    <div class="issue-message">${escapeHtml(issue.message)}</div>
                    ${explanation}
                    <div class="issue-suggestions">${suggestions}</div>
                    <div class="issue-actions">
                        <button class="btn-apply" onclick="applyIssueFix(${idx})"><i class="fas fa-check"></i> Accept</button>
                        <button class="btn-ignore" onclick="ignoreIssue(${idx})"><i class="fas fa-times"></i> Ignore</button>
                        <button class="btn-ignore-all" onclick="ignoreAllInstances(${idx})"><i class="fas fa-eye-slash"></i> Ignore All</button>
                        ${dictBtn}
                    </div>
                </div>
            `;
        }).join('');
    }

    function updateScore(score) {
        scoreValue.textContent = Math.round(score);
        let level, cls;
        if (score >= 90) { level = 'Excellent'; cls = 'excellent'; }
        else if (score >= 70) { level = 'Good'; cls = 'good'; }
        else if (score >= 50) { level = 'Fair'; cls = 'fair'; }
        else { level = 'Needs Work'; cls = 'poor'; }

        scoreLevel.textContent = level;
        scoreCircle.className = `score-circle mx-auto ${cls}`;

        if (scoreChart) scoreChart.destroy();
        const ctx = document.getElementById('scoreChart').getContext('2d');
        scoreChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Spelling', 'Grammar', 'Style', 'Punctuation', 'Context'],
                datasets: [{
                    data: [
                        parseInt(document.getElementById('spellingCount').textContent),
                        parseInt(document.getElementById('grammarCount').textContent),
                        parseInt(document.getElementById('styleCount').textContent),
                        parseInt(document.getElementById('punctuationCount').textContent),
                        parseInt(document.getElementById('contextCount').textContent)
                    ],
                    backgroundColor: ['#e63946', '#ff9f1c', '#4cc9f0', '#6c757d', '#4361ee'],
                    borderRadius: 4
                }]
            },
            options: {
                responsive: false,
                plugins: { legend: { display: false } },
                scales: {
                    y: { beginAtZero: true, ticks: { stepSize: 1, font: { size: 11 } } },
                    x: { ticks: { font: { size: 10 } } }
                }
            }
        });
    }

    function displayReadability(data) {
        readabilityValue.textContent = data.score;
        readabilityLevel.textContent = data.level;

        let color;
        if (data.score >= 80) color = 'var(--success)';
        else if (data.score >= 60) color = 'var(--primary)';
        else if (data.score >= 40) color = 'var(--warning)';
        else color = 'var(--danger)';
        readabilityValue.style.color = color;

        readabilityDetails.innerHTML = `
            <div class="readability-detail"><span>Grade Level</span><span>${data.grade}</span></div>
            <div class="readability-detail"><span>Sentences</span><span>${data.sentences}</span></div>
            <div class="readability-detail"><span>Words</span><span>${data.words}</span></div>
            <div class="readability-detail"><span>Syllables</span><span>${data.syllables}</span></div>
            <div class="readability-detail"><span>Avg Words/Sentence</span><span>${data.avg_words_sentence}</span></div>
            <div class="readability-detail"><span>Reading Time</span><span>${data.reading_time} min</span></div>
            <div class="readability-detail"><span>Complex Words</span><span>${data.complex_word_pct}%</span></div>
            <div class="readability-detail"><span>Assessment</span><span>${escapeHtml(data.explanation)}</span></div>
        `;
    }

    function displayScores(scores) {
        const dims = {
            'dimCorrectness': scores.correctness,
            'dimClarity': scores.clarity,
            'dimEngagement': scores.engagement,
            'dimDelivery': scores.delivery,
            'dimOverall': scores.overall,
        };
        for (const [id, data] of Object.entries(dims)) {
            const el = document.getElementById(id);
            if (!el) continue;
            const s = data.score;
            const label = data.label;
            el.textContent = `${s} — ${label}`;
            if (s >= 80) el.className = 'badge bg-success';
            else if (s >= 60) el.className = 'badge bg-primary';
            else if (s >= 40) el.className = 'badge bg-warning text-dark';
            else el.className = 'badge bg-danger';
        }
    }

    async function detectTone(text) {
        try {
            const response = await fetch('/api/tone', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });
            const data = await response.json();
            toneDominant.textContent = data.dominant_tone;
            toneConfidence.textContent = `${data.confidence}% confidence`;

            const sorted = Object.entries(data.tones).sort((a, b) => b[1] - a[1]);
            toneBreakdown.innerHTML = sorted.map(([tone, score]) => `
                <div class="mb-2">
                    <div class="tone-label">${tone}</div>
                    <div class="tone-bar">
                        <div class="tone-bar-fill" style="width:${Math.max(score, 3)}%"></div>
                    </div>
                    <div class="text-end" style="font-size:11px;color:var(--text-muted);">${score}%</div>
                </div>
            `).join('');
        } catch (error) {
            console.error('Tone detection failed:', error);
        }
    }

    async function fixAll() {
        const text = editor.value.trim();
        if (!text) return;

        previousText = editor.value;
        btnUndo.disabled = false;
        btnFixAll.disabled = true;
        btnFixAll.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Fixing...';

        try {
            const response = await fetch('/api/fix-all', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });
            const data = await response.json();
            editor.value = data.corrected;
            updateCounts();
            await checkGrammar();
        } catch (error) {
            console.error('Fix all failed:', error);
        }

        btnFixAll.disabled = false;
        btnFixAll.innerHTML = '<i class="fas fa-wand-magic-sparkles"></i> Fix All';
    }

    function undo() {
        if (previousText) {
            editor.value = previousText;
            previousText = '';
            btnUndo.disabled = true;
            updateCounts();
        }
    }

    function clearEditor() {
        editor.value = '';
        previousText = '';
        currentIssues = [];
        updateAiBadge(false);
        btnUndo.disabled = true;
        updateCounts();
        highlightedText.innerHTML = '<p class="text-muted mb-0">Text will be highlighted with error underlines after checking.</p>';
        issuesList.innerHTML = '<p class="text-muted mb-0">No issues found yet. Click "Check Grammar" to start.</p>';
        issueCountBadge.textContent = '0 issues';
        scoreValue.textContent = '--';
        scoreLevel.textContent = 'Not analyzed';
        scoreCircle.className = 'score-circle mx-auto';
        toneDominant.textContent = '--';
        toneConfidence.textContent = 'Analyze text to detect tone';
        toneBreakdown.innerHTML = '';
        readabilityValue.textContent = '--';
        readabilityLevel.textContent = 'Not analyzed';
        readabilityDetails.innerHTML = '';
        rephraseResult.style.display = 'none';
        if (scoreChart) { scoreChart.destroy(); scoreChart = null; }
    }

    async function pasteFromClipboard() {
        try {
            const text = await navigator.clipboard.readText();
            editor.value = text;
            updateCounts();
        } catch (err) {
            alert('Unable to paste from clipboard.');
        }
    }

    async function rephraseText() {
        const text = editor.value.trim();
        if (!text) { alert('Please enter text to rephrase.'); return; }

        btnRephrase.disabled = true;
        btnRephrase.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Rephrasing...';

        try {
            const response = await fetch('/api/rephrase', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text, tone: selectedTone })
            });
            const data = await response.json();
            rephraseOutput.textContent = data.rephrased;
            rephraseResult.style.display = 'block';
        } catch (error) {
            console.error('Rephrase failed:', error);
        }

        btnRephrase.disabled = false;
        btnRephrase.innerHTML = '<i class="fas fa-rotate"></i> Rephrase Text';
    }

    document.getElementById('btnCopyRephrase').addEventListener('click', function() {
        navigator.clipboard.writeText(rephraseOutput.textContent);
        this.innerHTML = '<i class="fas fa-check"></i> Copied!';
        setTimeout(() => { this.innerHTML = '<i class="fas fa-copy"></i> Copy'; }, 2000);
    });

    function exportReport() {
        const text = editor.value;
        if (!text.trim()) { alert('No text to export.'); return; }

        let report = 'WriteMaster AI - Grammar Check Report\n';
        report += '='.repeat(40) + '\n\n';
        report += `Date: ${new Date().toLocaleString()}\n`;
        report += `Text Length: ${text.length} characters, ${text.split(/\s+/).length} words\n\n`;
        report += 'TEXT:\n' + text + '\n\n';

        if (currentIssues.length) {
            report += `ISSUES FOUND: ${currentIssues.length}\n`;
            report += '-'.repeat(30) + '\n';
            currentIssues.forEach((issue, idx) => {
                report += `\n${idx + 1}. [${issue.severity.toUpperCase()}] ${issue.word || 'Issue'}\n`;
                report += `   ${issue.message}\n`;
                if (issue.suggestions && issue.suggestions.length) {
                    report += `   Suggestions: ${issue.suggestions.join(', ')}\n`;
                }
            });
        } else {
            report += 'NO ISSUES FOUND\n';
        }

        const blob = new Blob([report], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'writemaster-report.txt';
        a.click();
        URL.revokeObjectURL(url);
    }

    async function lookupSynonym() {
        const word = synonymInput.value.trim();
        if (!word) return;

        try {
            const response = await fetch('/api/synonyms', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ word })
            });
            const data = await response.json();

            if (data.synonyms && data.synonyms.length) {
                synonymResults.innerHTML = data.synonyms.map(s =>
                    `<span class="synonym-chip" onclick="insertSynonym('${escapeHtml(s)}')">${escapeHtml(s)}</span>`
                ).join('');
            } else {
                synonymResults.innerHTML = '<span class="text-muted">No synonyms found.</span>';
            }
        } catch (error) {
            console.error('Synonym lookup failed:', error);
        }
    }

    window.insertSynonym = function(synonym) {
        const word = synonymInput.value.trim();
        const text = editor.value;
        const idx = text.toLowerCase().indexOf(word.toLowerCase());
        if (idx !== -1) {
            editor.value = text.substring(0, idx) + synonym + text.substring(idx + word.length);
            updateCounts();
        }
    };

    window.applySuggestion = function(issueIndex, suggestion) {
        if (issueIndex !== undefined && issueIndex !== null) {
            applyIssueFix(issueIndex);
            return;
        }
        editor.value = suggestion;
        updateCounts();
    };

    window.applyIssueFix = async function(index) {
        const issue = currentIssues[index];
        if (!issue) return;
        previousText = editor.value;
        btnUndo.disabled = false;

        const startPos = issue.start_position !== undefined ? issue.start_position : issue.position;
        const endPos = issue.end_position !== undefined ? issue.end_position : (startPos + (issue.word || '').length);

        if (issue.corrected_text && issue.corrected_text !== issue.word) {
            const text = editor.value;
            const word = issue.word || '';
            const sentence = issue.sentence || '';
            if (sentence && sentence.length > word.length + 2) {
                const sentIdx = text.indexOf(sentence);
                if (sentIdx !== -1) {
                    const newSentence = sentence.replace(new RegExp('\\b' + escapeRegex(word) + '\\b', 'i'), issue.corrected_text);
                    editor.value = text.substring(0, sentIdx) + newSentence + text.substring(sentIdx + sentence.length);
                    updateCounts();
                    await checkGrammar();
                    return;
                }
            }
            const wordIdx = text.toLowerCase().indexOf(word.toLowerCase());
            if (wordIdx !== -1) {
                editor.value = text.substring(0, wordIdx) + issue.corrected_text + text.substring(wordIdx + word.length);
                updateCounts();
                await checkGrammar();
                return;
            }
        }

        if (issue.suggestions && issue.suggestions.length && issue.suggestions[0]) {
            const text = editor.value;
            const word = issue.word || '';
            const wordIdx = text.toLowerCase().indexOf(word.toLowerCase());
            if (wordIdx !== -1) {
                editor.value = text.substring(0, wordIdx) + issue.suggestions[0] + text.substring(wordIdx + word.length);
                updateCounts();
                await checkGrammar();
            }
        }
    };

    window.ignoreIssue = function(index) {
        const card = document.querySelector(`.issue-card[data-index="${index}"]`);
        if (card) {
            card.style.opacity = '0.4';
            card.style.pointerEvents = 'none';
        }
    };

    window.addToDictionary = async function(index) {
        const issue = currentIssues[index];
        if (!issue) return;
        const word = issue.word || '';
        if (!word) return;
        try {
            const response = await fetch('/api/add-to-dictionary', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ word })
            });
            const data = await response.json();
            if (data.success) {
                const card = document.querySelector(`.issue-card[data-index="${index}"]`);
                if (card) {
                    card.style.opacity = '0.4';
                    card.style.pointerEvents = 'none';
                }
                await checkGrammar();
            }
        } catch (error) {
            console.error('Add to dictionary failed:', error);
        }
    };

    window.ignoreAllInstances = async function(index) {
        const issue = currentIssues[index];
        if (!issue) return;
        const word = issue.word || '';
        if (!word) return;
        try {
            const response = await fetch('/api/ignore-all', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ word })
            });
            const data = await response.json();
            if (data.success) {
                await checkGrammar();
            }
        } catch (error) {
            console.error('Ignore all failed:', error);
        }
    };

    function exportPdf() {
        const text = editor.value.trim();
        if (!text) { alert('No text to export.'); return; }
        const dialect = dialectSelect ? dialectSelect.value : 'us';
        const genre = genreSelect ? genreSelect.value : 'general';
        const w = window.open('', '_blank');
        const issues = currentIssues || [];
        let issuesHtml = '';
        if (issues.length) {
            issuesHtml = '<h3>Issues Found</h3><ul>';
            for (const issue of issues) {
                const sugg = issue.suggestions && issue.suggestions.length ? ' → ' + escapeHtml(issue.suggestions[0]) : '';
                issuesHtml += `<li><strong>${escapeHtml(issue.rule)}</strong>: ${escapeHtml(issue.message)}${sugg}</li>`;
            }
            issuesHtml += '</ul>';
        } else {
            issuesHtml = '<p style="color:green;">No issues found.</p>';
        }
        w.document.write(`
            <html><head><title>WriteMaster Report</title>
            <style>body{font-family:sans-serif;margin:40px;line-height:1.6;}
            h1{color:#2563eb;} h3{color:#555;} ul{margin:10px 0;}
            li{margin:5px 0;} pre{background:#f5f5f5;padding:15px;border-radius:5px;white-space:pre-wrap;}</style>
            </head><body>
            <h1>WriteMaster Writing Report</h1>
            <p><strong>Dialect:</strong> ${dialect.toUpperCase()} | <strong>Genre:</strong> ${genre}</p>
            <h3>Original Text</h3>
            <pre>${escapeHtml(text)}</pre>
            ${issuesHtml}
            <p style="color:#888;font-size:12px;margin-top:40px;">Generated by WriteMaster AI Grammar Checker</p>
            </body></html>`);
        w.document.close();
        w.print();
    }

    function escapeHtml(text) {
        if (!text) return '';
        const map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' };
        return text.replace(/[&<>"']/g, m => map[m]);
    }

    function escapeRegex(str) {
        return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }
});
