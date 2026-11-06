'use strict';
/* ==========================================================================
   SkinSense AI - front-end
   Plain JavaScript, no build step. Sections:
   helpers - toasts/dialog - theme - auth - navigation - profile - history
   - analysis (upload/results) - care finder/map - weather - chat
   ========================================================================== */

// API address: set in js/config.js (Vercel build) -> else same origin when the backend serves this page ->
// else localhost when the file is opened directly.
const API_URL = (window.SKINSENSE_API_URL || '').replace(/\/+$/, '')
    || (location.protocol.startsWith('http') ? location.origin : 'http://127.0.0.1:8000');

const $ = (id) => document.getElementById(id);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

/** Every dynamic string that goes into innerHTML MUST pass through esc(). */
function esc(value) {
    return String(value ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
const icon = (name, cls = '') => `<svg class="i ${cls}" aria-hidden="true"><use href="#i-${name}"/></svg>`;
const authHeaders = (extra = {}) => ({ 'Authorization': `Bearer ${localStorage.getItem('token')}`, ...extra });
const detailOf = (data, fallback) => (data && typeof data.detail === 'string') ? data.detail : fallback;

/** fetch with a timeout so a hung third-party service never leaves a spinner running forever. */
async function fetchExternal(url, ms = 15000) {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), ms);
    try {
        return await fetch(url, { signal: ctrl.signal });
    } catch (e) {
        throw new Error(e.name === 'AbortError' ? 'The service took too long to respond.' : 'Could not reach the service. Check your connection.');
    } finally {
        clearTimeout(timer);
    }
}

function setLoading(btn, on) {
    btn.classList.toggle('loading', on);
    btn.disabled = on;
    btn.setAttribute('aria-busy', on ? 'true' : 'false');
}

/** fetch wrapper: adds auth, turns network errors into readable ones, handles expired sessions. */
async function api(path, { method = 'GET', json, body } = {}) {
    const headers = authHeaders(json ? { 'Content-Type': 'application/json' } : {});
    let res;
    try {
        res = await fetch(API_URL + path, { method, headers, body: json ? JSON.stringify(json) : body });
    } catch (e) {
        throw new Error('Cannot reach the server. Please check that it is running.');
    }
    if (res.status === 401) {
        logout('Your session has expired. Please sign in again.');
        throw new Error('Session expired');
    }
    return res;
}

/* ---------- Risk presentation ---------- */
const RISK = {
    'Melanoma':             { tone: 'high',   label: 'High priority' },
    'Basal Cell Carcinoma': { tone: 'high',   label: 'High priority' },
    'Actinic Keratoses':    { tone: 'medium', label: 'Review advised' },
    'Benign Keratosis':     { tone: 'low',    label: 'Typically benign' },
    'Nevi':                 { tone: 'low',    label: 'Typically benign' },
    'Dermatofibroma':       { tone: 'low',    label: 'Typically benign' },
    'Vascular':             { tone: 'low',    label: 'Typically benign' },
};
function classify(prediction) {
    const p = String(prediction || '');
    if (p.startsWith('Melanoma')) return RISK['Melanoma'];
    if (p.startsWith('Uncertain')) return { tone: 'neutral', label: 'Inconclusive' };
    return RISK[p] || { tone: 'neutral', label: 'Review' };
}
const shortResult = (p) => String(p).startsWith('Melanoma (') ? 'Melanoma flag' : (String(p).startsWith('Uncertain') ? 'Inconclusive' : p);

/* ==========================================================================
   Toasts and dialog (replace alert / confirm / prompt)
   ========================================================================== */
function toast(message, type = 'info', ms = 5000) {
    const el = document.createElement('div');
    el.className = `toast ${type}`;
    el.setAttribute('role', type === 'error' ? 'alert' : 'status');
    const ic = { success: 'check', error: 'x-circle', warn: 'alert', info: 'info' }[type] || 'info';
    el.innerHTML = `${icon(ic)}<div></div>`;
    el.lastElementChild.textContent = message;
    $('toasts').appendChild(el);
    setTimeout(() => { el.classList.add('leaving'); setTimeout(() => el.remove(), 260); }, ms);
}

function askDialog({ title, message, confirmText = 'Confirm', danger = false, password = false }) {
    return new Promise(resolve => {
        const dlg = $('dialog');
        $('dlgTitle').textContent = title;
        $('dlgMessage').textContent = message;
        $('dlgConfirm').textContent = confirmText;
        $('dlgConfirm').className = `btn ${danger ? 'btn-danger' : 'btn-primary'}`;
        $('dlgPasswordWrap').classList.toggle('hidden', !password);
        $('dlgPassword').value = '';
        const finish = (ok) => { if (dlg.open) dlg.close(); resolve({ ok, value: $('dlgPassword').value }); };
        $('dlgForm').onsubmit = (e) => {
            e.preventDefault();
            if (password && !$('dlgPassword').value) { $('dlgPassword').focus(); return; }
            finish(true);
        };
        $('dlgCancel').onclick = () => finish(false);
        dlg.oncancel = (e) => { e.preventDefault(); finish(false); };
        dlg.showModal();
        (password ? $('dlgPassword') : $('dlgCancel')).focus();
    });
}

/* ==========================================================================
   Theme
   ========================================================================== */
function applyTheme(theme) {
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem('theme', theme); } catch (e) { /* private mode */ }
    const dark = theme === 'dark';
    $('themeIconSun').classList.toggle('hidden', !dark);   // show the icon of the theme you will switch TO
    $('themeIconMoon').classList.toggle('hidden', dark);
    $('themeToggleText').textContent = dark ? 'Light mode' : 'Dark mode';
    document.querySelector('meta[name="theme-color"]').content = dark ? '#0a1120' : '#0b63ce';
}
$('themeToggleBtn').onclick = () => applyTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark');

/* ==========================================================================
   Authentication
   ========================================================================== */
let authMode = 'login';

function showAuthError(msg) {
    const el = $('authError');
    el.textContent = msg || '';
    el.classList.toggle('hidden', !msg);
}

function setAuthMode(mode) {
    authMode = mode;
    const reg = mode === 'register';
    $('authForm').dataset.mode = mode;
    $('authTabLogin').setAttribute('aria-selected', String(!reg));
    $('authTabRegister').setAttribute('aria-selected', String(reg));
    $('entryTitle').textContent = reg ? 'Create your account' : 'Welcome back';
    $('entrySubtitle').textContent = reg ? 'Set up a private workspace for your scans.' : 'Sign in to analyse images and view your history.';
    $('entryAuthSubmit').textContent = reg ? 'Create account' : 'Sign in';
    $('entryPassword').autocomplete = reg ? 'new-password' : 'current-password';
    $('entryPassword').placeholder = reg ? 'Choose a strong password' : 'Your password';
    showAuthError('');
}
$('authTabLogin').onclick = () => setAuthMode('login');
$('authTabRegister').onclick = () => setAuthMode('register');

$('togglePassword').onclick = () => {
    const input = $('entryPassword');
    const show = input.type === 'password';
    input.type = show ? 'text' : 'password';
    $('togglePassword').setAttribute('aria-label', show ? 'Hide password' : 'Show password');
};

$('authForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    showAuthError('');
    const username = $('entryUsername').value.trim();
    const password = $('entryPassword').value;
    if (!username || !password) return showAuthError('Please enter your username and password.');

    const btn = $('entryAuthSubmit');
    setLoading(btn, true);
    try {
        let res;
        try {
            res = await fetch(`${API_URL}/${authMode}`, { method: 'POST', body: new URLSearchParams({ username, password }) });
        } catch (err) {
            throw new Error('Cannot reach the server. Please check that it is running.');
        }
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(detailOf(data, 'Please check your details and try again.'));

        $('entryPassword').value = '';
        if (authMode === 'login') {
            localStorage.setItem('token', data.access_token);
            checkAuth();
        } else {
            toast('Account created. Please sign in.', 'success');
            setAuthMode('login');
            $('entryPassword').focus();
        }
    } catch (err) {
        showAuthError(err.message);
    } finally {
        setLoading(btn, false);
    }
});

function logout(message) {
    localStorage.removeItem('token');
    try { if (message) sessionStorage.setItem('flash', message); } catch (e) { /* ignore */ }
    location.reload();
}
$('logoutBtn').onclick = () => logout();

$('deleteDataBtn').onclick = async () => {
    const { ok, value } = await askDialog({
        title: 'Delete your account and data?',
        message: 'This permanently deletes your account, every scan and all stored images. This cannot be undone. Enter your password to confirm.',
        confirmText: 'Delete everything', danger: true, password: true,
    });
    if (!ok) return;
    try {
        const res = await api('/account', { method: 'DELETE', body: new URLSearchParams({ password: value }) });
        if (res.ok) return logout('Your account and all stored data were deleted.');
        toast(detailOf(await res.json().catch(() => ({})), 'Could not delete the account.'), 'error');
    } catch (err) { toast(err.message, 'error'); }
};

function checkAuth() {
    const signedIn = !!localStorage.getItem('token');
    $('landingPage').classList.toggle('hidden', signedIn);
    $('mainDashboard').classList.toggle('hidden', !signedIn);
    if (signedIn) {
        showSection('dashboard');
        loadProfile();
        loadHistory();
    }
}

/* ==========================================================================
   Navigation
   ========================================================================== */
const TITLES = { dashboard: 'Analysis', progress: 'History', clinics: 'Find a dermatologist', advice: 'Skin profile', patients: 'Patients' };

function closeNav() { document.body.classList.remove('nav-open'); }
function showSection(id) {
    $$('.content-section').forEach(s => s.classList.toggle('hidden', s.id !== `section-${id}`));
    $$('.nav a').forEach(a => {
        const on = a.id === `nav-${id}`;
        a.classList.toggle('active', on);
        if (on) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current');
    });
    $('pageTitle').textContent = TITLES[id] || '';
    closeNav();
    window.scrollTo({ top: 0 });
    if (id === 'clinics') initMap();
    if (id === 'patients') loadPatients();
    if (id === 'advice') loadCare();
}
window.showSection = showSection;

$('menuBtn').onclick = () => document.body.classList.toggle('nav-open');
document.addEventListener('click', (e) => { if (e.target === document.body) closeNav(); });
document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape') return;
    closeNav();
    if (!$('chatWidget').classList.contains('hidden')) toggleChat(false);
});

/* ==========================================================================
   Profile and personalised tips
   ========================================================================== */
let userProfile = {};

const SKIN_TIPS = {
    Oily: 'Cleanse gently twice a day and use a lightweight, non-comedogenic moisturiser.',
    Dry: 'Use a creamy cleanser and a rich moisturiser, and avoid very hot water.',
    Combination: 'Treat oily and dry areas differently and keep textures light.',
    Sensitive: 'Choose fragrance-free products with few ingredients and patch-test new ones.',
    Normal: 'Keep it simple: a gentle cleanser, a moisturiser and daily sunscreen.',
};
const TRIGGER_TIPS = [
    [/dairy/, 'Dairy is a common trigger for excess oil and acne-prone skin in some people.'],
    [/spicy/, 'Spicy food can trigger flushing and rosacea flare-ups.'],
    [/oily|fried/, 'Very oily or fried food can contribute to excess sebum.'],
    [/junk|processed|outside/, 'Highly processed food tends to lack nutrients the skin needs for repair.'],
    [/sugar/, 'A lot of sugar may worsen acne and speed up skin ageing.'],
    [/salt/, 'High salt intake can cause puffiness and dehydration.'],
    [/coffee/, 'Lots of coffee can dehydrate the skin; balance it with water.'],
    [/low water/, 'Low water intake weakens the skin barrier. Aim for around 2 litres a day.'],
    [/soap/, 'Body soap on the face can strip the skin barrier. Use a gentle facial cleanser instead.'],
    [/perfume|scented/, 'Fragrances can cause contact dermatitis. Consider fragrance-free products.'],
    [/makeup/, 'Cleanse thoroughly at night to remove makeup and avoid clogged pores.'],
    [/recently changed/, 'A recent change in products can cause irritation. Introduce new ones slowly.'],
    [/sunscreen/, 'Daily sunscreen is the best protection against premature ageing and skin cancer. Keep it up!'],
    [/acne creams/, 'Medicated acne products can be drying, so moisturise well.'],
];

async function loadProfile() {
    try {
        const res = await api('/profile');
        if (res.ok) { userProfile = await res.json(); renderProfile(); }
    } catch (e) { /* toast already shown for auth errors */ }
}

function renderProfile() {
    const name = userProfile.username || 'Signed in';
    $('userName').textContent = name;
    $('userAvatar').textContent = (name[0] || 'U').toUpperCase();
    const isClinician = userProfile.role === 'clinician';
    $('nav-patients').classList.toggle('hidden', !isClinician);
    $('userRole').textContent = isClinician ? 'Clinician account' : 'Private workspace';
    $('careCard').classList.toggle('hidden', isClinician);

    const box = $('currentSkinTypeText');
    if (userProfile.skin_type) {
        box.innerHTML = `<span class="skin-badge">${esc(userProfile.skin_type)} skin</span><p class="muted">${esc(SKIN_TIPS[userProfile.skin_type] || '')}</p>`;
        $('skinTypeSelect').value = userProfile.skin_type;
    }
    const foods = (userProfile.food_habits || '').split(',').map(s => s.trim());
    $$('#foodHabitsGrid input').forEach(cb => { cb.checked = foods.includes(cb.value); });
    const products = (userProfile.products_used || '').split(',').map(s => s.trim());
    $$('#productsUsedGrid input').forEach(cb => { cb.checked = products.includes(cb.value); });
    renderTriggerTips();
}

function renderTriggerTips() {
    const text = `${userProfile.food_habits || ''} ${userProfile.products_used || ''}`.toLowerCase();
    const tips = TRIGGER_TIPS.filter(([re]) => re.test(text)).map(([, t]) => t);
    $('triggerSuggestions').innerHTML = tips.length
        ? tips.map(t => `<div class="trigger-alert">${esc(t)}</div>`).join('')
        : (text.trim() ? '<p class="muted">No common triggers detected from your selections.</p>' : '');
}

$('skinTypeSelect').onchange = async (e) => {
    const skin_type = e.target.value;
    if (!skin_type) return;
    try {
        const res = await api('/profile', { method: 'POST', json: { skin_type } });
        if (!res.ok) throw new Error('Could not save your skin type.');
        userProfile.skin_type = skin_type;
        renderProfile();
        toast('Skin type saved.', 'success');
    } catch (err) { toast(err.message, 'error'); }
};

$('triggerForm').onsubmit = async (e) => {
    e.preventDefault();
    const food_habits = $$('#foodHabitsGrid input:checked').map(cb => cb.value).join(', ');
    const products_used = $$('#productsUsedGrid input:checked').map(cb => cb.value).join(', ');
    try {
        const res = await api('/profile', { method: 'POST', json: { food_habits, products_used } });
        if (!res.ok) throw new Error('Could not save your selections.');
        userProfile.food_habits = food_habits;
        userProfile.products_used = products_used;
        renderTriggerTips();
        toast('Profile updated.', 'success');
    } catch (err) { toast(err.message, 'error'); }
};

/* ==========================================================================
   History
   ========================================================================== */
let blobUrls = [];

async function loadHistory() {
    try {
        const res = await api('/history');
        if (!res.ok) return;
        const history = await res.json();
        renderTimeline(history);
        $('kpiScans').textContent = history.length;
        $('kpiLast').textContent = history.length ? new Date(history[0].timestamp).toLocaleDateString([], { dateStyle: 'medium' }) : '-';
        $('kpiResult').textContent = history.length ? shortResult(history[0].prediction) : '-';
        loadLesions();
    } catch (e) { /* handled */ }
}

function renderTimeline(history) {
    const root = $('progressTimeline');
    blobUrls.forEach(u => URL.revokeObjectURL(u));
    blobUrls = [];
    root.innerHTML = '';
    if (!history.length) {
        root.innerHTML = `<div class="empty card"><span class="card-icon">${icon('image', 'i-lg')}</span>
            <h3>No scans yet</h3><p>Your analyses will appear here once you run one.</p>
            <button class="btn btn-primary" type="button" onclick="showSection('dashboard')">Start an analysis</button></div>`;
        return;
    }
    history.forEach(item => {
        const r = classify(item.prediction);
        const card = document.createElement('article');
        card.className = 'card scan-card';
        card.innerHTML = `
            <div class="scan-img">${item.image_url ? '<img alt="Uploaded lesion" loading="lazy">' : icon('image', 'i-lg')}</div>
            <div class="scan-body">
                <span class="badge" data-tone="${r.tone}">${esc(r.label)}</span>
                <h4>${esc(item.prediction)}</h4>
                <div class="scan-meta">
                    <span>${esc(new Date(item.timestamp).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' }))} &middot; ${esc(item.confidence)}%</span>
                    <button class="btn btn-danger-ghost btn-sm history-delete" type="button" aria-label="Delete this scan">${icon('trash')}Delete</button>
                </div>
            </div>`;
        root.appendChild(card);
        card.querySelector('.history-delete').onclick = () => deleteHistoryItem(item.id);
        const bodyEl = card.querySelector('.scan-body');
        const rb = reviewBlock(item.reviews);
        if (rb) bodyEl.appendChild(rb);
        bodyEl.appendChild(lesionAssign(item));
        const histImg = card.querySelector('img');
        if (histImg && item.image_url) bodyEl.appendChild(focusButton(histImg, `/history/${item.id}/heatmap`));
        const img = card.querySelector('img');
        if (img) loadProtectedImage(img, item.image_url);
    });
}

/** Images are private: fetch with the auth header and display through a blob URL. */
async function loadProtectedImage(img, url) {
    try {
        const res = await fetch(`${API_URL}${url}`, { headers: authHeaders() });
        if (!res.ok) return;
        const u = URL.createObjectURL(await res.blob());
        blobUrls.push(u);
        img.src = u;
    } catch (e) { /* keep placeholder */ }
}

async function deleteHistoryItem(id) {
    const { ok } = await askDialog({ title: 'Delete this scan?', message: 'The result and its stored image will be permanently removed.', confirmText: 'Delete', danger: true });
    if (!ok) return;
    try {
        const res = await api(`/history/${id}`, { method: 'DELETE' });
        if (!res.ok) throw new Error('Could not delete the scan.');
        toast('Scan deleted.', 'success');
        loadHistory();
    } catch (err) { toast(err.message, 'error'); }
}

/* ==========================================================================
   Analysis: ABCDE, upload, results
   ========================================================================== */
function updateABCDE() {
    const n = $$('.abcde-checkbox').filter(cb => cb.checked).length;
    const tone = n <= 1 ? 'low' : (n <= 3 ? 'medium' : 'high');
    $('abcdeRiskBarFill').style.width = `${(n / 5) * 100}%`;
    $('abcdeMeter').dataset.tone = tone;
    $('abcdeRiskBadge').dataset.tone = tone;
    $('abcdeRiskBadge').textContent = { low: 'Low', medium: 'Moderate', high: 'High' }[tone] + (n ? ` (${n}/5)` : '');
}
$$('.abcde-checkbox').forEach(cb => cb.addEventListener('change', updateABCDE));

const MAX_BYTES = 8 * 1024 * 1024;
const OK_TYPES = ['image/jpeg', 'image/png', 'image/webp'];
let currentFile = null;
let previewUrl = null;
let lastResultClass = '';
let lastScanId = null;

function setStep(step) {
    const order = ['upload', 'analyze', 'review'];
    const idx = order.indexOf(step);
    order.forEach((s, i) => {
        const el = $(`step${s[0].toUpperCase()}${s.slice(1)}`);
        el.classList.toggle('done', i < idx);
        el.classList.toggle('now', i === idx);
    });
}
const fmtSize = (b) => b > 1024 * 1024 ? `${(b / 1024 / 1024).toFixed(1)} MB` : `${Math.max(1, Math.round(b / 1024))} KB`;

function setFile(file) {
    if (!file) return;
    if (!OK_TYPES.includes(file.type)) return toast('Please choose a JPEG, PNG or WebP image.', 'error');
    if (file.size > MAX_BYTES) return toast('That image is larger than 8 MB. Please choose a smaller one.', 'error');
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    currentFile = file;
    previewUrl = URL.createObjectURL(file);
    $('imagePreview').src = previewUrl;
    $('fileName').textContent = file.name;
    $('fileSize').textContent = fmtSize(file.size);
    $('uploadBox').classList.add('hidden');
    $('imagePreviewContainer').classList.remove('hidden');
    $('results').classList.add('hidden');
    setStep('analyze');
    checkPhotoQuality(file);
}

/** Instant, model-free feedback ("too blurry", "too dark"...) before the user spends an analysis on a bad photo. */
async function checkPhotoQuality(file) {
    const note = $('qualityNote');
    note.classList.add('hidden');
    try {
        const body = new FormData();
        body.append('file', file);
        const res = await api('/quality', { method: 'POST', body });
        if (!res.ok || currentFile !== file) return;
        const q = await res.json();
        note.classList.toggle('warn', !q.ok);
        $('qualityIcon').firstElementChild.setAttribute('href', q.ok ? '#i-check' : '#i-camera');
        $('qualityText').textContent = q.ok ? 'Photo quality looks good.' : q.tips.join(' ');
        note.classList.remove('hidden');
    } catch (e) { /* quality hints are optional */ }
}

function resetAnalysis() {
    currentFile = null;
    $('fileInput').value = '';
    $('imagePreview').removeAttribute('src');
    $('imagePreviewContainer').classList.add('hidden');
    $('qualityNote').classList.add('hidden');
    $('uploadBox').classList.remove('hidden');
    $('results').classList.add('hidden');
    setStep('upload');
}

const dropZone = $('dropZone');
dropZone.onclick = () => $('fileInput').click();
dropZone.onkeydown = (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); $('fileInput').click(); } };
$('fileInput').onchange = (e) => setFile(e.target.files[0]);
['dragenter', 'dragover'].forEach(ev => dropZone.addEventListener(ev, (e) => { e.preventDefault(); dropZone.classList.add('dragover'); }));
['dragleave', 'drop'].forEach(ev => dropZone.addEventListener(ev, (e) => { e.preventDefault(); dropZone.classList.remove('dragover'); }));
dropZone.addEventListener('drop', (e) => setFile(e.dataTransfer.files[0]));
$('removeFileBtn').onclick = resetAnalysis;
$('nextAnalysisBtn').onclick = () => { resetAnalysis(); window.scrollTo({ top: 0, behavior: 'smooth' }); };
$('printBtn').onclick = () => window.print();

$('analyzeBtn').onclick = async () => {
    if (!currentFile) return;
    const btn = $('analyzeBtn');
    setLoading(btn, true);
    setStep('analyze');
    const body = new FormData();
    body.append('file', currentFile);
    try {
        const lesionId = await resolveLesionChoice();
        if (lesionId) body.append('lesion_id', lesionId);
        if ($('ctxAge').value) body.append('age', $('ctxAge').value);
        if ($('ctxSex').value) body.append('sex', $('ctxSex').value);
        if ($('ctxSite').value) body.append('site', $('ctxSite').value);
        const res = await api('/predict', { method: 'POST', body });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(detailOf(data, 'The analysis failed. Please try another image.'));
        displayResults(data);
        loadHistory();
    } catch (err) {
        if (err.message !== 'Session expired') toast(err.message, 'error');
    } finally {
        setLoading(btn, false);
    }
};

function populateList(id, items) {
    const el = $(id);
    el.innerHTML = '';
    (items || []).forEach(text => {
        const li = document.createElement('li');
        li.textContent = text;
        el.appendChild(li);
    });
}

function displayResults(data) {
    const r = classify(data.prediction);
    $('resultCard').dataset.tone = r.tone;
    $('confRow').dataset.tone = r.tone;
    $('resultBadge').dataset.tone = r.tone;
    $('resultBadge').textContent = r.label;
    $('predictionText').textContent = data.prediction;
    $('confidenceLabel').textContent = data.escalated ? 'Melanoma probability' : 'Model confidence';
    $('confidenceText').textContent = `${data.confidence}%`;
    $('resultImage').src = previewUrl || '';
    lastScanId = data.scan_id || null;
    $('resultImage').dataset.showing = '';
    $('focusBtn').classList.toggle('hidden', !lastScanId);
    $('focusNote').classList.add('hidden');
    $('focusBtn').lastChild.textContent = 'Show model focus';
    $('descriptionText').textContent = data.description || '';
    $('resultDisclaimer').textContent = data.disclaimer || '';

    const ref = data.referral || {};
    $('referralCard').dataset.level = ref.level || 'routine';
    $('referralTitle').textContent = ref.title || '';
    $('referralMessage').textContent = ref.message || '';
    $('referralTime').textContent = ref.timeframe || '';
    $('referralTime').dataset.tone = { priority: 'high', soon: 'medium', routine: 'low' }[ref.level] || '';
    $('referralAdvice').textContent = ref.advice || '';
    $('contextNote').classList.toggle('hidden', !data.context_used);
    const q = data.quality;
    $('resultQuality').classList.toggle('hidden', !(q && !q.ok));
    $('resultQualityText').textContent = q && !q.ok ? q.tips.join(' ') : '';

    $('resultNote').classList.toggle('hidden', !data.note);
    $('resultNoteText').textContent = data.note || '';

    const fill = $('confidenceFill');
    fill.style.width = '0';
    requestAnimationFrame(() => requestAnimationFrame(() => { fill.style.width = `${Math.min(data.confidence, 100)}%`; }));

    const probs = Object.entries(data.probabilities || {}).sort((a, b) => b[1] - a[1]);
    $('probabilityList').innerHTML = probs.map(([name, pct]) => `
        <div class="prob-row${name === data.top_prediction ? ' top' : ''}">
            <span class="name" title="${esc(name)}">${esc(name)}</span>
            <div class="meter"><span style="width:${Math.min(Number(pct), 100)}%"></span></div>
            <span class="val">${esc(pct)}%</span>
        </div>`).join('');

    populateList('symptomsList', data.symptoms);
    populateList('recommendationsList', data.recommendations);
    populateList('dosList', data.dos);
    populateList('dontsList', data.donts);
    $('nearbyOut').innerHTML = '';

    lastResultClass = data.escalated ? 'Melanoma' : (RISK[data.prediction] ? data.prediction : '');
    $('results').classList.remove('hidden');
    setStep('review');
    $('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/* ==========================================================================
   Location helpers, clinic finder and map
   ========================================================================== */
function getPosition() {
    return new Promise((resolve, reject) => {
        if (!navigator.geolocation) return reject(new Error('Location is not supported by this browser.'));
        navigator.geolocation.getCurrentPosition(
            (p) => resolve(p.coords),
            (err) => reject(new Error(err.code === 1 ? 'Location permission was denied. You can search by city instead.' : 'Could not determine your location.')),
            { timeout: 12000 });
    });
}

async function geocode(query) {
    const res = await fetchExternal(`https://nominatim.openstreetmap.org/search?format=json&limit=1&q=${encodeURIComponent(query)}`);
    if (!res.ok) throw new Error('The location service is unavailable right now.');
    const data = await res.json();
    if (!data.length) return null;
    return { lat: Number(data[0].lat), lon: Number(data[0].lon), name: data[0].display_name };
}

function distanceKm(lat1, lon1, lat2, lon2) {
    const rad = (d) => d * Math.PI / 180;
    const a = Math.sin(rad(lat2 - lat1) / 2) ** 2 + Math.cos(rad(lat1)) * Math.cos(rad(lat2)) * Math.sin(rad(lon2 - lon1) / 2) ** 2;
    return 6371 * 2 * Math.asin(Math.sqrt(a));
}

async function findClinics(lat, lon, radius = 10000) {
    if (![lat, lon].every(Number.isFinite)) throw new Error('Invalid location.');
    const around = `(around:${radius},${lat},${lon})`;
    const query = `[out:json][timeout:20];(
        node["healthcare"="dermatology"]${around};way["healthcare"="dermatology"]${around};
        node["healthcare:speciality"="dermatology"]${around};node["speciality"="dermatology"]${around};
        node["name"~"Skin|Dermat",i]${around};way["name"~"Skin|Dermat",i]${around};
    );out center 40;`;
    const res = await fetchExternal('https://overpass-api.de/api/interpreter?data=' + encodeURIComponent(query), 25000);
    if (!res.ok) throw new Error('The clinic directory is unavailable right now.');
    const data = await res.json();
    return (data.elements || [])
        .map(el => {
            const clat = el.lat ?? el.center?.lat, clon = el.lon ?? el.center?.lon, t = el.tags || {};
            if (!Number.isFinite(clat) || !Number.isFinite(clon)) return null;
            return {
                name: t.name || 'Dermatology clinic', lat: clat, lon: clon,
                phone: t.phone || t['contact:phone'] || '',
                address: [t['addr:housenumber'], t['addr:street'], t['addr:city']].filter(Boolean).join(' '),
                km: distanceKm(lat, lon, clat, clon),
            };
        })
        .filter(Boolean)
        .sort((a, b) => a.km - b.km);
}
const directionsUrl = (c) => `https://www.google.com/maps/dir/?api=1&destination=${encodeURIComponent(c.lat + ',' + c.lon)}`;

/* ----- "Find care near you" inside the results ----- */
function renderNearby(clinics, where) {
    const out = $('nearbyOut');
    if (!clinics.length) {
        out.innerHTML = `<div class="state-msg">No dermatology listings found near ${esc(where)}. Try the
            <a href="#clinics" onclick="showSection('clinics'); return false;">clinic finder</a> with a wider area.</div>`;
        return;
    }
    out.innerHTML = clinics.slice(0, 3).map(c => `
        <div class="clinic-hit">
            <div><strong>${esc(c.name)}</strong><span class="fine">${esc(c.km.toFixed(1))} km away${c.address ? ' &middot; ' + esc(c.address) : ''}</span></div>
            <a class="btn btn-secondary btn-sm" href="${directionsUrl(c)}" target="_blank" rel="noopener noreferrer">Directions</a>
        </div>`).join('');
}

async function nearbyFrom(btn, getPoint, label) {
    setLoading(btn, true);
    $('nearbyOut').innerHTML = '<div class="state-msg">Searching...</div>';
    try {
        const pt = await getPoint();
        if (!pt) { $('nearbyOut').innerHTML = `<div class="state-msg">Could not find "${esc(label())}". Try a different name.</div>`; return; }
        renderNearby(await findClinics(pt.lat, pt.lon, 20000), label());
    } catch (err) {
        $('nearbyOut').innerHTML = `<div class="state-msg">${esc(err.message)}</div>`;
    } finally { setLoading(btn, false); }
}
$('nearbyBtn').onclick = () => nearbyFrom($('nearbyBtn'), async () => { const c = await getPosition(); return { lat: c.latitude, lon: c.longitude }; }, () => 'you');
$('nearbyCityBtn').onclick = () => {
    const city = $('nearbyCity').value.trim();
    if (!city) return $('nearbyCity').focus();
    nearbyFrom($('nearbyCityBtn'), () => geocode(city), () => city);
};
$('nearbyCity').onkeydown = (e) => { if (e.key === 'Enter') $('nearbyCityBtn').click(); };

/* ----- Clinic finder page ----- */
let map = null, markers = null;

function initMap() {
    if (!map) {
        map = L.map('map').setView([20.5937, 78.9629], 4);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { attribution: '&copy; OpenStreetMap contributors', maxZoom: 19 }).addTo(map);
        markers = L.layerGroup().addTo(map);
    }
    setTimeout(() => map.invalidateSize(), 60);
}

async function searchClinicsAt(lat, lon, label, youAreHere = false) {
    const list = $('clinicList');
    list.innerHTML = '<div class="state-msg">Searching for dermatology clinics...</div>';
    map.setView([lat, lon], 13);
    markers.clearLayers();
    if (youAreHere) L.marker([lat, lon]).addTo(markers).bindPopup('You are here');
    try {
        const clinics = await findClinics(lat, lon, 10000);
        if (!clinics.length) {
            list.innerHTML = `<div class="state-msg">No dermatology listings found within 10 km of ${esc(label)}. Try a nearby larger city.</div>`;
            return;
        }
        list.innerHTML = '';
        clinics.slice(0, 15).forEach(c => {
            L.marker([c.lat, c.lon]).addTo(markers).bindPopup(`<strong>${esc(c.name)}</strong>`);
            const card = document.createElement('div');
            card.className = 'clinic-card';
            card.innerHTML = `<strong>${esc(c.name)}</strong>
                <p>${esc(c.km.toFixed(1))} km away${c.address ? ' &middot; ' + esc(c.address) : ''}${c.phone ? '<br>' + esc(c.phone) : ''}</p>
                <a class="btn btn-secondary btn-sm" href="${directionsUrl(c)}" target="_blank" rel="noopener noreferrer">${icon('pin')}Directions</a>`;
            list.appendChild(card);
        });
    } catch (err) {
        list.innerHTML = `<div class="state-msg">${esc(err.message)} Try again, or search Google Maps for a dermatologist near you.</div>`;
    }
}

$('searchLocationBtn').onclick = async () => {
    const q = $('manualLocationInput').value.trim();
    if (!q) return $('manualLocationInput').focus();
    initMap();
    const btn = $('searchLocationBtn');
    setLoading(btn, true);
    try {
        const pt = await geocode(q);
        if (!pt) { $('clinicList').innerHTML = `<div class="state-msg">Could not find "${esc(q)}". Try a different name.</div>`; return; }
        await searchClinicsAt(pt.lat, pt.lon, q);
    } catch (err) { $('clinicList').innerHTML = `<div class="state-msg">${esc(err.message)}</div>`; }
    finally { setLoading(btn, false); }
};
$('manualLocationInput').onkeydown = (e) => { if (e.key === 'Enter') $('searchLocationBtn').click(); };

$('myLocationBtn').onclick = async () => {
    initMap();
    const btn = $('myLocationBtn');
    setLoading(btn, true);
    try {
        const c = await getPosition();
        await searchClinicsAt(c.latitude, c.longitude, 'your location', true);
    } catch (err) { toast(err.message, 'warn'); }
    finally { setLoading(btn, false); }
};

/* ==========================================================================
   Weather advice (Open-Meteo, no API key; requested only on click)
   ========================================================================== */
function weatherAdvice(temp, humidity, uv) {
    const tips = [];
    if (uv !== null && uv >= 6) tips.push('High UV index: apply SPF 50+, wear a hat and avoid midday sun.');
    else if (uv !== null && uv >= 3) tips.push('Moderate UV index: apply SPF 30+ on exposed skin.');
    if (humidity !== null && humidity < 30) tips.push('Dry air: use a rich moisturiser and avoid very hot showers.');
    if (humidity !== null && humidity > 75) tips.push('High humidity: prefer lightweight, non-comedogenic products.');
    if (temp !== null && temp > 32) tips.push('Hot weather: stay hydrated and rinse off sweat promptly.');
    if (temp !== null && temp < 5) tips.push('Cold weather: protect your skin from wind and use a barrier cream.');
    return tips.length ? tips.join(' ') : 'Conditions look mild. Keep up your usual routine and daily SPF.';
}

$('weatherBtn').onclick = async () => {
    const btn = $('weatherBtn');
    setLoading(btn, true);
    try {
        const c = await getPosition();
        const res = await fetchExternal(`https://api.open-meteo.com/v1/forecast?latitude=${c.latitude}&longitude=${c.longitude}&current=temperature_2m,relative_humidity_2m,uv_index`);
        if (!res.ok) throw new Error('Live weather is unavailable right now.');
        const cur = (await res.json()).current || {};
        const temp = cur.temperature_2m ?? null, humidity = cur.relative_humidity_2m ?? null, uv = cur.uv_index ?? null;
        $('weatherContent').innerHTML = `
            <div class="weather">
                <div class="temp">${temp !== null ? esc(Math.round(temp)) + '&deg;C' : '--'}</div>
                <div class="cond">${humidity !== null ? 'Humidity ' + esc(Math.round(humidity)) + '%' : ''}${uv !== null ? '<br>UV index ' + esc(uv) : ''}</div>
            </div>
            <p class="advice-text">${esc(weatherAdvice(temp, humidity, uv))}</p>`;
    } catch (err) { toast(err.message, 'warn'); }
    finally { setLoading(btn, false); }
};

/* ==========================================================================
   Assistant chat
   ========================================================================== */
function toggleChat(force) {
    const w = $('chatWidget');
    const open = typeof force === 'boolean' ? force : w.classList.contains('hidden');
    w.classList.toggle('hidden', !open);
    $('openChatBtn').setAttribute('aria-expanded', String(open));
    if (open) $('chatInput').focus();
}
$('openChatBtn').onclick = () => toggleChat();
$('closeChatBtn').onclick = () => toggleChat(false);

function addMessage(text, who) {
    const box = $('chatMessages');
    const div = document.createElement('div');
    div.className = `msg ${who}`;
    div.textContent = text;              // textContent: replies are never parsed as HTML
    box.appendChild(div);
    box.scrollTop = box.scrollHeight;
    return div;
}

async function sendChat(message) {
    message = message.trim();
    if (!message) return;
    $('chatInput').value = '';
    $('chatChips').classList.add('hidden');
    addMessage(message, 'user');
    const pending = addMessage('...', 'bot typing');
    $('sendChatBtn').disabled = true;
    try {
        const res = await api('/chat', { method: 'POST', json: { message, context: lastResultClass } });
        const data = await res.json().catch(() => ({}));
        pending.classList.remove('typing');
        pending.textContent = res.ok ? data.response : detailOf(data, 'Sorry, I could not answer that.');
    } catch (err) {
        pending.classList.remove('typing');
        pending.textContent = err.message;
    } finally { $('sendChatBtn').disabled = false; }
}
$('chatForm').onsubmit = (e) => { e.preventDefault(); sendChat($('chatInput').value); };
$$('#chatChips .chip').forEach(chip => chip.onclick = () => sendChat(chip.textContent));

/* ==========================================================================
   Care team: tracked spots (lesions), sharing, access log, clinician workspace
   ========================================================================== */
const BODY_SITES = ['abdomen', 'acral', 'back', 'chest', 'ear', 'face', 'foot', 'genital', 'hand',
                    'lower extremity', 'neck', 'scalp', 'trunk', 'upper extremity', 'unknown'];
const REVIEW_LABELS = ['Actinic Keratoses', 'Basal Cell Carcinoma', 'Benign Keratosis', 'Dermatofibroma',
                       'Melanoma', 'Nevi', 'Vascular', 'Other / not a skin lesion'];
let lesionsCache = [];

function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
}
function addOptions(select, options) {
    select.innerHTML = '';
    options.forEach(([value, label]) => { const o = el('option', '', label); o.value = value; select.appendChild(o); });
}
const fmtDate = (ts) => new Date(ts).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' });

/* ----- optional details on the analysis page ----- */
addOptions($('ctxSite'), [['', 'Not specified'], ...BODY_SITES.map(x => [x, x])]);

function renderLesionPicker() {
    const sel = $('ctxLesion');
    const keep = sel.value;
    addOptions(sel, [['', 'Not tracked'], ...lesionsCache.map(l => [String(l.id), l.name]), ['new', '+ New spot...']]);
    if ([...sel.options].some(o => o.value === keep)) sel.value = keep;
}
$('ctxLesion').onchange = () => $('ctxNewLesionWrap').classList.toggle('hidden', $('ctxLesion').value !== 'new');

async function resolveLesionChoice() {
    const choice = $('ctxLesion').value;
    if (choice !== 'new') return choice || null;
    const name = $('ctxNewLesion').value.trim();
    if (!name) throw new Error('Enter a name for the new spot, or choose "Not tracked".');
    const res = await api('/lesions', { method: 'POST', json: { name, body_site: $('ctxSite').value || null } });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(detailOf(data, 'Could not create the spot.'));
    await loadLesions();
    $('ctxLesion').value = String(data.id);
    $('ctxNewLesionWrap').classList.add('hidden');
    $('ctxNewLesion').value = '';
    return data.id;
}

/* ----- history: reviews + assigning scans to a spot ----- */
function reviewBlock(reviews) {
    if (!reviews || !reviews.length) return null;
    const box = el('div', 'review-box');
    reviews.forEach(r => {
        box.appendChild(el('strong', '', `Reviewed by ${r.clinician}: ${r.label}`));
        if (r.note) box.appendChild(el('span', '', r.note));
    });
    return box;
}

function lesionAssign(item) {
    const wrap = el('div', 'field');
    wrap.style.margin = '.5rem 0 0';
    const sel = el('select', 'input input-sm');
    sel.setAttribute('aria-label', 'Tracked spot for this scan');
    addOptions(sel, [['', 'Not tracked'], ...lesionsCache.map(l => [String(l.id), l.name])]);
    sel.value = item.lesion_id ? String(item.lesion_id) : '';
    sel.onchange = async () => {
        const res = await api(`/history/${item.id}/lesion`, { method: 'POST', json: { lesion_id: sel.value ? Number(sel.value) : null } });
        if (res.ok) { toast('Saved.', 'success', 2000); loadLesions(); } else toast('Could not update the scan.', 'error');
    };
    wrap.appendChild(sel);
    return wrap;
}

/* ----- "model focus" heat-maps (Grad-CAM) ----- */
async function fetchBlobUrl(url) {
    const res = await fetch(`${API_URL}${url}`, { headers: authHeaders() });
    if (!res.ok) throw new Error(res.status === 503 ? 'The model is not loaded on the server.' : 'The heat-map is not available.');
    const u = URL.createObjectURL(await res.blob());
    blobUrls.push(u);
    return u;
}

/** Swap an <img> between the original photo and its heat-map. `onState(showingHeatmap)` updates labels. */
async function toggleHeatmap(img, heatUrl, onState) {
    if (img.dataset.showing === 'heat') {
        img.src = img.dataset.orig;
        img.dataset.showing = '';
        return onState(false);
    }
    try {
        if (!img.dataset.heatSrc || img.dataset.heatFor !== heatUrl) {
            img.dataset.heatSrc = await fetchBlobUrl(heatUrl);
            img.dataset.heatFor = heatUrl;
        }
        if (!img.dataset.showing) img.dataset.orig = img.src;
        img.src = img.dataset.heatSrc;
        img.dataset.showing = 'heat';
        onState(true);
    } catch (err) { toast(err.message, 'error'); }
}

function focusButton(img, heatUrl) {
    const btn = el('button', 'btn btn-ghost btn-sm');
    btn.type = 'button';
    btn.innerHTML = icon('eye');
    const label = document.createTextNode('Model focus');
    btn.appendChild(label);
    btn.onclick = () => toggleHeatmap(img, heatUrl, (on) => { label.textContent = on ? 'Show original' : 'Model focus'; });
    return btn;
}

$('focusBtn').onclick = () => {
    if (!lastScanId) return;
    toggleHeatmap($('resultImage'), `/history/${lastScanId}/heatmap`, (on) => {
        $('focusBtn').lastChild.textContent = on ? 'Show original' : 'Show model focus';
        $('focusNote').classList.toggle('hidden', !on);
    });
};

/* ----- tracked spots ----- */
async function loadLesions() {
    try {
        const res = await api('/lesions');
        if (!res.ok) return;
        lesionsCache = await res.json();
        renderLesionPicker();
        renderLesionList();
    } catch (e) { /* handled */ }
}

function setHistoryTab(tab) {
    const lesions = tab === 'lesions';
    $('histTabScans').setAttribute('aria-selected', String(!lesions));
    $('histTabLesions').setAttribute('aria-selected', String(lesions));
    $('progressTimeline').classList.toggle('hidden', lesions);
    $('lesionList').classList.toggle('hidden', !lesions);
    $('lesionDetail').classList.add('hidden');
}
$('histTabScans').onclick = () => setHistoryTab('scans');
$('histTabLesions').onclick = () => setHistoryTab('lesions');

function renderLesionList() {
    const root = $('lesionList');
    root.innerHTML = '';
    if (!lesionsCache.length) {
        root.innerHTML = `<div class="empty card"><span class="card-icon">${icon('pin', 'i-lg')}</span>
            <h3>No tracked spots yet</h3><p>When you analyse a photo, open "Add details" and choose "+ New spot" to follow it over time.</p></div>`;
        return;
    }
    lesionsCache.forEach(l => {
        const alert = l.change && l.change.alert;
        const card = el('article', 'card');
        card.innerHTML = `<div class="card-head"><div><h3>${esc(l.name)}</h3><p>${esc(l.body_site || 'No body site')} &middot; ${esc(l.n_scans)} scan${l.n_scans === 1 ? '' : 's'}</p></div>
            <span class="badge" data-tone="${alert ? 'high' : 'low'}">${alert ? 'Needs attention' : 'No flag'}</span></div>
            <p class="fine">${l.last_scan ? 'Last scan ' + esc(fmtDate(l.last_scan)) : 'No scans yet'}</p>
            <button class="btn btn-secondary btn-sm" type="button" style="margin-top:.75rem">Open timeline</button>`;
        card.querySelector('button').onclick = () => openLesion(l.id);
        root.appendChild(card);
    });
}

async function openLesion(id) {
    const res = await api(`/lesions/${id}`);
    if (!res.ok) return toast('Could not open that spot.', 'error');
    const l = await res.json();
    const box = $('lesionDetail');
    $('lesionList').classList.add('hidden');
    box.classList.remove('hidden');
    box.innerHTML = '';

    const head = el('div', 'card');
    head.innerHTML = `<div class="card-head"><div><h2>${esc(l.name)}</h2><p>${esc(l.body_site || 'No body site')} &middot; ${esc(l.n_scans)} scan${l.n_scans === 1 ? '' : 's'}</p></div>
        <div style="display:flex;gap:.5rem"><button class="btn btn-secondary btn-sm" data-act="back" type="button">Back</button>
        <button class="btn btn-danger-ghost btn-sm" data-act="delete" type="button">${icon('trash')}Stop tracking</button></div></div>`;
    box.appendChild(head);

    if (l.change) {
        const d = l.change.melanoma_prob_delta;
        const text = d === null ? 'Not enough data to compare.'
            : `Melanoma probability moved ${d >= 0 ? 'up' : 'down'} by ${Math.abs(d)} percentage points over ${l.change.days} day${l.change.days === 1 ? '' : 's'} (${l.change.first_prediction} to ${l.change.last_prediction}).`;
        const callout = el('div', 'callout' + (l.change.alert ? ' warn' : ''));
        callout.appendChild(el('span', '', ''));
        const msg = el('div', '', (l.change.alert ? 'Worth showing to a doctor. ' : '') + text + ' ' + l.note);
        callout.appendChild(msg);
        callout.style.marginTop = '1rem';
        head.appendChild(callout);
    }

    const scans = l.scans || [];
    if (scans.length >= 2) {
        const cmp = el('div', 'lesion-compare');
        [['First photo', scans[0]], ['Latest photo', scans[scans.length - 1]]].forEach(([cap, sc]) => {
            const fig = el('figure');
            const img = el('img');
            img.alt = cap;
            if (sc.image_url) loadProtectedImage(img, sc.image_url);
            fig.appendChild(img);
            fig.appendChild(el('figcaption', '', `${cap} - ${fmtDate(sc.timestamp)} - ${sc.prediction}`));
            cmp.appendChild(fig);
        });
        head.appendChild(cmp);
    }

    const list = el('div', 'timeline');
    list.style.marginTop = '1rem';
    scans.slice().reverse().forEach(sc => {
        const r = classify(sc.prediction);
        const card = el('article', 'card');
        card.innerHTML = `<span class="badge" data-tone="${r.tone}">${esc(r.label)}</span><h4 style="margin-top:.5rem">${esc(sc.prediction)}</h4>
            <p class="fine">${esc(fmtDate(sc.timestamp))} &middot; ${esc(sc.confidence)}% &middot; melanoma ${esc(sc.melanoma_prob ?? '-')}%</p>`;
        const rb = reviewBlock(sc.reviews);
        if (rb) card.appendChild(rb);
        list.appendChild(card);
    });
    box.appendChild(list);

    head.querySelector('[data-act="back"]').onclick = () => { box.classList.add('hidden'); $('lesionList').classList.remove('hidden'); };
    head.querySelector('[data-act="delete"]').onclick = async () => {
        const { ok } = await askDialog({ title: 'Stop tracking this spot?', message: 'Its scans stay in your history; only the grouping is removed.', confirmText: 'Stop tracking', danger: true });
        if (!ok) return;
        const r = await api(`/lesions/${id}`, { method: 'DELETE' });
        if (r.ok) { toast('Spot removed.', 'success'); box.classList.add('hidden'); loadHistory(); setHistoryTab('lesions'); }
    };
}

/* ----- care team (patient): sharing + access log ----- */
async function loadCare() {
    if (userProfile.role === 'clinician') return;
    try {
        const res = await api('/sharing');
        if (!res.ok) return;
        const shares = await res.json();
        const list = $('shareList');
        list.innerHTML = '';
        shares.forEach(sh => {
            const li = el('li', 'share-item');
            li.appendChild(el('span', '', `${sh.clinician} can view your scans (since ${new Date(sh.created).toLocaleDateString()})`));
            const btn = el('button', 'btn btn-danger-ghost btn-sm', 'Revoke');
            btn.type = 'button';
            btn.onclick = async () => {
                const r = await api(`/sharing/${sh.id}`, { method: 'DELETE' });
                if (r.ok) { toast('Access revoked.', 'success'); loadCare(); } else toast('Could not revoke.', 'error');
            };
            li.appendChild(btn);
            list.appendChild(li);
        });
        if (!shares.length) list.appendChild(el('li', 'fine', 'You are not sharing your scans with anyone.'));
    } catch (e) { /* handled */ }
}

$('careShareBtn').onclick = async () => {
    const name = $('careUsername').value.trim();
    if (!name) return $('careUsername').focus();
    const res = await api('/sharing', { method: 'POST', json: { clinician_username: name } });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) return toast(detailOf(data, 'Could not share.'), 'error');
    $('careUsername').value = '';
    toast(`Shared with ${data.clinician}.`, 'success');
    loadCare();
};

$('auditBtn').onclick = async () => {
    const list = $('auditList');
    const res = await api('/audit');
    if (!res.ok) return;
    const rows = await res.json();
    list.innerHTML = '';
    const verbs = { share_granted: 'was given access', share_revoked: 'access was removed', viewed_history: 'viewed your scan list', viewed_image: 'viewed a photo', reviewed_scan: 'reviewed a scan' };
    rows.forEach(r => {
        const li = el('li', 'audit-item');
        li.appendChild(el('strong', '', r.who));
        li.appendChild(el('span', '', verbs[r.action] || r.action));
        if (r.detail) li.appendChild(el('span', 'fine', r.detail));
        li.appendChild(el('span', 'when', fmtDate(r.when)));
        list.appendChild(li);
    });
    if (!rows.length) list.appendChild(el('li', 'fine', 'No one has accessed your data yet.'));
    list.classList.remove('hidden');
};

/* ----- clinician workspace ----- */
async function loadPatients() {
    const root = $('patientList');
    $('patientDetail').classList.add('hidden');
    root.classList.remove('hidden');
    root.innerHTML = '';
    const res = await api('/clinician/patients');
    if (res.status === 403) { root.innerHTML = '<p class="muted">This page is for clinician accounts.</p>'; return; }
    if (!res.ok) return;
    const patients = await res.json();
    if (!patients.length) {
        root.innerHTML = `<div class="empty card"><span class="card-icon">${icon('user', 'i-lg')}</span><h3>No shared patients yet</h3>
            <p>Patients appear here after they share their scans with your username.</p></div>`;
        return;
    }
    patients.forEach(p => {
        const card = el('article', 'card');
        card.innerHTML = `<div class="card-head"><div><h3>${esc(p.username)}</h3><p>${esc(p.n_scans)} scan${p.n_scans === 1 ? '' : 's'} &middot; shared ${esc(new Date(p.shared_since).toLocaleDateString())}</p></div>
            <span class="badge" data-tone="${p.needs_review ? 'medium' : 'low'}">${esc(p.needs_review)} to review</span></div>
            <p class="fine">${p.last_scan ? 'Last scan ' + esc(fmtDate(p.last_scan)) + ' - ' + esc(p.last_prediction) : 'No scans yet'}</p>
            <button class="btn btn-secondary btn-sm" type="button" style="margin-top:.75rem">Open</button>`;
        card.querySelector('button').onclick = () => openPatient(p.patient_id);
        root.appendChild(card);
    });
}

async function openPatient(pid) {
    const res = await api(`/clinician/patients/${pid}/history`);
    if (!res.ok) return toast('Could not open this patient (access may have been revoked).', 'error');
    const data = await res.json();
    const box = $('patientDetail');
    $('patientList').classList.add('hidden');
    box.classList.remove('hidden');
    box.innerHTML = '';
    const head = el('div', 'card');
    head.innerHTML = `<div class="card-head"><div><h2>${esc(data.patient)}</h2><p>${esc(data.scans.length)} scans${data.skin_type ? ' &middot; ' + esc(data.skin_type) + ' skin' : ''}. This view is recorded in the patient's access log.</p></div>
        <button class="btn btn-secondary btn-sm" data-act="back" type="button">Back</button></div>`;
    box.appendChild(head);
    const grid = el('div', 'patient-scan-grid');
    data.scans.forEach(sc => {
        const r = classify(sc.prediction);
        const card = el('article', 'card scan-card');
        card.innerHTML = `<div class="scan-img">${sc.image_url ? '<img alt="Patient photo" loading="lazy">' : icon('image', 'i-lg')}</div>
            <div class="scan-body"><span class="badge" data-tone="${r.tone}">${esc(r.label)}</span>
            <h4>${esc(sc.prediction)}</h4>
            <p class="fine">${esc(fmtDate(sc.timestamp))} &middot; model ${esc(sc.confidence)}% &middot; melanoma ${esc(sc.melanoma_prob ?? '-')}%</p>
            <p class="fine">${esc([sc.age ? sc.age + ' y' : '', sc.sex || '', sc.site || ''].filter(Boolean).join(' / ') || 'No context given')}</p></div>`;
        const img = card.querySelector('img');
        if (img) loadProtectedImage(img, sc.image_url);
        const body = card.querySelector('.scan-body');
        if (img && sc.image_url) body.appendChild(focusButton(img, sc.image_url.replace(/\/image$/, '/heatmap')));
        const rb = reviewBlock(sc.reviews);
        if (rb) body.appendChild(rb);

        const form = el('div', 'review-form');
        const sel = el('select', 'input input-sm');
        addOptions(sel, REVIEW_LABELS.map(x => [x, x]));
        const mine = sc.reviews && sc.reviews.length ? sc.reviews[sc.reviews.length - 1] : null;
        if (mine) sel.value = mine.label; else sel.value = REVIEW_LABELS.includes(sc.prediction) ? sc.prediction : 'Melanoma';
        const note = el('textarea', 'input input-sm');
        note.placeholder = 'Note (e.g. biopsy result) - visible to the patient';
        note.maxLength = 1000;
        const save = el('button', 'btn btn-primary btn-sm', mine ? 'Update review' : 'Save review');
        save.type = 'button';
        save.onclick = async () => {
            const r2 = await api(`/clinician/scans/${sc.id}/review`, { method: 'POST', json: { label: sel.value, note: note.value || null } });
            if (r2.ok) { toast('Review saved.', 'success'); openPatient(pid); } else toast(detailOf(await r2.json().catch(() => ({})), 'Could not save the review.'), 'error');
        };
        form.append(sel, note, save);
        body.appendChild(form);
        grid.appendChild(card);
    });
    box.appendChild(grid);
    head.querySelector('[data-act="back"]').onclick = loadPatients;
}

/* ----- is the analysis server reachable? (matters when the front end is hosted separately, e.g. on Vercel) ----- */
async function checkServer() {
    const banner = $('serverBanner');
    try {
        const res = await fetchExternal(`${API_URL}/health`, 8000);
        if (!res.ok) throw new Error('unhealthy');
        const health = await res.json();
        banner.classList.toggle('hidden', health.status === 'ok');
    } catch (e) {
        $('serverBannerText').textContent = window.SKINSENSE_API_URL
            ? 'The analysis server is not reachable right now, so you cannot sign in or analyse photos yet.'
            : 'This page is running without an analysis server, so you cannot sign in or analyse photos. The server address has not been configured.';
        banner.classList.remove('hidden');
    }
}

/* ==========================================================================
   Boot
   ========================================================================== */
applyTheme(document.documentElement.dataset.theme || 'light');
setAuthMode('login');
updateABCDE();
try {
    const flash = sessionStorage.getItem('flash');
    if (flash) { sessionStorage.removeItem('flash'); toast(flash, 'info'); }
} catch (e) { /* ignore */ }
checkAuth();
checkServer();
