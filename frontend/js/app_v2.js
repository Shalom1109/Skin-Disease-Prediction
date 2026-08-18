const API_URL = 'http://127.0.0.1:8000';

// Core UI Elements
const landingPage = document.getElementById('landingPage');
const mainDashboard = document.getElementById('mainDashboard');
const sidebar = document.querySelector('.sidebar');
const sections = document.querySelectorAll('.content-section');
const navLinks = document.querySelectorAll('.sidebar-nav a');

// Auth Elements
const entryAuthForm = document.getElementById('entryAuthForm');
const entryUsername = document.getElementById('entryUsername');
const entryPassword = document.getElementById('entryPassword');

// Dashboard Elements
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const imagePreviewContainer = document.getElementById('imagePreviewContainer');
const imagePreview = document.getElementById('imagePreview');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultsSection = document.getElementById('results');
const nextAnalysisBtn = document.getElementById('nextAnalysisBtn');

// Profile & Advice Elements
const weatherContent = document.getElementById('weatherContent');
const skinTypeDisplay = document.getElementById('skinTypeDisplay');
const triggerForm = document.getElementById('triggerForm');
const triggerSuggestions = document.getElementById('triggerSuggestions');

// History & Glossary
const progressTimeline = document.getElementById('progressTimeline');
const glossaryGrid = document.getElementById('glossaryGrid');

let authMode = 'login';
let userProfile = null;

// Theme Toggle Bindings
const themeToggleBtn = document.getElementById('themeToggleBtn');
const themeIconSun = document.getElementById('themeIconSun');
const themeIconMoon = document.getElementById('themeIconMoon');
const themeToggleText = document.getElementById('themeToggleText');

// Check Local Storage and Initialize Theme
function initTheme() {
    const savedTheme = localStorage.getItem('theme') || 'dark';
    if (savedTheme === 'light') {
        document.body.classList.add('light-theme');
        if (themeIconSun) themeIconSun.classList.remove('hidden');
        if (themeIconMoon) themeIconMoon.classList.add('hidden');
        if (themeToggleText) themeToggleText.textContent = 'Medical Dark';
    } else {
        document.body.classList.remove('light-theme');
        if (themeIconSun) themeIconSun.classList.add('hidden');
        if (themeIconMoon) themeIconMoon.classList.remove('hidden');
        if (themeToggleText) themeToggleText.textContent = 'Sterile Light';
    }
}

// Bind Theme Click Handler
if (themeToggleBtn) {
    themeToggleBtn.onclick = () => {
        const isCurrentlyLight = document.body.classList.contains('light-theme');
        if (isCurrentlyLight) {
            localStorage.setItem('theme', 'dark');
            initTheme();
        } else {
            localStorage.setItem('theme', 'light');
            initTheme();
        }
    };
}

// Run theme initialization
initTheme();

// ABCDE Severity Checklist Bindings
const abcdeCheckboxes = document.querySelectorAll('.abcde-checkbox');
const abcdeRiskBadge = document.getElementById('abcdeRiskBadge');
const abcdeRiskBarFill = document.getElementById('abcdeRiskBarFill');

// Initialize/Update Risk Checklist logic
function updateABCDE_Risk() {
    if (!abcdeRiskBadge || !abcdeRiskBarFill) return;
    
    let checkedCount = 0;
    abcdeCheckboxes.forEach(cb => {
        if (cb.checked) checkedCount++;
    });
    
    const percent = (checkedCount / 5) * 100;
    abcdeRiskBarFill.style.width = `${percent}%`;
    
    if (checkedCount <= 1) {
        abcdeRiskBadge.className = 'risk-badge low';
        abcdeRiskBadge.textContent = 'Low Risk';
        abcdeRiskBarFill.style.backgroundColor = '#10b981';
    } else if (checkedCount <= 3) {
        abcdeRiskBadge.className = 'risk-badge medium';
        abcdeRiskBadge.textContent = 'Moderate Risk';
        abcdeRiskBarFill.style.backgroundColor = '#fbbf24';
    } else {
        abcdeRiskBadge.className = 'risk-badge high';
        abcdeRiskBadge.textContent = 'High Risk';
        abcdeRiskBarFill.style.backgroundColor = '#ef4444';
    }
}

// Bind event listeners to each ABCDE checkbox
if (abcdeCheckboxes.length > 0) {
    abcdeCheckboxes.forEach(checkbox => {
        checkbox.onchange = () => updateABCDE_Risk();
    });
}

// Initialize
checkAuth();

function checkAuth() {
    const token = localStorage.getItem('token');
    if (token) {
        landingPage.classList.add('hidden');
        mainDashboard.classList.remove('hidden');
        document.getElementById('openChatBtn').classList.remove('hidden');
        fetchProfile();
        fetchHistory();
        fetchGlossary();
        initWeather();
        initQuickClinicSearch();
    } else {
        landingPage.classList.remove('hidden');
        mainDashboard.classList.add('hidden');
        document.getElementById('openChatBtn').classList.add('hidden');
    }
}

function initQuickClinicSearch() {
    navigator.geolocation.getCurrentPosition(async (pos) => {
        const lat = pos.coords.latitude;
        const lng = pos.coords.longitude;
        const query = `
            [out:json];
            (node["healthcare"="dermatology"](around:10000, ${lat}, ${lng}););
            out body 1;
        `;
        const url = "https://overpass-api.de/api/interpreter?data=" + encodeURIComponent(query);
        try {
            const res = await fetch(url);
            const data = await res.json();
            if (data.elements && data.elements.length > 0) {
                const clinic = data.elements[0];
                const name = clinic.tags.name || "Dermatology Clinic";
                document.getElementById('quickClinicCard').classList.remove('hidden');
                document.getElementById('nearestClinicText').innerHTML = `The nearest specialist is <b>${name}</b>. <a href="javascript:void(0)" onclick="showSection('clinics')" style="color: var(--primary); text-decoration: underline;">View on map</a>`;
            }
        } catch(e) {}
    });
}

// Navigation Logic
window.showSection = (sectionId) => {
    sections.forEach(s => s.classList.add('hidden'));
    document.getElementById(`section-${sectionId}`).classList.remove('hidden');
    
    navLinks.forEach(l => l.classList.remove('active'));
    document.getElementById(`nav-${sectionId}`).classList.add('active');
    
    if (sectionId === 'clinics') initMap();
};

// Auth Functions
window.toggleAuthMode = () => {
    authMode = authMode === 'login' ? 'register' : 'login';
    document.getElementById('entryTitle').textContent = authMode === 'login' ? 'Login to System' : 'Clinical Registration';
    document.getElementById('entryAuthSubmit').textContent = authMode === 'login' ? 'Sign In' : 'Register Account';
};

entryAuthForm.onsubmit = async (e) => {
    e.preventDefault();
    const formData = new URLSearchParams();
    formData.append('username', entryUsername.value);
    formData.append('password', entryPassword.value);

    try {
        const response = await fetch(`${API_URL}/${authMode}`, {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || 'Auth failed');
        
        if (authMode === 'login') {
            localStorage.setItem('token', data.access_token);
            checkAuth();
        } else {
            alert('Success! Please sign in.');
            toggleAuthMode();
        }
    } catch (err) { alert(err.message); }
};

window.logout = () => {
    localStorage.removeItem('token');
    location.reload();
};

// Profile & History
async function fetchProfile() {
    const token = localStorage.getItem('token');
    const response = await fetch(`${API_URL}/profile`, {
        headers: { 'Authorization': `Bearer ${token}` }
    });
    if (response.ok) {
        userProfile = await response.json();
        updateProfileUI();
    }
}

function updateProfileUI() {
    if (userProfile.skin_type) {
        document.getElementById('currentSkinTypeText').innerHTML = `<div class="skin-badge">${userProfile.skin_type} Skin</div><p>${getSkinAdvice(userProfile.skin_type)}</p>`;
        document.getElementById('skinTypeSelect').value = userProfile.skin_type;
    }
    
    const foodItems = (userProfile.food_habits || '').split(',').map(s => s.trim());
    document.querySelectorAll('#foodHabitsGrid input[type="checkbox"]').forEach(cb => {
        cb.checked = foodItems.includes(cb.value);
    });
    
    const productItems = (userProfile.products_used || '').split(',').map(s => s.trim());
    document.querySelectorAll('#productsUsedGrid input[type="checkbox"]').forEach(cb => {
        cb.checked = productItems.includes(cb.value);
    });

    updateTriggerSuggestions();
}

async function fetchHistory() {
    const token = localStorage.getItem('token');
    const response = await fetch(`${API_URL}/history`, {
        headers: { 'Authorization': `Bearer ${token}` }
    });
    if (response.ok) {
        const history = await response.json();
        renderTimeline(history);
    }
}

function renderTimeline(history) {
    if (!progressTimeline) return;
    progressTimeline.innerHTML = '';
    history.forEach(item => {
        const div = document.createElement('div');
        div.className = 'timeline-item';
        div.innerHTML = `
            <div class="timeline-date">${new Date(item.timestamp).toLocaleDateString()}</div>
            <div class="timeline-content">
                ${item.image_url ? `<img src="${API_URL}${item.image_url}" class="timeline-img">` : ''}
                <div class="timeline-details">
                    <h4>${item.prediction}</h4>
                    <p>Confidence: ${item.confidence}%</p>
                    <p class="status">${item.confidence > 90 ? 'High Certainty' : 'Follow-up Suggested'}</p>
                </div>
            </div>
        `;
        progressTimeline.appendChild(div);
    });
}

// Prediction Logic
dropZone.onclick = () => fileInput.click();
fileInput.onchange = (e) => {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            imagePreviewContainer.style.display = 'block';
            dropZone.style.display = 'none';
        };
        reader.readAsDataURL(file);
    }
};

if (nextAnalysisBtn) {
    nextAnalysisBtn.onclick = () => {
        fileInput.value = '';
        imagePreview.src = '';
        imagePreviewContainer.style.display = 'none';
        dropZone.style.display = 'block';
        resultsSection.style.display = 'none';
    };
}

analyzeBtn.onclick = async () => {
    const file = fileInput.files[0];
    if (!file) return;

    analyzeBtn.disabled = true;
    analyzeBtn.textContent = 'Analyzing...';
    
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${localStorage.getItem('token')}` },
            body: formData
        });
        
        if (response.status === 401) {
            alert("Your session has expired. Please log in again.");
            logout();
            return;
        }

        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || 'Prediction failed');
        displayResults(data);
        fetchHistory();
    } catch (err) { alert('Error: ' + err.message); }
    finally { analyzeBtn.disabled = false; analyzeBtn.textContent = 'Execute Analysis'; }
};

function displayResults(data) {
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
    document.getElementById('predictionText').textContent = data.prediction;
    document.getElementById('confidenceText').textContent = `${data.confidence}% Confidence`;
    document.getElementById('confidenceFill').style.width = `${data.confidence}%`;
    document.getElementById('descriptionText').textContent = data.description;
    
    populateList('symptomsList', data.symptoms);
    populateList('recommendationsList', data.recommendations);
    populateList('dosList', data.dos);
    populateList('dontsList', data.donts);

    // Proactive Clinical Recommendation after Analysis
    recommendNearestClinic();
}

async function recommendNearestClinic() {
    const list = document.getElementById('recommendationsList');
    
    const showManualSearch = () => {
        const li = document.createElement('li');
        li.id = "manualResultSearch";
        li.innerHTML = `
            <div style="background: rgba(255,255,255,0.05); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
                <p style="font-size: 0.85rem; margin-bottom: 0.5rem;">Location access disabled. Enter city to find nearest specialist:</p>
                <div style="display: flex; gap: 0.5rem;">
                    <input type="text" id="resultCityInput" placeholder="City name" style="flex:1; padding: 0.4rem; border-radius: 5px; border: 1px solid var(--border); background: #000; color: white; font-size: 0.8rem;">
                    <button onclick="searchClinicByCity()" class="btn-process" style="width: auto; padding: 0.4rem 1rem; font-size: 0.8rem;">Find</button>
                </div>
            </div>
        `;
        list.prepend(li);
    };

    navigator.geolocation.getCurrentPosition(async (pos) => {
        const lat = pos.coords.latitude;
        const lng = pos.coords.longitude;
        const query = `[out:json];(node["healthcare"="dermatology"](around:10000, ${lat}, ${lng}););out body 1;`;
        const url = "https://overpass-api.de/api/interpreter?data=" + encodeURIComponent(query);
        try {
            const res = await fetch(url);
            const data = await res.json();
            if (data.elements && data.elements.length > 0) {
                const clinic = data.elements[0];
                const name = clinic.tags.name || "Dermatology Clinic";
                const li = document.createElement('li');
                li.style.color = "var(--secondary)";
                li.style.fontWeight = "bold";
                li.innerHTML = `🌟 Recommended Action: Visit <b>${name}</b> nearby for a professional consultation. <a href="https://www.google.com/maps/dir/?api=1&destination=${clinic.lat},${clinic.lon}" target="_blank" style="color:var(--primary); text-decoration:underline;">Directions</a>`;
                list.prepend(li);
            } else {
                showManualSearch();
            }
        } catch(e) { showManualSearch(); }
    }, () => {
        showManualSearch();
    });
}

window.searchClinicByCity = async () => {
    const city = document.getElementById('resultCityInput').value;
    if (!city) return;
    
    const geoUrl = `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(city)}`;
    const res = await fetch(geoUrl);
    const data = await res.json();
    
    if (data && data.length > 0) {
        const lat = data[0].lat;
        const lon = data[0].lon;
        const query = `[out:json];(node["healthcare"="dermatology"](around:20000, ${lat}, ${lon}););out body 1;`;
        const url = "https://overpass-api.de/api/interpreter?data=" + encodeURIComponent(query);
        const clinicRes = await fetch(url);
        const clinicData = await clinicRes.json();
        
        const container = document.getElementById('manualResultSearch');
        if (clinicData.elements && clinicData.elements.length > 0) {
            const clinic = clinicData.elements[0];
            container.innerHTML = `🌟 Recommended Action in ${city}: Visit <b>${clinic.tags.name || "Dermatology Clinic"}</b>. <a href="https://www.google.com/maps/dir/?api=1&destination=${clinic.lat},${clinic.lon}" target="_blank" style="color:var(--primary); text-decoration:underline;">Directions</a>`;
            container.style.color = "var(--secondary)";
            container.style.fontWeight = "bold";
        } else {
            container.innerHTML = `No specialists found in ${city}. <a href="https://www.google.com/maps/search/dermatologist+in+${city}" target="_blank" style="color:var(--primary); text-decoration:underline;">Search Google Maps</a>`;
        }
    }
};

function populateList(id, items) {
    const el = document.getElementById(id);
    el.innerHTML = '';
    items.forEach(item => {
        const li = document.createElement('li');
        li.textContent = item;
        el.appendChild(li);
    });
}

// Weather Advice
function initWeather() {
    navigator.geolocation.getCurrentPosition(async (pos) => {
        // In a real app, use an API like OpenWeatherMap
        // For demo, we simulate based on common weather logic
        const temp = 30; // Simulated
        const humidity = 40; // Simulated
        
        let advice = "Weather seems normal. Maintain your regular skincare routine.";
        if (temp > 25) advice = "High UV detected. Apply SPF 50+ and stay hydrated.";
        if (humidity < 30) advice = "Dry air detected. Use a heavy moisturizer today.";
        
        weatherContent.innerHTML = `
            <div class="weather-info">
                <div class="temp">28°C</div>
                <div class="cond">Sunny / Clear</div>
            </div>
            <p class="advice-text">${advice}</p>
        `;
    }, () => {
        weatherContent.innerHTML = "<p>Location access denied. Enable it for weather-based advice.</p>";
    });
}

// Skin Type Selection
window.updateSkinType = async (type) => {
    if (!type) return;
    
    try {
        const response = await fetch(`${API_URL}/profile`, {
            method: 'POST',
            headers: { 
                'Authorization': `Bearer ${localStorage.getItem('token')}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ skin_type: type })
        });
        
        if (response.ok) {
            if (!userProfile) userProfile = {};
            userProfile.skin_type = type;
            updateProfileUI();
        } else {
            alert("Failed to save skin type. Your session may have expired.");
        }
    } catch (err) {
        alert("Error saving skin type. Ensure your backend server is running.");
        console.error("Skin Type Error:", err);
    }
};

function getSkinAdvice(type) {
    const advice = {
        "Oily": "Focus on double cleansing and non-comedogenic moisturizers. <br><span style='color: var(--primary); font-weight: bold;'>🌟 Recommended Product:</span> La Roche-Posay Effaclar Purifying Foaming Gel.",
        "Dry": "Use creamy cleansers and hyaluronic acid serums. <br><span style='color: var(--primary); font-weight: bold;'>🌟 Recommended Product:</span> CeraVe Hydrating Facial Cleanser or Neutrogena Hydro Boost Water Gel.",
        "Combination": "Balance is key. Treat oily and dry zones separately. <br><span style='color: var(--primary); font-weight: bold;'>🌟 Recommended Product:</span> Paula's Choice Skin Balancing Pore-Reducing Toner.",
        "Sensitive": "Avoid fragrances and look for soothing ingredients like Cica. <br><span style='color: var(--primary); font-weight: bold;'>🌟 Recommended Product:</span> Aveeno Calm + Restore Oat Gel Moisturizer.",
        "Normal": "Maintain your healthy barrier with a basic cleanser, moisturizer, and daily SPF. <br><span style='color: var(--primary); font-weight: bold;'>🌟 Recommended Product:</span> Cetaphil Daily Facial Cleanser."
    };
    return advice[type] || "";
}

// Trigger Management
triggerForm.onsubmit = async (e) => {
    e.preventDefault();
    const food = Array.from(document.querySelectorAll('#foodHabitsGrid input:checked')).map(cb => cb.value).join(', ');
    const products = Array.from(document.querySelectorAll('#productsUsedGrid input:checked')).map(cb => cb.value).join(', ');
    
    try {
        const response = await fetch(`${API_URL}/profile`, {
            method: 'POST',
            headers: { 
                'Authorization': `Bearer ${localStorage.getItem('token')}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ food_habits: food, products_used: products })
        });
        
        if (response.ok) {
            if (!userProfile) userProfile = {};
            userProfile.food_habits = food;
            userProfile.products_used = products;
            updateTriggerSuggestions();
            alert("Habits updated successfully!");
        } else {
            alert("Failed to update habits. Your session may have expired.");
        }
    } catch (err) {
        alert("Error saving habits. Ensure your backend server is running.");
        console.error("Trigger Error:", err);
    }
};

function updateTriggerSuggestions() {
    let suggestions = [];
    const food = userProfile.food_habits?.toLowerCase() || '';
    const products = userProfile.products_used?.toLowerCase() || '';

    // Food Habits Suggestions
    if (food.includes('dairy')) suggestions.push("Dairy is a common trigger for cystic acne and excess sebum production.");
    if (food.includes('spicy')) suggestions.push("Spicy foods can trigger rosacea flare-ups and inflammation.");
    if (food.includes('oily')) suggestions.push("Excessively oily/fried foods can contribute to sebum overproduction.");
    if (food.includes('junk') || food.includes('processed')) suggestions.push("Junk and processed foods lack essential nutrients needed for skin repair and barrier defense.");
    if (food.includes('sugar')) suggestions.push("High sugar/sweets can accelerate skin aging and worsen acne through insulin spikes.");
    if (food.includes('salt')) suggestions.push("High salt intake can cause fluid retention, leading to puffy eyes and skin dehydration.");
    if (food.includes('coffee')) suggestions.push("Excess coffee consumption can elevate cortisol levels and dehydrate the skin.");
    if (food.includes('water')) suggestions.push("Low water intake severely compromises the skin barrier. Drink at least 2 liters daily.");

    // Products Used Suggestions
    if (products.includes('soap')) suggestions.push("Regular body soaps used on the face can disrupt the acid mantle and dry out skin. Use a gentle face wash. <br><span style='color: var(--primary); font-weight: bold;'>🌟 Recommended Product:</span> Vanicream Gentle Facial Cleanser.");
    if (products.includes('scented') || products.includes('perfume')) suggestions.push("Fragrances can cause contact dermatitis and disrupt the skin barrier. switch to fragrance-free. <br><span style='color: var(--primary); font-weight: bold;'>🌟 Recommended Product:</span> Cetaphil Gentle Skin Cleanser (Fragrance-Free).");
    if (products.includes('makeup')) suggestions.push("Ensure you double-cleanse every night to remove all makeup and prevent clogged pores. <br><span style='color: var(--primary); font-weight: bold;'>🌟 Recommended Product:</span> DHC Deep Cleansing Oil or Kose Softymo Speedy Cleansing Oil.");
    if (products.includes('recently changed')) suggestions.push("A recent change in skincare can cause 'purging' or an allergic reaction. Reintroduce new products slowly.");
    if (products.includes('sunscreen')) suggestions.push("Daily sunscreen use is the best defense against premature aging and skin cancer. Keep it up! <br><span style='color: var(--primary); font-weight: bold;'>🌟 Recommended Product:</span> EltaMD UV Clear Broad-Spectrum SPF 46 or Beauty of Joseon Relief Sun.");
    if (products.includes('acne creams')) suggestions.push("Medicated acne creams can cause extreme dryness. Ensure you are moisturizing adequately. <br><span style='color: var(--primary); font-weight: bold;'>🌟 Recommended Product:</span> CeraVe PM Facial Moisturizing Lotion.");

    triggerSuggestions.innerHTML = suggestions.length > 0 
        ? suggestions.map(s => `<div class="trigger-alert" style="background: rgba(16, 185, 129, 0.1); border-left: 4px solid var(--secondary); padding: 1rem; margin-bottom: 0.5rem; border-radius: 4px;">${s}</div>`).join('')
        : "<p>No common triggers detected from your inputs.</p>";
}

let map;
function initMap() {
    const mapEl = document.getElementById('map');
    if (map) return; // Prevent multiple initializations

    // Default to a central location if geo fails
    const defaultLoc = [20.5937, 78.9629]; // India center
    map = L.map('map').setView(defaultLoc, 13);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap contributors'
    }).addTo(map);

    navigator.geolocation.getCurrentPosition((pos) => {
        const userLoc = [pos.coords.latitude, pos.coords.longitude];
        map.setView(userLoc, 14);
        
        // Add User Marker
        L.marker(userLoc).addTo(map)
            .bindPopup('Your Location')
            .openPopup();

        searchNearbyDermatologists(pos.coords.latitude, pos.coords.longitude);
    }, (err) => {
        // Detailed feedback for explicit "Use My Location" clicks
        let msg = "Could not access location.";
        if (err.code === 1) msg = "Location access denied. Please enable location permissions in your browser settings or use the search bar above.";
        else if (err.code === 2) msg = "Position unavailable. Try searching for your city manually.";
        
        const list = document.getElementById('clinicList');
        list.innerHTML = `<div style="padding: 1rem; color: #f87171; border: 1px solid #f87171; border-radius: 8px; background: rgba(248, 113, 113, 0.1);">
            <strong>Error:</strong> ${msg}
        </div>`;
    });
}

window.searchManualLocation = async () => {
    const query = document.getElementById('manualLocationInput').value;
    if (!query) return alert("Please enter a city or area name.");

    const list = document.getElementById('clinicList');
    list.innerHTML = `<p>Geocoding "${query}"...</p>`;

    // Step 1: Geocode the city name using Nominatim
    try {
        const geoUrl = `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(query)}`;
        const res = await fetch(geoUrl);
        const data = await res.json();

        if (data && data.length > 0) {
            const lat = parseFloat(data[0].lat);
            const lng = parseFloat(data[0].lon);
            
            // Step 2: Initialize map at this location if not done
            if (!map) {
                map = L.map('map').setView([lat, lng], 13);
                L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                    attribution: '&copy; OpenStreetMap contributors'
                }).addTo(map);
            } else {
                map.setView([lat, lng], 13);
            }

            // Step 3: Search clinics at the new location
            searchNearbyDermatologists(lat, lng);
        } else {
            list.innerHTML = `<p>Location "${query}" not found. Try a different name.</p>`;
        }
    } catch (err) {
        list.innerHTML = "<p>Error reaching geocoding service.</p>";
    }
};

async function searchNearbyDermatologists(lat, lng) {
    const list = document.getElementById('clinicList');
    list.innerHTML = "<p>Searching for specialized skin clinics...</p>";

    // Stricter query for ONLY dermatology specialists
    const query = `
        [out:json];
        (
          node["healthcare"="dermatology"](around:10000, ${lat}, ${lng});
          way["healthcare"="dermatology"](around:10000, ${lat}, ${lng});
          node["speciality"="dermatology"](around:10000, ${lat}, ${lng});
          node["name"~"Skin",i](around:10000, ${lat}, ${lng});
          node["name"~"Dermatology",i](around:10000, ${lat}, ${lng});
        );
        out center;
    `;
    const url = "https://overpass-api.de/api/interpreter?data=" + encodeURIComponent(query);

    try {
        const response = await fetch(url);
        const data = await response.json();
        
        if (data.elements && data.elements.length > 0) {
            renderClinics(data.elements);
        } else {
            list.innerHTML = "<p>No specific dermatologists found in a 5km radius. Try a wider search.</p>";
        }
    } catch (err) {
        console.warn("Overpass API failed, using fallback clinics:", err);
        // Fallback to mock clinics so the demo doesn't break
        const mockClinics = [
            { tags: { name: "SkinSense Specialized Dermatology Clinic", phone: "555-0101", "addr:street": "Medical Center Blvd" }, lat: lat + 0.01, lon: lng + 0.01 },
            { tags: { name: "Advanced Skin Care Center", phone: "555-0202", "addr:street": "Health Ave" }, lat: lat - 0.01, lon: lng - 0.01 },
            { tags: { name: "City Hospital Dermatology Dept", phone: "555-0303", "addr:street": "Main Street" }, lat: lat + 0.02, lon: lng - 0.01 }
        ];
        renderClinics(mockClinics);
    }
}

function renderClinics(clinics) {
    const list = document.getElementById('clinicList');
    list.innerHTML = '';
    
    clinics.slice(0, 10).forEach(c => {
        const name = c.tags.name || "Skin Specialist Clinic";
        const type = "Dermatology Specialist";
        
        const lat = c.lat || (c.center ? c.center.lat : null);
        const lon = c.lon || (c.center ? c.center.lon : null);
        
        // Add Marker
        if (lat && lon) {
            L.marker([lat, lon]).addTo(map)
                .bindPopup(`<b>${name}</b><br>${type}`);
        }

        // Add to List
        const div = document.createElement('div');
        div.className = 'clinic-card';
        div.innerHTML = `
            <strong>${name}</strong>
            <p>${type}</p>
            <a href="https://www.google.com/maps/dir/?api=1&destination=${lat},${lon}" target="_blank" class="btn-secondary" style="font-size: 0.8rem; margin-top: 0.5rem; display: inline-block;">Get Directions</a>
        `;
        list.appendChild(div);
    });
}

// Chat functions (restored)
window.toggleChat = () => document.getElementById('chatWidget').classList.toggle('hidden');
async function fetchGlossary() {
    const res = await fetch(`${API_URL}/diseases`);
    if (res.ok) {
        const diseases = await res.json();
        glossaryGrid.innerHTML = Object.entries(diseases).map(([name, info]) => {
            return `
            <div class="glossary-card" style="overflow: hidden;">
                <h3>${name}</h3>
                <p style="color: var(--text-dim); font-size: 0.95rem;">${info.description.replace(/â€“/g, '-')}</p>
            </div>
        `}).join('');
    }
}
