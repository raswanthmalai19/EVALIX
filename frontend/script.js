/**
 * Evalix - Credit Risk Assessment Frontend
 * Compatible with Loan_Prediction.csv dataset
 * Connects to FastAPI backend at localhost:8000
 */

// ========================================
// Configuration
// ========================================
const API_BASE_URL = 'http://localhost:8000';

// ========================================
// DOM Elements
// ========================================
const loanForm      = document.getElementById('loanForm');
const submitBtn      = document.getElementById('submitBtn');
const resultsPlaceholder = document.getElementById('resultsPlaceholder');
const resultsContainer   = document.getElementById('resultsContainer');
const errorMessage       = document.getElementById('errorMessage');
const errorText          = document.getElementById('errorText');

// Status
const apiStatus     = document.getElementById('apiStatus');
const apiStatusText = document.getElementById('apiStatusText');

// Decision
const decisionCard  = document.getElementById('decisionCard');
const decisionBadge = document.getElementById('decisionBadge');
const decisionIcon  = document.getElementById('decisionIcon');
const decisionText  = document.getElementById('decisionText');
const riskScore     = document.getElementById('riskScore');
const riskLevel     = document.getElementById('riskLevel');
const confidence    = document.getElementById('confidence');
const riskBar       = document.getElementById('riskBar');
const riskMarker    = document.getElementById('riskMarker');

// Explanations
const riskFactorsList       = document.getElementById('riskFactorsList');
const protectiveFactorsList = document.getElementById('protectiveFactorsList');

// Suggestions
const suggestionsCard = document.getElementById('suggestionsCard');
const suggestionsList = document.getElementById('suggestionsList');

// ========================================
// API Health Check
// ========================================
async function checkApiStatus() {
    try {
        const res = await fetch(`${API_BASE_URL}/health`, { method: 'GET' });
        if (res.ok) {
            const data = await res.json();
            apiStatus.classList.add('online');
            apiStatus.classList.remove('offline');
            apiStatusText.textContent = `Online (${data.model})`;
        } else {
            throw new Error('Not OK');
        }
    } catch {
        apiStatus.classList.add('offline');
        apiStatus.classList.remove('online');
        apiStatusText.textContent = 'Offline';
    }
}

// Check every 15 s
checkApiStatus();
setInterval(checkApiStatus, 15000);

// ========================================
// Collect Form Data
// ========================================
function collectFormData() {
    return {
        age:                  parseFloat(document.getElementById('age').value),
        income:               parseFloat(document.getElementById('income').value),
        assets:               parseFloat(document.getElementById('assets').value),
        credit_score:         parseFloat(document.getElementById('credit_score').value),
        debt_to_income_ratio: parseFloat(document.getElementById('debt_to_income_ratio').value),
        existing_loan:        parseInt(document.getElementById('existing_loan').value, 10),
        criminal_record:      parseInt(document.getElementById('criminal_record').value, 10),
    };
}

// ========================================
// Submit Handler
// ========================================
loanForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    setLoading(true);
    hideError();

    const payload = collectFormData();

    try {
        const res = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || `Server error (${res.status})`);
        }

        const result = await res.json();
        displayResults(result);
    } catch (err) {
        showError(err.message || 'Unable to connect to the API. Make sure the backend is running.');
    } finally {
        setLoading(false);
    }
});

// ========================================
// Display Results
// ========================================
function displayResults(result) {
    resultsPlaceholder.classList.add('hidden');
    resultsContainer.classList.remove('hidden');

    // --- Decision card (3-state) ---
    const decision = result.decision;
    let decisionClass, iconClass;
    if (decision === 'Approved') {
        decisionClass = 'approved';
        iconClass = 'fas fa-check-circle';
    } else if (decision === 'Review Needed') {
        decisionClass = 'review';
        iconClass = 'fas fa-exclamation-circle';
    } else {
        decisionClass = 'rejected';
        iconClass = 'fas fa-times-circle';
    }
    decisionCard.className = `decision-card ${decisionClass}`;
    decisionBadge.className = `decision-badge ${decisionClass}`;
    decisionIcon.className = iconClass;
    decisionText.textContent = decision.toUpperCase();

    // Approval odds = probability (direct display)
    const approvalPct = (result.probability * 100).toFixed(2);
    riskScore.textContent = `${approvalPct}%`;
    riskLevel.textContent = result.risk_level.toUpperCase();
    riskLevel.className   = `metric-badge ${result.risk_level.toLowerCase().replace(' ', '-')}`;
    confidence.textContent = `${result.confidence}%`;

    riskBar.style.width  = `${approvalPct}%`;
    riskMarker.style.left = `${approvalPct}%`;

    // Color the gauge (green = high approval, red = low)
    const approvalColor = approvalPct >= 60 ? '#27ae60' : approvalPct >= 30 ? '#f39c12' : '#e74c3c';
    riskBar.style.background = approvalColor;

    // --- Risk & Protective Factors ---
    riskFactorsList.innerHTML = '';
    protectiveFactorsList.innerHTML = '';

    if (result.top_risk_factors && result.top_risk_factors.length) {
        result.top_risk_factors.forEach(f => {
            const li = document.createElement('li');
            li.className = 'factor-item risk';
            li.innerHTML = `
                <div class="factor-info">
                    <span class="factor-name">${f.feature}</span>
                    <span class="factor-impact">${Math.abs(f.impact).toFixed(4)}</span>
                </div>
                <div class="factor-bar">
                    <div class="factor-bar-fill risk" style="width:${Math.min(Math.abs(f.impact) * 500, 100)}%"></div>
                </div>`;
            riskFactorsList.appendChild(li);
        });
    } else {
        riskFactorsList.innerHTML = '<li class="factor-item empty">No significant risk factors</li>';
    }

    if (result.top_protective_factors && result.top_protective_factors.length) {
        result.top_protective_factors.forEach(f => {
            const li = document.createElement('li');
            li.className = 'factor-item protective';
            li.innerHTML = `
                <div class="factor-info">
                    <span class="factor-name">${f.feature}</span>
                    <span class="factor-impact">+${Math.abs(f.impact).toFixed(4)}</span>
                </div>
                <div class="factor-bar">
                    <div class="factor-bar-fill protective" style="width:${Math.min(Math.abs(f.impact) * 500, 100)}%"></div>
                </div>`;
            protectiveFactorsList.appendChild(li);
        });
    } else {
        protectiveFactorsList.innerHTML = '<li class="factor-item empty">No significant protective factors</li>';
    }

    // --- Improvement Suggestions ---
    if (result.improvement_suggestions && result.improvement_suggestions.length) {
        suggestionsCard.classList.remove('hidden');
        suggestionsList.innerHTML = '';
        result.improvement_suggestions.forEach(tip => {
            const li = document.createElement('li');
            li.className = 'suggestion-item';
            li.innerHTML = `<i class="fas fa-arrow-circle-up"></i><span>${tip}</span>`;
            suggestionsList.appendChild(li);
        });
    } else {
        suggestionsCard.classList.add('hidden');
    }

    // Smooth scroll to results on mobile
    if (window.innerWidth < 992) {
        resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

// ========================================
// Utilities
// ========================================
function setLoading(on) {
    submitBtn.disabled = on;
    submitBtn.classList.toggle('loading', on);
}

function showError(msg) {
    errorMessage.classList.remove('hidden');
    errorText.textContent = msg;
    resultsPlaceholder.classList.add('hidden');
    resultsContainer.classList.add('hidden');
}

function hideError() {
    errorMessage.classList.add('hidden');
}

function resetForm() {
    loanForm.reset();
    resultsContainer.classList.add('hidden');
    errorMessage.classList.add('hidden');
    resultsPlaceholder.classList.remove('hidden');
}

// Add smooth scroll for nav links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) target.scrollIntoView({ behavior: 'smooth' });
    });
});
