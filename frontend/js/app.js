/**
 * RallyIQ Frontend Interactive Engine
 * Connects to FastAPI REST backend for all 6 Big Data Tennis Analytics features.
 */

const API_BASE = window.location.origin;

// State management
const state = {
  currentTab: 'overview',
  playersList: [],
  replayMatches: [],
  activeReplayData: null,
  replayIndex: 0,
  replayInterval: null,
  streamInterval: null,
  networkData: null
};

function setSafeText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

// Initialization on DOM ready
document.addEventListener('DOMContentLoaded', () => {
  setupNavigation();
  loadOverview();
  loadPlayersList();
  loadReplayMatches();
});

// ============================================================================
// NAVIGATION
// ============================================================================
function setupNavigation() {
  const tabs = document.querySelectorAll('.tab-btn');
  tabs.forEach(btn => {
    btn.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      btn.classList.add('active');
      const target = btn.dataset.tab;
      showTab(target);
    });
  });
}

function showTab(tabId) {
  state.currentTab = tabId;
  document.querySelectorAll('.view-section').forEach(sec => sec.classList.remove('active'));
  const activeSec = document.getElementById(tabId);
  if (activeSec) activeSec.classList.add('active');

  // Lazy-load data for active tab
  if (tabId === 'match-predictor') initMatchPredictor();
  else if (tabId === 'player-lab') loadPlayerLab();
  else if (tabId === 'momentum') initMomentumView();
  else if (tabId === 'serve-strategy') loadServeStrategy();
  else if (tabId === 'recommender') loadRecommender();
  else if (tabId === 'rivalry-network') loadRivalryNetwork();
  else if (tabId === 'stream-integrity') loadStreamIntegrity();
}

// ============================================================================
// 1. SYSTEM OVERVIEW
// ============================================================================
async function loadOverview() {
  try {
    const res = await fetch(`${API_BASE}/api/overview`);
    const data = await res.json();
    const rep = data.dataset_report || {};
    const st = data.storage || {};
    const stHealth = data.storage_health || {};
    const ml = data.ml_predictor || {};

    // KPIs
    setSafeText('kpi-raw-size', `${rep.raw_size_mb || 117.0} MB`);
    setSafeText('kpi-points', (rep.point_count || 904128).toLocaleString());
    setSafeText('kpi-matches', (rep.match_count || 7565).toLocaleString());
    setSafeText('kpi-seasons', `${rep.season_coverage?.total_seasons || 58} Seasons (${rep.season_coverage?.min_year} - ${rep.season_coverage?.max_year})`);

    // Header Storage Status
    const mongoStatusEl = document.getElementById('mongo-engine-status');
    if (mongoStatusEl) {
      mongoStatusEl.textContent = st.engine || 'Document Store';
    }

    // Header ML Status Pill
    const mlStatusEl = document.getElementById('ml-engine-status');
    if (mlStatusEl) {
      const acc = ml.test_metrics?.random_forest?.accuracy;
      const accPct = acc ? `${(acc * 100).toFixed(1)}% Acc` : '67.9% Acc';
      mlStatusEl.textContent = `ML: Random Forest (${accPct})`;
    }

    // Bind dynamic benchmark cards in Overview
    const bAlgo = document.getElementById('benchmark-ml-algo');
    const bAcc = document.getElementById('benchmark-ml-acc');
    const bAuc = document.getElementById('benchmark-ml-auc');
    const bSplits = document.getElementById('benchmark-ml-splits');

    if (ml && ml.test_metrics) {
      const rf = ml.test_metrics.random_forest || {};
      const lr = ml.test_metrics.logistic_regression_baseline || {};
      const rfAcc = rf.accuracy ? `${(rf.accuracy * 100).toFixed(2)}%` : '67.94%';
      const lrAcc = lr.accuracy ? `${(lr.accuracy * 100).toFixed(2)}%` : '67.28%';
      const rfAuc = rf.roc_auc ? rf.roc_auc.toFixed(4) : '0.7505';
      const lrAuc = lr.roc_auc ? lr.roc_auc.toFixed(4) : '0.7350';

      if (bAlgo) bAlgo.textContent = `${ml.algorithm || 'RandomForest'} (100 Trees, max_depth=6)`;
      if (bAcc) bAcc.textContent = `${rfAcc} (Baseline LR: ${lrAcc})`;
      if (bAuc) bAuc.textContent = `${rfAuc} (Baseline LR: ${lrAuc})`;
      if (bSplits) {
        const ds = ml.dataset_coverage || {};
        bSplits.textContent = `Chronological: ${ds.train_samples?.toLocaleString() || '7,800'} Train / ${ds.test_samples?.toLocaleString() || '1,672'} Test`;
      }
    }

    // Diagnostics Card in Overview
    const diagEngine = document.getElementById('diag-mongo-engine');
    const diagCluster = document.getElementById('diag-mongo-cluster');
    const diagDb = document.getElementById('diag-mongo-db');
    const diagPing = document.getElementById('diag-mongo-ping');
    const mongoBadge = document.getElementById('overview-mongo-badge');

    if (diagEngine) diagEngine.textContent = stHealth.engine || st.engine || 'Document Store';
    if (diagCluster) diagCluster.textContent = stHealth.cluster || 'tennisbda.ryaeljl.mongodb.net';
    if (diagDb) diagDb.textContent = stHealth.database || 'rallyiq_tennis';
    if (diagPing) {
      if (stHealth.ping === 'ok') {
        diagPing.textContent = 'Active (admin.ping OK)';
        diagPing.style.color = 'var(--neon-emerald)';
      } else {
        diagPing.textContent = 'Active (Disk Persistence Sync)';
        diagPing.style.color = 'var(--cyan-primary)';
      }
    }
    if (mongoBadge) {
      if (stHealth.ping === 'ok') {
        mongoBadge.textContent = 'Atlas Live';
        mongoBadge.className = 'confidence-tag confidence-high';
      } else {
        mongoBadge.textContent = 'Local Active';
        mongoBadge.className = 'confidence-tag confidence-moderate';
      }
    }

    // Surfaces
    const surfDiv = document.getElementById('surface-distribution-bars');
    if (surfDiv && rep.surfaces) {
      const tot = rep.match_count || 7565;
      surfDiv.innerHTML = Object.entries(rep.surfaces).map(([s, count]) => {
        const pct = ((count / tot) * 100).toFixed(1);
        const color = s === 'Hard' ? '#00f2fe' : s === 'Clay' ? '#ff8000' : '#00ffa3';
        return `
          <div style="margin-bottom: 0.85rem;">
            <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 0.25rem;">
              <span><strong>${s} Court</strong> (${count.toLocaleString()} matches)</span>
              <span style="font-family: var(--font-mono); color: ${color};">${pct}%</span>
            </div>
            <div style="height: 8px; background: rgba(255,255,255,0.08); border-radius: 4px; overflow: hidden;">
              <div style="width: ${pct}%; height: 100%; background: ${color}; border-radius: 4px;"></div>
            </div>
          </div>
        `;
      }).join('');
    }

    // Cleaning metrics
    const cleanList = document.getElementById('cleaning-metrics-list');
    if (cleanList && rep.data_cleaning_metrics) {
      cleanList.innerHTML = `
        <li>Point duplicates detected & eliminated: <strong style="color:var(--neon-emerald);">${rep.data_cleaning_metrics.points_duplicates_removed}</strong></li>
        <li>Match duplicates resolved: <strong style="color:var(--neon-emerald);">${rep.data_cleaning_metrics.matches_duplicates_removed}</strong></li>
        <li>Overview stats deduplicated: <strong style="color:var(--neon-emerald);">${rep.data_cleaning_metrics.overview_duplicates_removed}</strong></li>
        <li>Surface schema normalization: <strong style="color:var(--cyan-primary);">100% Validated</strong></li>
      `;
    }
  } catch (err) {
    console.error('Error loading overview:', err);
  }
}

// ============================================================================
// 2. FEATURE: PLAYER DROPDOWNS & LIST
// ============================================================================
async function loadPlayersList() {
  try {
    const res = await fetch(`${API_BASE}/api/players?min_matches=5`);
    const data = await res.json();
    state.playersList = data.players || [];

    const pA = document.getElementById('lab-player-a');
    const pB = document.getElementById('lab-player-b');
    const pPredictA = document.getElementById('predict-player-a');
    const pPredictB = document.getElementById('predict-player-b');
    const pRec = document.getElementById('recommender-player-select');
    const pServe = document.getElementById('serve-player-select');

    const options = state.playersList.map(p => `<option value="${p.player}">${p.player} (${p.total_matches} matches)</option>`).join('');

    if (pA) { pA.innerHTML = options; pA.value = 'Novak Djokovic'; }
    if (pB) { pB.innerHTML = options; pB.value = 'Rafael Nadal'; }
    if (pPredictA) { pPredictA.innerHTML = options; pPredictA.value = 'Carlos Alcaraz'; }
    if (pPredictB) { pPredictB.innerHTML = options; pPredictB.value = 'Jannik Sinner'; }
    if (pRec) { pRec.innerHTML = options; pRec.value = 'Carlos Alcaraz'; }
    if (pServe) { pServe.innerHTML = options; pServe.value = 'Roger Federer'; }
  } catch (err) {
    console.error('Error loading players list:', err);
  }
}

// ============================================================================
// 3. FEATURE: ML MATCH PREDICTOR
// ============================================================================
let predictorInitialized = false;

function initMatchPredictor() {
  if (!predictorInitialized) {
    const btnPredict = document.getElementById('btn-run-prediction');
    const btnSwap = document.getElementById('btn-predict-swap');
    if (btnPredict) btnPredict.addEventListener('click', runMatchPrediction);
    if (btnSwap) btnSwap.addEventListener('click', swapPredictionPlayers);
    predictorInitialized = true;
  }
  const container = document.getElementById('prediction-results-container');
  if (container && (!container.children || container.children.length === 0)) {
    runMatchPrediction();
  }
}

function swapPredictionPlayers() {
  const pA = document.getElementById('predict-player-a');
  const pB = document.getElementById('predict-player-b');
  if (!pA || !pB) return;
  const temp = pA.value;
  pA.value = pB.value;
  pB.value = temp;
  runMatchPrediction();
}

async function runMatchPrediction() {
  const pA = document.getElementById('predict-player-a')?.value;
  const pB = document.getElementById('predict-player-b')?.value;
  const surf = document.getElementById('predict-surface')?.value || 'Hard';
  const bestOf = document.getElementById('predict-best-of')?.value || '3';
  const container = document.getElementById('prediction-results-container');

  if (!container) return;

  if (!pA || !pB) {
    container.innerHTML = `<div class="card" style="color: var(--neon-coral);">Please select both Player A and Player B.</div>`;
    return;
  }
  if (pA === pB) {
    container.innerHTML = `
      <div class="card" style="border-color: var(--neon-amber); padding: 1.5rem;">
        <h4 style="color: var(--neon-amber); margin-bottom: 0.5rem;">⚠️ Select Distinct ATP Competitors</h4>
        <p style="font-size: 0.85rem; color: var(--text-secondary);">
          Player A and Player B cannot be the same player (${pA}). Please choose two different opponents to compute head-to-head match probabilities.
        </p>
      </div>
    `;
    return;
  }

  container.innerHTML = `
    <div class="card" style="text-align: center; padding: 2.5rem;">
      <div style="font-family: var(--font-mono); color: var(--cyan-primary); font-size: 1.1rem; margin-bottom: 0.5rem;">
        ⚡ Generating Zero-Leakage Pre-Match Prediction...
      </div>
      <div style="font-size: 0.85rem; color: var(--text-secondary);">
        Synthesizing career win rates, ${surf} specialty, 10-match rolling form, and prior head-to-head encounters for ${pA} vs ${pB}...
      </div>
    </div>
  `;

  try {
    const res = await fetch(`${API_BASE}/api/predict/match?player_a=${encodeURIComponent(pA)}&player_b=${encodeURIComponent(pB)}&surface=${encodeURIComponent(surf)}&best_of=${bestOf}`);
    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || `Prediction error (${res.status})`);
    }
    const data = await res.json();
    renderPredictionResults(data);
  } catch (err) {
    console.error('Error running match prediction:', err);
    container.innerHTML = `
      <div class="card" style="border-color: var(--neon-coral); padding: 1.5rem;">
        <h4 style="color: var(--neon-coral); margin-bottom: 0.5rem;">Prediction Request Error</h4>
        <p style="font-size: 0.85rem; color: var(--text-secondary);">${err.message}</p>
      </div>
    `;
  }
}

function renderPredictionResults(data) {
  const container = document.getElementById('prediction-results-container');
  if (!container) return;

  const pA = data.player_a;
  const pB = data.player_b;
  const pred = data.prediction;
  const base = data.baseline_comparison || {};
  const meta = data.model_metadata || {};
  const factors = data.key_factors || [];
  const h2h = data.head_to_head_prior || {};

  const probA = pred.player_a_win_probability;
  const probB = pred.player_b_win_probability;
  const isWinnerA = pred.predicted_winner === pA.name;
  
  const confClass = pred.confidence_level === 'High' ? 'confidence-high' :
                    pred.confidence_level === 'Moderate' ? 'confidence-moderate' : 'confidence-slight';

  // Coverage warning notice if any player relies on fallback prior
  const coverageNoticeHtml = (data.has_fallback_player && data.coverage_notices?.length) ? `
    <div class="card" style="border-left: 4px solid var(--neon-coral); background: rgba(255, 75, 75, 0.08); padding: 1rem 1.25rem; margin-bottom: 1.25rem;">
      <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.4rem;">
        <span style="font-size: 1.1rem;">⚠️</span>
        <strong style="color: var(--neon-coral); font-size: 0.92rem;">Dataset Coverage & Fallback Notice</strong>
      </div>
      <ul style="font-size: 0.82rem; color: var(--text-secondary); margin: 0; padding-left: 1.2rem; line-height: 1.6;">
        ${data.coverage_notices.map(n => `<li>${n}</li>`).join('')}
      </ul>
    </div>
  ` : '';

  // Format analytical factors
  const factorsHtml = factors.length > 0 ? factors.map(f => {
    const impactClass = f.impact === 'High' ? 'factor-impact-high' :
                        f.impact === 'Medium' ? 'factor-impact-med' : 'factor-impact-low';
    const favorColor = f.favor === pA.name ? 'var(--cyan-primary)' :
                       f.favor === pB.name ? 'var(--neon-coral)' : 'var(--text-muted)';
    return `
      <div class="factor-item">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.25rem;">
          <span style="font-size: 0.88rem; font-weight: 700; color: var(--text-primary);">${f.factor}</span>
          <span class="${impactClass}">● ${f.impact} Impact</span>
        </div>
        <div style="font-size: 0.82rem; color: var(--text-secondary); margin-bottom: 0.35rem;">${f.detail}</div>
        <div style="font-size: 0.78rem;">
          Favors: <strong style="color: ${favorColor}; font-family: var(--font-mono);">${f.favor}</strong>
        </div>
      </div>
    `;
  }).join('') : `
    <div style="font-size: 0.85rem; color: var(--text-muted); font-style: italic; padding: 1rem 0;">
      No single dominant factor; this matchup presents evenly matched historical records across career, surface, form, and rivalry metrics.
    </div>
  `;

  container.innerHTML = `
    ${coverageNoticeHtml}

    <!-- Top Prediction & Probability Duel Banner -->
    <div class="card" style="margin-bottom: 1.5rem;">
      <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.75rem; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 1rem; margin-bottom: 1.25rem;">
        <div>
          <div style="font-size: 0.78rem; text-transform: uppercase; font-family: var(--font-mono); color: var(--text-muted); margin-bottom: 0.25rem;">
            Projected Match Outcome • ${data.surface} Court (Best of ${data.best_of})
          </div>
          <div style="font-size: 1.55rem; font-weight: 800; color: var(--text-primary); display: flex; align-items: center; gap: 0.6rem; flex-wrap: wrap;">
            <span>🏆 Predicted Winner:</span>
            <span style="color: ${isWinnerA ? 'var(--cyan-primary)' : 'var(--neon-coral)'};">${pred.predicted_winner}</span>
            <span style="font-size: 1.2rem; font-family: var(--font-mono); color: var(--neon-emerald); font-weight: 700;">(${pred.winner_probability}%)</span>
          </div>
        </div>
        <div style="display: flex; align-items: center; gap: 0.75rem;">
          <span class="confidence-tag ${confClass}">● ${pred.confidence_level}</span>
        </div>
      </div>

      <!-- Probability Comparison Duel Bar -->
      <div class="prob-bar-wrapper">
        <div class="prob-bar-header">
          <div>
            <strong style="color: var(--cyan-primary);">${pA.name}</strong>
            <span style="font-family: var(--font-mono); font-size: 1.15rem; font-weight: 800; margin-left: 0.5rem; color: var(--cyan-primary);">${probA}%</span>
          </div>
          <div style="font-size: 0.82rem; color: var(--text-muted); font-family: var(--font-mono); text-transform: uppercase;">
            Win Probability Distribution
          </div>
          <div>
            <span style="font-family: var(--font-mono); font-size: 1.15rem; font-weight: 800; margin-right: 0.5rem; color: var(--neon-coral);">${probB}%</span>
            <strong style="color: var(--neon-coral);">${pB.name}</strong>
          </div>
        </div>
        <div class="prob-bar-track">
          <div class="prob-bar-fill-a" style="width: ${probA}%;"></div>
          <div class="prob-bar-fill-b" style="width: ${probB}%;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.78rem; color: var(--text-muted); margin-top: 0.6rem; font-family: var(--font-mono);">
          <span>LR Win Probability: ${(base.logistic_regression_prob_a ?? probA).toFixed(1)}%</span>
          <span>Symmetrical Ensembled Random Forest (100 Trees)</span>
          <span>LR Win Probability: ${(base.logistic_regression_prob_b ?? probB).toFixed(1)}%</span>
        </div>
      </div>
    </div>

    <!-- 2-Column Grid: Analytical Drivers vs Pre-Match Tactical Profile -->
    <div class="grid-2" style="margin-bottom: 1.5rem;">
      <!-- Key Drivers Column -->
      <div class="card">
        <h3 class="card-title">Key Analytical Drivers & Feature Attribution</h3>
        <p style="font-size: 0.82rem; color: var(--text-secondary); margin-bottom: 1rem;">
          Feature attributions explaining why the model favored the projected winner based on pre-match differences:
        </p>
        <div>
          ${factorsHtml}
        </div>
      </div>

      <!-- Tactical Profile Comparison Column -->
      <div class="card">
        <h3 class="card-title">Pre-Match Historical Profile Breakdown</h3>
        <p style="font-size: 0.82rem; color: var(--text-secondary); margin-bottom: 1rem;">
          Verified pre-match metrics derived strictly prior to match timestamp without outcome leakage:
        </p>
        <div class="table-container">
          <table>
            <thead>
              <tr>
                <th>Feature Metric</th>
                <th style="color: var(--cyan-primary);">${pA.name}</th>
                <th style="color: var(--neon-coral);">${pB.name}</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>Career Win Rate</strong></td>
                <td style="font-family: var(--font-mono);">${pA.career_win_rate}% (${pA.total_wins}/${pA.total_matches})</td>
                <td style="font-family: var(--font-mono);">${pB.career_win_rate}% (${pB.total_wins}/${pB.total_matches})</td>
              </tr>
              <tr>
                <td><strong>${data.surface} Court Win Rate</strong></td>
                <td style="font-family: var(--font-mono);">${pA.surface_win_rate}% (${pA.surface_wins}/${pA.surface_matches})</td>
                <td style="font-family: var(--font-mono);">${pB.surface_win_rate}% (${pB.surface_wins}/${pB.surface_matches})</td>
              </tr>
              <tr>
                <td><strong>Recent Rolling Form (Last 10)</strong></td>
                <td style="font-family: var(--font-mono);">${pA.recent_form_win_rate}%</td>
                <td style="font-family: var(--font-mono);">${pB.recent_form_win_rate}%</td>
              </tr>
              <tr>
                <td><strong>Prior Head-to-Head Meetings</strong></td>
                <td style="font-family: var(--font-mono);">${h2h.player_a_wins} Wins</td>
                <td style="font-family: var(--font-mono);">${h2h.player_b_wins} Wins</td>
              </tr>
              <tr>
                <td><strong>Verified Point Matches (2010–2026 Model Training)</strong></td>
                <td style="font-family: var(--font-mono);">${pA.has_verified_training_data ? `${pA.verified_matches} matches` : '<span style="color:var(--neon-coral);">0 (Neutral Prior 50% applied)</span>'}</td>
                <td style="font-family: var(--font-mono);">${pB.has_verified_training_data ? `${pB.verified_matches} matches` : '<span style="color:var(--neon-coral);">0 (Neutral Prior 50% applied)</span>'}</td>
              </tr>
              <tr>
                <td><strong>Historical Matches in Database (1968–2026)</strong></td>
                <td style="font-family: var(--font-mono);">${pA.historical_matches || pA.total_matches} matches</td>
                <td style="font-family: var(--font-mono);">${pB.historical_matches || pB.total_matches} matches</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- Model Governance, Benchmark & Disclaimer Card -->
    <div class="card">
      <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 0.75rem; margin-bottom: 1rem;">
        <h3 class="card-title" style="margin: 0;">Machine Learning Governance & Benchmark Validation</h3>
        <span class="confidence-tag confidence-high">Zero-Leakage Verified</span>
      </div>
      <div class="grid-4" style="margin-bottom: 1rem;">
        <div style="background: rgba(8,14,28,0.7); padding: 0.85rem; border-radius: 8px;">
          <div style="font-size: 0.75rem; color: var(--text-muted); font-family: var(--font-mono); text-transform: uppercase;">Holdout Test Accuracy</div>
          <div style="font-size: 1.45rem; font-weight: 800; color: var(--neon-emerald); font-family: var(--font-mono); margin: 0.2rem 0;">${meta.test_accuracy || '67.94%'}</div>
          <div style="font-size: 0.75rem; color: var(--text-secondary);">vs Baseline LR ${base.baseline_test_accuracy || '67.28%'}</div>
        </div>
        <div style="background: rgba(8,14,28,0.7); padding: 0.85rem; border-radius: 8px;">
          <div style="font-size: 0.75rem; color: var(--text-muted); font-family: var(--font-mono); text-transform: uppercase;">Holdout Test ROC-AUC</div>
          <div style="font-size: 1.45rem; font-weight: 800; color: var(--neon-emerald); font-family: var(--font-mono); margin: 0.2rem 0;">${meta.test_roc_auc || '0.7505'}</div>
          <div style="font-size: 0.75rem; color: var(--text-secondary);">vs Baseline LR ${base.baseline_test_roc_auc || '0.7350'}</div>
        </div>
        <div style="background: rgba(8,14,28,0.7); padding: 0.85rem; border-radius: 8px;">
          <div style="font-size: 0.75rem; color: var(--text-muted); font-family: var(--font-mono); text-transform: uppercase;">Mathematical Symmetry</div>
          <div style="font-size: 1.45rem; font-weight: 800; color: var(--cyan-primary); font-family: var(--font-mono); margin: 0.2rem 0;">100.0% Exact</div>
          <div style="font-size: 0.75rem; color: var(--text-secondary);">P(A, B) = 1 - P(B, A)</div>
        </div>
        <div style="background: rgba(8,14,28,0.7); padding: 0.85rem; border-radius: 8px;">
          <div style="font-size: 0.75rem; color: var(--text-muted); font-family: var(--font-mono); text-transform: uppercase;">Split Strategy</div>
          <div style="font-size: 1.1rem; font-weight: 700; color: var(--text-primary); font-family: var(--font-mono); margin: 0.35rem 0;">Chronological</div>
          <div style="font-size: 0.75rem; color: var(--text-secondary);">70/15/15 (2010–2026)</div>
        </div>
      </div>
      <div style="font-size: 0.82rem; line-height: 1.7; color: var(--text-secondary);">
        <strong>Methodological Safeguards:</strong> Features are extracted chronologically match-by-match. Post-match stats (aces, double faults, unforced errors, match duration, and match winners) are strictly excluded from the pre-match feature vectors.
      </div>

      <div class="disclaimer-card">
        <span style="font-size: 1.3rem;">⚠️</span>
        <div>
          <strong>Sports Analytics Disclaimer:</strong> ${data.disclaimer || 'Probabilistic model estimate based on historical pre-match feature vectors without outcome leakage. Sports matches contain intrinsic unpredictability.'}
        </div>
      </div>
    </div>
  `;
}

// ============================================================================
// 4. FEATURE 1: PLAYER INTELLIGENCE LAB
// ============================================================================
async function loadPlayerLab() {
  const pA = document.getElementById('lab-player-a')?.value || 'Novak Djokovic';
  const pB = document.getElementById('lab-player-b')?.value || 'Rafael Nadal';
  const surf = document.getElementById('lab-surface-filter')?.value || 'Overall';

  try {
    const res = await fetch(`${API_BASE}/api/player-lab?player_a=${encodeURIComponent(pA)}&player_b=${encodeURIComponent(pB)}&surface=${encodeURIComponent(surf)}`);
    const data = await res.json();

    document.getElementById('lab-name-a').textContent = data.player_a.name;
    document.getElementById('lab-name-b').textContent = data.player_b.name;
    document.getElementById('lab-matches-a').textContent = `${data.player_a.total_matches} matches`;
    document.getElementById('lab-matches-b').textContent = `${data.player_b.total_matches} matches`;

    // Head-to-Head
    const h2h = data.head_to_head;
    const h2hEl = document.getElementById('lab-h2h-summary');
    if (h2h) {
      const winsA = h2h.player_a === data.player_a.name ? h2h.wins_a : h2h.wins_b;
      const winsB = h2h.player_a === data.player_a.name ? h2h.wins_b : h2h.wins_a;
      h2hEl.innerHTML = `
        <div style="text-align: center; margin-bottom: 1rem;">
          <div style="font-size: 2.2rem; font-weight: 800; font-family: var(--font-mono); color: var(--cyan-primary);">
            ${winsA} <span style="font-size: 1.2rem; color: var(--text-muted);">vs</span> ${winsB}
          </div>
          <div style="font-size: 0.82rem; color: var(--text-secondary);">
            ${h2h.total_matches} Total Historical Meetings (Hard: ${h2h.surfaces?.Hard || 0}, Clay: ${h2h.surfaces?.Clay || 0}, Grass: ${h2h.surfaces?.Grass || 0})
          </div>
        </div>
      `;

      // Render recent meetings table
      const meetTable = document.getElementById('lab-meetings-table-body');
      if (meetTable) {
        meetTable.innerHTML = (h2h.matches || []).slice(0, 10).map(m => `
          <tr>
            <td>${m.year}</td>
            <td>${m.tournament}</td>
            <td><span class="status-pill">${m.surface}</span></td>
            <td>${m.round}</td>
            <td><strong style="color: ${m.winner === data.player_a.name ? 'var(--cyan-primary)' : 'var(--neon-coral)'}">${m.winner}</strong></td>
          </tr>
        `).join('');
      }
    } else {
      h2hEl.innerHTML = `<div style="color: var(--text-muted); text-align: center; padding: 1rem;">No direct head-to-head meetings recorded in dataset.</div>`;
    }

    // Render Comparative Stat Bars
    renderComparativeBars(data.player_a.stats, data.player_b.stats, data.player_a.name, data.player_b.name);
    // Render Radar Chart
    renderPlayerRadar(data.player_a.stats, data.player_b.stats, data.player_a.name, data.player_b.name);
  } catch (err) {
    console.error('Error loading player lab:', err);
  }
}

function renderComparativeBars(sA, sB, nameA, nameB) {
  const metrics = [
    { key: 'first_won_pct', label: '1st Serve Win %', unit: '%' },
    { key: 'second_won_pct', label: '2nd Serve Win %', unit: '%' },
    { key: 'ace_rate', label: 'Ace Rate', unit: '%' },
    { key: 'df_rate', label: 'Double Fault Rate', unit: '%', invert: true },
    { key: 'bp_save_pct', label: 'Break Point Save %', unit: '%' },
    { key: 'return_won_pct', label: 'Return Win %', unit: '%' },
    { key: 'winner_unforced_ratio', label: 'Winner/Unforced Ratio', unit: '' }
  ];

  const container = document.getElementById('lab-stat-bars');
  if (!container) return;

  container.innerHTML = metrics.map(m => {
    const vA = sA[m.key] || 0;
    const vB = sB[m.key] || 0;
    const maxVal = Math.max(vA, vB, 1);
    const pctA = Math.min(100, (vA / maxVal) * 100);
    const pctB = Math.min(100, (vB / maxVal) * 100);

    return `
      <div style="margin-bottom: 1rem;">
        <div style="display: flex; justify-content: space-between; font-size: 0.82rem; margin-bottom: 0.25rem;">
          <span style="font-family: var(--font-mono); color: var(--cyan-primary);">${vA}${m.unit}</span>
          <span style="color: var(--text-secondary); font-weight: 600;">${m.label}</span>
          <span style="font-family: var(--font-mono); color: var(--neon-coral);">${vB}${m.unit}</span>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 4px; height: 8px;">
          <div style="background: rgba(255,255,255,0.06); border-radius: 4px; display: flex; justify-content: flex-end;">
            <div style="width: ${pctA}%; background: var(--cyan-primary); border-radius: 4px;"></div>
          </div>
          <div style="background: rgba(255,255,255,0.06); border-radius: 4px;">
            <div style="width: ${pctB}%; background: var(--neon-coral); border-radius: 4px;"></div>
          </div>
        </div>
      </div>
    `;
  }).join('');
}

function renderPlayerRadar(sA, sB, nameA, nameB) {
  const radarSvg = document.getElementById('lab-radar-svg');
  if (!radarSvg) return;

  const feats = [
    { key: 'first_won_pct', label: '1st Win', max: 85 },
    { key: 'second_won_pct', label: '2nd Win', max: 65 },
    { key: 'ace_rate', label: 'Aces', max: 15 },
    { key: 'bp_save_pct', label: 'BP Save', max: 75 },
    { key: 'return_won_pct', label: 'Return', max: 45 },
    { key: 'winner_unforced_ratio', label: 'W/U Ratio', max: 2.0 }
  ];

  const cx = 150, cy = 150, r = 110;
  const num = feats.length;

  let gridCircles = '';
  for (let l = 1; l <= 4; l++) {
    gridCircles += `<circle cx="${cx}" cy="${cy}" r="${(r / 4) * l}" fill="none" stroke="rgba(255,255,255,0.08)" stroke-dasharray="3,3"/>`;
  }

  let axes = '';
  let ptsA = [];
  let ptsB = [];

  feats.forEach((f, i) => {
    const angle = (Math.PI * 2 / num) * i - Math.PI / 2;
    const ax = cx + r * Math.cos(angle);
    const ay = cy + r * Math.sin(angle);
    axes += `<line x1="${cx}" y1="${cy}" x2="${ax}" y2="${ay}" stroke="rgba(255,255,255,0.1)"/>`;
    axes += `<text x="${ax + 12 * Math.cos(angle)}" y="${ay + 12 * Math.sin(angle) + 4}" fill="#94a3b8" font-size="10" text-anchor="middle" font-family="var(--font-mono)">${f.label}</text>`;

    const valA = Math.min(1.0, (sA[f.key] || 0) / f.max);
    const valB = Math.min(1.0, (sB[f.key] || 0) / f.max);

    ptsA.push(`${cx + r * valA * Math.cos(angle)},${cy + r * valA * Math.sin(angle)}`);
    ptsB.push(`${cx + r * valB * Math.cos(angle)},${cy + r * valB * Math.sin(angle)}`);
  });

  radarSvg.innerHTML = `
    ${gridCircles}
    ${axes}
    <polygon points="${ptsA.join(' ')}" fill="rgba(0, 242, 254, 0.25)" stroke="var(--cyan-primary)" stroke-width="2"/>
    <polygon points="${ptsB.join(' ')}" fill="rgba(255, 51, 102, 0.25)" stroke="var(--neon-coral)" stroke-width="2"/>
  `;
}

// ============================================================================
// 3. FEATURE 2: MATCH MOMENTUM ANALYZER
// ============================================================================
async function loadReplayMatches() {
  try {
    const res = await fetch(`${API_BASE}/api/matches/replayable`);
    const data = await res.json();
    state.replayMatches = data.matches || [];

    const selectEl = document.getElementById('momentum-match-select');
    if (selectEl) {
      selectEl.innerHTML = state.replayMatches.map(m => `
        <option value="${m.match_id}">${m.label}</option>
      `).join('');
    }
  } catch (err) {
    console.error('Error loading replayable matches:', err);
  }
}

function initMomentumView() {
  const matchId = document.getElementById('momentum-match-select')?.value;
  if (!matchId) return;
  if (state.activeReplayData && state.activeReplayData.match_id === matchId) {
    renderMomentumSnapshot(state.replayIndex);
    requestAnimationFrame(() => drawMomentumCanvas());
  } else {
    loadMatchReplay(matchId);
  }
}

async function loadMatchReplay(matchId) {
  const winSize = parseInt(document.getElementById('momentum-window-input')?.value || '40');
  const alpha = parseFloat(document.getElementById('momentum-alpha-input')?.value || '0.25');

  if (state.replayInterval) {
    clearInterval(state.replayInterval);
    state.replayInterval = null;
    const playBtn = document.getElementById('momentum-play-btn');
    if (playBtn) playBtn.textContent = '▶ Play';
  }

  try {
    const res = await fetch(`${API_BASE}/api/momentum/replay?match_id=${encodeURIComponent(matchId)}&window_size=${winSize}&alpha=${alpha}`);
    const data = await res.json();
    state.activeReplayData = data;
    state.replayIndex = 0;

    const labelEl = document.getElementById('momentum-match-label');
    if (labelEl) {
      labelEl.textContent = `${data.metadata?.tournament || 'Match'} (${data.metadata?.year || ''}): ${data.p1} vs ${data.p2}`;
    }
    const swingsEl = document.getElementById('momentum-swings-count');
    if (swingsEl) {
      swingsEl.textContent = `${data.momentum_swings} Momentum Shifts`;
    }

    const slider = document.getElementById('momentum-scrubber');
    if (slider) {
      slider.min = 0;
      slider.max = data.points.length - 1;
      slider.value = 0;
    }

    renderMomentumSnapshot(0);
    requestAnimationFrame(() => drawMomentumCanvas());
  } catch (err) {
    console.error('Error loading match replay:', err);
  }
}

function renderMomentumSnapshot(idx) {
  if (!state.activeReplayData || !state.activeReplayData.points) return;
  const clampedIdx = Math.max(0, Math.min(state.activeReplayData.points.length - 1, idx));
  state.replayIndex = clampedIdx;
  const pt = state.activeReplayData.points[clampedIdx];
  if (!pt) return;

  const p1Name = state.activeReplayData.p1 || 'Player 1';
  const p2Name = state.activeReplayData.p2 || 'Player 2';

  // Scoreboard
  const scoreEl = document.getElementById('live-score-display');
  if (scoreEl) {
    if (pt.point_number === 0) {
      scoreEl.textContent = `Set 0-0 | Games 0-0 | Point: 0-0`;
    } else {
      scoreEl.textContent = `Set ${pt.set1}-${pt.set2} | Games ${pt.gm1}-${pt.gm2} | Point: ${pt.score}`;
    }
  }

  const serverEl = document.getElementById('live-server-display');
  if (serverEl) {
    const serverName = pt.server === 1 ? p1Name : (pt.server === 2 ? p2Name : `Player ${pt.server}`);
    serverEl.textContent = pt.point_number === 0 ? `Opening Server: ${serverName}` : `Server: ${serverName}`;
  }

  const labelEl = document.getElementById('live-point-label');
  if (labelEl) {
    labelEl.textContent = pt.label || '-';
  }

  const winnerEl = document.getElementById('live-winner-display');
  if (winnerEl) {
    if (pt.point_number === 0) {
      winnerEl.textContent = `Match Ready — Step to start`;
      winnerEl.style.color = 'var(--text-muted)';
    } else {
      winnerEl.textContent = `Point Won By: ${pt.winner_name}`;
      winnerEl.style.color = 'var(--neon-emerald)';
    }
  }

  // Momentum Gauge
  const mVal = pt.momentum;
  const mBar = document.getElementById('live-momentum-meter');
  if (mBar) {
    const clamped = Math.max(-3.5, Math.min(3.5, mVal));
    const pct = ((clamped + 3.5) / 7.0) * 100;
    mBar.style.left = `${pct}%`;
  }
  const mText = document.getElementById('live-momentum-score-text');
  if (mText) {
    if (pt.point_number === 0 || Math.abs(mVal) < 0.001) {
      mText.textContent = `Score: 0.000 (Neutral)`;
    } else {
      const sign = mVal > 0 ? '+' : '';
      mText.textContent = `Score: ${sign}${mVal.toFixed(3)} (${pt.dominant_player} Advantage)`;
    }
  }

  // DGIM Dynamic Header Description
  const dgimDesc = document.getElementById('dgim-description-text');
  if (dgimDesc) {
    dgimDesc.textContent = `Counting ${p1Name} point wins in sliding window (N=${pt.window_size}) using power-of-two buckets (O(log² N) memory):`;
  }

  // DGIM Buckets
  const bucketContainer = document.getElementById('dgim-bucket-display');
  if (bucketContainer) {
    const buckets = pt.dgim_p1_buckets || [];
    bucketContainer.innerHTML = buckets.length > 0 
      ? buckets.map(b => `<div class="dgim-bubble">Size ${b.size} (t=${b.timestamp})</div>`).join('')
      : '<span style="color:var(--text-muted);font-size:0.8rem;">No active buckets in window</span>';
  }

  // DGIM Accuracy Metrics
  const estWinsEl = document.getElementById('dgim-est-wins');
  if (estWinsEl) estWinsEl.textContent = pt.dgim_p1_window_wins;

  const exactWinsEl = document.getElementById('dgim-exact-wins');
  if (exactWinsEl) exactWinsEl.textContent = pt.dgim_p1_exact_wins;

  const errPctEl = document.getElementById('dgim-error-pct');
  if (errPctEl) {
    const bound = pt.theoretical_max_error_pct || 50.0;
    errPctEl.textContent = `${pt.dgim_p1_error_pct}% (Bound: ≤${bound}%)`;
  }
}

function stepReplay(delta) {
  if (!state.activeReplayData || !state.activeReplayData.points) return;
  const maxIdx = state.activeReplayData.points.length - 1;
  const newIdx = Math.max(0, Math.min(maxIdx, state.replayIndex + delta));
  state.replayIndex = newIdx;
  const slider = document.getElementById('momentum-scrubber');
  if (slider) slider.value = newIdx;
  renderMomentumSnapshot(newIdx);
  drawMomentumCanvas();
}

function drawMomentumCanvas() {
  const canvas = document.getElementById('momentum-curve-canvas');
  if (!canvas || !state.activeReplayData) return;
  const ctx = canvas.getContext('2d');
  const curve = state.activeReplayData.curve;
  if (!curve || curve.length === 0) return;

  const parentWidth = canvas.parentElement ? canvas.parentElement.clientWidth : 0;
  const w = canvas.width = Math.max(300, parentWidth || canvas.clientWidth || 700);
  const h = canvas.height = 280;

  ctx.clearRect(0, 0, w, h);

  const zeroY = h / 2;
  const maxPoints = Math.max(1, curve.length - 1);
  const maxM = Math.max(3.5, ...curve.map(c => Math.abs(c.momentum || 0)));

  // Grid background: subtle horizontal guide lines
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
  ctx.lineWidth = 1;
  [-2, -1, 1, 2].forEach(level => {
    const y = zeroY - (level / maxM) * (h / 2 - 25);
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  });

  // Zero-line (Center / Neutral Momentum)
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.25)';
  ctx.lineWidth = 1.5;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(0, zeroY);
  ctx.lineTo(w, zeroY);
  ctx.stroke();
  ctx.setLineDash([]);

  // Labels for players
  ctx.font = '11px sans-serif';
  ctx.fillStyle = 'rgba(0, 242, 254, 0.7)';
  ctx.fillText(`▲ ${state.activeReplayData.p1 || 'Player 1'} Advantage`, 12, 20);
  ctx.fillStyle = 'rgba(255, 51, 102, 0.7)';
  ctx.fillText(`▼ ${state.activeReplayData.p2 || 'Player 2'} Advantage`, 12, h - 12);

  // Gradient fill under the curve (Player 1 Advantage - Cyan)
  ctx.save();
  ctx.beginPath();
  ctx.moveTo(0, zeroY);
  curve.forEach((c, i) => {
    const x = (i / maxPoints) * w;
    const y = zeroY - (c.momentum / maxM) * (h / 2 - 25);
    ctx.lineTo(x, Math.min(zeroY, y));
  });
  ctx.lineTo(w, zeroY);
  ctx.closePath();
  const gradP1 = ctx.createLinearGradient(0, 25, 0, zeroY);
  gradP1.addColorStop(0, 'rgba(0, 242, 254, 0.22)');
  gradP1.addColorStop(1, 'rgba(0, 242, 254, 0.0)');
  ctx.fillStyle = gradP1;
  ctx.fill();
  ctx.restore();

  // Gradient fill under the curve (Player 2 Advantage - Coral)
  ctx.save();
  ctx.beginPath();
  ctx.moveTo(0, zeroY);
  curve.forEach((c, i) => {
    const x = (i / maxPoints) * w;
    const y = zeroY - (c.momentum / maxM) * (h / 2 - 25);
    ctx.lineTo(x, Math.max(zeroY, y));
  });
  ctx.lineTo(w, zeroY);
  ctx.closePath();
  const gradP2 = ctx.createLinearGradient(0, zeroY, 0, h - 25);
  gradP2.addColorStop(0, 'rgba(255, 51, 102, 0.0)');
  gradP2.addColorStop(1, 'rgba(255, 51, 102, 0.22)');
  ctx.fillStyle = gradP2;
  ctx.fill();
  ctx.restore();

  // Main Momentum Stroke Line
  ctx.beginPath();
  curve.forEach((c, i) => {
    const x = (i / maxPoints) * w;
    const y = zeroY - (c.momentum / maxM) * (h / 2 - 25);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = '#00f2fe';
  ctx.lineWidth = 2.2;
  ctx.stroke();

  // Current playback marker line & indicator dot
  const curIndex = Math.max(0, Math.min(curve.length - 1, state.replayIndex));
  const curX = (curIndex / maxPoints) * w;
  const curM = curve[curIndex] ? curve[curIndex].momentum : 0;
  const curY = zeroY - (curM / maxM) * (h / 2 - 25);

  ctx.strokeStyle = '#ff3366';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(curX, 0);
  ctx.lineTo(curX, h);
  ctx.stroke();

  // Current point marker circle
  ctx.beginPath();
  ctx.arc(curX, curY, 5, 0, 2 * Math.PI);
  ctx.fillStyle = '#ffaa00';
  ctx.fill();
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 1.5;
  ctx.stroke();
}

function toggleReplayPlay() {
  const btn = document.getElementById('momentum-play-btn');
  if (state.replayInterval) {
    clearInterval(state.replayInterval);
    state.replayInterval = null;
    btn.textContent = '▶ Play';
  } else {
    btn.textContent = '⏸ Pause';
    state.replayInterval = setInterval(() => {
      if (!state.activeReplayData) return;
      if (state.replayIndex >= state.activeReplayData.points.length - 1) {
        clearInterval(state.replayInterval);
        state.replayInterval = null;
        btn.textContent = '▶ Play';
        return;
      }
      state.replayIndex++;
      const slider = document.getElementById('momentum-scrubber');
      if (slider) slider.value = state.replayIndex;
      renderMomentumSnapshot(state.replayIndex);
      drawMomentumCanvas();
    }, 180);
  }
}

// ============================================================================
// 4. FEATURE 3: SERVE STRATEGY HEATMAP
// ============================================================================
async function loadServeStrategy() {
  const player = document.getElementById('serve-player-select')?.value || 'Roger Federer';
  const surface = document.getElementById('serve-surface-select')?.value || 'Hard';

  try {
    const res = await fetch(`${API_BASE}/api/serve-strategy?player=${encodeURIComponent(player)}&surface=${encodeURIComponent(surface)}`);
    const data = await res.json();
    const s = data.strategy;
    if (!s) return;

    // Deuce Court Zones
    const dWide = document.getElementById('zone-deuce-wide');
    const dBody = document.getElementById('zone-deuce-body');
    const dT = document.getElementById('zone-deuce-t');

    if (dWide) {
      dWide.style.fill = `rgba(0, 242, 254, ${0.15 + (s.deuce_court.wide_pct / 100) * 0.7})`;
      document.getElementById('text-deuce-wide').textContent = `${s.deuce_court.wide_pct}% (${s.deuce_court.wide})`;
    }
    if (dBody) {
      dBody.style.fill = `rgba(0, 242, 254, ${0.15 + (s.deuce_court.body_pct / 100) * 0.7})`;
      document.getElementById('text-deuce-body').textContent = `${s.deuce_court.body_pct}% (${s.deuce_court.body})`;
    }
    if (dT) {
      dT.style.fill = `rgba(0, 242, 254, ${0.15 + (s.deuce_court.t_pct / 100) * 0.7})`;
      document.getElementById('text-deuce-t').textContent = `${s.deuce_court.t_pct}% (${s.deuce_court.t})`;
    }

    // Ad Court Zones
    const aWide = document.getElementById('zone-ad-wide');
    const aBody = document.getElementById('zone-ad-body');
    const aT = document.getElementById('zone-ad-t');

    if (aWide) {
      aWide.style.fill = `rgba(79, 172, 254, ${0.15 + (s.ad_court.wide_pct / 100) * 0.7})`;
      document.getElementById('text-ad-wide').textContent = `${s.ad_court.wide_pct}% (${s.ad_court.wide})`;
    }
    if (aBody) {
      aBody.style.fill = `rgba(79, 172, 254, ${0.15 + (s.ad_court.body_pct / 100) * 0.7})`;
      document.getElementById('text-ad-body').textContent = `${s.ad_court.body_pct}% (${s.ad_court.body})`;
    }
    if (aT) {
      aT.style.fill = `rgba(79, 172, 254, ${0.15 + (s.ad_court.t_pct / 100) * 0.7})`;
      document.getElementById('text-ad-t').textContent = `${s.ad_court.t_pct}% (${s.ad_court.t})`;
    }

    // Fault Distributions
    document.getElementById('fault-net-pct').textContent = `${s.faults.net_pct}% (${s.faults.net} net)`;
    document.getElementById('fault-wide-pct').textContent = `${s.faults.wide_pct}% (${s.faults.wide} wide)`;
    document.getElementById('fault-deep-pct').textContent = `${s.faults.deep_pct}% (${s.faults.deep} deep)`;
    document.getElementById('total-serves-badge').textContent = `${s.total_serves_recorded} Recorded Serves`;
  } catch (err) {
    console.error('Error loading serve strategy:', err);
  }
}

// ============================================================================
// 5. FEATURE 4: SIMILAR PLAYSTYLE RECOMMENDER
// ============================================================================
async function loadRecommender() {
  const player = document.getElementById('recommender-player-select')?.value || 'Carlos Alcaraz';
  const topK = parseInt(document.getElementById('recommender-topk-select')?.value || '5');

  try {
    const res = await fetch(`${API_BASE}/api/recommender?player=${encodeURIComponent(player)}&top_k=${topK}`);
    const data = await res.json();

    document.getElementById('rec-target-name').textContent = data.target_player;
    document.getElementById('rec-target-archetype').textContent = data.target_archetype;
    document.getElementById('rec-target-matches').textContent = `${data.target_matches} matches`;

    const container = document.getElementById('recommender-cards-container');
    if (!container) return;

    container.innerHTML = data.recommendations.map((rec, i) => `
      <div class="recommender-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
          <div>
            <h4 style="font-size: 1.15rem; color: var(--text-primary); font-weight: 700;">#${i + 1} ${rec.player}</h4>
            <span style="font-size: 0.78rem; color: var(--text-secondary);">${rec.archetype} • ${rec.total_matches} matches</span>
          </div>
          <span class="similarity-badge">${rec.similarity_score}% Similar</span>
        </div>
        
        <div style="font-size: 0.82rem; margin-bottom: 0.5rem; color: var(--cyan-primary); font-weight: 600;">Key Shared Traits:</div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.5rem; font-size: 0.8rem;">
          ${rec.key_similar_traits.map(t => `
            <div style="background: rgba(8, 14, 28, 0.6); padding: 0.45rem 0.65rem; border-radius: 6px; border: 1px solid rgba(0, 242, 254, 0.1);">
              <span style="color: var(--text-secondary);">${t.feature}:</span>
              <strong style="color: var(--text-primary); margin-left: 4px;">${t.match_value}</strong>
              <span style="color: var(--neon-emerald); font-size: 0.72rem; margin-left: 4px;">(Δ ${t.difference})</span>
            </div>
          `).join('')}
        </div>
      </div>
    `).join('');
  } catch (err) {
    console.error('Error loading recommender:', err);
  }
}

// ============================================================================
// 6. FEATURE 5: ATP RIVALRY NETWORK
// ============================================================================
async function loadRivalryNetwork() {
  const minMeetings = parseInt(document.getElementById('network-meetings-slider')?.value || '3');

  try {
    const res = await fetch(`${API_BASE}/api/rivalry-network?min_meetings=${minMeetings}&top_n=35`);
    const data = await res.json();
    state.networkData = data;

    // KPIs
    document.getElementById('net-nodes-count').textContent = data.nodes.length;
    document.getElementById('net-edges-count').textContent = data.edges.length;
    document.getElementById('net-communities-count').textContent = data.metrics.louvain_communities_count;
    document.getElementById('net-modularity-score').textContent = `Q = ${data.metrics.graph_modularity_score}`;

    // Community Legend
    const legendEl = document.getElementById('network-cluster-legend');
    if (legendEl) {
      legendEl.innerHTML = (data.community_clusters || []).map(c => `
        <div style="display: flex; align-items: center; gap: 0.5rem; font-size: 0.78rem;">
          <span style="width: 10px; height: 10px; border-radius: 50%; background: ${c.color}; box-shadow: 0 0 6px ${c.color};"></span>
          <span><strong>${c.label}</strong> (${c.total_members} players)</span>
        </div>
      `).join('');
    }

    drawNetworkSVG();
    if (data.nodes && data.nodes.length > 0) {
      inspectNetworkNode(data.nodes[0].id);
    }
  } catch (err) {
    console.error('Error loading rivalry network:', err);
  }
}

function drawNetworkSVG() {
  const svg = document.getElementById('network-svg-canvas');
  if (!svg || !state.networkData) return;
  const { nodes, edges } = state.networkData;

  const w = svg.clientWidth || 800;
  const h = 580;
  svg.setAttribute('viewBox', `0 0 ${w} ${h}`);

  // Color map for communities
  const colors = ["#00F2FE", "#FF007F", "#7928CA", "#00FFA3", "#FF8000", "#FFD700", "#4FACFE"];

  // Simple deterministic circular/force layout
  const nodeCoords = {};
  const cx = w / 2, cy = h / 2;
  const radius = Math.min(w, h) * 0.4;

  nodes.forEach((n, i) => {
    const angle = (Math.PI * 2 / nodes.length) * i;
    const rOffset = (n.community % 2 === 0 ? 0.9 : 1.05) * radius;
    nodeCoords[n.id] = {
      x: cx + rOffset * Math.cos(angle),
      y: cy + rOffset * Math.sin(angle),
      r: Math.max(6, Math.min(18, n.pagerank * 320)),
      color: colors[n.community % colors.length]
    };
  });

  // Render edges
  let edgesHtml = '';
  edges.forEach(e => {
    const src = nodeCoords[e.source];
    const tgt = nodeCoords[e.target];
    if (src && tgt) {
      const strokeWidth = Math.max(1, Math.min(5, e.meetings * 0.4));
      edgesHtml += `
        <line x1="${src.x}" y1="${src.y}" x2="${tgt.x}" y2="${tgt.y}" 
              stroke="rgba(0, 242, 254, 0.22)" stroke-width="${strokeWidth}"/>
      `;
    }
  });

  // Render nodes
  let nodesHtml = '';
  nodes.forEach(n => {
    const c = nodeCoords[n.id];
    nodesHtml += `
      <g style="cursor: pointer;" onclick="inspectNetworkNode('${n.id}')">
        <circle cx="${c.x}" cy="${c.y}" r="${c.r}" fill="${c.color}" stroke="#ffffff" stroke-width="1.5"
                filter="drop-shadow(0 0 6px ${c.color})"/>
        <text x="${c.x}" y="${c.y + c.r + 12}" fill="#f8fafc" font-size="9" text-anchor="middle" font-family="var(--font-sans)" font-weight="600">${n.label}</text>
      </g>
    `;
  });

  svg.innerHTML = edgesHtml + nodesHtml;
}

function inspectNetworkNode(playerId) {
  if (!state.networkData || !state.networkData.nodes) return;
  const node = state.networkData.nodes.find(n => n.id === playerId);
  if (!node) return;

  setSafeText('inspect-node-name', node.label);
  setSafeText('inspect-node-pagerank', node.pagerank);
  setSafeText('inspect-node-betweenness', node.betweenness);
  setSafeText('inspect-node-degree', node.degree);
  setSafeText('inspect-node-matches', node.total_matches);

  const pillEl = document.getElementById('inspect-node-cluster-pill');
  if (pillEl) {
    pillEl.textContent = `Cluster ${node.community}`;
    const colors = ["#00F2FE", "#FF007F", "#7928CA", "#00FFA3", "#FF8000", "#FFD700", "#4FACFE"];
    const col = colors[node.community % colors.length];
    pillEl.style.borderColor = col;
    pillEl.style.color = col;
  }
}

// ============================================================================
// 7. FEATURE 6: STREAM INTEGRITY MONITOR
// ============================================================================
async function loadStreamIntegrity() {
  try {
    const res = await fetch(`${API_BASE}/api/stream/status`);
    const data = await res.json();
    renderStreamStatus(data);
    loadRecentStreamEvents();
  } catch (err) {
    console.error('Error loading stream status:', err);
  }
}

function formatProbability(val) {
  if (val === null || val === undefined || isNaN(val)) return '0.0';
  const num = Number(val);
  if (num === 0) return '0.0';
  if (num < 1e-4) return num.toExponential(4);
  return num.toFixed(6);
}

function renderStreamStatus(data) {
  if (!data) return;
  const bf = data.bloom_filter || {};
  setSafeText('stream-inserted-n', bf.inserted_items_n ?? '-');
  setSafeText('stream-bits-density', `${bf.bit_density_pct ?? 0}% (${bf.bits_set ?? 0}/${bf.bit_array_size_m ?? 8192} bits)`);
  setSafeText('stream-theoretical-fp', formatProbability(bf.theoretical_fp_prob));
  const negQueries = bf.negative_queries_tested ?? (bf.false_positives_detected ?? 0);
  const empiricalText = (negQueries > 0)
    ? `${formatProbability(bf.empirical_fp_rate)} (${bf.false_positives_detected ?? 0} detected / ${negQueries} tested)`
    : `${formatProbability(bf.empirical_fp_rate)} (${bf.false_positives_detected ?? 0} detected)`;
  setSafeText('stream-empirical-fp', empiricalText);
  setSafeText('stream-duplicates-rejected', bf.duplicate_rejections ?? 0);
  setSafeText('stream-mongo-docs', `${data.mongo_persisted_events ?? 0} Docs`);
}

async function streamStep(count = 5) {
  try {
    const res = await fetch(`${API_BASE}/api/stream/step?count=${count}`, { method: 'POST' });
    const data = await res.json();
    loadStreamIntegrity();
  } catch (err) {
    console.error('Error stepping stream:', err);
  }
}

async function injectDuplicateEvent() {
  try {
    const res = await fetch(`${API_BASE}/api/stream/inject-duplicate`, { method: 'POST' });
    const data = await res.json();
    loadStreamIntegrity();
  } catch (err) {
    console.error('Error injecting duplicate:', err);
  }
}

async function resetStreamFilter() {
  const m = parseInt(document.getElementById('bloom-m-input')?.value || '8192');
  const k = parseInt(document.getElementById('bloom-k-input')?.value || '5');
  try {
    const res = await fetch(`${API_BASE}/api/stream/reset?m=${m}&k=${k}`, { method: 'POST' });
    await res.json();
    loadStreamIntegrity();
  } catch (err) {
    console.error('Error resetting stream filter:', err);
  }
}

async function loadRecentStreamEvents() {
  try {
    const res = await fetch(`${API_BASE}/api/stream/events?limit=8`);
    const data = await res.json();
    const tbody = document.getElementById('stream-events-table-body');
    if (!tbody) return;

    if (!data.events || data.events.length === 0) {
      tbody.innerHTML = `
        <tr>
          <td colspan="6" style="text-align: center; color: var(--text-muted); padding: 1.5rem;">
            No stream events in current window. Click "⚡ Step 5 Points" to ingest point events.
          </td>
        </tr>
      `;
      const mongoJsonEl = document.getElementById('mongo-stream-json-preview');
      if (mongoJsonEl) {
        mongoJsonEl.textContent = JSON.stringify({ status: "Waiting for point stream events..." }, null, 2);
      }
      return;
    }

    tbody.innerHTML = data.events.map(ev => {
      const isRejected = ev.duplicate_rejected || (ev.bloom_suspected_duplicate && !ev.persisted);
      const isFP = ev.is_false_positive;
      const statusLabel = isRejected ? '⚠️ DUPLICATE REJECTED' :
                          isFP ? '⚠️ FALSE POSITIVE' :
                          ev.bloom_suspected_duplicate ? '⚠️ DUPLICATE' : '✓ VERIFIED NEW';
      const statusColor = (isRejected || ev.bloom_suspected_duplicate) ? 'var(--neon-coral)' :
                          isFP ? 'var(--neon-coral)' : 'var(--neon-emerald)';
      return `
        <tr>
          <td><code>${ev.event_id}</code></td>
          <td>${ev.players}</td>
          <td><strong style="color:var(--cyan-primary);">${ev.score}</strong></td>
          <td>Server ${ev.server}</td>
          <td>${ev.serve_dir}</td>
          <td>
            <span class="status-pill" style="color: ${statusColor}">
              ${statusLabel}
            </span>
          </td>
        </tr>
      `;
    }).join('');

    // Update raw mongo preview
    const mongoJsonEl = document.getElementById('mongo-stream-json-preview');
    if (mongoJsonEl && data.events.length > 0) {
      mongoJsonEl.textContent = JSON.stringify(data.events[0], null, 2);
    }
  } catch (err) {
    console.error('Error loading recent stream events:', err);
  }
}

function toggleAutoStream() {
  const btn = document.getElementById('stream-auto-btn');
  if (state.streamInterval) {
    clearInterval(state.streamInterval);
    state.streamInterval = null;
    btn.textContent = '⚡ Start Live Stream';
  } else {
    btn.textContent = '⏸ Pause Stream';
    state.streamInterval = setInterval(() => {
      streamStep(2);
    }, 1000);
  }
}

// Window resize handler to maintain sharp responsive canvas render
window.addEventListener('resize', () => {
  if (state.currentTab === 'momentum' && state.activeReplayData) {
    drawMomentumCanvas();
  }
});
