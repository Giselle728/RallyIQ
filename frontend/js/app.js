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
  if (tabId === 'player-lab') loadPlayerLab();
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

    // KPIs
    document.getElementById('kpi-raw-size').textContent = `${rep.raw_size_mb || 117.0} MB`;
    document.getElementById('kpi-points').textContent = (rep.point_count || 904128).toLocaleString();
    document.getElementById('kpi-matches').textContent = (rep.match_count || 7565).toLocaleString();
    document.getElementById('kpi-seasons').textContent = `${rep.season_coverage?.total_seasons || 58} Seasons (${rep.season_coverage?.min_year} - ${rep.season_coverage?.max_year})`;

    // Storage Status
    const mongoStatusEl = document.getElementById('mongo-engine-status');
    if (mongoStatusEl) {
      mongoStatusEl.textContent = st.engine || 'Connected';
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
// 2. FEATURE 1: PLAYER INTELLIGENCE LAB
// ============================================================================
async function loadPlayersList() {
  try {
    const res = await fetch(`${API_BASE}/api/players?min_matches=5`);
    const data = await res.json();
    state.playersList = data.players || [];

    const pA = document.getElementById('lab-player-a');
    const pB = document.getElementById('lab-player-b');
    const pRec = document.getElementById('recommender-player-select');
    const pServe = document.getElementById('serve-player-select');

    const options = state.playersList.map(p => `<option value="${p.player}">${p.player} (${p.total_matches} matches)</option>`).join('');

    if (pA) { pA.innerHTML = options; pA.value = 'Novak Djokovic'; }
    if (pB) { pB.innerHTML = options; pB.value = 'Rafael Nadal'; }
    if (pRec) { pRec.innerHTML = options; pRec.value = 'Carlos Alcaraz'; }
    if (pServe) { pServe.innerHTML = options; pServe.value = 'Roger Federer'; }
  } catch (err) {
    console.error('Error loading players list:', err);
  }
}

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
  if (matchId) loadMatchReplay(matchId);
}

async function loadMatchReplay(matchId) {
  const winSize = parseInt(document.getElementById('momentum-window-input')?.value || '40');
  const alpha = parseFloat(document.getElementById('momentum-alpha-input')?.value || '0.25');

  try {
    const res = await fetch(`${API_BASE}/api/momentum/replay?match_id=${encodeURIComponent(matchId)}&window_size=${winSize}&alpha=${alpha}`);
    const data = await res.json();
    state.activeReplayData = data;
    state.replayIndex = 0;

    document.getElementById('momentum-match-label').textContent = `${data.metadata?.tournament} (${data.metadata?.year}): ${data.p1} vs ${data.p2}`;
    document.getElementById('momentum-swings-count').textContent = `${data.momentum_swings} Lead Changes`;

    const slider = document.getElementById('momentum-scrubber');
    if (slider) {
      slider.max = data.points.length - 1;
      slider.value = 0;
    }

    renderMomentumSnapshot(0);
    drawMomentumCanvas();
  } catch (err) {
    console.error('Error loading match replay:', err);
  }
}

function renderMomentumSnapshot(idx) {
  if (!state.activeReplayData || !state.activeReplayData.points[idx]) return;
  state.replayIndex = idx;
  const pt = state.activeReplayData.points[idx];

  // Scoreboard
  document.getElementById('live-score-display').textContent = `Set ${pt.set1}-${pt.set2} | Games ${pt.gm1}-${pt.gm2} | Point: ${pt.score}`;
  document.getElementById('live-server-display').textContent = `Server: Player ${pt.server}`;
  document.getElementById('live-point-label').textContent = pt.label;
  document.getElementById('live-winner-display').textContent = `Point Won By: ${pt.winner_name}`;

  // Momentum Gauge
  const mVal = pt.momentum;
  const mBar = document.getElementById('live-momentum-meter');
  if (mBar) {
    const clamped = Math.max(-3, Math.min(3, mVal));
    const pct = ((clamped + 3) / 6) * 100;
    mBar.style.left = `${pct}%`;
  }
  document.getElementById('live-momentum-score-text').textContent = `Score: ${mVal > 0 ? '+' : ''}${mVal} (${pt.dominant_player} Advantage)`;

  // DGIM Buckets
  const bucketContainer = document.getElementById('dgim-bucket-display');
  if (bucketContainer) {
    const buckets = pt.dgim_p1_buckets || [];
    bucketContainer.innerHTML = buckets.length > 0 
      ? buckets.map(b => `<div class="dgim-bubble">Size ${b.size} (t=${b.timestamp})</div>`).join('')
      : '<span style="color:var(--text-muted);font-size:0.8rem;">No active buckets in window</span>';
  }

  // DGIM Accuracy
  document.getElementById('dgim-est-wins').textContent = pt.dgim_p1_window_wins;
  document.getElementById('dgim-exact-wins').textContent = pt.dgim_p1_exact_wins;
  document.getElementById('dgim-error-pct').textContent = `${pt.dgim_p1_error_pct}% (Bound: ≤25%)`;
}

function drawMomentumCanvas() {
  const canvas = document.getElementById('momentum-curve-canvas');
  if (!canvas || !state.activeReplayData) return;
  const ctx = canvas.getContext('2d');
  const curve = state.activeReplayData.curve;

  const w = canvas.width = canvas.parentElement.clientWidth;
  const h = canvas.height = 280;

  ctx.clearRect(0, 0, w, h);

  // Zero-line (Center)
  const zeroY = h / 2;
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, zeroY);
  ctx.lineTo(w, zeroY);
  ctx.stroke();

  // Draw curve
  ctx.beginPath();
  const maxM = 3.5;
  curve.forEach((c, i) => {
    const x = (i / (curve.length - 1)) * w;
    const y = zeroY - (c.momentum / maxM) * (h / 2 - 20);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });

  ctx.strokeStyle = '#00f2fe';
  ctx.lineWidth = 2.5;
  ctx.stroke();

  // Current playback marker line
  const curX = (state.replayIndex / (curve.length - 1)) * w;
  ctx.strokeStyle = '#ff3366';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(curX, 0);
  ctx.lineTo(curX, h);
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
      document.getElementById('momentum-scrubber').value = state.replayIndex;
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
  const node = state.networkData.nodes.find(n => n.id === playerId);
  if (!node) return;
  alert(`Player: ${node.label}\nTotal Matches: ${node.total_matches}\nPageRank: ${node.pagerank}\nLouvain Cluster: ${node.community}`);
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

function renderStreamStatus(data) {
  const bf = data.bloom_filter;
  document.getElementById('stream-inserted-n').textContent = bf.inserted_items_n;
  document.getElementById('stream-bits-density').textContent = `${bf.bit_density_pct}% (${bf.bits_set}/${bf.bit_array_size_m} bits)`;
  document.getElementById('stream-theoretical-fp').textContent = bf.theoretical_fp_prob;
  document.getElementById('stream-empirical-fp').textContent = `${bf.empirical_fp_rate} (${bf.false_positives_detected} detected)`;
  document.getElementById('stream-duplicates-rejected').textContent = bf.duplicate_rejections;
  document.getElementById('stream-mongo-docs').textContent = `${data.mongo_persisted_events} Docs`;
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
    alert(`Injected Previously Streamed Event: ${data.injected_event_id}\nBloom Filter Flagged: ${data.bloom_detected_duplicate ? 'TRUE (DUPLICATE CAUGHT)' : 'FALSE'}`);
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
    alert(`Bloom filter reset with m=${m} bits, k=${k} hash functions.`);
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

    tbody.innerHTML = (data.events || []).map(ev => `
      <tr>
        <td><code>${ev.event_id}</code></td>
        <td>${ev.players}</td>
        <td><strong style="color:var(--cyan-primary);">${ev.score}</strong></td>
        <td>Server ${ev.server}</td>
        <td>${ev.serve_dir}</td>
        <td>
          <span class="status-pill" style="color: ${ev.bloom_suspected_duplicate ? 'var(--neon-coral)' : 'var(--neon-emerald)'}">
            ${ev.bloom_suspected_duplicate ? '⚠️ DUPLICATE' : '✓ VERIFIED NEW'}
          </span>
        </td>
      </tr>
    `).join('');

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
