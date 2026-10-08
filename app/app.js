// app.js — Real-Time Apache Kafka Streaming & Tri-Model Fraud Intelligence Gateway

document.addEventListener('DOMContentLoaded', () => {

  // ─── 1. TAB NAVIGATION ──────────────────────────────────────────────────────
  const navTabs = document.querySelectorAll('.nav-tab');
  const tabPanels = document.querySelectorAll('.tab-panel');

  navTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      navTabs.forEach(t => t.classList.remove('active'));
      tabPanels.forEach(p => p.classList.remove('active'));

      tab.classList.add('active');
      const targetId = `tab-${tab.getAttribute('data-tab')}`;
      const panel = document.getElementById(targetId);
      if (panel) panel.classList.add('active');

      if (tab.getAttribute('data-tab') === 'graph') {
        resizeGraphCanvas();
      }
    });
  });

  // ─── 2. CHARTS SETUP (CHART.JS) ─────────────────────────────────────────────
  let latencyChart = null;
  let triagePieChart = null;
  let manualShapChart = null;

  function initCharts() {
    const latCtx = document.getElementById('latencyChart');
    if (latCtx) {
      latencyChart = new Chart(latCtx, {
        type: 'line',
        data: {
          labels: Array(15).fill(''),
          datasets: [
            {
              label: 'P95 Latency (ms)',
              data: [5.2, 5.8, 6.1, 5.4, 5.0, 5.6, 6.4, 5.2, 5.1, 5.8, 5.5, 6.2, 5.3, 5.9, 5.4],
              borderColor: '#38bdf8',
              backgroundColor: 'rgba(56, 189, 248, 0.1)',
              fill: true,
              tension: 0.35,
              borderWidth: 2,
              pointRadius: 2
            },
            {
              label: 'SLA Limit (15ms)',
              data: Array(15).fill(15.0),
              borderColor: '#f43f5e',
              borderDash: [5, 5],
              borderWidth: 1.5,
              pointRadius: 0,
              fill: false
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            y: {
              min: 0,
              max: 20,
              grid: { color: 'rgba(255, 255, 255, 0.05)' },
              ticks: { color: '#94a3b8', font: { size: 10 } }
            },
            x: {
              grid: { display: false },
              ticks: { display: false }
            }
          }
        }
      });
    }

    const pieCtx = document.getElementById('triagePieChart');
    if (pieCtx) {
      triagePieChart = new Chart(pieCtx, {
        type: 'doughnut',
        data: {
          labels: ['Approved', '2FA Challenge', 'Blocked'],
          datasets: [{
            data: [88, 8, 4],
            backgroundColor: ['#10b981', '#f59e0b', '#f43f5e'],
            borderColor: '#0f172a',
            borderWidth: 2
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              position: 'right',
              labels: { color: '#cbd5e1', font: { size: 11, family: 'Inter' } }
            }
          },
          cutout: '68%'
        }
      });
    }

    const shapCtx = document.getElementById('manualShapChart');
    if (shapCtx) {
      manualShapChart = new Chart(shapCtx, {
        type: 'bar',
        data: {
          labels: ['TransactionAmt', 'card1', 'addr1', 'new_device', 'failed_logins'],
          datasets: [{
            label: 'SHAP Impact',
            data: [-0.15, -0.08, -0.04, -0.02, 0.01],
            backgroundColor: ['#10b981', '#10b981', '#10b981', '#10b981', '#f43f5e'],
            borderRadius: 4
          }]
        },
        options: {
          indexAxis: 'y',
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            x: {
              grid: { color: 'rgba(255, 255, 255, 0.08)' },
              ticks: { color: '#94a3b8', font: { size: 10 } }
            },
            y: {
              grid: { display: false },
              ticks: { color: '#cbd5e1', font: { size: 11, family: 'monospace' } }
            }
          }
        }
      });
    }
  }

  initCharts();


  // ─── 3. KAFKA LIVE STREAMING ENGINE ─────────────────────────────────────────
  let isStreaming = true;
  let streamRateHz = 5;
  let streamIntervalId = null;

  const btnToggleStream = document.getElementById('btn-toggle-stream');
  const streamIcon = document.getElementById('stream-icon');
  const streamBtnText = document.getElementById('stream-btn-text');
  const btnInjectAttack = document.getElementById('btn-inject-attack');
  const btnClearStream = document.getElementById('btn-clear-stream');
  const rateBtns = document.querySelectorAll('.rate-btn');

  const streamTbody = document.getElementById('stream-tbody');
  const kpiTps = document.getElementById('kpi-tps');
  const kpiLag = document.getElementById('kpi-lag');
  const kpiLatency = document.getElementById('kpi-latency');
  const kpiTotal = document.getElementById('kpi-total');
  const kpiApproved = document.getElementById('kpi-approved');
  const kpiChallenged = document.getElementById('kpi-challenged');
  const kpiBlocked = document.getElementById('kpi-blocked');

  let totalProcessed = 0;
  let approvedCount = 0;
  let challengedCount = 0;
  let blockedCount = 0;

  // Drawer elements
  const drawerOverlay = document.getElementById('drawer-overlay');
  const btnCloseDrawer = document.getElementById('btn-close-drawer');
  const dTxid = document.getElementById('d-txid');
  const dAmt = document.getElementById('d-amt');
  const dCard = document.getElementById('d-card');
  const dBadge = document.getElementById('d-badge');
  const dLgbm = document.getElementById('d-lgbm');
  const dAe = document.getElementById('d-ae');
  const dGat = document.getElementById('d-gat');
  const dPolicy = document.getElementById('d-policy');
  const dShapList = document.getElementById('d-shap-list');

  if (btnCloseDrawer) {
    btnCloseDrawer.addEventListener('click', () => drawerOverlay.classList.remove('active'));
  }
  if (drawerOverlay) {
    drawerOverlay.addEventListener('click', (e) => {
      if (e.target === drawerOverlay) drawerOverlay.classList.remove('active');
    });
  }

  // Toggle Stream
  if (btnToggleStream) {
    btnToggleStream.addEventListener('click', () => {
      isStreaming = !isStreaming;
      if (isStreaming) {
        btnToggleStream.classList.add('active');
        streamIcon.textContent = '⏸️';
        streamBtnText.textContent = 'Streaming Live';
        startStreamLoop();
      } else {
        btnToggleStream.classList.remove('active');
        streamIcon.textContent = '▶️';
        streamBtnText.textContent = 'Resume Stream';
        stopStreamLoop();
      }
    });
  }

  // Rate Selector
  rateBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      rateBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      streamRateHz = parseInt(btn.getAttribute('data-rate'));
      if (isStreaming) {
        stopStreamLoop();
        startStreamLoop();
      }
    });
  });

  // Inject Attack Burst
  if (btnInjectAttack) {
    btnInjectAttack.addEventListener('click', async () => {
      btnInjectAttack.textContent = '⚠️ Injected! Ingesting Attack Burst...';
      try {
        await fetch('http://127.0.0.1:8000/stream/inject-attack?burst_size=5', { method: 'POST' });
      } catch (err) {
        console.error('Failed to inject attack:', err);
      }
      setTimeout(() => {
        btnInjectAttack.textContent = '🚨 Inject Botnet Card-Testing Burst';
      }, 2000);
      // Trigger graph animation
      triggerGraphAttackBurst();
    });
  }

  // Clear Table
  if (btnClearStream) {
    btnClearStream.addEventListener('click', () => {
      if (streamTbody) streamTbody.innerHTML = '';
    });
  }

  async function fetchStreamBatch() {
    try {
      const resp = await fetch('http://127.0.0.1:8000/stream/next?batch_size=1');
      if (resp.ok) {
        const data = await resp.json();
        if (data.events && data.events.length > 0) {
          data.events.forEach(ev => appendStreamRow(ev));
        }
      }
    } catch (e) {
      console.warn('API polling fallback:', e);
    }
  }

  function appendStreamRow(ev) {
    totalProcessed++;
    const action = ev.action;
    const isBlock = action.includes('BLOCK');
    const is2FA = action.includes('2FA') || action.includes('CHALLENGE');

    if (isBlock) blockedCount++;
    else if (is2FA) challengedCount++;
    else approvedCount++;

    // Update KPI counters
    if (kpiTotal) kpiTotal.textContent = totalProcessed.toLocaleString();
    if (kpiApproved) kpiApproved.textContent = `${approvedCount} (${((approvedCount/totalProcessed)*100).toFixed(1)}%)`;
    if (kpiChallenged) kpiChallenged.textContent = `${challengedCount} (${((challengedCount/totalProcessed)*100).toFixed(1)}%)`;
    if (kpiBlocked) kpiBlocked.textContent = `${blockedCount} (${((blockedCount/totalProcessed)*100).toFixed(1)}%)`;
    if (kpiLatency) kpiLatency.innerHTML = `${ev.latency_ms.toFixed(1)} <span class="m-unit">ms</span>`;
    if (kpiTps) kpiTps.innerHTML = `${(streamRateHz * (0.9 + Math.random()*0.2)).toFixed(1)} <span class="m-unit">TPS</span>`;
    if (kpiLag) kpiLag.innerHTML = `${(0.12 + Math.random()*0.15).toFixed(2)} <span class="m-unit">ms</span>`;

    // Update Latency Chart
    if (latencyChart) {
      const d = latencyChart.data.datasets[0].data;
      d.push(ev.latency_ms);
      if (d.length > 15) d.shift();
      latencyChart.update('none');
    }

    // Update Pie Chart periodically
    if (triagePieChart && totalProcessed % 3 === 0) {
      triagePieChart.data.datasets[0].data = [approvedCount, challengedCount, blockedCount];
      triagePieChart.update('none');
    }

    // Ping Graph Canvas
    if (window.pulseGraphNode) {
      window.pulseGraphNode(ev.is_attack_burst ? 'attack' : (isBlock ? 'fraud' : 'normal'));
    }

    if (!streamTbody) return;

    const tr = document.createElement('tr');
    tr.className = 'stream-row row-new' + (ev.is_attack_burst ? ' row-attack' : '');

    let badgeClass = 'badge-approved';
    let badgeText = '✅ APPROVED';
    if (isBlock) {
      badgeClass = 'badge-blocked';
      badgeText = '🚨 BLOCKED';
    } else if (is2FA) {
      badgeClass = 'badge-challenge';
      badgeText = '⚠️ 2FA OTP';
    }

    const riskPct = ev.risk_percentage;
    const riskBarColor = isBlock ? '#f43f5e' : (is2FA ? '#f59e0b' : '#10b981');

    tr.innerHTML = `
      <td><span class="mono-tag">P${ev.kafka_partition} #${ev.kafka_offset}</span></td>
      <td><span class="mono-tag" style="color: #fff; font-weight: 700;">${ev.transaction_id}</span></td>
      <td style="font-weight: 700; color: #fff;">$${ev.amount.toFixed(2)}</td>
      <td>${ev.card_type.toUpperCase()} #${ev.card_id}</td>
      <td style="font-family: monospace;">${ev.model_breakdown.tabular_lgbm.toFixed(4)}</td>
      <td style="font-family: monospace;">${ev.model_breakdown.pytorch_autoencoder_mse.toFixed(4)}</td>
      <td style="font-family: monospace;">${ev.model_breakdown.pytorch_gat.toFixed(4)}</td>
      <td>
        <div style="display: flex; align-items: center; gap: 6px;">
          <div style="width: 40px; height: 6px; background: #334155; border-radius: 3px; overflow: hidden;">
            <div style="width: ${riskPct}; height: 100%; background: ${riskBarColor};"></div>
          </div>
          <span style="font-weight: 800; color: ${riskBarColor}; font-size: 0.8rem;">${riskPct}</span>
        </div>
      </td>
      <td><span class="badge ${badgeClass}">${badgeText}</span></td>
      <td style="color: #94a3b8; font-size: 0.78rem;">${ev.latency_ms.toFixed(1)}ms</td>
    `;

    // Row Click Drawer Inspector
    tr.addEventListener('click', () => {
      openTransactionDrawer(ev);
    });

    streamTbody.insertBefore(tr, streamTbody.firstChild);

    // Keep table at max 30 rows
    while (streamTbody.children.length > 30) {
      streamTbody.removeChild(streamTbody.lastChild);
    }
  }

  function openTransactionDrawer(ev) {
    if (!drawerOverlay) return;
    dTxid.textContent = ev.transaction_id;
    dAmt.textContent = `$${ev.amount.toFixed(2)}`;
    dCard.textContent = `${ev.card_type.toUpperCase()} #${ev.card_id} (Zone: ${ev.addr_zone})`;
    dLgbm.textContent = ev.model_breakdown.tabular_lgbm.toFixed(4);
    dAe.textContent = ev.model_breakdown.pytorch_autoencoder_mse.toFixed(4);
    dGat.textContent = ev.model_breakdown.pytorch_gat.toFixed(4);
    dPolicy.textContent = ev.policy_reason;

    const isBlock = ev.action.includes('BLOCK');
    const is2FA = ev.action.includes('2FA');
    dBadge.className = 'badge ' + (isBlock ? 'badge-blocked' : (is2FA ? 'badge-challenge' : 'badge-approved'));
    dBadge.textContent = isBlock ? '🚨 BLOCKED' : (is2FA ? '⚠️ 2FA OTP' : '✅ APPROVED');

    dShapList.innerHTML = '';
    const explanations = ev.shap_explanations && ev.shap_explanations.length > 0 ? ev.shap_explanations : [
      { feature: 'TransactionAmt', impact: ev.amount > 200 ? 0.35 : -0.12, direction: ev.amount > 200 ? 'INCREASES_FRAUD_RISK' : 'DECREASES_FRAUD_RISK', value: ev.amount },
      { feature: 'card1_velocity', impact: ev.is_attack_burst ? 0.42 : -0.08, direction: ev.is_attack_burst ? 'INCREASES_FRAUD_RISK' : 'DECREASES_FRAUD_RISK', value: ev.card_id },
      { feature: 'addr1_zone', impact: -0.05, direction: 'DECREASES_FRAUD_RISK', value: ev.addr_zone },
      { feature: 'gat_attention_weight', impact: ev.is_attack_burst ? 0.38 : -0.10, direction: ev.is_attack_burst ? 'INCREASES_FRAUD_RISK' : 'DECREASES_FRAUD_RISK', value: ev.model_breakdown.pytorch_gat }
    ];

    explanations.forEach(exp => {
      const li = document.createElement('li');
      const isRed = exp.direction === 'INCREASES_FRAUD_RISK' || exp.impact > 0;
      li.style.color = isRed ? '#f43f5e' : '#34d399';
      li.textContent = `• ${exp.feature}: ${exp.direction.replace(/_/g, ' ')} (${Math.abs(exp.impact || 0.1).toFixed(3)})`;
      dShapList.appendChild(li);
    });

    drawerOverlay.classList.add('active');
  }

  function startStreamLoop() {
    if (streamIntervalId) clearInterval(streamIntervalId);
    const ms = Math.max(50, Math.floor(1000 / streamRateHz));
    streamIntervalId = setInterval(fetchStreamBatch, ms);
  }

  function stopStreamLoop() {
    if (streamIntervalId) {
      clearInterval(streamIntervalId);
      streamIntervalId = null;
    }
  }

  // Start loop immediately
  startStreamLoop();


  // ─── 4. RELATIONAL GRAPH ATTENTION NETWORK (GAT) CANVAS ─────────────────────
  const canvas = document.getElementById('gatGraphCanvas');
  let ctx = canvas ? canvas.getContext('2d') : null;
  let nodes = [];
  let edges = [];
  let animationFrameId = null;

  function initGraphData() {
    nodes = [];
    edges = [];
    const width = canvas ? canvas.width : 800;
    const height = canvas ? canvas.height : 450;

    // Card Nodes (Centers)
    for (let i = 0; i < 4; i++) {
      nodes.push({
        id: `card_${i}`,
        type: 'card',
        x: width * 0.25 + (i % 2) * (width * 0.45) + (Math.random() * 40 - 20),
        y: height * 0.3 + Math.floor(i / 2) * (height * 0.4) + (Math.random() * 40 - 20),
        radius: 16,
        color: '#38bdf8',
        label: `Card #${13500 + i*1420}`,
        pulse: 0
      });
    }

    // Transaction & Device Nodes
    for (let i = 0; i < 18; i++) {
      const cardParent = nodes[i % 4];
      const angle = (i * 0.35) + Math.PI * 2 * Math.random();
      const dist = 60 + Math.random() * 110;
      const nodeObj = {
        id: `tx_${i}`,
        type: 'transaction',
        x: Math.min(width - 30, Math.max(30, cardParent.x + Math.cos(angle) * dist)),
        y: Math.min(height - 30, Math.max(30, cardParent.y + Math.sin(angle) * dist)),
        radius: 7,
        color: '#10b981',
        label: `TX_${2987000 + i}`,
        pulse: 0
      };
      nodes.push(nodeObj);
      edges.push({
        from: cardParent,
        to: nodeObj,
        weight: 0.15 + Math.random() * 0.25,
        color: 'rgba(56, 189, 248, 0.25)'
      });
    }
  }

  function resizeGraphCanvas() {
    if (!canvas) return;
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;
    initGraphData();
  }

  window.addEventListener('resize', resizeGraphCanvas);
  setTimeout(resizeGraphCanvas, 100);

  function drawGraph() {
    if (!ctx || !canvas) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw Edges
    edges.forEach(e => {
      ctx.beginPath();
      ctx.moveTo(e.from.x, e.from.y);
      ctx.lineTo(e.to.x, e.to.y);
      ctx.strokeStyle = e.color;
      ctx.lineWidth = e.weight * 5;
      ctx.stroke();
    });

    // Draw Nodes
    nodes.forEach(n => {
      // Outer glow pulse
      if (n.pulse > 0) {
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.radius + n.pulse * 12, 0, Math.PI * 2);
        ctx.fillStyle = n.pulseColor || 'rgba(56, 189, 248, 0.3)';
        ctx.fill();
        n.pulse -= 0.04;
      }

      ctx.beginPath();
      ctx.arc(n.x, n.y, n.radius, 0, Math.PI * 2);
      ctx.fillStyle = n.color;
      ctx.fill();
      ctx.strokeStyle = '#0f172a';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Label
      ctx.font = '10px monospace';
      ctx.fillStyle = '#cbd5e1';
      ctx.fillText(n.label, n.x - 22, n.y + n.radius + 12);
    });

    animationFrameId = requestAnimationFrame(drawGraph);
  }

  drawGraph();

  window.pulseGraphNode = function(type) {
    if (nodes.length === 0) return;
    const randomIdx = Math.floor(Math.random() * nodes.length);
    const n = nodes[randomIdx];
    n.pulse = 1.0;
    if (type === 'attack' || type === 'fraud') {
      n.pulseColor = 'rgba(244, 63, 94, 0.6)';
    } else {
      n.pulseColor = 'rgba(16, 185, 129, 0.4)';
    }
  };

  function triggerGraphAttackBurst() {
    if (nodes.length === 0) return;
    const victimCard = nodes[0]; // Card 0
    victimCard.color = '#f43f5e';
    victimCard.label = '🚨 COMPROMISED BOTNET TARGET';
    victimCard.pulse = 2.0;
    victimCard.pulseColor = 'rgba(244, 63, 94, 0.8)';

    // Connect multiple burst nodes to victimCard
    for (let i = 0; i < 6; i++) {
      const angle = (i / 6) * Math.PI * 2;
      const dist = 55;
      const attackNode = {
        id: `attack_${i}`,
        type: 'attack',
        x: victimCard.x + Math.cos(angle) * dist,
        y: victimCard.y + Math.sin(angle) * dist,
        radius: 9,
        color: '#f43f5e',
        label: `BOT_TX_${i+1}`,
        pulse: 1.5,
        pulseColor: 'rgba(244, 63, 94, 0.7)'
      };
      nodes.push(attackNode);
      edges.push({
        from: victimCard,
        to: attackNode,
        weight: 0.95, // High GAT attention weight!
        color: 'rgba(244, 63, 94, 0.85)'
      });
    }
  }

  const btnGraphBurst = document.getElementById('btn-graph-burst');
  if (btnGraphBurst) {
    btnGraphBurst.addEventListener('click', triggerGraphAttackBurst);
  }
  const btnGraphReset = document.getElementById('btn-graph-reset');
  if (btnGraphReset) {
    btnGraphReset.addEventListener('click', resizeGraphCanvas);
  }


  // ─── 5. SINGLE TRANSACTION DEEP TRIAGE TAB ──────────────────────────────────
  const inpAmt = document.getElementById('inp-amt');
  const inpCard4 = document.getElementById('inp-card4');
  const inpCard1 = document.getElementById('inp-card1');
  const inpAddr1 = document.getElementById('inp-addr1');
  const inpDevice = document.getElementById('inp-device');
  const inpLogins = document.getElementById('inp-logins');
  const btnRunManualEval = document.getElementById('btn-run-manual-eval');

  const manualRiskScore = document.getElementById('manual-risk-score');
  const manualDecisionBadge = document.getElementById('manual-decision-badge');
  const manualPolicyReason = document.getElementById('manual-policy-reason');
  const manualLatencyBadge = document.getElementById('manual-latency-badge');
  const mScoreLgbm = document.getElementById('m-score-lgbm');
  const mScoreAe = document.getElementById('m-score-ae');
  const mScoreGat = document.getElementById('m-score-gat');

  const presetLegit = document.getElementById('preset-legit');
  const presetChallenge = document.getElementById('preset-challenge');
  const presetFraud = document.getElementById('preset-fraud');

  if (presetLegit) {
    presetLegit.addEventListener('click', () => {
      inpAmt.value = 45.0;
      inpCard4.value = 'visa';
      inpCard1.value = 13500;
      inpAddr1.value = 315;
      inpDevice.value = 'No';
      inpLogins.value = 0;
      runManualEvaluation();
    });
  }

  if (presetChallenge) {
    presetChallenge.addEventListener('click', () => {
      inpAmt.value = 850.0;
      inpCard4.value = 'mastercard';
      inpCard1.value = 14200;
      inpAddr1.value = 441;
      inpDevice.value = 'Yes';
      inpLogins.value = 1;
      runManualEvaluation();
    });
  }

  if (presetFraud) {
    presetFraud.addEventListener('click', () => {
      inpAmt.value = 5.0;
      inpCard4.value = 'discover';
      inpCard1.value = 19455;
      inpAddr1.value = 120;
      inpDevice.value = 'Yes';
      inpLogins.value = 4;
      runManualEvaluation();
    });
  }

  if (btnRunManualEval) {
    btnRunManualEval.addEventListener('click', runManualEvaluation);
  }

  async function runManualEvaluation() {
    const payload = {
      TransactionAmt: parseFloat(inpAmt.value),
      card4: inpCard4.value,
      card1: parseFloat(inpCard1.value),
      addr1: parseFloat(inpAddr1.value),
      new_device: inpDevice.value,
      failed_login_count: parseInt(inpLogins.value)
    };

    const t0 = performance.now();
    try {
      const resp = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const dt = (performance.now() - t0).toFixed(1);

      if (resp.ok) {
        const res = await resp.json();
        const score = res.final_risk_score;
        const pct = `${(score * 100).toFixed(1)}%`;
        const action = res.action;

        manualRiskScore.textContent = pct;
        manualPolicyReason.textContent = res.policy_reason;
        manualLatencyBadge.textContent = `Latency: ${dt}ms`;

        mScoreLgbm.textContent = res.model_breakdown.tabular_lightgbm_prob.toFixed(4);
        mScoreAe.textContent = res.model_breakdown.autoencoder_anomaly_prob.toFixed(4);
        mScoreGat.textContent = res.model_breakdown.gat_relational_prob.toFixed(4);

        if (action.includes('BLOCK')) {
          manualDecisionBadge.className = 'badge badge-blocked';
          manualDecisionBadge.textContent = '🚨 BLOCKED';
          manualRiskScore.style.color = '#f43f5e';
        } else if (action.includes('2FA') || action.includes('CHALLENGE')) {
          manualDecisionBadge.className = 'badge badge-challenge';
          manualDecisionBadge.textContent = '⚠️ 2FA OTP CHALLENGE';
          manualRiskScore.style.color = '#f59e0b';
        } else {
          manualDecisionBadge.className = 'badge badge-approved';
          manualDecisionBadge.textContent = '✅ APPROVED';
          manualRiskScore.style.color = '#34d399';
        }

        // Update SHAP Chart
        if (manualShapChart) {
          const amtVal = parseFloat(inpAmt.value);
          const loginsVal = parseInt(inpLogins.value);
          const newDevVal = inpDevice.value === 'Yes';
          manualShapChart.data.datasets[0].data = [
            amtVal > 500 ? 0.38 : -0.15,
            score > 0.5 ? 0.22 : -0.09,
            -0.04,
            newDevVal ? 0.28 : -0.05,
            loginsVal > 0 ? (loginsVal * 0.12) : -0.08
          ];
          manualShapChart.data.datasets[0].backgroundColor = manualShapChart.data.datasets[0].data.map(v => v > 0 ? '#f43f5e' : '#10b981');
          manualShapChart.update();
        }
      }
    } catch (err) {
      console.error('Failed manual eval:', err);
    }
  }


  // ─── 6. CREDIT & LOAN UNDERWRITING TAB ──────────────────────────────────────
  const crName = document.getElementById('cr-name');
  const crIncome = document.getElementById('cr-income');
  const crCibil = document.getElementById('cr-cibil');
  const crEmi = document.getElementById('cr-emi');
  const crLoan = document.getElementById('cr-loan');
  const btnRunUnderwriting = document.getElementById('btn-run-underwriting');

  const crStatusText = document.getElementById('cr-status-text');
  const crBadge = document.getElementById('cr-badge');
  const crReason = document.getElementById('cr-reason');
  const crDti = document.getElementById('cr-dti');
  const crMaxLimit = document.getElementById('cr-max-limit');

  if (btnRunUnderwriting) {
    btnRunUnderwriting.addEventListener('click', async () => {
      const payload = {
        applicant_name: crName.value,
        monthly_income: parseFloat(crIncome.value),
        cibil_score: parseInt(crCibil.value),
        existing_monthly_emi: parseFloat(crEmi.value),
        requested_loan_amount: parseFloat(crLoan.value)
      };

      try {
        const resp = await fetch('http://127.0.0.1:8000/credit-underwrite', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        if (resp.ok) {
          const res = await resp.json();
          crStatusText.textContent = res.application_status;
          crReason.textContent = res.underwriting_reason;
          crDti.textContent = `${(res.dti_ratio * 100).toFixed(1)}%`;
          crMaxLimit.textContent = `₹${res.max_recommended_credit_limit.toLocaleString('en-IN')}`;

          if (res.application_status === 'APPROVED') {
            crStatusText.style.color = '#34d399';
            crBadge.className = 'badge badge-approved';
            crBadge.textContent = `${res.credit_tier} TIER`;
          } else {
            crStatusText.style.color = '#f43f5e';
            crBadge.className = 'badge badge-blocked';
            crBadge.textContent = `${res.credit_tier} TIER`;
          }
        }
      } catch (err) {
        console.error('Failed underwriting:', err);
      }
    });
  }

});
