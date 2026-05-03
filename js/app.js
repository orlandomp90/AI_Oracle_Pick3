document.addEventListener('DOMContentLoaded', async () => {
    const ui = {
        btnSyncCloud: document.getElementById('btn-sync-cloud'),
        btnManualAdd: document.getElementById('btn-manual-add'),
        manualDate: document.getElementById('manual-date'),
        manualShift: document.getElementById('manual-shift'),
        manualN1: document.getElementById('manual-n1'),
        manualN2: document.getElementById('manual-n2'),
        manualN3: document.getElementById('manual-n3'),
        dbCount: document.getElementById('db-count'),
        modelStatus: document.getElementById('model-status'),
        regimeStatus: document.getElementById('regime-status'),
        btnSeed: document.getElementById('btn-seed'),
        csvUpload: document.getElementById('csv-upload'),
        pdfUpload: document.getElementById('pdf-upload'),
        btnClear: document.getElementById('btn-clear'),
        historyBody: document.getElementById('history-body'),
        btnTrain: document.getElementById('btn-train'),
        btnStopTrain: document.getElementById('btn-stop-train'),
        btnPredict: document.getElementById('btn-predict'),
        trainProgress: document.getElementById('train-progress'),
        epochText: document.getElementById('epoch-text'),
        lossText: document.getElementById('loss-text'),
        dataProgressContainer: document.getElementById('data-progress-container'),
        dataProgressFill: document.getElementById('data-progress-fill'),
        dataProgressText: document.getElementById('data-progress-text'),
        predBallsMD: [
            [document.getElementById('pred-md-a1'), document.getElementById('pred-md-a2'), document.getElementById('pred-md-a3')],
            [document.getElementById('pred-md-b1'), document.getElementById('pred-md-b2'), document.getElementById('pred-md-b3')],
            [document.getElementById('pred-md-c1'), document.getElementById('pred-md-c2'), document.getElementById('pred-md-c3')]
        ],
        predBallsEV: [
            [document.getElementById('pred-ev-a1'), document.getElementById('pred-ev-a2'), document.getElementById('pred-ev-a3')],
            [document.getElementById('pred-ev-b1'), document.getElementById('pred-ev-b2'), document.getElementById('pred-ev-b3')],
            [document.getElementById('pred-ev-c1'), document.getElementById('pred-ev-c2'), document.getElementById('pred-ev-c3')]
        ],
        confBadgesMD: [document.getElementById('conf-md-a'), document.getElementById('conf-md-b'), document.getElementById('conf-md-c')],
        confBadgesEV: [document.getElementById('conf-ev-a'), document.getElementById('conf-ev-b'), document.getElementById('conf-ev-c')],
        trainNotification: document.getElementById('train-notification'),
        pendingCount: document.getElementById('pending-count'),
        btnTrainNow: document.getElementById('btn-train-now'),
        statCombo: document.getElementById('stat-combo'),
        statHot: document.getElementById('stat-hot'),
        statCold: document.getElementById('stat-cold'),
        backtestPanel: document.getElementById('backtest-panel'),
        btPrecision: document.getElementById('bt-precision'),
        btHits: document.getElementById('bt-hits'),
        btResultsBody: document.getElementById('bt-results-body'),
        probContainer: document.getElementById('prob-container'),
        // New v3.0
        confidenceGauge: document.getElementById('confidence-gauge'),
        confidenceValue: document.getElementById('confidence-value'),
        confidenceBar: document.getElementById('confidence-bar'),
        regimeLabel: document.getElementById('regime-label'),
        entropyValue: document.getElementById('entropy-value'),

        wfPanel: document.getElementById('walkforward-panel'),
        wfAvg: document.getElementById('wf-avg-precision'),
        wfConsistency: document.getElementById('wf-consistency'),
        wfBody: document.getElementById('wf-results-body'),
        phPanel: document.getElementById('pred-history-panel'),
        phTotal: document.getElementById('ph-total'),
        phAvgHits: document.getElementById('ph-avg-hits'),
        phFullHits: document.getElementById('ph-full-hits'),
        phBody: document.getElementById('ph-body'),
    };

    let allDraws = [];
    let lossChart = null;
    const RETRAIN_THRESHOLD = 5;
    const AUTO_SYNC_DAYS = 5;

    function getPendingDraws() { return parseInt(localStorage.getItem('oraclePendingDraws') || '0'); }
    function setPendingDraws(n) { localStorage.setItem('oraclePendingDraws', n.toString()); updateTrainNotification(); }
    function addPendingDraws(c) { setPendingDraws(getPendingDraws() + c); }
    function updateTrainNotification() {
        const c = getPendingDraws();
        if (c > 0 && ui.trainNotification) { ui.trainNotification.style.display = 'flex'; ui.pendingCount.textContent = c; }
        else if (ui.trainNotification) { ui.trainNotification.style.display = 'none'; }
    }
    function shouldAutoSync() { const l = localStorage.getItem('oracleLastSync'); if (!l) return true; return (Date.now()-parseInt(l)) >= AUTO_SYNC_DAYS*24*60*60*1000; }
    function markSynced() { localStorage.setItem('oracleLastSync', Date.now().toString()); }

    async function cleanAndSort(newDraws) {
        if (!newDraws || newDraws.length === 0) return 0;
        const current = await dbManager.getAllDraws();
        const keys = new Set(current.map(d => `${d.date}|${d.n1}${d.n2}${d.n3}`));
        const filtered = newDraws.filter(d => { const k = `${d.date}|${d.n1}${d.n2}${d.n3}`; if (keys.has(k)) return false; keys.add(k); return true; });
        if (filtered.length > 0) await dbManager.addDraws(filtered);
        return filtered.length;
    }

    // Cloud Sync
    async function syncOnline() {
        ui.btnSyncCloud.disabled = true;
        const prev = ui.btnSyncCloud.innerHTML;
        ui.btnSyncCloud.innerHTML = '<i data-lucide="loader-2" class="spin"></i> Sync...';
        if (window.lucide) lucide.createIcons();
        try {
            ui.dataProgressContainer.style.display = 'block';
            ui.dataProgressFill.style.width = '30%';
            ui.dataProgressText.textContent = 'Conectando (Midday + Evening)...';
            const resp = await fetch('/api/scrape');
            ui.dataProgressFill.style.width = '70%';
            ui.dataProgressText.textContent = 'Procesando datos multi-fuente...';
            const data = await resp.json();
            ui.dataProgressFill.style.width = '100%';
            if (data.success && data.draws) {
                const added = await cleanAndSort(data.draws);
                markSynced();
                if (added > 0) {
                    await reloadData();
                    addPendingDraws(added);
                    ui.dataProgressText.textContent = `✅ ${added} sorteos nuevos!`;
                    if (getPendingDraws() >= RETRAIN_THRESHOLD) { alert(`${added} nuevos. Entrenamiento iniciado.`); forceRetrain(); }
                    else alert(`${added} nuevos. (${getPendingDraws()}/${RETRAIN_THRESHOLD} para re-entrenar)`);
                } else { ui.dataProgressText.textContent = 'Todo al día.'; alert('No hay sorteos nuevos.'); }
            } else throw new Error('Server error');
        } catch (err) { console.error('Sync:', err); alert('Error: Ejecuta server.js primero.'); }
        setTimeout(() => { ui.dataProgressContainer.style.display = 'none'; }, 3000);
        ui.btnSyncCloud.disabled = false;
        ui.btnSyncCloud.innerHTML = prev;
        if (window.lucide) lucide.createIcons();
    }
    if (ui.btnSyncCloud) ui.btnSyncCloud.addEventListener('click', syncOnline);

    // Chart
    function initChart() {
        lossChart = new Chart(document.getElementById('lossChart').getContext('2d'), {
            type: 'line',
            data: { labels: [], datasets: [{ label: 'Loss', data: [], borderColor: '#38bdf8', backgroundColor: 'rgba(56,189,248,0.1)', borderWidth: 2, tension: 0.4, fill: true, pointRadius: 0 }] },
            options: { responsive: true, maintainAspectRatio: false, scales: { x: { display: false }, y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8', font: { size: 10 } } } }, plugins: { legend: { display: false } } }
        });
    }

    // Worker Init & Stop Logic
    let trainingWorker = null;

    function initWorker() {
        if (trainingWorker) trainingWorker.terminate();
        trainingWorker = new Worker('js/training-worker.js');
        trainingWorker.onmessage = async (e) => {
            const { type, epoch, total, loss, message, modelName } = e.data;
            switch(type) {
                case 'epoch_end':
                    ui.epochText.textContent = `${modelName || 'IA'}: ${epoch}/${total}`;
                    ui.lossText.textContent = `Loss: ${loss}`;
                    ui.trainProgress.style.width = (epoch/total*100)+'%';
                    if (epoch%2===0) { lossChart.data.labels.push(epoch); lossChart.data.datasets[0].data.push(parseFloat(loss)); lossChart.update(); }
                    break;
                case 'training_complete':
                    updateModelStatus('✅ IA v4.0 Ensemble (5 modelos)', '#10b981');
                    await aiModel.loadModel();
                    ui.btnPredict.disabled = false;
                    aiModel.buildMarkovChains(allDraws);
                    runAllAnalytics();
                    ui.btnTrain.disabled = false;
                    if (ui.btnStopTrain) ui.btnStopTrain.style.display = 'none';
                    break;
                case 'log': console.log(message); break;
                case 'error': 
                    console.error('Worker:', message); 
                    updateModelStatus('⚠️ ' + message.substring(0,50), 'red'); 
                    ui.btnTrain.disabled = false; 
                    if (ui.btnStopTrain) ui.btnStopTrain.style.display = 'none';
                    break;
            }
        };
    }
    initWorker();

    if (ui.btnStopTrain) {
        ui.btnStopTrain.addEventListener('click', async () => {
            initWorker(); // Terminates the current worker instantly
            ui.btnTrain.disabled = false;
            ui.btnStopTrain.style.display = 'none';
            updateModelStatus('⚠️ Entrenamiento abortado por el usuario', '#f59e0b');
            ui.epochText.textContent = 'Parado';
            ui.lossText.textContent = '—';
            ui.trainProgress.style.width = '0%';
            // If we aborted but had good old models, reload them to allow predicting
            const loaded = await aiModel.loadModel();
            if (loaded) ui.btnPredict.disabled = false;
        });
    }

    async function forceRetrain() {
        if (allDraws.length < aiModel.windowSize + 5) return;
        ui.btnTrain.disabled = true;
        if (ui.btnStopTrain) ui.btnStopTrain.style.display = 'inline-flex';
        setPendingDraws(0);
        updateModelStatus('Entrenamiento v4.0 (4 modelos + Markov + Python)...', '#ff7e5f');
        lossChart.data.labels = []; lossChart.data.datasets[0].data = []; lossChart.update();
        trainingWorker.postMessage({ type: 'start_training', draws: allDraws });
        
        // Disparar background task para entrenar Python
        fetch('http://localhost:5001/api/python/train', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ draws: allDraws.map(d => [d.n1, d.n2, d.n3, d.date]) })
        }).then(r => r.json()).then(data => console.log("[Python Train]", data))
          .catch(e => console.warn("API de Python no disponible, asegúrate de correr INICIAR_TITAN.bat", e));
    }

    // Init
    try {
        try {
            if (tf.engine().backendName !== 'webgl') await tf.setBackend('webgl');
            const t = tf.tensor1d([1,2,3]); t.dataSync(); t.dispose();
            console.log('✅ GPU WebGL');
        } catch (e) { await tf.setBackend('cpu'); console.log('✅ CPU fallback'); }

        await dbManager.init();
        initChart();
        await reloadData();

        const loaded = await aiModel.loadModel();
        if (loaded) {
            const mCount = Object.values(aiModel.models).filter(m => m).length;
            updateModelStatus(`IA v4.0 ${mCount}+1 modelos (${tf.engine().backendName.toUpperCase()})`, '#10b981');
            ui.btnPredict.disabled = false;
            aiModel.buildMarkovChains(allDraws);
            runAllAnalytics();
        } else if (allDraws.length > aiModel.windowSize + 10) forceRetrain();

        if (shouldAutoSync() && ui.btnSyncCloud) setTimeout(() => syncOnline(), 2000);
        updateTrainNotification();
        updatePredictionHistory();
    } catch (e) { console.error('Init:', e); }

    async function reloadData() {
        allDraws = await dbManager.getAllDraws();
        ui.dbCount.textContent = allDraws.length.toLocaleString();
        ui.historyBody.innerHTML = '';
        allDraws.slice(-20).reverse().forEach(draw => {
            const tr = document.createElement('tr');
            tr.innerHTML = `<td><span class="date-chip">${draw.date}</span></td><td><span class="num-cell">${draw.n1}</span></td><td><span class="num-cell">${draw.n2}</span></td><td><span class="num-cell">${draw.n3}</span></td>`;
            ui.historyBody.appendChild(tr);
        });

        if (allDraws.length >= aiModel.windowSize + 5) {
            ui.btnTrain.disabled = false;
            calculateHistoricalStatistics(allDraws);
        } else { ui.btnTrain.disabled = true; ui.btnPredict.disabled = true; }

        function resetManual() {
            ui.manualN1.value = ui.manualN2.value = ui.manualN3.value = '';
            ui.manualDate.value = new Date().toLocaleDateString();
        }
        resetManual();

        ui.btnManualAdd.onclick = async () => {
            const d = ui.manualDate.value + ' (' + ui.manualShift.value + ')';
            const n1 = parseInt(ui.manualN1.value), n2 = parseInt(ui.manualN2.value), n3 = parseInt(ui.manualN3.value);
            if (isNaN(n1)||isNaN(n2)||isNaN(n3)) { alert('Ingresa 3 dígitos (0-9)'); return; }
            const added = await cleanAndSort([{ date: d, n1, n2, n3 }]);
            if (added > 0) {
                await reloadData(); resetManual(); addPendingDraws(1);
                // Online learning: micro-update
                if (aiModel.isTrained) { await aiModel.incrementalUpdate(allDraws); console.log('🔄 Online learning applied'); }
                if (getPendingDraws() >= RETRAIN_THRESHOLD) forceRetrain();
            } else alert('Sorteo ya existe.');
        };
        if (window.lucide) lucide.createIcons();
    }

    function calculateHistoricalStatistics(draws) {
        if (!draws || draws.length === 0) return;
        const comboFreq = new Map();
        const numFreq = new Uint32Array(10);
        for (const d of draws) {
            const k = `${d.n1}-${d.n2}-${d.n3}`;
            comboFreq.set(k, (comboFreq.get(k)||0)+1);
            numFreq[d.n1]++; numFreq[d.n2]++; numFreq[d.n3]++;
        }
        let maxCombo = { key: '-- -- --', count: -1 };
        for (let [k,c] of comboFreq) if (c > maxCombo.count) maxCombo = { key: k, count: c };
        ui.statCombo.textContent = maxCombo.key.replace(/-/g, ' ');
        const sorted = [...Array(10).keys()].sort((a,b) => numFreq[b]-numFreq[a]);
        ui.statHot.textContent = `${sorted[0]} ${sorted[1]} ${sorted[2]}`;
        ui.statCold.textContent = `${sorted[9]} ${sorted[8]} ${sorted[7]}`;
    }

    // PDF Parser
    ui.pdfUpload.addEventListener('change', async (event) => {
        const file = event.target.files[0]; if (!file) return;
        event.target.value = '';
        ui.dataProgressContainer.style.display = 'block';
        ui.dataProgressFill.style.width = '5%'; ui.dataProgressText.textContent = 'Leyendo PDF...';
        const reader = new FileReader();
        reader.onload = async function() {
            try {
                pdfjsLib.GlobalWorkerOptions.workerSrc = '/js/pdf.worker.min.js';
                const pdf = await pdfjsLib.getDocument({ data: new Uint8Array(this.result), verbosity: 0 }).promise;
                const allItems = []; let allText = '';
                for (let i = 1; i <= pdf.numPages; i++) {
                    const page = await pdf.getPage(i);
                    const tc = await page.getTextContent();
                    const items = tc.items.map(it => it.str.trim()).filter(s => s.length > 0);
                    allItems.push(...items); allText += items.join(' ') + '\n';
                    ui.dataProgressFill.style.width = (5 + Math.round((i/pdf.numPages)*80)) + '%';
                    ui.dataProgressText.textContent = `Leyendo: ${i}/${pdf.numPages}`;
                }
                let newDraws = [];
                const allDates = [...allText.matchAll(/\d{1,2}\/\d{1,2}\/\d{2,4}/g)].map(m => m[0]);
                const allDigits = allItems.filter(it => /^[0-9]$/.test(it)).map(Number);
                if (allDates.length > 0 && allDigits.length > 0) {
                    const pairs = Math.min(allDates.length, Math.floor(allDigits.length/3));
                    for (let i = 0; i < pairs; i++) newDraws.push({ date: allDates[i], n1: allDigits[i*3], n2: allDigits[i*3+1], n3: allDigits[i*3+2] });
                }
                if (newDraws.length === 0) {
                    const re = /(\d{1,2}\/\d{1,2}\/\d{2,4})\s+(\d)\s+(\d)\s+(\d)(?!\d)/g;
                    let m; while ((m = re.exec(allText)) !== null) newDraws.push({ date: m[1], n1: +m[2], n2: +m[3], n3: +m[4] });
                }
                if (newDraws.length > 0) {
                    const added = await cleanAndSort(newDraws.reverse());
                    if (added > 0) { ui.dataProgressText.textContent = `✅ ${added} nuevos`; await reloadData(); addPendingDraws(added); if (getPendingDraws() >= RETRAIN_THRESHOLD) forceRetrain(); }
                    else { ui.dataProgressText.textContent = 'Ya registrados.'; }
                } else { alert('No se encontraron sorteos en el PDF.'); }
            } catch(e) { console.error('PDF:', e); alert('Error PDF: ' + e.message); }
            setTimeout(() => { ui.dataProgressContainer.style.display = 'none'; }, 3000);
        };
        reader.readAsArrayBuffer(file);
    });

    // CSV
    ui.csvUpload.addEventListener('change', async (e) => {
        const file = e.target.files[0]; if (!file) return;
        ui.dataProgressContainer.style.display = 'block';
        const reader = new FileReader();
        reader.onload = async (ev) => {
            const rows = ev.target.result.split('\n');
            const nd = [];
            rows.forEach(r => { const c = r.split(','); if (c.length >= 4) { const d1=parseInt(c[1]),d2=parseInt(c[2]),d3=parseInt(c[3]); if (!isNaN(d1)) nd.push({date:c[0].trim(),n1:d1,n2:d2,n3:d3}); } });
            if (nd.length > 0) { const added = await cleanAndSort(nd); await reloadData(); if (added > 0) { addPendingDraws(added); if (getPendingDraws() >= RETRAIN_THRESHOLD) forceRetrain(); } }
            setTimeout(() => { ui.dataProgressContainer.style.display = 'none'; }, 3000);
        };
        reader.readAsText(file);
    });

    // Seed
    ui.btnSeed.addEventListener('click', async () => {
        const mock = []; const base = new Date();
        for (let i = 1000; i >= 0; i--) {
            const d = new Date(base); d.setDate(d.getDate() - Math.floor(i/2));
            mock.push({ date: d.toLocaleDateString() + (i%2===0 ? ' MD' : ' EVE'), n1: Math.floor(Math.random()*10), n2: Math.floor(Math.random()*10), n3: Math.floor(Math.random()*10) });
        }
        await dbManager.addDraws(mock);
        await reloadData();
        forceRetrain();
    });

    // Clear
    ui.btnClear.addEventListener('click', async () => {
        if (confirm('¿Borrar TODO (datos + IA)?')) {
            try {
                await dbManager.deleteFullDatabase();
                try { await tf.io.removeModel('indexeddb://florida-pick3-pro-model'); } catch(e) {}
                localStorage.removeItem('oraclePendingDraws');
                localStorage.removeItem('oracleLastSync');
                localStorage.removeItem('oraclePredLog');
                alert('Limpieza completa.'); window.location.href = window.location.pathname + '?v=' + Date.now();
            } catch(e) { alert('Error: ' + e.message); }
        }
    });



    // ═══════════════════════════════════════
    // ALL ANALYTICS
    // ═══════════════════════════════════════
    async function runAllAnalytics() {
        if (!aiModel.isTrained) return;
        await runBacktest();
        await runWalkForward();

        updateProbabilitiesView();
        updatePredictionHistory();
    }

    async function runBacktest() {
        if (!aiModel.isTrained || allDraws.length < aiModel.windowSize + 5) return;
        const report = await aiModel.runBacktest(allDraws, 15);
        if (!report) return;
        ui.backtestPanel.style.display = 'block';
        ui.btPrecision.textContent = report.precision + '%';
        ui.btHits.textContent = report.fullHits;
        ui.btResultsBody.innerHTML = '';
        report.results.reverse().forEach(res => {
            const tr = document.createElement('tr');
            let ht = 'hit-none'; if (res.digitHits===3) ht='hit-full'; else if (res.digitHits>0) ht='hit-partial';
            tr.innerHTML = `<td><small>${res.date}</small></td><td>${res.actual.map(n=>`<span class="bt-num-actual">${n}</span>`).join('')}</td><td>${res.predicted.map(n=>`<span class="bt-num-pred">${n}</span>`).join('')}</td><td><span class="hit-badge ${ht}">${res.digitHits===3?'¡PLENO!':res.digitHits+'/3'}</span></td>`;
            ui.btResultsBody.appendChild(tr);
        });
    }

    async function runWalkForward() {
        if (!aiModel.isTrained || allDraws.length < aiModel.windowSize + 30) return;
        const wf = await aiModel.runWalkForward(allDraws, 5);
        if (!wf) return;
        ui.wfPanel.style.display = 'block';
        ui.wfAvg.textContent = wf.avgPrecision + '%';
        ui.wfConsistency.textContent = wf.consistency;
        ui.wfBody.innerHTML = '';
        wf.folds.forEach(f => {
            const tr = document.createElement('tr');
            tr.innerHTML = `<td>${f.fold}</td><td><small>${f.period}</small></td><td>${f.total}</td><td>${f.fullHits}</td><td><strong>${f.digitPrecision}%</strong></td>`;
            ui.wfBody.appendChild(tr);
        });
    }



    async function updateProbabilitiesView() {
        if (!aiModel.isTrained || allDraws.length < aiModel.windowSize) return;
        const probs = await aiModel.getProbabilities(allDraws);
        if (!probs) return;
        ui.probContainer.innerHTML = '';
        probs.forEach((dp, i) => {
            const row = document.createElement('div'); row.className = 'prob-row';
            const label = document.createElement('div'); label.className = 'prob-label'; label.textContent = `D${i+1}`;
            const bars = document.createElement('div'); bars.className = 'prob-bars';
            const maxP = Math.max(...dp);
            Array.from(dp).forEach((p, val) => {
                const bar = document.createElement('div'); bar.className = 'prob-bar';
                if (p === maxP) bar.classList.add('high');
                bar.style.flexGrow = (p*100)+1;
                bar.setAttribute('data-val', val);
                bar.title = `${val}: ${(p*100).toFixed(1)}%`;
                bars.appendChild(bar);
            });
            row.appendChild(label); row.appendChild(bars); ui.probContainer.appendChild(row);
        });
    }

    function updatePredictionHistory() {
        const stats = aiModel.getPredictionStats();
        if (stats.total > 0) {
            ui.phPanel.style.display = 'block';
            ui.phTotal.textContent = stats.total;
            ui.phAvgHits.textContent = stats.avgHits;
            ui.phFullHits.textContent = stats.fullHits;
            ui.phBody.innerHTML = '';
            (stats.history || []).reverse().forEach(e => {
                const tr = document.createElement('tr');
                const preds = e.predictions ? e.predictions.map(p => p.join('-')).join(' | ') : '—';
                const actual = e.actual ? e.actual.join('-') : '—';
                const hits = e.hits || 0;
                let ht = 'hit-none'; if (hits===3) ht='hit-full'; else if (hits>0) ht='hit-partial';
                tr.innerHTML = `<td><small>${new Date(e.timestamp).toLocaleString()}</small></td><td>${e.shift||'—'}</td><td><small>${preds}</small></td><td><span class="hit-badge ${ht}">${hits}/3</span></td>`;
                ui.phBody.appendChild(tr);
            });
        }
    }

    function updateConfidenceGauge(result) {
        if (!result || !result.predictions || result.predictions.length === 0) return;
        ui.confidenceGauge.style.display = 'block';
        const sc = result.predictions[0].systemConfidence;
        ui.confidenceValue.textContent = sc.label;
        ui.confidenceValue.style.color = sc.color;
        ui.confidenceBar.style.width = (sc.confidence*100) + '%';
        ui.confidenceBar.style.background = sc.color;
        ui.regimeLabel.textContent = result.regime || '—';
        ui.regimeStatus.textContent = result.regime || '—';
        ui.regimeStatus.style.color = result.regime === 'stable' ? '#10b981' : result.regime === 'shift' ? '#ef4444' : '#f59e0b';
        const avgEnt = result.finalProbs ? (result.finalProbs.reduce((s,fp) => s + aiModel.calculateEntropy(fp), 0)/3*100).toFixed(0) : '—';
        ui.entropyValue.textContent = avgEnt + '%';
    }

    // ═══════════════════════════════════════
    // PREDICTIONS WITH ALL MODULES
    // ═══════════════════════════════════════
    ui.btnPredict.addEventListener('click', async () => {
        if (!aiModel.isTrained) return;
        ui.btnPredict.disabled = true;
        const allBalls = [...ui.predBallsMD.flat(), ...ui.predBallsEV.flat()];
        let iter = 0;
        const matrix = setInterval(() => {
            allBalls.forEach(b => { b.textContent = Math.floor(Math.random()*10); b.classList.remove('predicted'); });
            if (iter++ > 20) { clearInterval(matrix); finishPrediction(); }
        }, 50);

        async function finishPrediction() {
            // Midday
            const mdResult = await aiModel.predictUltraEnsemble(allDraws, { numOptions: 3, targetShift: 'midday', numRuns: 80 });
            mdResult.predictions.forEach((g, i) => {
                g.numbers.forEach((n, j) => { const b = ui.predBallsMD[i][j]; b.textContent = n; b.classList.add('predicted'); });
                if (ui.confBadgesMD[i]) {
                    ui.confBadgesMD[i].textContent = g.confidence + '%';
                    ui.confBadgesMD[i].style.color = parseFloat(g.confidence) > 15 ? '#10b981' : parseFloat(g.confidence) > 8 ? '#f59e0b' : '#94a3b8';
                }
            });
            aiModel.logPrediction('midday', mdResult.predictions);

            // Evening
            const evResult = await aiModel.predictUltraEnsemble(allDraws, { numOptions: 3, targetShift: 'evening', numRuns: 80 });
            evResult.predictions.forEach((g, i) => {
                g.numbers.forEach((n, j) => { const b = ui.predBallsEV[i][j]; b.textContent = n; b.classList.add('predicted'); });
                if (ui.confBadgesEV[i]) {
                    ui.confBadgesEV[i].textContent = g.confidence + '%';
                    ui.confBadgesEV[i].style.color = parseFloat(g.confidence) > 15 ? '#10b981' : parseFloat(g.confidence) > 8 ? '#f59e0b' : '#94a3b8';
                }
            });
            aiModel.logPrediction('evening', evResult.predictions);

            updateConfidenceGauge(mdResult);
            updatePredictionHistory();
            ui.btnPredict.disabled = false;
        }
    });

    function updateModelStatus(text, color) { ui.modelStatus.textContent = text; ui.modelStatus.style.color = color; }
    ui.btnTrain.addEventListener('click', forceRetrain);
    if (ui.btnTrainNow) ui.btnTrainNow.addEventListener('click', forceRetrain);
});
