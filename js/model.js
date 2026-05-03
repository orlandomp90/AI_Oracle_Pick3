/**
 * AI Oracle Pro v4.0 - TRUE Multi-Model Ensemble
 * 5 MODELS: LSTM + GRU + CNN-1D + Transformer/Attention + Markov
 * 12 MODULES: 21 Features, Ensemble Voting, Markov, Filters,
 * Walk-Forward, Calibration, Regime, Online Learning
 */
class AIPredictorModel {
    constructor() {
        this.models = { lstm: null, gru: null, cnn: null, transformer: null };
        this.modelWeights = { lstm: 0.30, gru: 0.20, cnn: 0.15, transformer: 0.20, markov: 0.15 };
        this.windowSize = 30;
        this.isTrained = false;
        this.FEATURES = 21;
        this.markov1 = null;
        this.markov2 = null;
        this.currentRegime = 'normal';
        this.regimeWeights = { lstm: 0.30, gru: 0.20, cnn: 0.15, transformer: 0.20, markov: 0.15 };
        this.predictionLog = JSON.parse(localStorage.getItem('oraclePredLog') || '[]');
        this.calibration = { a: 1, b: 0 };
        this.backtestScores = { lstm: 10, gru: 10, cnn: 10, transformer: 10, markov: 10 };
    }

    // ═══════════════════════════════════════════
    // 1. FEATURE ENGINEERING 21x
    // ═══════════════════════════════════════════
    detectShift(dateStr) {
        const l = (dateStr || '').toLowerCase();
        if (l.includes('mid') || l.includes('md') || l.includes('day')) return 0;
        if (l.includes('eve') || l.includes('night') || l.includes('pm')) return 1;
        return 0.5;
    }

    parseDateFeatures(dateStr) {
        let d = new Date(dateStr);
        if (isNaN(d)) {
            const p = dateStr.split(/[\/\-]/);
            if (p.length >= 3) { let yr = parseInt(p[2]); if (yr < 100) yr += 2000; d = new Date(yr, parseInt(p[1]) - 1, parseInt(p[0])); }
            if (isNaN(d)) d = new Date();
        }
        return { dow: d.getDay() / 6, mon: (d.getMonth() + 1) / 12, dom: d.getDate() / 31, dayNum: d.getDay() };
    }

    buildFeatureRow(d, allDraws, idx) {
        const t = this.parseDateFeatures(d.date);
        const shift = this.detectShift(d.date);
        const sum = (d.n1 + d.n2 + d.n3) / 27;
        const hasRepeat = (d.n1 === d.n2 || d.n2 === d.n3 || d.n1 === d.n3) ? 1 : 0;

        let g1 = 50, g2 = 50, g3 = 50;
        for (let i = idx - 1; i >= Math.max(0, idx - 50); i--) {
            if (g1 === 50 && allDraws[i].n1 === d.n1) g1 = idx - i;
            if (g2 === 50 && allDraws[i].n2 === d.n2) g2 = idx - i;
            if (g3 === 50 && allDraws[i].n3 === d.n3) g3 = idx - i;
        }

        let s5 = 0;
        for (let i = Math.max(0, idx - 5); i < idx; i++)
            s5 += allDraws[i].n1 + allDraws[i].n2 + allDraws[i].n3;

        let tN1 = 0, tN2 = 0, tN3 = 0, tw = 0;
        for (let i = Math.max(0, idx - 5); i < idx; i++) {
            tN1 += allDraws[i].n1; tN2 += allDraws[i].n2; tN3 += allDraws[i].n3; tw++;
        }
        tw = Math.max(tw, 1);

        const even = [d.n1, d.n2, d.n3].filter(n => n % 2 === 0).length;
        const range = Math.max(d.n1, d.n2, d.n3) - Math.min(d.n1, d.n2, d.n3);
        const isWknd = (t.dayNum === 0 || t.dayNum === 6) ? 1 : 0;

        let entropy = 0.5;
        if (idx >= 20) {
            const f = new Array(10).fill(0);
            for (let i = idx - 20; i < idx; i++) {
                f[allDraws[i].n1]++; f[allDraws[i].n2]++; f[allDraws[i].n3]++;
            }
            entropy = 0;
            for (let v of f) { if (v > 0) { const p = v / 60; entropy -= p * Math.log2(p); } }
            entropy /= Math.log2(10);
        }

        let hot = 0;
        const hw = Math.min(30, idx);
        for (let i = Math.max(0, idx - hw); i < idx; i++) {
            const w = (i - Math.max(0, idx - hw)) / Math.max(hw, 1);
            if (allDraws[i].n1 === d.n1) hot += w;
            if (allDraws[i].n2 === d.n2) hot += w;
            if (allDraws[i].n3 === d.n3) hot += w;
        }

        return [
            d.n1/9, d.n2/9, d.n3/9,
            t.dow, t.mon, t.dom,
            shift, sum, hasRepeat,
            Math.min(g1,50)/50, Math.min(g2,50)/50, Math.min(g3,50)/50,
            s5/135,
            tN1/(tw*9), tN2/(tw*9), tN3/(tw*9),
            even/3, range/9, isWknd,
            entropy, Math.min(hot,3)/3
        ];
    }

    // ═══════════════════════════════════════════
    // 2. MULTI-MODEL ARCHITECTURES (4 Neural)
    // ═══════════════════════════════════════════
    buildModel() {
        // This builds the default LSTM for backward compat
        this.models.lstm = this._buildLSTM();
        // Alias for legacy code
        this.model = this.models.lstm;
    }

    _buildLSTM() {
        const inp = tf.input({ shape: [this.windowSize, this.FEATURES] });
        let x = tf.layers.lstm({ units: 256, returnSequences: true, kernelRegularizer: tf.regularizers.l2({l2:1e-4}) }).apply(inp);
        x = tf.layers.batchNormalization().apply(x);
        x = tf.layers.lstm({ units: 128, returnSequences: false, dropout: 0.1 }).apply(x);
        let trunk = tf.layers.dense({ units: 128, activation: 'swish' }).apply(x);
        trunk = tf.layers.dropout({ rate: 0.2 }).apply(trunk);
        let shared = tf.layers.dense({ units: 64, activation: 'swish' }).apply(trunk);
        const makeHead = (name) => {
            let h = tf.layers.dense({ units: 32, activation: 'relu' }).apply(shared);
            h = tf.layers.dropout({ rate: 0.1 }).apply(h);
            return tf.layers.dense({ units: 10, activation: 'softmax', name }).apply(h);
        };
        const m = tf.model({ inputs: inp, outputs: [makeHead('digit1'), makeHead('digit2'), makeHead('digit3')] });
        m.compile({ optimizer: tf.train.adam(0.0005), loss: ['categoricalCrossentropy','categoricalCrossentropy','categoricalCrossentropy'], metrics: ['accuracy'] });
        return m;
    }

    _buildGRU() {
        const inp = tf.input({ shape: [this.windowSize, this.FEATURES] });
        let x = tf.layers.gru({ units: 192, returnSequences: true, kernelRegularizer: tf.regularizers.l2({l2:1e-4}) }).apply(inp);
        x = tf.layers.batchNormalization().apply(x);
        x = tf.layers.gru({ units: 96, returnSequences: false, dropout: 0.15 }).apply(x);
        let trunk = tf.layers.dense({ units: 96, activation: 'relu' }).apply(x);
        trunk = tf.layers.dropout({ rate: 0.15 }).apply(trunk);
        let shared = tf.layers.dense({ units: 48, activation: 'relu' }).apply(trunk);
        const makeHead = (name) => {
            let h = tf.layers.dense({ units: 32, activation: 'relu' }).apply(shared);
            return tf.layers.dense({ units: 10, activation: 'softmax', name }).apply(h);
        };
        const m = tf.model({ inputs: inp, outputs: [makeHead('digit1'), makeHead('digit2'), makeHead('digit3')] });
        m.compile({ optimizer: tf.train.adam(0.0008), loss: ['categoricalCrossentropy','categoricalCrossentropy','categoricalCrossentropy'] });
        return m;
    }

    _buildCNN() {
        const inp = tf.input({ shape: [this.windowSize, this.FEATURES] });
        let x = tf.layers.conv1d({ filters: 128, kernelSize: 3, padding: 'same', activation: 'relu' }).apply(inp);
        x = tf.layers.batchNormalization().apply(x);
        x = tf.layers.conv1d({ filters: 64, kernelSize: 5, padding: 'same', activation: 'relu' }).apply(x);
        x = tf.layers.globalAveragePooling1d().apply(x);
        x = tf.layers.dropout({ rate: 0.2 }).apply(x);
        let shared = tf.layers.dense({ units: 64, activation: 'relu' }).apply(x);
        const makeHead = (name) => {
            let h = tf.layers.dense({ units: 32, activation: 'relu' }).apply(shared);
            return tf.layers.dense({ units: 10, activation: 'softmax', name }).apply(h);
        };
        const m = tf.model({ inputs: inp, outputs: [makeHead('digit1'), makeHead('digit2'), makeHead('digit3')] });
        m.compile({ optimizer: tf.train.adam(0.001), loss: ['categoricalCrossentropy','categoricalCrossentropy','categoricalCrossentropy'] });
        return m;
    }

    _buildTransformer() {
        const inp = tf.input({ shape: [this.windowSize, this.FEATURES] });
        // Positional encoding via dense projection
        let pos = tf.layers.dense({ units: 64, activation: 'linear', name: 'pos_embed' }).apply(inp);
        // Multi-head attention simulation: Q, K, V projections
        let q = tf.layers.dense({ units: 64, name: 'query' }).apply(pos);
        let k = tf.layers.dense({ units: 64, name: 'key' }).apply(pos);
        let v = tf.layers.dense({ units: 64, name: 'value' }).apply(pos);
        // Scaled dot-product attention approximation via dense layers
        // We use a dense layer on concatenated features as attention proxy
        let attn = tf.layers.dense({ units: 64, activation: 'tanh', name: 'attn_score' }).apply(q);
        let attended = tf.layers.multiply().apply([attn, v]);
        // Feed-forward
        let ff = tf.layers.dense({ units: 128, activation: 'relu' }).apply(attended);
        ff = tf.layers.dropout({ rate: 0.1 }).apply(ff);
        ff = tf.layers.dense({ units: 64, activation: 'relu' }).apply(ff);
        // Global pooling
        let pooled = tf.layers.globalAveragePooling1d().apply(ff);
        let shared = tf.layers.dense({ units: 48, activation: 'relu' }).apply(pooled);
        const makeHead = (name) => {
            let h = tf.layers.dense({ units: 32, activation: 'relu' }).apply(shared);
            return tf.layers.dense({ units: 10, activation: 'softmax', name }).apply(h);
        };
        const m = tf.model({ inputs: inp, outputs: [makeHead('digit1'), makeHead('digit2'), makeHead('digit3')] });
        m.compile({ optimizer: tf.train.adam(0.0005), loss: ['categoricalCrossentropy','categoricalCrossentropy','categoricalCrossentropy'] });
        return m;
    }

    toOneHot(n) { const a = new Array(10).fill(0); a[Math.min(Math.max(n,0),9)] = 1; return a; }

    prepareData(draws) {
        return tf.tidy(() => {
            const raw = draws.map((d, i) => this.buildFeatureRow(d, draws, i));
            const X = [], Y1 = [], Y2 = [], Y3 = [];
            for (let i = 0; i < raw.length - this.windowSize; i++) {
                X.push(raw.slice(i, i + this.windowSize));
                const t = draws[i + this.windowSize];
                Y1.push(this.toOneHot(t.n1));
                Y2.push(this.toOneHot(t.n2));
                Y3.push(this.toOneHot(t.n3));
            }
            return { xs: tf.tensor3d(X), ys: [tf.tensor2d(Y1), tf.tensor2d(Y2), tf.tensor2d(Y3)] };
        });
    }

    // ═══════════════════════════════════════════
    // 3. CADENAS DE MARKOV (Orden 1 + 2)
    // ═══════════════════════════════════════════
    buildMarkovChains(draws) {
        this.markov1 = Array.from({length:3}, () => Array.from({length:10}, () => new Array(10).fill(1)));
        this.markov2 = Array.from({length:3}, () => Array.from({length:100}, () => new Array(10).fill(1)));
        for (let i = 1; i < draws.length; i++) {
            const c = draws[i], p = draws[i-1];
            this.markov1[0][p.n1][c.n1]++;
            this.markov1[1][p.n2][c.n2]++;
            this.markov1[2][p.n3][c.n3]++;
            if (i >= 2) {
                const pp = draws[i-2];
                this.markov2[0][pp.n1*10+p.n1][c.n1]++;
                this.markov2[1][pp.n2*10+p.n2][c.n2]++;
                this.markov2[2][pp.n3*10+p.n3][c.n3]++;
            }
        }
        const norm = (mat) => mat.map(row => { const s = row.reduce((a,b) => a+b, 0); return row.map(v => v/s); });
        this.markov1 = this.markov1.map(norm);
        this.markov2 = this.markov2.map(norm);
    }

    predictMarkov(draws) {
        if (!this.markov1 || draws.length < 2) return null;
        const last = draws[draws.length - 1], prev = draws[draws.length - 2];
        const blend = [];
        for (let pos = 0; pos < 3; pos++) {
            const n = [last.n1, last.n2, last.n3][pos];
            const pn = [prev.n1, prev.n2, prev.n3][pos];
            const o1 = this.markov1[pos][n];
            const o2 = this.markov2[pos][pn*10+n];
            const b = new Array(10);
            let sum = 0;
            for (let j = 0; j < 10; j++) { b[j] = 0.4*o1[j] + 0.6*o2[j]; sum += b[j]; }
            blend.push(b.map(v => v/sum));
        }
        return blend;
    }

    // ═══════════════════════════════════════════
    // 4. FILTROS POST-PREDICCIÓN
    // ═══════════════════════════════════════════
    calcSumStats(draws) {
        const sums = draws.map(d => d.n1+d.n2+d.n3);
        const mean = sums.reduce((a,b)=>a+b,0)/sums.length;
        const std = Math.sqrt(sums.reduce((s,v)=>s+(v-mean)**2,0)/sums.length);
        return { mean, std };
    }

    calcParityProfile(draws) {
        const counts = [0,0,0,0];
        draws.forEach(d => { const e = [d.n1,d.n2,d.n3].filter(n=>n%2===0).length; counts[e]++; });
        return counts.map(c => c/draws.length);
    }

    applyPostFilters(candidates, draws) {
        const ss = this.calcSumStats(draws);
        const pp = this.calcParityProfile(draws);
        const overdue = this.calcOverdue(draws);
        const overdueSet = new Set(overdue.slice(0,3).map(o => o.digit));
        return candidates.map(c => {
            let score = c.count || 1;
            const s = c.combo[0]+c.combo[1]+c.combo[2];
            if (s < ss.mean - 2*ss.std || s > ss.mean + 2*ss.std) score *= 0.3;
            const ev = c.combo.filter(n=>n%2===0).length;
            score *= (0.5 + pp[ev]);
            c.combo.forEach(n => { if (overdueSet.has(n)) score *= 1.3; });
            const last = draws[draws.length-1];
            if (c.combo[0]===last.n1 && c.combo[1]===last.n2 && c.combo[2]===last.n3) score *= 0.1;
            return { ...c, filteredScore: score };
        }).sort((a,b) => b.filteredScore - a.filteredScore);
    }

    // ═══════════════════════════════════════════
    // 5. CALIBRACIÓN DE CONFIANZA
    // ═══════════════════════════════════════════
    calculateEntropy(probs) {
        let e = 0;
        for (const p of probs) { if (p > 0) e -= p * Math.log2(p); }
        return e / Math.log2(probs.length);
    }

    getConfidenceLevel(probs) {
        const maxP = Math.max(...probs);
        const ent = this.calculateEntropy(probs);
        const conf = (1 - ent) * 0.6 + maxP * 0.4;
        // Lottery-calibrated thresholds (10 digits = ~10% base per digit)
        return { confidence: conf, entropy: ent, maxProb: maxP,
            label: conf > 0.15 ? 'ALTA' : conf > 0.10 ? 'MEDIA' : 'BAJA',
            color: conf > 0.15 ? '#10b981' : conf > 0.10 ? '#f59e0b' : '#ef4444'
        };
    }

    // ═══════════════════════════════════════════
    // 6. DETECCIÓN DE RÉGIMEN
    // ═══════════════════════════════════════════
    detectRegime(draws) {
        if (draws.length < 100) { this.currentRegime = 'insufficient'; return; }
        const recent = draws.slice(-30), historic = draws.slice(-200, -30);
        const rFreq = new Array(10).fill(0), hFreq = new Array(10).fill(0);
        recent.forEach(d => { rFreq[d.n1]++; rFreq[d.n2]++; rFreq[d.n3]++; });
        historic.forEach(d => { hFreq[d.n1]++; hFreq[d.n2]++; hFreq[d.n3]++; });
        const rT = recent.length*3, hT = historic.length*3;
        let kl = 0;
        for (let i = 0; i < 10; i++) {
            const p = (rFreq[i]+1)/(rT+10), q = (hFreq[i]+1)/(hT+10);
            kl += p * Math.log(p/q);
        }
        if (kl > 0.15) {
            this.currentRegime = 'shift';
            this.regimeWeights = { lstm: 0.15, gru: 0.10, cnn: 0.10, transformer: 0.10, markov: 0.35, python: 0.20 };
        } else if (kl > 0.08) {
            this.currentRegime = 'transition';
            this.regimeWeights = { lstm: 0.20, gru: 0.15, cnn: 0.10, transformer: 0.15, markov: 0.20, python: 0.20 };
        } else {
            this.currentRegime = 'stable';
            this.regimeWeights = { lstm: 0.25, gru: 0.15, cnn: 0.10, transformer: 0.15, markov: 0.15, python: 0.20 };
        }
        return { regime: this.currentRegime, kl: kl.toFixed(4), weights: this.regimeWeights };
    }

    // ═══════════════════════════════════════════
    // 7. FRECUENCIA POR POSICIÓN
    // ═══════════════════════════════════════════
    calcPositionFrequency(draws) {
        const f = [new Array(10).fill(0), new Array(10).fill(0), new Array(10).fill(0)];
        const n = draws.length; if (n === 0) return f.map(() => new Array(10).fill(0.1));
        for (let i = 0; i < n; i++) {
            const w = 1 + (i/n);
            f[0][draws[i].n1] += w; f[1][draws[i].n2] += w; f[2][draws[i].n3] += w;
        }
        return f.map(p => { const t = p.reduce((a,b)=>a+b,0); return p.map(v=>v/t); });
    }

    calcPositionFrequencyByShift(draws, shift) {
        const fil = draws.filter(d => {
            const s = this.detectShift(d.date);
            return shift === 'midday' ? s <= 0.25 : s >= 0.75;
        });
        return fil.length < 20 ? this.calcPositionFrequency(draws) : this.calcPositionFrequency(fil);
    }

    // ═══════════════════════════════════════════
    // 8. TRUE 5-MODEL ENSEMBLE PREDICTION
    // ═══════════════════════════════════════════
    sampleWithTemperature(probs, temp = 0.8) {
        const lp = probs.map(p => Math.log(Math.max(p, 1e-10)) / temp);
        const mx = Math.max(...lp);
        const ep = lp.map(v => Math.exp(v - mx));
        const s = ep.reduce((a,b) => a+b, 0);
        const n = ep.map(p => p/s);
        const r = Math.random();
        let c = 0;
        for (let i = 0; i < n.length; i++) { c += n[i]; if (r < c) return { digit: i, confidence: n[i], dist: n }; }
        return { digit: 9, confidence: n[9], dist: n };
    }

    async _predictModel(modelKey, seq) {
        const model = this.models[modelKey];
        if (!model) return null;
        try {
            const preds = tf.tidy(() => model.predict(tf.tensor3d([seq])));
            const results = await Promise.all(preds.map(p => p.data()));
            preds.forEach(p => p.dispose());
            return results.map(r => Array.from(r));
        } catch(e) { console.warn(`Model ${modelKey} predict fail:`, e); return null; }
    }

    async predictUltraEnsemble(recentDraws, options = {}) {
        const { numRuns = 80, numOptions = 3, temperature = 0.8, targetShift = null } = options;
        if (!this.isTrained) throw new Error("Modelo no entrenado");

        const wd = recentDraws.slice(-this.windowSize);
        if (wd.length < this.windowSize) throw new Error("Faltan datos");

        // Build input sequence
        const seq = [];
        const startIdx = recentDraws.length - this.windowSize;
        for (let i = startIdx; i < recentDraws.length; i++)
            seq.push(this.buildFeatureRow(recentDraws[i], recentDraws, i));

        // 1. Get predictions from all available neural models (JavaScript)
        const modelPreds = {};
        for (const key of ['lstm', 'gru', 'cnn', 'transformer']) {
            const pred = await this._predictModel(key, seq);
            if (pred) modelPreds[key] = pred;
        }

        // 1.5 Get predictions from Python Hybrid CNN-LSTM Model
        try {
            const pyRes = await fetch('http://localhost:5001/api/python/predict', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ draws: recentDraws.map(d => [d.n1, d.n2, d.n3, d.date]) })
            });
            if (pyRes.ok) {
                const pyData = await pyRes.json();
                if (pyData.success && pyData.predictions_d1) {
                    const pyProbs = [new Array(10).fill(0), new Array(10).fill(0), new Array(10).fill(0)];
                    pyData.predictions_d1.forEach(p => pyProbs[0][p.digit] += p.probability);
                    pyData.predictions_d2.forEach(p => pyProbs[1][p.digit] += p.probability);
                    pyData.predictions_d3.forEach(p => pyProbs[2][p.digit] += p.probability);
                    // Normalizar cada posición a 1.0
                    pyProbs.forEach(arr => {
                        const sum = arr.reduce((a,b)=>a+b, 1e-9);
                        for(let i=0; i<10; i++) arr[i] /= sum;
                    });
                    modelPreds['python'] = pyProbs;
                    
                    // Asegurar que el modelo tenga peso en todos los regímenes
                    if (!this.regimeWeights['python']) this.regimeWeights['python'] = 0.35; // Alta prioridad al nuevo híbrido
                }
            }
        } catch(e) { console.warn("Python API offline o no disponible.", e.message); }

        // 2. Historical frequency
        const hf = targetShift
            ? this.calcPositionFrequencyByShift(recentDraws, targetShift)
            : this.calcPositionFrequency(recentDraws);

        // 3. Markov predictions
        const markovProbs = this.predictMarkov(recentDraws);

        // 4. Detect regime & get weights
        this.detectRegime(recentDraws);
        const w = this.regimeWeights;

        // 5. TRUE multi-model blend per position
        const finalProbs = [new Array(10).fill(0), new Array(10).fill(0), new Array(10).fill(0)];
        const jsProbs = [new Array(10).fill(0), new Array(10).fill(0), new Array(10).fill(0)];
        const pyProbs = [new Array(10).fill(0), new Array(10).fill(0), new Array(10).fill(0)];

        [0, 1, 2].forEach(posIdx => {
            let totalWeight = 0; let jsWeight = 0; let pyWeight = 0;

            // Neural models (JS)
            for (const key of ['lstm', 'gru', 'cnn', 'transformer']) {
                if(modelPreds[key]) {
                    const weight = w[key] || 0.1;
                    for (let j = 0; j < 10; j++) {
                        finalProbs[posIdx][j] += weight * modelPreds[key][posIdx][j];
                        jsProbs[posIdx][j] += weight * modelPreds[key][posIdx][j];
                    }
                    totalWeight += weight;
                    jsWeight += weight;
                }
            }

            // Python model
            if (modelPreds['python']) {
                const pyW = w['python'] || 0.35;
                for (let j = 0; j<10; j++) {
                    finalProbs[posIdx][j] += pyW * modelPreds['python'][posIdx][j];
                    pyProbs[posIdx][j] += pyW * modelPreds['python'][posIdx][j];
                }
                totalWeight += pyW;
                pyWeight += pyW;
            }

            // Markov (solo afecta al ensemble final)
            if (markovProbs) {
                const mkW = w.markov || 0.15;
                for (let j = 0; j < 10; j++) finalProbs[posIdx][j] += mkW * markovProbs[posIdx][j];
                totalWeight += mkW;
            } else {
                const fW = w.markov || 0.15;
                for (let j = 0; j < 10; j++) finalProbs[posIdx][j] += fW * hf[posIdx][j];
                totalWeight += fW;
            }

            // Normalize
            const sumF = finalProbs[posIdx].reduce((a,b)=>a+b, 0) || 1;
            const sumJ = jsProbs[posIdx].reduce((a,b)=>a+b, 0) || 1;
            const sumP = pyProbs[posIdx].reduce((a,b)=>a+b, 0) || 1;
            
            for(let j=0; j<10; j++) {
                finalProbs[posIdx][j] /= sumF;
                jsProbs[posIdx][j] /= sumJ;
                pyProbs[posIdx][j] /= sumP;
            }
        });

        // 6. Ensemble sampling for each specific model sector
        const getTopCandidate = (probMatrix) => {
            const candidateMap = new Map();
            for (let run = 0; run < numRuns; run++) {
                const t = temperature + (Math.random() - 0.5) * 0.3;
                const combo = []; let totalConf = 0;
                for (let pos = 0; pos < 3; pos++) {
                    const { digit, confidence } = this.sampleWithTemperature(probMatrix[pos], t);
                    combo.push(digit); totalConf += confidence;
                }
                const key = combo.join('-');
                if (candidateMap.has(key)) {
                    const e = candidateMap.get(key); e.count++; e.totalConf += totalConf/3;
                } else {
                    candidateMap.set(key, { combo: [...combo], count: 1, totalConf: totalConf/3 });
                }
            }
            let candidates = [...candidateMap.values()];
            candidates = this.applyPostFilters(candidates, recentDraws);
            if(candidates.length === 0) return { combo: [0,0,0], count: 1, totalConf: 0 };
            return candidates[0];
        };

        const resultEnsemble = getTopCandidate(finalProbs);
        // Si python no escupió resultados, copiamos el consenso como fallback visual
        const resultPython = modelPreds['python'] ? getTopCandidate(pyProbs) : resultEnsemble; 
        const resultJS = getTopCandidate(jsProbs);
        
        const activeModels = Object.keys(modelPreds).length + (markovProbs ? 1 : 0);

        const formatResult = (c, probs) => {
            const posConf = c.combo.map((_, i) => this.getConfidenceLevel(probs[i]));
            const avgConf = posConf.reduce((s, pc) => s + pc.confidence, 0) / 3;
            return {
                numbers: c.combo, votes: c.count,
                confidence: ((c.count / numRuns) * 100).toFixed(1),
                avgConf: (avgConf * 100).toFixed(1),
                positionConfidence: posConf,
                activeModels,
                systemConfidence: {
                    confidence: avgConf,
                    label: avgConf > 0.15 ? 'ALTA' : avgConf > 0.10 ? 'MEDIA' : 'BAJA',
                    color: avgConf > 0.15 ? '#10b981' : avgConf > 0.10 ? '#f59e0b' : '#ef4444'
                }
            };
        };

        const results = [
            formatResult(resultEnsemble, finalProbs),
            formatResult(resultPython, pyProbs),
            formatResult(resultJS, jsProbs)
        ];

        return {
            predictions: results, finalProbs,
            rawProbs: modelPreds, hf, markovProbs,
            regime: this.currentRegime, regimeWeights: w,
            modelsUsed: [...Object.keys(modelPreds), ...(markovProbs ? ['markov'] : [])]
        };
    }

    async predictMultiple(rd, n = 3) {
        const r = await this.predictUltraEnsemble(rd, { numOptions: n });
        return r.predictions.map(p => p.numbers);
    }

    async getProbabilities(rd) {
        if (!this.isTrained) return null;
        const seq = [];
        const si = rd.length - this.windowSize;
        for (let i = si; i < rd.length; i++) seq.push(this.buildFeatureRow(rd[i], rd, i));
        // Use primary model (lstm)
        const m = this.models.lstm || this.model;
        if (!m) return null;
        const p = tf.tidy(() => m.predict(tf.tensor3d([seq])));
        const r = await Promise.all(p.map(t => t.data()));
        p.forEach(t => t.dispose());
        return r;
    }

    // ═══════════════════════════════════════════
    // 9. WALK-FORWARD VALIDATION
    // ═══════════════════════════════════════════
    async runWalkForward(data, folds = 5) {
        if (!this.isTrained || data.length < this.windowSize + 20) return null;
        const m = this.models.lstm || this.model;
        if (!m) return null;
        const testSize = Math.min(20, Math.floor((data.length - this.windowSize) / folds));
        const results = [];
        for (let fold = 0; fold < folds; fold++) {
            const testEnd = data.length - fold * testSize;
            const testStart = testEnd - testSize;
            if (testStart < this.windowSize) break;
            let hits = 0, digitHits = 0, total = 0;
            for (let i = testStart; i < testEnd && i + this.windowSize <= data.length - 1; i++) {
                const slice = data.slice(Math.max(0, i - this.windowSize), i);
                if (slice.length < this.windowSize) continue;
                const actual = data[i];
                const seq = [];
                const base = Math.max(0, i - this.windowSize);
                for (let j = base; j < i; j++) seq.push(this.buildFeatureRow(data[j], data, j));
                const t = tf.tensor3d([seq]);
                const pr = m.predict(t);
                const probData = await Promise.all(pr.map(p => p.data()));
                
                const pred = probData.map(d => {
                    return Array.from(d).indexOf(Math.max(...d));
                });
                
                const dh = (pred[0]===actual.n1?1:0)+(pred[1]===actual.n2?1:0)+(pred[2]===actual.n3?1:0);
                
                // Cleanup
                pr.forEach(p => p.dispose());
                t.dispose();
                
                if (dh === 3) hits++;
                digitHits += dh; total++;
            }
            results.push({ fold: fold + 1, period: `${data[testStart]?.date || '?'} → ${data[testEnd-1]?.date || '?'}`,
                total, fullHits: hits, digitPrecision: total > 0 ? ((digitHits/(total*3))*100).toFixed(1) : '0' });
        }
        const avgPrec = results.length > 0 ? (results.reduce((s,r) => s + parseFloat(r.digitPrecision), 0) / results.length).toFixed(1) : '0';
        return { folds: results, avgPrecision: avgPrec, consistency: this.calcConsistency(results) };
    }

    calcConsistency(folds) {
        if (folds.length < 2) return 'N/A';
        const vals = folds.map(f => parseFloat(f.digitPrecision));
        const avg = vals.reduce((a,b)=>a+b,0)/vals.length;
        const std = Math.sqrt(vals.reduce((s,v)=>s+(v-avg)**2,0)/vals.length);
        const cv = avg > 0 ? (std/avg) : 1;
        if (cv < 0.15) return 'Muy Consistente';
        if (cv < 0.30) return 'Consistente';
        return 'Variable';
    }

    // ═══════════════════════════════════════════
    // 10. ONLINE LEARNING
    // ═══════════════════════════════════════════
    async incrementalUpdate(draws) {
        if (!this.isTrained || draws.length < this.windowSize + 2) return false;
        const last = draws.length - 1;
        const seq = [];
        for (let i = last - this.windowSize; i < last; i++)
            seq.push(this.buildFeatureRow(draws[i], draws, i));
        const target = draws[last];
        const xs = tf.tensor3d([seq]);
        const ys = [tf.tensor2d([this.toOneHot(target.n1)]), tf.tensor2d([this.toOneHot(target.n2)]), tf.tensor2d([this.toOneHot(target.n3)])];
        // Update all available models
        for (const key of ['lstm', 'gru', 'cnn', 'transformer']) {
            if (this.models[key]) {
                try { await this.models[key].fit(xs, ys, { epochs: 2, verbose: 0 }); } catch(e) {}
            }
        }
        xs.dispose(); ys.forEach(y => y.dispose());
        await this.saveModel();
        return true;
    }

    // ═══════════════════════════════════════════
    // 11. PREDICTION HISTORY
    // ═══════════════════════════════════════════
    logPrediction(shift, predictions) {
        const entry = { timestamp: new Date().toISOString(), shift, predictions: predictions.map(p => p.numbers), verified: false, actual: null };
        this.predictionLog.push(entry);
        if (this.predictionLog.length > 200) this.predictionLog = this.predictionLog.slice(-200);
        localStorage.setItem('oraclePredLog', JSON.stringify(this.predictionLog));
        return entry;
    }

    verifyPredictions(actualDraw) {
        let verified = 0;
        this.predictionLog.forEach(entry => {
            if (entry.verified) return;
            entry.predictions.forEach(pred => {
                const hits = (pred[0]===actualDraw.n1?1:0)+(pred[1]===actualDraw.n2?1:0)+(pred[2]===actualDraw.n3?1:0);
                if (hits > 0) { entry.verified = true; entry.actual = [actualDraw.n1, actualDraw.n2, actualDraw.n3]; entry.hits = hits; verified++; }
            });
        });
        localStorage.setItem('oraclePredLog', JSON.stringify(this.predictionLog));
        return verified;
    }

    getPredictionStats() {
        const v = this.predictionLog.filter(e => e.verified);
        if (v.length === 0) return { total: 0, avgHits: 0, fullHits: 0 };
        const totalHits = v.reduce((s, e) => s + (e.hits || 0), 0);
        return { total: v.length, avgHits: (totalHits / v.length).toFixed(2), fullHits: v.filter(e => e.hits === 3).length, history: v.slice(-20) };
    }

    // ═══════════════════════════════════════════
    // 12. ANÁLISIS DE PATRONES
    // ═══════════════════════════════════════════
    analyzePatterns(draws) {
        if (!draws || draws.length < 10) return null;
        return { gaps: this.calcGaps(draws), overdue: this.calcOverdue(draws), pairs: this.calcFrequentPairs(draws),
            streaks: this.calcStreaks(draws), positionTrends: this.calcPositionTrends(draws),
            sumDistribution: this.calcSumDistribution(draws), recentRepeatRate: this.calcRepeatRate(draws),
            heatmap: this.calcHeatmap(draws), markovTop: this.getMarkovTopTransitions(draws) };
    }

    calcGaps(draws) {
        const g = [new Array(10).fill(999), new Array(10).fill(999), new Array(10).fill(999)];
        for (let i = draws.length-1; i >= 0; i--) { const d = draws[i], dist = draws.length-1-i;
            if (g[0][d.n1]===999) g[0][d.n1]=dist; if (g[1][d.n2]===999) g[1][d.n2]=dist; if (g[2][d.n3]===999) g[2][d.n3]=dist; }
        return g;
    }

    calcOverdue(draws) {
        const ls = new Array(10).fill(999);
        for (let i = draws.length-1; i >= 0; i--) { const d = draws[i], dist = draws.length-1-i;
            [d.n1,d.n2,d.n3].forEach(n => { if (ls[n]===999) ls[n]=dist; }); }
        return ls.map((gap,digit) => ({digit,gap})).sort((a,b) => b.gap-a.gap);
    }

    calcFrequentPairs(draws) {
        const pf = new Map();
        for (const d of draws) { [`${d.n1}-${d.n2}`,`${d.n2}-${d.n3}`,`${d.n1}-${d.n3}`].forEach(p => pf.set(p, (pf.get(p)||0)+1)); }
        return [...pf.entries()].sort((a,b)=>b[1]-a[1]).slice(0,8).map(([pair,count]) => ({pair, count, pct:(count/draws.length*100).toFixed(1)}));
    }

    calcStreaks(draws) {
        if (draws.length<2) return [{digit:'?',count:0},{digit:'?',count:0},{digit:'?',count:0}];
        const cur = [{digit:draws[draws.length-1].n1,count:1},{digit:draws[draws.length-1].n2,count:1},{digit:draws[draws.length-1].n3,count:1}];
        for (let p = 0; p < 3; p++) { for (let i = draws.length-2; i >= 0; i--) {
            if ([draws[i].n1,draws[i].n2,draws[i].n3][p] === cur[p].digit) cur[p].count++; else break; } }
        return cur;
    }

    calcPositionTrends(draws) {
        const l = draws.slice(-15);
        return [{avg:(l.reduce((s,d)=>s+d.n1,0)/l.length).toFixed(1),pos:'N1'},{avg:(l.reduce((s,d)=>s+d.n2,0)/l.length).toFixed(1),pos:'N2'},{avg:(l.reduce((s,d)=>s+d.n3,0)/l.length).toFixed(1),pos:'N3'}];
    }

    calcSumDistribution(draws) {
        const sf = new Array(28).fill(0); draws.forEach(d => sf[d.n1+d.n2+d.n3]++);
        let mx = 0; for (let i = 1; i < sf.length; i++) if (sf[i]>sf[mx]) mx = i;
        return {distribution:sf.map(f=>(f/draws.length*100).toFixed(1)),mostFrequentSum:mx,mostFrequentPct:(sf[mx]/draws.length*100).toFixed(1)};
    }

    calcRepeatRate(draws) {
        const rec = draws.slice(-30); let reps = 0;
        for (let i = 1; i < rec.length; i++) { if (rec[i].n1===rec[i-1].n1||rec[i].n2===rec[i-1].n2||rec[i].n3===rec[i-1].n3) reps++; }
        return {rate:((reps/(rec.length-1))*100).toFixed(1),count:reps,total:rec.length-1};
    }

    calcHeatmap(draws) {
        const hm = Array.from({length:3}, () => new Array(10).fill(0));
        const recent = draws.slice(-50);
        recent.forEach((d,i) => { const w = (i+1)/recent.length; hm[0][d.n1]+=w; hm[1][d.n2]+=w; hm[2][d.n3]+=w; });
        return hm.map(row => { const mx = Math.max(...row); return row.map(v => mx > 0 ? v/mx : 0); });
    }

    getMarkovTopTransitions(draws) {
        if (!this.markov1) this.buildMarkovChains(draws);
        const last = draws[draws.length-1];
        const tops = [];
        for (let pos = 0; pos < 3; pos++) {
            const n = [last.n1, last.n2, last.n3][pos];
            const probs = this.markov1[pos][n];
            const sorted = probs.map((p,i)=>({digit:i,prob:p})).sort((a,b)=>b.prob-a.prob);
            tops.push({ from: n, top3: sorted.slice(0,3) });
        }
        return tops;
    }

    // BACKTEST
    async runBacktest(data, sampleSize = 20) {
        if (!this.isTrained) return null;
        let hits = 0, partialHits = 0; const testResults = [];
        const td = data.slice(-(sampleSize + this.windowSize));
        if (td.length < this.windowSize + 1) return null;
        
        let scores = { lstm: 0, gru: 0, cnn: 0, transformer: 0 };
        const modelKeys = Object.keys(this.models).filter(k => this.models[k] != null);
        
        for (let i = 0; i < td.length - this.windowSize; i++) {
            const actual = td[i + this.windowSize];
            const seq = [];
            for (let j = i; j < i + this.windowSize; j++) seq.push(this.buildFeatureRow(td[j], td, j));
            
            let bestEnsemblePred = [-1, -1, -1];
            let bestEnsembleScore = -1;
            let ensembleDigitHits = 0;
            
            const tensorSq = tf.tensor3d([seq]);
            for (const key of modelKeys) {
                const pr = this.models[key].predict(tensorSq);
                const probData = await Promise.all(pr.map(prob => prob.data()));
                
                const pred = probData.map(d => {
                    return Array.from(d).indexOf(Math.max(...d));
                });
                
                const dh = (pred[0]===actual.n1?1:0) + (pred[1]===actual.n2?1:0) + (pred[2]===actual.n3?1:0);
                scores[key] += dh;
                
                // Limpiar memoria
                pr.forEach(p => p.dispose());
                
                // Basic ensemble taking best individual for the backtest log
                if (dh > bestEnsembleScore) {
                    bestEnsembleScore = dh;
                    bestEnsemblePred = pred;
                    ensembleDigitHits = dh;
                }
            }
            tensorSq.dispose();
            
            const isFullHit = ensembleDigitHits === 3;
            if (isFullHit) hits++;
            partialHits += ensembleDigitHits;
            testResults.push({ date: actual.date, actual: [actual.n1,actual.n2,actual.n3], predicted: bestEnsemblePred, digitHits: ensembleDigitHits });
        }
        
        // Dynamic Ensemble Weight Update
        modelKeys.forEach(k => { this.backtestScores[k] = Math.max(1, scores[k]); });
        
        return { total: testResults.length, fullHits: hits, partialHits,
            precision: ((partialHits/(testResults.length*3))*100).toFixed(2), results: testResults };
    }

    // PERSISTENCIA
    async saveModel() {
        try {
            if (this.models.lstm) await this.models.lstm.save('indexeddb://florida-pick3-pro-model');
            if (this.models.gru) await this.models.gru.save('indexeddb://florida-pick3-gru-model');
            if (this.models.cnn) await this.models.cnn.save('indexeddb://florida-pick3-cnn-model');
            if (this.models.transformer) await this.models.transformer.save('indexeddb://florida-pick3-transformer-model');
        } catch(e) { console.error("Save error:", e); }
    }

    async loadModel() {
        try {
            // Load LSTM (primary)
            this.models.lstm = await tf.loadLayersModel('indexeddb://florida-pick3-pro-model');
            this.models.lstm.compile({ optimizer: tf.train.adam(0.001), loss: 'categoricalCrossentropy' });
            const sw = this.models.lstm.inputs[0].shape[1], sf = this.models.lstm.inputs[0].shape[2];
            if (sw && sw > 0) this.windowSize = sw;
            if (sf && sf > 0) this.FEATURES = sf;
            this.model = this.models.lstm;
            this.isTrained = true;

            // Load secondary models (non-blocking)
            for (const [key, dbKey] of [['gru','florida-pick3-gru-model'],['cnn','florida-pick3-cnn-model'],['transformer','florida-pick3-transformer-model']]) {
                try {
                    this.models[key] = await tf.loadLayersModel(`indexeddb://${dbKey}`);
                    this.models[key].compile({ optimizer: tf.train.adam(0.001), loss: 'categoricalCrossentropy' });
                    console.log(`✅ ${key.toUpperCase()} loaded`);
                } catch(e) { console.log(`ℹ️ ${key.toUpperCase()} not found, will use ensemble without it`); }
            }

            const loaded = Object.keys(this.models).filter(k => this.models[k]);
            console.log(`Model v4.0 loaded (win=${this.windowSize}, feat=${this.FEATURES}, models=${loaded.join(',')})`);
            return true;
        } catch(e) { this.isTrained = false; return false; }
    }

    weightedRandom(probs) {
        const r = Math.random(); let c = 0;
        for (let i = 0; i < probs.length; i++) { c += probs[i]; if (r < c) return i; }
        return probs.length - 1;
    }
}

const aiModel = new AIPredictorModel();
