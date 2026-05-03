/**
 * Worker de Entrenamiento - AI Oracle v4.0
 * Trains 4 models: LSTM + GRU + CNN-1D + Transformer/Attention
 * CPU-only for Web Worker stability
 * 21 FEATURES synchronized with model.js
 */
importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js');
tf.setBackend('cpu');

const FEATURES = 21;

const CONFIG = {
    getWindowSize: (n) => {
        if (n < 100) return 8;
        if (n < 300) return 10;
        return 15;
    },
    getUnits: (n) => {
        if (n < 150) return { lstm: [24, 12], gru: [20, 10], cnn: [20, 10], trans: 12 };
        if (n < 500) return { lstm: [32, 16], gru: [28, 14], cnn: [24, 12], trans: 16 };
        return { lstm: [32, 16], gru: [28, 14], cnn: [32, 16], trans: 20 };
    },
    getTrainParams: (n) => {
        if (n < 150) return { epochs: 50, patience: 10, lr: 0.003, batch: 32 };
        if (n < 500) return { epochs: 60, patience: 12, lr: 0.002, batch: 64 };
        return { epochs: 60, patience: 12, lr: 0.002, batch: 256 };
    }
};

class TrainingWorkerModel {
    constructor() { this.windowSize = 30; }

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
        for (let i = Math.max(0, idx - 5); i < idx; i++) s5 += allDraws[i].n1 + allDraws[i].n2 + allDraws[i].n3;
        let tN1 = 0, tN2 = 0, tN3 = 0, tw = 0;
        for (let i = Math.max(0, idx - 5); i < idx; i++) { tN1 += allDraws[i].n1; tN2 += allDraws[i].n2; tN3 += allDraws[i].n3; tw++; }
        tw = Math.max(tw, 1);
        const even = [d.n1, d.n2, d.n3].filter(n => n % 2 === 0).length;
        const range = Math.max(d.n1, d.n2, d.n3) - Math.min(d.n1, d.n2, d.n3);
        const isWknd = (t.dayNum === 0 || t.dayNum === 6) ? 1 : 0;
        let entropy = 0.5;
        if (idx >= 20) {
            const f = new Array(10).fill(0);
            for (let i = idx - 20; i < idx; i++) { f[allDraws[i].n1]++; f[allDraws[i].n2]++; f[allDraws[i].n3]++; }
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
        return [d.n1/9, d.n2/9, d.n3/9, t.dow, t.mon, t.dom, shift, sum, hasRepeat,
            Math.min(g1,50)/50, Math.min(g2,50)/50, Math.min(g3,50)/50, s5/135,
            tN1/(tw*9), tN2/(tw*9), tN3/(tw*9), even/3, range/9, isWknd, entropy, Math.min(hot,3)/3];
    }

    oneHot(n) { const a = new Array(10).fill(0); a[Math.min(Math.max(n,0),9)] = 1; return a; }

    // ═══════════════════════════════════════════
    // 4 MODEL BUILDERS
    // ═══════════════════════════════════════════
    buildLSTM(ws, u) {
        const inp = tf.input({ shape: [ws, FEATURES] });
        let x = tf.layers.lstm({ units: u.lstm[0], returnSequences: true, kernelInitializer: 'glorotUniform', recurrentInitializer: 'glorotUniform', kernelRegularizer: tf.regularizers.l2({ l2: 5e-5 }) }).apply(inp);
        x = tf.layers.batchNormalization().apply(x);
        x = tf.layers.dropout({ rate: 0.2 }).apply(x);
        x = tf.layers.lstm({ units: u.lstm[1], returnSequences: false, kernelInitializer: 'glorotUniform', recurrentInitializer: 'glorotUniform', kernelRegularizer: tf.regularizers.l2({ l2: 5e-5 }) }).apply(x);
        x = tf.layers.batchNormalization().apply(x);
        x = tf.layers.dropout({ rate: 0.15 }).apply(x);
        x = tf.layers.dense({ units: u.lstm[1], activation: 'relu' }).apply(x);
        x = tf.layers.dropout({ rate: 0.1 }).apply(x);
        const shared = tf.layers.dense({ units: Math.floor(u.lstm[1] / 2), activation: 'relu' }).apply(x);
        const out1 = tf.layers.dense({ units: 10, activation: 'softmax', name: 'digit1' }).apply(shared);
        const out2 = tf.layers.dense({ units: 10, activation: 'softmax', name: 'digit2' }).apply(shared);
        const out3 = tf.layers.dense({ units: 10, activation: 'softmax', name: 'digit3' }).apply(shared);
        return tf.model({ inputs: inp, outputs: [out1, out2, out3] });
    }

    buildGRU(ws, u) {
        const inp = tf.input({ shape: [ws, FEATURES] });
        let x = tf.layers.gru({ units: u.gru[0], returnSequences: true, kernelInitializer: 'glorotUniform', recurrentInitializer: 'glorotUniform', kernelRegularizer: tf.regularizers.l2({ l2: 5e-5 }) }).apply(inp);
        x = tf.layers.batchNormalization().apply(x);
        x = tf.layers.gru({ units: u.gru[1], returnSequences: false, kernelInitializer: 'glorotUniform', recurrentInitializer: 'glorotUniform', dropout: 0.15 }).apply(x);
        x = tf.layers.dense({ units: u.gru[1], activation: 'relu' }).apply(x);
        x = tf.layers.dropout({ rate: 0.15 }).apply(x);
        const shared = tf.layers.dense({ units: Math.floor(u.gru[1] / 2), activation: 'relu' }).apply(x);
        const out1 = tf.layers.dense({ units: 10, activation: 'softmax', name: 'digit1' }).apply(shared);
        const out2 = tf.layers.dense({ units: 10, activation: 'softmax', name: 'digit2' }).apply(shared);
        const out3 = tf.layers.dense({ units: 10, activation: 'softmax', name: 'digit3' }).apply(shared);
        return tf.model({ inputs: inp, outputs: [out1, out2, out3] });
    }

    buildCNN(ws, u) {
        const inp = tf.input({ shape: [ws, FEATURES] });
        let x = tf.layers.conv1d({ filters: u.cnn[0], kernelSize: 3, padding: 'same', activation: 'relu' }).apply(inp);
        x = tf.layers.batchNormalization().apply(x);
        x = tf.layers.conv1d({ filters: u.cnn[1], kernelSize: 5, padding: 'same', activation: 'relu' }).apply(x);
        x = tf.layers.globalAveragePooling1d().apply(x);
        x = tf.layers.dropout({ rate: 0.2 }).apply(x);
        const shared = tf.layers.dense({ units: Math.floor(u.cnn[1] / 2), activation: 'relu' }).apply(x);
        const out1 = tf.layers.dense({ units: 10, activation: 'softmax', name: 'digit1' }).apply(shared);
        const out2 = tf.layers.dense({ units: 10, activation: 'softmax', name: 'digit2' }).apply(shared);
        const out3 = tf.layers.dense({ units: 10, activation: 'softmax', name: 'digit3' }).apply(shared);
        return tf.model({ inputs: inp, outputs: [out1, out2, out3] });
    }

    buildTransformer(ws, u) {
        const inp = tf.input({ shape: [ws, FEATURES] });
        let pos = tf.layers.dense({ units: u.trans, activation: 'linear' }).apply(inp);
        let q = tf.layers.dense({ units: u.trans }).apply(pos);
        let v = tf.layers.dense({ units: u.trans }).apply(pos);
        let attn = tf.layers.dense({ units: u.trans, activation: 'tanh' }).apply(q);
        let attended = tf.layers.multiply().apply([attn, v]);
        let ff = tf.layers.dense({ units: u.trans * 2, activation: 'relu' }).apply(attended);
        ff = tf.layers.dropout({ rate: 0.1 }).apply(ff);
        ff = tf.layers.dense({ units: u.trans, activation: 'relu' }).apply(ff);
        let pooled = tf.layers.globalAveragePooling1d().apply(ff);
        const shared = tf.layers.dense({ units: Math.floor(u.trans / 2), activation: 'relu' }).apply(pooled);
        const out1 = tf.layers.dense({ units: 10, activation: 'softmax', name: 'digit1' }).apply(shared);
        const out2 = tf.layers.dense({ units: 10, activation: 'softmax', name: 'digit2' }).apply(shared);
        const out3 = tf.layers.dense({ units: 10, activation: 'softmax', name: 'digit3' }).apply(shared);
        return tf.model({ inputs: inp, outputs: [out1, out2, out3] });
    }

    prepareData(draws) {
        const ws = this.windowSize;
        const raw = draws.map((d, i) => this.buildFeatureRow(d, draws, i));
        const X = [], Y1 = [], Y2 = [], Y3 = [];
        for (let i = 0; i < raw.length - ws; i++) {
            X.push(raw.slice(i, i + ws));
            const t = draws[i + ws];
            Y1.push(this.oneHot(t.n1)); Y2.push(this.oneHot(t.n2)); Y3.push(this.oneHot(t.n3));
        }
        return { xs: tf.tensor3d(X), ys: [tf.tensor2d(Y1), tf.tensor2d(Y2), tf.tensor2d(Y3)] };
    }

    async trainSingleModel(model, name, xs, ys, tp, totalModels, modelIdx) {
        model.compile({
            optimizer: tf.train.adam(tp.lr),
            loss: ['categoricalCrossentropy', 'categoricalCrossentropy', 'categoricalCrossentropy'],
            metrics: ['accuracy']
        });

        // Fewer epochs for secondary models to speed up training
        const epochScale = name === 'LSTM' ? 1.0 : 0.6;
        const maxEpochs = Math.round(tp.epochs * epochScale);
        let bestLoss = Infinity, patienceLeft = tp.patience;

        self.postMessage({ type: 'log', message: `[${name}] Training ${maxEpochs} epochs...` });

        await model.fit(xs, ys, {
            epochs: maxEpochs, batchSize: tp.batch, shuffle: true,
            validationSplit: xs.shape[0] > 200 ? 0.1 : 0,
            callbacks: {
                onEpochEnd: async (epoch, logs) => {
                    const loss = logs.loss;
                    // Map epoch across all models for progress
                    const globalEpoch = modelIdx * maxEpochs + epoch + 1;
                    const globalTotal = totalModels * maxEpochs;
                    self.postMessage({
                        type: 'epoch_end', epoch: globalEpoch, total: globalTotal,
                        loss: loss.toFixed(5), modelName: name
                    });
                    if (loss < bestLoss - 1e-5) { bestLoss = loss; patienceLeft = tp.patience; }
                    else { patienceLeft--; if (patienceLeft <= 0) { model.stopTraining = true; } }
                    if (patienceLeft === Math.floor(tp.patience / 2)) {
                        const lr = model.optimizer.learningRate;
                        if (lr > 1e-5) model.optimizer.learningRate = lr * 0.5;
                    }
                }
            }
        });
        self.postMessage({ type: 'log', message: `[${name}] Done. Best loss: ${bestLoss.toFixed(5)}` });
        return model;
    }

    async train(draws) {
        await tf.ready();
        if (tf.getBackend() !== 'cpu') await tf.setBackend('cpu');

        const maxDraws = 1500;
        const data = draws.length > maxDraws ? draws.slice(-maxDraws) : draws;
        const ws = CONFIG.getWindowSize(data.length);
        const units = CONFIG.getUnits(data.length);
        const tp = CONFIG.getTrainParams(data.length);
        this.windowSize = ws;

        self.postMessage({ type: 'log', message: `[Worker v4.0] ${data.length} draws | win=${ws} | feat=${FEATURES} | 4 MODELS` });

        if (data.length < ws + 5) throw new Error(`Need at least ${ws + 5} draws.`);

        self.postMessage({ type: 'log', message: `[Worker] Computing 21 features...` });
        const { xs, ys } = this.prepareData(data);
        self.postMessage({ type: 'log', message: `[Worker] Sequences: ${xs.shape[0]} | Shape: [${xs.shape}]` });

        const modelDefs = [
            { name: 'LSTM', build: () => this.buildLSTM(ws, units), dbKey: 'florida-pick3-pro-model' },
            { name: 'GRU', build: () => this.buildGRU(ws, units), dbKey: 'florida-pick3-gru-model' },
            { name: 'CNN', build: () => this.buildCNN(ws, units), dbKey: 'florida-pick3-cnn-model' },
            { name: 'TRANSFORMER', build: () => this.buildTransformer(ws, units), dbKey: 'florida-pick3-transformer-model' }
        ];

        for (let i = 0; i < modelDefs.length; i++) {
            const def = modelDefs[i];
            self.postMessage({ type: 'log', message: `\n═══ Building ${def.name} (${i+1}/${modelDefs.length}) ═══` });
            try {
                const model = def.build();
                await this.trainSingleModel(model, def.name, xs, ys, tp, modelDefs.length, i);
                await model.save(`indexeddb://${def.dbKey}`);
                self.postMessage({ type: 'log', message: `[${def.name}] ✅ Saved to IndexedDB` });
                model.dispose();
            } catch (err) {
                self.postMessage({ type: 'log', message: `[${def.name}] ⚠️ Failed: ${err.message}` });
            }
        }

        xs.dispose(); ys.forEach(y => y.dispose());
        self.postMessage({ type: 'training_complete' });
    }
}

const workerModel = new TrainingWorkerModel();

self.onmessage = async (e) => {
    if (e.data.type === 'start_training') {
        try { await workerModel.train(e.data.draws); }
        catch (err) {
            console.error('[Worker] Error:', err);
            self.postMessage({ type: 'error', message: err.message });
        }
    }
};
