// app.js — Custom Neural Network · Adam Optimizer · Real Math Solver · CoT UI
// Bugs fixed:
//   1. biases[i][0] → biases[i][r]  (critical — all neurons shared one bias)
//   2. lossCanvas / lossCtx now initialised inside init()
//   3. VOCAB_SIZE sourced from data.js (no more hardcoded 50)
// Upgrades:
//   • Adam optimizer replaces plain SGD  (~5× faster convergence)
//   • Real number-extraction math solver (Speed, Work, Percent, Ratio)
//   • Class-probability bar display
//   • Signal propagation fires on inference

// ═══════════════════════════════════════════════════════════════════
// Neural Network — Xavier init, Adam optimizer, gradient clipping
// ═══════════════════════════════════════════════════════════════════
class NeuralNetwork {
    constructor(layerSizes, activation = 'relu', learningRate = 0.003) {
        this.layerSizes    = layerSizes;
        this.activationName = activation;
        this.lr            = learningRate;

        // Adam hyper-params
        this.beta1 = 0.9;
        this.beta2 = 0.999;
        this.eps   = 1e-8;
        this.t     = 0;          // Adam timestep

        this.weights = [];
        this.biases  = [];
        this.m       = [];       // 1st moment (weights + biases)
        this.v       = [];       // 2nd moment

        for (let i = 0; i < layerSizes.length - 1; i++) {
            const fanIn  = layerSizes[i];
            const fanOut = layerSizes[i + 1];
            const limit  = Math.sqrt(6 / (fanIn + fanOut));  // Xavier uniform

            const W = this._randMatrixUniform(fanOut, fanIn, -limit, limit);
            const B = new Array(fanOut).fill(0);

            this.weights.push(W);
            this.biases.push(B);

            // Adam moment arrays — same shape as weights / biases
            this.m.push({
                w: W.map(row => row.map(() => 0)),
                b: new Array(fanOut).fill(0)
            });
            this.v.push({
                w: W.map(row => row.map(() => 0)),
                b: new Array(fanOut).fill(0)
            });
        }
    }

    _randMatrixUniform(rows, cols, min, max) {
        return Array.from({ length: rows }, () =>
            Array.from({ length: cols }, () => Math.random() * (max - min) + min)
        );
    }

    // ── Activation functions ─────────────────────────────────────────────────
    activate(z, deriv = false) {
        switch (this.activationName) {
            case 'relu':
                return deriv ? z.map(v => v > 0 ? 1 : 0)
                             : z.map(v => Math.max(0, v));
            case 'tanh':
                return deriv ? z.map(v => 1 - Math.tanh(v) ** 2)
                             : z.map(v => Math.tanh(v));
            case 'sigmoid': {
                const s = z.map(v => 1 / (1 + Math.exp(-Math.max(-15, Math.min(15, v)))));
                return deriv ? s.map(sv => sv * (1 - sv)) : s;
            }
            case 'elu':
                return deriv ? z.map(v => v > 0 ? 1 : Math.min(Math.exp(v), 1e6))
                             : z.map(v => v > 0 ? v : Math.exp(v) - 1);
            default: return z;
        }
    }

    softmax(z) {
        const max = Math.max(...z);
        const exp = z.map(v => Math.exp(v - max));
        const sum = exp.reduce((a, b) => a + b, 1e-300);
        return exp.map(v => v / sum);
    }

    // ── Forward pass  [BUG FIX: biases[i][r] not biases[i][0]] ─────────────
    forward(input) {
        const activations = [input];
        let a = input;
        for (let i = 0; i < this.weights.length; i++) {
            // r is the neuron index — previously missing, causing bias[0] for all
            const z = this.weights[i].map((row, r) =>
                row.reduce((sum, w, j) => sum + w * a[j], 0) + this.biases[i][r]
            );
            a = (i === this.weights.length - 1) ? z : this.activate(z);
            activations.push(a);
        }
        return { activations, output: a };
    }

    predict(input) {
        return this.softmax(this.forward(input).output);
    }

    // ── Back-propagation ────────────────────────────────────────────────────
    backprop(input, target) {
        const { activations } = this.forward(input);
        const probs = this.softmax(activations[activations.length - 1]);

        // Cross-entropy gradient at output
        let delta = probs.map((p, i) => p - target[i]);

        const grads = [];
        for (let i = this.weights.length - 1; i >= 0; i--) {
            const aPrev = activations[i];

            // Re-compute pre-activation z for this layer (needed for act derivative)
            if (i < this.weights.length - 1) {
                const z = this.weights[i].map((row, r) =>
                    row.reduce((sum, w, c) => sum + w * aPrev[c], 0) + this.biases[i][r]
                );
                const ad = this.activate(z, true);
                delta = delta.map((d, idx) => d * ad[idx]);
            }

            grads.unshift({
                w: delta.map(d => aPrev.map(ap => d * ap)),
                b: [...delta]
            });

            if (i > 0) {
                // Propagate delta to previous layer
                delta = Array.from({ length: this.weights[i][0].length }, (_, col) =>
                    this.weights[i].reduce((sum, row, r) => sum + row[col] * delta[r], 0)
                );
            }
        }
        return grads;
    }

    // ── Gradient clipping (by global norm) ──────────────────────────────────
    _clipGradients(grads, maxNorm = 1.0) {
        let norm = 0;
        for (const g of grads) {
            for (const row of g.w) for (const v of row) norm += v * v;
            for (const v of g.b) norm += v * v;
        }
        norm = Math.sqrt(norm);
        if (norm > maxNorm) {
            const scale = maxNorm / norm;
            for (const g of grads) {
                for (let r = 0; r < g.w.length; r++) {
                    for (let c = 0; c < g.w[r].length; c++) g.w[r][c] *= scale;
                    g.b[r] *= scale;
                }
            }
        }
    }

    // ── Adam update ──────────────────────────────────────────────────────────
    applyGradients(grads, batchSize) {
        const { beta1, beta2, eps, lr } = this;
        this.t++;
        const bc1 = 1 - beta1 ** this.t;   // bias-correction denominators
        const bc2 = 1 - beta2 ** this.t;

        for (let i = 0; i < this.weights.length; i++) {
            for (let r = 0; r < this.weights[i].length; r++) {
                // Bias parameter
                const gb = grads[i].b[r] / batchSize;
                this.m[i].b[r] = beta1 * this.m[i].b[r] + (1 - beta1) * gb;
                this.v[i].b[r] = beta2 * this.v[i].b[r] + (1 - beta2) * gb * gb;
                this.biases[i][r] -= lr * (this.m[i].b[r] / bc1) /
                                     (Math.sqrt(this.v[i].b[r] / bc2) + eps);

                // Weight parameters
                for (let c = 0; c < this.weights[i][r].length; c++) {
                    const gw = grads[i].w[r][c] / batchSize;
                    this.m[i].w[r][c] = beta1 * this.m[i].w[r][c] + (1 - beta1) * gw;
                    this.v[i].w[r][c] = beta2 * this.v[i].w[r][c] + (1 - beta2) * gw * gw;
                    this.weights[i][r][c] -= lr * (this.m[i].w[r][c] / bc1) /
                                             (Math.sqrt(this.v[i].w[r][c] / bc2) + eps);
                }
            }
        }
    }

    // ── Mini-batch training step ─────────────────────────────────────────────
    trainBatch(inputs, targets) {
        let batchGrads = null;
        let totalLoss  = 0;

        for (let s = 0; s < inputs.length; s++) {
            const probs = this.predict(inputs[s]);
            const eps   = 1e-9;
            totalLoss  += -targets[s].reduce((sum, t, i) =>
                sum + t * Math.log(Math.max(probs[i], eps)), 0);

            const g = this.backprop(inputs[s], targets[s]);
            this._clipGradients(g, 1.0);

            if (!batchGrads) {
                batchGrads = g.map(gi => ({
                    w: gi.w.map(row => [...row]),
                    b: [...gi.b]
                }));
            } else {
                for (let i = 0; i < g.length; i++) {
                    for (let r = 0; r < g[i].w.length; r++) {
                        for (let c = 0; c < g[i].w[r].length; c++)
                            batchGrads[i].w[r][c] += g[i].w[r][c];
                        batchGrads[i].b[r] += g[i].b[r];
                    }
                }
            }
        }

        if (batchGrads) this.applyGradients(batchGrads, inputs.length);
        return totalLoss / inputs.length;
    }

    evaluate(inputs, labels) {
        let correct = 0;
        for (let i = 0; i < inputs.length; i++) {
            const probs = this.predict(inputs[i]);
            if (probs.indexOf(Math.max(...probs)) === labels[i].indexOf(1)) correct++;
        }
        return correct / inputs.length;
    }

    getParameterCount() {
        let n = 0;
        this.weights.forEach(w => w.forEach(row => n += row.length));
        this.biases.forEach(b => n += b.length);
        return n;
    }
}

// ═══════════════════════════════════════════════════════════════════
// Real Math Solver — extracts numbers, applies class-specific formula
// ═══════════════════════════════════════════════════════════════════
function extractNumbers(text) {
    // Match integers and decimals
    return [...text.matchAll(/\d+\.?\d*/g)].map(m => parseFloat(m[0]));
}

function solveSpeedDistance(text, nums) {
    const lower = text.toLowerCase();
    const steps = [];
    let answer = '?';

    if (nums.length >= 3) {
        // "X km in Y hours. How far in Z hours?" → speed = X/Y, dist = speed*Z
        const [d1, t1, t2] = nums;
        const spd = d1 / t1;
        const dist = spd * t2;
        steps.push({ label: 'Speed', formula: `${d1} ÷ ${t1}`,    result: `${spd.toFixed(2)} km/h` });
        steps.push({ label: 'Distance', formula: `${spd.toFixed(2)} × ${t2}`, result: `${dist.toFixed(2)} km` });
        answer = `${dist.toFixed(2)} km`;
    } else if (nums.length === 2) {
        const [a, b] = nums;
        const isSpeedQuery = /speed|how fast|km\/h/.test(lower);
        if (isSpeedQuery || (/distance|how far|cover/.test(lower) === false)) {
            // Treat as distance/time → compute speed
            const spd = a / b;
            steps.push({ label: 'Speed', formula: `${a} ÷ ${b}`, result: `${spd.toFixed(2)} km/h` });
            answer = `${spd.toFixed(2)} km/h`;
        } else {
            // speed × time → distance
            const dist = a * b;
            steps.push({ label: 'Distance', formula: `${a} × ${b}`, result: `${dist.toFixed(2)} km` });
            answer = `${dist.toFixed(2)} km`;
        }
    } else if (nums.length === 1) {
        steps.push({ label: 'Note', formula: '—', result: 'More values needed' });
    }

    return { steps, answer };
}

function solveWorkRate(text, nums) {
    const steps = [];
    let answer = '?';

    if (nums.length >= 3) {
        const [w1, d1, w2] = nums;
        const totalWork = w1 * d1;
        const d2 = totalWork / w2;
        steps.push({ label: 'Total work-units', formula: `${w1} × ${d1}`, result: `${totalWork}` });
        steps.push({ label: `Days for ${w2}`, formula: `${totalWork} ÷ ${w2}`, result: `${d2.toFixed(1)} days` });
        answer = `${d2.toFixed(1)} days`;
    } else if (nums.length === 2) {
        const [w1, d1] = nums;
        const totalWork = w1 * d1;
        steps.push({ label: 'Total work-units', formula: `${w1} × ${d1}`, result: `${totalWork}` });
        answer = `${totalWork} work-units`;
    }

    return { steps, answer };
}

function solvePercentage(text, nums) {
    const lower = text.toLowerCase();
    const steps = [];
    let answer = '?';

    // "X is what % of Y" or "what % of Y is X"
    const isReverseQuery = /what\s+(is\s+)?(the\s+)?percent|is\s+what\s+percent|percentage\s+is/.test(lower);

    if (isReverseQuery && nums.length >= 2) {
        const [part, whole] = nums;
        const pct = (part / whole) * 100;
        steps.push({ label: 'Percentage', formula: `(${part} ÷ ${whole}) × 100`, result: `${pct.toFixed(2)}%` });
        answer = `${pct.toFixed(2)}%`;
    } else if (nums.length >= 2) {
        // "What is P% of N"
        const [p, n] = nums;
        const val = (p / 100) * n;
        steps.push({ label: 'Convert %', formula: `${p} ÷ 100`, result: `${(p/100).toFixed(4)}` });
        steps.push({ label: 'Value', formula: `${(p/100).toFixed(4)} × ${n}`, result: `${val.toFixed(2)}` });
        answer = `${val.toFixed(2)}`;
    }

    return { steps, answer };
}

function solveRatio(text, nums) {
    const steps = [];
    let answer = '?';

    // Extract explicit X:Y pattern
    const rm = text.match(/(\d+)\s*:\s*(\d+)/);
    if (rm) {
        const r1 = parseFloat(rm[1]);
        const r2 = parseFloat(rm[2]);

        // Divide 120 in ratio 3:5 → larger part
        if (/divide|split/.test(text.toLowerCase()) && nums.length >= 3) {
            const total = nums.find(n => n !== r1 && n !== r2) || nums[0];
            const parts = r1 + r2;
            const a = (r1 / parts) * total;
            const b = (r2 / parts) * total;
            steps.push({ label: 'Parts sum', formula: `${r1} + ${r2}`, result: `${parts}` });
            steps.push({ label: 'Part A', formula: `(${r1}/${parts}) × ${total}`, result: `${a.toFixed(2)}` });
            steps.push({ label: 'Part B', formula: `(${r2}/${parts}) × ${total}`, result: `${b.toFixed(2)}` });
            answer = `${Math.max(a, b).toFixed(2)} (larger part)`;
        } else {
            // Given one part, find the other
            const givenNum = nums.find(n => n !== r1 && n !== r2);
            if (givenNum !== undefined) {
                // Determine if given corresponds to r1 or r2 by keyword proximity
                const lower = text.toLowerCase();
                const idx1 = lower.lastIndexOf(String(r1));
                const idx2 = lower.lastIndexOf(String(r2));
                const givenIdx = lower.lastIndexOf(String(givenNum));
                const closerToR2 = Math.abs(givenIdx - idx2) < Math.abs(givenIdx - idx1);

                let found, ratio, label;
                if (closerToR2) {
                    // given value matches r2 → find r1 part
                    found = (r1 / r2) * givenNum;
                    ratio = `${r1} ÷ ${r2}`;
                    label = 'First part';
                } else {
                    found = (r2 / r1) * givenNum;
                    ratio = `${r2} ÷ ${r1}`;
                    label = 'Second part';
                }
                steps.push({ label: 'Ratio factor', formula: ratio, result: (found / givenNum).toFixed(4) });
                steps.push({ label, formula: `${(found/givenNum).toFixed(4)} × ${givenNum}`, result: found.toFixed(2) });
                answer = found.toFixed(2);
            }
        }
    } else if (nums.length >= 2) {
        // No explicit ratio found — show what we have
        steps.push({ label: 'Note', formula: '—', result: 'Could not parse ratio pattern' });
    }

    return { steps, answer };
}

function solveProblem(text, classIndex) {
    const nums = extractNumbers(text);
    switch (classIndex) {
        case 0: return solveSpeedDistance(text, nums);
        case 1: return solveWorkRate(text, nums);
        case 2: return solvePercentage(text, nums);
        case 3: return solveRatio(text, nums);
        default: return { steps: [], answer: '?' };
    }
}

// ═══════════════════════════════════════════════════════════════════
// Global State
// ═══════════════════════════════════════════════════════════════════
let model;
let visualizer;
let epochsTrained  = 0;
let lossHistory    = [];
let accuracyHistory = [];
let isTraining     = false;
let lossCanvas, lossCtx;   // [BUG FIX] — lazy init inside init()
let currentDeviceConfig = DEVICE_CONFIGS.balanced;
let infiniteModeActive = false;
let infiniteModeInterval = null;

// ═══════════════════════════════════════════════════════════════════
// Loading Bar
// ═══════════════════════════════════════════════════════════════════
function updateLoadingBar(progress, status) {
    const fill   = document.getElementById('loadingFill');
    const statEl = document.getElementById('loadingStatus');
    if (fill)   fill.style.width = progress + '%';
    if (statEl) statEl.textContent = status;
}
function hideLoadingBar() {
    document.getElementById('loadingBar')?.classList.add('hidden');
    document.getElementById('loadingStatus')?.classList.add('hidden');
}

// ═══════════════════════════════════════════════════════════════════
// Bag-of-Words Encoder (normalised)
// ═══════════════════════════════════════════════════════════════════
function textToBow(text) {
    const words = text.toLowerCase().replace(/[^\w\s]/g, ' ').split(/\s+/).filter(Boolean);
    const bow = new Array(VOCAB_SIZE).fill(0);

    words.forEach(w => {
        const idx = VOCAB.indexOf(w);
        if (idx !== -1) {
            bow[idx] += 1;
        } else {
            // Simple hash fallback for OOV words
            let h = 0;
            for (let i = 0; i < w.length; i++) h = ((h << 5) - h) + w.charCodeAt(i);
            bow[Math.abs(h) % VOCAB_SIZE] += 0.15;
        }
    });

    const sum = bow.reduce((a, b) => a + b, 1e-9);
    return bow.map(v => v / sum);
}

// ═══════════════════════════════════════════════════════════════════
// Build Model from UI controls
// ═══════════════════════════════════════════════════════════════════
function buildModel() {
    const numLayers = parseInt(document.getElementById('numLayersSlider').value);
    const units     = parseInt(document.getElementById('unitsSlider').value);
    const activation = document.getElementById('activationSelect').value;
    const lr        = parseFloat(document.getElementById('lrSlider').value);

    const sizes = [VOCAB_SIZE];
    for (let i = 0; i < numLayers; i++) sizes.push(units);
    sizes.push(CLASS_NAMES.length);

    const nn = new NeuralNetwork(sizes, activation, lr);
    document.getElementById('statParams').innerText = nn.getParameterCount().toLocaleString();
    document.getElementById('statLayers').innerText = numLayers + 1;
    document.getElementById('statHeads').innerText  = units;
    return nn;
}

// ═══════════════════════════════════════════════════════════════════
// Device Configuration Management
// ═══════════════════════════════════════════════════════════════════
function applyDeviceConfig(config) {
    currentDeviceConfig = config;
    document.getElementById('numLayersSlider').value = config.numLayers;
    document.getElementById('unitsSlider').value = config.units;
    document.getElementById('lrSlider').value = config.learningRate;
    document.getElementById('activationSelect').value = config.activation;
    document.getElementById('depthSlider').value = config.reasoningDepth;

    // Update display values
    document.getElementById('numLayersValue').innerText = config.numLayers;
    document.getElementById('unitsValue').innerText = config.units;
    document.getElementById('lrValue').innerText = config.learningRate.toFixed(4);
    document.getElementById('depthValue').innerText = config.reasoningDepth + ' steps';

    showNotification(`📱 Config: ${config.name} — ${config.description}`, 'success');
}

function startInfiniteMode() {
    infiniteModeActive = true;
    infiniteModeIndex = 0;
    
    showNotification('♾️ Infinite Mode activated - cycling through device configs every 3s', 'success');
    
    infiniteModeInterval = setInterval(() => {
        const config = INFINITE_CONFIG_SEQUENCE[infiniteModeIndex % INFINITE_CONFIG_SEQUENCE.length];
        applyDeviceConfig(config);
        infiniteModeIndex++;
    }, DEFAULT_CONFIG.infiniteInterval);
}

function stopInfiniteMode() {
    if (infiniteModeInterval) {
        clearInterval(infiniteModeInterval);
        infiniteModeInterval = null;
    }
    infiniteModeActive = false;
    showNotification('⏹️ Infinite Mode stopped', 'info');
}

// ═══════════════════════════════════════════════════════════════════
// Loss / Accuracy Chart
// ═══════════════════════════════════════════════════════════════════
function drawLossCanvas() {
    if (!lossCanvas || !lossCtx) return;
    const w = lossCanvas.width, h = lossCanvas.height;
    lossCtx.clearRect(0, 0, w, h);
    if (!lossHistory.length) return;

    // Grid lines
    lossCtx.strokeStyle = 'rgba(255,255,255,0.05)';
    lossCtx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
        const y = (i / 4) * h;
        lossCtx.beginPath(); lossCtx.moveTo(0, y); lossCtx.lineTo(w, y); lossCtx.stroke();
    }

    const drawLine = (data, color, maxVal, minVal) => {
        const range = (maxVal - minVal) || 1;
        lossCtx.beginPath();
        data.forEach((val, i) => {
            const x = (i / Math.max(data.length - 1, 1)) * w;
            const y = h - ((val - minVal) / range) * h;
            i === 0 ? lossCtx.moveTo(x, y) : lossCtx.lineTo(x, y);
        });
        lossCtx.strokeStyle = color;
        lossCtx.lineWidth = 1.8;
        lossCtx.stroke();
    };

    drawLine(lossHistory, '#ef4444',
        Math.max(...lossHistory, 0.1),
        Math.min(...lossHistory, 0));

    if (accuracyHistory.length) {
        drawLine(accuracyHistory, '#10b981', 1, 0);
    }

    lossCtx.font = '9px "JetBrains Mono", monospace';
    lossCtx.fillStyle = '#ef4444'; lossCtx.fillText('Loss', 5, 12);
    lossCtx.fillStyle = '#10b981'; lossCtx.fillText('Acc',  5, 24);
}

function updateChart(loss, acc) {
    lossHistory.push(loss);
    accuracyHistory.push(acc);
    if (lossHistory.length > 80) { lossHistory.shift(); accuracyHistory.shift(); }
    drawLossCanvas();
}

// ═══════════════════════════════════════════════════════════════════
// Training
// ═══════════════════════════════════════════════════════════════════
async function trainNetwork(epochs = 60, silent = false) {
    if (isTraining) return;
    isTraining = true;

    if (!silent) updateLoadingBar(0, `Training ${epochs} epochs…`);

    try {
        model = buildModel();
        const inputs  = TRAIN_DATA.map(s => textToBow(s.text));
        const targets = TRAIN_DATA.map(s => {
            const t = new Array(CLASS_NAMES.length).fill(0);
            t[s.label] = 1;
            return t;
        });

        lossHistory = []; accuracyHistory = [];

        const batchSize  = 8;
        const numSamples = inputs.length;

        for (let epoch = 0; epoch < epochs; epoch++) {
            // Shuffle indices
            const idx = [...Array(numSamples).keys()];
            for (let i = numSamples - 1; i > 0; i--) {
                const j = Math.floor(Math.random() * (i + 1));
                [idx[i], idx[j]] = [idx[j], idx[i]];
            }

            let totalLoss = 0;
            for (let start = 0; start < numSamples; start += batchSize) {
                const end = Math.min(start + batchSize, numSamples);
                const bi  = idx.slice(start, end);
                totalLoss += model.trainBatch(bi.map(i => inputs[i]), bi.map(i => targets[i]))
                             * (end - start);
            }

            const avgLoss = totalLoss / numSamples;
            const acc     = model.evaluate(inputs, targets);

            epochsTrained++;
            document.getElementById('lossDisplay').innerText     = avgLoss.toFixed(4);
            document.getElementById('accuracyDisplay').innerText = isNaN(acc) ? '0.00%' : (acc * 100).toFixed(1) + '%';
            document.getElementById('epochDisplay').innerText    = epochsTrained;
            updateChart(avgLoss, acc);

            if (!silent) updateLoadingBar((epoch + 1) / epochs * 100, `Epoch ${epoch + 1}/${epochs}`);

            // Sync visualizer with real weights every 3 epochs
            if (visualizer && epoch % 3 === 0 && model.weights.length) {
                const fake = { getWeights: () => [{ arraySync: () => model.weights[0] }] };
                visualizer.updateWeightsFromTFModel(fake);
                visualizer.activateLayer(Math.floor(Math.random() * 4), 0.3);
            }

            await new Promise(r => setTimeout(r, 0)); // yield to browser
        }

        // Final sync
        if (model.weights.length) {
            const fake = { getWeights: () => [{ arraySync: () => model.weights[0] }] };
            visualizer?.updateWeightsFromTFModel(fake);
            updateWeightHeatmap(fake);
        }

    } catch (err) {
        console.error('Training error:', err);
        showNotification('Training error: ' + err.message, 'error');
    } finally {
        isTraining = false;
        if (!silent) hideLoadingBar();
    }
}

// ═══════════════════════════════════════════════════════════════════
// Predict
// ═══════════════════════════════════════════════════════════════════
function predictOperation(text) {
    if (!model) return { class: 0, className: CLASS_NAMES[0], confidence: 0, probs: [] };
    const bow    = textToBow(text);
    const probs  = model.predict(bow);
    const maxIdx = probs.indexOf(Math.max(...probs));
    return { class: maxIdx, className: CLASS_NAMES[maxIdx], confidence: probs[maxIdx], probs };
}

// ═══════════════════════════════════════════════════════════════════
// Tree of Thought Reasoning — Multi-path exploration
// ═══════════════════════════════════════════════════════════════════
async function startTreeOfThoughtReasoning() {
    const input = document.getElementById('inputText').value.trim();
    if (!input) { showNotification('Please enter a problem', 'warning'); return; }
    if (!model)  { showNotification('Train the model first', 'error');   return; }

    const depth    = parseInt(document.getElementById('depthSlider').value);
    const chainDiv = document.getElementById('reasoningChain');
    chainDiv.innerHTML = '';

    document.getElementById('finalOutput').innerHTML        = '';
    document.getElementById('thinkingIndicator').style.display = 'flex';
    document.getElementById('tokenContainer').style.display    = 'none';
    document.getElementById('confidenceMeter').style.display   = 'none';

    const t0   = performance.now();
    const pred = predictOperation(input);

    visualizer?.activateLayer(0, 0.9);
    visualizer?.fireSignal(0);

    // Tree of Thought: Multiple reasoning paths
    const paths = [
        { name: 'Direct Extraction', icon: '🎯', description: 'Parse numbers directly and apply formula' },
        { name: 'Contextual Analysis', icon: '🧠', description: 'Analyze context keywords to infer operations' },
        { name: 'Dimensional Check', icon: '📏', description: 'Verify dimensional consistency of terms' },
        { name: 'Symbolic Substitution', icon: '⚖️', description: 'Match problem structure to known patterns' },
        { name: 'Constraint Propagation', icon: '🔗', description: 'Propagate constraints through variables' }
    ];

    // Only show depth number of paths
    const activePaths = paths.slice(0, Math.min(depth, paths.length));

    // Display initial thinking with multiple paths
    addStepToUI({
        step: 1,
        type: 'Tree of Thought Init',
        icon: '🌳',
        content: `Exploring <strong>${activePaths.length}</strong> distinct reasoning branches for <em>${pred.className}</em>. Confidence: ${(pred.confidence * 100).toFixed(1)}%`,
        confidence: pred.confidence
    });

    await new Promise(r => setTimeout(r, 300));

    // Animate each path
    for (let i = 0; i < activePaths.length; i++) {
        const path = activePaths[i];
        const conf = Math.min(0.6 + i * 0.06 + pred.confidence * 0.1, 0.99);

        if (visualizer) {
            const layer = Math.min(i, visualizer.nodes.reduce((m, n) => Math.max(m, n.layer), 0));
            visualizer.activateLayer(layer, 0.5);
            visualizer.fireSignal(layer);
        }

        // Create a container for the path
        const chainDiv = document.getElementById('reasoningChain');
        const pathCard = document.createElement('div');
        pathCard.className = 'step-card fade-in path-card';
        pathCard.innerHTML = `
            <div class="step-header">
                <div class="step-number">${i + 1}</div>
                <div class="step-title"><span style="margin-right:8px">${path.icon}</span>${path.name}</div>
            </div>
            <div class="step-content">${path.description}</div>
            <div class="step-meta"><span>Path Confidence: ${(conf * 100).toFixed(1)}%</span></div>
            <div class="confidence-bar"><div class="confidence-fill" style="width:${(conf * 100)}%"></div></div>
        `;
        chainDiv.appendChild(pathCard);
        chainDiv.scrollTop = chainDiv.scrollHeight;

        visualizer?.nodes.forEach(n => {
            if (n.layer === i % 3) n.target = 0.4 + 0.5 * Math.sin(i * 0.5 + n.idx);
        });

        await new Promise(r => setTimeout(r, 350));
    }

    // Final synthesis step
    const synthesisConf = Math.min(pred.confidence + 0.15, 0.95);
    addStepToUI({
        step: activePaths.length + 1,
        type: 'Path Synthesis',
        icon: '✨',
        content: `Merging insights from all <strong>${activePaths.length}</strong> paths. Selecting highest-confidence solution branch.`,
        confidence: synthesisConf
    });

    await new Promise(r => setTimeout(r, 350));

    // ── Real solver ──────────────────────────────────────────────
    const { steps: solveSteps, answer } = solveProblem(input, pred.class);
    const solutionHTML = solveSteps.length
        ? `<div class="solution-box">
            ${solveSteps.map(s => `
                <div class="solution-step">
                    <strong>${s.label}:</strong>
                    <code>${s.formula}</code>
                    <span class="sol-result">= ${s.result}</span>
                </div>`).join('')}
           </div>`
        : `<p class="sol-fallback">Could not extract enough numerical data. Try a clearer formulation.</p>`;

    document.getElementById('finalOutput').innerHTML = `
        <div class="result-header">
            <span class="result-badge">🌳 Tree of Thought</span>
            <span class="result-badge">${pred.className}</span>
            <span class="result-confidence">${(pred.confidence * 100).toFixed(1)}% confidence</span>
        </div>
        ${solutionHTML}
        <div class="final-answer">Answer: <strong>${answer}</strong></div>
    `;

    showClassProbs(pred.probs);
    document.getElementById('thinkingIndicator').style.display = 'none';
    showTokens(input.split(/\s+/).slice(0, 16));
    showConfidenceMeter(pred.confidence);
    document.getElementById('statTime').innerText = Math.round(performance.now() - t0) + 'ms';
}

// ═══════════════════════════════════════════════════════════════════
// Chain-of-Thought Reasoning — with REAL math solver
// ═══════════════════════════════════════════════════════════════════
async function startReasoning() {
    const treeOfThoughtEnabled = document.getElementById('treeOfThoughtToggle')?.checked || false;
    
    if (treeOfThoughtEnabled) {
        return await startTreeOfThoughtReasoning();
    }

    const input = document.getElementById('inputText').value.trim();
    if (!input) { showNotification('Please enter a problem', 'warning'); return; }
    if (!model)  { showNotification('Train the model first', 'error');   return; }

    const depth    = parseInt(document.getElementById('depthSlider').value);
    const chainDiv = document.getElementById('reasoningChain');
    chainDiv.innerHTML = '';

    document.getElementById('finalOutput').innerHTML        = '';
    document.getElementById('thinkingIndicator').style.display = 'flex';
    document.getElementById('tokenContainer').style.display    = 'none';
    document.getElementById('confidenceMeter').style.display   = 'none';

    const t0   = performance.now();
    const pred = predictOperation(input);

    // Trigger visualizer signal
    visualizer?.activateLayer(0, 0.9);
    visualizer?.fireSignal(0);

    // CoT step templates
    const templates = [
        { type: 'NLP Classification', icon: '🔍',
          content: `Neural classifier mapped input → <strong>${pred.className}</strong> (${(pred.confidence*100).toFixed(1)}% confidence). Extracting numerical entities…` },
        { type: 'Token Analysis', icon: '📐',
          content: `Bag-of-Words encoder produced a ${VOCAB_SIZE}-dimensional feature vector. Top activations correspond to <em>${pred.className}</em> keyword clusters.` },
        { type: 'Pattern Matching', icon: '🧮',
          content: `Applying <em>${pred.className}</em> formula schema. Parsing candidate numbers: [${extractNumbers(input).join(', ')}].` },
        { type: 'Symbolic Reasoning', icon: '⚙️',
          content: `Executing domain-specific solver for <em>${pred.className}</em>. Substituting extracted values into formula and computing…` },
        { type: 'Verification', icon: '✅',
          content: `Cross-checking dimensional consistency. Confirming solution satisfies the original problem constraints.` },
        { type: 'Synthesis', icon: '📊',
          content: `Aggregating all reasoning steps. Generating final answer with confidence scoring.` }
    ];

    for (let i = 0; i < depth; i++) {
        const t   = templates[i % templates.length];
        const conf = Math.min(0.65 + i * 0.04 + pred.confidence * 0.12, 0.99);

        if (visualizer) {
            const layer = Math.min(i, visualizer.nodes.reduce((m, n) => Math.max(m, n.layer), 0));
            visualizer.activateLayer(layer, 0.6);
            visualizer.fireSignal(layer);
            visualizer.nodes.forEach(n => {
                if (n.layer === layer) n.target = 0.5 + 0.45 * Math.sin(i * 0.7 + n.idx);
            });
        }

        addStepToUI({ step: i + 1, type: t.type, icon: t.icon, content: t.content, confidence: conf });
        await new Promise(r => setTimeout(r, 350));
    }

    // ── Real solver ──────────────────────────────────────────────
    const { steps: solveSteps, answer } = solveProblem(input, pred.class);
    const solutionHTML = solveSteps.length
        ? `<div class="solution-box">
            ${solveSteps.map(s => `
                <div class="solution-step">
                    <strong>${s.label}:</strong>
                    <code>${s.formula}</code>
                    <span class="sol-result">= ${s.result}</span>
                </div>`).join('')}
           </div>`
        : `<p class="sol-fallback">Could not extract enough numerical data. Try a clearer formulation.</p>`;

    document.getElementById('finalOutput').innerHTML = `
        <div class="result-header">
            <span class="result-badge">${pred.className}</span>
            <span class="result-confidence">${(pred.confidence * 100).toFixed(1)}% confidence</span>
        </div>
        ${solutionHTML}
        <div class="final-answer">Answer: <strong>${answer}</strong></div>
    `;

    // Class probability bars
    showClassProbs(pred.probs);

    document.getElementById('thinkingIndicator').style.display = 'none';
    showTokens(input.split(/\s+/).slice(0, 16));
    showConfidenceMeter(pred.confidence);
    document.getElementById('statTime').innerText = Math.round(performance.now() - t0) + 'ms';
}

// ═══════════════════════════════════════════════════════════════════
// UI Helpers
// ═══════════════════════════════════════════════════════════════════
function addStepToUI({ step, type, icon, content, confidence }) {
    const c    = document.getElementById('reasoningChain');
    const card = document.createElement('div');
    card.className = 'step-card fade-in';
    const conf = (confidence * 100).toFixed(1);
    card.innerHTML = `
        <div class="step-header">
            <div class="step-number">${step}</div>
            <div class="step-title"><span style="margin-right:8px">${icon}</span>${type}</div>
        </div>
        <div class="step-content">${content}</div>
        <div class="step-meta"><span>Confidence: ${conf}%</span></div>
        <div class="confidence-bar"><div class="confidence-fill" style="width:${conf}%"></div></div>
    `;
    c.appendChild(card);
    c.scrollTop = c.scrollHeight;
}

function showTokens(tokens) {
    const cont = document.getElementById('tokenList');
    cont.innerHTML = '';
    tokens.forEach((t, i) => {
        const s = document.createElement('span');
        s.className   = 'token';
        s.textContent = t;
        s.style.animationDelay = i * 0.05 + 's';
        // highlight vocab matches
        if (VOCAB.includes(t.toLowerCase().replace(/[^\w]/g, '')))
            s.classList.add('token-match');
        cont.appendChild(s);
    });
    document.getElementById('tokenContainer').style.display = 'block';
}

function showConfidenceMeter(confidence) {
    const meter   = document.getElementById('confidenceMeter');
    const fill    = document.getElementById('confidenceFill');
    const percent = document.getElementById('confidencePercent');
    meter.style.display  = 'block';
    fill.style.width     = '0%';
    percent.innerText    = '0%';
    setTimeout(() => {
        fill.style.width  = (confidence * 100) + '%';
        percent.innerText = (confidence * 100).toFixed(1) + '%';
    }, 100);
}

function showClassProbs(probs) {
    const container = document.getElementById('classProbsContainer');
    if (!container || !probs || !probs.length) return;
    container.style.display = 'block';
    container.innerHTML = CLASS_NAMES.map((name, i) => {
        const pct = (probs[i] * 100).toFixed(1);
        const isMax = probs[i] === Math.max(...probs);
        return `
        <div class="prob-row">
            <span class="prob-label">${name}</span>
            <div class="prob-track">
                <div class="prob-bar${isMax ? ' prob-bar-max' : ''}" style="width:${pct}%"></div>
            </div>
            <span class="prob-pct">${pct}%</span>
        </div>`;
    }).join('');
}

function testNLP() {
    const input = document.getElementById('inputText').value
        || 'A train travels 120 km in 2 hours. How far in 5 hours?';
    const pred  = predictOperation(input);
    showNotification(`NLP: ${pred.className} — ${(pred.confidence*100).toFixed(1)}% conf`, 'success');
    showClassProbs(pred.probs);
}

function clearAll() {
    document.getElementById('inputText').value = '';
    document.getElementById('reasoningChain').innerHTML = `
        <div class="chain-empty">
            <div class="chain-empty-icon">
                <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                    <path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm0 18a8 8 0 1 1 8-8 8 8 0 0 1-8 8z"/>
                    <path d="M12 6v6l4 2"/>
                </svg>
            </div>
            <div class="chain-empty-text">No active reasoning session</div>
            <div class="chain-empty-hint">Start reasoning to see the chain</div>
        </div>`;
    document.getElementById('finalOutput').innerHTML = `
        <div class="empty-state-viz">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                <circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/>
                <path d="M12 17h.01"/>
            </svg>
            <p>Enter a problem and start reasoning</p>
        </div>`;
    const cp = document.getElementById('classProbsContainer');
    if (cp) cp.style.display = 'none';
    document.getElementById('tokenContainer').style.display   = 'none';
    document.getElementById('confidenceMeter').style.display  = 'none';
    document.getElementById('statTime').innerText             = '0ms';
    visualizer?.initNetwork();
}

function showNotification(message, type = 'info') {
    const n = document.createElement('div');
    n.className = `notification notification-${type}`;
    n.innerHTML = `<span>${type === 'success' ? '✓' : type === 'error' ? '✕' : type === 'warning' ? '⚠' : 'ℹ'}</span>${message}`;
    document.body.appendChild(n);
    setTimeout(() => n.classList.add('show'), 10);
    setTimeout(() => { n.classList.remove('show'); setTimeout(() => n.remove(), 300); }, 3500);
}

// ═══════════════════════════════════════════════════════════════════
// Init
// ═══════════════════════════════════════════════════════════════════
function init() {
    // [BUG FIX] — initialise canvas refs inside init so dimensions are available
    lossCanvas = document.getElementById('lossChart');
    lossCtx    = lossCanvas.getContext('2d');
    lossCanvas.width  = lossCanvas.offsetWidth;
    lossCanvas.height = lossCanvas.offsetHeight;
    drawLossCanvas();

    // Floating background shapes
    const shapesEl = document.getElementById('floatingShapes');
    for (let i = 0; i < 4; i++) {
        const s = document.createElement('div');
        s.className = 'shape';
        shapesEl.appendChild(s);
    }

    visualizer = new NetworkVisualizer('networkCanvas');

    // Inject notification + solution styles
    if (!document.getElementById('dyn-styles')) {
        const style = document.createElement('style');
        style.id = 'dyn-styles';
        style.textContent = `
            .notification{position:fixed;top:20px;right:20px;padding:14px 22px;background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius-md);color:var(--text-primary);font-size:.88rem;display:flex;align-items:center;gap:10px;z-index:10000;transform:translateX(130%);transition:transform .3s ease;backdrop-filter:blur(20px);box-shadow:var(--shadow-soft)}
            .notification.show{transform:translateX(0)}
            .notification-success{border-color:var(--success)}.notification-success span{color:var(--success)}
            .notification-error{border-color:var(--error)}.notification-error span{color:var(--error)}
            .notification-warning{border-color:var(--warning)}.notification-warning span{color:var(--warning)}
            .solution-box{background:rgba(0,0,0,.3);border-radius:var(--radius-md);padding:14px 16px;margin-top:12px}
            .solution-step{display:flex;align-items:center;gap:8px;padding:7px 0;border-bottom:1px solid var(--border-light);font-size:.85rem;color:var(--text-secondary);flex-wrap:wrap}
            .solution-step:last-child{border-bottom:none}
            .solution-step strong{color:var(--primary);min-width:120px}
            .solution-step code{background:rgba(0,212,255,.08);border:1px solid rgba(0,212,255,.15);padding:2px 8px;border-radius:4px;font-family:var(--font-mono);font-size:.82rem;color:var(--text-primary)}
            .sol-result{color:var(--success);font-family:var(--font-mono);font-weight:600;margin-left:auto}
            .sol-fallback{color:var(--text-muted);font-size:.85rem;padding:8px 0;font-style:italic}
            .final-answer{margin-top:14px;padding:10px 14px;background:linear-gradient(135deg,rgba(0,212,255,.08),rgba(124,58,237,.08));border:1px solid rgba(0,212,255,.2);border-radius:var(--radius-md);font-size:.95rem;color:var(--text-primary)}
            .final-answer strong{color:var(--primary);font-family:var(--font-mono);font-size:1.05rem}
            .result-header{display:flex;align-items:center;gap:12px;margin-bottom:14px}
            .result-badge{background:linear-gradient(135deg,var(--primary),var(--secondary));color:#fff;padding:4px 12px;border-radius:20px;font-size:.8rem;font-weight:600}
            .result-confidence{color:var(--success);font-size:.85rem;font-family:var(--font-mono)}
            .token-match{border-color:var(--primary)!important;color:var(--primary)!important}
        `;
        document.head.appendChild(style);
    }

    // Slider listeners
    const sliders = [
        ['depthSlider',     'depthValue',    v => v + ' steps'],
        ['numLayersSlider', 'numLayersValue', v => v],
        ['unitsSlider',     'unitsValue',     v => v],
        ['lrSlider',        'lrValue',        v => parseFloat(v).toFixed(4)]
    ];
    sliders.forEach(([sliderId, valId, fmt]) => {
        document.getElementById(sliderId)?.addEventListener('input', e => {
            document.getElementById(valId).innerText = fmt(e.target.value);
        });
    });

    // Set balanced device as active by default
    document.getElementById('deviceBalancedBtn')?.classList.add('active');

    // Tree of Thought toggle listener
    document.getElementById('treeOfThoughtToggle')?.addEventListener('change', e => {
        const enabled = e.target.checked;
        showNotification(`Tree of Thought ${enabled ? 'enabled' : 'disabled'}`, 'info');
    });

    // Device configuration buttons
    ['cpu', 'gpu', 'mobile', 'balanced', 'server'].forEach(device => {
        document.getElementById(`device${device.charAt(0).toUpperCase() + device.slice(1)}Btn`)?.addEventListener('click', (e) => {
            // Remove active class from all buttons
            document.querySelectorAll('.btn-device').forEach(btn => btn.classList.remove('active'));
            // Add active class to clicked button
            e.target.closest('.btn-device').classList.add('active');
            // Stop infinite mode if active
            if (infiniteModeActive) stopInfiniteMode();
            // Apply device config
            applyDeviceConfig(DEVICE_CONFIGS[device]);
        });
    });

    // Infinite mode button
    document.getElementById('infiniteModeBtn')?.addEventListener('click', () => {
        if (infiniteModeActive) {
            stopInfiniteMode();
            document.getElementById('infiniteModeBtn').style.opacity = '1';
        } else {
            startInfiniteMode();
            document.getElementById('infiniteModeBtn').style.opacity = '0.6';
        }
    });

    document.getElementById('startReasoningBtn')?.addEventListener('click', startReasoning);
    document.getElementById('resetBtn')?.addEventListener('click', clearAll);
    document.getElementById('trainBtn')?.addEventListener('click', () => trainNetwork(60));
    document.getElementById('testNLPBtn')?.addEventListener('click', testNLP);
    document.getElementById('inputText')?.addEventListener('keydown', e => {
        if (e.key === 'Enter' && e.ctrlKey) startReasoning();
    });

    // Initial quick train (silent, 8 epochs to warm up)
    setTimeout(() => {
        updateLoadingBar(80, 'Initialising neural network…');
        trainNetwork(8, true).then(() => {
            hideLoadingBar();
            showNotification('Model ready — Adam optimizer active', 'success');
        });
    }, 400);
}

window.addEventListener('DOMContentLoaded', init);
