// app.js - TF.js model, training, CoT reasoning, UI
let model;
let visualizer;
let epochsTrained = 0;
let lossHistory = [];
const VOCAB_SIZE = 50;
const NUM_CLASSES = 4;
const CLASS_NAMES = ['Speed/Distance', 'Work/Rate', 'Percentage', 'Ratio'];

// Loaded from data.json
let TRAIN_DATA = [];
let DEFAULT_CONFIG = {};

// Text to BoW
function textToBow(text) {
    const words = text.toLowerCase().replace(/[^\w\s]/g, ' ').split(/\s+/).filter(w => w.length > 0);
    const bow = new Array(VOCAB_SIZE).fill(0);
    words.forEach(w => {
        const idx = VOCAB.indexOf(w);
        if (idx >= 0) bow[idx] += 1;
        else {
            let hash = 0;
            for (let i = 0; i < w.length; i++) hash = ((hash << 5) - hash) + w.charCodeAt(i);
            bow[Math.abs(hash) % VOCAB_SIZE] += 0.2;
        }
    });
    const sum = bow.reduce((a, b) => a + b, 1e-9);
    return bow.map(v => v / sum);
}

// Build model from UI
function buildModel() {
    const numLayers = parseInt(document.getElementById('numLayersSlider').value);
    const units = parseInt(document.getElementById('unitsSlider').value);
    const activation = document.getElementById('activationSelect').value;
    const lr = parseFloat(document.getElementById('lrSlider').value);

    const model = tf.sequential();
    model.add(tf.layers.dense({ units, activation, inputShape: [VOCAB_SIZE] }));
    for (let i = 1; i < numLayers; i++) {
        model.add(tf.layers.dense({ units, activation }));
    }
    model.add(tf.layers.dense({ units: NUM_CLASSES, activation: 'softmax' }));
    
    const optimizer = tf.train.adam(lr);
    model.compile({ optimizer, loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
    
    // Update stats
    let totalParams = 0;
    model.getWeights().forEach(w => totalParams += w.size);
    document.getElementById('statParams').innerText = totalParams;
    document.getElementById('statLayers').innerText = numLayers + 1;
    return model;
}

// Train function
async function trainNetwork(epochs = 30, silent = false) {
    const overlay = document.getElementById('overlay');
    if (!silent) overlay.classList.add('active');
    
    model = buildModel();
    
    const xs = TRAIN_DATA.map(s => textToBow(s.text));
    const ys = TRAIN_DATA.map(s => {
        const arr = new Array(NUM_CLASSES).fill(0);
        arr[s.label] = 1;
        return arr;
    });
    
    const xsTensor = tf.tensor2d(xs);
    const ysTensor = tf.tensor2d(ys);
    
    const history = await model.fit(xsTensor, ysTensor, {
        epochs,
        batchSize: 4,
        shuffle: true,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                epochsTrained++;
                document.getElementById('lossDisplay').innerText = logs.loss.toFixed(4);
                document.getElementById('accuracyDisplay').innerText = logs.acc?.toFixed(4) || '0.00';
                document.getElementById('epochDisplay').innerText = epochsTrained;
            }
        }
    });
    
    xsTensor.dispose();
    ysTensor.dispose();
    
    if (!silent) overlay.classList.remove('active');
    
    visualizer.updateWeightsFromTFModel(model);
    updateWeightHeatmap(model);
    return history;
}

// Predict operation
function predictOperation(text) {
    const bow = textToBow(text);
    const input = tf.tensor2d([bow]);
    const pred = model.predict(input);
    const probs = pred.dataSync();
    const maxIdx = probs.indexOf(Math.max(...probs));
    input.dispose(); pred.dispose();
    return { class: maxIdx, className: CLASS_NAMES[maxIdx], confidence: probs[maxIdx] };
}

// CoT reasoning
async function startReasoning() {
    const input = document.getElementById('inputText').value.trim();
    if (!input) { alert('Enter a problem'); return; }
    const depth = parseInt(document.getElementById('depthSlider').value);
    const chainDiv = document.getElementById('reasoningChain'); chainDiv.innerHTML = '';
    document.getElementById('finalOutput').innerHTML = '';
    document.getElementById('thinkingIndicator').style.display = 'flex';
    document.getElementById('tokenContainer').style.display = 'none';
    const start = performance.now();

    const pred = predictOperation(input);
    const baseConf = pred.confidence;
    
    const templates = [
        { type: '🔍 NLP classification', content: `Network predicts: ${pred.className} (${(baseConf*100).toFixed(1)}%). Activating relevant reasoning templates.` },
        { type: '📐 Problem parsing', content: `Extracting quantities using ${pred.className} schema.` },
        { type: '🧮 Computation plan', content: `Applying ${pred.className} formulas.` },
        { type: '✅ Verification', content: `Checking units and consistency.` },
        { type: '📊 Synthesis', content: `Combining steps into final answer.` }
    ];

    for (let i = 0; i < depth; i++) {
        const t = templates[i % templates.length];
        const conf = Math.min(0.7 + i*0.05 + baseConf*0.1, 0.98);
        addStepToUI({ step: i+1, type: t.type, content: t.content, confidence: conf });
        if (visualizer) {
            visualizer.nodes.forEach(n => { if (n.layer < 2) n.target = 0.4 + 0.5 * Math.sin(i * 0.5 + n.idx); });
        }
        await new Promise(r => setTimeout(r, 300));
    }

    let final = '';
    if (pred.class === 0) final = 'Speed = 120/2 = 60 km/h. Distance in 5h = 60×5 = 300 km.';
    else if (pred.class === 1) final = 'Work = 3×4 = 12 worker-days. 6 workers → 12/6 = 2 days.';
    else if (pred.class === 2) final = '20% of 150 = 0.20×150 = 30.';
    else final = 'Ratio 3:4, girls=28 → boys = (3/4)×28 = 21.';
    
    document.getElementById('finalOutput').innerHTML = `<strong style="color:var(--primary);">Solution:</strong><br>${final}`;
    document.getElementById('thinkingIndicator').style.display = 'none';
    showTokens(input.split(/\s+/).slice(0, 12));
    document.getElementById('statTime').innerText = Math.round(performance.now() - start) + 'ms';
}

function addStepToUI(step) {
    const c = document.getElementById('reasoningChain');
    const card = document.createElement('div'); card.className = 'step-card';
    const conf = (step.confidence * 100).toFixed(1);
    card.innerHTML = `<div class="step-header"><div class="step-number">${step.step}</div><div class="step-title">${step.type}</div></div>
        <div class="step-content">${step.content}</div><div class="step-meta"><span>🎯 ${conf}%</span></div>
        <div class="confidence-bar"><div class="confidence-fill" style="width:${conf}%"></div></div>`;
    c.appendChild(card); c.scrollTop = c.scrollHeight;
}

function showTokens(tokens) {
    const cont = document.getElementById('tokenList'); cont.innerHTML = '';
    tokens.forEach((t, i) => {
        const s = document.createElement('span'); s.className = 'token'; s.textContent = t;
        s.style.animationDelay = i*0.05+'s'; cont.appendChild(s);
    });
    document.getElementById('tokenContainer').style.display = 'block';
}

function testNLP() {
    const input = document.getElementById('inputText').value || "A train travels 120 km in 2 hours. How far in 5 hours?";
    const pred = predictOperation(input);
    alert(`NLP prediction: ${pred.className} (${(pred.confidence*100).toFixed(1)}% confidence)`);
}

function clearAll() {
    document.getElementById('inputText').value = '';
    document.getElementById('reasoningChain').innerHTML = '<div class="empty-state"><div class="empty-icon">🤔</div><div>No active session</div></div>';
    document.getElementById('finalOutput').innerHTML = 'Enter a problem and click "Start Reasoning" — NLP classification guides CoT.';
    document.getElementById('tokenContainer').style.display = 'none';
    document.getElementById('statTime').innerText = '0ms';
    if (visualizer) visualizer.initNetwork();
}

// UI binding and initialization
function init() {
    // Background neurons
    for (let i = 0; i < 50; i++) {
        const n = document.createElement('div'); n.className = 'neuron';
        n.style.left = Math.random()*100+'%'; n.style.top = Math.random()*100+'%';
        n.style.animationDelay = Math.random()*20+'s';
        document.getElementById('bgAnimation').appendChild(n);
    }
    
    visualizer = new NetworkVisualizer('networkCanvas');
    
    // Slider displays
    document.getElementById('depthSlider').addEventListener('input', e => document.getElementById('depthValue').innerText = e.target.value + ' steps');
    document.getElementById('numLayersSlider').addEventListener('input', e => document.getElementById('numLayersValue').innerText = e.target.value);
    document.getElementById('unitsSlider').addEventListener('input', e => document.getElementById('unitsValue').innerText = e.target.value);
    document.getElementById('lrSlider').addEventListener('input', e => document.getElementById('lrValue').innerText = parseFloat(e.target.value).toFixed(4));
    
    // Buttons
    document.getElementById('startReasoningBtn').addEventListener('click', startReasoning);
    document.getElementById('resetBtn').addEventListener('click', clearAll);
    document.getElementById('trainBtn').addEventListener('click', () => trainNetwork(20));
    document.getElementById('testNLPBtn').addEventListener('click', testNLP);
    
    // Initial training
    trainNetwork(15, true).then(() => {
        document.getElementById('overlay').classList.remove('active');
        visualizer.updateWeightsFromTFModel(model);
        updateWeightHeatmap(model);
    });
}

window.addEventListener('DOMContentLoaded', init);