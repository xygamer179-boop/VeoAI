// veo.js - Canvas network visualizer and weight heatmap
class NetworkVisualizer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.resize();
        window.addEventListener('resize', () => this.resize());
        this.nodes = [];
        this.connections = [];
        this.particles = [];
        this.initNetwork();
        this.animationId = null;
        this.start();
    }

    resize() {
        this.canvas.width = this.canvas.offsetWidth;
        this.canvas.height = this.canvas.offsetHeight;
    }

    initNetwork(layerSizes = [8, 12, 8, 4]) {
        this.nodes = [];
        this.connections = [];
        const layers = layerSizes.length;
        const layerX = layerSizes.map((_, i) => (i + 0.5) * (this.canvas.width / layers));
        let nodeId = 0;
        layerSizes.forEach((n, l) => {
            for (let i = 0; i < n; i++) {
                this.nodes.push({
                    id: nodeId++,
                    x: layerX[l],
                    y: (i + 1) * (this.canvas.height / (n + 1)),
                    layer: l,
                    idx: i,
                    activation: 0.3,
                    target: 0.3,
                    phase: Math.random() * Math.PI * 2
                });
            }
        });
        // Create connections between consecutive layers
        let offset = 0;
        for (let l = 0; l < layers - 1; l++) {
            const fromCount = layerSizes[l];
            const toCount = layerSizes[l + 1];
            for (let i = 0; i < fromCount; i++) {
                for (let j = 0; j < toCount; j++) {
                    this.connections.push({
                        from: offset + i,
                        to: offset + fromCount + j,
                        weight: Math.random(),
                        active: false,
                        pulse: 0
                    });
                }
            }
            offset += fromCount;
        }
        // Particles
        for (let i = 0; i < 20; i++) {
            this.particles.push({
                x: Math.random() * this.canvas.width,
                y: Math.random() * this.canvas.height,
                vx: (Math.random() - 0.5) * 0.5,
                vy: (Math.random() - 0.5) * 0.5,
                life: Math.random()
            });
        }
    }

    updateWeightsFromTFModel(model) {
        // Extract weights from first Dense layer and map to connections between layer0 and layer1
        try {
            const weights = model.getWeights();
            if (weights.length >= 2) {
                const kernel = weights[0].arraySync(); // shape [inputDim, units]
                const inputDim = kernel.length;
                const units = kernel[0].length;
                this.connections.forEach(conn => {
                    const fromNode = this.nodes[conn.from];
                    const toNode = this.nodes[conn.to];
                    if (fromNode.layer === 0 && toNode.layer === 1) {
                        const row = Math.min(fromNode.idx, inputDim - 1);
                        const col = Math.min(toNode.idx, units - 1);
                        const w = kernel[row][col];
                        conn.weight = (w + 1.5) / 3; // normalize for visualization
                        conn.active = conn.weight > 0.6;
                    }
                });
            }
        } catch (e) {
            console.warn('Could not update weights from model', e);
        }
    }

    draw() {
        this.ctx.fillStyle = 'rgba(10,14,39,0.15)';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        // Draw connections
        this.connections.forEach(c => {
            const from = this.nodes[c.from];
            const to = this.nodes[c.to];
            const w = isNaN(c.weight) ? 0.2 : c.weight;
            const gradient = this.ctx.createLinearGradient(from.x, from.y, to.x, to.y);
            gradient.addColorStop(0, `rgba(0,210,255,${0.2 + w * 0.6})`);
            gradient.addColorStop(1, `rgba(58,123,213,${0.2 + w * 0.6})`);
            this.ctx.beginPath();
            this.ctx.moveTo(from.x, from.y);
            this.ctx.lineTo(to.x, to.y);
            this.ctx.strokeStyle = gradient;
            this.ctx.lineWidth = c.active ? 2.5 : 1;
            this.ctx.stroke();
        });
        // Draw nodes
        this.nodes.forEach(n => {
            n.activation += (n.target - n.activation) * 0.08;
            n.phase += 0.02;
            const r = 5 + n.activation * 8;
            const grad = this.ctx.createRadialGradient(n.x, n.y, 0, n.x, n.y, r * 2.5);
            grad.addColorStop(0, `rgba(0,210,255,${0.5 + n.activation * 0.5})`);
            grad.addColorStop(1, 'rgba(0,210,255,0)');
            this.ctx.beginPath();
            this.ctx.arc(n.x, n.y, r * 2.5, 0, Math.PI * 2);
            this.ctx.fillStyle = grad;
            this.ctx.fill();
            this.ctx.beginPath();
            this.ctx.arc(n.x, n.y, r * 0.6, 0, Math.PI * 2);
            this.ctx.fillStyle = 'rgba(255,255,255,0.9)';
            this.ctx.fill();
        });
        // Particles
        this.particles.forEach(p => {
            p.x += p.vx;
            p.y += p.vy;
            p.life -= 0.005;
            if (p.life <= 0 || p.x < 0 || p.x > this.canvas.width || p.y < 0 || p.y > this.canvas.height) {
                p.x = Math.random() * this.canvas.width;
                p.y = Math.random() * this.canvas.height;
                p.life = 1;
            }
            this.ctx.beginPath();
            this.ctx.arc(p.x, p.y, 2, 0, 2 * Math.PI);
            this.ctx.fillStyle = `rgba(0,210,255,${p.life * 0.4})`;
            this.ctx.fill();
        });
    }

    animate() {
        this.draw();
        this.animationId = requestAnimationFrame(() => this.animate());
    }

    start() {
        if (!this.animationId) this.animate();
    }

    stop() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
    }
}

// Weight heatmap updater
function updateWeightHeatmap(model) {
    const grid = document.getElementById('attentionGrid');
    grid.innerHTML = '';
    try {
        const weights = model.getWeights();
        if (weights.length > 0) {
            const kernel = weights[0].arraySync(); // shape [50, units]
            const rows = Math.min(kernel.length, 8);
            const cols = Math.min(kernel[0].length, 8);
            for (let r = 0; r < 8; r++) {
                for (let c = 0; c < 8; c++) {
                    const cell = document.createElement('div');
                    cell.className = 'attention-cell';
                    const row = r % rows;
                    const col = c % cols;
                    let val = kernel[row]?.[col] ?? 0;
                    val = (val + 1.2) / 2.5; // normalize to 0..1
                    val = Math.min(Math.max(val, 0.1), 0.95);
                    cell.style.backgroundColor = `rgba(0,210,255,${val})`;
                    cell.style.opacity = 0.4 + val * 0.6;
                    grid.appendChild(cell);
                }
            }
        }
    } catch (e) {
        console.warn('Heatmap update failed', e);
    }
}