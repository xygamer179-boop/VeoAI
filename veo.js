// veo.js — Neural Network Canvas Visualizer
// Bug fixed: bezier control points now stored per-connection (no more per-frame random jitter)
// Added:    fireSignal() for forward-pass propagation animation

class NetworkVisualizer {
    constructor(canvasId) {
        this.canvas  = document.getElementById(canvasId);
        this.ctx     = this.canvas.getContext('2d');
        this.nodes   = [];
        this.connections = [];
        this.particles   = [];
        this.signals     = [];      // moving "signal" dots along connections
        this.mouseX = 0;
        this.mouseY = 0;
        this.hoveredNode = null;
        this.animationId = null;

        this.canvas.addEventListener('mousemove', e => {
            const r = this.canvas.getBoundingClientRect();
            this.mouseX = e.clientX - r.left;
            this.mouseY = e.clientY - r.top;
        });

        this.resize();
        window.addEventListener('resize', () => { this.resize(); this.initNetwork(); });
        this.initNetwork();
        this.start();
    }

    resize() {
        this.canvas.width  = this.canvas.offsetWidth  || 400;
        this.canvas.height = this.canvas.offsetHeight || 300;
    }

    // ── Build network topology ───────────────────────────────────────────────
    initNetwork(layerSizes = [6, 10, 8, 6, 4]) {
        this.nodes       = [];
        this.connections = [];
        this.particles   = [];
        this.signals     = [];

        const L   = layerSizes.length;
        const W   = this.canvas.width;
        const H   = this.canvas.height;
        const lxs = layerSizes.map((_, i) => (i + 0.5) * (W / L));

        let nodeId = 0;
        layerSizes.forEach((n, l) => {
            for (let i = 0; i < n; i++) {
                this.nodes.push({
                    id:         nodeId++,
                    x:          lxs[l],
                    y:          (i + 1) * (H / (n + 1)),
                    layer:      l,
                    idx:        i,
                    activation: 0.2 + Math.random() * 0.3,
                    target:     0.2 + Math.random() * 0.3,
                    phase:      Math.random() * Math.PI * 2,
                    speed:      0.008 + Math.random() * 0.015,
                    size:       4 + Math.random() * 2.5
                });
            }
        });

        // Build connections and store STABLE bezier control points
        let offset = 0;
        for (let l = 0; l < L - 1; l++) {
            const fc = layerSizes[l];
            const tc = layerSizes[l + 1];
            for (let i = 0; i < fc; i++) {
                for (let j = 0; j < tc; j++) {
                    const from = this.nodes[offset + i];
                    const to   = this.nodes[offset + fc + j];
                    const midX = (from.x + to.x) / 2;
                    const midY = (from.y + to.y) / 2;
                    this.connections.push({
                        from:       offset + i,
                        to:         offset + fc + j,
                        weight:     Math.random() * 0.5 + 0.2,
                        active:     false,
                        pulse:      0,
                        // [BUG FIX] — control points stored once, not recomputed every frame
                        cpX:        midX + (Math.random() - 0.5) * 18,
                        cpY:        midY + (Math.random() - 0.5) * 18,
                        flowOffset: Math.random()
                    });
                }
            }
            offset += fc;
        }

        // Background particles
        for (let i = 0; i < 28; i++) {
            this.particles.push({
                x: Math.random() * W,
                y: Math.random() * H,
                vx: (Math.random() - 0.5) * 0.7,
                vy: (Math.random() - 0.5) * 0.7,
                life: Math.random(),
                maxLife: 0.4 + Math.random() * 0.6,
                size: 1 + Math.random() * 1.8
            });
        }
    }

    // ── Sync weights from trained model ────────────────────────────────────
    updateWeightsFromTFModel(model) {
        try {
            const weights = model.getWeights();
            if (weights.length > 0) {
                const kernel   = weights[0].arraySync();
                const inputDim = kernel.length;
                const units    = kernel[0]?.length || 1;
                this.connections.forEach(c => {
                    const fn = this.nodes[c.from];
                    const tn = this.nodes[c.to];
                    if (fn.layer === 0 && tn.layer === 1) {
                        const row = Math.min(fn.idx, inputDim - 1);
                        const col = Math.min(tn.idx, units - 1);
                        const w   = kernel[row]?.[col] ?? 0;
                        c.weight = Math.min(Math.max((w + 1.5) / 3, 0.05), 1);
                        c.active = c.weight > 0.55;
                    }
                });
            }
        } catch (e) {
            console.warn('Weight sync failed', e);
        }
    }

    activateLayer(layerIndex, intensity = 1) {
        this.nodes.forEach(n => {
            if (n.layer === layerIndex)
                n.target = Math.min(n.target + intensity * 0.5, 1);
        });
    }

    // ── Fire a "signal" that travels forward through the network ──────────
    fireSignal(fromLayer = 0) {
        const srcNodes = this.nodes.filter(n => n.layer === fromLayer);
        // Pick a random subset to avoid flooding
        const picks = srcNodes.filter(() => Math.random() > 0.5).slice(0, 4);
        picks.forEach(src => {
            const conns = this.connections.filter(c => c.from === src.id);
            conns.slice(0, 3).forEach(c => {
                this.signals.push({ conn: c, t: 0, speed: 0.025 + Math.random() * 0.025 });
            });
        });
    }

    pulseConnection(fromId, toId) {
        this.connections.forEach(c => {
            if (c.from === fromId && c.to === toId) c.pulse = 1;
        });
    }

    // ── Main draw ───────────────────────────────────────────────────────────
    draw() {
        const ctx = this.ctx;
        const W   = this.canvas.width;
        const H   = this.canvas.height;

        // Fade trail
        ctx.fillStyle = 'rgba(3, 7, 18, 0.18)';
        ctx.fillRect(0, 0, W, H);

        // Subtle grid
        ctx.strokeStyle = 'rgba(0, 212, 255, 0.025)';
        ctx.lineWidth   = 1;
        const gs = 32;
        for (let x = 0; x < W; x += gs) {
            ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
        }
        for (let y = 0; y < H; y += gs) {
            ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
        }

        // ── Connections ────────────────────────────────────────────────────
        this.connections.forEach(c => {
            const from = this.nodes[c.from];
            const to   = this.nodes[c.to];
            const w    = isNaN(c.weight) ? 0.2 : c.weight;

            // Pulse glow at midpoint
            if (c.pulse > 0) {
                c.pulse -= 0.018;
                const mx = (from.x + to.x) / 2;
                const my = (from.y + to.y) / 2;
                const pg = ctx.createRadialGradient(mx, my, 0, mx, my, 22);
                pg.addColorStop(0, `rgba(0,212,255,${c.pulse * 0.9})`);
                pg.addColorStop(1, 'rgba(0,212,255,0)');
                ctx.beginPath(); ctx.arc(mx, my, 22, 0, Math.PI * 2);
                ctx.fillStyle = pg; ctx.fill();
            }

            // Gradient line — uses STORED cpX/cpY (no random per frame)
            const grad = ctx.createLinearGradient(from.x, from.y, to.x, to.y);
            const baseA = 0.1 + w * 0.45;
            grad.addColorStop(0,   `rgba(0,212,255,${baseA})`);
            grad.addColorStop(0.5, `rgba(124,58,237,${baseA * 1.3})`);
            grad.addColorStop(1,   `rgba(244,114,182,${baseA})`);

            ctx.beginPath();
            ctx.moveTo(from.x, from.y);
            ctx.quadraticCurveTo(c.cpX, c.cpY, to.x, to.y);
            ctx.strokeStyle = grad;
            ctx.lineWidth   = c.active ? 2.2 : 0.8;
            ctx.stroke();
        });

        // ── Signal particles along connections ────────────────────────────
        this.signals = this.signals.filter(sig => {
            sig.t += sig.speed;
            if (sig.t >= 1) return false;

            const c    = sig.conn;
            const from = this.nodes[c.from];
            const to   = this.nodes[c.to];
            const t    = sig.t;

            // Quadratic bezier position
            const sx = (1-t)*(1-t)*from.x + 2*(1-t)*t*c.cpX + t*t*to.x;
            const sy = (1-t)*(1-t)*from.y + 2*(1-t)*t*c.cpY + t*t*to.y;

            const sg = ctx.createRadialGradient(sx, sy, 0, sx, sy, 5);
            sg.addColorStop(0, 'rgba(0,212,255,0.9)');
            sg.addColorStop(1, 'rgba(0,212,255,0)');
            ctx.beginPath(); ctx.arc(sx, sy, 5, 0, Math.PI * 2);
            ctx.fillStyle = sg; ctx.fill();

            // Wake up target node when signal arrives
            if (sig.t > 0.9) {
                const tn = this.nodes[c.to];
                tn.target = Math.min(tn.target + 0.3, 1);
            }
            return true;
        });

        // ── Nodes ──────────────────────────────────────────────────────────
        this.nodes.forEach(n => {
            n.activation += (n.target - n.activation) * 0.07;
            n.target     *= 0.995;   // slow decay
            n.phase      += n.speed;

            const isHovered = Math.hypot(this.mouseX - n.x, this.mouseY - n.y) < 18;
            const r          = n.size + n.activation * 9;

            // Outer glow
            const glow = ctx.createRadialGradient(n.x, n.y, 0, n.x, n.y, r * 4);
            glow.addColorStop(0, `rgba(0,212,255,${0.12 + n.activation * 0.22})`);
            glow.addColorStop(0.4, `rgba(124,58,237,${0.05 + n.activation * 0.08})`);
            glow.addColorStop(1, 'rgba(0,212,255,0)');
            ctx.beginPath(); ctx.arc(n.x, n.y, r * 4, 0, Math.PI * 2);
            ctx.fillStyle = glow; ctx.fill();

            // Node body
            const body = ctx.createRadialGradient(n.x - r * 0.3, n.y - r * 0.3, 0, n.x, n.y, r);
            body.addColorStop(0, `rgba(255,255,255,${0.9 + n.activation * 0.1})`);
            body.addColorStop(1, `rgba(0,212,255,${0.6 + n.activation * 0.4})`);
            ctx.beginPath(); ctx.arc(n.x, n.y, r * 0.55, 0, Math.PI * 2);
            ctx.fillStyle = body; ctx.fill();

            // Hover ring
            if (isHovered) {
                ctx.beginPath(); ctx.arc(n.x, n.y, r + 6, 0, Math.PI * 2);
                ctx.strokeStyle = 'rgba(0,212,255,0.85)';
                ctx.lineWidth   = 1.5;
                ctx.stroke();
            }
        });

        // ── Background particles ────────────────────────────────────────────
        this.particles.forEach(p => {
            p.x += p.vx; p.y += p.vy; p.life -= 0.0025;
            if (p.life <= 0 || p.x < 0 || p.x > W || p.y < 0 || p.y > H) {
                p.x = Math.random() * W; p.y = Math.random() * H;
                p.life = p.maxLife;
                p.vx = (Math.random() - 0.5) * 0.7;
                p.vy = (Math.random() - 0.5) * 0.7;
            }
            ctx.beginPath(); ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(0,212,255,${p.life * 0.45})`; ctx.fill();
        });
    }

    animate() {
        this.draw();
        this.animationId = requestAnimationFrame(() => this.animate());
    }

    start() { if (!this.animationId) this.animate(); }
    stop()  {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
    }
}

// ─── Weight Heatmap ──────────────────────────────────────────────────────────
function updateWeightHeatmap(model) {
    const grid = document.getElementById('attentionGrid');
    if (!grid) return;
    grid.innerHTML = '';

    try {
        const weights = model.getWeights();
        if (!weights.length) return;
        const kernel = weights[0].arraySync();
        const rows   = Math.min(kernel.length, 8);
        const cols   = Math.min(kernel[0]?.length || 0, 8);

        for (let r = 0; r < 8; r++) {
            for (let c = 0; c < 8; c++) {
                const cell = document.createElement('div');
                cell.className = 'attention-cell';
                let val = kernel[r % rows]?.[c % cols] ?? 0;
                val = Math.min(Math.max((val + 1.2) / 2.5, 0.05), 0.98);
                const hue = 180 + val * 65;
                cell.style.backgroundColor = `hsla(${hue},80%,50%,${0.15 + val * 0.85})`;
                cell.style.boxShadow = val > 0.55 ? `0 0 7px hsla(${hue},80%,55%,0.55)` : 'none';
                cell.title = `w[${r}][${c}] = ${val.toFixed(3)}`;
                grid.appendChild(cell);
            }
        }
    } catch (e) {
        console.warn('Heatmap update failed', e);
    }
}
