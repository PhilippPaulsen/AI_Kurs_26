class Grid {
    constructor(width, height, coverage) {
        this.width = width;
        this.height = height;
        this.grid = Array(height).fill().map(() => Array(width).fill(0)); // 0:Empty, 1:Wall, 2:Goal
        this.startPos = { r: 0, c: 0 };
        this.agentPos = { ...this.startPos };
        this.goalPos = { r: height - 1, c: width - 1 };
        this.walls = [];
        this.visited = new Set();
        this.visited.add(`0,0`);

        this.generateWalls(coverage);
        this.grid[this.goalPos.r][this.goalPos.c] = 2;
    }

    generateWalls(coverage) {
        let numWalls = Math.floor(this.width * this.height * coverage);
        for (let i = 0; i < numWalls; i++) {
            let r, c;
            do {
                r = Math.floor(Math.random() * this.height);
                c = Math.floor(Math.random() * this.width);
            } while ((r === 0 && c === 0) || (r === this.goalPos.r && c === this.goalPos.c) || this.grid[r][c] === 1);
            this.grid[r][c] = 1;
            this.walls.push({ r, c });
        }
    }

    step(action) {
        // 0:Up, 1:Down, 2:Left, 3:Right
        let { r, c } = this.agentPos;
        let nr = r, nc = c;

        if (action === 0) nr--;
        if (action === 1) nr++;
        if (action === 2) nc--;
        if (action === 3) nc++;

        if (nr >= 0 && nr < this.height && nc >= 0 && nc < this.width) {
            if (this.grid[nr][nc] === 1) {
                return { state: this.agentPos, reward: -1, done: false, type: 'wall' };
            } else if (this.grid[nr][nc] === 2) {
                this.agentPos = { r: nr, c: nc };
                this.visited.add(`${nr},${nc}`);
                return { state: this.agentPos, reward: 100, done: true, type: 'goal' };
            } else {
                this.agentPos = { r: nr, c: nc };
                this.visited.add(`${nr},${nc}`);
                return { state: this.agentPos, reward: -0.1, done: false, type: 'move' };
            }
        }
        return { state: this.agentPos, reward: -1, done: false, type: 'bound' };
    }

    getPercept() {
        let { r, c } = this.agentPos;
        const moves = [{ dr: -1, dc: 0, k: 'up' }, { dr: 1, dc: 0, k: 'down' }, { dr: 0, dc: -1, k: 'left' }, { dr: 0, dc: 1, k: 'right' }];
        let p = {};
        moves.forEach(m => {
            let nr = r + m.dr, nc = c + m.dc;
            if (nr < 0 || nr >= this.height || nc < 0 || nc >= this.width) p[m.k] = 'limit';
            else if (this.grid[nr][nc] === 1) p[m.k] = 'wall';
            else if (this.grid[nr][nc] === 2) p[m.k] = 'goal';
            else p[m.k] = 'empty'; // Visited check handled by agent memory if needed
        });
        return p;
    }
}

// --- Agents ---

class Agent {
    constructor() { this.log = ""; }
    act(percept) { return null; }
}

class SimpleReflexAgent extends Agent {
    act(percept) {
        const dirs = ['up', 'down', 'left', 'right'];
        // Rule 1: Goal
        for (let i = 0; i < 4; i++) {
            if (percept[dirs[i]] === 'goal') {
                this.log = `Found GOAL at ${dirs[i]}!`;
                return i;
            }
        }
        // Rule 2: Random Safe
        let safe = [];
        for (let i = 0; i < 4; i++) {
            if (percept[dirs[i]] !== 'wall' && percept[dirs[i]] !== 'limit') safe.push(i);
        }
        if (safe.length > 0) {
            const action = safe[Math.floor(Math.random() * safe.length)];
            this.log = `Moving ${dirs[action]} (Safe random)`;
            return action;
        }
        this.log = "Trapped!";
        return Math.floor(Math.random() * 4);
    }
}

class ModelBasedReflexAgent extends Agent {
    constructor() {
        super();
        this.visited = new Set(['0,0']);
    }

    updateModel(r, c) {
        this.visited.add(`${r},${c}`);
    }

    act(percept, agentPos) {
        const dirs = ['up', 'down', 'left', 'right'];
        const moves = [{ dr: -1, dc: 0 }, { dr: 1, dc: 0 }, { dr: 0, dc: -1 }, { dr: 0, dc: 1 }];

        // Rule 1: Goal
        for (let i = 0; i < 4; i++) {
            if (percept[dirs[i]] === 'goal') {
                this.log = `Memory: Goal at ${dirs[i]}`;
                return i;
            }
        }

        // Rule 2: Prefer Unvisited
        let unvisited = [];
        let visitedSafe = [];

        for (let i = 0; i < 4; i++) {
            if (percept[dirs[i]] === 'wall' || percept[dirs[i]] === 'limit') continue;

            let nr = agentPos.r + moves[i].dr;
            let nc = agentPos.c + moves[i].dc;

            if (!this.visited.has(`${nr},${nc}`)) unvisited.push(i);
            else visitedSafe.push(i);
        }

        if (unvisited.length > 0) {
            const action = unvisited[Math.floor(Math.random() * unvisited.length)];
            this.log = `Exploring new cell: ${dirs[action]}`;
            return action;
        }

        if (visitedSafe.length > 0) {
            const action = visitedSafe[Math.floor(Math.random() * visitedSafe.length)];
            this.log = `Backtracking: ${dirs[action]}`;
            return action;
        }

        return Math.floor(Math.random() * 4);
    }
}

class QLearningAgent extends Agent {
    constructor(alpha, gamma, epsilon) {
        super();
        this.qTable = {}; // key: "r,c", value: [0,0,0,0]
        this.alpha = alpha;
        this.gamma = gamma;
        this.epsilon = epsilon;
        this.prevState = null;
        this.prevAction = null;
    }

    getQ(stateKey) {
        if (!this.qTable[stateKey]) this.qTable[stateKey] = [0, 0, 0, 0];
        return this.qTable[stateKey];
    }

    act(stateKey) {
        if (Math.random() < this.epsilon) {
            this.log = "Exploring (Random)";
            this.prevAction = Math.floor(Math.random() * 4);
        } else {
            const qs = this.getQ(stateKey);
            const maxQ = Math.max(...qs);
            const maxActions = qs.map((v, i) => v === maxQ ? i : -1).filter(i => i !== -1);
            this.prevAction = maxActions[Math.floor(Math.random() * maxActions.length)];
            this.log = `Exploiting (Max Q: ${maxQ.toFixed(2)})`;
        }
        this.prevState = stateKey;
        return this.prevAction;
    }

    learn(nextStateKey, reward) {
        if (this.prevState !== null) {
            const oldQ = this.getQ(this.prevState)[this.prevAction];
            const nextMaxQ = Math.max(...this.getQ(nextStateKey));
            const newQ = oldQ + this.alpha * (reward + this.gamma * nextMaxQ - oldQ);
            this.qTable[this.prevState][this.prevAction] = newQ;
        }
    }
}

// --- App Logic ---

let env, agent, timer;
let isRunning = false;
let stepCount = 0;
let totalReward = 0;

// DOM Elements
const gridEl = document.getElementById('gridContainer');
const logEl = document.getElementById('agentLog');
const stepEl = document.getElementById('stepCount');
const rewardEl = document.getElementById('totalReward');

function init() {
    const size = parseInt(document.getElementById('gridSize').value);
    const coverage = parseFloat(document.getElementById('wallCoverage').value);
    const type = document.getElementById('agentType').value;

    env = new Grid(size, size, coverage);

    if (type === 'manual') agent = new Agent();
    else if (type === 'simple') agent = new SimpleReflexAgent();
    else if (type === 'model') agent = new ModelBasedReflexAgent();
    else if (type === 'qlearning') {
        const a = parseFloat(document.getElementById('alpha').value);
        const g = parseFloat(document.getElementById('gamma').value);
        const e = parseFloat(document.getElementById('epsilon').value);
        agent = new QLearningAgent(a, g, e);
    }

    if (type === 'qlearning') {
        document.getElementById('qParams').style.display = 'block';
        document.getElementById('qTableContainer').style.display = 'block';
    } else {
        document.getElementById('qParams').style.display = 'none';
        document.getElementById('qTableContainer').style.display = 'none';
    }

    stepCount = 0;
    totalReward = 0;
    document.getElementById('autoBtn').textContent = "Auto Run";
    clearInterval(timer);
    isRunning = false;

    render();
}

function render() {
    let html = "";
    const agentView = document.getElementById('agentView').checked;
    const type = document.getElementById('agentType').value;

    for (let r = 0; r < env.height; r++) {
        let rowStr = "";
        for (let c = 0; c < env.width; c++) {
            let char = ". "; // Empty
            let cellClass = "";
            let hidden = false;

            // Agent View Logic
            if (agentView) {
                const dist = Math.abs(env.agentPos.r - r) + Math.abs(env.agentPos.c - c);
                // Fog of War: Show if adjacent (dist<=1) OR if visited (for model-based)
                let visible = dist <= 1;
                if (type === 'model' && agent.visited.has(`${r},${c}`)) visible = true;
                if (!visible) hidden = true;
            }

            if (hidden) {
                char = "  "; // Hidden
            } else {
                if (r === env.agentPos.r && c === env.agentPos.c) char = "A ";
                else if (r === env.goalPos.r && c === env.goalPos.c) char = "G ";
                else if (env.grid[r][c] === 1) char = "# ";
                else if (env.visited.has(`${r},${c}`) && (r !== 0 || c !== 0)) char = "* "; // Visited trail
            }
            rowStr += char;
        }
        html += rowStr + "\n";
    }
    gridEl.innerText = html;

    stepEl.innerText = stepCount;
    rewardEl.innerText = totalReward.toFixed(1);

    if (agent.log) logEl.innerText = agent.log;

    // Q-Table Render (simplified)
    if (agent instanceof QLearningAgent) {
        renderQTable();
    }
}

function renderQTable() {
    // A simple grid view of max Q values
    // TODO: A detailed table is hard to fit. Let's just dump values for visited states?
    // Or maybe a small heatmap grid?
    // For now, let's keep it simple text or skip complex render to avoid clutter
    // Actually, user asked for Heatmap table directly under grid.
    // Let's make a grid of divs
    const qContainer = document.getElementById('qTable');
    qContainer.style.gridTemplateColumns = `repeat(${env.width}, 1fr)`;
    qContainer.innerHTML = "";

    for (let r = 0; r < env.height; r++) {
        for (let c = 0; c < env.width; c++) {
            const val = agent.qTable[`${r},${c}`];
            const div = document.createElement('div');
            div.className = 'q-cell';
            if (val) {
                const maxQ = Math.max(...val);
                // Color based on value?
                const intensity = Math.min(255, Math.floor(maxQ * 2)); // simple scale
                div.style.color = maxQ > 0 ? '#0f0' : '#f00';
                div.innerText = maxQ.toFixed(1);
            } else {
                div.innerText = "-";
            }
            qContainer.appendChild(div);
        }
    }
}

function step(manualAction = null) {
    if (stepCount > 0 && env.agentPos.r === env.goalPos.r && env.agentPos.c === env.goalPos.c) return;

    let action = null;
    const type = document.getElementById('agentType').value;

    if (type === 'manual') {
        if (manualAction === null) return;
        action = manualAction;
    } else {
        const percept = env.getPercept();
        if (type === 'simple') action = agent.act(percept);
        else if (type === 'model') action = agent.act(percept, env.agentPos);
        else if (type === 'qlearning') action = agent.act(`${env.agentPos.r},${env.agentPos.c}`);
    }

    if (action !== null) {
        const res = env.step(action);
        stepCount++;
        totalReward += res.reward;

        if (type === 'model') agent.updateModel(res.state.r, res.state.c);
        if (type === 'qlearning') agent.learn(`${res.state.r},${res.state.c}`, res.reward);

        if (res.done) {
            agent.log = "GOAL REACHED!";
            if (isRunning) toggleAuto();
        }
    }
    render();
}

function toggleAuto() {
    if (isRunning) {
        clearInterval(timer);
        isRunning = false;
        document.getElementById('autoBtn').textContent = "Auto Run";
    } else {
        const speed = parseInt(document.getElementById('speed').value);
        timer = setInterval(() => step(), speed);
        isRunning = true;
        document.getElementById('autoBtn').textContent = "Stop";
    }
}

// Event Listeners
document.getElementById('resetBtn').onclick = init;
document.getElementById('stepBtn').onclick = () => step();
document.getElementById('autoBtn').onclick = toggleAuto;
document.getElementById('agentView').onchange = render;

// Sliders update labels
['gridSize', 'wallCoverage', 'speed', 'alpha', 'gamma', 'epsilon'].forEach(id => {
    document.getElementById(id).oninput = (e) => {
        document.getElementById(id + 'Val').innerText = e.target.value;
        if ((id === 'alpha' || id === 'gamma' || id === 'epsilon') && agent instanceof QLearningAgent) {
            agent[id] = parseFloat(e.target.value);
        }
    };
});

document.getElementById('agentType').onchange = init;

// Initialization
init();

// Manual Keys
window.onkeydown = (e) => {
    if (document.getElementById('agentType').value === 'manual') {
        if (e.key === 'ArrowUp') step(0);
        if (e.key === 'ArrowDown') step(1);
        if (e.key === 'ArrowLeft') step(2);
        if (e.key === 'ArrowRight') step(3);
    }
};
