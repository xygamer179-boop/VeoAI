// data.js — Vocabulary, training data, and default configuration
// VOCAB_SIZE is auto-computed from VOCAB.length — no hardcoding

const VOCAB = [
    // Speed / Distance (16 tokens)
    "train","travel","travels","km","kilometer","speed","distance",
    "hour","hours","car","bus","airplane","cyclist","road","move","moves",
    // Work / Rate (15 tokens)
    "worker","workers","day","days","job","complete","finish",
    "machine","machines","task","project","take","takes","painter","painters",
    // Percentage (7 tokens)
    "percent","percentage","what","calculate","find","of","out",
    // Ratio (13 tokens)
    "ratio","boys","girls","apples","oranges","red","blue",
    "yellow","paint","mix","sugar","flour","balls",
    // Shared / Common (13 tokens)
    "how","many","much","long","far","if","in","the","a","per","rate","time","more"
];
// Total: 64 unique tokens — matches VOCAB_SIZE below
const VOCAB_SIZE = VOCAB.length; // 64

const CLASS_NAMES = ['Speed/Distance', 'Work/Rate', 'Percentage', 'Ratio'];

const TRAIN_DATA = [
    // ── Speed / Distance — label 0 ──────────────────────────────────────────
    { text: "A train travels 120 km in 2 hours. How far in 5 hours?",            label: 0 },
    { text: "A car moves at 60 km/h for 3 hours. What distance does it cover?",  label: 0 },
    { text: "Train speed is 80 km/h traveling for 2.5 hours. Distance?",         label: 0 },
    { text: "A cyclist travels 45 km in 3 hours. Speed?",                        label: 0 },
    { text: "A bus travels 200 km in 4 hours. How far in 7 hours?",              label: 0 },
    { text: "An airplane flies 1500 km in 3 hours. Speed?",                      label: 0 },
    { text: "A car travels at 90 km/h for 2 hours. How far does it go?",         label: 0 },
    { text: "Speed is 50 km/h and time is 4 hours. What is the distance?",       label: 0 },
    { text: "Distance 360 km time 6 hours. What is average speed?",              label: 0 },
    { text: "A runner covers 10 km in 1 hour. How far in 3.5 hours?",            label: 0 },

    // ── Work / Rate — label 1 ────────────────────────────────────────────────
    { text: "If 3 workers complete a job in 4 days how long for 6 workers?",     label: 1 },
    { text: "5 workers take 10 days to complete a project. Days for 2 workers?", label: 1 },
    { text: "8 workers can complete a job in 6 days. How long for 4 workers?",   label: 1 },
    { text: "12 workers finish work in 8 days. 6 workers will take?",            label: 1 },
    { text: "9 workers complete a task in 12 days. 3 workers take how long?",    label: 1 },
    { text: "4 machines produce 100 items in 5 hours. 2 machines take?",         label: 1 },
    { text: "6 painters paint a wall in 4 days. How long for 3 painters?",       label: 1 },
    { text: "10 workers take 6 days to build a wall. How long for 15 workers?",  label: 1 },
    { text: "7 workers can finish a task in 14 days. How long for 14 workers?",  label: 1 },
    { text: "2 machines finish a job in 8 hours. How long for 4 machines?",      label: 1 },

    // ── Percentage — label 2 ─────────────────────────────────────────────────
    { text: "What is 20 percent of 150?",                                         label: 2 },
    { text: "Find 15% of 200",                                                    label: 2 },
    { text: "What percentage is 25 of 200?",                                      label: 2 },
    { text: "Calculate 30% of 80",                                                label: 2 },
    { text: "What is 8% of 250?",                                                 label: 2 },
    { text: "45 is what percent of 90?",                                          label: 2 },
    { text: "What is 12.5 percent of 400?",                                       label: 2 },
    { text: "60 is what percentage of 240?",                                      label: 2 },
    { text: "Find the percentage: 18 out of 72",                                  label: 2 },
    { text: "What percent of 500 is 125?",                                        label: 2 },

    // ── Ratio — label 3 ──────────────────────────────────────────────────────
    { text: "The ratio of boys to girls is 3:4. If there are 28 girls how many boys?", label: 3 },
    { text: "Mix paint in ratio 2:5. If 10 liters of blue how much yellow?",           label: 3 },
    { text: "The ratio of apples to oranges is 2:3. If 15 oranges how many apples?",   label: 3 },
    { text: "In a class ratio of boys to girls is 5:7. 21 girls. Boys?",               label: 3 },
    { text: "Ratio of red to blue balls is 3:4. 20 blue balls. Red balls?",            label: 3 },
    { text: "Sugar and flour ratio 1:4. 200g flour. How much sugar?",                  label: 3 },
    { text: "Ratio of cats to dogs is 2:3. If 12 cats how many dogs?",                 label: 3 },
    { text: "A to B ratio is 5:3. If B is 15 find A.",                                 label: 3 },
    { text: "The ratio of men to women is 4:5. 25 women. How many men?",               label: 3 },
    { text: "Divide 120 in ratio 3:5. What is the larger part?",                       label: 3 }
];

const DEFAULT_CONFIG = {
    numLayers: 2,
    units: 32,
    learningRate: 0.003,
    activation: "relu",
    reasoningDepth: 5,
    batchSize: 8,
    epochs: 60
};
