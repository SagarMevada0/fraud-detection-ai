// ---------------------------------------------------------------
// UDA7 Credit Card Fraud Detection - Results / Model Evaluation page
//
// Fetches GET /results from the backend and renders:
//   - metric cards (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC)
//   - the confusion matrix as a table
//   - the evaluation plots (confusion matrix heatmap, ROC curve,
//     PR curve, threshold analysis, feature importance)
// ---------------------------------------------------------------

// Same API base URL detection as script.js.
const API_URL = (() => {
  const { protocol, hostname, port } = window.location;
  if (protocol === "file:") return "http://127.0.0.1:8000";
  if (port && port !== "8000" && ["localhost", "127.0.0.1"].includes(hostname)) {
    return `${protocol}//${hostname}:8000`;
  }
  return ""; // same origin as the page
})();

document.getElementById("apiUrl").textContent =
  API_URL || `${window.location.origin} (same origin)`;

const METRIC_CARDS = [
  { key: "Accuracy", label: "Accuracy", hint: "Correct predictions overall" },
  { key: "Precision", label: "Precision", hint: "Flagged frauds that are real" },
  { key: "Recall", label: "Recall", hint: "Real frauds that were caught" },
  { key: "F1", label: "F1 Score", hint: "Balance of precision & recall" },
  { key: "ROC_AUC", label: "ROC-AUC", hint: "Ranking quality (all thresholds)" },
  { key: "PR_AUC", label: "PR-AUC", hint: "Precision-recall trade-off quality" },
];

async function loadResults() {
  const statusEl = document.getElementById("resultsStatus");
  try {
    const response = await fetch(`${API_URL}/results`);
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      throw new Error(detail.detail || `Backend returned ${response.status}`);
    }
    render(await response.json());
    statusEl.textContent = "";
  } catch (err) {
    document.getElementById("modelInfo").textContent = "";
    statusEl.textContent =
      `Could not load results (${err.message}). Is the API running?`;
    statusEl.className = "status error";
  }
}

function render(data) {
  const m = data.test_metrics;

  document.getElementById("modelInfo").textContent =
    `Model: ${data.model_name} · Alert threshold: ` +
    `${(data.threshold * 100).toFixed(0)}% · Evaluated on ` +
    `${data.test_set_size.toLocaleString()} unseen test transactions ` +
    `(${data.test_fraud_count} frauds).`;

  // Metric cards
  const grid = document.getElementById("metricsGrid");
  grid.innerHTML = "";
  METRIC_CARDS.forEach(({ key, label, hint }) => {
    const card = document.createElement("div");
    card.className = "metric-card";
    card.innerHTML = `
      <div class="metric-value">${(m[key] * 100).toFixed(2)}%</div>
      <div class="metric-label">${label}</div>
      <div class="metric-hint">${hint}</div>
    `;
    grid.appendChild(card);
  });

  // Confusion matrix table
  const cm = data.confusion_matrix;
  document.getElementById("cmTN").textContent = cm.true_negatives.toLocaleString();
  document.getElementById("cmFP").textContent = cm.false_positives.toLocaleString();
  document.getElementById("cmFN").textContent = cm.false_negatives.toLocaleString();
  document.getElementById("cmTP").textContent = cm.true_positives.toLocaleString();
  document.getElementById("cmNote").textContent =
    `Out of ${data.test_fraud_count} real frauds, the model caught ` +
    `${cm.true_positives} and missed ${cm.false_negatives}, while raising ` +
    `only ${cm.false_positives} false alarms on legitimate transactions.`;
  document.getElementById("cmCard").classList.remove("hidden");

  // Plots
  const plotsGrid = document.getElementById("plotsGrid");
  plotsGrid.innerHTML = "";
  (data.images || []).forEach(({ title, url }) => {
    const fig = document.createElement("figure");
    fig.className = "plot";
    fig.innerHTML = `
      <img src="${API_URL}${url}" alt="${title}" loading="lazy" />
      <figcaption>${title}</figcaption>
    `;
    plotsGrid.appendChild(fig);
  });
  document.getElementById("plotsCard").classList.remove("hidden");
}

loadResults();
