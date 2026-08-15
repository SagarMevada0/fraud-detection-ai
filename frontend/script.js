// ---------------------------------------------------------------
// UDA7 Credit Card Fraud Detection - Frontend logic
//
// Talks to the FastAPI backend:
//   GET  /sample   -> fills the form with one random real transaction
//   POST /predict  -> returns fraud probability, decision and alert
// ---------------------------------------------------------------

// Backend base URL. Change this if you run the API on another port/host.
const API_URL = "http://127.0.0.1:8000";

const V_FEATURES = Array.from({ length: 28 }, (_, i) => `V${i + 1}`);
const ALL_FEATURES = ["Time", "Amount", ...V_FEATURES];

// Keep the ground-truth class of the last loaded sample (for demo comparison).
let lastSampleActualClass = null;

// ----- Build the V1-V28 inputs inside the "Advanced" section -----
const vGrid = document.getElementById("vGrid");
V_FEATURES.forEach((name) => {
  const wrapper = document.createElement("div");
  wrapper.className = "field";
  wrapper.innerHTML = `
    <label for="${name}">${name}</label>
    <input type="number" id="${name}" step="any" value="0" />
  `;
  vGrid.appendChild(wrapper);
});

// Show which API we are talking to in the footer.
document.getElementById("apiUrl").textContent = API_URL;

const statusEl = document.getElementById("status");
const loadSampleBtn = document.getElementById("loadSampleBtn");
const checkBtn = document.getElementById("checkBtn");

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.className = isError ? "status error" : "status";
}

// ----- Load sample transaction -----
loadSampleBtn.addEventListener("click", async () => {
  loadSampleBtn.disabled = true;
  setStatus("Loading a random sample transaction...");

  try {
    const response = await fetch(`${API_URL}/sample`);
    if (!response.ok) throw new Error(`Backend returned ${response.status}`);
    const data = await response.json();

    // Fill every input field with the sample's values.
    ALL_FEATURES.forEach((name) => {
      const input = document.getElementById(name);
      if (input && data.transaction[name] !== undefined) {
        input.value = data.transaction[name];
      }
    });

    lastSampleActualClass = data.actual_class;
    setStatus("Sample loaded. Click \u201CCheck transaction\u201D to analyse it.");
  } catch (err) {
    setStatus(
      `Could not reach the backend (${err.message}). ` +
      "Is the API running on port 8000?",
      true
    );
  } finally {
    loadSampleBtn.disabled = false;
  }
});

// ----- Check transaction -----
checkBtn.addEventListener("click", async () => {
  // Collect all 30 features from the form.
  const transaction = {};
  for (const name of ALL_FEATURES) {
    const value = document.getElementById(name).value;
    if (value === "") {
      setStatus(`Please fill in the "${name}" field (or load a sample).`, true);
      return;
    }
    transaction[name] = parseFloat(value);
  }

  checkBtn.disabled = true;
  setStatus("Analysing transaction...");

  try {
    const response = await fetch(`${API_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(transaction),
    });
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      throw new Error(detail.detail || `Backend returned ${response.status}`);
    }
    const result = await response.json();
    showResult(result);
    setStatus("");
  } catch (err) {
    setStatus(`Prediction failed: ${err.message}`, true);
  } finally {
    checkBtn.disabled = false;
  }
});

// ----- Display the prediction result -----
function showResult(result) {
  const card = document.getElementById("resultCard");
  const badge = document.getElementById("resultBadge");
  const alertMessage = document.getElementById("alertMessage");
  const probText = document.getElementById("probText");
  const gaugeFill = document.getElementById("gaugeFill");
  const thresholdMarker = document.getElementById("thresholdMarker");
  const thresholdNote = document.getElementById("thresholdNote");
  const groundTruth = document.getElementById("groundTruth");

  const probabilityPct = (result.fraud_probability * 100).toFixed(2);
  const thresholdPct = (result.threshold * 100).toFixed(0);
  const isFraud = result.decision === "FRAUD";

  badge.textContent = isFraud ? "🚨 FRAUD ALERT" : "✅ LEGITIMATE";
  badge.className = `badge ${isFraud ? "fraud" : "legit"}`;
  alertMessage.textContent = result.alert;

  probText.textContent = `${probabilityPct}%`;
  gaugeFill.style.width = `${probabilityPct}%`;
  thresholdMarker.style.left = `${thresholdPct}%`;

  thresholdNote.textContent =
    `Alert threshold: ${thresholdPct}% \u2014 transactions with a fraud ` +
    `probability at or above this value trigger a fraud alert.`;

  // If the values came from "Load sample", show the dataset's true label.
  if (lastSampleActualClass === 0 || lastSampleActualClass === 1) {
    const label = lastSampleActualClass === 1 ? "FRAUD" : "LEGITIMATE";
    groundTruth.textContent =
      `Ground truth for the loaded sample (from the dataset): ${label}.`;
  } else {
    groundTruth.textContent = "";
  }

  card.classList.remove("hidden");
  card.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// Editing any field manually invalidates the stored ground-truth label.
document.addEventListener("input", (event) => {
  if (event.target.tagName === "INPUT") lastSampleActualClass = null;
});
