/**
 * DeepProtein Studio - Client Logic
 * Handles interactive validation, preset loading, prediction calling, and dynamic rendering.
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const sequenceInput = document.getElementById("sequenceInput");
  const sequenceCounter = document.getElementById("sequenceCounter");
  const validationWarning = document.getElementById("validationWarning");
  const validationWarningText = document.getElementById("validationWarningText");
  
  const presetButtons = document.getElementById("presetButtons");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const clearInputBtn = document.getElementById("clearInputBtn");
  const analyzeSpinner = document.getElementById("analyzeSpinner");

  const modeCalibratedBtn = document.getElementById("modeCalibratedBtn");
  const modeCustomBtn = document.getElementById("modeCustomBtn");
  const customThresholdGroup = document.getElementById("customThresholdGroup");
  const thresholdSlider = document.getElementById("thresholdSlider");
  const sliderValue = document.getElementById("sliderValue");

  const resultsSection = document.getElementById("resultsSection");
  const statLength = document.getElementById("statLength");
  const statMW = document.getElementById("statMW");
  const statPI = document.getElementById("statPI");
  const statGRAVY = document.getElementById("statGRAVY");

  const heatmapContainer = document.getElementById("heatmapContainer");
  const predictionsList = document.getElementById("predictionsList");
  const domainTabs = document.getElementById("domainTabs");
  const filterInput = document.getElementById("filterInput");
  const showActiveOnlyCheckbox = document.getElementById("showActiveOnlyCheckbox");
  const summaryActiveCount = document.getElementById("summaryActiveCount");

  const countAll = document.getElementById("countAll");
  const countMF = document.getElementById("countMF");
  const countBP = document.getElementById("countBP");
  const countCC = document.getElementById("countCC");

  const exportCsvBtn = document.getElementById("exportCsvBtn");
  const copyJsonBtn = document.getElementById("copyJsonBtn");

  // State
  let currentPredictions = [];
  let currentAttentionWeights = [];
  let currentCleanSequence = "";
  let currentMode = "calibrated";
  let activeTab = "all";
  let presetSamples = [];

  const VALID_AMINO_ACIDS = new Set("ACDEFGHIKLMNPQRSTVWY");

  // ==========================================
  // 1. SEQUENCE INPUT VALIDATION & COUNTER
  // ==========================================
  function parseAndCleanSequence(rawText) {
    const lines = rawText.trim().split("\n");
    let seq = "";
    if (lines.length > 0 && lines[0].startsWith(">")) {
      seq = lines.slice(1).join("").toUpperCase().replace(/\s+/g, "");
    } else {
      seq = lines.join("").toUpperCase().replace(/\s+/g, "");
    }
    return seq;
  }

  function validateInput() {
    const raw = sequenceInput.value;
    const clean = parseAndCleanSequence(raw);
    currentCleanSequence = clean;
    sequenceCounter.textContent = `${clean.length} / 1000 Residues`;

    if (clean.length === 0) {
      validationWarning.style.display = "none";
      sequenceInput.style.borderColor = "var(--bg-glass-border)";
      return true;
    }

    const invalidLetters = [];
    for (const char of clean) {
      if (!VALID_AMINO_ACIDS.has(char)) {
        invalidLetters.push(char);
      }
    }

    if (invalidLetters.length > 0) {
      const uniqueInvalid = Array.from(new Set(invalidLetters)).join(", ");
      validationWarningText.textContent = `Invalid amino acid characters detected: [ ${uniqueInvalid} ]. Standard 20 amino acids allowed only.`;
      validationWarning.style.display = "flex";
      sequenceInput.style.borderColor = "var(--accent-red)";
      return false;
    } else {
      validationWarning.style.display = "none";
      sequenceInput.style.borderColor = "var(--primary-cyan)";
      return true;
    }
  }

  sequenceInput.addEventListener("input", validateInput);

  clearInputBtn.addEventListener("click", () => {
    sequenceInput.value = "";
    validateInput();
    resultsSection.style.display = "none";
  });

  // ==========================================
  // 2. PRESET SAMPLES LOADER
  // ==========================================
  const FALLBACK_PRESETS = [
    {
      id: "INSULIN",
      name: "Human Insulin (P01308)",
      desc: "Key hormone regulating carbohydrate and lipid metabolism. Binds insulin receptor.",
      sequence: "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN"
    },
    {
      id: "HBB",
      name: "Hemoglobin Subunit Beta (P68871)",
      desc: "Iron-containing oxygen transport metalloprotein in erythrocytes. Contains heme cofactor binding pockets.",
      sequence: "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH"
    },
    {
      id: "P53",
      name: "Cellular Tumor Antigen p53 (P04637)",
      desc: "Master sequence-specific DNA-binding transcription factor and tumor suppressor regulating cell cycle arrest and apoptosis.",
      sequence: "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD"
    },
    {
      id: "CYP3A4",
      name: "Cytochrome P450 3A4 (P08684)",
      desc: "Major endoplasmic reticulum monooxygenase metabolizing approximately 50% of therapeutic drugs.",
      sequence: "MALIPDLAMETWLLLAVSLVLLYLYGTHSHGLFKKLGIPGPTPLPFLGNILSYHKGFCMFDMECHKKYGKVWGFYDGQQPVLAITDPDMIKTVLVKECYSVFTNRRPFGPVGFMKSAISIAEDEEWKRLRSLLSPTFTSGKLKEMVPIIAQYGDVLVRNLRREAETGKPVTLKDVFGAYSMDVITSTSFGVNIDSLNNPQDPFVENTKKLLRFDFLDPFFLSITVFPFLIPILEVLNICVFPREVTNFLRKSVKRMKESRLEDTQKHRVDFLQLMIDSQNSKETESHKALSDLELMAQSIIFIFAGYETTSSVLSFIMYELATHPDVQQKLQEEIDAVLPNKAPPTYDTVLQMEYLDMVVNETLRLFPIAMRLERVCKKDVEINGMFIPKGVVVMIPSYALHRDPKYWTEPEKFLPERFSKKNKDNIDPYIYTPFGSGPRNCIGMRFALMNMKLALIRVLQNFSFKPCKETQIPLKLSLGGLLQPEKPVVLKVESRDGTVSGA"
    },
    {
      id: "RHODOPSIN",
      name: "Rhodopsin (P02699)",
      desc: "7-transmembrane G-protein coupled photoreceptor converting photons into visual neurochemical signals.",
      sequence: "MNGTEGPNFYVPFSNKTGVVRSPFEAPQYYLAEPWQFSMLAAYMFLLIMLGFPINFLTLYVTVQHKKLRTPLNYILLNLAVADLFMVFGGFTTTLYTSLHGYFVFGPTGCNLEGFFATLGGEIALWSLVVLAIERYVVVCKPMSNFRFGENHAIMGVAFTWVMALACAAPPLVGWSRYIPEGMQCSCGIDYYTPHEETNNESFVIYMFVVHFIIPLIVIFFCYGQLVFTVKEAAAQQQESATTQKAEKEVTRMVIIMVIAFLICWLPYAGVAFYIFTHQGSDFGPIFMTIPAFFAKTSAVYNPVIYIMMNKQFRNCMVTTLCCGKNPLGDDEASTTVSKTETSQVAPA"
    }
  ];

  function renderPresetPills(samples) {
    presetButtons.innerHTML = "";
    samples.forEach(sample => {
      const btn = document.createElement("button");
      btn.className = "preset-pill";
      btn.textContent = `⚡ ${sample.name.split(" ")[0]} (${sample.id})`;
      btn.title = sample.desc;
      btn.addEventListener("click", () => {
        sequenceInput.value = `>${sample.id} | ${sample.name}\n${sample.sequence}`;
        validateInput();
        triggerPrediction();
      });
      presetButtons.appendChild(btn);
    });
  }

  async function loadPresetSamples() {
    try {
      const res = await fetch("/api/samples");
      if (res.ok) {
        presetSamples = await res.json();
        renderPresetPills(presetSamples);
        return;
      }
    } catch (err) {
      console.warn("Could not reach /api/samples backend, using embedded fallback presets.", err);
    }
    presetSamples = FALLBACK_PRESETS;
    renderPresetPills(presetSamples);
  }
  loadPresetSamples();

  // ==========================================
  // 3. INFERENCE MODE & SLIDER TOGGLE
  // ==========================================
  modeCalibratedBtn.addEventListener("click", () => {
    currentMode = "calibrated";
    modeCalibratedBtn.classList.add("active");
    modeCustomBtn.classList.remove("active");
    customThresholdGroup.style.display = "none";
    if (currentPredictions.length > 0) triggerPrediction();
  });

  modeCustomBtn.addEventListener("click", () => {
    currentMode = "custom";
    modeCustomBtn.classList.add("active");
    modeCalibratedBtn.classList.remove("active");
    customThresholdGroup.style.display = "block";
    if (currentPredictions.length > 0) triggerPrediction();
  });

  thresholdSlider.addEventListener("input", (e) => {
    sliderValue.textContent = parseFloat(e.target.value).toFixed(2);
    if (currentPredictions.length > 0) {
      updateActiveStatuses(parseFloat(e.target.value));
      renderPredictions();
    }
  });

  function updateActiveStatuses(customVal) {
    currentPredictions.forEach(p => {
      if (currentMode === "calibrated") {
        p.is_active = (p.probability >= p.threshold);
      } else {
        p.is_active = (p.probability >= customVal);
      }
    });
  }

  // ==========================================
  // 4. API PREDICTION CALL
  // ==========================================
  async function triggerPrediction() {
    const isValid = validateInput();
    if (!isValid || currentCleanSequence.length === 0) {
      alert("Please provide a valid protein sequence containing standard 20 amino acids.");
      return;
    }

    // Set UI loading state
    analyzeBtn.disabled = true;
    analyzeSpinner.style.display = "block";

    try {
      const payload = {
        sequence: sequenceInput.value,
        mode: currentMode,
        custom_threshold: parseFloat(thresholdSlider.value)
      };

      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || "Prediction request failed.");
      }

      const data = await res.json();
      currentPredictions = data.predictions;
      currentAttentionWeights = data.attention_weights;

      // Populate Biophysical Stats
      statLength.textContent = `${data.biophysics.length} AA`;
      statMW.textContent = `${data.biophysics.molecular_weight_kda} kDa`;
      statPI.textContent = `${data.biophysics.isoelectric_point}`;
      statGRAVY.textContent = `${data.biophysics.gravy_hydropathy}`;

      // Render Attention Heatmap & Functional Annotations
      renderAttentionHeatmap(data.cleaned_sequence, data.attention_weights);
      updateDomainCounts(data.predictions);
      renderPredictions();

      // Show results
      resultsSection.style.display = "block";
      resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });

    } catch (err) {
      alert(`Prediction Error: ${err.message}`);
    } finally {
      analyzeBtn.disabled = false;
      analyzeSpinner.style.display = "none";
    }
  }

  analyzeBtn.addEventListener("click", triggerPrediction);

  // ==========================================
  // 5. ATTENTION HEATMAP RENDERER
  // ==========================================
  function renderAttentionHeatmap(sequence, weights) {
    heatmapContainer.innerHTML = "";
    if (!weights || weights.length === 0) return;

    const minW = Math.min(...weights);
    const maxW = Math.max(...weights);
    const range = (maxW - minW) || 1e-6;

    const fragment = document.createDocumentFragment();

    sequence.split("").forEach((aa, idx) => {
      const w = weights[idx] || minW;
      const norm = (w - minW) / range; // 0.0 to 1.0

      const tile = document.createElement("div");
      tile.className = "aa-tile";
      tile.textContent = aa;

      // Dynamic heatmap color interpolation
      // Low (0.0): Deep Slate (#1e293b) -> Mid (0.5): Cyan/Emerald -> High (1.0): Golden Yellow
      let bg = "";
      if (norm < 0.35) {
        bg = `rgba(30, 41, 59, ${0.4 + norm * 0.8})`;
      } else if (norm < 0.70) {
        bg = `rgba(6, 182, 212, ${0.4 + (norm - 0.35) * 1.2})`;
      } else {
        bg = `rgba(245, 158, 11, ${0.6 + (norm - 0.70) * 1.3})`;
      }

      tile.style.backgroundColor = bg;
      tile.style.color = (norm > 0.6) ? "#000000" : "#FFFFFF";
      tile.title = `Residue #${idx + 1}: ${aa} | Attention Focus: ${(w * 100).toFixed(3)}%`;

      fragment.appendChild(tile);
    });

    heatmapContainer.appendChild(fragment);
  }

  // ==========================================
  // 6. FUNCTIONAL ANNOTATIONS RENDERER
  // ==========================================
  function updateDomainCounts(predictions) {
    let mf = 0, bp = 0, cc = 0;
    predictions.forEach(p => {
      if (p.domain === "MF") mf++;
      if (p.domain === "BP") bp++;
      if (p.domain === "CC") cc++;
    });
    countAll.textContent = predictions.length;
    countMF.textContent = mf;
    countBP.textContent = bp;
    countCC.textContent = cc;
  }

  function renderPredictions() {
    predictionsList.innerHTML = "";
    const filterText = filterInput.value.toLowerCase().trim();
    const activeOnly = showActiveOnlyCheckbox.checked;

    const filtered = currentPredictions.filter(p => {
      // Tab filter
      if (activeTab !== "all" && p.domain !== activeTab) return false;
      // Active only filter
      if (activeOnly && !p.is_active) return false;
      // Search filter
      if (filterText) {
        const matchesName = p.name.toLowerCase().includes(filterText);
        const matchesId = p.go_id.toLowerCase().includes(filterText);
        const matchesDesc = p.description.toLowerCase().includes(filterText);
        if (!matchesName && !matchesId && !matchesDesc) return false;
      }
      return true;
    });

    summaryActiveCount.textContent = `Showing ${filtered.length} terms (${filtered.filter(p => p.is_active).length} Active)`;

    if (filtered.length === 0) {
      predictionsList.innerHTML = `
        <div style="text-align: center; padding: 40px; color: var(--text-muted);">
          No Gene Ontology terms match your current filter settings.
        </div>
      `;
      return;
    }

    const fragment = document.createDocumentFragment();

    filtered.forEach(p => {
      const card = document.createElement("div");
      card.className = `prediction-card ${p.is_active ? 'is-active' : ''}`;

      const probPct = (p.probability * 100).toFixed(1);
      const threshPct = (p.threshold * 100).toFixed(1);
      const domainPillClass = `pill-${p.domain.toLowerCase()}`;

      card.innerHTML = `
        <div class="pred-meta">
          <span class="domain-pill ${domainPillClass}">${p.domain_name}</span>
          <a href="${p.quickgo_url}" target="_blank" rel="noopener noreferrer" class="go-link" title="Open QuickGO Database">
            ${p.go_id} &nearr;
          </a>
        </div>
        
        <div class="pred-details">
          <div class="term-name">${p.name}</div>
          <div class="term-desc">${p.description}</div>
          <div class="progress-container">
            <div class="progress-track" title="Confidence: ${probPct}% | Decision Cutoff: ${threshPct}%">
              <div class="progress-fill ${p.is_active ? '' : 'below-thresh'}" style="width: ${probPct}%;"></div>
              <div class="threshold-marker" style="left: ${threshPct}%;" title="Optimal Cutoff: ${threshPct}%"></div>
            </div>
          </div>
        </div>

        <div class="pred-status">
          <span class="prob-badge">${probPct}%</span>
          <span class="status-badge ${p.is_active ? 'status-active' : 'status-inactive'}">
            ${p.is_active ? '● Active' : 'Below Cutoff'}
          </span>
        </div>
      `;

      fragment.appendChild(card);
    });

    predictionsList.appendChild(fragment);
  }

  // Filter Event Listeners
  domainTabs.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      domainTabs.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      activeTab = btn.getAttribute("data-tab");
      renderPredictions();
    });
  });

  filterInput.addEventListener("input", renderPredictions);
  showActiveOnlyCheckbox.addEventListener("change", renderPredictions);

  // ==========================================
  // 7. EXPORT CONTROLS
  // ==========================================
  exportCsvBtn.addEventListener("click", () => {
    if (currentPredictions.length === 0) return;
    let csv = "GO_ID,Term_Name,Domain,Probability,Threshold,Is_Active,QuickGO_URL\n";
    currentPredictions.forEach(p => {
      csv += `"${p.go_id}","${p.name.replace(/"/g, '""')}","${p.domain}",${p.probability},${p.threshold},${p.is_active},"${p.quickgo_url}"\n`;
    });

    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `DeepProtein_Predictions_${new Date().toISOString().slice(0, 10)}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  });

  copyJsonBtn.addEventListener("click", () => {
    if (currentPredictions.length === 0) return;
    const jsonStr = JSON.stringify(currentPredictions, null, 2);
    navigator.clipboard.writeText(jsonStr).then(() => {
      const originalText = copyJsonBtn.textContent;
      copyJsonBtn.textContent = "✓ Copied JSON!";
      setTimeout(() => copyJsonBtn.textContent = originalText, 2000);
    });
  });

  // Smooth scroll to documentation
  document.getElementById("viewDocsBtn").addEventListener("click", (e) => {
    e.preventDefault();
    document.getElementById("docs").scrollIntoView({ behavior: "smooth" });
  });
});
