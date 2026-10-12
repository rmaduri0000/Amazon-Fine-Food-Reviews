/**
 * FineFoods AI — Batch & Product Health Analytics Controller (Page 2)
 * Features:
 * - CSV File Upload & Parsing
 * - Paste Review Input & Counter
 * - Configurable Negative Defect Alert Threshold
 * - Preliminary Result Sample Size Warnings (N < 10)
 * - Sentiment Distribution Visual Charts
 * - Dynamic Review-by-Review Filtering & Search
 * - CSV Export Download
 * - 1-Click Redirect to Live Sentiment Studio
 */

let currentBatchData = null;
let filteredReviews = [];

document.addEventListener('DOMContentLoaded', () => {
  initThemeToggle();
  initModeTabs();
  initBatchPresets();
  initCsvUpload();
  initBatchAnalyzer();
  initTableFilters();
  initCsvExport();
  initClearBatch();
});

const batchPresets = {
  coffee: {
    name: "☕ Colombian Roast Coffee (QA Demo)",
    reviews: [
      "This Colombian roast has an incredible aroma and rich nutty taste. Not bitter at all, perfectly smooth and flavorful.",
      "Five stars! The whole beans are fresh with a lovely oily sheen, grinds evenly in my burr grinder.",
      "Best morning brew I have discovered this year. Delicious dark cocoa notes.",
      "Smooth and rich, pairs wonderfully with oat milk. Will order another bag.",
      "Great value gourmet roast, arrived quickly and sealed tightly."
    ]
  },
  chips: {
    name: "🥨 Kettle Chips (Mixed Feedback Demo)",
    reviews: [
      "Crunchy and flavorful salt & vinegar chips, delicious snack!",
      "Decent flavor, but several chips were broken into crumbs during shipping.",
      "Completely stale and expired. Bag arrived punctured and smelled like cardboard.",
      "Too salty for my liking, left an intense burn on the tongue.",
      "Kids loved them, good portion size for lunchboxes."
    ]
  },
  tea: {
    name: "🍵 Organic Matcha Tea (High Satisfaction Demo)",
    reviews: [
      "Five stars! High quality organic Japanese matcha powder. Beautiful vibrant green color.",
      "Crisp, clean taste with zero bitterness. Dissolves easily in warm water.",
      "Authentic ceremonial grade matcha at a reasonable price. Highly recommended.",
      "Great packaging preserves the delicate aroma and freshness.",
      "My daily morning energizer! Smooth, refreshing, and delicious."
    ]
  },
  pet: {
    name: "🐕 Dog Food (Defect Alert QA Demo)",
    reviews: [
      "Beware! My dog refused to eat this after the first bite, and vomited within an hour.",
      "Formula has clearly changed, food arrived with a foul rancid smell. Very bad batch.",
      "Caused severe allergic reaction and diarrhea. Threw the bag straight into the garbage.",
      "Disappointed with the quality control. Returning immediately.",
      "Not the same recipe as before. Pet refused to touch the bowl."
    ]
  }
};

function initThemeToggle() {
  const toggle = document.getElementById('ios-theme-toggle');
  const savedTheme = localStorage.getItem('finefoods_theme') || 'light';
  document.documentElement.setAttribute('data-theme', savedTheme);

  if (toggle) {
    toggle.checked = (savedTheme === 'dark');
    toggle.addEventListener('change', () => {
      const next = toggle.checked ? 'dark' : 'light';
      document.documentElement.setAttribute('data-theme', next);
      localStorage.setItem('finefoods_theme', next);
    });
  }
}

function initModeTabs() {
  const tabPaste = document.getElementById('tab-mode-paste');
  const tabUpload = document.getElementById('tab-mode-upload');
  const panePaste = document.getElementById('pane-paste-mode');
  const paneUpload = document.getElementById('pane-upload-mode');

  if (tabPaste && tabUpload && panePaste && paneUpload) {
    tabPaste.addEventListener('click', () => {
      tabPaste.classList.add('active');
      tabUpload.classList.remove('active');
      panePaste.style.display = 'block';
      paneUpload.style.display = 'none';
    });

    tabUpload.addEventListener('click', () => {
      tabUpload.classList.add('active');
      tabPaste.classList.remove('active');
      paneUpload.style.display = 'block';
      panePaste.style.display = 'none';
    });
  }
}

function initBatchPresets() {
  const textarea = document.getElementById('batch-textarea-main');
  const setupBtn = (btnId, key) => {
    const btn = document.getElementById(btnId);
    if (btn && textarea) {
      btn.addEventListener('click', () => {
        const preset = batchPresets[key];
        textarea.value = preset.reviews.join('\n');
        updateBatchCount();
        runBatchExecution(preset.name);
      });
    }
  };

  setupBtn('btn-batch-coffee', 'coffee');
  setupBtn('btn-batch-chips', 'chips');
  setupBtn('btn-batch-tea', 'tea');
  setupBtn('btn-batch-pet', 'pet');
}

function initCsvUpload() {
  const dropzone = document.getElementById('csv-dropzone');
  const fileInput = document.getElementById('batch-file-input');
  const btnBrowse = document.getElementById('btn-browse-csv');
  const statusDiv = document.getElementById('upload-file-status');
  const textarea = document.getElementById('batch-textarea-main');

  if (btnBrowse && fileInput) {
    btnBrowse.addEventListener('click', (e) => {
      e.stopPropagation();
      fileInput.click();
    });
  }

  if (dropzone && fileInput) {
    dropzone.addEventListener('click', () => fileInput.click());
    dropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropzone.style.borderColor = '#3b82f6';
      dropzone.style.background = 'rgba(59, 130, 246, 0.08)';
    });
    dropzone.addEventListener('dragleave', () => {
      dropzone.style.borderColor = 'var(--border-glass)';
      dropzone.style.background = 'rgba(255, 255, 255, 0.03)';
    });
    dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.style.borderColor = 'var(--border-glass)';
      dropzone.style.background = 'rgba(255, 255, 255, 0.03)';
      if (e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener('change', () => {
      if (fileInput.files.length) {
        handleFile(fileInput.files[0]);
      }
    });
  }

  function handleFile(file) {
    if (!file.name.endsWith('.csv')) {
      alert('Please select a valid CSV file.');
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target.result;
      const lines = text.split(/\r?\n/).filter(l => l.trim().length > 0);
      if (lines.length <= 1) {
        alert('CSV file appears to be empty.');
        return;
      }

      // Parse headers
      const headers = lines[0].split(',').map(h => h.trim().replace(/^["']|["']$/g, '').toLowerCase());
      let colIdx = headers.findIndex(h => h === 'text' || h === 'review' || h === 'review_text' || h === 'body');
      if (colIdx === -1) {
        colIdx = headers.findIndex(h => h === 'summary');
      }
      if (colIdx === -1) {
        colIdx = 0; // fallback to first column
      }

      const extracted = [];
      for (let i = 1; i < lines.length; i++) {
        const line = lines[i];
        if (!line.trim()) continue;
        // Simple CSV quote handler
        const parts = line.split(',');
        const val = parts.length > colIdx ? parts[colIdx].replace(/^["']|["']$/g, '').trim() : line.trim();
        if (val.length > 5) {
          extracted.push(val);
        }
        if (extracted.length >= 250) break; // cap for client batch analysis
      }

      if (extracted.length > 0) {
        textarea.value = extracted.join('\n');
        updateBatchCount();
        if (statusDiv) {
          statusDiv.style.display = 'block';
          statusDiv.textContent = `✓ Loaded ${extracted.length} reviews from "${file.name}"`;
        }
        runBatchExecution(`Uploaded CSV: ${file.name}`);
      } else {
        alert('Could not extract review text from CSV. Ensure a Text or Summary column is present.');
      }
    };
    reader.readAsText(file);
  }
}

function initBatchAnalyzer() {
  const textarea = document.getElementById('batch-textarea-main');
  const btnRun = document.getElementById('btn-run-batch');

  if (textarea) {
    textarea.addEventListener('input', updateBatchCount);
  }

  if (btnRun) {
    btnRun.addEventListener('click', () => {
      runBatchExecution();
    });
  }
}

function initClearBatch() {
  const btnClear = document.getElementById('btn-clear-batch');
  const textarea = document.getElementById('batch-textarea-main');
  const resultsArea = document.getElementById('batch-results-area');
  const statusDiv = document.getElementById('upload-file-status');

  if (btnClear) {
    btnClear.addEventListener('click', () => {
      if (textarea) textarea.value = '';
      updateBatchCount();
      if (resultsArea) resultsArea.style.display = 'none';
      if (statusDiv) statusDiv.style.display = 'none';
      currentBatchData = null;
    });
  }
}

function updateBatchCount() {
  const textarea = document.getElementById('batch-textarea-main');
  const counter = document.getElementById('batch-reviews-counter');
  if (textarea && counter) {
    const lines = textarea.value.split('\n').filter(l => l.trim().length > 0);
    counter.textContent = `${lines.length} review${lines.length === 1 ? '' : 's'} loaded`;
  }
}

async function runBatchExecution(productNameOverride = null) {
  const textarea = document.getElementById('batch-textarea-main');
  const modelSelect = document.getElementById('batch-model-select');
  const thresholdInput = document.getElementById('batch-threshold-input');
  const btnRun = document.getElementById('btn-run-batch');
  const resultsArea = document.getElementById('batch-results-area');

  const text = textarea ? textarea.value.trim() : '';
  if (!text) {
    alert('Please enter or upload customer reviews first.');
    return;
  }

  const reviewsList = text.split('\n')
    .map(r => r.trim())
    .filter(r => r.length > 0);

  if (reviewsList.length === 0) {
    alert('No non-empty review lines detected.');
    return;
  }

  const selectedModel = modelSelect ? modelSelect.value : 'ensemble_avg';
  const threshold = thresholdInput ? parseFloat(thresholdInput.value) || 25.0 : 25.0;
  const productName = productNameOverride || "Custom Review Batch";

  if (btnRun) {
    btnRun.disabled = true;
    const btnSpan = btnRun.querySelector('.btn-text');
    if (btnSpan) btnSpan.textContent = 'Evaluating Batch...';
  }

  try {
    const res = await fetch('/api/predict-batch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        reviews: reviewsList,
        model: selectedModel,
        threshold: threshold,
        product_name: productName
      })
    });

    const json = await res.json();
    if (json.status === 'success') {
      currentBatchData = json.data;
      renderBatchResults(json.data);
      if (resultsArea) {
        resultsArea.style.display = 'block';
        resultsArea.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
    } else {
      alert('Batch evaluation error: ' + json.message);
    }
  } catch (err) {
    console.error('Batch error:', err);
    alert('Server communication error during batch evaluation.');
  } finally {
    if (btnRun) {
      btnRun.disabled = false;
      const btnSpan = btnRun.querySelector('.btn-text');
      if (btnSpan) btnSpan.textContent = 'Check All Reviews';
    }
  }
}

function renderBatchResults(data) {
  // Banner metrics
  const banner = document.getElementById('batch-health-banner');
  const emoji = document.getElementById('batch-health-emoji');
  const verdict = document.getElementById('batch-health-verdict');
  const productTag = document.getElementById('batch-product-tag');
  const samplePill = document.getElementById('batch-sample-pill');
  const alertMsg = document.getElementById('batch-alert-message');

  const mTotal = document.getElementById('batch-metric-total');
  const mPos = document.getElementById('batch-metric-pos');
  const mNeg = document.getElementById('batch-metric-neg');
  const mLat = document.getElementById('batch-metric-lat');

  if (productTag) productTag.textContent = data.product_name || 'Batch Customer Satisfaction';
  if (emoji) emoji.textContent = data.satisfaction_emoji || '📊';

  // Sample size indicator
  if (samplePill) {
    samplePill.style.display = data.is_preliminary ? 'inline-flex' : 'none';
  }

  // Defect crisis alert styling
  if (banner) {
    banner.classList.toggle('negative', data.defect_alert);
  }

  if (verdict) {
    if (data.defect_alert) {
      verdict.textContent = `🚨 Quality Alert: ${data.negative_pct}% Unhappy (Exceeds ${data.threshold_used_pct}% Threshold)`;
    } else {
      verdict.textContent = `${data.satisfaction_status} (${data.positive_pct}% Happy)`;
    }
  }

  if (alertMsg) {
    if (data.defect_alert) {
      alertMsg.textContent = `High proportion of dissatisfied customers detected in this batch. Immediate quality review recommended.`;
      alertMsg.style.color = '#ef4444';
    } else {
      alertMsg.textContent = `Customer feedback polarity is healthy and conforms to quality expectations.`;
      alertMsg.style.color = '#10b981';
    }
  }

  if (mTotal) mTotal.textContent = data.total_analyzed;
  if (mPos) mPos.textContent = `${data.positive_count} (${data.positive_pct}%)`;
  if (mNeg) mNeg.textContent = `${data.negative_count} (${data.negative_pct}%)`;
  if (mLat) mLat.textContent = `${data.avg_latency_ms} ms`;

  // Sentiment Distribution Progress Gauge
  const barPos = document.getElementById('batch-bar-pos');
  const barNeg = document.getElementById('batch-bar-neg');
  const chartPosPct = document.getElementById('chart-pos-pct');
  const chartPosCount = document.getElementById('chart-pos-count');
  const chartNegPct = document.getElementById('chart-neg-pct');
  const chartNegCount = document.getElementById('chart-neg-count');
  const threshMarker = document.getElementById('batch-threshold-marker');

  if (chartPosPct) chartPosPct.textContent = `${data.positive_pct}%`;
  if (chartPosCount) chartPosCount.textContent = data.positive_count;
  if (chartNegPct) chartNegPct.textContent = `${data.negative_pct}%`;
  if (chartNegCount) chartNegCount.textContent = data.negative_count;
  if (threshMarker) threshMarker.textContent = `Alert Trigger Threshold: ${data.threshold_used_pct}% Unhappy`;

  if (barPos && barNeg) {
    barPos.style.width = `${data.positive_pct}%`;
    barNeg.style.width = `${data.negative_pct}%`;
  }

  // Populate Table
  filteredReviews = [...data.reviews];
  renderTableRows();
}

function initTableFilters() {
  const searchInput = document.getElementById('batch-search-input');
  const filterSelect = document.getElementById('batch-filter-sentiment');

  const applyFilters = () => {
    if (!currentBatchData) return;
    const query = searchInput ? searchInput.value.toLowerCase().trim() : '';
    const sentFilter = filterSelect ? filterSelect.value : 'all';

    filteredReviews = currentBatchData.reviews.filter(item => {
      const matchText = !query || item.full_text.toLowerCase().includes(query);
      const matchSent = sentFilter === 'all' ||
        (sentFilter === 'positive' && item.prediction === 'Positive') ||
        (sentFilter === 'negative' && item.prediction === 'Negative');
      return matchText && matchSent;
    });

    renderTableRows();
  };

  if (searchInput) searchInput.addEventListener('input', applyFilters);
  if (filterSelect) filterSelect.addEventListener('change', applyFilters);
}

function renderTableRows() {
  const tbody = document.getElementById('batch-table-body');
  const countSpan = document.getElementById('batch-table-count');

  if (countSpan && currentBatchData) {
    countSpan.textContent = `Showing ${filteredReviews.length} of ${currentBatchData.total_analyzed} reviews`;
  }

  if (!tbody) return;
  tbody.innerHTML = '';

  if (filteredReviews.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="6" style="text-align: center; padding: 2rem; color: var(--text-muted);">
          No reviews matched your search filter.
        </td>
      </tr>
    `;
    return;
  }

  filteredReviews.forEach((item) => {
    const isPos = item.prediction === 'Positive';
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td style="font-weight: 700; color: var(--text-muted);">${item.index}</td>
      <td>
        <span style="font-size: 0.88rem; line-height: 1.4; color: var(--text-primary);">
          ${escapeHtml(item.text)}
        </span>
      </td>
      <td>
        <span class="tag-pill ${isPos ? 'happy' : 'unhappy'}">
          ${isPos ? '👍 Happy' : '👎 Unhappy'}
        </span>
      </td>
      <td style="font-weight: 700;">
        ${item.confidence}%
      </td>
      <td>
        <span style="font-size: 0.82rem; color: var(--text-muted);">
          ${item.agreement_pct}% agree
        </span>
      </td>
      <td style="text-align: center;">
        <button class="btn-inspect-studio" data-review="${escapeAttr(item.full_text)}" title="Inspect in Live Studio">
          Inspect ↗
        </button>
      </td>
    `;
    tbody.appendChild(tr);
  });

  // Attach Studio inspect links
  document.querySelectorAll('.btn-inspect-studio').forEach(btn => {
    btn.addEventListener('click', () => {
      const reviewText = btn.getAttribute('data-review');
      if (reviewText) {
        window.location.href = `/?review=${encodeURIComponent(reviewText)}`;
      }
    });
  });
}

function initCsvExport() {
  const btnExport = document.getElementById('btn-export-batch-csv');
  if (btnExport) {
    btnExport.addEventListener('click', () => {
      if (!currentBatchData || !currentBatchData.reviews || currentBatchData.reviews.length === 0) {
        alert('No analyzed reviews available to export.');
        return;
      }

      const rows = [
        ["Index", "Prediction", "Confidence_Pct", "Agreement_Pct", "Review_Snippet", "Full_Review_Text"]
      ];

      currentBatchData.reviews.forEach(r => {
        rows.push([
          r.index,
          r.prediction,
          r.confidence,
          r.agreement_pct,
          `"${r.text.replace(/"/g, '""')}"`,
          `"${r.full_text.replace(/"/g, '""')}"`
        ]);
      });

      const csvContent = rows.map(e => e.join(",")).join("\n");
      const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.setAttribute("href", url);
      link.setAttribute("download", `sentiment_predictions_${Date.now()}.csv`);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    });
  }
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function escapeAttr(str) {
  if (!str) return '';
  return String(str).replace(/"/g, '&quot;');
}
