/**
 * FineFoods AI — Frontend Controller
 * Plain Language Sentiment Intelligence & Liquid Glass Interactions
 */

// Application State with Varied, Authentic Examples
const appState = {
  activeResultType: 'sentiment', // 'sentiment', 'consensus', 'words'
  samplePresets: [
    {
      title: "☕ Gourmet Hazelnut Coffee",
      text: "This coffee has an incredible aroma and rich nutty taste. Not bitter at all, perfectly smooth and flavorful. Five stars, ordering another bag!"
    },
    {
      title: "🥨 Stale Snack Chips",
      text: "Extremely disappointed with this purchase. The bag arrived punctured and the chips were completely stale and tasted like cardboard. Threw it straight into the garbage!"
    },
    {
      title: "🍵 Organic Green Tea",
      text: "Five stars! High quality organic Japanese green tea leaves. Beautiful vibrant color and crisp clean taste. Great packaging keeps it fresh."
    },
    {
      title: "🍬 Chemical Aftertaste",
      text: "The description promised pure fruit flavor, but it tastes entirely like artificial sweetener and chemicals. Left a horrific lingering aftertaste in my mouth. Terrible purchase."
    },
    {
      title: "🐕 Dog Food Problem",
      text: "Beware! My dog refused to eat this after the first bite, and when he did, he vomited within an hour. Very bad batch, returning immediately."
    },
    {
      title: "🍫 85% Dark Chocolate",
      text: "Absolute perfection. Rich 85% cocoa with silky smooth texture and subtle notes of vanilla. Melt in your mouth goodness without excessive sugar. Worth every single penny!"
    }
  ]
};

document.addEventListener('DOMContentLoaded', () => {
  initThemeToggle();
  initPresets();
  initResultTypeSwitcher();
  const modelSelect = document.getElementById('model-select-compact');
  if (modelSelect) {
    enhanceSelect(modelSelect);
  }
  initAnalyzer();
  initScrollReveal();
  initHeroDetections();
  loadLiveReviewsFeed();
  loadTopWords();
  checkUrlParams();
});

function checkUrlParams() {
  const urlParams = new URLSearchParams(window.location.search);
  const reviewParam = urlParams.get('review');
  if (reviewParam) {
    const textarea = document.getElementById('review-input');
    if (textarea) {
      textarea.value = reviewParam;
      updateCharCount();
      scrollToSection('analyzer');
      runSentimentAnalysis();
    }
  }
}

/* ================= 1. THEME TOGGLE (iOS SWITCH) ================= */
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
      updateSegmentedThumb();
    });
  }
}

/* ================= 2. SCROLL REVEAL (INTERSECTION OBSERVER) ================= */
function initScrollReveal() {
  const revealElements = document.querySelectorAll('#analyzer .reveal-item');
  if (!revealElements.length) return;

  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (!('IntersectionObserver' in window) || prefersReducedMotion) {
    revealElements.forEach(el => el.classList.add('in-view'));
    return;
  }

  const observer = new IntersectionObserver((entries, obs) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('in-view');
        obs.unobserve(entry.target);
      }
    });
  }, {
    rootMargin: '0px 0px -40px 0px',
    threshold: 0.12
  });

  revealElements.forEach(el => observer.observe(el));
}

/* ================= 3. QUICK PRESET CHIPS ================= */
function initPresets() {
  const container = document.getElementById('presets-container');
  if (!container) return;
  container.innerHTML = '';

  appState.samplePresets.forEach((preset, index) => {
    const btn = document.createElement('button');
    btn.className = 'preset-chip-btn';
    btn.textContent = preset.title;
    btn.style.animationDelay = `${index * 60}ms`;
    btn.addEventListener('click', () => {
      const textarea = document.getElementById('review-input');
      if (textarea) {
        textarea.value = preset.text;
        updateCharCount();
        runSentimentAnalysis();
      }
    });
    container.appendChild(btn);
  });
}

/* ================= 4. RESULT TYPE SWITCHER (iOS SEGMENTED CONTROL) ================= */
function updateSegmentedThumb() {
  const container = document.getElementById('result-type-pills');
  const thumb = document.getElementById('segmented-thumb');
  if (!container || !thumb) return;

  const activeBtn = container.querySelector('.segment-btn.active');
  if (!activeBtn) return;

  const left = activeBtn.offsetLeft;
  const width = activeBtn.offsetWidth;

  if (width > 0) {
    thumb.style.transform = `translateX(${left}px)`;
    thumb.style.width = `${width}px`;
    thumb.style.opacity = '1';
  }
}

function initResultTypeSwitcher() {
  const pills = document.querySelectorAll('#result-type-pills .segment-btn');
  pills.forEach(pill => {
    pill.addEventListener('click', () => {
      pills.forEach(p => {
        p.classList.remove('active');
        p.setAttribute('aria-selected', 'false');
      });
      pill.classList.add('active');
      pill.setAttribute('aria-selected', 'true');
      const targetType = pill.getAttribute('data-type');
      appState.activeResultType = targetType;
      updateSegmentedThumb();
      switchResultView(targetType);
    });
  });

  window.addEventListener('resize', updateSegmentedThumb);
  window.addEventListener('load', updateSegmentedThumb);
  if (document.fonts) {
    document.fonts.ready.then(updateSegmentedThumb);
  }
  requestAnimationFrame(() => {
    updateSegmentedThumb();
    setTimeout(updateSegmentedThumb, 60);
    setTimeout(updateSegmentedThumb, 250);
  });
}

function switchResultView(viewType) {
  const viewSentiment = document.getElementById('view-sentiment');
  const viewConsensus = document.getElementById('view-consensus');
  const viewWords = document.getElementById('view-words');

  if (viewSentiment) viewSentiment.classList.toggle('active', viewType === 'sentiment');
  if (viewConsensus) viewConsensus.classList.toggle('active', viewType === 'consensus');
  if (viewWords) viewWords.classList.toggle('active', viewType === 'words');
}

/* ================= 5. ENHANCE SELECT (CUSTOM LIQUID GLASS DROPDOWN) ================= */
function enhanceSelect(selectEl) {
  if (!selectEl) return null;

  // 1. Hide the native select
  selectEl.style.display = 'none';
  selectEl.setAttribute('aria-hidden', 'true');
  selectEl.tabIndex = -1;

  // Remove existing enhanced instance if any
  const existingDropdown = selectEl.parentNode ? selectEl.parentNode.querySelector('.custom-glass-dropdown') : null;
  if (existingDropdown) {
    existingDropdown.remove();
  }

  // 2. Build button + <ul role="listbox"> from its options
  const dropdown = document.createElement('div');
  dropdown.className = 'custom-glass-dropdown';
  dropdown.id = 'custom-model-dropdown';

  const trigger = document.createElement('button');
  trigger.type = 'button';
  trigger.className = 'glass-dropdown-trigger';
  trigger.id = 'model-dropdown-trigger';
  trigger.setAttribute('aria-haspopup', 'listbox');
  trigger.setAttribute('aria-expanded', 'false');

  const selectedOpt = selectEl.options[selectEl.selectedIndex] || selectEl.options[0];

  const textSpan = document.createElement('span');
  textSpan.className = 'dropdown-selected-text';
  textSpan.id = 'model-dropdown-text';
  textSpan.textContent = selectedOpt ? selectedOpt.text : '';

  const chevron = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  chevron.setAttribute('class', 'dropdown-chevron');
  chevron.setAttribute('width', '14');
  chevron.setAttribute('height', '14');
  chevron.setAttribute('viewBox', '0 0 24 24');
  chevron.setAttribute('fill', 'none');
  chevron.setAttribute('stroke', 'currentColor');
  chevron.setAttribute('stroke-width', '2.5');
  chevron.setAttribute('stroke-linecap', 'round');
  chevron.setAttribute('stroke-linejoin', 'round');
  const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  path.setAttribute('d', 'm6 9 6 6 6-6');
  chevron.appendChild(path);

  trigger.appendChild(textSpan);
  trigger.appendChild(chevron);

  const menu = document.createElement('ul');
  menu.className = 'glass-dropdown-menu';
  menu.id = 'model-dropdown-menu';
  menu.setAttribute('role', 'listbox');
  menu.tabIndex = -1;

  const items = [];

  Array.from(selectEl.options).forEach((opt, idx) => {
    const li = document.createElement('li');
    li.className = 'glass-dropdown-option';
    li.setAttribute('role', 'option');
    li.setAttribute('data-value', opt.value);
    li.setAttribute('data-index', idx);
    li.tabIndex = -1;

    const isSelected = idx === selectEl.selectedIndex;
    if (isSelected) {
      li.classList.add('is-selected');
      li.setAttribute('aria-selected', 'true');
    } else {
      li.setAttribute('aria-selected', 'false');
    }

    const optLabel = document.createElement('span');
    optLabel.className = 'glass-option-label';
    optLabel.textContent = opt.text;

    const optCheck = document.createElement('span');
    optCheck.className = 'glass-dropdown-check';
    optCheck.setAttribute('aria-hidden', 'true');
    optCheck.textContent = '✓';

    li.appendChild(optLabel);
    li.appendChild(optCheck);

    li.addEventListener('click', (e) => {
      e.stopPropagation();
      chooseItem(idx, true);
    });

    menu.appendChild(li);
    items.push(li);
  });

  dropdown.appendChild(trigger);
  dropdown.appendChild(menu);

  // Insert custom dropdown next to native select
  selectEl.parentNode.insertBefore(dropdown, selectEl.nextSibling);

  let focusedIndex = selectEl.selectedIndex >= 0 ? selectEl.selectedIndex : 0;

  function isOpen() {
    return dropdown.classList.contains('open');
  }

  function openDropdown() {
    dropdown.classList.add('open');
    trigger.setAttribute('aria-expanded', 'true');
    focusedIndex = selectEl.selectedIndex >= 0 ? selectEl.selectedIndex : 0;
    if (items[focusedIndex]) {
      items[focusedIndex].focus();
    }
  }

  function closeDropdown(refocus = true) {
    dropdown.classList.remove('open');
    trigger.setAttribute('aria-expanded', 'false');
    if (refocus) {
      trigger.focus();
    }
  }

  function chooseItem(index, refocus = true) {
    if (index < 0 || index >= selectEl.options.length) return;

    selectEl.selectedIndex = index;
    textSpan.textContent = selectEl.options[index].text;

    items.forEach((item, idx) => {
      const match = idx === index;
      item.classList.toggle('is-selected', match);
      item.setAttribute('aria-selected', match ? 'true' : 'false');
    });

    // Dispatch change event on selectEl
    selectEl.dispatchEvent(new Event('change', { bubbles: true }));

    closeDropdown(refocus);
  }

  // Trigger click
  trigger.addEventListener('click', (e) => {
    e.stopPropagation();
    if (isOpen()) {
      closeDropdown(false);
    } else {
      openDropdown();
    }
  });

  // Keyboard navigation on trigger button
  trigger.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp' || e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      if (!isOpen()) {
        openDropdown();
      } else {
        closeDropdown(false);
      }
    } else if (e.key === 'Escape' && isOpen()) {
      e.preventDefault();
      closeDropdown(true);
    }
  });

  // Keyboard navigation on menu items
  menu.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      e.preventDefault();
      closeDropdown(true);
    } else if (e.key === 'Tab') {
      closeDropdown(false);
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      focusedIndex = (focusedIndex + 1) % items.length;
      items[focusedIndex].focus();
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      focusedIndex = (focusedIndex - 1 + items.length) % items.length;
      items[focusedIndex].focus();
    } else if (e.key === 'Home') {
      e.preventDefault();
      focusedIndex = 0;
      items[focusedIndex].focus();
    } else if (e.key === 'End') {
      e.preventDefault();
      focusedIndex = items.length - 1;
      items[focusedIndex].focus();
    } else if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      chooseItem(focusedIndex, true);
    }
  });

  // Close when clicking outside
  document.addEventListener('click', (e) => {
    if (!dropdown.contains(e.target) && isOpen()) {
      closeDropdown(false);
    }
  });

  // Sync if external script modifies selectEl.selectedIndex
  selectEl.addEventListener('change', () => {
    const idx = selectEl.selectedIndex;
    if (idx >= 0 && idx < selectEl.options.length) {
      textSpan.textContent = selectEl.options[idx].text;
      items.forEach((item, i) => {
        const match = i === idx;
        item.classList.toggle('is-selected', match);
        item.setAttribute('aria-selected', match ? 'true' : 'false');
      });
    }
  });

  return dropdown;
}

/* ================= 6. ANALYZER CONTROLLER ================= */
function initAnalyzer() {
  const textarea = document.getElementById('review-input');
  const btnAnalyze = document.getElementById('btn-analyze');
  const heroSampleBtn = document.getElementById('hero-btn-sample');

  if (textarea) {
    textarea.addEventListener('input', updateCharCount);
    textarea.addEventListener('keyup', updateCharCount);
    textarea.addEventListener('change', updateCharCount);
  }

  if (btnAnalyze) {
    btnAnalyze.addEventListener('click', () => {
      runSentimentAnalysis();
    });
  }

  const btnClear = document.getElementById('btn-clear');
  if (btnClear) {
    btnClear.addEventListener('click', () => {
      if (textarea) {
        textarea.value = '';
        updateCharCount();
        textarea.focus();
      }
      const emptyState = document.getElementById('empty-state');
      const activeBox = document.getElementById('active-result-box');
      if (emptyState) emptyState.style.display = 'block';
      if (activeBox) activeBox.style.display = 'none';
    });
  }

  // "Try an Example" button fills a sample review and auto-runs analysis!
  if (heroSampleBtn) {
    heroSampleBtn.addEventListener('click', () => {
      if (textarea) {
        textarea.value = appState.samplePresets[0].text;
        updateCharCount();
        scrollToSection('analyzer');
        runSentimentAnalysis();
      }
    });
  }
}

function updateCharCount() {
  const textarea = document.getElementById('review-input');
  const counter = document.getElementById('char-counter');
  if (textarea && counter) {
    const len = textarea.value.length;
    counter.textContent = `${len} character${len === 1 ? '' : 's'}`;
  }
}

/* ================= 7. COUNT-UP ANIMATION UTILITY ================= */
function animateCountUp(element, endVal, duration = 800, suffix = '') {
  if (!element) return;
  const numVal = parseFloat(endVal);
  if (isNaN(numVal)) {
    element.textContent = `${endVal}${suffix}`;
    return;
  }

  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    element.textContent = `${endVal}${suffix}`;
    return;
  }

  const startVal = 0;
  const startTime = performance.now();
  const isDecimal = String(endVal).includes('.');

  function update(now) {
    const elapsed = now - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const ease = 1 - Math.pow(1 - progress, 3); // cubic ease-out
    const current = startVal + (numVal - startVal) * ease;

    element.textContent = `${isDecimal ? current.toFixed(1) : Math.round(current)}${suffix}`;

    if (progress < 1) {
      requestAnimationFrame(update);
    } else {
      element.textContent = `${endVal}${suffix}`;
    }
  }

  requestAnimationFrame(update);
}

/* ================= 8. INFERENCE & RESULTS RENDERING ================= */
async function runSentimentAnalysis() {
  const textarea = document.getElementById('review-input');
  const modelSelect = document.getElementById('model-select-compact');
  const btnAnalyze = document.getElementById('btn-analyze');

  const text = textarea ? textarea.value.trim() : '';
  if (!text) {
    if (textarea) textarea.focus();
    return;
  }

  if (btnAnalyze) {
    btnAnalyze.disabled = true;
    btnAnalyze.classList.add('loading');
    const textSpan = btnAnalyze.querySelector('.btn-text');
    if (textSpan) textSpan.textContent = 'Reading review...';
  }

  // Minimum 400ms delay to ensure shimmer & spinner animation is polished and visible
  const minDelay = new Promise(resolve => setTimeout(resolve, 400));

  try {
    const selectedModelKey = modelSelect ? modelSelect.value : 'ensemble_avg';
    const fetchPromise = fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: text,
        model: selectedModelKey
      })
    });

    const [res] = await Promise.all([fetchPromise, minDelay]);
    const json = await res.json();
    if (json.status === 'success') {
      renderSentimentResults(json.data);
      if (btnAnalyze) {
        btnAnalyze.classList.remove('loading');
        btnAnalyze.classList.add('btn-success-pulse');
        setTimeout(() => btnAnalyze.classList.remove('btn-success-pulse'), 350);
      }
    } else {
      alert('Error: ' + json.message);
    }
  } catch (err) {
    console.error('Inference error:', err);
    alert('Server communication error.');
  } finally {
    if (btnAnalyze) {
      btnAnalyze.disabled = false;
      btnAnalyze.classList.remove('loading');
      const textSpan = btnAnalyze.querySelector('.btn-text');
      if (textSpan) textSpan.textContent = 'Check Review';
    }
  }
}

function renderSentimentResults(data) {
  const emptyState = document.getElementById('empty-state');
  const activeBox = document.getElementById('active-result-box');

  if (emptyState) emptyState.style.display = 'none';
  if (activeBox) activeBox.style.display = 'block';

  const isPos = data.prediction === 'Positive';

  // 1. Plain Language Sentiment Result ("Happy customer (96% sure)")
  const banner = document.getElementById('sentiment-banner');
  const emoji = document.getElementById('sentiment-emoji');
  const verdict = document.getElementById('banner-verdict');
  const activeAlgoTag = document.getElementById('banner-active-algorithm');
  const metricConf = document.getElementById('metric-confidence');
  const metricLatency = document.getElementById('metric-latency');

  if (banner) {
    banner.classList.toggle('negative', !isPos);
    banner.classList.remove('verdict-pop');
    void banner.offsetWidth; // restart pop-in animation
    banner.classList.add('verdict-pop');
  }
  if (emoji) emoji.textContent = isPos ? '👍' : '👎';
  if (verdict) {
    verdict.textContent = isPos
      ? `Happy customer (${data.confidence}% sure)`
      : `Unhappy customer (${data.confidence}% sure)`;
  }
  if (activeAlgoTag) {
    activeAlgoTag.textContent = data.selected_model === 'ensemble_avg'
      ? 'All 6 models combined'
      : data.selected_model_name;
  }
  
  // Count-up confidence from 0 to target over 800ms
  animateCountUp(metricConf, data.confidence, 800, '%');
  
  // Measured latency from API
  if (metricLatency) metricLatency.textContent = `${data.latency_ms} ms`;

  // Probability Track ("Chance of happy" / "Chance of unhappy")
  const posVal = document.getElementById('val-pos-pct');
  const negVal = document.getElementById('val-neg-pct');
  const posFill = document.getElementById('prob-fill-positive');
  const negFill = document.getElementById('prob-fill-negative');

  animateCountUp(posVal, data.positive_prob, 800, '%');
  animateCountUp(negVal, data.negative_prob, 800, '%');

  if (posFill && negFill) {
    posFill.style.width = '0%';
    negFill.style.width = '0%';
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        posFill.style.width = `${data.positive_prob}%`;
        negFill.style.width = `${data.negative_prob}%`;
      });
    });
  }

  // 2. "Do the models agree?"
  const consensusContainer = document.getElementById('consensus-cards-container');
  const consensusBadge = document.getElementById('consensus-badge');

  if (consensusContainer && data.models_consensus) {
    consensusContainer.innerHTML = '';
    
    const coreKeys = ['logistic_regression', 'naive_bayes', 'svm', 'xgboost', 'random_forest', 'knn'];
    let agreementCount = 0;

    coreKeys.forEach(k => {
      const model = data.models_consensus[k];
      if (model && model.prediction === data.prediction) {
        agreementCount++;
      }
    });

    if (consensusBadge) {
      const agreePct = data.model_agreement_pct !== undefined ? data.model_agreement_pct : Math.round(agreementCount / 6 * 100);
      consensusBadge.textContent = `${agreementCount} of 6 models agree (${agreePct}%)`;
      consensusBadge.className = `tag-pill ${agreementCount >= 4 ? 'happy' : 'unhappy'}`;
    }

    const displayKeys = ['ensemble_avg', 'logistic_regression', 'naive_bayes', 'svm', 'xgboost', 'random_forest', 'knn'];

    displayKeys.forEach((k, idx) => {
      const model = data.models_consensus[k];
      if (!model) return;

      const isModelPos = model.prediction === 'Positive';
      const isEnsemble = k === 'ensemble_avg';
      const isSelected = data.selected_model === k;

      const card = document.createElement('div');
      card.className = `consensus-box ${isEnsemble ? 'consensus-box-ensemble' : ''} ${isSelected ? 'active-selection' : ''}`;
      card.style.animationDelay = `${idx * 80}ms`;
      
      const plainModelLabel = isEnsemble ? 'All 6 Models Combined' : model.name;

      card.innerHTML = `
        <div class="consensus-box-header">
          <span class="consensus-box-title">${plainModelLabel}</span>
          ${isEnsemble ? '<span class="tag-pill happy" style="font-size: 0.68rem; padding: 0.15rem 0.45rem;">Average</span>' : ''}
        </div>
        <div class="consensus-box-vote ${isModelPos ? 'pos' : 'neg'}">
          ${isModelPos ? 'Happy' : 'Unhappy'}
        </div>
        <div class="consensus-box-conf">${model.confidence}% sure</div>
        <div class="score-bar-track" style="margin-bottom: 0.35rem; height: 5px;">
          <div class="consensus-mini-bar" style="width: 0%; height: 100%; background: ${isModelPos ? '#10b981' : '#f43f5e'};"></div>
        </div>
      `;
      consensusContainer.appendChild(card);

      const miniBar = card.querySelector('.consensus-mini-bar');
      if (miniBar) {
        setTimeout(() => {
          miniBar.style.width = `${model.confidence}%`;
        }, idx * 80 + 120);
      }
    });
  }

  // 3. "Words that mattered most"
  const wordsContainer = document.getElementById('words-chips-container');
  if (wordsContainer) {
    wordsContainer.innerHTML = '';
    if (data.word_attributions && data.word_attributions.length) {
      data.word_attributions.slice(0, 8).forEach((attr, idx) => {
        const isWordPos = attr.sentiment === 'Positive';
        const chip = document.createElement('div');
        chip.className = `word-chip ${isWordPos ? 'pos' : 'neg'}`;
        chip.style.animationDelay = `${idx * 60}ms`;
        chip.innerHTML = `
          <span>${isWordPos ? '▲' : '▼'} ${escapeHtml(attr.word)}</span>
          <span style="font-size: 0.72rem; opacity: 0.85;">(${attr.contribution > 0 ? '+' : ''}${attr.contribution})</span>
        `;
        wordsContainer.appendChild(chip);
      });
    } else {
      wordsContainer.innerHTML = '<span style="color: var(--text-muted); font-size: 0.85rem;">Standard everyday food words found.</span>';
    }
  }

  switchResultView(appState.activeResultType);
  updateSegmentedThumb();
}

/* ================= 9. HERO DETECTION ROWS ================= */
function initHeroDetections() {
  const feedItems = document.querySelectorAll('.preview-example-item.clickable');
  feedItems.forEach(item => {
    item.addEventListener('click', () => {
      const sampleIdx = parseInt(item.getAttribute('data-sample') || '0', 10);
      const preset = appState.samplePresets[sampleIdx] || appState.samplePresets[0];

      const textarea = document.getElementById('review-input');
      if (textarea) {
        textarea.value = preset.text;
        updateCharCount();
        scrollToSection('analyzer');
        runSentimentAnalysis();
      }
    });
  });
}

/* ================= 10. REVIEWS FEED (REAL DATASET ROWS) ================= */
async function loadLiveReviewsFeed() {
  const container = document.getElementById('reviews-feed-container');
  if (!container) return;

  try {
    const res = await fetch('/api/reviews?per_page=6');
    const data = await res.json();
    if (data.status === 'success' && data.reviews) {
      container.innerHTML = '';
      data.reviews.forEach(r => {
        const card = document.createElement('div');
        card.className = 'liquid-card feed-review-card';
        const stars = '★'.repeat(r.score) + '☆'.repeat(5 - r.score);
        const isPos = r.sentiment.toLowerCase() === 'positive';

        card.innerHTML = `
          <div class="feed-card-top">
            <span class="stars-row">${stars}</span>
            <span class="tag-pill ${isPos ? 'happy' : 'unhappy'}">${isPos ? 'Happy customer' : 'Unhappy customer'}</span>
          </div>
          <div class="feed-product">${escapeHtml(r.product_name || 'Amazon Fine Food Product')}</div>
          <div class="feed-summary">${escapeHtml(r.summary)}</div>
          <p class="feed-body">${escapeHtml(r.text)}</p>
          <button class="btn-ios-secondary" style="align-self: flex-start; padding: 0.35rem 0.85rem; font-size: 0.78rem;">
            ⚡ Check this review
          </button>
        `;

        card.querySelector('button').addEventListener('click', () => {
          const textarea = document.getElementById('review-input');
          if (textarea) {
            textarea.value = r.text;
            updateCharCount();
            scrollToSection('analyzer');
            runSentimentAnalysis();
          }
        });

        container.appendChild(card);
      });
    }
  } catch (err) {
    console.error('Failed to load reviews feed:', err);
  }
}

/* Helper Functions */
async function loadTopWords() {
  const happyContainer = document.getElementById('happy-words-chips');
  const unhappyContainer = document.getElementById('unhappy-words-chips');
  const contentWrapper = document.getElementById('top-words-content');
  const fallbackWrapper = document.getElementById('top-words-fallback');

  if (!happyContainer || !unhappyContainer) return;

  try {
    const res = await fetch('/api/top-words');
    if (!res.ok) throw new Error('API request failed');
    const json = await res.json();

    const happyWords = json.happy_words || (json.data && json.data.happy_words) || [];
    const unhappyWords = json.unhappy_words || (json.data && json.data.unhappy_words) || [];

    if (!happyWords.length && !unhappyWords.length) {
      throw new Error('No words returned');
    }

    // Render Happy words (8 green chips)
    happyContainer.innerHTML = '';
    happyWords.slice(0, 8).forEach(item => {
      const chip = document.createElement('span');
      chip.className = 'word-chip happy';
      chip.textContent = typeof item === 'string' ? item : item.word;
      if (item.weight) {
        chip.title = `Impact score: +${Math.abs(item.weight).toFixed(2)}`;
      }
      happyContainer.appendChild(chip);
    });

    // Render Unhappy words (8 red chips)
    unhappyContainer.innerHTML = '';
    unhappyWords.slice(0, 8).forEach(item => {
      const chip = document.createElement('span');
      chip.className = 'word-chip unhappy';
      chip.textContent = typeof item === 'string' ? item : item.word;
      if (item.weight) {
        chip.title = `Impact score: -${Math.abs(item.weight).toFixed(2)}`;
      }
      unhappyContainer.appendChild(chip);
    });

    if (contentWrapper) contentWrapper.style.display = 'flex';
    if (fallbackWrapper) fallbackWrapper.style.display = 'none';

  } catch (err) {
    console.warn('Failed to load /api/top-words:', err);
    if (contentWrapper) contentWrapper.style.display = 'none';
    if (fallbackWrapper) {
      fallbackWrapper.style.display = 'block';
      fallbackWrapper.innerHTML = '<p class="top-words-error">Word list is not available right now.</p>';
    }
  }
}

function scrollToSection(id) {
  const el = document.getElementById(id);
  if (el) {
    el.scrollIntoView({ behavior: 'smooth' });
  }
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
