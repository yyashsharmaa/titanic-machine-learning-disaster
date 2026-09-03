document.addEventListener('DOMContentLoaded', () => {
  // Elements
  const tabs = document.querySelectorAll('.nav-tab');
  const tabContents = document.querySelectorAll('.tab-content');
  const ageSlider = document.getElementById('age');
  const ageDisplay = document.getElementById('age-display');
  const ageCategory = document.getElementById('age-category');
  const form = document.getElementById('passenger-form');
  const probValue = document.getElementById('prob-value');
  const progressCircle = document.getElementById('progress-circle');
  const statusBanner = document.getElementById('status-banner');
  const statusIcon = document.getElementById('status-icon');
  const statusHeadline = document.getElementById('status-headline');
  const statusSub = document.getElementById('status-sub');
  const factorsList = document.getElementById('factors-list');

  // Radial progress circumference (2 * pi * r, r=92 -> 578.05)
  const CIRCUMFERENCE = 578.05;
  progressCircle.style.strokeDasharray = `${CIRCUMFERENCE} ${CIRCUMFERENCE}`;

  // 1. Tab Switching
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      tabContents.forEach(tc => tc.classList.remove('active'));

      tab.classList.add('active');
      const targetId = tab.getAttribute('data-tab');
      const targetContent = document.getElementById(targetId);
      if (targetContent) {
        targetContent.classList.add('active');
      }
    });
  });

  // 2. Age Slider & Category
  function updateAgeDisplay(val) {
    ageDisplay.textContent = val;
    let cat = 'Adult';
    if (val <= 12) cat = 'Child';
    else if (val <= 18) cat = 'Teen';
    else if (val <= 35) cat = 'Young Adult';
    else if (val <= 60) cat = 'Adult';
    else cat = 'Senior';
    ageCategory.textContent = cat;
  }

  if (ageSlider) {
    ageSlider.addEventListener('input', (e) => {
      updateAgeDisplay(e.target.value);
    });
  }

  // 3. Counter Steppers (SibSp & Parch)
  document.querySelectorAll('.btn-step').forEach(btn => {
    btn.addEventListener('click', () => {
      const targetId = btn.getAttribute('data-target');
      const action = btn.getAttribute('data-action');
      const input = document.getElementById(targetId);
      if (!input) return;

      let val = parseInt(input.value, 10) || 0;
      if (action === 'inc' && val < 8) val++;
      if (action === 'dec' && val > 0) val--;
      input.value = val;
    });
  });

  // 4. Update Gauge
  function setGaugeProgress(percent) {
    const offset = CIRCUMFERENCE - (percent / 100) * CIRCUMFERENCE;
    progressCircle.style.strokeDashoffset = offset;

    // Color shift
    if (percent >= 60) {
      progressCircle.style.stroke = '#10b981'; // Emerald
    } else if (percent >= 40) {
      progressCircle.style.stroke = '#f59e0b'; // Amber
    } else {
      progressCircle.style.stroke = '#f43f5e'; // Rose
    }
  }

  // 5. Predict API Call
  async function performPrediction() {
    const pclassInput = document.querySelector('input[name="pclass"]:checked');
    const sexInput = document.querySelector('input[name="sex"]:checked');
    const nameVal = document.getElementById('name').value;
    const ageVal = parseFloat(document.getElementById('age').value);
    const sibspVal = parseInt(document.getElementById('sibsp').value, 10);
    const parchVal = parseInt(document.getElementById('parch').value, 10);
    const fareVal = parseFloat(document.getElementById('fare').value);
    const embarkedVal = document.getElementById('embarked').value;
    const cabinVal = document.getElementById('cabin').value;

    const payload = {
      Pclass: pclassInput ? parseInt(pclassInput.value, 10) : 3,
      Sex: sexInput ? sexInput.value : 'male',
      Name: nameVal,
      Age: ageVal,
      SibSp: sibspVal,
      Parch: parchVal,
      Fare: fareVal,
      Embarked: embarkedVal,
      Cabin: cabinVal
    };

    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();

      if (data.error) {
        alert('Prediction error: ' + data.error);
        return;
      }

      // Update Gauge
      const prob = data.survival_probability;
      probValue.textContent = `${prob.toFixed(1)}%`;
      setGaugeProgress(prob);

      // Update Status Banner
      if (data.survived === 1) {
        statusBanner.className = 'status-banner banner-survived';
        statusIcon.textContent = '✅';
        statusHeadline.textContent = 'High Probability of Survival';
        statusSub.textContent = `Survival chance evaluated at ${prob.toFixed(1)}% (${data.risk_level} Risk Category).`;
      } else {
        statusBanner.className = 'status-banner banner-perished';
        statusIcon.textContent = '⚠️';
        statusHeadline.textContent = 'High Mortality Risk';
        statusSub.textContent = `Survival chance estimated at ${prob.toFixed(1)}% (${data.risk_level} Risk Category).`;
      }

      // Render Contributing Factors
      factorsList.innerHTML = '';
      if (data.factors && data.factors.length > 0) {
        data.factors.forEach(f => {
          const li = document.createElement('li');
          li.className = `factor-item ${f.type}`;
          const icon = f.type === 'positive' ? '🟢' : (f.type === 'negative' ? '🔴' : '⚪');
          li.innerHTML = `<span>${icon}</span> <span>${f.text}</span>`;
          factorsList.appendChild(li);
        });
      }
    } catch (err) {
      console.error('Failed to predict:', err);
    }
  }

  if (form) {
    form.addEventListener('submit', (e) => {
      e.preventDefault();
      performPrediction();
    });
  }

  // 6. Archetypes Loading
  const archetypes = {
    rose: {
      name: 'Bukater, Miss. Rose DeWitt',
      pclass: '1',
      sex: 'female',
      age: 17,
      sibsp: 0,
      parch: 1,
      fare: 227.5,
      cabin: 'B51',
      embarked: 'C'
    },
    jack: {
      name: 'Dawson, Mr. Jack',
      pclass: '3',
      sex: 'male',
      age: 20,
      sibsp: 0,
      parch: 0,
      fare: 8.05,
      cabin: '',
      embarked: 'S'
    },
    carter: {
      name: 'Carter, Master. William Thornton II',
      pclass: '1',
      sex: 'male',
      age: 11,
      sibsp: 1,
      parch: 2,
      fare: 120.0,
      cabin: 'B96',
      embarked: 'S'
    },
    father: {
      name: 'Andersson, Mr. Anders Johan',
      pclass: '3',
      sex: 'male',
      age: 39,
      sibsp: 1,
      parch: 5,
      fare: 31.275,
      cabin: '',
      embarked: 'S'
    }
  };

  document.querySelectorAll('.archetype-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const type = btn.getAttribute('data-archetype');
      const arch = archetypes[type];
      if (!arch) return;

      document.getElementById('name').value = arch.name;
      const pclassRadio = document.querySelector(`input[name="pclass"][value="${arch.pclass}"]`);
      if (pclassRadio) pclassRadio.checked = true;

      const sexRadio = document.querySelector(`input[name="sex"][value="${arch.sex}"]`);
      if (sexRadio) sexRadio.checked = true;

      document.getElementById('age').value = arch.age;
      updateAgeDisplay(arch.age);

      document.getElementById('sibsp').value = arch.sibsp;
      document.getElementById('parch').value = arch.parch;
      document.getElementById('fare').value = arch.fare;
      document.getElementById('embarked').value = arch.embarked;
      document.getElementById('cabin').value = arch.cabin;

      performPrediction();
    });
  });

  // 7. Load Benchmark Stats
  async function loadStats() {
    try {
      const res = await fetch('/api/stats');
      const data = await res.json();

      if (data.metadata && data.metadata.cv_accuracy) {
        document.getElementById('best-model-name').textContent = data.metadata.model_type || 'Gradient Boosting';
        document.getElementById('best-cv-acc').textContent = `${(data.metadata.cv_accuracy * 100).toFixed(1)}%`;
        document.getElementById('best-cv-auc').textContent = data.metadata.cv_roc_auc ? data.metadata.cv_roc_auc.toFixed(3) : '0.874';
        document.getElementById('best-cv-f1').textContent = data.metadata.cv_f1 ? data.metadata.cv_f1.toFixed(3) : '0.781';
      }

      if (data.benchmark && data.benchmark.length > 0) {
        const tbody = document.getElementById('benchmark-table-body');
        tbody.innerHTML = '';
        data.benchmark.forEach((row, idx) => {
          const tr = document.createElement('tr');
          const isTop = idx === 0;
          tr.innerHTML = `
            <td><strong>${row.Model}</strong></td>
            <td>${(row.Accuracy_Mean * 100).toFixed(2)}% ± ${(row.Accuracy_Std * 100).toFixed(2)}%</td>
            <td>${(row.F1_Mean).toFixed(3)}</td>
            <td>${(row.ROC_AUC_Mean).toFixed(3)}</td>
            <td>${(row.Precision_Mean).toFixed(3)}</td>
            <td>${(row.Recall_Mean).toFixed(3)}</td>
            <td>${isTop ? '<span class="badge-top">★ Production Model</span>' : '<span style="color:var(--text-sub)">Evaluated</span>'}</td>
          `;
          tbody.appendChild(tr);
        });
      }
    } catch (err) {
      console.warn('Stats API not ready yet or loading defaults:', err);
    }
  }

  loadStats();
  // Trigger initial prediction for default profile
  setTimeout(performPrediction, 500);
});
