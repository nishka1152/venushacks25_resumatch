async function submitMatch() {
  const fileInput = document.getElementById('resumeUpload');
  const jd = document.getElementById('jobDescription').value;
  const resultsDiv = document.getElementById('results');
  const loader = document.getElementById('loader');
  resultsDiv.innerHTML = '';
  loader.style.display = 'block';

  if (!fileInput.files.length || !jd.trim()) {
    alert('Please upload a PDF resume and enter a job description.');
    loader.style.display = 'none';
    return;
  }

  const formData = new FormData();
  formData.append('resume', fileInput.files[0]);
  formData.append('job_description', jd);

  try {
    const res = await fetch('/api/match', {
      method: 'POST',
      body: formData
    });
    const data = await res.json();
    loader.style.display = 'none';

    if (data && data.score) {
      resultsDiv.innerHTML = `
        <div class="result-item"><strong>Total Match:</strong> ${data.score.total_score}%</div>
        <div class="result-item"><strong>Experience:</strong> ${data.score.experience_score}%</div>
        <div class="result-item"><strong>Degree Match:</strong> ${data.score.degree_match}%</div>
        <div class="result-item"><strong>Tech Skill Match:</strong> ${data.score.tech_skill_match}%</div>
        <div class="result-item"><strong>Soft Skill Match:</strong> ${data.score.soft_skill_match}%</div>
        <div class="result-item"><strong>Language Match:</strong> ${data.score.language_match}%</div>
      `;
    } else {
      resultsDiv.innerHTML = `<div class="result-item" style="background:#f8d7da;border-left-color:#dc3545;">❌ Unable to compute match.</div>`;
    }
  } catch (err) {
    loader.style.display = 'none';
    resultsDiv.innerHTML = `<div class="result-item" style="background:#f8d7da;border-left-color:#dc3545;">❌ Server error occurred.</div>`;
  }
}

// Theme toggling — this should be AFTER submitMatch(), not wrapped around it
function toggleTheme() {
  const body = document.body;
  const themeSwitch = document.getElementById('themeSwitch');
  const isDark = themeSwitch.checked;

  if (isDark) {
    body.classList.add('dark');
    localStorage.setItem('theme', 'dark');
  } else {
    body.classList.remove('dark');
    localStorage.setItem('theme', 'light');
  }
}

window.onload = function () {
  const savedTheme = localStorage.getItem('theme');
  const themeSwitch = document.getElementById('themeSwitch');
  if (savedTheme === 'dark') {
    document.body.classList.add('dark');
    if (themeSwitch) themeSwitch.checked = true;
  }
}
