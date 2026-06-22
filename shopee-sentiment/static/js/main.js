/* ============================================================
   main.js – Shopee Sentiment Analyzer Frontend Logic
   ============================================================ */

"use strict";

/* ── Contoh ulasan untuk tombol cepat ───────────────────── */
const EXAMPLES = {
  positif:
    "Barang datang cepat banget, packing sangat aman dan rapi. Kualitas produk sesuai foto bahkan lebih bagus dari ekspektasi. Seller responsif dan ramah. Sangat puas dan pasti akan beli lagi! Highly recommended!",
  negatif:
    "Sangat kecewa dengan produk ini. Barang yang datang tidak sesuai foto sama sekali, kualitas sangat buruk dan mudah rusak. Seller tidak mau bertanggung jawab dan tidak responsif. Harga mahal tapi kualitas seperti barang murahan. Tidak akan beli lagi!",
  netral:
    "Barang sudah datang sesuai estimasi. Kondisi produk biasa saja, tidak terlalu bagus tapi tidak jelek juga. Packing standar.  ",
};

/* ── Hitung karakter textarea ───────────────────────────── */
const textarea = document.getElementById("reviewInput");
const charCount = document.getElementById("charCount");

if (textarea) {
  textarea.addEventListener("input", () => {
    charCount.textContent = `${textarea.value.length} karakter`;
  });
  textarea.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") analyze();
  });
}

function fillExample(type) {
  if (!textarea) return;
  textarea.value = EXAMPLES[type] || "";
  charCount.textContent = `${textarea.value.length} karakter`;
  textarea.focus();
}

/* ── Warna per sentimen ─────────────────────────────────── */
const COLORS = {
  positif: { fill: "#10B981", bg: "pos-bg", cls: "pos" },
  negatif: { fill: "#E53935", bg: "neg-bg", cls: "neg" },
  netral: { fill: "#F59E0B", bg: "neu-bg", cls: "neu" },
};

/* ── Fungsi Analisis Sentimen ───────────────────────────── */
async function analyze() {
  const text = textarea ? textarea.value.trim() : "";
  if (!text) {
    alert("Mohon masukkan teks ulasan terlebih dahulu!");
    return;
  }

  const btn = document.getElementById("btnAnalyze");
  const resultCard = document.getElementById("resultCard");
  const loadingCard = document.getElementById("loadingCard");

  btn.disabled = true;
  resultCard.style.display = "none";
  loadingCard.style.display = "flex";

  try {
    const resp = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    if (!resp.ok) throw new Error(`HTTP error: ${resp.status}`);
    const data = await resp.json();
    renderResult(data);
  } catch (err) {
    console.error(err);
    alert("Terjadi kesalahan saat memproses. Pastikan server Flask berjalan!");
  } finally {
    btn.disabled = false;
    loadingCard.style.display = "none";
  }
}

/* ── Render hasil prediksi ──────────────────────────────── */
function renderResult(data) {
  const resultCard = document.getElementById("resultCard");
  const sentiment = data.sentiment || "netral";
  const palette = COLORS[sentiment] || COLORS.netral;

  // Header
  const header = document.getElementById("resultHeader");
  header.className = `result-header ${palette.bg}`;

  // SVG emoji per sentimen
  const emojiEl = document.getElementById("resultEmoji");
  const svgMap = {
    positif: `<svg width="36" height="36" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="10" stroke="#10b981" stroke-width="2"/><path d="M8 14s1.5 2 4 2 4-2 4-2" stroke="#10b981" stroke-width="2" stroke-linecap="round"/><circle cx="9" cy="10" r="1.5" fill="#10b981"/><circle cx="15" cy="10" r="1.5" fill="#10b981"/></svg>`,
    negatif: `<svg width="36" height="36" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="10" stroke="#ef4444" stroke-width="2"/><path d="M8 16s1.5-2 4-2 4 2 4 2" stroke="#ef4444" stroke-width="2" stroke-linecap="round"/><circle cx="9" cy="10" r="1.5" fill="#ef4444"/><circle cx="15" cy="10" r="1.5" fill="#ef4444"/></svg>`,
    netral: `<svg width="36" height="36" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="10" stroke="#f59e0b" stroke-width="2"/><path d="M8 15h8" stroke="#f59e0b" stroke-width="2" stroke-linecap="round"/><circle cx="9" cy="10" r="1.5" fill="#f59e0b"/><circle cx="15" cy="10" r="1.5" fill="#f59e0b"/></svg>`,
  };
  emojiEl.innerHTML = svgMap[sentiment] || svgMap.netral;

  const sentEl = document.getElementById("resultSentiment");
  sentEl.textContent = sentiment.toUpperCase();
  sentEl.className = `result-sentiment ${palette.cls}`;

  document.getElementById("confValue").textContent = `${data.confidence}%`;
  // Probability bars
  const container = document.getElementById("probaBars");
  container.innerHTML = "";

  const proba = data.probabilities || {};
  const sortedKeys = Object.keys(proba).sort((a, b) => proba[b] - proba[a]);

  sortedKeys.forEach((key) => {
    const pct = proba[key];
    const p = COLORS[key] || {};
    const fillClr = p.fill || "#888";

    const row = document.createElement("div");
    row.className = "proba-bar-row";
    row.innerHTML = `
      <span class="proba-name">${key.charAt(0).toUpperCase() + key.slice(1)}</span>
      <div class="proba-track">
        <div class="proba-fill" style="width:0%; background:${fillClr};"></div>
      </div>
      <span class="proba-pct">${pct}%</span>
    `;
    container.appendChild(row);

    // Animasi bar
    requestAnimationFrame(() => {
      setTimeout(() => {
        row.querySelector(".proba-fill").style.width = `${pct}%`;
      }, 50);
    });
  });

  // Cleaned text
  document.getElementById("cleanText").textContent =
    data.cleaned_text || "(teks terlalu pendek setelah preprocessing)";

  resultCard.style.display = "block";
  resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

/* ── Chart.js: Distribusi Dataset ──────────────────────── */
function initDistChart(metrics) {
  const ctx = document.getElementById("distChart");
  if (!ctx || !metrics) return;

  const dist = metrics.distribution || {};
  const labels = Object.keys(dist).map(
    (k) => k.charAt(0).toUpperCase() + k.slice(1),
  );
  const values = Object.values(dist);

  new Chart(ctx, {
    type: "doughnut",
    data: {
      labels,
      datasets: [
        {
          data: values,
          backgroundColor: ["#10B981", "#E53935", "#F59E0B"],
          borderWidth: 3,
          borderColor: "#fff",
          hoverOffset: 10,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: {
          position: "bottom",
          labels: { padding: 16, font: { size: 13, weight: "600" } },
        },
        tooltip: {
          callbacks: {
            label: (ctx) =>
              `  ${ctx.label}: ${ctx.parsed} data (${Math.round((ctx.parsed / values.reduce((a, b) => a + b, 0)) * 100)}%)`,
          },
        },
      },
    },
  });
}

/* ── Chart.js: Metrik Evaluasi ──────────────────────────── */
function initMetricChart(metrics) {
  const ctx = document.getElementById("metricChart");
  if (!ctx || !metrics) return;

  const labels = ["Akurasi", "Precision", "Recall", "F1-Score", "CV Mean"];
  const values = [
    metrics.accuracy,
    metrics.precision,
    metrics.recall,
    metrics.f1_score,
    metrics.cv_mean,
  ];

  new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Nilai (%)",
          data: values,
          backgroundColor: [
            "rgba(238,77,45,.8)",
            "rgba(16,185,129,.8)",
            "rgba(37,99,235,.8)",
            "rgba(245,158,11,.8)",
            "rgba(139,92,246,.8)",
          ],
          borderRadius: 8,
          borderSkipped: false,
        },
      ],
    },
    options: {
      responsive: true,
      scales: {
        y: {
          min: 0,
          max: 100,
          ticks: { callback: (v) => `${v}%` },
          grid: { color: "rgba(0,0,0,.05)" },
        },
        x: { grid: { display: false } },
      },
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: (ctx) => ` ${ctx.parsed.y}%` } },
      },
    },
  });
}

/* ── Confusion Matrix HTML ──────────────────────────────── */
function initConfusionMatrix(metrics) {
  const wrapper = document.getElementById("cmWrapper");
  if (!wrapper || !metrics) return;

  const cm = metrics.confusion_matrix;
  const labels = metrics.labels || [];
  if (!cm || !cm.length) return;

  let html = `<table class="cm-table">
    <thead><tr><th>Aktual ↓ / Prediksi →</th>`;
  labels.forEach(
    (l) => (html += `<th>${l.charAt(0).toUpperCase() + l.slice(1)}</th>`),
  );
  html += `</tr></thead><tbody>`;

  cm.forEach((row, i) => {
    html += `<tr><th>${labels[i] ? labels[i].charAt(0).toUpperCase() + labels[i].slice(1) : i}</th>`;
    row.forEach((val, j) => {
      const cls = i === j ? "cm-diag" : "cm-off";
      html += `<td class="${cls}">${val}</td>`;
    });
    html += `</tr>`;
  });

  html += `</tbody></table>`;
  wrapper.innerHTML = html;
}

/* ── Top Words ──────────────────────────────────────────── */
async function loadTopWords(sentiment, btn) {
  // Update tab aktif
  document
    .querySelectorAll(".wc-tab")
    .forEach((t) => t.classList.remove("active"));
  if (btn) btn.classList.add("active");

  const container = document.getElementById("topWordsContainer");
  container.innerHTML = `<span style="color:var(--text-3);font-size:13px;">Memuat kata...</span>`;

  try {
    const resp = await fetch(`/wordcloud/${sentiment}`);
    const words = await resp.json();

    if (!words || words.length === 0) {
      container.innerHTML = `<span style="color:var(--text-3);font-size:13px;">Data tidak tersedia.</span>`;
      return;
    }

    // Normalisasi ukuran font berdasarkan frekuensi
    const maxVal = Math.max(...words.map((w) => w.value));
    container.innerHTML = words
      .slice(0, 40)
      .map((w) => {
        const size = 11 + Math.round((w.value / maxVal) * 12);
        return `<span class="tw-chip" style="font-size:${size}px;" title="${w.value}x">${w.text}</span>`;
      })
      .join("");
  } catch {
    container.innerHTML = `<span style="color:var(--text-3);font-size:13px;">Gagal memuat data.</span>`;
  }
}

/* ── Init semua chart saat halaman dimuat ─────────────────── */
document.addEventListener("DOMContentLoaded", () => {
  const metrics = window.METRICS || {};

  initDistChart(metrics);
  initMetricChart(metrics);
  initConfusionMatrix(metrics);
  loadTopWords("positif", document.querySelector(".wc-tab.active"));

  // Smooth scroll navigasi
  document.querySelectorAll(".nav-link").forEach((link) => {
    link.addEventListener("click", (e) => {
      const href = link.getAttribute("href");
      if (href && href.startsWith("#")) {
        e.preventDefault();
        document.querySelector(href)?.scrollIntoView({ behavior: "smooth" });
        document
          .querySelectorAll(".nav-link")
          .forEach((l) => l.classList.remove("active"));
        link.classList.add("active");
      }
    });
  });
});
document.addEventListener("DOMContentLoaded", function () {
  // 1. Efek Animasi Ketik/Ganti Otomatis pada Card Visualizer Kanan (Ala Widget Contoh)
  const previewText = document.getElementById("previewText");
  const previewBadge = document.getElementById("previewBadge");

  const sampleReviews = [
    {
      text: '"Barangnya bagus banget, sesuai deskripsi!"',
      label: "😊 Positive Sentiment",
      color: "#10b981",
      bg: "#ecfdf5",
    },
    {
      text: '"Kecewa, pengirimannya lama banget dan pecah."',
      label: "😠 Negative Sentiment",
      color: "#ef4444",
      bg: "#fef2f2",
    },
    {
      text: '"Biasa aja sih, kualitas standar sesuai harga."',
      label: "😐 Neutral Sentiment",
      color: "#f59e0b",
      bg: "#fffbeb",
    },
  ];

  let currentIndex = 0;

  function rotatePreview() {
    if (!previewText || !previewBadge) return;

    // Animasikan keluar (fade out)
    previewText.style.opacity = 0;
    previewText.style.transform = "translateY(-10px)";
    previewBadge.style.opacity = 0;

    setTimeout(() => {
      currentIndex = (currentIndex + 1) % sampleReviews.length;
      const current = sampleReviews[currentIndex];

      // Ganti Konten Data
      previewText.innerText = current.text;
      previewBadge.innerText = current.label;
      previewBadge.style.color = current.color;
      previewBadge.style.backgroundColor = current.bg;

      // Animasikan masuk (fade in)
      previewText.style.opacity = 1;
      previewText.style.transform = "translateY(0)";
      previewBadge.style.opacity = 1;
    }, 400);
  }

  // Set CSS transisi dasar pada element visualizer
  if (previewText && previewBadge) {
    previewText.style.transition = "all 0.4s ease";
    previewBadge.style.transition = "all 0.4s ease";
    setInterval(rotatePreview, 4000); // Berganti setiap 4 detik
  }

  // 2. Scroll Reveal Animation menggunakan Intersection Observer API
  const revealElements = document.querySelectorAll(
    ".reveal-init, .stat-card, .naive-card, .viz-card, .pipe-step",
  );

  const observerOptions = {
    root: null,
    threshold: 0.1,
    rootMargin: "0px 0px -50px 0px",
  };

  const observer = new IntersectionObserver((entries, observer) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("reveal-visible");
        // Jika element adalah stat-card, kita bisa tambahkan delay berantai kecil
        if (entry.target.classList.contains("stat-card")) {
          entry.target.style.transitionDelay = `${Array.from(entry.target.parentNode.children).indexOf(entry.target) * 0.05}s`;
        }
        observer.unobserve(entry.target);
      }
    });
  }, observerOptions);

  revealElements.forEach((el) => {
    el.classList.add("reveal-init");
    observer.observe(el);
  });
});
