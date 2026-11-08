"use strict";

(function () {
  var root = document.documentElement;

  function applyTheme(theme) {
    var button = document.getElementById("themeToggle");
    var isDark = theme === "dark";

    root.dataset.theme = theme;
    if (!button) return;

    button.setAttribute("aria-pressed", String(isDark));
    button.setAttribute("aria-label", isDark ? "Ganti ke tema terang" : "Ganti ke tema gelap");
    button.querySelector(".tt-label").textContent = isDark ? "Tema terang" : "Tema gelap";
  }

  function saveTheme(theme) {
    try {
      sessionStorage.setItem("theme", theme);
    } catch (error) {
      /* penyimpanan diblokir: pilihan hanya berlaku di halaman ini */
    }
  }

  function setupThemeToggle() {
    var button = document.getElementById("themeToggle");

    applyTheme(root.dataset.theme === "dark" ? "dark" : "light");
    if (!button) return;

    button.addEventListener("click", function () {
      var next = root.dataset.theme === "dark" ? "light" : "dark";
      applyTheme(next);
      saveTheme(next);
    });
  }

  function setupExampleButton() {
    var button = document.getElementById("fillExample");
    if (!button) return;

    var example = JSON.parse(button.dataset.example);
    button.addEventListener("click", function () {
      Object.keys(example).forEach(function (name) {
        var input = document.getElementById("f-" + name);
        if (input) input.value = example[name];
      });
    });
  }

  function placeMeterTicks() {
    document.querySelectorAll(".meter[data-threshold]").forEach(function (meter) {
      var tick = meter.querySelector(".meter-tick");
      tick.style.left = meter.dataset.threshold + "%";
      tick.classList.add("is-placed");
    });
  }

  function setupMobileMenu() {
    var header = document.querySelector(".topbar");
    var button = document.getElementById("menuToggle");
    if (!header || !button) return;

    function setOpen(open) {
      header.classList.toggle("is-open", open);
      button.setAttribute("aria-expanded", String(open));
      button.setAttribute("aria-label", open ? "Tutup menu" : "Buka menu");
    }

    button.addEventListener("click", function () {
      setOpen(!header.classList.contains("is-open"));
    });
    header.querySelectorAll(".nav a").forEach(function (link) {
      link.addEventListener("click", function () { setOpen(false); });
    });
    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape") setOpen(false);
    });
  }

  function centerWideFigures() {
    /* Pohon yang lebar digeser ke tengah supaya akarnya langsung terlihat di layar kecil. */
    document.querySelectorAll(".tree-scroll").forEach(function (scroller) {
      scroller.scrollLeft = (scroller.scrollWidth - scroller.clientWidth) / 2;
    });
  }

  function focusFeedback() {
    /* Setelah kirim formulir: tunjukkan kolom yang salah, atau hasil jika sudah ada. */
    var invalid = document.querySelector("[aria-invalid='true']");
    var result = document.querySelector("#hasil[data-has-result]");
    if (invalid) {
      invalid.focus();
    } else if (result) {
      result.scrollIntoView();
    }
  }

  setupThemeToggle();
  setupMobileMenu();
  setupExampleButton();
  placeMeterTicks();
  window.addEventListener("load", centerWideFigures);
  focusFeedback();
})();
