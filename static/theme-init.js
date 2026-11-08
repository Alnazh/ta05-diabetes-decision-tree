/* Dijalankan di <head> sebelum halaman digambar supaya tidak ada kedipan tema.
   Setiap kunjungan baru selalu mulai dengan tema terang. Pilihan gelap hanya bertahan
   selama sesi browser, supaya tetap gelap saat berpindah halaman. */
try {
  localStorage.removeItem("theme"); /* hapus sisa pilihan lama yang dulu disimpan permanen */
  if (sessionStorage.getItem("theme") === "dark") {
    document.documentElement.dataset.theme = "dark";
  }
} catch (error) {
  /* penyimpanan diblokir: tetap pakai tema terang */
}
