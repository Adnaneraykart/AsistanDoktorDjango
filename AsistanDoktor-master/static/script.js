// JavaScript dosyası: theme-toggle.js

const themeToggle = document.getElementById('theme-toggle');
const body = document.body;

// Tema durumu kontrol et ve ikonu güncelle
const setTheme = () => {
  const currentTheme = body.getAttribute('data-theme');
  if (currentTheme === 'dark') {
    body.setAttribute('data-theme', 'light');
    themeToggle.textContent = '🌞';  // Güneş emojisi
    localStorage.setItem('theme', 'light');  // Tema bilgisini localStorage'a kaydet
  } else {
    body.setAttribute('data-theme', 'dark');
    themeToggle.textContent = '🌚';  // Ay emojisi
    localStorage.setItem('theme', 'dark');  // Tema bilgisini localStorage'a kaydet
  }
};

// Sayfa yüklendiğinde tema kontrolü
const savedTheme = localStorage.getItem('theme'); // localStorage'dan temayı al

if (savedTheme) {
  // Eğer tema bilgisi varsa, onu uygula
  body.setAttribute('data-theme', savedTheme);
  themeToggle.textContent = savedTheme === 'dark' ? '🌚' : '🌞'; // Tema durumuna göre emojiyi ayarla
} else {
  // Varsayılan olarak light tema uygula
  body.setAttribute('data-theme', 'light');
  themeToggle.textContent = '🌞';  // Güneş emojisi
}

themeToggle.addEventListener('click', setTheme);

