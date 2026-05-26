App.applyTheme = function(mode){
  App.currentThemeMode = mode;
  var effective = mode;
  if(mode === 'system'){
    effective = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }
  document.documentElement.setAttribute('data-theme', effective);
  document.querySelectorAll('.theme-seg button').forEach(function(b){
    b.classList.toggle('active', b.getAttribute('data-tv') === mode);
  });
  try{localStorage.setItem('theme', mode)}catch(e){}
};

document.querySelectorAll('.theme-seg button').forEach(function(btn){
  btn.addEventListener('click', function(){
    App.applyTheme(btn.getAttribute('data-tv'));
  });
});

window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function(){
  if(App.currentThemeMode === 'system') App.applyTheme('system');
});

try{var savedTheme = localStorage.getItem('theme'); App.applyTheme(savedTheme || 'system')}catch(e){App.applyTheme('system')}
