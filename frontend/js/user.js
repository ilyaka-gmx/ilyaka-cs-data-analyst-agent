try{var savedUser=localStorage.getItem('userId');if(savedUser)App.$userId.value=savedUser}catch(e){}

App.selectUser = function(u){
  App.$userId.value = u;
  App.$userDropdown.classList.remove('open');
  try{localStorage.setItem('userId', u)}catch(ex){}
  App.activeTagFilter = null;
  App.startNewChat();
  App.loadChatList();
};

App.renderUserDropdown = function(showAll){
  App.$userDropdown.innerHTML='';
  var current = App.getUser();
  var filter = showAll ? '' : (App.$userId.value||'').trim().toLowerCase();
  var filtered = App.allUsers.filter(function(u){
    return !filter || u.toLowerCase().indexOf(filter) >= 0;
  });
  if(!filtered.length && filter && App.allUsers.length){
    var hint = document.createElement('div');
    hint.className = 'user-dropdown-item';
    hint.style.color = 'var(--text-faint)';
    hint.textContent = 'Press Enter to create "' + filter + '"';
    App.$userDropdown.appendChild(hint);
    return;
  }
  filtered.forEach(function(u){
    var item = document.createElement('div');
    item.className = 'user-dropdown-item' + (u === current ? ' active' : '');
    item.textContent = u;
    item.addEventListener('mousedown', function(e){
      e.preventDefault();
      e.stopPropagation();
      App.selectUser(u);
    });
    App.$userDropdown.appendChild(item);
  });
};

App.$userId.addEventListener('focus', function(){
  if(App._usersLoaded && App.allUsers.length){
    App.renderUserDropdown(true);
    App.$userDropdown.classList.add('open');
  } else {
    App.loadUsers().then(function(){
      App._usersLoaded = true;
      App.renderUserDropdown(true);
      App.$userDropdown.classList.add('open');
    });
  }
});
App.$userId.addEventListener('blur', function(){
  setTimeout(function(){App.$userDropdown.classList.remove('open')}, 200);
});
App.$userId.addEventListener('input', function(){
  App.renderUserDropdown();
  if(App.allUsers.length) App.$userDropdown.classList.add('open');
});
App.$userId.addEventListener('keydown', function(e){
  if(e.key === 'Enter'){
    App.$userDropdown.classList.remove('open');
    App.$userId.blur();
    try{localStorage.setItem('userId', App.$userId.value.trim())}catch(ex){}
    App.activeTagFilter = null;
    App._usersLoaded = false;
    App.startNewChat();
    App.loadChatList();
    App.loadUsers();
  }
});
