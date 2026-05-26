App.setMode = function(admin){
  App.isAdminMode = admin;
  App.$tabChat.style.display = admin ? 'none' : '';
  App.$tabAdmin.style.display = admin ? '' : 'none';
  App.$tabMemory.style.display = admin ? '' : 'none';
  App.$adminModeBtn.textContent = admin ? '\u2190 Back to Chat' : '\u2699 Admin';
  App.$adminModeBtn.classList.toggle('active', admin);

  App.$sbUser.style.display = admin ? 'none' : '';
  App.$sbActions.style.display = admin ? 'none' : '';
  App.$sbSearch.style.display = admin ? 'none' : '';
  App.$sbChats.style.display = admin ? 'none' : '';
  App.$tagFilter.style.display = admin ? 'none' : '';
  App.$sbSections.forEach(function(s){s.style.display = admin ? 'none' : ''});
  App.$recToggles.style.display = admin ? 'none' : '';
  App.$adminSidebar.style.display = admin ? '' : 'none';

  document.querySelectorAll('.tab-btn').forEach(function(b){b.classList.remove('active')});
  document.querySelectorAll('.tab-content').forEach(function(c){c.classList.remove('active')});

  if(admin){
    App.$tabAdmin.classList.add('active');
    document.getElementById('adminTab').classList.add('active');
    App.$exportBtn.style.display = 'none';
    App.loadAdmin();
    App.loadAdminSidebar();
  } else {
    App.$tabChat.classList.add('active');
    document.getElementById('chatTab').classList.add('active');
    App.$exportBtn.style.display = '';
  }
};

App.$adminModeBtn.addEventListener('click', function(){ App.setMode(!App.isAdminMode) });

App.updateExportVisibility = function(tab){
  App.$exportBtn.style.display = (tab === 'chat') ? '' : 'none';
};

document.querySelectorAll('.tab-btn').forEach(function(btn){
  btn.addEventListener('click',function(){
    document.querySelectorAll('.tab-btn').forEach(function(b){b.classList.remove('active')});
    document.querySelectorAll('.tab-content').forEach(function(c){c.classList.remove('active')});
    btn.classList.add('active');
    var tab = btn.dataset.tab;
    document.getElementById(tab+'Tab').classList.add('active');
    App.updateExportVisibility(tab);
    if(tab==='admin') App.loadAdmin();
    if(tab==='memory') App.loadMemory();
  });
});

App.updateExportVisibility('chat');

/* ── Sidebar panels ── */
document.getElementById('tagToggle').addEventListener('click',function(){
  var arrow=document.getElementById('tagsArrow');
  arrow.classList.toggle('open');
  App.$tagsPanel.classList.toggle('open');
  if(App.$tagsPanel.classList.contains('open')) App.renderTagPanel();
});

/* ── Recommendation toggles ── */
var $autoRecToggle = document.getElementById('autoRecToggle');
var $pastSessionsToggle = document.getElementById('pastSessionsToggle');
$autoRecToggle.checked = App.autoRecEnabled;
$pastSessionsToggle.checked = App.usePastSessions;
$autoRecToggle.addEventListener('change', function(){
  App.autoRecEnabled = this.checked;
  localStorage.setItem('autoRec', App.autoRecEnabled);
});
$pastSessionsToggle.addEventListener('change', function(){
  App.usePastSessions = this.checked;
  localStorage.setItem('pastSessions', App.usePastSessions);
});
var $qualityScoringToggle = document.getElementById('qualityScoringToggle');
$qualityScoringToggle.checked = App.qualityScoringEnabled;
$qualityScoringToggle.addEventListener('change', function(){
  App.qualityScoringEnabled = this.checked;
  localStorage.setItem('qualityScoring', App.qualityScoringEnabled);
});
var $reflectionToggle = document.getElementById('reflectionToggle');
$reflectionToggle.checked = App.reflectionEnabled;
$reflectionToggle.addEventListener('change', function(){
  App.reflectionEnabled = this.checked;
  localStorage.setItem('reflection', App.reflectionEnabled);
  fetch(App.API+'/api/reflection',{
    method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({enabled:App.reflectionEnabled})
  });
});
App.$reflectionToggle = $reflectionToggle;

var $decompositionToggle = document.getElementById('decompositionToggle');
$decompositionToggle.checked = App.decompositionEnabled;
$decompositionToggle.addEventListener('change', function(){
  App.decompositionEnabled = this.checked;
  localStorage.setItem('decomposition', App.decompositionEnabled);
  fetch(App.API+'/api/decomposition',{
    method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({enabled:App.decompositionEnabled})
  });
});
App.$decompositionToggle = $decompositionToggle;

/* ── Export ── */
App.$exportBtn.addEventListener('click',function(){
  if(!App.chatMessages.length){alert('No messages to export.');return}
  var lines=['# Conversation Export\n'];
  App.chatMessages.forEach(function(m){
    lines.push((m.role==='user'?'**User**: ':'**Agent**: ')+m.content+'\n');
  });
  var blob=new Blob([lines.join('\n')],{type:'text/markdown'});
  var a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='conversation.md';a.click();
});

/* ── Welcome screen ── */
App.showWelcome = function(){
  App.$messages.innerHTML='';
  var w=document.createElement('div');
  w.className='welcome';
  w.id='welcomeScreen';
  w.innerHTML='<h2>What would you like to know?</h2>'
    +'<p>Ask about categories, intents, distributions, examples, or anything in the customer service dataset.</p>';
  var chips=document.createElement('div');
  chips.className='chips';
  App.SUGGESTIONS.forEach(function(s){
    var c=document.createElement('div');
    c.className='chip';
    c.textContent=s.label;
    c.addEventListener('click',function(){App.$queryInput.value=s.query;App.sendQuery()});
    chips.appendChild(c);
  });
  w.appendChild(chips);
  App.$messages.appendChild(w);
};

App.showSuggestionChips = function(recommendations){
  var old=document.getElementById('suggestionsRow');
  if(old)old.remove();
  var row=document.createElement('div');
  row.id='suggestionsRow';
  row.className='chips';
  row.style.padding='4px 0 8px';

  if(recommendations && recommendations.length){
    recommendations.forEach(function(r){
      var c=document.createElement('div');
      c.className='chip rec';
      c.textContent=r.query.length>60 ? r.query.substring(0,57)+'...' : r.query;
      if(r.reason) c.title=r.reason;
      c.addEventListener('click',function(){App.$queryInput.value=r.query;App.sendQuery()});
      row.appendChild(c);
    });
  } else {
    var picks=App.SUGGESTIONS.slice(0,3);
    picks.forEach(function(s){
      var c=document.createElement('div');
      c.className='chip';
      c.textContent=s.label;
      c.addEventListener('click',function(){App.$queryInput.value=s.query;App.sendQuery()});
      row.appendChild(c);
    });
  }

  App.$messages.appendChild(row);
  App.scrollBottom();
};

App.fetchRecommendations = async function(){
  App.showSuggestionChips();
  try{
    var url=App.API+'/api/recommend?user_id='+encodeURIComponent(App.getUser())
      +'&use_past_sessions='+App.usePastSessions;
    if(App.activeThread) url+='&session_id='+encodeURIComponent(App.activeThread);
    var resp=await fetch(url);
    var data=await resp.json();
    if(data.recommendations && data.recommendations.length){
      App.showSuggestionChips(data.recommendations);
    }
  }catch(e){
    console.warn('Recommendation fetch failed:',e);
  }
};

/* ── New Chat ── */
App.startNewChat = function(){
  App.activeThread=App.makeId();
  App.chatMessages=[];
  App.showWelcome();
  App.renderChatList();
  App.renderTagPanel();
  App.$queryInput.value='';
  App.$sendBtn.disabled=false;
  App.$queryInput.focus();
};
document.getElementById('btnNew').addEventListener('click',App.startNewChat);
