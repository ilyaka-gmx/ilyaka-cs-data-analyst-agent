App.healthPopoverEl = null;

App.loadHealth = async function(){
  try{
    var r=await fetch(App.API+'/api/health');
    var d=await r.json();
    var dot={healthy:'\uD83D\uDFE2',warning:'\uD83D\uDFE1',error:'\uD83D\uDD34',loading:'\u26AA'}[d.status]||'\u26AA';
    var label=d.status==='loading'?'Loading...':d.status.charAt(0).toUpperCase()+d.status.slice(1);
    App.$healthSeg.textContent=dot+' '+label;

    if(App.healthPopoverEl){App.healthPopoverEl.remove();App.healthPopoverEl=null}
    if(d.checks&&d.checks.length){
      App.healthPopoverEl=document.createElement('div');
      App.healthPopoverEl.className='health-popover';
      d.checks.forEach(function(c){
        var icon={pass:'\u2705',warn:'\u26A0\uFE0F',fail:'\u274C'}[c.status]||'';
        var row=document.createElement('div');
        row.className='hc';
        row.textContent=icon+' '+c.name+': '+c.status;
        App.healthPopoverEl.appendChild(row);
        var msg=document.createElement('div');
        msg.className='hc-msg';
        msg.textContent=c.message;
        App.healthPopoverEl.appendChild(msg);
      });
      var refreshBtn=document.createElement('button');
      refreshBtn.className='refresh-btn';
      refreshBtn.textContent='Refresh';
      refreshBtn.addEventListener('click',function(e){
        e.stopPropagation();
        fetch(App.API+'/api/health/refresh',{method:'POST'});
        App.$healthSeg.textContent='\u26AA Refreshing...';
        setTimeout(App.loadHealth,3000);
      });
      App.healthPopoverEl.appendChild(refreshBtn);
      App.$healthSeg.appendChild(App.healthPopoverEl);
    }

    if(d.status==='loading'){
      setTimeout(App.loadHealth,2000);
    }
  }catch(e){
    App.$healthSeg.textContent='\u26AA offline';
  }
};

App.$healthSeg.addEventListener('click',function(e){
  if(App.healthPopoverEl) App.healthPopoverEl.classList.toggle('open');
});
document.addEventListener('click',function(e){
  if(App.healthPopoverEl && !App.$healthSeg.contains(e.target)){
    App.healthPopoverEl.classList.remove('open');
  }
});

App.refreshMeta = async function(){
  try{
    var r=await fetch(App.API+'/api/meta');
    var d=await r.json();
    App.$datasetSeg.textContent=d.dataset.row_count.toLocaleString()+' rows \u00b7 '+d.dataset.num_categories+' categories \u00b7 '+d.dataset.num_intents+' intents';

    var hasJudge=App.qualityScoringEnabled && d.judge && d.judge.tokens && d.judge.tokens.total>0;
    if(hasJudge){
      App.$tokenSeg.innerHTML='<span class="status-group">Agent: '+d.tokens.total.toLocaleString()+' tok \u00b7 ~$'+d.tokens.cost.toFixed(4)+' \u00b7 '+App.esc(d.model_short)+'</span>'
        +'<span class="status-group">\u2696\uFE0F Judge: '+d.judge.tokens.total.toLocaleString()+' tok \u00b7 ~$'+d.judge.cost.toFixed(4)+' \u00b7 '+App.esc(d.judge.model_short)+'</span>';
      App.$modelSeg.textContent='';
    } else {
      App.$tokenSeg.textContent=d.tokens.total.toLocaleString()+' tok \u00b7 ~$'+d.tokens.cost.toFixed(4);
      App.$modelSeg.textContent=d.model_short;
    }
  }catch(e){}
};

App.loadUsers = async function(){
  try{
    var r=await fetch(App.API+'/api/users');
    var d=await r.json();
    App.allUsers=d.users||[];
    var current=App.getUser();
    if(current && current !== 'default' && App.allUsers.indexOf(current)<0){
      App.allUsers.push(current);
      App.allUsers.sort();
    }
    App._usersLoaded = true;
  }catch(e){App.allUsers=[];}
};

/* ── INIT ── */
App.init = function(){
  App.startNewChat();
  App.loadChatList();
  App.loadHealth();
  App.refreshMeta();
  App.loadUsers();

  fetch(App.API+'/api/reflection').then(function(r){return r.json()}).then(function(d){
    App.reflectionEnabled=d.enabled;
    App.$reflectionToggle.checked=App.reflectionEnabled;
    localStorage.setItem('reflection',App.reflectionEnabled);
  }).catch(function(){});

  fetch(App.API+'/api/decomposition').then(function(r){return r.json()}).then(function(d){
    App.decompositionEnabled=d.enabled;
    App.$decompositionToggle.checked=App.decompositionEnabled;
    localStorage.setItem('decomposition',App.decompositionEnabled);
  }).catch(function(){});
};
