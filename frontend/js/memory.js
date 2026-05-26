App.loadMemory = async function(){
  App.$memoryContent.innerHTML='<h3>Memory Insights <button class="mem-refresh" id="memRefreshBtn">Refresh</button></h3>';
  document.getElementById('memRefreshBtn').addEventListener('click', App.loadMemory);

  try{
    var url=App.API+'/api/memory';
    if(App.activeThread) url+='?session_id='+encodeURIComponent(App.activeThread);
    var r=await fetch(url);
    var data=await r.json();

    var grid=document.createElement('div');
    grid.className='memory-grid';
    grid.appendChild(buildWorkingPanel(data.working));
    grid.appendChild(buildEpisodicPanel(data.episodic));
    grid.appendChild(buildSemanticPanel(data.semantic));
    grid.appendChild(buildProceduralPanel(data.procedural));
    App.$memoryContent.appendChild(grid);
  }catch(e){
    App.$memoryContent.innerHTML+='<div class="admin-empty">Failed to load memory data.</div>';
  }
};

function buildWorkingPanel(w){
  var panel=document.createElement('div');
  panel.className='memory-panel';
  var header=document.createElement('div');
  header.className='mp-header';
  header.innerHTML='<div class="mp-icon" style="background:var(--mt-working-bg);color:var(--mt-working-fg)">\u26A1</div>'
    +'<div class="mp-info"><div class="mp-title">Working Memory</div><div class="mp-sub">Current session context</div></div>'
    +'<div class="mp-kpi">'+(w.message_count||0)+' msg</div>';
  panel.appendChild(header);

  var body=document.createElement('div');
  body.className='mp-body';
  if(!w.message_count){
    body.innerHTML='<div class="mp-empty">Start a chat to see working memory</div>';
  } else {
    body.innerHTML=
      '<div class="mp-row"><span class="mp-label">User messages</span><span class="mp-val">'+w.user_msgs+'</span></div>'
      +'<div class="mp-row"><span class="mp-label">Assistant messages</span><span class="mp-val">'+w.assistant_msgs+'</span></div>'
      +'<div class="mp-row"><span class="mp-label">Tool messages</span><span class="mp-val">'+w.tool_msgs+'</span></div>'
      +'<div class="mp-row"><span class="mp-label">Est. tokens</span><span class="mp-val">~'+(w.estimated_tokens||0).toLocaleString()+'</span></div>';
  }
  panel.appendChild(body);
  return panel;
}

function buildEpisodicPanel(ep){
  var panel=document.createElement('div');
  panel.className='memory-panel';
  var header=document.createElement('div');
  header.className='mp-header';
  header.innerHTML='<div class="mp-icon" style="background:var(--mt-episodic-bg);color:var(--mt-episodic-fg)">\uD83D\uDCC5</div>'
    +'<div class="mp-info"><div class="mp-title">Conversation Records</div><div class="mp-sub">Chat history across all users</div></div>'
    +'<div class="mp-kpi">'+ep.total_sessions+' sessions</div>';
  panel.appendChild(header);

  var body=document.createElement('div');
  body.className='mp-body';
  body.innerHTML=
    '<div class="mp-row"><span class="mp-label">Total messages</span><span class="mp-val">'+ep.total_messages+'</span></div>'
    +'<div class="mp-row"><span class="mp-label">Avg depth</span><span class="mp-val">'+ep.avg_depth+' msg/session</span></div>';

  if(ep.sessions&&ep.sessions.length){
    var title=document.createElement('div');
    title.className='mp-section-title';
    title.textContent='Recent Sessions';
    body.appendChild(title);

    var list=document.createElement('div');
    list.className='mp-list';
    ep.sessions.slice(0,8).forEach(function(s){
      var item=document.createElement('div');
      item.className='mp-list-item';
      item.innerHTML='<span class="li-title">'+App.esc(s.title)+'</span>'
        +'<span class="li-meta">'+s.message_count+' msg &middot; '+(s.created_at||'').substring(0,10)+'</span>';
      list.appendChild(item);
    });
    body.appendChild(list);
  }
  panel.appendChild(body);
  return panel;
}

function buildSemanticPanel(sem){
  var panel=document.createElement('div');
  panel.className='memory-panel';
  var header=document.createElement('div');
  header.className='mp-header';
  header.innerHTML='<div class="mp-icon" style="background:var(--mt-semantic-bg);color:var(--mt-semantic-fg)">\uD83E\uDDE9</div>'
    +'<div class="mp-info"><div class="mp-title">Semantic Memory</div><div class="mp-sub">mem0 — facts &amp; preferences</div></div>'
    +'<div class="mp-kpi">'+sem.total_facts+' facts</div>';
  panel.appendChild(header);

  var body=document.createElement('div');
  body.className='mp-body';

  if(sem.backend){
    var bk=sem.backend;
    var st=sem.storage||{};
    var infoEl=document.createElement('div');
    infoEl.style.cssText='font-size:12px;color:var(--text-secondary);margin-bottom:10px;padding:8px 10px;background:var(--bg);border-radius:6px;border:1px solid var(--border);line-height:1.6';
    var utilPct=st.utilization_pct||0;
    var utilColor=utilPct>80?'#dc2626':utilPct>50?'#d97706':'var(--text-secondary)';
    var barHtml='<div style="display:flex;align-items:center;gap:6px;margin-top:4px">'
      +'<div style="flex:1;height:6px;background:var(--border);border-radius:3px;overflow:hidden">'
      +'<div style="width:'+Math.min(utilPct,100)+'%;height:100%;background:'+utilColor+';border-radius:3px"></div></div>'
      +'<span style="font-size:11px;color:'+utilColor+'">'+utilPct+'%</span></div>';
    infoEl.innerHTML='<strong>Backend:</strong> '+bk.engine+' &middot; '+bk.vector_store
      +'<br><strong>LLM:</strong> '+bk.llm_model
      +'<br><strong>Embedder:</strong> '+bk.embedding_model
      +'<br><strong>Storage:</strong> '+(bk.storage_exists?'\u2705':'\u274C')+' <code style="font-size:11px">'+bk.storage_path+'</code>'
      +' &middot; '+(st.size_display||'0 B')+' ('+sem.total_facts+' / '+((st.point_limit||20000).toLocaleString())+' points)'
      +barHtml;
    body.appendChild(infoEl);
  }

  if(!sem.profiles||!sem.profiles.length){
    body.innerHTML+='<div class="mp-empty">No user profiles yet. The agent remembers facts when you share them.</div>';
  } else {
    sem.profiles.forEach(function(p){
      var uTitle=document.createElement('div');
      uTitle.className='fact-user';
      uTitle.textContent=p.user_id+' ('+p.fact_count+' fact'+(p.fact_count!==1?'s':'')+')';
      body.appendChild(uTitle);
      if(p.facts&&p.facts.length){
        p.facts.forEach(function(f){
          var fEl=document.createElement('div');
          fEl.className='fact-item';
          fEl.textContent=f;
          body.appendChild(fEl);
        });
      }
    });
  }

  if(App.isAdminMode){
    var delRow=document.createElement('div');
    delRow.className='mp-delete-row';
    var sel=document.createElement('select');
    sel.innerHTML='<option value="">Select user to delete...</option><option value="__all__">All users</option>';
    var knownUsers=new Set();
    (sem.profiles||[]).forEach(function(p){knownUsers.add(p.user_id)});
    App.allUsers.forEach(function(u){knownUsers.add(u)});
    Array.from(knownUsers).sort().forEach(function(uid){
      var opt=document.createElement('option');
      opt.value=uid;
      opt.textContent=uid;
      sel.appendChild(opt);
    });
    var delBtn=document.createElement('button');
    delBtn.textContent='\u2715 Delete';
    delBtn.addEventListener('click',function(){
      var uid=sel.value;
      if(!uid){alert('Select a user first.');return}
      if(uid==='__all__'){
        if(!confirm('Delete ALL users?\n\nThis will remove:\n- All profiles and stored facts\n- All chat sessions\n- All checkpoint data\n\nThis cannot be undone.'))return;
        App.deleteAllUsers();
      } else {
        if(!confirm('Delete user "'+uid+'"?\n\nThis will remove:\n- Their profile and stored facts\n- All their chat sessions\n- All checkpoint data\n\nThis cannot be undone.'))return;
        App.deleteUser(uid);
      }
    });
    delRow.appendChild(sel);
    delRow.appendChild(delBtn);
    body.appendChild(delRow);
  }

  panel.appendChild(body);
  return panel;
}

function buildProceduralPanel(proc){
  var panel=document.createElement('div');
  panel.className='memory-panel';
  var header=document.createElement('div');
  header.className='mp-header';
  header.innerHTML='<div class="mp-icon" style="background:var(--mt-procedural-bg);color:var(--mt-procedural-fg)">\u2699\uFE0F</div>'
    +'<div class="mp-info"><div class="mp-title">Procedural Memory</div><div class="mp-sub">Tool usage &amp; patterns</div></div>'
    +'<div class="mp-kpi">'+proc.total_tool_calls+' calls</div>';
  panel.appendChild(header);

  var body=document.createElement('div');
  body.className='mp-body';
  var usage=proc.tool_usage||{};
  var toolNames=Object.keys(usage);
  if(!toolNames.length){
    body.innerHTML='<div class="mp-empty">No tool usage recorded yet.</div>';
  } else {
    var maxCount=Math.max.apply(null,toolNames.map(function(n){return usage[n]}));

    var title1=document.createElement('div');
    title1.className='mp-section-title';
    title1.textContent='Tool Usage';
    body.appendChild(title1);

    toolNames.forEach(function(name){
      var count=usage[name];
      var row=document.createElement('div');
      row.className='tool-bar-row';
      row.innerHTML='<span class="tool-bar-name">'+App.esc(name)+'</span>'
        +'<div class="tool-bar"><div class="tool-bar-fill" style="width:'+Math.round(count/maxCount*100)+'%"></div></div>'
        +'<span class="tool-bar-count">'+count+'</span>';
      body.appendChild(row);
    });

    var qtDist=proc.query_type_distribution||{};
    var qtKeys=Object.keys(qtDist);
    if(qtKeys.length){
      var title2=document.createElement('div');
      title2.className='mp-section-title';
      title2.textContent='Query Types';
      body.appendChild(title2);
      var distRow=document.createElement('div');
      distRow.className='qt-dist';
      var totalQt=qtKeys.reduce(function(s,k){return s+qtDist[k]},0);
      qtKeys.forEach(function(k){
        var pct=totalQt?Math.round(qtDist[k]/totalQt*100):0;
        var badge=document.createElement('span');
        badge.className='qt-badge';
        badge.textContent=k+': '+pct+'% ('+qtDist[k]+')';
        distRow.appendChild(badge);
      });
      body.appendChild(distRow);
    }
  }
  panel.appendChild(body);
  return panel;
}
