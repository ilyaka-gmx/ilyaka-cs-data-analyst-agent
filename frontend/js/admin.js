App.loadAdmin = async function(){
  App.$adminContent.innerHTML='';
  var heading=document.createElement('div');
  heading.style.cssText='display:flex;align-items:center;gap:12px;margin-bottom:12px';
  heading.innerHTML='<h3 style="margin:0;font-size:16px;font-weight:600">Chat Tracing</h3>';
  App.$adminContent.appendChild(heading);

  var cleanup=document.createElement('div');
  cleanup.className='admin-cleanup';
  var btnCleanAll=document.createElement('button');
  btnCleanAll.textContent='\u2715 Clean All Chats';
  btnCleanAll.addEventListener('click',function(){
    if(!confirm('Delete ALL chat sessions and checkpoint data? User profiles will be preserved. This cannot be undone.'))return;
    App.deleteAllChats();
  });
  cleanup.appendChild(btnCleanAll);
  App.$adminContent.appendChild(cleanup);

  try{
    var r=await fetch(App.API+'/api/chats?all_users=true');
    var chats=await r.json();
    if(!chats.length){
      App.$adminContent.innerHTML+='<div class="admin-empty">No sessions recorded yet. Start a conversation in the Chat tab.</div>';
      return;
    }
    for(var i=0;i<chats.length;i++){
      var chat=chats[i];
      var cr=await fetch(App.API+'/api/chats/'+chat.thread_id);
      var detail=await cr.json();
      var queries=detail.queries||[];
      var hasFallback=queries.some(function(q){return q.hit_fallback});
      var cardId='ac_'+chat.thread_id.replace(/[^a-z0-9]/gi,'');

      var card=document.createElement('div');
      card.className='admin-card';

      var header=document.createElement('div');
      header.className='admin-card-header';
      header.innerHTML='<span class="arrow" id="'+cardId+'_a">&#9654;</span>'
        +(hasFallback?'&#9888;':'&#9989;')
        +' <span class="session-id">'+App.esc(chat.thread_id)+'</span> '
        +'<span class="admin-user-badge">'+App.esc(chat.user_id||'default')+'</span> '
        +App.esc(chat.title)
        +' &mdash; '+(chat.updated_at||'').substring(0,16)
        +' &middot; '+chat.query_count+' queries &middot; '+(chat.total_tokens||0).toLocaleString()+' tok';

      var delBtn=document.createElement('button');
      delBtn.className='admin-del-btn';
      delBtn.textContent='\u2715';
      delBtn.title='Delete this chat';
      (function(tid){
        delBtn.addEventListener('click',function(e){
          e.stopPropagation();
          if(!confirm('Delete this chat and its traces?'))return;
          App.deleteSingleChat(tid);
        });
      })(chat.thread_id);
      header.appendChild(delBtn);

      var body=document.createElement('div');
      body.className='admin-card-body';
      body.id=cardId+'_b';

      var info=document.createElement('div');
      info.style.cssText='color:var(--text-faint);margin-bottom:6px';
      info.innerHTML='User: '+App.esc(chat.user_id||'default');
      body.appendChild(info);

      if(chat.tags&&chat.tags.length){
        var tagRow=document.createElement('div');
        tagRow.className='tag-mgmt-row';
        tagRow.style.marginBottom='6px';
        chat.tags.forEach(function(t){
          var pill=document.createElement('span');
          pill.className='tag-rm';
          pill.innerHTML=App.esc(t)+'<span class="x">\u00d7</span>';
          (function(tid,tag){
            pill.querySelector('.x').addEventListener('click',function(){
              App.removeTagAdmin(tid,tag);
            });
          })(chat.thread_id,t);
          tagRow.appendChild(pill);
        });
        body.appendChild(tagRow);
      }

      var addDiv=document.createElement('div');
      addDiv.className='tag-add';
      addDiv.style.marginBottom='8px';
      var tagInp=document.createElement('input');
      tagInp.placeholder='Add tag...';
      var tagBtn=document.createElement('button');
      tagBtn.textContent='+';
      (function(tid,inpRef){
        tagInp.addEventListener('keydown',function(e){if(e.key==='Enter'){App.addTagAdmin(tid,inpRef)}});
        tagBtn.addEventListener('click',function(){App.addTagAdmin(tid,inpRef)});
      })(chat.thread_id,tagInp);
      addDiv.appendChild(tagInp);
      addDiv.appendChild(tagBtn);
      body.appendChild(addDiv);

      var hr=document.createElement('hr');
      hr.style.cssText='border:none;border-top:1px solid var(--border);margin:6px 0';
      body.appendChild(hr);

      queries.forEach(function(q){
        var qRow=document.createElement('div');
        qRow.className='q-row';
        var preview=q.user_message.length>60?q.user_message.substring(0,60)+'...':q.user_message;
        var qh=document.createElement('div');
        qh.className='q-header';
        qh.innerHTML=(q.hit_fallback?'&#9888;':'&#9989;')+' Q'+(q.query_index+1)+': &ldquo;'+App.esc(preview)+'&rdquo; ('+App.esc(q.query_type)+') &mdash; '+q.total_duration_ms+'ms &middot; '+(q.tokens.total||0)+' tok';
        qRow.appendChild(qh);
        if(q.tool_calls){
          q.tool_calls.forEach(function(tc){
            var t=document.createElement('div');
            t.className='q-tool';
            t.textContent='\u2192 '+(tc.name||'?')+'('+JSON.stringify(tc.args||{})+')';
            qRow.appendChild(t);
          });
        }
        if(q.final_response_preview){
          var resp=document.createElement('div');
          resp.className='q-response';
          resp.textContent='Response: '+q.final_response_preview;
          qRow.appendChild(resp);
        }
        if(q.quality_score && q.quality_score.overall){
          var qs=q.quality_score;
          var qj=document.createElement('div');
          qj.className='q-judge'+(qs.overall<3?' low':'');
          qj.textContent='\u2696\uFE0F '+qs.overall+'/5 (G:'+qs.data_grounded+' R:'+qs.addresses_question+' C:'+qs.conciseness+') \u00b7 '
            +(qs.judge_tokens?qs.judge_tokens.total:0)+' tok \u00b7 '+qs.judge_duration_ms+'ms';
          qRow.appendChild(qj);
          if(qs.issue){
            var qi=document.createElement('div');
            qi.className='q-judge-issue';
            qi.textContent=qs.issue;
            qRow.appendChild(qi);
          }
        }
        body.appendChild(qRow);
      });

      (function(hdr,bdy,arrowId){
        hdr.addEventListener('click',function(e){
          if(e.target.classList.contains('admin-del-btn'))return;
          bdy.classList.toggle('open');
          var a=document.getElementById(arrowId);
          if(a) a.classList.toggle('open');
        });
      })(header,body,cardId+'_a');

      card.appendChild(header);
      card.appendChild(body);
      App.$adminContent.appendChild(card);
    }
  }catch(e){
    App.$adminContent.innerHTML+='<div class="admin-empty">Error loading sessions.</div>';
  }
};

App.deleteSingleChat = async function(threadId){
  await fetch(App.API+'/api/chats/'+threadId,{method:'DELETE'});
  if(App.activeThread===threadId) App.startNewChat();
  await App.loadChatList();
  App.loadAdmin();
  App.loadAdminSidebar();
};

App.deleteAllChats = async function(){
  await fetch(App.API+'/api/chats',{method:'DELETE'});
  App.startNewChat();
  await App.loadChatList();
  if(App.isAdminMode){App.loadAdmin();App.loadAdminSidebar();}
};

/* ── Model Selector Widget (collapsible, inside admin sidebar) ── */

App.loadModelSelector = async function(container){
  try{
    var r=await fetch(App.API+'/api/models');
    var d=await r.json();

    var card=document.createElement('div');
    card.className='stat-card';
    card.style.cursor='pointer';

    var headerDiv=document.createElement('div');
    headerDiv.className='stat-label';
    headerDiv.style.display='flex';headerDiv.style.alignItems='center';headerDiv.style.justifyContent='space-between';
    headerDiv.innerHTML='\u2699\uFE0F Models <span class="ms-arrow" style="font-size:10px;transition:transform .15s;display:inline-block">\u25B6</span>';

    var bodyDiv=document.createElement('div');
    bodyDiv.style.display='none';
    bodyDiv.style.paddingTop='8px';
    bodyDiv.style.overflow='hidden';

    var isOpen=false;
    headerDiv.addEventListener('click',function(){
      isOpen=!isOpen;
      bodyDiv.style.display=isOpen?'block':'none';
      headerDiv.querySelector('.ms-arrow').style.transform=isOpen?'rotate(90deg)':'';
    });

    function makeDropdown(label,models,currentId,roleKey){
      var wrap=document.createElement('div');
      wrap.style.marginBottom='8px';
      var lbl=document.createElement('div');
      lbl.style.fontSize='11px';lbl.style.fontWeight='500';lbl.style.marginBottom='3px';
      lbl.textContent=label;
      wrap.appendChild(lbl);

      var sel=document.createElement('select');
      sel.style.width='100%';sel.style.fontSize='10px';sel.style.padding='4px 6px';
      sel.style.borderRadius='4px';sel.style.border='1px solid var(--border)';
      sel.style.background='var(--bg)';sel.style.color='var(--text)';

      models.forEach(function(m){
        var opt=document.createElement('option');
        opt.value=m.id;
        var stars='\u2605'.repeat(m.strength)+'\u2606'.repeat(5-m.strength);
        opt.textContent=m.short_name+' '+stars+' $'+m.cost_input.toFixed(2)+'/$'+m.cost_output.toFixed(2);
        opt.title=m.short_name+' ('+m.strength+'/5)\nInput: $'+m.cost_input.toFixed(2)+'/M | Output: $'+m.cost_output.toFixed(2)+'/M\nPros: '+m.pros.join('; ')+'\nCons: '+m.cons.join('; ');
        if(m.id===currentId) opt.selected=true;
        sel.appendChild(opt);
      });
      sel.dataset.role=roleKey;
      wrap.appendChild(sel);

      var info=document.createElement('div');
      info.className='ms-info';
      info.style.fontSize='10px';info.style.color='var(--text-faint)';info.style.marginTop='2px';info.style.wordWrap='break-word';
      function updateInfo(){
        var mid=sel.value;
        var found=models.find(function(x){return x.id===mid});
        if(!found){info.innerHTML='';return;}
        info.innerHTML='<span style="color:var(--step-tool)">+ '+found.pros.join(' | ')+'</span>'
          +'<br><span style="color:var(--text-dim)">\u2013 '+found.cons.join(' | ')+'</span>';
      }
      updateInfo();
      sel.addEventListener('change',updateInfo);
      wrap.appendChild(info);
      return wrap;
    }

    var appliedModels={agent_model:d.current_agent, judge_model:d.current_judge};

    var applyBtn=document.createElement('button');
    applyBtn.textContent='Apply';
    applyBtn.disabled=true;
    applyBtn.style.cssText='width:100%;padding:5px 0;font-size:11px;border-radius:4px;border:1px solid var(--border);background:var(--accent-bg);color:var(--text);cursor:pointer;margin-top:4px;opacity:0.4';

    function checkDirty(){
      var selects=bodyDiv.querySelectorAll('select');
      var changed=false;
      selects.forEach(function(s){
        if(s.value!==appliedModels[s.dataset.role]) changed=true;
      });
      applyBtn.disabled=!changed;
      applyBtn.style.opacity=changed?'1':'0.4';
      applyBtn.style.cursor=changed?'pointer':'default';
    }

    bodyDiv.appendChild(makeDropdown('Agent Model',d.agent_models,d.current_agent,'agent_model'));
    bodyDiv.appendChild(makeDropdown('Judge Model',d.judge_models,d.current_judge,'judge_model'));

    bodyDiv.querySelectorAll('select').forEach(function(s){
      s.addEventListener('change',checkDirty);
    });

    applyBtn.addEventListener('click',async function(){
      if(applyBtn.disabled) return;
      var selects=bodyDiv.querySelectorAll('select');
      var payload={};
      selects.forEach(function(s){payload[s.dataset.role]=s.value;});
      applyBtn.disabled=true;applyBtn.textContent='Applying...';applyBtn.style.opacity='0.4';
      try{
        var resp=await fetch(App.API+'/api/models',{
          method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)
        });
        var result=await resp.json();
        if(result.ok){
          appliedModels.agent_model=result.agent_model;
          appliedModels.judge_model=result.judge_model;
          applyBtn.textContent='\u2714 Applied';
          setTimeout(function(){applyBtn.textContent='Apply';checkDirty();},1500);
          App.refreshMeta();
          App.loadAdminSidebar();
        }else{
          applyBtn.textContent='Error';
          setTimeout(function(){applyBtn.textContent='Apply';checkDirty();},1500);
        }
      }catch(e){
        applyBtn.textContent='Error';
        setTimeout(function(){applyBtn.textContent='Apply';checkDirty();},1500);
      }
    });
    bodyDiv.appendChild(applyBtn);

    card.appendChild(headerDiv);
    card.appendChild(bodyDiv);
    container.appendChild(card);
  }catch(e){/* models API not available, skip */}
};

App.loadAdminSidebar = async function(){
  try{
    var r=await fetch(App.API+'/api/admin/stats');
    var d=await r.json();
    App.$adminSidebar.innerHTML='';

    var c1=document.createElement('div');c1.className='stat-card';
    c1.innerHTML='<div class="stat-label">Users</div><div class="stat-val">'+d.total_users+'</div>'
      +'<div class="stat-detail">'+d.users.join(', ')+'</div>';
    App.$adminSidebar.appendChild(c1);

    var c2=document.createElement('div');c2.className='stat-card';
    c2.innerHTML='<div class="stat-label">Sessions</div><div class="stat-val">'+d.total_sessions+'</div>';
    var perUser=d.sessions_per_user||{};
    var puKeys=Object.keys(perUser);
    if(puKeys.length){
      var detail=document.createElement('div');detail.className='stat-detail';
      puKeys.forEach(function(u){
        var su=document.createElement('div');su.className='su';
        su.textContent=u+': '+perUser[u]+' session'+(perUser[u]!==1?'s':'');
        detail.appendChild(su);
      });
      c2.appendChild(detail);
    }
    App.$adminSidebar.appendChild(c2);

    var c3=document.createElement('div');c3.className='stat-card';
    c3.innerHTML='<div class="stat-label">Semantic Memory</div><div class="stat-val">'+d.total_profiles+'</div>'
      +'<div class="stat-detail">'+d.total_facts+' fact'+(d.total_facts!==1?'s':'')+' in mem0</div>';
    App.$adminSidebar.appendChild(c3);

    var c4=document.createElement('div');c4.className='stat-card';
    c4.innerHTML='<div class="stat-label">Queries</div><div class="stat-val">'+d.total_queries+'</div>';
    App.$adminSidebar.appendChild(c4);

    if(d.quality && d.quality.scored > 0){
      var c5=document.createElement('div');c5.className='stat-card';
      c5.innerHTML='<div class="stat-label">\u2696\uFE0F Scored</div><div class="stat-val">'+d.quality.scored+' of '+d.quality.total+'</div>'
        +'<div class="stat-detail">'
        +'<div class="su">\u25CF\u25CF\u25CF High (4-5): '+d.quality.high+'</div>'
        +'<div class="su">\u25CF\u25CF Med (3-4): '+d.quality.med+'</div>'
        +'<div class="su">\u25CF Low (&lt;3): '+d.quality.low+'</div>'
        +'</div>';
      App.$adminSidebar.appendChild(c5);
    }

    if(d.costs){
      var at=d.costs.agent_tokens||{};
      var jt=d.costs.judge_tokens||{};
      var c6=document.createElement('div');c6.className='stat-card';
      c6.innerHTML='<div class="stat-label">\uD83D\uDCB0 Costs &amp; Tokens</div>'
        +'<div class="stat-detail">'
        +'<div class="su">Agent: $'+d.costs.agent.toFixed(4)+' \u2022 '+(at.total||0).toLocaleString()+' tok</div>'
        +(d.costs.judge>0?'<div class="su">Judge: $'+d.costs.judge.toFixed(4)+' \u2022 '+(jt.total||0).toLocaleString()+' tok</div>':'')
        +'</div>';
      App.$adminSidebar.appendChild(c6);
    }

    App.loadModelSelector(App.$adminSidebar);
  }catch(e){
    App.$adminSidebar.innerHTML='<div style="padding:12px;font-size:12px;color:var(--text-faint)">Stats unavailable</div>';
  }
};

App.deleteUser = async function(userId){
  await fetch(App.API+'/api/users/'+encodeURIComponent(userId),{method:'DELETE'});
  await App.loadChatList();
  App.loadAdmin();
  App.loadAdminSidebar();
  App.loadMemory();
  App.loadUsers();
  App._usersLoaded=false;
};

App.deleteAllUsers = async function(){
  await fetch(App.API+'/api/users',{method:'DELETE'});
  App.startNewChat();
  await App.loadChatList();
  App.loadAdmin();
  App.loadAdminSidebar();
  App.loadMemory();
  App.loadUsers();
  App._usersLoaded=false;
};

App.addTagAdmin = async function(threadId,inpEl){
  var tag=inpEl.value.trim().toLowerCase();
  if(!tag)return;
  await fetch(App.API+'/api/chats/'+threadId+'/tags',{
    method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({tag:tag})
  });
  inpEl.value='';
  App.loadAdmin();
};

App.removeTagAdmin = async function(threadId,tag){
  await fetch(App.API+'/api/chats/'+threadId+'/tags/'+encodeURIComponent(tag),{method:'DELETE'});
  App.loadAdmin();
};
