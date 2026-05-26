App.switchToChat = async function(threadId){
  App.activeThread=threadId;
  App.chatMessages=[];
  App.renderChatList();
  try{
    var r=await fetch(App.API+'/api/chats/'+threadId);
    var data=await r.json();
    App.chatMessages=data.messages||[];
    App.renderLoadedMessages(data.queries||[]);
  }catch(e){
    App.$messages.innerHTML='<div style="padding:20px;color:var(--red)">Failed to load chat.</div>';
  }
  App.renderTagPanel();
};

App.renderLoadedMessages = function(queries){
  App.$messages.innerHTML='';
  var qIdx = 0;
  App.chatMessages.forEach(function(m){
    if(m.role==='user'){
      App.appendUserBubble(m.content);
    } else {
      var q = queries[qIdx] || null;
      if(q) qIdx++;
      var answerLike = q ? {
        query_type: q.query_type,
        tokens: q.tokens,
        duration_ms: q.total_duration_ms,
        tool_count: (q.tool_calls||[]).length,
        memory_ops: q.memory_ops || {}
      } : null;
      var bEl=App.appendBotBubble(m.content, q?q.tool_calls:[], answerLike);
      if(q && q.quality_score && q.quality_score.overall) App.renderJudgeCard(q.quality_score, bEl);
    }
  });
  if(App.autoRecEnabled && App.chatMessages.length > 0){
    App.fetchRecommendations();
  } else {
    App.showSuggestionChips();
  }
  App.scrollBottom();
};

App.$queryInput.addEventListener('keydown',function(e){if(e.key==='Enter'&&!e.shiftKey)App.sendQuery()});
App.$sendBtn.addEventListener('click',function(){App.sendQuery()});

App.sendQuery = async function(){
  if(App.isSending)return;
  var q=App.$queryInput.value.trim();
  if(!q)return;

  App.isSending=true;
  App.$queryInput.value='';
  App.$sendBtn.disabled=true;

  var welcome=document.getElementById('welcomeScreen');
  if(welcome)welcome.remove();
  var oldSug=document.getElementById('suggestionsRow');
  if(oldSug)oldSug.remove();

  App.appendUserBubble(q);
  App.chatMessages.push({role:'user',content:q});

  var typingEl=App.showTyping();
  App.scrollBottom();

  var steps=[];
  var finalText='';
  var answerData=null;
  var qualityData=null;
  var liveStepsEl=null;

  try{
    var resp=await fetch(App.API+'/api/chat',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({query:q,session_id:App.activeThread,user_id:App.getUser(),use_past_sessions:App.usePastSessions,quality_scoring:App.qualityScoringEnabled})
    });

    var reader=resp.body.getReader();
    var decoder=new TextDecoder();
    var buffer='';
    var eventType=null;

    while(true){
      var chunk=await reader.read();
      if(chunk.done)break;
      buffer+=decoder.decode(chunk.value,{stream:true});
      var lines=buffer.split('\n');
      buffer=lines.pop();
      for(var i=0;i<lines.length;i++){
        var line=lines[i];
        if(line.startsWith('event: ')){eventType=line.substring(7).trim();continue}
        if(line.startsWith('data: ')&&eventType){
          try{
            var d=JSON.parse(line.substring(6));
            if(eventType==='route'){steps.push({type:'route',query_type:d.query_type})}
            else if(eventType==='decompose'){steps.push({type:'decompose',plan:d.plan})}
            else if(eventType==='tool_call'){steps.push({type:'tool_call',name:d.name,args:d.args})}
            else if(eventType==='tool_result'){steps.push({type:'tool_result',name:d.name,content:d.content})}
            else if(eventType==='answer'){answerData=d;finalText=d.text}
            else if(eventType==='quality'){qualityData=d}
            else if(eventType==='error'){finalText='Error: '+d.message}
            if(eventType!=='answer'){
              liveStepsEl=App.updateLiveSteps(liveStepsEl,steps);
              App.scrollBottom();
            }
          }catch(e){}
          eventType=null;
        }
      }
    }
  }catch(e){
    finalText='Connection error. Please try again.';
  }

  if(typingEl)typingEl.remove();
  if(liveStepsEl)liveStepsEl.remove();

  var botEl=App.appendBotBubble(finalText,steps,answerData);
  if(qualityData) App.renderJudgeCard(qualityData, botEl);
  App.chatMessages.push({role:'assistant',content:finalText});

  App.scrollBottom();
  App.isSending=false;
  App.$sendBtn.disabled=false;
  App.$queryInput.focus();

  if(App.autoRecEnabled){
    App.fetchRecommendations();
  } else {
    App.showSuggestionChips();
  }
  await App.loadChatList();
  App.renderTagPanel();
  App.refreshMeta();
};

/* ── Message rendering helpers ── */

App.appendUserBubble = function(text){
  var el=document.createElement('div');
  el.className='msg user';
  el.textContent=text;
  App.$messages.appendChild(el);
  App.scrollBottom();
};

App.getMemoryBadges = function(steps, memOps){
  var mo = memOps || {};
  var ctxLabel = mo.context_messages ? ' ('+mo.context_messages+' msg)' : '';
  var badges=[
    {type:'working', label:'Working'+ctxLabel},
    {type:'episodic', label:'Episodic'}
  ];
  var toolCalls=(steps||[]).filter(function(s){return s.type==='tool_call'});
  if(toolCalls.length){
    badges.push({type:'procedural',label:'Procedural: '+toolCalls.length+' tool'+(toolCalls.length>1?'s':'')});
  }
  var semTools = (mo.semantic_tools && mo.semantic_tools.length) ? mo.semantic_tools : [];
  var semFromSteps = toolCalls.filter(function(s){return s.name==='remember_fact'||s.name==='recall_profile'});
  if(semTools.length){
    var ops = semTools.map(function(n){return n==='remember_fact'?'write':'read'});
    badges.push({type:'semantic',label:'Semantic: '+ops.join(', ')});
  } else if(semFromSteps.length){
    var ops = semFromSteps.map(function(s){return s.name==='remember_fact'?'write':'read'});
    badges.push({type:'semantic',label:'Semantic: '+ops.join(', ')});
  }
  return badges;
};

App.appendBotBubble = function(text,steps,answerData){
  var el=document.createElement('div');
  el.className='msg bot';

  var content=document.createElement('span');
  content.textContent=text||'No response.';
  el.appendChild(content);

  if(steps&&steps.length){
    var togId='r'+App.makeId();
    var toggle=document.createElement('div');
    toggle.className='reasoning-toggle';
    toggle.textContent='Show reasoning ('+steps.length+' steps)';

    var reasoningBox=document.createElement('div');
    reasoningBox.className='reasoning-content';
    reasoningBox.id=togId;

    steps.forEach(function(s){
      var stepEl=document.createElement('div');
      stepEl.className='step';
      if(s.type==='route'){stepEl.classList.add('route');stepEl.textContent='Route: '+(s.query_type||'')}
      else if(s.type==='decompose'){stepEl.classList.add('route');stepEl.textContent='Plan: '+(s.plan||'')}
      else if(s.type==='tool_call'){stepEl.classList.add('tool');stepEl.textContent=s.name+'('+JSON.stringify(s.args||{})+')'}
      else if(s.type==='tool_result'){stepEl.classList.add('result');stepEl.textContent=(s.content||'').substring(0,200)}
      reasoningBox.appendChild(stepEl);
    });

    toggle.addEventListener('click',function(){
      reasoningBox.classList.toggle('open');
      toggle.textContent=(reasoningBox.classList.contains('open')?'Hide':'Show')+' reasoning ('+steps.length+' steps)';
    });
    el.appendChild(toggle);
    el.appendChild(reasoningBox);
  }

  var memOps = answerData ? (answerData.memory_ops || {}) : {};
  var badges=App.getMemoryBadges(steps, memOps);
  if(badges.length){
    var traceEl=document.createElement('div');
    traceEl.className='memory-trace';
    badges.forEach(function(b){
      var badge=document.createElement('span');
      badge.className='mt-badge '+b.type;
      badge.textContent=b.label;
      traceEl.appendChild(badge);
    });
    el.appendChild(traceEl);
  }

  if(answerData){
    var footer=document.createElement('div');
    footer.className='msg-footer';
    var badge={structured:'struct',unstructured:'open',out_of_scope:'oos'}[answerData.query_type]||'';
    var tok=(answerData.tokens||{}).total||0;
    var dur=answerData.duration_ms||0;
    var tc=answerData.tool_count||0;
    footer.textContent=badge+' \u00b7 '+tok.toLocaleString()+' tok \u00b7 '+(dur/1000).toFixed(1)+'s \u00b7 '+tc+' calls';
    el.appendChild(footer);
  }

  App.$messages.appendChild(el);
  return el;
};

App.showTyping = function(){
  var el=document.createElement('div');
  el.className='typing-indicator visible';
  el.innerHTML='<div class="dot"></div><div class="dot"></div><div class="dot"></div><span class="typing-label">Thinking...</span>';
  App.$messages.appendChild(el);
  return el;
};

App.updateLiveSteps = function(existing,steps){
  if(!existing){
    existing=document.createElement('div');
    existing.className='msg bot';
    existing.style.opacity='0.7';
    App.$messages.appendChild(existing);
  }
  existing.innerHTML='';
  steps.forEach(function(s){
    var stepEl=document.createElement('div');
    stepEl.className='step';
    if(s.type==='route'){stepEl.classList.add('route');stepEl.textContent='Route: '+(s.query_type||'')}
    else if(s.type==='decompose'){stepEl.classList.add('route');stepEl.textContent='Plan: '+(s.plan||'')}
    else if(s.type==='tool_call'){stepEl.classList.add('tool');stepEl.textContent=s.name+'('+JSON.stringify(s.args||{})+')'}
    else if(s.type==='tool_result'){stepEl.classList.add('result');stepEl.textContent=(s.content||'').substring(0,200)}
    existing.appendChild(stepEl);
  });
  return existing;
};
