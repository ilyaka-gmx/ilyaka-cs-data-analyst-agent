App.loadAllTags = async function(){
  try{
    var r=await fetch(App.API+'/api/tags?user_id='+encodeURIComponent(App.getUser()));
    var d=await r.json();
    App.allTags=d.tags||[];
  }catch(e){App.allTags=[];}
  App.renderTagFilter();
};

App.renderTagFilter = function(){
  if(!App.allTags.length){App.$tagFilter.innerHTML='';return}
  var wrap=document.createElement('div');
  wrap.className='sb-tag-filter';
  App.allTags.forEach(function(tag){
    var pill=document.createElement('span');
    pill.className='tag-pill'+(App.activeTagFilter===tag?' active':'');
    pill.textContent=tag;
    pill.addEventListener('click',function(){
      if(App.activeTagFilter===tag){App.activeTagFilter=null}
      else{App.activeTagFilter=tag}
      App.loadChatList();
    });
    wrap.appendChild(pill);
  });
  App.$tagFilter.innerHTML='';
  App.$tagFilter.appendChild(wrap);
};

App.renderTagPanel = function(){
  if(!App.activeThread){
    App.$tagsPanel.innerHTML='<div style="font-size:12px;color:var(--text-faint)">Start a chat to manage tags</div>';
    return;
  }
  var chat=App.allChats.find(function(c){return c.thread_id===App.activeThread});
  var tags=chat?chat.tags:[];

  App.$tagsPanel.innerHTML='';

  if(tags.length){
    var row=document.createElement('div');
    row.className='tag-mgmt-row';
    tags.forEach(function(t){
      var pill=document.createElement('span');
      pill.className='tag-rm';
      pill.innerHTML=App.esc(t)+'<span class="x">\u00d7</span>';
      pill.querySelector('.x').addEventListener('click',function(){App.removeTagFromChat(t)});
      row.appendChild(pill);
    });
    App.$tagsPanel.appendChild(row);
  } else {
    var empty=document.createElement('div');
    empty.style.cssText='font-size:11px;color:var(--text-faint);margin-bottom:4px';
    empty.textContent='No tags on this chat';
    App.$tagsPanel.appendChild(empty);
  }

  var addRow=document.createElement('div');
  addRow.className='tag-add';
  var inp=document.createElement('input');
  inp.placeholder='Add tag...';
  inp.addEventListener('keydown',function(e){if(e.key==='Enter')doAddTag()});
  var btn=document.createElement('button');
  btn.textContent='+';
  btn.addEventListener('click',doAddTag);
  addRow.appendChild(inp);
  addRow.appendChild(btn);
  App.$tagsPanel.appendChild(addRow);

  function doAddTag(){
    var tag=inp.value.trim().toLowerCase();
    if(!tag)return;
    App.addTagToChat(tag);
    inp.value='';
  }
};

App.addTagToChat = async function(tag){
  if(!App.activeThread)return;
  await fetch(App.API+'/api/chats/'+App.activeThread+'/tags',{
    method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({tag:tag})
  });
  await App.loadChatList();
  App.renderTagPanel();
};

App.removeTagFromChat = async function(tag){
  if(!App.activeThread)return;
  await fetch(App.API+'/api/chats/'+App.activeThread+'/tags/'+encodeURIComponent(tag),{method:'DELETE'});
  await App.loadChatList();
  App.renderTagPanel();
};
