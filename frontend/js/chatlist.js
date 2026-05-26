App.loadChatList = async function(){
  try{
    var url=App.API+'/api/chats?user_id='+encodeURIComponent(App.getUser());
    if(App.activeTagFilter) url+='&tag='+encodeURIComponent(App.activeTagFilter);
    var r=await fetch(url);
    App.allChats=await r.json();
  }catch(e){App.allChats=[];}
  App.renderChatList();
  await App.loadAllTags();
};

App.renderChatList = function(){
  var search=(App.$search.value||'').toLowerCase();
  var filtered=App.allChats.filter(function(c){return !search||c.title.toLowerCase().indexOf(search)>=0});
  if(!filtered.length){
    App.$chatList.innerHTML='<div style="padding:12px 16px;font-size:12px;color:var(--text-faint)">'
      +(App.allChats.length?'No matches':'No conversations yet')+'</div>';
    return;
  }
  App.$chatList.innerHTML='';
  filtered.forEach(function(c){
    var item=document.createElement('div');
    item.className='chat-item'+(c.thread_id===App.activeThread?' active':'');
    var title=document.createElement('div');
    title.className='ci-title';
    title.textContent=(c.thread_id===App.activeThread?'\u25B6 ':'')+c.title;
    item.appendChild(title);

    var actions=document.createElement('div');
    actions.className='ci-actions';
    var expBtn=document.createElement('button');
    expBtn.className='ci-act';
    expBtn.title='Export this chat';
    expBtn.textContent='\u2913';
    (function(tid,ttl){expBtn.addEventListener('click',function(e){e.stopPropagation();App.exportSingleChat(tid,ttl)})})(c.thread_id,c.title);
    actions.appendChild(expBtn);
    var delBtn=document.createElement('button');
    delBtn.className='ci-act del';
    delBtn.title='Delete this chat';
    delBtn.textContent='\u2715';
    (function(tid){delBtn.addEventListener('click',function(e){e.stopPropagation();if(confirm('Delete this chat?'))App.deleteSingleChat(tid)})})(c.thread_id);
    actions.appendChild(delBtn);
    item.appendChild(actions);

    if(c.tags&&c.tags.length){
      var tagsEl=document.createElement('div');
      tagsEl.className='ci-tags';
      c.tags.forEach(function(t){
        var pill=document.createElement('span');
        pill.className='ci-tag';
        pill.textContent=t;
        tagsEl.appendChild(pill);
      });
      item.appendChild(tagsEl);
    }
    var meta=document.createElement('div');
    meta.className='ci-meta';
    meta.textContent=(c.updated_at||'').substring(0,16)+' \u00b7 '+c.message_count+' msg';
    item.appendChild(meta);
    item.addEventListener('click',function(){App.switchToChat(c.thread_id)});
    App.$chatList.appendChild(item);
  });

  if(filtered.length > 1){
    var delAll=document.createElement('button');
    delAll.className='delete-all-user';
    delAll.textContent='\u2715 Delete all my chats';
    delAll.addEventListener('click',function(){
      if(!confirm('Delete all your chats? This cannot be undone.'))return;
      App.deleteUserChats();
    });
    App.$chatList.appendChild(delAll);
  }
};

App.exportSingleChat = async function(threadId,title){
  try{
    var r=await fetch(App.API+'/api/chats/'+threadId);
    var data=await r.json();
    var lines=['# '+App.esc(title||'Chat Export')+'\n'];
    (data.messages||[]).forEach(function(m){
      lines.push((m.role==='user'?'**User**: ':'**Agent**: ')+m.content+'\n');
    });
    var blob=new Blob([lines.join('\n')],{type:'text/markdown'});
    var a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=(title||'chat').replace(/[^a-z0-9]/gi,'_')+'.md';a.click();
  }catch(e){alert('Export failed.')}
};

App.deleteUserChats = async function(){
  await fetch(App.API+'/api/chats?user_id='+encodeURIComponent(App.getUser()),{method:'DELETE'});
  App.startNewChat();
  await App.loadChatList();
};

App.$search.addEventListener('input',function(){App.renderChatList()});
