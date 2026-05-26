App.renderJudgeCard = function(qd, botEl){
  var card=document.createElement('div');
  card.className='judge-card';
  var passed=qd.passed;
  var icon=passed?'\u2705':'\u26A0\uFE0F';
  function dots(score){
    var s='';
    for(var i=1;i<=5;i++) s+=(i<=score?'\u25CF':'\u25CB');
    return s;
  }
  card.innerHTML=
    '<div class="judge-header">'
      +'<span class="judge-icon">\u2696\uFE0F</span>'
      +'<span class="judge-label">'+App.esc(qd.judge_model?qd.judge_model.split('/').pop():'')+'</span>'
      +'<span class="judge-score '+(passed?'pass':'fail')+'">'+qd.overall+'/5 '+icon+'</span>'
    +'</div>'
    +'<div class="judge-dims">'
      +'<span class="judge-dim"><span class="dim-label">Grounded</span><span class="dim-dots">'+dots(qd.data_grounded)+'</span></span>'
      +'<span class="judge-dim"><span class="dim-label">Relevant</span><span class="dim-dots">'+dots(qd.addresses_question)+'</span></span>'
      +'<span class="judge-dim"><span class="dim-label">Concise</span><span class="dim-dots">'+dots(qd.conciseness)+'</span></span>'
    +'</div>'
    +(qd.issue?'<div class="judge-issue">'+App.esc(qd.issue)+'</div>':'')
    +'<div class="judge-meta">'+(qd.judge_tokens?qd.judge_tokens.total||0:0)+' tok &middot; '
      +(qd.judge_duration_ms||0)+'ms</div>';

  if(botEl && botEl.parentNode === App.$messages){
    var wrapper=document.createElement('div');
    wrapper.className='msg-with-judge';
    App.$messages.insertBefore(wrapper, botEl);
    wrapper.appendChild(botEl);
    wrapper.appendChild(card);
  } else {
    App.$messages.appendChild(card);
  }
  App.scrollBottom();
};
