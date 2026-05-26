"use strict";
var App = {};

/* ── Config ── */
App.API = '';
App.SUGGESTIONS = [
  {label:'What categories exist?', query:'What categories exist in the dataset?'},
  {label:'How many refund requests?', query:'How many refund requests did we get?'},
  {label:'Summarize FEEDBACK', query:'Summarize the FEEDBACK category.'},
  {label:'3 examples from SHIPPING', query:'Show me 3 examples from the SHIPPING category.'},
  {label:'Intent dist. in ACCOUNT', query:'What is the distribution of intents in the ACCOUNT category?'},
  {label:'Top customer issues', query:'What are the top customer issues in the dataset?'}
];

/* ── State ── */
App.activeThread = null;
App.chatMessages = [];
App.allChats = [];
App.allTags = [];
App.allUsers = [];
App.activeTagFilter = null;
App.isSending = false;
App.currentThemeMode = 'system';
App.autoRecEnabled = localStorage.getItem('autoRec') !== 'false';
App.usePastSessions = localStorage.getItem('pastSessions') === 'true';
App.qualityScoringEnabled = localStorage.getItem('qualityScoring') === 'true';
App.reflectionEnabled = localStorage.getItem('reflection') !== 'false';
App.decompositionEnabled = localStorage.getItem('decomposition') !== 'false';
App._usersLoaded = false;
App.isAdminMode = false;

/* ── DOM refs ── */
App.$messages      = document.getElementById('messages');
App.$queryInput    = document.getElementById('queryInput');
App.$sendBtn       = document.getElementById('sendBtn');
App.$chatList      = document.getElementById('chatList');
App.$search        = document.getElementById('searchInput');
App.$userId        = document.getElementById('userId');
App.$userDropdown  = document.getElementById('userDropdown');
App.$healthSeg     = document.getElementById('healthSeg');
App.$datasetSeg    = document.getElementById('datasetSeg');
App.$tokenSeg      = document.getElementById('tokenSeg');
App.$modelSeg      = document.getElementById('modelSeg');
App.$tagsPanel     = document.getElementById('tagsPanel');
App.$tagFilter     = document.getElementById('tagFilterArea');
App.$adminContent  = document.getElementById('adminContent');
App.$memoryContent = document.getElementById('memoryContent');
App.$exportBtn     = document.getElementById('exportBtn');
App.$tabChat       = document.getElementById('tabChat');
App.$tabAdmin      = document.getElementById('tabAdmin');
App.$tabMemory     = document.getElementById('tabMemory');
App.$adminModeBtn  = document.getElementById('adminModeBtn');
App.$adminSidebar  = document.getElementById('adminSidebar');
App.$sbUser        = document.querySelector('.sb-user');
App.$sbActions     = document.querySelector('.sb-actions');
App.$sbSearch      = document.querySelector('.sb-search');
App.$sbChats       = document.getElementById('chatList');
App.$sbSections    = document.querySelectorAll('.sb-section');
App.$recToggles    = document.getElementById('recToggles');

/* ── Util ── */
App.esc = function(s){if(!s)return '';var d=document.createElement('div');d.textContent=s;return d.innerHTML};
App.makeId = function(){return Math.random().toString(36).substring(2,10)};
App.scrollBottom = function(){App.$messages.scrollTop=App.$messages.scrollHeight};
App.getUser = function(){return App.$userId.value.trim()||'default'};
