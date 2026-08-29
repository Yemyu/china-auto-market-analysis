// China Auto Market Analysis shared shell and chart helpers.
var __dataCache = {};
var APP_DATA_VERSION = '20260829-1';

function showAppError(message) {
  var main = document.querySelector('.content');
  if (!main || document.querySelector('.app-error')) return;
  var box = document.createElement('div');
  box.className = 'app-error';
  box.textContent = message || 'Data failed to load. Start the dashboard with a local HTTP server and retry.';
  main.insertBefore(box, main.firstChild);
}

function loadJSON(name) {
  if (!__dataCache[name]) {
    __dataCache[name] = fetch('static/data/' + name + '.json?v=' + APP_DATA_VERSION).then(function (r) {
      if (!r.ok) throw new Error(name + '.json: HTTP ' + r.status);
      return r.json();
    }).catch(function (err) {
      showAppError((window.CHINA_AUTO_MARKET && window.CHINA_AUTO_MARKET.lang === 'en')
        ? 'Dashboard data could not be loaded: ' + err.message
        : '看板数据加载失败：' + err.message);
      throw err;
    });
  }
  return __dataCache[name];
}

function initChart(id) {
  var el = document.getElementById(id);
  if (!el) return null;
  var existing = echarts.getInstanceByDom(el);
  if (existing) return existing;
  var chart = echarts.init(el, null, { renderer: 'canvas' });
  window.addEventListener('resize', function () { chart.resize(); });
  return chart;
}

function baseGrid(extra) {
  return Object.assign({ left: 64, right: 28, top: 48, bottom: 56, containLabel: true }, extra || {});
}

function axisStyle() {
  return {
    axisLine: { lineStyle: { color: window.CHINA_AUTO_MARKET.axisLine } },
    axisTick: { show: false },
    axisLabel: { color: window.CHINA_AUTO_MARKET.muted, fontSize: 12 },
    splitLine: { lineStyle: { color: window.CHINA_AUTO_MARKET.splitLine } }
  };
}

function titleStyle() { return { show: false }; }
function tooltipStyle() {
  return {
    trigger: 'axis', backgroundColor: 'rgba(15, 28, 47, .96)', borderWidth: 0,
    padding: [10, 12], textStyle: { color: '#f7fbff', fontSize: 13 },
    extraCssText: 'box-shadow:0 12px 28px rgba(15,28,47,.2);border-radius:8px;'
  };
}

function labelLang(d) {
  return window.CHINA_AUTO_MARKET.lang === 'en' ? (d.name_en || d.name || d.name_zh) : (d.name_zh || d.name || d.name_en);
}
function nameFromAspect(d) { return labelLang(d); }
function formatNumber(value, digits) {
  if (value === null || value === undefined || value === '') return '—';
  return Number(value).toLocaleString(undefined, { minimumFractionDigits: digits || 0, maximumFractionDigits: digits || 0 });
}
function escapeHTML(value) {
  return String(value === null || value === undefined ? '' : value)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#039;');
}

function currentPage() {
  var file = location.pathname.split('/').pop() || 'index.html';
  return ({
    'index.html': 'overview', 'forecast.html': 'forecast', 'absa.html': 'absa',
    'attribution.html': 'attribution', 'alerts.html': 'alerts',
    'drilldown.html': 'drilldown'
  })[file] || 'overview';
}
function shellLabel(key, fallback) { return window.I18N ? window.I18N.get(key) : fallback; }

function renderShell() {
  var sidebar = document.querySelector('.sidebar');
  if (!sidebar) return;
  var active = currentPage();
  var items = [
    ['overview', 'index.html', 'nav_overview', '01'],
    ['forecast', 'forecast.html', 'nav_forecast', '02'],
    ['absa', 'absa.html', 'nav_absa', '03'],
    ['attribution', 'attribution.html', 'nav_attribution', '04'],
    ['alerts', 'alerts.html', 'nav_alerts', '05'],
    ['drilldown', 'drilldown.html', 'nav_drilldown', '06']
  ];
  sidebar.innerHTML = '<div class="nav-caption">' + escapeHTML(shellLabel('nav_caption', '中国车市分析 · 研究看板')) + '</div><div class="nav-list">' +
    items.map(function (item) {
      return '<a href="' + item[1] + '" class="' + (active === item[0] ? 'active' : '') + '">' +
        '<span class="nav-index">' + item[3] + '</span><span>' +
        escapeHTML(shellLabel(item[2], item[0])) + '</span></a>';
    }).join('') + '</div><div class="nav-foot"><span class="status-dot"></span><span>' +
    escapeHTML(shellLabel('nav_data_status', 'Data through 2026-07')) + '</span></div>';

  var brand = document.querySelector('.topbar .brand');
  if (brand) brand.innerHTML = '<a href="index.html" aria-label="' + escapeHTML(shellLabel('project_home_label', '中国车市分析首页')) + '"><span class="brand-mark">CM</span><span class="brand-word">' + escapeHTML(shellLabel('project_name_short', '中国车市分析')) + '</span></a>';
  var topbar = document.querySelector('.topbar');
  if (topbar && !document.getElementById('navToggle')) {
    var toggle = document.createElement('button');
    toggle.id = 'navToggle';
    toggle.className = 'nav-toggle';
    toggle.setAttribute('aria-label', 'Toggle navigation');
    toggle.innerHTML = '<span></span><span></span><span></span>';
    topbar.insertBefore(toggle, topbar.firstChild);
    var overlay = document.createElement('button');
    overlay.className = 'nav-overlay';
    overlay.setAttribute('aria-label', 'Close navigation');
    document.body.appendChild(overlay);
    function closeNav() { document.body.classList.remove('nav-open'); }
    toggle.addEventListener('click', function () { document.body.classList.toggle('nav-open'); });
    overlay.addEventListener('click', closeNav);
    sidebar.addEventListener('click', function (e) { if (e.target.closest('a')) closeNav(); });
  }
}

document.addEventListener('DOMContentLoaded', renderShell);
document.addEventListener('i18nChanged', function () {
  renderShell();
  if (window.__renderers) {
    Object.keys(window.__renderers).forEach(function (key) {
      try { window.__renderers[key](); } catch (e) { /* keep last valid chart */ }
    });
  }
});
window.__renderers = window.__renderers || {};
