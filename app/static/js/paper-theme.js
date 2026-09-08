// Read chart colors from the opt-in report theme rather than duplicating CSS tokens.
(function () {
  if (!document.body.classList.contains('paper-theme')) return;
  var css = getComputedStyle(document.body);
  function token(name) { return css.getPropertyValue(name).trim(); }
  Object.assign(window.CHINA_AUTO_MARKET, {
    primary: token('--primary'), text: token('--ink'), muted: token('--muted'),
    bg: token('--paper'), axisLine: token('--report-rule'), splitLine: token('--line'),
    sentimentPos: token('--positive'), sentimentNeg: token('--negative'),
    palette: [token('--primary'), '#64847b', '#9b8056', '#837687', token('--negative')],
    comparison: '#8b989c'
  });
})();
