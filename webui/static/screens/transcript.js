/* Meetscribe — transcript screen.
 * Design: design/Meetscribe.dc.html lines 70-109 (body + "Who spoke" rail),
 * export dialog 345-377. Inline style strings are carried over from the design.
 * Data: GET /api/meeting?id=…  (plus /api/review and /api/settings, lazily, for
 * the two lines in the rail that need them). Nothing here invents a number.
 */
(function () {
  'use strict';

  var MS = (window.MS = window.MS || {});
  MS.screens = MS.screens || {};

  /* one screen instance at a time; render() supersedes whatever was live */
  var live = null;
  var epoch = 0;

  // ---------------------------------------------------------------- helpers
  function el(tag, style, text) {
    var n = document.createElement(tag);
    if (style) n.setAttribute('style', style);
    if (text !== null && text !== undefined) n.textContent = String(text);
    return n;
  }

  function pad(n) { return String(n).padStart(2, '0'); }

  function hmsLocal(s) {
    s = Math.max(0, Math.floor(Number(s) || 0));
    return Math.floor(s / 3600) + ':' + pad(Math.floor((s % 3600) / 60)) + ':' + pad(s % 60);
  }

  /* the design's ms(): m:ss, used for the byline and the roll */
  function msLocal(s) {
    s = Math.max(0, Number(s) || 0);
    return Math.floor(s / 60) + ':' + pad(Math.floor(s % 60));
  }

  function srtTime(s) {
    s = Math.max(0, Number(s) || 0);
    var w = Math.floor(s);
    return pad(Math.floor(w / 3600)) + ':' + pad(Math.floor((w % 3600) / 60)) + ':' +
      pad(w % 60) + ',' + String(Math.round((s - w) * 1000)).padStart(3, '0');
  }

  function num(x) { return typeof x === 'number' && isFinite(x); }

  /* Use the shell's formatter when it demonstrably formats elapsed seconds the
   * way the design wants — i.e. when its output parses back to the seconds we
   * gave it. Otherwise fall back, rather than print a number that means
   * something else. */
  function fmtPick(ctx, name, probe, fallback) {
    var f = (ctx.fmt && typeof ctx.fmt[name] === 'function') ? ctx.fmt[name] : null;
    if (!f) return fallback;
    var s;
    try { s = f(probe); } catch (e) { return fallback; }
    if (typeof s !== 'string') return fallback;
    var parts = s.trim().split(':');
    if (parts.length < 2 || parts.length > 3) return fallback;
    var v = 0;
    for (var i = 0; i < parts.length; i++) {
      if (!/^\d+(\.\d+)?$/.test(parts[i])) return fallback;
      v = v * 60 + parseFloat(parts[i]);
    }
    if (Math.abs(v - probe) > 1) return fallback;
    return function (x) {
      try {
        var r = f(x);
        return typeof r === 'string' ? r : fallback(x);
      } catch (e) { return fallback(x); }
    };
  }

  function plural(n, one, many) { return n === 1 ? '1 ' + one : n + ' ' + many; }

  function trimNum(x) {
    var v = Number(x);
    if (!isFinite(v)) return String(x);
    return v % 1 === 0 ? String(v) : String(Math.round(v * 10) / 10);
  }

  function cleanup() {
    epoch++;                       // in-flight fetches from the old screen bail out
    if (!live) return;
    var l = live;
    live = null;
    if (l.timer) clearInterval(l.timer);
    (l.offs || []).forEach(function (off) { try { off(); } catch (e) {} });
    if (l.closeDialog) { try { l.closeDialog(); } catch (e) {} }
  }

  // -------------------------------------------------------------- the screen
  MS.screens.transcript = {
    title: 'Transcript',
    render: render,
    destroy: cleanup
  };

  async function render(root, ctx) {
    cleanup();
    var token = epoch;
    ctx = ctx || {};
    root.textContent = '';

    var api = typeof ctx.api === 'function' ? ctx.api : null;
    var go = typeof ctx.go === 'function' ? ctx.go.bind(ctx) : function () {};
    var state = (ctx.state && typeof ctx.state === 'object') ? ctx.state : {};
    var ink = inkFn(ctx);
    var hms = fmtPick(ctx, 'hms', 3661, hmsLocal);      // 1:01:01 — the turn stamp
    var mmss = fmtPick(ctx, 'clock', 125, msLocal);     // 2:05 — byline and roll

    /* the design shows 11:32 for a 692 s meeting: m:ss under an hour, h:mm:ss over */
    function meetingDur(s) {
      s = Math.max(0, Number(s) || 0);
      return s >= 3600 ? hms(s) : mmss(s);
    }

    if (!api) { note(root, 'The transcript screen needs ctx.api and did not get it.'); return; }

    // -- which meeting
    var id = pickId(ctx);
    if (!id) {
      try {
        var lib = await api('/api/library');
        if (token !== epoch) return;
        var list = (lib && lib.meetings) || [];
        if (list.length) id = list[0].id;
      } catch (e) { /* handled by the empty state below */ }
    }
    if (token !== epoch) return;
    if (!id) {
      backLink(root, go);
      note(root, 'No transcript is open, and the library has none to fall back on.');
      return;
    }

    var m;
    try {
      m = await api('/api/meeting?id=' + encodeURIComponent(id));
    } catch (e) {
      if (token !== epoch) return;
      backLink(root, go);
      note(root, '“' + id + '” is not a transcript in this library. (' +
        ((e && e.message) || 'request failed') + ')');
      return;
    }
    if (token !== epoch) return;
    if (!m || typeof m !== 'object') { note(root, 'The server returned no meeting.'); return; }

    /* let the shell and the other screens know which meeting is open */
    try { state.meeting = m.id; state.meetingId = m.id; } catch (e) {}

    // -- speakers: the API's own order (longest first) is the ink order
    var speakers = Array.isArray(m.speakers) ? m.speakers : [];
    var order = speakers.map(function (s) { return s.id; });
    function inkOf(g, variant) {
      var i = order.indexOf(g);
      return ink(i < 0 ? 4 : i, variant);   // a cluster the API left out (G-1) reads as unknown
    }

    // -- flatten turns into the design's one-paragraph-per-segment rows
    var rows = [];
    (Array.isArray(m.turns) ? m.turns : []).forEach(function (t) {
      if (!t) return;
      var lines = (Array.isArray(t.lines) && t.lines.length)
        ? t.lines
        : [{ start: t.start, end: t.end, text: t.text }];
      lines.forEach(function (ln) {
        if (!ln) return;
        var text = (ln.text || '').trim();
        if (!text) return;
        rows.push({
          g: t.speaker, name: t.name || 'Unknown', named: !!t.named,
          start: num(+ln.start) ? +ln.start : 0,
          end: num(+ln.end) ? +ln.end : 0,
          text: text
        });
      });
    });
    rows.forEach(function (r, i) { r.head = (i === 0) || rows[i - 1].g !== r.g; });

    var starts = rows.map(function (r) { return r.start; });

    // ------------------------------------------------------------ layout
    var grid = el('div', 'display:grid;grid-template-columns:minmax(0,1fr) 236px;gap:56px;align-items:start;max-width:1180px');
    var main = el('div');
    var aside = el('aside', 'padding-top:34px;position:sticky;top:0');
    grid.appendChild(main);
    grid.appendChild(aside);
    root.appendChild(grid);

    // back
    var back = el('div', 'display:inline-block;font-size:12.5px;color:var(--color-accent-700);cursor:pointer;margin-bottom:14px', '← Library');
    back.onclick = function () { go('library'); };
    main.appendChild(back);

    // kicker
    var kick = ['Meeting'];
    if (m.date) kick.push(m.date);
    if (num(m.duration_s) && m.duration_s > 0) kick.push(meetingDur(m.duration_s));
    if (num(m.n_speakers)) kick.push(plural(m.n_speakers, 'voice', 'voices'));
    var nTurns = num(m.n_turns) ? m.n_turns : rows.length;
    kick.push(plural(nTurns, 'turn', 'turns'));
    main.appendChild(el('div', 'font-size:10.5px;letter-spacing:0.14em;text-transform:uppercase;color:var(--color-neutral-600);margin-bottom:9px', kick.join(' · ')));

    main.appendChild(el('h1', 'font-size:42px;margin:0 0 12px;max-width:24ch', m.title || m.id));

    // ------------------------------------------------------------ byline
    var focus = null;
    var bylineNodes = [];
    var bl = el('div', 'display:flex;flex-wrap:wrap;gap:4px 18px;margin-bottom:34px');
    speakers.forEach(function (s) {
      var d = el('div', 'font-size:12.5px;cursor:pointer;color:' + inkOf(s.id, 'text') + ';');
      d.appendChild(document.createTextNode((s.name || s.id) + ' '));
      d.appendChild(el('span', 'opacity:0.55;font-variant-numeric:tabular-nums', mmss(s.seconds)));
      d.onclick = function () { setFocus(s.id); };
      bylineNodes.push({ g: s.id, node: d });
      bl.appendChild(d);
    });
    if (speakers.length) main.appendChild(bl);

    // ------------------------------------------------------------- turns
    var body = el('div', 'max-width:64ch');
    main.appendChild(body);

    var refs = [];
    if (!rows.length) {
      body.appendChild(el('div', 'font-size:15px;color:var(--color-neutral-700);font-style:italic',
        'This transcript has no lines with text in it.'));
    } else {
      var frag = document.createDocumentFragment();
      var flagged = {};
      rows.forEach(function (r, i) {
        var wrap = el('div', 'cursor:pointer;');
        wrap.setAttribute('data-active', '0');
        var ref = { wrap: wrap, head: null, time: null, body: null, row: r };

        if (r.head) {
          var head = el('div', headStyle(r, false, i === 0));
          var time = el('span', timeStyle(false), hms(r.start));
          head.appendChild(time);
          head.appendChild(document.createTextNode(r.name));
          /* The design flags an unplaced voice in its turn head. With nobody
           * named yet that is every head on the page, so it is flagged once per
           * voice — the first time you meet them — instead of 218 times. */
          if (!r.named && !flagged[r.g] && order.indexOf(r.g) >= 0) {
            flagged[r.g] = true;
            var flag = el('span', 'font-size:10px;letter-spacing:0.06em;text-transform:none;color:var(--color-accent-2-700);border-bottom:1px solid var(--color-accent-2-300)', 'name this voice');
            flag.style.cursor = 'pointer';
            flag.onclick = function (ev) {
              ev.stopPropagation();
              go('review', { meeting: m.id, cluster: r.g });
            };
            head.appendChild(flag);
          }
          wrap.appendChild(head);
          ref.head = head;
          ref.time = time;
        }
        var para = el('div', bodyStyle(false), r.text);
        wrap.appendChild(para);
        ref.body = para;

        wrap.onclick = function () {
          if (ctx.player && typeof ctx.player.seek === 'function') {
            try { ctx.player.seek(r.start); } catch (e) {}
          }
          setActive(i, false);
        };
        refs.push(ref);
        frag.appendChild(wrap);
      });
      body.appendChild(frag);
      body.appendChild(el('div', 'height:40px'));
    }

    function headStyle(r, on, first) {
      return 'display:flex;align-items:baseline;gap:9px;margin:' + (first ? '0' : '22px') +
        ' 0 5px;font-size:11px;letter-spacing:0.1em;text-transform:uppercase;color:' +
        (on ? 'var(--color-accent-700)' : inkOf(r.g, 'text'));
    }
    function timeStyle(on) {
      return 'font-variant-numeric:tabular-nums;letter-spacing:0.04em;color:var(--color-neutral-600);width:52px;flex:none;' +
        (on ? 'color:var(--color-accent-700);' : '');
    }
    function bodyStyle(on) {
      return 'font-size:16.5px;line-height:1.62;margin-bottom:3px;color:' +
        (on ? 'var(--color-text)' : 'color-mix(in srgb, var(--color-text) 72%, transparent)') + ';' +
        (on ? 'font-weight:400;' : '');
    }

    // ------------------------------------------------------- who spoke
    aside.appendChild(el('div', 'font-size:10.5px;letter-spacing:0.12em;text-transform:uppercase;color:var(--color-neutral-600);margin-bottom:14px', 'Who spoke'));

    var rollNodes = [];
    speakers.forEach(function (s) {
      var wrap = el('div', 'margin-bottom:15px;cursor:pointer;');
      var line = el('div', 'display:flex;justify-content:space-between;align-items:baseline;font-size:13px;margin-bottom:4px');
      line.appendChild(el('span', 'color:' + inkOf(s.id, 'text') + ';' + (s.named ? '' : 'font-style:italic;'), s.name || s.id));
      line.appendChild(el('span', 'color:var(--color-neutral-600);font-variant-numeric:tabular-nums;font-size:11.5px', mmss(s.seconds)));
      wrap.appendChild(line);
      var track = el('div', 'height:5px;background:color-mix(in srgb, var(--color-text) 8%, transparent)');
      var pct = num(s.share) ? Math.max(0, Math.min(1, s.share)) * 100 : 0;
      track.appendChild(el('div', 'height:5px;width:' + pct.toFixed(1) + '%;background:' + inkOf(s.id, 'fill')));
      wrap.appendChild(track);
      /* the design prints a match score here. The server has one only for a
       * cluster still open in review, so the line stays hidden until /api/review
       * hands over a real number — see below. */
      var score = el('div', 'display:none');
      wrap.appendChild(score);
      wrap.onclick = function () { setFocus(s.id); };
      rollNodes.push({ g: s.id, node: wrap, score: score });
      aside.appendChild(wrap);
    });
    if (!speakers.length) {
      aside.appendChild(el('div', 'font-size:12.5px;color:var(--color-neutral-700);line-height:1.5',
        'No speaker was assigned in this transcript.'));
    }

    // unnamed → review
    var unnamed = speakers.filter(function (s) { return !s.named; }).length;
    if (unnamed) {
      var rev = el('div', 'margin-top:24px;font-size:12.5px;color:var(--color-accent-2-700);cursor:pointer;border-bottom:1px solid var(--color-accent-2-300);display:inline-block;padding-bottom:2px',
        (unnamed === 1 ? 'One voice unnamed' : unnamed + ' voices unnamed') + ' — review');
      rev.onclick = function () { go('review', { meeting: m.id }); };
      aside.appendChild(rev);
    }

    // export + the small print
    var exportBlock = el('div', 'margin-top:26px;display:flex;flex-direction:column;gap:8px');
    var xb = el('button', 'margin:0', 'Export transcript');
    xb.className = 'btn btn-secondary btn-block';
    xb.onclick = function () { openExport(); };
    exportBlock.appendChild(xb);

    var factsParts = [];
    if (num(m.coverage)) {
      /* 0.9999 is not 100%, and rounding it there would read as a claim */
      var cov = m.coverage * 100;
      factsParts.push('Coverage ' + (cov < 100 && cov >= 99.95 ? cov.toFixed(2) : cov.toFixed(1)) + '%');
    }
    if (num(m.window_s)) factsParts.push('window ' + trimNum(m.window_s) + ' s');
    var facts = el('div', 'font-size:11.5px;color:var(--color-neutral-600);line-height:1.45', factsParts.join(' · '));
    if (factsParts.length) exportBlock.appendChild(facts);

    if (m.has_audio === false) {
      exportBlock.appendChild(el('div', 'font-size:11.5px;color:var(--color-neutral-600);line-height:1.45',
        'No audio sits beside this transcript, so there is nothing to play. Clicking a line still moves the mark.'));
    }
    aside.appendChild(exportBlock);

    var idNote = el('div', 'display:none');
    aside.appendChild(idNote);

    // ------------------------------------------------------------ focus
    function setFocus(g) {
      focus = (focus === g) ? null : g;
      bylineNodes.forEach(function (b) { b.node.style.opacity = (focus && focus !== b.g) ? '0.4' : ''; });
      rollNodes.forEach(function (b) { b.node.style.opacity = (focus && focus !== b.g) ? '0.4' : ''; });
      refs.forEach(function (r) { r.wrap.style.opacity = (focus && focus !== r.row.g) ? '0.32' : ''; });
    }

    // ----------------------------------------------------------- playhead
    var active = -1;

    function findActive(t) {
      if (!starts.length) return -1;
      var lo = 0, hi = starts.length - 1, best = -1;
      while (lo <= hi) {
        var mid = (lo + hi) >> 1;
        if (starts[mid] <= t) { best = mid; lo = mid + 1; } else { hi = mid - 1; }
      }
      if (best < 0) return -1;
      return t < rows[best].end + 0.6 ? best : -1;
    }

    function paint(i, on) {
      var r = refs[i];
      if (!r) return;
      r.wrap.setAttribute('data-active', on ? '1' : '0');
      if (r.head) r.head.style.color = on ? 'var(--color-accent-700)' : inkOf(r.row.g, 'text');
      if (r.time) r.time.style.color = on ? 'var(--color-accent-700)' : 'var(--color-neutral-600)';
      if (r.body) {
        r.body.style.color = on ? 'var(--color-text)' : 'color-mix(in srgb, var(--color-text) 72%, transparent)';
      }
    }

    function setActive(i, moving) {
      if (i === active) return;
      if (active >= 0) paint(active, false);
      active = i;
      if (active >= 0) {
        paint(active, true);
        if (moving) follow(refs[active] && refs[active].wrap);
      }
    }

    var scroller = null;
    function findScroller() {
      if (scroller && document.contains(scroller) &&
          scroller.scrollHeight > scroller.clientHeight + 4) return scroller;
      scroller = null;
      var n = root.parentElement;
      while (n && n !== document.body && n !== document.documentElement) {
        var oy = '';
        try { oy = getComputedStyle(n).overflowY; } catch (e) {}
        if ((oy === 'auto' || oy === 'scroll' || oy === 'overlay') &&
            n.scrollHeight > n.clientHeight + 4) { scroller = n; break; }
        n = n.parentElement;
      }
      return scroller;
    }

    function follow(node) {
      if (!node) return;
      var r = node.getBoundingClientRect();
      var sc = findScroller();
      if (sc) {
        var c = sc.getBoundingClientRect();
        if (r.top < c.top + 90 || r.bottom > c.bottom - 90) sc.scrollTop += r.top - c.top - 190;
      } else {
        var h = window.innerHeight || document.documentElement.clientHeight || 0;
        if (r.top < 90 || r.bottom > h - 90) window.scrollBy(0, r.top - 190);
      }
    }

    /* The player is the clock. Poll it (the contract guarantees time()) and,
     * where the shell also emits events, take those too — both funnel into the
     * same no-op-if-unchanged path. A player that is not running never moves
     * the mark, so a click on a line stays put. */
    var lastPlayerT = null;
    function sample() {
      var p = ctx.player;
      if (!p || typeof p.time !== 'function') return;
      var t;
      try { t = p.time(); } catch (e) { return; }
      if (!num(t)) return;
      if (lastPlayerT !== null && Math.abs(t - lastPlayerT) < 0.005) return;
      lastPlayerT = t;
      setActive(findActive(t), true);
    }

    var offs = [];
    if (ctx.player && typeof ctx.player.on === 'function') {
      ['time', 'timeupdate'].forEach(function (evt) {
        try {
          var off = ctx.player.on(evt, function (t) {
            if (num(t)) { lastPlayerT = t; setActive(findActive(t), true); } else { sample(); }
          });
          if (typeof off === 'function') offs.push(off);
        } catch (e) {}
      });
    }

    live = { timer: setInterval(sample, 120), offs: offs, closeDialog: null };
    sample();

    // ------------------------------------------- lazy, honest extra facts
    api('/api/settings').then(function (s) {
      if (token !== epoch || !s || !s.models || !s.models.transcribe) return;
      factsParts.push(String(s.models.transcribe));
      facts.textContent = factsParts.join(' · ');
      if (!facts.parentNode) exportBlock.insertBefore(facts, exportBlock.children[1] || null);
    }).catch(function () {});

    api('/api/review').then(function (rv) {
      if (token !== epoch || !rv) return;
      var open = {};
      ['pending', 'elsewhere'].forEach(function (k) {
        (Array.isArray(rv[k]) ? rv[k] : []).forEach(function (it) {
          if (it && (it.meeting === m.id || it.meeting_id === m.id)) open[it.cluster] = it;
        });
      });
      rollNodes.forEach(function (b) {
        var it = open[b.g];
        if (!it || !it.best || !num(it.best.score)) return;
        b.score.setAttribute('style', 'font-size:10.5px;color:var(--color-neutral-600);margin-top:3px;font-variant-numeric:tabular-nums');
        b.score.textContent = 'best guess ' + it.best.score.toFixed(2) +
          (it.best.name ? ' · ' + it.best.name : '');
      });
      var un = (Array.isArray(rv.unidentified) ? rv.unidentified : []).filter(function (u) {
        return u && u.meeting === m.id;
      })[0];
      if (un) {
        idNote.setAttribute('style', 'margin-top:14px;font-size:11.5px;color:var(--color-neutral-600);line-height:1.45');
        idNote.textContent = un.detail || ('Not identified — ' + (un.reason || 'reason not given') + '.');
      }
    }).catch(function () {});

    // ------------------------------------------------------ export dialog
    function exportWho(g, name, namesOn) {
      if (namesOn) return name || 'Unknown';
      var i = order.indexOf(g);
      return i < 0 ? 'UNKNOWN' : 'SPEAKER ' + (i + 1);
    }

    function build(fmt, opts, preview) {
      var take = preview ? rows.slice(0, 5) : rows;
      var out, last = null;
      if (fmt === 'txt') {
        out = (m.title || m.id) + '   ' + ((Number(m.duration_s) || 0) / 60).toFixed(1) +
          ' min   ' + rows.length + ' segments\n';
        if (opts.roll) {
          out += 'speakers: ' + speakers.map(function (s) {
            return exportWho(s.id, s.name, opts.names);
          }).join(', ') + '\n';
        }
        out += new Array(49).join('=') + '\n';
        take.forEach(function (r) {
          if (r.g !== last) {
            out += '\n' + (opts.stamps ? '[' + hms(r.start) + '] ' : '') +
              exportWho(r.g, r.name, opts.names) + '\n';
            last = r.g;
          }
          out += '  ' + r.text + '\n';
        });
        return out + (preview ? '\n…' : '');
      }
      if (fmt === 'md') {
        out = '## ' + (m.title || m.id) + '\n*' +
          [m.date, meetingDur(m.duration_s), plural(speakers.length, 'voice', 'voices')]
            .filter(Boolean).join(' · ') + '*\n';
        take.forEach(function (r) {
          out += '\n**' + exportWho(r.g, r.name, opts.names) + '**' +
            (opts.stamps ? ' · ' + hms(r.start) : '') + '  \n' + r.text + '\n';
        });
        return out + (preview ? '\n…' : '');
      }
      if (fmt === 'srt') {
        out = take.map(function (r, i) {
          return (i + 1) + '\n' + srtTime(r.start) + ' --> ' + srtTime(r.end) + '\n' +
            (opts.names ? exportWho(r.g, r.name, true) + ': ' : '') + r.text;
        }).join('\n\n');
        return out + (preview ? '\n\n…' : '\n');
      }
      // json
      if (preview) {
        return '{\n "meeting": "' + (m.id || '') + '",\n "duration_s": ' +
          (Number(m.duration_s) || 0).toFixed(1) + ',\n "segments": [\n' +
          take.map(function (r) {
            return '  {"t": ' + r.start.toFixed(2) + ', "e": ' + r.end.toFixed(2) +
              ', "s": ' + JSON.stringify(opts.names ? (r.named ? r.name : null) : r.g) +
              ', "x": ' + JSON.stringify(r.text.slice(0, 34) + '…') + '}';
          }).join(',\n') + '\n ]\n}';
      }
      return JSON.stringify({
        meeting: m.id,
        duration_s: Number(m.duration_s) || 0,
        segments: rows.map(function (r) {
          return {
            t: r.start, e: r.end,
            s: opts.names ? (r.named ? r.name : null) : r.g,
            x: r.text
          };
        })
      }, null, 1);
    }

    function openExport() {
      if (typeof ctx.dialog !== 'function') return;
      var st = { fmt: 'txt', opts: { stamps: true, names: false, roll: true } };

      var dlg = el('div', 'width:min(620px,100%);gap:var(--space-3)');
      dlg.className = 'dialog';
      var dtitle = el('div', null, 'Export — ' + (m.title || m.id));
      dtitle.className = 'dialog-title';
      dlg.appendChild(dtitle);

      var cols = el('div', 'display:grid;grid-template-columns:200px minmax(0,1fr);gap:26px');
      var left = el('div');
      var right = el('div');
      cols.appendChild(left);
      cols.appendChild(right);
      dlg.appendChild(cols);

      var LBL = 'font-size:10.5px;letter-spacing:0.12em;text-transform:uppercase;color:var(--color-neutral-600);margin-bottom:10px';
      left.appendChild(el('div', LBL, 'Format'));
      var fmts = el('div', 'display:flex;flex-direction:column;gap:9px;margin-bottom:20px');
      var dots = [];
      [['txt', 'Plain text'], ['md', 'Markdown'], ['srt', 'Subtitles (SRT)'], ['json', 'JSON with timings']]
        .forEach(function (f) {
          var lab = el('label', 'display:flex;align-items:center;gap:8px;font-size:13.5px;cursor:pointer;position:relative');
          var inp = document.createElement('input');
          inp.type = 'radio';
          inp.name = 'ms-export-fmt';
          inp.checked = st.fmt === f[0];
          inp.setAttribute('style', 'position:absolute;opacity:0;width:0;height:0;margin:0');
          var dot = el('span', dotStyle(st.fmt === f[0]));
          inp.onchange = function () {
            st.fmt = f[0];
            dots.forEach(function (d) { d.dot.setAttribute('style', dotStyle(st.fmt === d.k)); });
            refresh();
          };
          lab.appendChild(inp);
          lab.appendChild(dot);
          lab.appendChild(document.createTextNode(f[1]));
          dots.push({ k: f[0], dot: dot });
          fmts.appendChild(lab);
        });
      left.appendChild(fmts);

      left.appendChild(el('div', LBL, 'Include'));
      var incl = el('div', 'display:flex;flex-direction:column;gap:9px');
      [['stamps', 'Timestamps'], ['names', 'Speaker names'], ['roll', 'Speaker roll-up']]
        .forEach(function (o) {
          var row = el('div', 'display:flex;align-items:center;gap:8px;font-size:13.5px;cursor:pointer');
          var box = el('span', boxStyle(st.opts[o[0]]));
          row.appendChild(box);
          row.appendChild(document.createTextNode(o[1]));
          row.onclick = function () {
            st.opts[o[0]] = !st.opts[o[0]];
            box.setAttribute('style', boxStyle(st.opts[o[0]]));
            refresh();
          };
          incl.appendChild(row);
        });
      left.appendChild(incl);

      right.appendChild(el('div', LBL, 'Preview'));
      var pre = el('div', 'background:var(--color-bg);padding:14px 16px;border-radius:var(--radius-md);font-size:12px;line-height:1.6;white-space:pre-wrap;height:206px;overflow:hidden;font-variant-numeric:tabular-nums');
      right.appendChild(pre);
      var nameNote = el('div');
      right.appendChild(nameNote);

      var actions = el('div');
      actions.className = 'dialog-actions';
      var cancel = el('button', null, 'Cancel');
      cancel.className = 'btn btn-secondary';
      var doit = el('button', null, 'Export TXT');
      doit.className = 'btn btn-primary';
      actions.appendChild(cancel);
      actions.appendChild(doit);
      dlg.appendChild(actions);

      function dotStyle(on) {
        return 'width:15px;height:15px;flex:none;border-radius:50%;border:1.5px solid ' +
          (on ? 'var(--color-accent)' : 'var(--color-divider)') + ';' +
          (on ? 'background:var(--color-accent);box-shadow:inset 0 0 0 3px var(--color-surface);' : '');
      }
      function boxStyle(on) {
        return 'width:15px;height:15px;flex:none;border-radius:var(--radius-sm);border:1.5px solid ' +
          (on ? 'var(--color-accent)' : 'var(--color-divider)') + ';' +
          (on ? 'background:var(--color-accent);box-shadow:inset 0 0 0 3px var(--color-surface);' : '');
      }
      function refresh() {
        pre.textContent = build(st.fmt, st.opts, true);
        nameNote.setAttribute('style', 'margin-top:10px;font-size:11.5px;line-height:1.45;color:' +
          (st.opts.names ? 'var(--color-accent-2-700)' : 'var(--color-neutral-700)'));
        nameNote.textContent = st.opts.names
          ? 'Names are on. This file identifies everyone who spoke.'
          : 'Names off. Speakers export as SPEAKER 1, SPEAKER 2 — safe to send outside.';
        doit.textContent = 'Export ' + st.fmt.toUpperCase();
      }
      refresh();

      var closed = false;
      var close = ctx.dialog(dlg);
      function shut() {
        if (closed) return;
        closed = true;
        document.removeEventListener('keydown', onKey, true);
        if (live) live.closeDialog = null;
        if (typeof close === 'function') { try { close(); } catch (e) {} }
      }
      function onKey(ev) { if (ev.key === 'Escape') shut(); }
      document.addEventListener('keydown', onKey, true);
      if (live) live.closeDialog = shut;

      cancel.onclick = shut;
      doit.onclick = function () {
        var text = build(st.fmt, st.opts, false);
        var name = String(m.title || m.id || 'transcript').replace(/[\\/:*?"<>|]+/g, '-');
        var blob = new Blob([text], {
          type: st.fmt === 'json' ? 'application/json;charset=utf-8' : 'text/plain;charset=utf-8'
        });
        var url = URL.createObjectURL(blob);
        var a = document.createElement('a');
        a.href = url;
        a.download = name + '.' + st.fmt;
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(function () { URL.revokeObjectURL(url); }, 10000);
        shut();
      };
    }
  }

  // ------------------------------------------------------------- plumbing
  function pickId(ctx) {
    var s = (ctx.state && typeof ctx.state === 'object') ? ctx.state : {};
    var p = ctx.params || s.params || {};
    var cands = [p && p.id, p && p.meeting, s.meeting, s.meetingId, s.meeting_id, s.id];
    for (var i = 0; i < cands.length; i++) {
      var c = cands[i];
      if (typeof c === 'string' && c) return c;
      if (c && typeof c === 'object' && typeof c.id === 'string' && c.id) return c.id;
    }
    try {
      var hay = String(location.hash || '') + '&' + String(location.search || '');
      var mm = /[?&#](?:id|meeting)=([^&]+)/.exec(hay);
      if (mm) return decodeURIComponent(mm[1]);
    } catch (e) {}
    return null;
  }

  /* The shell owns the palette. It is unclear from the contract whether the
   * variant is 0/1 or 'fill'/'text', so probe an index where the two colours
   * differ and then stick with whichever call form the shell answers. */
  function inkFn(ctx) {
    var f = typeof ctx.ink === 'function' ? ctx.ink.bind(ctx) : null;
    if (!f) {
      return function (i, variant) {
        return variant === 'text' ? 'var(--color-text)' : 'var(--color-neutral-500)';
      };
    }
    function safe(i, v) {
      try {
        var r = arguments.length > 1 ? f(i, v) : f(i);
        return (typeof r === 'string' && r) ? r : null;
      } catch (e) { return null; }
    }
    var mode = 'plain';
    var a = safe(1, 'text'), b = safe(1, 'fill');
    if (a && b && a !== b) {
      mode = 'str';
    } else {
      var c = safe(1, 1), d = safe(1, 0);
      if (c && d && c !== d) mode = 'num';
    }
    return function (i, variant) {
      var want = variant === 'text';
      var r = mode === 'str' ? safe(i, want ? 'text' : 'fill')
        : mode === 'num' ? safe(i, want ? 1 : 0)
          : safe(i);
      if (r) return r;
      r = safe(i);
      return r || (want ? 'var(--color-text)' : 'var(--color-neutral-500)');
    };
  }

  function backLink(root, go) {
    var back = el('div', 'display:inline-block;font-size:12.5px;color:var(--color-accent-700);cursor:pointer;margin-bottom:14px', '← Library');
    back.onclick = function () { go('library'); };
    root.appendChild(back);
  }

  function note(root, text) {
    root.appendChild(el('div', 'font-size:15px;color:var(--color-neutral-700);font-style:italic;max-width:64ch;line-height:1.6', text));
  }
})();
