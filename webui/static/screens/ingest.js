/* Ingest — design/Meetscribe.dc.html lines 111-141.

   Two real inputs, because the server reads audio off this machine and the
   browser never uploads anything:
     · a drop zone. A file dragged from the desktop only hands the page a path
       in some browsers (Firefox gives text/uri-list, Chrome does not), so when
       the path is withheld we fall through to the picker with the names ticked.
     · a picker over /api/browse, which lists directories on this machine.
   The queue table polls /api/queue and shows the stages the backend actually
   emits: queued/starting, upload, transcribe, embed, link, retrieve, done. */
(function () {
  'use strict';

  var MS = (window.MS = window.MS || {});
  MS.screens = MS.screens || {};

  /* the extensions server.py accepts (AUDIO_EXT) — the design's line lists all
     but .opus, and is kept verbatim; this list is what we actually test. */
  var AUDIO_EXT = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.opus', '.mp4', '.webm'];
  /* stages backend.py emits while a file is in flight */
  var RUNNING = ['starting', 'upload', 'transcribe', 'embed', 'link', 'retrieve'];

  var ZONE = 'border:1px dashed color-mix(in srgb, var(--color-text) 28%, transparent);' +
    'border-radius:var(--radius-md);padding:38px 30px;text-align:center;margin-bottom:12px';
  var ZONE_HOT = 'border:1px dashed var(--color-accent);' +
    'background:color-mix(in srgb, var(--color-accent) 7%, transparent);' +
    'border-radius:var(--radius-md);padding:38px 30px;text-align:center;margin-bottom:12px';
  var LABEL = 'font-size:10.5px;letter-spacing:0.12em;text-transform:uppercase;color:var(--color-neutral-600)';
  var SMALL = 'font-size:11.5px;color:var(--color-neutral-700)';

  // ---------------------------------------------------------------- helpers
  function h(tag, attrs, kids) {
    var n = document.createElement(tag), k, v, i, c, list;
    if (attrs) {
      for (k in attrs) {
        if (!Object.prototype.hasOwnProperty.call(attrs, k)) continue;
        v = attrs[k];
        if (v === null || v === undefined || v === false) continue;
        if (k === 'text') n.textContent = String(v);
        else if (k.indexOf('on') === 0 && typeof v === 'function') n.addEventListener(k.slice(2), v);
        else n.setAttribute(k, String(v));
      }
    }
    if (kids !== null && kids !== undefined) {
      list = Array.isArray(kids) ? kids : [kids];
      for (i = 0; i < list.length; i++) {
        c = list[i];
        if (c === null || c === undefined || c === false) continue;
        n.appendChild(typeof c === 'object' ? c : document.createTextNode(String(c)));
      }
    }
    return n;
  }

  function clear(node) { while (node && node.firstChild) node.removeChild(node.firstChild); }

  function errText(e) {
    if (!e) return 'unknown error';
    var d = e.body || e.data || e.payload || e.json;
    if (d && typeof d === 'object' && d.error) return String(d.error);
    if (typeof e.error === 'string') return e.error;
    if (e.message) return String(e.message);
    return String(e);
  }

  function pad2(n) { return (n < 10 ? '0' : '') + n; }

  function fallbackDur(s) {
    s = Math.round(s);
    var hh = Math.floor(s / 3600), mm = Math.floor((s % 3600) / 60), ss = s % 60;
    return hh ? hh + ':' + pad2(mm) + ':' + pad2(ss) : mm + ':' + pad2(ss);
  }

  function durOf(ctx, s) {
    if (typeof s !== 'number' || !isFinite(s) || s <= 0) return null;
    try {
      if (ctx && ctx.fmt && typeof ctx.fmt.dur === 'function') {
        var out = ctx.fmt.dur(s);
        if (out) return String(out);
      }
    } catch (e) { /* fall through to our own */ }
    return fallbackDur(s);
  }

  function elapsedText(s) {
    if (typeof s !== 'number' || !isFinite(s) || s < 0) return '—';
    if (s < 60) return Math.round(s) + ' s';
    return Math.floor(s / 60) + ':' + pad2(Math.round(s % 60) % 60);
  }

  function isAudio(name) {
    var lower = String(name || '').toLowerCase(), i;
    for (i = 0; i < AUDIO_EXT.length; i++) {
      if (lower.length > AUDIO_EXT[i].length && lower.slice(-AUDIO_EXT[i].length) === AUDIO_EXT[i]) return true;
    }
    return false;
  }

  function joinPath(dir, name) {
    if (!dir) return name;
    return dir.charAt(dir.length - 1) === '/' ? dir + name : dir + '/' + name;
  }

  function parentOf(p) {
    var s = String(p || '');
    while (s.length > 1 && s.charAt(s.length - 1) === '/') s = s.slice(0, -1);
    var i = s.lastIndexOf('/');
    if (i <= 0) return '/';
    return s.slice(0, i);
  }

  function baseOf(p) {
    var s = String(p || '');
    while (s.length > 1 && s.charAt(s.length - 1) === '/') s = s.slice(0, -1);
    return s.slice(s.lastIndexOf('/') + 1);
  }

  /* Paths out of a drop. Firefox and most terminals/editors put file:// URIs or
     a bare path on the dataTransfer; Chrome deliberately withholds them for OS
     file drags, in which case this returns [] and the caller falls back. */
  function pathsFromDrop(dt) {
    var raw = '', out = [], seen = {};
    try { raw = dt.getData('text/uri-list') || ''; } catch (e) { raw = ''; }
    if (!raw) { try { raw = dt.getData('text/plain') || ''; } catch (e2) { raw = ''; } }
    String(raw).split(/[\r\n]+/).forEach(function (line) {
      var s = line.trim();
      if (!s || s.charAt(0) === '#') return;
      if (/^file:\/\//i.test(s)) {
        s = s.replace(/^file:\/\/[^/]*/i, '');
        try { s = decodeURIComponent(s); } catch (e3) { /* keep it raw */ }
      } else if (s.charAt(0) !== '/' && s.charAt(0) !== '~') {
        return;
      }
      if (s && !seen[s]) { seen[s] = 1; out.push(s); }
    });
    return out;
  }

  function stageRank(st) {
    if (st === 'queued') return 1;
    if (st === 'failed') return 2;
    if (st === 'done') return 3;
    return 0;
  }

  /* running first, then waiting, then failures and finished work newest-first —
     the order the design's queue is drawn in. */
  function ordered(queue) {
    var rows = (queue || []).filter(function (q) { return q && typeof q === 'object'; })
      .map(function (q, i) { return { q: q, i: i }; });
    rows.sort(function (a, b) {
      var ra = stageRank(a.q.stage), rb = stageRank(b.q.stage);
      if (ra !== rb) return ra - rb;
      return ra >= 2 ? b.i - a.i : a.i - b.i;
    });
    return rows.map(function (r) { return r.q; });
  }

  function counts(items) {
    var c = { running: 0, waiting: 0, done: 0, failed: 0 };
    items.forEach(function (q) {
      var st = q.stage;
      if (st === 'queued') c.waiting++;
      else if (st === 'done') c.done++;
      else if (st === 'failed') c.failed++;
      else c.running++;
    });
    return c;
  }

  // ------------------------------------------------------------------ copy
  /* The design asserts things about storage that are only true of one of the
     two deployments, so this sentence is built from /api/settings instead of
     being hardcoded. See the report: the design's wording is not kept here. */
  function storageNote(s) {
    if (!s || !s.backend || !s.paths) {
      return 'Audio is read from the folder you point at. Two things are written: the transcript ' +
        'into the library, and one database holding the voice profiles.';
    }
    var lib = s.paths.library || 'the library';
    var db = s.paths.speaker_db || 'the profile store';
    if (s.backend.host) {
      return 'Transcription and embedding run on ' + (s.backend.host || 'the pipeline box') +
        ' over ssh — the audio is copied there and the transcript comes back. Everything else stays ' +
        'on this machine: the transcript and the audio in ' + lib + ' (hard-linked where the ' +
        'filesystem allows, so the copy costs nothing), and the voice profiles in ' + db + '. ' +
        'Naming, matching and merging never leave this machine.';
    }
    return 'Nothing leaves this machine. Audio is read from the folder you point at, and two things ' +
      'are written: the transcript and the audio beside it in ' + lib + ' (hard-linked where the ' +
      'filesystem allows, so the copy costs nothing), and one database holding the voice profiles, ' + db + '.';
  }

  /* The design's closing paragraph claims a pipeline stage this build does not
     run and an overlap this queue does not do; the true stage list comes from
     backend.py and depends on the mode. */
  function stagesNote(s) {
    var chain = (s && s.backend && s.backend.host)
      ? 'upload → transcribe → embed → link → retrieve'
      : 'transcribe → embed → link';
    return 'Stages run ' + chain + ', one file at a time — this build does not overlap one meeting’s ' +
      'embedding with the next one’s transcription. The first file in a session also pays for loading ' +
      'the model, so no countdown is shown until the engine is resident. Putting names to the voices is a ' +
      'separate step, on Review.';
  }

  // ------------------------------------------------------------------ state
  var live = null;

  function teardown(S) {
    if (!S) return;
    S.dead = true;
    if (S.timer) { clearTimeout(S.timer); S.timer = null; }
    if (S.winDrag) {
      window.removeEventListener('dragover', S.winDrag);
      window.removeEventListener('drop', S.winDrag);
      S.winDrag = null;
    }
    if (S.closeDialog) { try { S.closeDialog(); } catch (e) { /* already gone */ } S.closeDialog = null; }
  }

  // ------------------------------------------------------------------ queue
  function busy(S) {
    return (S.queue || []).some(function (q) {
      return q && (q.stage === 'queued' || RUNNING.indexOf(q.stage) >= 0);
    });
  }

  function schedule(S, ms) {
    if (S.dead) return;
    if (S.timer) clearTimeout(S.timer);
    S.timer = setTimeout(function () { poll(S); }, ms);
  }

  function poll(S) {
    if (S.dead) return;
    S.ctx.api('/api/queue').then(function (d) {
      if (S.dead) return;
      S.queueErr = null;
      S.queue = (d && d.queue) || [];
      drawQueue(S);
      loadLengths(S);
      schedule(S, busy(S) ? 1000 : 5000);
    }, function (e) {
      if (S.dead) return;
      S.queueErr = errText(e);
      drawQueue(S);
      schedule(S, 5000);
    });
  }

  /* A queue row carries no duration — the length of the audio is only known
     once the transcript exists, so it is read back off /api/library for
     finished items and left as an em dash for everything else. */
  function loadLengths(S) {
    if (S.lenBusy) return;
    var want = (S.queue || []).some(function (q) {
      return q && q.stage === 'done' && q.meeting && !(q.meeting in S.lens);
    });
    if (!want) return;
    S.lenBusy = true;
    S.ctx.api('/api/library').then(function (d) {
      S.lenBusy = false;
      if (S.dead) return;
      ((d && d.meetings) || []).forEach(function (m) {
        if (m && m.id) S.lens[m.id] = m.duration_s;
      });
      drawQueue(S);
    }, function () {
      S.lenBusy = false;
      /* leave the dashes; nothing here is worth inventing */
    });
  }

  function openMeeting(S, id) {
    try { S.ctx.go('transcript', { id: id }); } catch (e) { /* no transcript screen */ }
  }

  function queueRow(S, q) {
    var st = String(q.stage || 'queued');
    var running = RUNNING.indexOf(st) >= 0 || (st !== 'queued' && st !== 'done' && st !== 'failed');
    var tag = 'tag ' + (st === 'queued' ? 'tag-outline'
      : st === 'done' ? 'tag-neutral'
        : st === 'failed' ? 'tag-accent-2' : 'tag-accent');

    var pct = typeof q.pct === 'number' && isFinite(q.pct) ? Math.max(0, Math.min(100, q.pct)) : 0;
    var barInk = st === 'done' ? 'var(--color-neutral-400)'
      : st === 'failed' ? 'var(--color-accent-2-600)' : 'var(--color-accent-600)';
    var bar = 'height:5px;width:' + pct.toFixed(0) + '%;background:' + barInk;

    var el = null;
    if (typeof q.elapsed === 'number' && isFinite(q.elapsed)) el = q.elapsed;
    if (running && typeof q.started === 'number' && isFinite(q.started)) {
      /* the server only stamps elapsed at a stage change, and a stage can be
         minutes long; the server is on this machine, so its clock is ours. */
      var livesecs = Date.now() / 1000 - q.started;
      if (livesecs > 0 && livesecs < 31536000) el = Math.max(el || 0, livesecs);
    }
    var elapsed = st === 'queued' ? '—' : elapsedText(el);

    var lenSecs = q.stage === 'done' && q.meeting ? S.lens[q.meeting] : null;
    var len = durOf(S.ctx, lenSecs);

    var nameNode;
    if (st === 'done' && q.meeting) {
      nameNode = h('div', {
        style: 'color:var(--color-accent-700);cursor:pointer',
        onclick: function () { openMeeting(S, q.meeting); }
      }, String(q.file || q.meeting));
    } else {
      nameNode = h('div', null, String(q.file || '(unnamed)'));
    }

    var fileCell = h('td', { title: q.path ? String(q.path) : null }, [
      nameNode,
      st === 'failed' && q.error
        ? h('div', { style: 'font-size:11.5px;color:var(--color-accent-2-700);margin-top:3px;line-height:1.4' },
          String(q.error))
        : null
    ]);

    return h('tr', null, [
      fileCell,
      h('td', {
        style: 'font-variant-numeric:tabular-nums;color:var(--color-neutral-700)',
        title: len ? null : 'the length is only known once the transcript is written'
      }, len || '—'),
      h('td', null, h('span', { class: tag }, st)),
      h('td', null,
        h('div', { style: 'height:5px;background:color-mix(in srgb, var(--color-text) 8%, transparent)' },
          h('div', { style: bar }))),
      h('td', { style: 'text-align:right;font-variant-numeric:tabular-nums;color:var(--color-neutral-700)' },
        elapsed)
    ]);
  }

  function drawQueue(S) {
    if (S.dead || !S.tbody) return;
    var items = ordered(S.queue), c = counts(items), parts = [];

    if (S.queueErr) {
      S.qnote.textContent = 'the queue could not be read';
    } else if (!items.length) {
      S.qnote.textContent = 'nothing queued yet';
    } else {
      if (c.running) parts.push(c.running + ' running');
      if (c.waiting) parts.push(c.waiting + ' waiting');
      if (c.done) parts.push(c.done + ' done');
      if (c.failed) parts.push(c.failed + ' failed');
      S.qnote.textContent = parts.join(', ') +
        " · held in this server's memory, so a restart empties it";
    }

    clear(S.tbody);
    if (!items.length) {
      S.tbody.appendChild(h('tr', null,
        h('td', { colspan: '5', style: 'color:var(--color-neutral-700);font-size:13px' },
          S.queueErr
            ? 'The queue could not be read — ' + S.queueErr
            : 'Nothing queued. This list is what the server has been asked to do since it started; ' +
              'finished transcripts live in the library, not here.')));
      return;
    }
    items.forEach(function (q) { S.tbody.appendChild(queueRow(S, q)); });
  }

  // ----------------------------------------------------------------- ingest
  function say(S, msg, bad) {
    if (S.dead || !S.status) return;
    S.status.textContent = msg || '';
    S.status.setAttribute('style', 'font-size:11.5px;line-height:1.5;max-width:64ch;margin-bottom:' +
      (msg ? '12px' : '0') + ';color:' + (bad ? 'var(--color-accent-2-700)' : 'var(--color-neutral-700)'));
  }

  function send(S, paths) {
    if (!paths || !paths.length) return Promise.resolve(0);
    say(S, 'Adding ' + paths.length + (paths.length === 1 ? ' file…' : ' files…'), false);
    return S.ctx.api('/api/ingest', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ files: paths })
    }).then(function (r) {
      if (S.dead) return 0;
      var n = (r && r.queued && r.queued.length) || 0;
      say(S, n + (n === 1 ? ' file added to the queue.' : ' files added to the queue.'), false);
      schedule(S, 120);
      return n;
    }, function (e) {
      if (S.dead) return 0;
      var msg = errText(e);
      if (/no readable files/i.test(msg)) {
        msg = 'The server could not read any of those paths. It opens the files itself, off this ' +
          'machine, so the path has to exist and be readable by the server process.';
      }
      say(S, msg, true);
      throw e;
    });
  }

  // ----------------------------------------------------------------- picker
  /* /api/browse lists one directory at a time: {dir, parent, files, dirs}. It
     returns names, not sizes or durations, so this lists names. */
  function openPicker(S, opts) {
    opts = opts || {};
    var selected = Object.create(null);
    var pending = (opts.pending || []).slice(0, 40);
    var cur = null;
    var retried = false;

    var panel = h('div', { class: 'dialog', style: 'width:min(620px,100%);gap:var(--space-3)' });
    var close = null;
    function shut() {
      if (close) { try { close(); } catch (e) { /* already gone */ } }
      close = null;
      if (S.closeDialog) S.closeDialog = null;
    }

    var pathLine = h('div', {
      style: 'font-size:12.5px;font-variant-numeric:tabular-nums;overflow-wrap:anywhere;' +
        'color:var(--color-neutral-800);flex:1;min-width:0'
    }, '…');
    var upBtn = h('button', {
      class: 'btn btn-secondary',
      style: 'padding:4px 10px;font-size:12.5px',
      onclick: function () { if (cur) go(parentOf(cur)); }
    }, '↑ Up');

    var jump = h('input', {
      class: 'input', type: 'text', placeholder: '/path/to/a/folder',
      style: 'font-size:13px'
    });
    jump.addEventListener('keydown', function (e) {
      if (e.key === 'Enter') { e.preventDefault(); if (jump.value.trim()) go(jump.value.trim()); }
    });

    var listBox = h('div', {
      style: 'max-height:300px;overflow:auto;border:1px solid var(--color-divider);' +
        'border-radius:var(--radius-md);background:var(--color-bg)'
    });

    var tally = h('div', { style: SMALL });
    var allBtn = h('button', {
      class: 'btn btn-ghost', style: 'font-size:12.5px;padding:2px 6px',
      onclick: function () { pickAll(); }
    }, 'Select every file here');
    var addBtn = h('button', { class: 'btn btn-primary', onclick: function () { commit(); } }, 'Add to the queue');
    var dialogErr = h('div', { style: 'font-size:11.5px;color:var(--color-accent-2-700);line-height:1.5' });

    var here = [];

    function count() {
      var n = 0, k;
      for (k in selected) if (selected[k]) n++;
      return n;
    }

    function refreshTally() {
      var n = count();
      tally.textContent = n === 0 ? 'nothing selected'
        : n + (n === 1 ? ' file selected' : ' files selected');
      addBtn.disabled = n === 0;
      addBtn.textContent = n === 0 ? 'Add to the queue'
        : 'Add ' + n + (n === 1 ? ' file' : ' files') + ' to the queue';
    }

    function pickAll() {
      here.forEach(function (name) { selected[joinPath(cur, name)] = true; });
      draw();
    }

    function draw() {
      clear(listBox);
      var dirs = (S.browse && S.browse.dirs) || [];
      var files = here;

      if (!dirs.length && !files.length) {
        listBox.appendChild(h('div', { style: 'padding:14px 12px;font-size:13px;color:var(--color-neutral-700)' },
          'No sub-folders and no audio in this folder.'));
      }

      dirs.forEach(function (name) {
        listBox.appendChild(h('div', {
          style: 'display:flex;align-items:center;gap:10px;padding:6px 12px;font-size:13.5px;cursor:pointer;' +
            'border-bottom:1px solid color-mix(in srgb, var(--color-text) 6%, transparent)',
          onclick: function () { go(joinPath(cur, name)); }
        }, [
          h('span', { style: 'color:var(--color-neutral-600)' }, '▸'),
          h('span', { style: 'overflow-wrap:anywhere' }, name)
        ]));
      });

      files.forEach(function (name) {
        var full = joinPath(cur, name);
        var box = h('input', { type: 'checkbox' });
        box.checked = !!selected[full];
        box.addEventListener('change', function () {
          if (box.checked) selected[full] = true; else delete selected[full];
          refreshTally();
        });
        listBox.appendChild(h('label', {
          style: 'display:flex;align-items:center;gap:10px;padding:6px 12px;font-size:13.5px;cursor:pointer;' +
            'border-bottom:1px solid color-mix(in srgb, var(--color-text) 6%, transparent)'
        }, [box, h('span', { style: 'overflow-wrap:anywhere' }, name)]));
      });

      refreshTally();
    }

    function go(dir) {
      dialogErr.textContent = '';
      var url = dir ? '/api/browse?dir=' + encodeURIComponent(dir) : '/api/browse';
      S.ctx.api(url).then(function (d) {
        if (S.dead) return;
        retried = false;
        S.browse = { dirs: (d && d.dirs) || [] };
        cur = (d && d.dir) || dir || '';
        here = (d && d.files) || [];
        if (cur) S.ctx.state.ingestDir = cur;
        pathLine.textContent = cur || '(unknown folder)';
        upBtn.disabled = !cur || cur === '/';
        /* tick anything whose name came off the drop but arrived without a path */
        var hit = 0;
        pending.forEach(function (name) {
          if (here.indexOf(name) >= 0) { selected[joinPath(cur, name)] = true; hit++; }
        });
        if (hit) dialogErr.textContent = 'Ticked ' + hit + ' of the dropped ' +
          (pending.length === 1 ? 'file' : 'files') + ' found in this folder.';
        draw();
      }, function (e) {
        if (S.dead) return;
        if (!retried && dir && dir !== '/' && parentOf(dir) !== dir) {
          /* a dropped folder-ish path that is really a file, or a typo */
          retried = true;
          go(parentOf(dir));
          return;
        }
        dialogErr.textContent = errText(e);
      });
    }

    function commit() {
      var files = [], k;
      for (k in selected) if (selected[k]) files.push(k);
      if (!files.length) return;
      addBtn.disabled = true;
      addBtn.textContent = 'Adding…';
      send(S, files).then(function () { shut(); }, function () {
        if (S.dead) return;
        dialogErr.textContent = S.status ? S.status.textContent : 'That did not work.';
        refreshTally();
      });
    }

    panel.appendChild(h('div', { class: 'dialog-title' }, 'Choose files'));
    panel.appendChild(h('div', { class: 'dialog-body' },
      'The server opens the audio itself, off this machine — nothing is uploaded from the browser, ' +
      'so pick the files where they already are.'));
    if (pending.length) {
      panel.appendChild(h('div', {
        style: 'font-size:11.5px;color:var(--color-accent-2-700);line-height:1.5'
      }, 'Dropped ' + pending.slice(0, 4).join(', ') + (pending.length > 4 ? ' and ' + (pending.length - 4) + ' more' : '') +
        ' — this browser hands the page the file but not the folder it came from. Find the folder and ' +
        'they will be ticked for you.'));
    }
    panel.appendChild(h('div', { style: 'display:flex;align-items:center;gap:12px' }, [upBtn, pathLine]));
    panel.appendChild(h('div', { class: 'field' }, [h('label', null, 'Or type a folder'), jump]));
    panel.appendChild(listBox);
    panel.appendChild(h('div', { style: 'display:flex;align-items:center;gap:12px' }, [tally, allBtn]));
    panel.appendChild(dialogErr);
    panel.appendChild(h('div', { class: 'dialog-actions' }, [
      h('button', { class: 'btn btn-secondary', onclick: function () { shut(); } }, 'Cancel'),
      addBtn
    ]));

    close = S.ctx.dialog(panel);
    S.closeDialog = shut;
    refreshTally();
    go(opts.dir || S.ctx.state.ingestDir || '');
  }

  // ----------------------------------------------------------------- screen
  MS.screens.ingest = {
    title: 'Ingest',

    render: function (root, ctx) {
      teardown(live);
      var S = live = {
        ctx: ctx, dead: false, timer: null, queue: [], queueErr: null,
        lens: Object.create(null), lenBusy: false, settings: null,
        closeDialog: null, winDrag: null, browse: null, hot: 0
      };
      if (!S.ctx.state) S.ctx.state = {};

      var page = h('div', { style: 'max-width:900px' });

      page.appendChild(h('h1', { style: 'font-size:40px;margin:0 0 4px' }, 'Ingest'));
      page.appendChild(h('p', {
        style: 'font-size:13.5px;color:var(--color-neutral-700);max-width:56ch;margin:0 0 26px'
      }, 'Point it at a folder. Each stage writes to disk before the next one starts — the raw ' +
        'transcript, the embeddings and the linked transcript are separate files in the library.'));

      // -- drop zone
      var zone = h('div', { style: ZONE }, [
        h('div', { style: 'font-size:21px;font-weight:600;margin-bottom:6px' }, 'Drop meeting audio here'),
        h('div', { style: 'font-size:12.5px;color:var(--color-neutral-700);margin-bottom:16px' },
          'wav · mp3 · m4a · flac · ogg · mp4 · webm'),
        h('button', {
          class: 'btn btn-primary',
          onclick: function () { openPicker(S, {}); }
        }, 'Choose files')
      ]);
      page.appendChild(zone);
      S.zone = zone;

      function hot(on) {
        if (S.dead) return;
        zone.setAttribute('style', on ? ZONE_HOT : ZONE);
      }

      // -- status + backend warning
      S.status = h('div', { 'aria-live': 'polite', style: 'font-size:11.5px;line-height:1.5;max-width:64ch' });
      page.appendChild(S.status);
      S.warn = h('div', { style: 'font-size:11.5px;line-height:1.5;max-width:64ch' });
      page.appendChild(S.warn);

      S.note = h('div', {
        style: 'font-size:11.5px;color:var(--color-neutral-700);margin-bottom:38px;line-height:1.5;max-width:64ch'
      }, storageNote(null));
      page.appendChild(S.note);

      // -- queue
      S.qnote = h('div', { style: SMALL }, 'reading…');
      page.appendChild(h('div', { style: 'display:flex;align-items:baseline;gap:12px;margin-bottom:8px' }, [
        h('div', { style: LABEL }, 'Queue'),
        S.qnote
      ]));

      S.tbody = h('tbody');
      page.appendChild(h('table', { class: 'table' }, [
        h('thead', null, h('tr', null, [
          h('th', { style: 'width:38%' }, 'File'),
          h('th', { style: 'width:80px' }, 'Length'),
          h('th', null, 'Stage'),
          h('th', { style: 'width:150px' }, 'Progress'),
          h('th', { style: 'width:110px;text-align:right' }, 'Elapsed')
        ])),
        S.tbody
      ]));

      S.stages = h('div', {
        style: 'font-size:11.5px;color:var(--color-neutral-700);margin-top:16px;line-height:1.5;max-width:70ch'
      }, stagesNote(null));
      page.appendChild(S.stages);

      root.appendChild(page);

      // -- drag and drop over the whole screen, highlighting the zone
      root.addEventListener('dragenter', function (e) {
        if (e.dataTransfer) e.preventDefault();
        S.hot++;
        hot(true);
      });
      root.addEventListener('dragover', function (e) {
        e.preventDefault();
        try { e.dataTransfer.dropEffect = 'copy'; } catch (err) { /* not settable everywhere */ }
      });
      root.addEventListener('dragleave', function () {
        S.hot = Math.max(0, S.hot - 1);
        if (!S.hot) hot(false);
      });
      root.addEventListener('drop', function (e) {
        e.preventDefault();
        e.stopPropagation();
        S.hot = 0;
        hot(false);
        var dt = e.dataTransfer;
        if (!dt) return;

        var paths = pathsFromDrop(dt);
        var audio = paths.filter(function (p) { return isAudio(p); });
        if (audio.length) { send(S, audio).catch(function () { /* reported already */ }); return; }
        if (paths.length) { openPicker(S, { dir: paths[0] }); return; }

        var names = [], i;
        if (dt.files) for (i = 0; i < dt.files.length; i++) names.push(dt.files[i].name);
        if (!names.length) {
          say(S, 'Nothing recognisable in that drop.', true);
          return;
        }
        openPicker(S, { pending: names });
      });

      /* stop a stray drop anywhere else from navigating the page away */
      S.winDrag = function (e) { e.preventDefault(); };
      window.addEventListener('dragover', S.winDrag);
      window.addEventListener('drop', S.winDrag);

      // -- real facts for the two prose blocks, and the reachability warning
      ctx.api('/api/settings').then(function (s) {
        if (S.dead) return;
        S.settings = s;
        S.note.textContent = storageNote(s);
        S.stages.textContent = stagesNote(s);
        if (s && s.backend && s.backend.reachable === false) {
          S.warn.textContent = 'The pipeline is not reachable' +
            (s.backend.host ? ' on ' + s.backend.host : '') +
            ' — the library still reads, but anything added to the queue will fail until it is back.';
          S.warn.setAttribute('style',
            'font-size:11.5px;color:var(--color-accent-2-700);line-height:1.5;max-width:64ch;margin-bottom:12px');
        }
      }, function () {
        /* leave the neutral wording; better than claiming a mode we cannot read */
      });

      poll(S);
      return Promise.resolve();
    },

    destroy: function () {
      teardown(live);
      live = null;
    }
  };
}());
