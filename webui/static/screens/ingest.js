/* Ingest — design/Meetscribe.dc.html lines 111-141.

   Audio reaches the server by being uploaded: a browser will not tell a page
   where a chosen file lives, so a system file picker can only work if the bytes
   travel with it. This screen therefore offers the OS picker for files and for a
   folder, plus drag and drop, and POSTs each file to /api/upload before queuing
   the paths it gets back. An earlier version browsed the server's own
   directories from the page to avoid uploading; that was a lot of custom UI
   standing in for a dialog every operating system already has.
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
    var c = { running: 0, waiting: 0, held: 0, done: 0, failed: 0 };
    items.forEach(function (q) {
      var st = q.stage;
      if (st === 'held') c.held++;
      else if (st === 'queued') c.waiting++;
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
      S.waiting = (d && d.waiting) || 0;
      S.running = !!(d && d.running);
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

  /* Queue control. An individual file cannot be stopped once its batch is
     running -- the server hands every queued file to ONE ./transcribe so the
     engine loads once, which is ~70s per batch instead of per file. Holding a
     file before it starts, and cancelling the batch that is running, give the
     same control without surrendering that. Holding everything but one and
     pressing Run is how you process a single file. */
  function act(S, payload) {
    return S.ctx.api('/api/queue/act', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    }).then(function (r) {
      poll(S);
      return r;
    }, function (e) {
      say(S, errText(e), true);
      throw e;
    });
  }

  function rowActions(S, q) {
    var st = String(q.stage || 'queued');
    var wrap = h('div', { style: 'display:flex;gap:8px;justify-content:flex-end' });
    function btn(label, payload, title) {
      return h('span', {
        style: 'font-size:11.5px;cursor:pointer;color:var(--color-accent-700);' +
          'border-bottom:1px solid var(--color-accent-300)',
        role: 'button', tabindex: '0', title: title || null,
        onclick: function (ev) { ev.stopPropagation(); act(S, payload); }
      }, label);
    }
    if (st === 'queued') {
      wrap.appendChild(btn('hold', { action: 'hold', id: q.id },
        'Keep this out of the next batch.'));
      wrap.appendChild(btn('remove', { action: 'remove', id: q.id }));
    } else if (st === 'held') {
      wrap.appendChild(btn('release', { action: 'release', id: q.id },
        'Put it back in line for the next batch.'));
      wrap.appendChild(btn('remove', { action: 'remove', id: q.id }));
    } else if (st === 'failed') {
      wrap.appendChild(btn('retry', { action: 'retry', id: q.id }));
      wrap.appendChild(btn('remove', { action: 'remove', id: q.id }));
    } else if (st === 'done') {
      wrap.appendChild(btn('clear', { action: 'remove', id: q.id },
        'Take it off this list. The transcript stays in the library.'));
    }
    return wrap;
  }

  function queueRow(S, q) {
    var st = String(q.stage || 'queued');
    var running = RUNNING.indexOf(st) >= 0
      || (st !== 'queued' && st !== 'held' && st !== 'done' && st !== 'failed');
    var tag = 'tag ' + (st === 'queued' || st === 'held' ? 'tag-outline'
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
        elapsed),
      h('td', null, rowActions(S, q))
    ]);
  }

  /* A row is rebuilt only when something it displays has changed. The whole
     table used to be cleared and reconstructed on every poll, which is fine for
     six files and not for six hundred: a second-by-second rebuild of hundreds of
     rows churns the DOM, drops any text you had selected, and grows with the
     queue rather than with what actually moved. */
  function rowSig(S, q) {
    return [q.stage, q.pct, q.elapsed, q.started, q.meeting, q.error,
            q.stage === 'done' && q.meeting ? S.lens[q.meeting] : null].join('\u0001');
  }

  var QFILTERS = [
    ['all', 'All', null],
    ['running', 'Running', function (q) { return RUNNING.indexOf(String(q.stage)) >= 0
        || (q.stage !== 'queued' && q.stage !== 'done' && q.stage !== 'failed'); }],
    ['queued', 'Waiting', function (q) { return q.stage === 'queued'; }],
    ['held', 'Held', function (q) { return q.stage === 'held'; }],
    ['failed', 'Failed', function (q) { return q.stage === 'failed'; }],
    ['done', 'Done', function (q) { return q.stage === 'done'; }]
  ];
  var QPAGE = 80;

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
      if (c.held) parts.push(c.held + ' held');
      if (c.done) parts.push(c.done + ' done');
      if (c.failed) parts.push(c.failed + ' failed');
      S.qnote.textContent = parts.join(', ') +
        " · held in this server's memory, so a restart empties it";
    }

    /* Filter chips, so a long queue can be narrowed to the part you care about
       -- with three running behind two hundred done, scrolling is not a way to
       find them. Built once, restyled on each draw; a filter matching nothing is
       not offered. */
    if (S.qfilters) {
      clear(S.qfilters);
      if (items.length > 12) {
        QFILTERS.forEach(function (f) {
          var n = f[2] ? items.filter(f[2]).length : items.length;
          if (f[0] !== 'all' && !n) return;
          var on = (S.qfilter || 'all') === f[0];
          var chip = h('span', {
            style: 'font-size:12px;cursor:pointer;padding-bottom:2px;' +
              (on ? 'color:var(--color-text);border-bottom:2px solid var(--color-accent-600);'
                  : 'color:var(--color-neutral-700);border-bottom:2px solid transparent;'),
            role: 'button', tabindex: '0',
            onclick: function () { S.qfilter = f[0]; S.qshown = QPAGE; drawQueue(S); }
          }, f[1] + ' ' + n);
          S.qfilters.appendChild(chip);
        });
      }
    }

    if (S.qctl) {
      clear(S.qctl);
      /* Adding a file never starts it. This is the button that does, so it is
         the only primary action on the screen and it says how much it is about
         to commit to. */
      if (c.waiting) {
        S.qctl.appendChild(h('button', {
          class: 'btn btn-primary', style: 'font-size:12.5px;padding:3px 12px',
          title: 'Runs everything waiting as one batch. The engine loads once for '
            + 'the batch, so this is far faster than starting files one at a time.',
          onclick: function () { act(S, { action: 'run' }); }
        }, 'Process queue (' + c.waiting + ')'));
      }
      if (S.running) {
        S.qctl.appendChild(h('button', {
          class: 'btn btn-secondary', style: 'font-size:12.5px;padding:3px 10px',
          title: 'Stops the batch that is running. Files already transcribed keep their transcripts.',
          onclick: function () { act(S, { action: 'cancel' }); }
        }, 'Cancel the run'));
      }
    }

    var want = QFILTERS.filter(function (f) { return f[0] === (S.qfilter || 'all'); })[0];
    var shownItems = (want && want[2]) ? items.filter(want[2]) : items;

    if (!shownItems.length) {
      clear(S.tbody);
      S.rowCache = Object.create(null);
      S.tbody.appendChild(
        h('tr', null,
          h('td', { colspan: '6', style: 'color:var(--color-neutral-700);font-size:13px' },
            S.queueErr
              ? 'The queue could not be read — ' + S.queueErr
              : items.length
                ? 'Nothing in the queue matches this filter.'
                : 'Nothing queued. This list is what the server has been asked to do since it '
                  + 'started; finished transcripts live in the library, not here.')));
      return;
    }

    var cap = Math.min(S.qshown || QPAGE, shownItems.length);
    var visible = shownItems.slice(0, cap);

    var cache = S.rowCache || (S.rowCache = Object.create(null));
    var seen = Object.create(null);
    var prev = null;
    visible.forEach(function (q) {
      var key = String(q.id || q.path || q.file);
      seen[key] = true;
      var sig = rowSig(S, q);
      var hit = cache[key];
      if (!hit || hit.sig !== sig) {
        var tr = queueRow(S, q);
        if (hit && hit.tr.parentNode === S.tbody) S.tbody.replaceChild(tr, hit.tr);
        hit = cache[key] = { tr: tr, sig: sig };
      }
      // keep DOM order in step with sort order, moving only what is out of place
      var want_after = prev ? prev.nextSibling : S.tbody.firstChild;
      if (hit.tr !== want_after) S.tbody.insertBefore(hit.tr, want_after);
      prev = hit.tr;
    });
    // drop rows that are no longer shown
    Object.keys(cache).forEach(function (k) {
      if (!seen[k]) {
        if (cache[k].tr.parentNode === S.tbody) S.tbody.removeChild(cache[k].tr);
        delete cache[k];
      }
    });
    if (S.qmore) {
      clear(S.qmore);
      if (cap < shownItems.length) {
        S.qmore.appendChild(h('span', {
          style: 'font-size:12.5px;cursor:pointer;color:var(--color-accent-700);'
            + 'border-bottom:1px solid var(--color-accent-300)',
          role: 'button', tabindex: '0',
          onclick: function () { S.qshown = cap + QPAGE; drawQueue(S); }
        }, 'Show ' + Math.min(QPAGE, shownItems.length - cap) + ' more'));
        S.qmore.appendChild(h('span', {
          style: 'font-size:12.5px;color:var(--color-neutral-600);margin-left:12px;'
            + 'font-variant-numeric:tabular-nums'
        }, cap + ' of ' + shownItems.length));
      }
    }
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

  /* Uploading, because the browser refuses to tell a page where a chosen file
     lives -- it gives the contents and a bare name and nothing else. That is a
     deliberate boundary, and it is why this screen used to carry its own
     directory browser: the server reads audio by path, so the page had to name a
     path some other way. Sending the bytes removes the need for any of that, and
     the system picker then works for one file, many files or a whole folder with
     no custom UI to get wrong.

     One request per file so progress is per file and one failure does not lose
     the batch. */
  function uploadOne(S, file) {
    return fetch('/api/upload', {
      method: 'POST',
      headers: { 'X-Filename': encodeURIComponent(file.name),
                 'Content-Type': 'application/octet-stream' },
      body: file
    }).then(function (res) {
      return res.json().catch(function () { return {}; }).then(function (j) {
        if (!res.ok || !j.path) throw new Error(j.error || (file.name + ': upload failed'));
        return j.path;
      });
    });
  }

  function uploadAndQueue(S, fileList) {
    var files = [];
    for (var i = 0; i < fileList.length; i++) {
      if (isAudio(fileList[i].name)) files.push(fileList[i]);
    }
    var skipped = fileList.length - files.length;
    if (!files.length) {
      say(S, skipped ? 'None of those ' + skipped + ' files is audio this can read.'
                     : 'Nothing to add.', true);
      return Promise.resolve();
    }
    var done = 0, paths = [], failed = [];
    say(S, 'Uploading 0 of ' + files.length + '…');
    // sequential: a folder of hour-long recordings saturates the link either
    // way, and one at a time keeps the count honest and the server's disk
    // writes ordered
    return files.reduce(function (chain, f) {
      return chain.then(function () {
        return uploadOne(S, f).then(function (path) {
          paths.push(path);
        }).catch(function (e) {
          failed.push(f.name + ' — ' + errText(e));
        }).then(function () {
          done++;
          if (!S.dead) say(S, 'Uploading ' + done + ' of ' + files.length + '…');
        });
      });
    }, Promise.resolve()).then(function () {
      if (!paths.length) {
        say(S, 'Nothing uploaded. ' + failed.join('; '), true);
        return;
      }
      return send(S, paths).then(function () {
        var note = [];
        if (skipped) note.push(skipped + ' not audio');
        if (failed.length) note.push(failed.length + ' failed to upload');
        if (note.length) say(S, paths.length + ' queued · ' + note.join(', '), !!failed.length);
      });
    });
  }

  // ----------------------------------------------------------------- picker
  /* /api/browse lists one directory at a time: {dir, parent, files, dirs}. It
     returns names, not sizes or durations, so this lists names. */
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

      // -- drop zone, with the SYSTEM pickers rather than a browser of our own
      var filesInput = h('input', {
        type: 'file', multiple: 'multiple', style: 'display:none',
        accept: AUDIO_EXT.join(',')
      });
      filesInput.addEventListener('change', function () {
        if (filesInput.files && filesInput.files.length) uploadAndQueue(S, filesInput.files);
        filesInput.value = '';
      });
      var dirInput = h('input', { type: 'file', multiple: 'multiple', style: 'display:none' });
      /* webkitdirectory is how a browser offers "pick a folder"; it is not in
         the HTML attribute list h() knows, and Firefox needs the mozdirectory
         alias, so both are set directly. Non-audio inside the folder is filtered
         out on the way through. */
      dirInput.setAttribute('webkitdirectory', '');
      dirInput.setAttribute('mozdirectory', '');
      dirInput.addEventListener('change', function () {
        if (dirInput.files && dirInput.files.length) uploadAndQueue(S, dirInput.files);
        dirInput.value = '';
      });

      var zone = h('div', { style: ZONE }, [
        h('div', { style: 'font-size:21px;font-weight:600;margin-bottom:6px' }, 'Drop meeting audio here'),
        h('div', { style: 'font-size:12.5px;color:var(--color-neutral-700);margin-bottom:16px' },
          'wav · mp3 · m4a · flac · ogg · mp4 · webm'),
        h('div', { style: 'display:flex;gap:10px;justify-content:center' }, [
          h('button', { class: 'btn btn-primary',
            onclick: function () { filesInput.click(); } }, 'Choose files'),
          h('button', { class: 'btn btn-secondary',
            onclick: function () { dirInput.click(); } }, 'Choose a folder')
        ]),
        filesInput, dirInput
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

      /* Filters sit above the table and only appear once the queue is long
         enough to need them (drawQueue decides), so a handful of files is not
         cluttered by controls for a problem it does not have. */
      S.qfilters = h('div', { style: 'display:flex;align-items:baseline;gap:15px;flex-wrap:wrap' });
      /* Queue-wide controls. "Hold new files" is what makes a batch something
         you assemble rather than something that starts under you: add forty
         recordings, look at them, then Run. */
      S.qctl = h('div', { style: 'display:flex;align-items:center;gap:14px;margin-left:auto' });
      page.appendChild(h('div', {
        style: 'display:flex;align-items:center;gap:15px;flex-wrap:wrap;margin-bottom:9px'
      }, [S.qfilters, S.qctl]));

      S.tbody = h('tbody');
      page.appendChild(h('table', { class: 'table' }, [
        h('thead', null, h('tr', null, [
          h('th', { style: 'width:38%' }, 'File'),
          h('th', { style: 'width:80px' }, 'Length'),
          h('th', null, 'Stage'),
          h('th', { style: 'width:150px' }, 'Progress'),
          h('th', { style: 'width:110px;text-align:right' }, 'Elapsed'),
          h('th', { style: 'width:120px' }, '')
        ])),
        S.tbody
      ]));

      S.qmore = h('div', { style: 'padding:12px 0;display:flex;align-items:baseline' });
      page.appendChild(S.qmore);

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

        /* A drop carries the FILES, which is all we need now that the bytes
           travel. The old code tried to recover paths from the drag data --
           which only Firefox provides -- and fell back to a picker when it
           could not. Both are gone. */
        if (dt.files && dt.files.length) {
          uploadAndQueue(S, dt.files);
          return;
        }
        say(S, 'Nothing recognisable in that drop.', true);
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
