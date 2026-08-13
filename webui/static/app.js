/* Meetscribe — the shell.
 *
 * Design: design/Meetscribe.dc.html lines 1-37 (masthead + scrolling content
 * area) and 312-343 (the player bar). Every inline style string below is
 * carried over from that file; the sample numbers in it are not — the counts,
 * the badge, the lanes and the speaker names all come from the server.
 *
 * This file owns:
 *   · the masthead, the nav row and the hash router
 *   · the ctx object handed to every screen (api / go / state / ink / fmt /
 *     dialog / player)
 *   · one real <audio> element on /api/audio?id=… and the transport bar over it
 *   · a visible failure state, because a screen that throws must say so
 *
 * It owns nothing inside a screen. Screens live in /static/screens/<name>.js
 * and register themselves on window.MS.screens.<name>.
 *
 * ONE DELIBERATE DEPARTURE FROM THE DESIGN'S COPY, at the top of the page:
 * design line 23 reads "Local transcription · nothing leaves this machine".
 * That is true of an all-in-one install and false of a split one, where the
 * audio is rsynced to a GPU box over ssh (see ui/backend.py, RemoteBackend).
 * The tagline is therefore chosen from the mode the server reports, and is
 * blank until the server has answered. A privacy claim is the last thing that
 * should be hardcoded.
 */
(function () {
  'use strict';

  var MS = (window.MS = window.MS || {});
  MS.screens = MS.screens || {};

  // ------------------------------------------------------------- small DOM
  function el(tag, style, text) {
    var n = document.createElement(tag);
    if (style) n.setAttribute('style', style);
    if (text != null) n.textContent = String(text);
    return n;
  }
  function clear(n) { while (n && n.firstChild) n.removeChild(n.firstChild); }
  function num(x) { return typeof x === 'number' && isFinite(x); }
  function toNum(x) {
    if (num(x)) return x;
    if (typeof x === 'string' && x.trim() !== '') {
      var v = Number(x);
      if (isFinite(v)) return v;
    }
    return null;
  }

  // ------------------------------------------------------------- the inks
  /* The speaker palette, from the design (lines 483-489). Index into the
     meeting's speaker order; wrap past the end. Variant 'fill'/0 is the block
     colour, 'text'/1 the readable one. This is the only copy of the palette in
     the app — screens are told to call it, not to restate it. */
  var INK = [
    ['var(--color-neutral-900)', 'var(--color-neutral-900)'],
    ['var(--color-accent-600)', 'var(--color-accent-700)'],
    ['var(--color-accent-2-600)', 'var(--color-accent-2-700)'],
    ['var(--color-process-yellow)', '#8a6b00'],
    ['var(--color-neutral-400)', 'var(--color-neutral-700)']
  ];
  function ink(i, variant) {
    var n = Number(i);
    if (!isFinite(n)) n = 0;
    n = Math.floor(n);
    var row = INK[((n % INK.length) + INK.length) % INK.length];
    var wantText = (variant === 'text' || variant === 1 || variant === '1' || variant === true);
    return row[wantText ? 1 : 0];
  }
  MS.ink = ink;

  // -------------------------------------------------------- the formatters
  function pad2(n) { return (n < 10 ? '0' : '') + n; }
  /* h:mm:ss, always with hours — the design's hms() (line 585). */
  function hms(s) {
    var w = Math.max(0, Math.floor(toNum(s) || 0));
    return Math.floor(w / 3600) + ':' + pad2(Math.floor((w % 3600) / 60)) + ':' + pad2(w % 60);
  }
  /* m:ss, minutes uncapped — the design's ms() (line 586). */
  function clock(s) {
    var w = Math.max(0, Math.floor(toNum(s) || 0));
    return Math.floor(w / 60) + ':' + pad2(w % 60);
  }
  /* A duration as it is written in the design's meeting rows: "42:10" under an
     hour, "1:48:20" over it. */
  function dur(s) {
    var w = Math.max(0, Math.floor(toNum(s) || 0));
    if (w < 3600) return Math.floor(w / 60) + ':' + pad2(w % 60);
    return Math.floor(w / 3600) + ':' + pad2(Math.floor((w % 3600) / 60)) + ':' + pad2(w % 60);
  }
  var fmt = { hms: hms, clock: clock, dur: dur };
  MS.fmt = fmt;

  // ---------------------------------------------------------------- fetch
  /* Throws on non-2xx, with the server's own JSON body attached. The screens
     read err.body/.data/.payload/.json for the `reason` codes the API returns
     (name-taken, below-enrolment-floor, candidate-unknown …), so the body has
     to survive the throw rather than be flattened into a string. */
  function api(path, opts) {
    var url = String(path);
    return fetch(url, opts || undefined).then(function (res) {
      var ct = res.headers.get('Content-Type') || '';
      var read = ct.indexOf('json') >= 0
        ? res.json().catch(function () { return null; })
        : res.text().catch(function () { return ''; });
      return read.then(function (body) {
        if (res.ok) return body;
        var obj = (body && typeof body === 'object') ? body : null;
        var msg = (obj && typeof obj.error === 'string' && obj.error) ? obj.error
          : (typeof body === 'string' && body.trim()) ? body.trim().slice(0, 400)
            : ('HTTP ' + res.status + (res.statusText ? ' ' + res.statusText : '') + ' from ' + url);
        var err = new Error(msg);
        err.status = res.status;
        err.statusText = res.statusText;
        err.url = url;
        err.body = obj;
        err.data = obj;
        err.payload = obj;
        err.json = obj;
        err.response = obj;
        if (obj && typeof obj.reason === 'string') err.reason = obj.reason;
        throw err;
      });
    }, function (netErr) {
      var e = new Error('the server did not answer (' +
        ((netErr && netErr.message) || 'network error') + ') — ' + url);
      e.status = 0;
      e.url = url;
      e.body = null;
      throw e;
    });
  }
  MS.api = api;

  // --------------------------------------------------------------- dialogs
  /* ctx.dialog(node): put a screen's .dialog panel inside a .dialog-backdrop
     and return the function that takes it away again. Escape closes the
     topmost one; navigating away closes all of them, so a modal cannot outlive
     the screen that opened it. */
  var openDialogs = [];
  function dialog(node) {
    var back = el('div', 'z-index:40');
    back.className = 'dialog-backdrop';
    if (node) back.appendChild(node);
    var entry = { back: back, closed: false, close: null };
    var restore = document.activeElement;

    function onKey(e) {
      if (e.key !== 'Escape') return;
      if (openDialogs[openDialogs.length - 1] !== entry) return;
      e.preventDefault();
      e.stopPropagation();
      close();
    }
    function close() {
      if (entry.closed) return;
      entry.closed = true;
      document.removeEventListener('keydown', onKey, true);
      var i = openDialogs.indexOf(entry);
      if (i >= 0) openDialogs.splice(i, 1);
      if (back.parentNode) back.parentNode.removeChild(back);
      try { if (restore && restore.focus && document.contains(restore)) restore.focus(); } catch (e) { /* gone */ }
    }
    entry.close = close;
    openDialogs.push(entry);
    document.addEventListener('keydown', onKey, true);
    document.body.appendChild(back);
    setTimeout(function () {
      if (entry.closed) return;
      var f = back.querySelector('input:not([type=hidden]),select,textarea,button,[tabindex]');
      try { if (f && f.focus) f.focus(); } catch (e) { /* not focusable */ }
    }, 0);
    return close;
  }
  function closeAllDialogs() {
    openDialogs.slice().forEach(function (d) { try { d.close(); } catch (e) { /* already gone */ } });
  }

  // ---------------------------------------------------------------- alerts
  /* Anything that fails outside a screen's render() — a late rejected promise,
     a throw inside an event handler — says so on the page. Silence would be
     indistinguishable from working. */
  var alerts = null;
  function notify(text, detail) {
    if (!alerts) return;
    if (alerts.childNodes.length >= 3) alerts.removeChild(alerts.firstChild);
    var row = el('div', 'margin:0 0 10px;padding:9px 12px;border-left:3px solid var(--color-accent-2-600);' +
      'background:var(--color-accent-2-100);color:var(--color-accent-2-800);font-size:12.5px;' +
      'line-height:1.5;display:flex;gap:12px;align-items:baseline');
    var body = el('div', 'min-width:0;flex:1');
    body.appendChild(el('div', 'font-weight:600', text));
    if (detail) body.appendChild(el('div', 'font-size:11.5px;opacity:0.85;word-break:break-word', detail));
    row.appendChild(body);
    var x = el('div', 'flex:none;cursor:pointer;font-size:11px;letter-spacing:0.1em;text-transform:uppercase', 'Dismiss');
    x.setAttribute('role', 'button');
    x.setAttribute('tabindex', '0');
    function go() { if (row.parentNode) row.parentNode.removeChild(row); }
    x.addEventListener('click', go);
    x.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(); }
    });
    row.appendChild(x);
    alerts.appendChild(row);
  }

  // ---------------------------------------------------------------- player
  /* A real <audio> element on /api/audio?id=…  The server answers byte ranges,
     so a seek is a range request and not a re-download.
     When a meeting has no audio on disk (both of the real ones do not, today)
     the element is never given a src: the transport is disabled and says so,
     and the clock becomes a plain variable, so clicking a line still moves the
     mark. That is what the transcript screen promises in its rail, and it is
     the honest half of a player with nothing to play. */
  function makePlayer() {
    var audio = document.createElement('audio');
    audio.preload = 'metadata';
    audio.setAttribute('style', 'display:none');

    var subs = Object.create(null);
    var meetingId = null;   // meeting the bar is currently showing
    var meta = null;        // {duration, lanes, rows, starts}
    var hasAudio = false;   // a src was set and the server had a file
    var vclock = 0;         // the clock when there is no audio to be one
    var rate = 1;
    var mounted = false;
    var ticker = null;
    var dragging = false;
    var loadFailed = false;

    // bar nodes, rebuilt per attach
    var bar = null, playBtn = null, timeNow = null, timeTotal = null,
      scrub = null, laneBox = null, tickBox = null, head = null, whoBox = null,
      rateBox = null, noteBox = null;

    function emit(evt, val) {
      var list = subs[evt];
      if (!list) return;
      list.slice().forEach(function (cb) {
        try { cb(val); } catch (e) { notify('A player listener threw.', String((e && e.message) || e)); }
      });
    }
    function on(evt, cb) {
      if (typeof cb !== 'function') return function () {};
      var key = (evt === 'timeupdate') ? 'time' : String(evt);
      (subs[key] = subs[key] || []).push(cb);
      return function off() {
        var l = subs[key];
        if (!l) return;
        var i = l.indexOf(cb);
        if (i >= 0) l.splice(i, 1);
      };
    }

    function duration() {
      if (meta && num(meta.duration) && meta.duration > 0) return meta.duration;
      if (hasAudio && num(audio.duration) && audio.duration > 0) return audio.duration;
      return 0;
    }
    function time() {
      if (hasAudio) {
        var t = audio.currentTime;
        return num(t) ? t : 0;
      }
      return vclock;
    }
    function playable() { return hasAudio && !loadFailed; }
    function seek(t) {
      var v = toNum(t);
      if (v === null) return;
      var d = duration();
      v = Math.max(0, d > 0 ? Math.min(d, v) : v);
      if (playable()) {
        try { audio.currentTime = v; } catch (e) { vclock = v; }
      } else {
        vclock = v;
      }
      paintHead();
      emit('time', time());
      emit('seek', time());
    }
    function play() {
      if (!playable()) return;
      var p = audio.play();
      if (p && typeof p.catch === 'function') {
        p.catch(function (e) {
          notify('The browser would not play this audio.', String((e && e.message) || e));
          paintTransport();
        });
      }
    }
    function pause() { try { audio.pause(); } catch (e) { /* nothing playing */ } }
    function toggle() {
      if (!playable()) return;
      if (audio.paused) play(); else pause();
    }
    function setRate(r) {
      var v = toNum(r);
      if (v === null || v <= 0) return;
      rate = v;
      try { audio.playbackRate = v; } catch (e) { /* not loaded */ }
      paintRates();
      emit('rate', v);
    }

    // ---- metadata for the lanes
    function buildMeta(m) {
      var d = toNum(m && m.duration_s) || 0;
      var speakers = (m && Array.isArray(m.speakers)) ? m.speakers : [];
      var index = Object.create(null);
      speakers.forEach(function (s, i) { if (s && s.id != null) index[s.id] = i; });

      var rows = [];
      ((m && Array.isArray(m.turns)) ? m.turns : []).forEach(function (t) {
        if (!t) return;
        var lines = Array.isArray(t.lines) && t.lines.length ? t.lines
          : [{ start: t.start, end: t.end }];
        lines.forEach(function (l) {
          var a = toNum(l && l.start), b = toNum(l && l.end);
          if (a === null || b === null || b < a) return;
          rows.push({ start: a, end: b, g: t.speaker, name: t.name || '' });
        });
      });
      rows.sort(function (x, y) { return x.start - y.start; });

      // one lane per speaker, in the order /api/meeting returns them (most
      // speech first) — the same order ctx.ink indexes into.
      var lanes = speakers.map(function (s, i) {
        return { key: s.id, name: s.name || '', colour: ink(i, 'fill'), blocks: [], title: (s.name || String(s.id)) };
      });
      // Segments the pipeline attributed to nobody (G-1) have no speaker and so
      // no ink; they get one grey strip at the foot rather than being dropped,
      // because they are real seconds of the recording.
      var loose = null;
      rows.forEach(function (r) {
        var i = index[r.g];
        if (i === undefined) {
          if (!loose) {
            loose = { key: '__loose__', name: 'Unattributed', colour: 'var(--color-neutral-300)', blocks: [],
              title: 'Unattributed — the pipeline assigned these seconds to no speaker' };
          }
          loose.blocks.push(r);
        } else {
          lanes[i].blocks.push(r);
        }
      });
      if (loose) lanes.push(loose);

      return { duration: d, lanes: lanes, rows: rows, starts: rows.map(function (r) { return r.start; }) };
    }

    function speakerAt(t) {
      if (!meta || !meta.rows.length) return null;
      var s = meta.starts, lo = 0, hi = s.length - 1, best = -1;
      while (lo <= hi) {
        var mid = (lo + hi) >> 1;
        if (s[mid] <= t) { best = mid; lo = mid + 1; } else { hi = mid - 1; }
      }
      if (best < 0) return null;
      var r = meta.rows[best];
      // the design's rule: a segment stays current for 0.6 s past its end
      return (t < r.end + 0.6) ? r : null;
    }

    // ---- the bar (design lines 312-343)
    function buildBar(host) {
      clear(host);
      bar = el('div', 'display:flex;align-items:center;gap:22px');
      host.appendChild(bar);

      playBtn = document.createElement('button');
      playBtn.className = 'btn btn-primary btn-icon';
      playBtn.setAttribute('style', 'flex:none;width:38px;height:38px');
      playBtn.addEventListener('click', toggle);
      bar.appendChild(playBtn);

      timeNow = el('div', 'flex:none;font-size:13px;font-variant-numeric:tabular-nums;width:104px;white-space:nowrap');
      bar.appendChild(timeNow);

      scrub = el('div', 'flex:1;min-width:0;position:relative;cursor:ew-resize;padding:2px 0');
      scrub.setAttribute('role', 'slider');
      scrub.setAttribute('tabindex', '0');
      scrub.setAttribute('aria-label', 'Position in the recording');
      laneBox = el('div');
      tickBox = el('div', 'position:relative;height:13px;margin-top:3px');
      head = el('div');
      scrub.appendChild(laneBox);
      scrub.appendChild(tickBox);
      scrub.appendChild(head);
      bar.appendChild(scrub);

      scrub.addEventListener('pointerdown', function (e) {
        if (!duration()) return;
        dragging = true;
        try { scrub.setPointerCapture(e.pointerId); } catch (_) { /* older browser */ }
        scrubTo(e);
        e.preventDefault();
      });
      scrub.addEventListener('pointermove', function (e) { if (dragging) scrubTo(e); });
      scrub.addEventListener('pointerup', function () { dragging = false; });
      scrub.addEventListener('pointercancel', function () { dragging = false; });
      scrub.addEventListener('keydown', function (e) {
        if (e.key === 'ArrowLeft') { seek(time() - 5); e.preventDefault(); }
        if (e.key === 'ArrowRight') { seek(time() + 5); e.preventDefault(); }
      });

      rateBox = el('div', 'display:flex;flex:none;border:1px solid var(--color-divider);' +
        'border-radius:var(--radius-md);overflow:hidden');
      bar.appendChild(rateBox);

      noteBox = el('div', 'display:none');
      bar.appendChild(noteBox);

      whoBox = el('div', 'flex:none;font-size:12.5px;color:var(--color-neutral-700);width:180px;text-align:right');
      bar.appendChild(whoBox);

      paintTransport();
      paintRates();
      paintNote();
      paintLanes();
      paintHead();
    }

    function scrubTo(e) {
      var d = duration();
      if (!d) return;
      var r = scrub.getBoundingClientRect();
      if (!r.width) return;
      seek((e.clientX - r.left) / r.width * d);
    }

    var PLAY_SVG = '<svg width="13" height="14" viewBox="0 0 13 14" aria-hidden="true">' +
      '<path d="M2 0.8 12 7 2 13.2Z" fill="currentColor"></path></svg>';
    var PAUSE_SVG = '<svg width="12" height="13" viewBox="0 0 12 13" aria-hidden="true">' +
      '<rect x="1" y="0.5" width="3.4" height="12" fill="currentColor"></rect>' +
      '<rect x="7.6" y="0.5" width="3.4" height="12" fill="currentColor"></rect></svg>';

    function paintTransport() {
      if (!playBtn) return;
      var playing = playable() && !audio.paused;
      playBtn.innerHTML = playing ? PAUSE_SVG : PLAY_SVG;
      playBtn.disabled = !playable();
      playBtn.setAttribute('aria-label', playing ? 'Pause' : 'Play');
      playBtn.title = playable()
        ? (playing ? 'Pause (space)' : 'Play (space)')
        : (loadFailed
          ? 'The server has audio for this meeting but it would not load — see /api/audio'
          : 'No audio sits beside this transcript, so there is nothing to play. The mark still moves.');
    }

    function paintRates() {
      if (!rateBox) return;
      clear(rateBox);
      if (!playable()) { rateBox.setAttribute('style', 'display:none'); return; }
      rateBox.setAttribute('style', 'display:flex;flex:none;border:1px solid var(--color-divider);' +
        'border-radius:var(--radius-md);overflow:hidden');
      [1, 1.5, 2].forEach(function (r, i) {
        var on = Math.abs(rate - r) < 1e-6;
        var lab = document.createElement('label');
        lab.setAttribute('style', 'position:relative;display:inline-flex;align-items:center;justify-content:center;' +
          'min-width:38px;padding:5px 8px;font-size:12px;cursor:pointer;font-variant-numeric:tabular-nums;' +
          (i ? 'border-left:1px solid var(--color-divider);' : '') +
          (on ? 'background:var(--color-accent);color:var(--color-bg);' : 'color:var(--color-neutral-700);'));
        var inp = document.createElement('input');
        inp.type = 'radio';
        inp.name = 'ms-rate';
        inp.checked = on;
        inp.setAttribute('style', 'position:absolute;opacity:0;width:0;height:0;margin:0');
        inp.addEventListener('change', function () { setRate(r); });
        lab.appendChild(inp);
        lab.appendChild(document.createTextNode(r + '×'));
        rateBox.appendChild(lab);
      });
    }

    function paintNote() {
      if (!noteBox) return;
      if (playable()) { noteBox.setAttribute('style', 'display:none'); noteBox.textContent = ''; return; }
      noteBox.setAttribute('style', 'flex:none;font-size:11.5px;color:var(--color-neutral-600);max-width:230px;line-height:1.4');
      noteBox.textContent = loadFailed ? 'audio would not load' : 'no audio on disk';
      noteBox.title = loadFailed
        ? 'The transcript says a recording exists, but /api/audio did not serve it.'
        : 'No recording sits beside this transcript, so there is nothing to play. ' +
          'The lanes below are the diarisation, and dragging still moves the mark.';
    }

    function tickStep(d) {
      var STEPS = [15, 30, 60, 120, 300, 600, 900, 1800, 3600];
      for (var i = 0; i < STEPS.length; i++) if (d / STEPS[i] <= 7) return STEPS[i];
      return STEPS[STEPS.length - 1];
    }

    function paintLanes() {
      if (!laneBox) return;
      clear(laneBox);
      clear(tickBox);
      var d = duration();
      if (!meta || !d) return;

      var frag = document.createDocumentFragment();
      meta.lanes.forEach(function (lane) {
        var row = el('div', 'position:relative;height:8px;margin-bottom:2px;background:color-mix(in srgb, var(--color-text) 7%, transparent)');
        if (lane.title) row.title = lane.title;
        lane.blocks.forEach(function (b) {
          row.appendChild(el('div',
            'position:absolute;top:0;bottom:0;left:' + (b.start / d * 100).toFixed(2) +
            '%;width:' + Math.max(0.35, (b.end - b.start) / d * 100).toFixed(2) +
            '%;background:' + lane.colour));
        });
        frag.appendChild(row);
      });
      laneBox.appendChild(frag);

      var step = tickStep(d);
      for (var t = 0; t < d; t += step) {
        tickBox.appendChild(el('div',
          'position:absolute;left:' + (t / d * 100).toFixed(2) + '%;top:0;font-size:9.5px;letter-spacing:0.08em;' +
          'color:var(--color-neutral-600);font-variant-numeric:tabular-nums;padding-left:3px;' +
          'border-left:1px solid color-mix(in srgb, var(--color-text) 22%, transparent);height:11px',
          clock(t)));
      }
    }

    var lastPainted = null;
    function paintHead(force) {
      if (!head) return;
      var d = duration(), t = time();
      if (!force && lastPainted !== null && Math.abs(lastPainted - t) < 0.02) return;
      lastPainted = t;
      head.setAttribute('style', 'position:absolute;left:' + (d ? (t / d * 100).toFixed(2) : '0') +
        '%;top:0;bottom:14px;width:1px;background:var(--color-accent);box-shadow:0 -4px 0 2px var(--color-accent)');
      if (timeNow) {
        clear(timeNow);
        var long = d >= 3600;
        timeNow.appendChild(document.createTextNode((long ? hms(t) : clock(t)) + ' '));
        timeNow.appendChild(el('span', 'color:var(--color-neutral-600)',
          '/ ' + (d ? (long ? hms(d) : clock(d)) : '—')));
        timeNow.setAttribute('style', 'flex:none;font-size:13px;font-variant-numeric:tabular-nums;width:' +
          (long ? '134px' : '104px') + ';white-space:nowrap');
      }
      if (scrub && d) {
        scrub.setAttribute('aria-valuemin', '0');
        scrub.setAttribute('aria-valuemax', String(Math.round(d)));
        scrub.setAttribute('aria-valuenow', String(Math.round(t)));
        scrub.setAttribute('aria-valuetext', hms(t));
      }
      if (whoBox) {
        var r = speakerAt(t);
        whoBox.textContent = r ? (r.name || String(r.g)) : 'silence';
      }
    }

    var lastEmitted = null;
    function tick() {
      paintHead();
      var t = time();
      if (lastEmitted === null || Math.abs(t - lastEmitted) > 0.005) {
        lastEmitted = t;
        emit('time', t);
      }
    }

    audio.addEventListener('play', function () { paintTransport(); emit('play', time()); });
    audio.addEventListener('pause', function () { paintTransport(); emit('pause', time()); });
    audio.addEventListener('ratechange', function () { rate = audio.playbackRate; paintRates(); });
    audio.addEventListener('timeupdate', tick);
    audio.addEventListener('loadedmetadata', function () {
      loadFailed = false;
      try { audio.playbackRate = rate; } catch (e) { /* ignore */ }
      paintTransport(); paintRates(); paintNote(); paintLanes(); paintHead(true);
      emit('duration', duration());
    });
    audio.addEventListener('error', function () {
      // The server said has_audio, and then would not serve it. Say so rather
      // than leave a dead play button.
      if (!audio.getAttribute('src')) return;
      loadFailed = true;
      hasAudio = false;
      paintTransport(); paintRates(); paintNote();
      notify('The audio for this meeting would not load.',
        'GET /api/audio?id=' + encodeURIComponent(meetingId || '') + ' failed. The transcript still works; the mark moves without sound.');
    });
    audio.addEventListener('ended', function () { paintTransport(); });

    /* Point the bar at a meeting. Cheap and idempotent when it is already
       there, because navigating transcript → review → transcript must not
       restart the recording. */
    function attach(id, host) {
      mounted = true;
      if (host) host.style.display = '';
      if (meetingId === id && meta) {
        if (host && (!bar || bar.parentNode !== host)) buildBar(host);
        paintTransport(); paintRates(); paintNote(); paintLanes(); paintHead(true);
        startTicker();
        return Promise.resolve(meta);
      }

      pause();
      meetingId = id;
      meta = null;
      hasAudio = false;
      loadFailed = false;
      vclock = 0;
      lastPainted = null;
      lastEmitted = null;
      audio.removeAttribute('src');
      try { audio.load(); } catch (e) { /* nothing to unload */ }

      if (host) buildBar(host);
      paintNote();
      startTicker();

      var mine = id;
      return api('/api/meeting?id=' + encodeURIComponent(id)).then(function (m) {
        if (meetingId !== mine) return null;
        meta = buildMeta(m);
        if (m && m.has_audio) {
          hasAudio = true;
          audio.setAttribute('src', '/api/audio?id=' + encodeURIComponent(id));
          try { audio.load(); } catch (e) { /* ignore */ }
        }
        paintTransport(); paintRates(); paintNote(); paintLanes(); paintHead(true);
        emit('duration', duration());
        return meta;
      }).catch(function (e) {
        if (meetingId !== mine) return null;
        paintNote();
        if (noteBox) {
          noteBox.setAttribute('style', 'flex:none;font-size:11.5px;color:var(--color-accent-2-700);max-width:260px;line-height:1.4');
          noteBox.textContent = 'the player could not read this meeting';
          noteBox.title = String((e && e.message) || e);
        }
        return null;
      });
    }

    function startTicker() {
      if (ticker) return;
      ticker = setInterval(tick, 100);
    }
    function stopTicker() {
      if (!ticker) return;
      clearInterval(ticker);
      ticker = null;
    }
    function detach(host) {
      mounted = false;
      pause();
      stopTicker();
      if (host) host.style.display = 'none';
    }

    document.body.appendChild(audio);

    return {
      // the contract
      seek: seek, toggle: toggle, on: on, time: time, duration: duration,
      // the shell drives these; screens are free to use them
      play: play, pause: pause, rate: function (r) { return r === undefined ? rate : setRate(r); },
      playable: playable, meeting: function () { return meetingId; },
      // shell-only
      _attach: attach, _detach: detach, _mounted: function () { return mounted; }
    };
  }

  var player = makePlayer();
  MS.player = player;

  // ------------------------------------------------------------- the state
  var state = {};
  MS.state = state;

  // ------------------------------------------------------------ the router
  var NAV = [['library', 'Library'], ['ingest', 'Ingest'], ['voices', 'Voices'],
    ['review', 'Review'], ['settings', 'Settings']];
  var KNOWN = ['library', 'transcript', 'ingest', 'voices', 'review', 'search', 'settings'];

  function parseHash() {
    var raw = String(location.hash || '');
    if (raw.charAt(0) === '#') raw = raw.slice(1);
    var qi = raw.indexOf('?');
    var qs = qi >= 0 ? raw.slice(qi + 1) : '';
    if (qi >= 0) raw = raw.slice(0, qi);
    if (raw.charAt(0) === '/') raw = raw.slice(1);

    var parts = raw.split('/').filter(function (s) { return s !== ''; });
    var name = parts[0] ? decodeURIComponent(parts[0]) : '';
    var params = {};
    if (qs) {
      qs.split('&').forEach(function (kv) {
        if (!kv) return;
        var i = kv.indexOf('=');
        var k = decodeURIComponent((i < 0 ? kv : kv.slice(0, i)).replace(/\+/g, ' '));
        var v = i < 0 ? '' : decodeURIComponent(kv.slice(i + 1).replace(/\+/g, ' '));
        if (k) params[k] = v;
      });
    }
    if (parts[1]) params.id = decodeURIComponent(parts[1]);
    ['t', 'seek'].forEach(function (k) {
      if (params[k] !== undefined) {
        var v = toNum(params[k]);
        if (v === null) delete params[k]; else params[k] = v;
      }
    });
    if (params.play !== undefined) params.play = (params.play === '1' || params.play === 'true');
    return { name: name, params: params };
  }

  function buildHash(name, params) {
    var p = {};
    var src = params && typeof params === 'object' ? params : {};
    Object.keys(src).forEach(function (k) {
      var v = src[k];
      if (v === undefined || v === null || v === '' || v === false ||
          typeof v === 'object' || typeof v === 'function') return;
      p[k] = v;
    });
    var h = '#/' + encodeURIComponent(name);
    if (name === 'transcript') {
      var id = p.id || p.meeting;
      if (id) {
        h += '/' + encodeURIComponent(String(id));
        if (p.meeting === id) delete p.meeting;
        delete p.id;
      }
    }
    var qs = Object.keys(p).map(function (k) {
      return encodeURIComponent(k) + '=' + encodeURIComponent(String(p[k]));
    }).join('&');
    return qs ? h + '?' + qs : h;
  }

  function go(name, params) {
    var target = buildHash(String(name || 'library'), params);
    if (location.hash === target) {
      route();          // same URL, still a navigation the user asked for
      return;
    }
    location.hash = target;
  }

  // -------------------------------------------------------------- masthead
  var nav = null, tagline = null, counts = null, screenHost = null, scroller = null, playerHost = null;
  var badges = {};
  var libStats = null;

  function paintNav(active) {
    if (!nav) return;
    clear(nav);
    badges = {};
    NAV.forEach(function (pair) {
      var k = pair[0], label = pair[1];
      var on = (active === k) || (active === 'transcript' && k === 'library');
      var item = el('div', 'font-size:13.5px;cursor:pointer;display:flex;align-items:baseline;gap:5px;padding-bottom:1px;' +
        (on ? 'color:var(--color-accent-700);border-bottom:2px solid var(--color-accent);'
          : 'color:var(--color-text);border-bottom:2px solid transparent;'));
      item.appendChild(document.createTextNode(label));
      var badge = el('span', 'display:none');
      item.appendChild(badge);
      badges[k] = badge;
      item.setAttribute('role', 'button');
      item.setAttribute('tabindex', '0');
      if (on) item.setAttribute('aria-current', 'page');
      item.addEventListener('click', function () { go(k); });
      item.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(k); }
      });
      nav.appendChild(item);
    });

    var search = el('div', 'margin-left:auto;font-size:12.5px;cursor:pointer;color:' +
      (active === 'search' ? 'var(--color-accent-700)' : 'var(--color-neutral-700)'), 'Search  /');
    search.setAttribute('role', 'button');
    search.setAttribute('tabindex', '0');
    search.title = 'Search every transcript — press /';
    if (active === 'search') search.setAttribute('aria-current', 'page');
    search.addEventListener('click', function () { go('search'); });
    search.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go('search'); }
    });
    nav.appendChild(search);
    paintBadges();
  }

  var pendingCount = null;      // null = not asked yet / not answerable
  var pendingReason = '';
  function paintBadges() {
    var b = badges.review;
    if (!b) return;
    if (num(pendingCount) && pendingCount > 0) {
      b.setAttribute('style', 'font-size:10.5px;color:var(--color-accent-2-600);font-variant-numeric:tabular-nums');
      b.textContent = String(pendingCount);
      b.title = pendingCount + ' voice' + (pendingCount === 1 ? '' : 's') + ' waiting to be placed';
    } else {
      b.setAttribute('style', 'display:none');
      b.textContent = '';
      b.title = pendingReason || '';
    }
  }

  /* design line 23. The all-in-one sentence is the design's, verbatim. The
     split sentence is not in the design, because the design assumed one
     deployment; printing the design's line on a split install would be a false
     privacy claim, so the mode the server reports decides. */
  function paintTagline(stats) {
    if (!tagline) return;
    var mode = stats && stats.backend;
    if (mode === 'all-in-one') {
      tagline.textContent = 'Local transcription · nothing leaves this machine';
      tagline.title = 'The pipeline runs on this machine (ui/backend.py, LocalBackend).';
      return;
    }
    if (mode === 'split') {
      api('/api/settings').then(function (s) {
        var host = s && s.backend && s.backend.host;
        tagline.textContent = 'Split install · audio goes to ' + (host || 'the GPU box') +
          ' to transcribe · names stay here';
        tagline.title = 'In split mode the audio is rsynced to ' + (host || 'the GPU box') +
          ' over ssh for transcription and embedding. The library, the voice profiles and ' +
          'every naming, matching and merging decision stay on this machine.';
      }).catch(function () {
        tagline.textContent = 'Split install · audio goes to the GPU box to transcribe · names stay here';
      });
      return;
    }
    tagline.textContent = '';
    tagline.title = '';
  }

  function paintCounts(stats, err) {
    if (!counts) return;
    if (!stats) {
      counts.textContent = err ? 'library unavailable' : '';
      counts.title = err ? ('GET /api/library failed: ' + err) : '';
      counts.style.color = err ? 'var(--color-accent-2-700)' : 'var(--color-neutral-700)';
      return;
    }
    counts.style.color = 'var(--color-neutral-700)';
    var bits = [];
    if (num(stats.meetings)) bits.push(stats.meetings + ' meeting' + (stats.meetings === 1 ? '' : 's'));
    if (num(stats.hours)) bits.push(stats.hours.toFixed(1) + ' h on disk');
    if (num(stats.profiles)) bits.push(stats.profiles + ' voice' + (stats.profiles === 1 ? '' : 's') + ' on file');
    counts.textContent = bits.join(' · ');
    counts.title = 'Meetings and hours are the transcripts in the library; voices on file is ' +
      'the number of profiles in speakers.db, which is not the same as the number of ' +
      'unnamed speakers inside those meetings.';
  }

  var lastRefresh = 0;
  var lastRefreshOk = false;
  /* Re-read the header numbers on every navigation, but no more than once every
     three seconds — /api/library parses every transcript in the library, and a
     count in the masthead is not worth doing that twice a second. The throttle
     is skipped while the last read failed, so the header stops saying
     "unavailable" as soon as the server is back. */
  function refreshMasthead(force) {
    var now = Date.now();
    if (!force && lastRefreshOk && now - lastRefresh < 3000) return Promise.resolve();
    lastRefresh = now;
    return api('/api/library').then(function (lib) {
      libStats = (lib && lib.stats) || null;
      lastRefreshOk = true;
      paintCounts(libStats, null);
      paintTagline(libStats);
    }).catch(function (e) {
      libStats = null;
      lastRefreshOk = false;
      paintCounts(null, String((e && e.message) || e));
      paintTagline(null);
    }).then(function () {
      return api('/api/review').then(function (rv) {
        var c = (rv && rv.counts) || {};
        pendingCount = num(c.pending) ? c.pending : 0;
        pendingReason = (rv && typeof rv.reason === 'string' && rv.reason) ? rv.reason : '';
        paintBadges();
      }).catch(function (e) {
        // A badge is not worth an alert; the Review screen states the reason.
        pendingCount = null;
        pendingReason = 'the review queue could not be read: ' + String((e && e.message) || e);
        paintBadges();
      });
    });
  }

  // ---------------------------------------------------------- error screen
  function errorScreen(root, name, err, retry) {
    clear(root);
    var wrap = el('div', 'max-width:70ch');
    wrap.appendChild(el('h1', 'font-size:40px;margin:0 0 4px', 'This screen did not render'));
    wrap.appendChild(el('p', 'font-size:13.5px;color:var(--color-neutral-700);max-width:60ch;margin:0 0 18px',
      'The shell caught an error while rendering the ' + name + ' screen. Nothing was ' +
      'written to disk. What follows is the error itself, not a guess at it.'));

    var msg = String((err && err.message) || err || 'no message');
    var box = el('div', 'background:var(--color-surface);padding:14px 16px;border-radius:var(--radius-md);' +
      'font-size:12.5px;line-height:1.6;white-space:pre-wrap;word-break:break-word;margin-bottom:6px');
    box.textContent = msg;
    wrap.appendChild(box);

    var where = [];
    if (err && err.url) where.push(String(err.url));
    if (err && num(err.status) && err.status) where.push('HTTP ' + err.status);
    where.push('/static/screens/' + name + '.js');
    wrap.appendChild(el('div', 'font-size:11px;letter-spacing:0.1em;text-transform:uppercase;' +
      'color:var(--color-neutral-600);margin-bottom:18px', where.join(' · ')));

    if (err && err.stack) {
      var det = document.createElement('details');
      det.setAttribute('style', 'margin-bottom:20px');
      var sum = document.createElement('summary');
      sum.setAttribute('style', 'font-size:12.5px;color:var(--color-accent-700);cursor:pointer');
      sum.textContent = 'Stack';
      det.appendChild(sum);
      var pre = el('pre', 'font-size:11.5px;line-height:1.5;white-space:pre-wrap;word-break:break-word;' +
        'color:var(--color-neutral-700);margin:8px 0 0');
      pre.textContent = String(err.stack);
      det.appendChild(pre);
      wrap.appendChild(det);
    }

    var row = el('div', 'display:flex;gap:10px');
    if (retry) {
      var again = el('button', 'margin:0', 'Try again');
      again.className = 'btn btn-primary';
      again.addEventListener('click', retry);
      row.appendChild(again);
    }
    var home = el('button', 'margin:0', 'Go to Library');
    home.className = 'btn btn-secondary';
    home.addEventListener('click', function () { go('library'); });
    row.appendChild(home);
    wrap.appendChild(row);
    root.appendChild(wrap);
  }

  function missingScreen(root, name) {
    clear(root);
    var wrap = el('div', 'max-width:70ch');
    wrap.appendChild(el('h1', 'font-size:40px;margin:0 0 4px', 'No screen called “' + name + '”'));
    wrap.appendChild(el('p', 'font-size:13.5px;color:var(--color-neutral-700);margin:0 0 18px',
      'The shell routes ' + KNOWN.map(function (k) { return '#/' + k; }).join(', ') +
      '. A screen registers itself as window.MS.screens.' + name +
      ' from /static/screens/' + name + '.js; if that file exists, it failed to load — ' +
      'the browser console will say why.'));
    var home = el('button', 'margin:0', 'Go to Library');
    home.className = 'btn btn-primary';
    home.addEventListener('click', function () { go('library'); });
    wrap.appendChild(home);
    root.appendChild(wrap);
  }

  // --------------------------------------------------------------- mounting
  var epoch = 0;
  var mounted = null;      // {name, screen}

  function unmount() {
    closeAllDialogs();
    if (mounted && mounted.screen && typeof mounted.screen.destroy === 'function') {
      try { mounted.screen.destroy(); } catch (e) {
        notify('The ' + mounted.name + ' screen threw while closing.', String((e && e.message) || e));
      }
    }
    mounted = null;
  }

  function route() {
    var r = parseHash();
    if (!r.name) {
      // a bare "/" — land on the library, and write it into the URL so back
      // and forward have something to return to. replaceState rather than
      // assigning location.hash, which would fire a second navigation.
      r.name = 'library';
      try { history.replaceState(null, '', location.pathname + location.search + '#/library'); }
      catch (e) { /* the old hash is harmless */ }
    }
    mount(r.name, r.params);
  }

  function mount(name, params) {
    var token = ++epoch;
    unmount();

    clear(screenHost);
    var root = el('div');
    screenHost.appendChild(root);
    if (scroller) scroller.scrollTop = 0;

    paintNav(name);
    var screen = MS.screens[name];
    document.title = 'Meetscribe' + (screen && screen.title ? ' — ' + screen.title : '');

    if (!screen || typeof screen.render !== 'function') {
      player._detach(playerHost);
      missingScreen(root, name);
      return;
    }

    var ctx = {
      api: api,
      go: go,
      state: state,
      ink: ink,
      fmt: fmt,
      dialog: dialog,
      params: params || {}
    };

    mounted = { name: name, screen: screen };

    var prep;
    if (name === 'transcript') {
      // the contract gives the player to the transcript screen and to nothing else
      ctx.player = player;
      prep = prepareTranscript(ctx, token);
    } else {
      player._detach(playerHost);
      prep = Promise.resolve();
    }

    prep.then(function () {
      if (token !== epoch) return;
      return Promise.resolve(screen.render(root, ctx));
    }).then(function () {
      if (token !== epoch) return;
      refreshMasthead();
    }).catch(function (e) {
      if (token !== epoch) return;
      errorScreen(root, name, e, function () { mount(name, params); });
    });
  }

  /* The transcript screen is the only one with a player. Work out which
     meeting it is about (the same way the screen itself does, so the two never
     disagree), point the bar at it, and honour a timestamp handed over by
     Search or Review before the screen paints. */
  function prepareTranscript(ctx, token) {
    var p = ctx.params || {};
    var namedInUrl = !!((typeof p.id === 'string' && p.id) || (typeof p.meeting === 'string' && p.meeting));
    var id = (typeof p.id === 'string' && p.id) ? p.id
      : (typeof p.meeting === 'string' && p.meeting) ? p.meeting
        : (typeof state.meeting === 'string' && state.meeting) ? state.meeting
          : (typeof state.meetingId === 'string' && state.meetingId) ? state.meetingId
            : null;

    var found = id ? Promise.resolve(id) : api('/api/library').then(function (lib) {
      var list = (lib && lib.meetings) || [];
      return list.length ? list[0].id : null;      // newest first, as the server sorts it
    }).catch(function () { return null; });

    return found.then(function (mid) {
      if (token !== epoch) return;
      if (!mid) {
        // Nothing to play and nothing to show — let the screen say so.
        player._detach(playerHost);
        return;
      }
      state.meeting = mid;
      state.meetingId = mid;

      // A bare #/transcript resolves to whichever meeting is actually shown —
      // the one handed over in ctx.state, or the newest in the library. Write
      // it into the URL so a reload or a bookmark lands in the same place.
      // replaceState, not location.hash: this is a correction of the current
      // history entry, not a navigation, and it must keep any ?t= that came
      // with it or back/forward would lose the timestamp.
      if (!namedInUrl) {
        var q = {};
        Object.keys(p).forEach(function (k) { q[k] = p[k]; });
        q.id = mid;
        var want = buildHash('transcript', q);
        if (location.hash !== want) {
          try { history.replaceState(null, '', location.pathname + location.search + want); }
          catch (e) { /* keep the old hash rather than fire a navigation */ }
        }
      }
      ctx.params.id = mid;

      var t = toNum(p.t);
      if (t === null) t = toNum(p.seek);
      if (t === null) t = toNum(state.t);
      if (t === null) t = toNum(state.seek);
      delete state.t;
      delete state.seek;

      return player._attach(mid, playerHost).then(function () {
        if (token !== epoch) return;
        if (t !== null) player.seek(t);
        if (p.play) player.play();
      });
    });
  }

  // -------------------------------------------------------------- keyboard
  function typing(t) {
    if (!t) return false;
    var tag = (t.tagName || '').toLowerCase();
    return tag === 'input' || tag === 'textarea' || tag === 'select' || t.isContentEditable === true;
  }

  function onKey(e) {
    if (e.defaultPrevented || e.metaKey || e.ctrlKey || e.altKey) return;
    if (typing(e.target)) return;
    if (openDialogs.length) return;
    var here = mounted && mounted.name;

    if (e.key === '/') {
      e.preventDefault();
      if (here === 'search') {
        var box = screenHost && screenHost.querySelector('input');
        if (box) { try { box.focus(); box.select(); } catch (_) { /* not selectable */ } return; }
      }
      go('search');
      return;
    }
    if (here !== 'transcript') return;
    if (e.key === ' ' || e.key === 'Spacebar') {
      // Only swallow the spacebar when there is something to play; otherwise it
      // has to keep scrolling the transcript.
      if (!player.playable()) return;
      e.preventDefault();
      player.toggle();
      return;
    }
    if (e.key === 'ArrowRight') { e.preventDefault(); player.seek(player.time() + 5); }
    if (e.key === 'ArrowLeft') { e.preventDefault(); player.seek(player.time() - 5); }
  }

  // ------------------------------------------------------------------ boot
  function boot() {
    alerts = document.getElementById('ms-alerts');
    nav = document.getElementById('ms-nav');
    tagline = document.getElementById('ms-tagline');
    counts = document.getElementById('ms-counts');
    screenHost = document.getElementById('ms-screen');
    scroller = document.getElementById('ms-scroll');
    playerHost = document.getElementById('ms-player');

    if (!screenHost) {
      document.body.appendChild(el('div', 'padding:40px;font-size:15px',
        'The shell could not find #ms-screen in index.html, so no screen can be mounted.'));
      return;
    }
    clear(screenHost);   // drops the "did not start" fallback

    var missing = KNOWN.filter(function (k) { return !MS.screens[k]; });
    if (missing.length) {
      notify('Screen file' + (missing.length === 1 ? '' : 's') + ' did not load: ' + missing.join(', ') + '.',
        'Expected /static/screens/<name>.js to set window.MS.screens.<name>. ' +
        'Those tabs will show the reason instead of a blank page.');
    }

    window.addEventListener('error', function (e) {
      var t = e && e.target;
      if (t && t !== window && (t.src || t.href)) {
        notify('A file the page needs did not load.', String(t.src || t.href));
        return;
      }
      if (!e || (!e.message && !e.error)) return;   // not something we can name
      notify('Something threw outside a screen.',
        String(e.message || (e.error && e.error.message) || 'error') +
        (e.filename ? ' — ' + e.filename + ':' + (e.lineno || 0) : ''));
    }, true);
    window.addEventListener('unhandledrejection', function (e) {
      var r = e && e.reason;
      notify('A background request failed.', String((r && r.message) || r || 'unknown reason'));
    });
    window.addEventListener('hashchange', route);
    window.addEventListener('keydown', onKey);

    refreshMasthead();
    route();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
