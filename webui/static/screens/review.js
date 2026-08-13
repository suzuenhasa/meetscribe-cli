/* Meetscribe — Review.
 *
 * Design: design/Meetscribe.dc.html lines 187-233, plus the name dialog at
 * 378-392. The inline style strings are carried over from that file.
 *
 * Data: GET /api/review, POST /api/review/resolve, and GET /api/library for
 * one fact the review payload does not carry (whether a meeting has audio, so
 * the copy can say "play" only when there is something to play).
 */
(function () {
  'use strict';

  var MS = (window.MS = window.MS || {});
  MS.screens = MS.screens || {};

  var AXIS = { min: 0.20, max: 1.00 };   // only a fallback; the server sends its own
  var session = null;                    // { alive, close } for destroy()

  // ------------------------------------------------------------------ atoms
  function el(tag, style, text) {
    var n = document.createElement(tag);
    if (style) n.style.cssText = style;
    if (text !== null && text !== undefined) n.textContent = String(text);
    return n;
  }

  function isNum(x) {
    return typeof x === 'number' && isFinite(x);
  }

  function sc(x) {                       // a similarity, always two places
    return isNum(x) ? x.toFixed(2) : '—';
  }

  function secs(x) {
    return isNum(x) ? String(Math.round(x)) : null;
  }

  function clockOf(ctx, t) {
    if (!isNum(t)) return '—';
    try {
      if (ctx && ctx.fmt && typeof ctx.fmt.clock === 'function') return ctx.fmt.clock(t);
      if (ctx && ctx.fmt && typeof ctx.fmt.hms === 'function') return ctx.fmt.hms(t);
    } catch (_) { /* fall through to the local formatter */ }
    var m = Math.floor(t / 60), s = Math.floor(t % 60);
    return m + ':' + (s < 10 ? '0' + s : String(s));
  }

  function axisOf(d) {
    var a = (d && d.axis) || {};
    if (isNum(a.min) && isNum(a.max) && a.max > a.min) return { min: a.min, max: a.max };
    return AXIS;
  }

  function pos(x, axis) {                // percent along the similarity scale
    var p = ((x - axis.min) / (axis.max - axis.min)) * 100;
    if (!isFinite(p)) return 0;
    return Math.max(0, Math.min(100, p));
  }

  // The server throws on non-2xx; the body carries the reason we need to act on.
  function apiErr(e) {
    var out = { body: null, message: '' };
    if (!e) { out.message = 'the server did not answer'; return out; }
    var maybe = [e.body, e.data, e.payload, e.json, e.response];
    for (var i = 0; i < maybe.length; i++) {
      if (maybe[i] && typeof maybe[i] === 'object') { out.body = maybe[i]; break; }
    }
    if (!out.body && typeof e.message === 'string') {
      var b = e.message.indexOf('{');
      if (b >= 0) { try { out.body = JSON.parse(e.message.slice(b)); } catch (_) { /* not JSON */ } }
    }
    out.message = (out.body && (out.body.error || out.body.detail)) ||
                  e.message || String(e);
    return out;
  }

  function post(ctx, path, body) {
    return ctx.api(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
  }

  // ------------------------------------------------------------------- copy
  // The denominator is what this machine actually scored, which is not the whole
  // library when some meetings have no embeddings here — so say which it is.
  function introLine(n, counts, allScored) {
    if (!n) return 'Nothing is waiting.';
    var inLib = null;
    if (isNum(counts.clusters) && isNum(counts.elsewhere)) inLib = counts.clusters - counts.elsewhere;
    var verb = n === 1 ? 'needs' : 'need';
    var tail = ' a person to decide. Everything else was matched or left unknown on its own.';
    if (inLib !== null && inLib >= n) {
      return n + ' of ' + inLib + ' cluster' + (inLib === 1 ? '' : 's') +
             (allScored ? ' in the library ' : ' scored on this machine ') + verb + tail;
    }
    return n + ' cluster' + (n === 1 ? '' : 's') + ' ' + verb + tail;
  }

  function thresholdSentence(t) {
    if (!isNum(t.accept) || !isNum(t.review) || !isNum(t.margin)) {
      return 'The server did not report the thresholds it matches at.';
    }
    var a = sc(t.accept), r = sc(t.review), m = sc(t.margin);
    return 'Accept at ' + a + ' with a margin of ' + m + ' over the runner-up. Between ' +
           r + ' and ' + a + ' a person decides. Below ' + r + ' the voice is treated as new.';
  }

  function outcomeWord(band) {
    if (band === 'unknown') return 'nobody close';
    if (band === 'margin') return 'margin too thin';
    if (band === 'review') return 'below the accept line';
    return band || 'open';
  }

  function meetingLine(c) {
    var s = c.meeting_title || c.meeting_id || c.meeting || '';
    if (c.source === 'log') {
      s += ' · from the decision log';
      if (c.decided_at_str) s += ', ' + c.decided_at_str;
    }
    return s;
  }

  function secondsLine(c) {
    var v = secs(c.seconds);
    if (v) return v + ' s of speech';
    if (c.seconds_reason === 'transcript-not-in-library') {
      return 'length unknown — the transcript is not in the library';
    }
    return 'length unknown';
  }

  function sampleNote(c) {
    if (c.samples_reason === 'no-lines-long-enough') {
      return 'No line from this voice is long enough to quote.';
    }
    if (c.samples_reason === 'transcript-not-in-library') {
      return 'The transcript is not in the library, so there is nothing here to read.';
    }
    return 'No sample lines came back for this cluster.';
  }

  function reasonText(c, t, hasAudio) {
    var best = c.best || {}, second = c.second || {};
    var min = isNum(t.min_enroll_sec) ? t.min_enroll_sec : null;
    var v = secs(c.seconds);

    if (c.band === 'unknown') {
      var head = 'Nothing on file comes close.';
      if (v && min !== null) {
        head += c.enough_to_enroll
          ? ' ' + v + ' s of speech is ' + (c.seconds >= min * 2 ? 'well past' : 'past') +
            ' the ' + sc(min).replace(/\.00$/, '') + ' s floor, so this voice is worth remembering.'
          : ' ' + v + ' s of speech is under the ' + sc(min).replace(/\.00$/, '') +
            ' s floor, so naming it stores nothing unless you insist.';
      } else if (v) {
        head += ' ' + v + ' s of speech.';
      }
      return head;
    }

    if (c.band === 'margin' && isNum(best.score) && isNum(second.score)) {
      return sc(best.score) + ' clears the accept line but beats ' +
             (second.name || 'the runner-up') + ' by only ' + sc(best.score - second.score) +
             '. One person cannot be two clusters in one meeting, so the tie is yours to break.';
    }

    var line = best.name
      ? 'Best match is ' + best.name + (isNum(best.score) ? ' at ' + sc(best.score) : '') +
        ' — above the review floor, below the accept line.'
      : 'The best match scores ' + sc(best.score) + ' — above the review floor, below the ' +
        'accept line — but the decision log did not record who it was.';
    if (c.samples && c.samples.length) {
      line += hasAudio ? ' Play a sample before you decide.' : ' Read a sample before you decide.';
    }
    return line;
  }

  function nameBody(c, t) {
    var best = c.best || {}, second = c.second || {};
    var min = isNum(t.min_enroll_sec) ? sc(t.min_enroll_sec).replace(/\.00$/, '') : null;
    var v = secs(c.seconds);
    var out;
    if (v && c.enough_to_enroll) out = v + ' s of speech, enough to remember.';
    else if (v && min) out = v + ' s of speech — under the ' + min + ' s floor.';
    else if (v) out = v + ' s of speech.';
    else out = 'The transcript does not say how much this voice speaks.';

    if (best.name && isNum(best.score)) {
      if (c.band === 'margin' && second.name && isNum(second.score)) {
        out += ' The nearest voice on file is ' + best.name + ' at ' + sc(best.score) + ', with ' +
               second.name + ' at ' + sc(second.score) + ' — too close to call.';
      } else if (c.band === 'unknown') {
        out += ' The nearest voice on file is ' + best.name + ' at ' + sc(best.score) +
               ', below the review floor.';
      } else {
        out += ' The nearest voice on file is ' + best.name + ' at ' + sc(best.score) +
               ' — close, but under the accept line.';
      }
    } else if (isNum(best.score)) {
      out += ' The decision log kept the nearest score, ' + sc(best.score) +
             ', but not who it belonged to.';
    }
    return out;
  }

  // ------------------------------------------------------------- navigation
  function openAt(ctx, meetingId, t, play) {
    if (!meetingId) return;
    try {
      if (ctx.state) {
        ctx.state.meeting = meetingId;
        ctx.state.meetingId = meetingId;
        if (isNum(t)) { ctx.state.seek = t; ctx.state.t = t; }
      }
      ctx.go('transcript', { id: meetingId, meeting: meetingId, t: t, seek: t, play: !!play });
    } catch (_) { /* the shell owns navigation; a failure here must not kill the screen */ }
  }

  // ------------------------------------------------------------------ axis
  function marker(k, i, axis) {
    var p = pos(k.score, axis);
    var under = k.score < axis.min;
    var shift = p < 8 ? 'translateX(0)' : (p > 92 ? 'translateX(-100%)' : 'translateX(-50%)');
    var n = el('div',
      'position:absolute;left:' + p.toFixed(1) + '%;top:' + (i ? 26 : 0) + 'px;transform:' + shift +
      ';white-space:nowrap;font-size:11px;font-variant-numeric:tabular-nums;display:flex;' +
      'align-items:center;gap:5px;color:' + (i ? 'var(--color-neutral-600)' : 'var(--color-text)'));
    n.appendChild(el('span', 'width:7px;height:7px;background:' +
      (i ? 'var(--color-neutral-500)' : 'var(--color-accent-600)') +
      ';display:inline-block;flex:none'));
    n.appendChild(document.createTextNode(
      (under ? '‹ ' : '') + (k.name || 'not recorded') + ' ' + sc(k.score)));
    if (under) {
      n.title = sc(k.score) + ' is below the ' + sc(axis.min) + ' floor of this scale, so it sits at the end';
    }
    return n;
  }

  function axisBlock(c, t, axis) {
    var box = el('div', 'margin:26px 0 8px;position:relative;height:34px');
    box.appendChild(el('div', 'position:absolute;left:0;right:0;top:22px;height:1px;' +
      'background:color-mix(in srgb, var(--color-text) 20%, transparent)'));

    if (isNum(t.review)) {
      var rt = el('div', 'position:absolute;left:' + pos(t.review, axis).toFixed(1) +
        '%;top:14px;bottom:0;width:1px;background:var(--color-neutral-400)');
      rt.title = 'review floor ' + sc(t.review);
      box.appendChild(rt);
    }
    if (isNum(t.accept)) {
      var at = el('div', 'position:absolute;left:' + pos(t.accept, axis).toFixed(1) +
        '%;top:8px;bottom:-4px;width:1px;background:var(--color-accent-2-600)');
      at.title = 'accept line ' + sc(t.accept);
      box.appendChild(at);
    }

    var pair = [c.best, c.second];
    for (var i = 0; i < pair.length; i++) {
      if (pair[i] && isNum(pair[i].score)) box.appendChild(marker(pair[i], i, axis));
    }
    return box;
  }

  // ------------------------------------------------------------ name dialog
  function askName(ctx, c, t, done) {
    var dlg = el('div');
    dlg.className = 'dialog';

    var title = el('div', null, 'Name this voice');
    title.className = 'dialog-title';
    dlg.appendChild(title);

    var body = el('div', null, nameBody(c, t));
    body.className = 'dialog-body';
    dlg.appendChild(body);

    var field = el('div');
    field.className = 'field';
    field.appendChild(el('label', null, 'Name'));
    var input = document.createElement('input');
    input.className = 'input';
    input.setAttribute('placeholder', 'Who is this?');
    input.value = '';
    field.appendChild(input);
    dlg.appendChild(field);

    var note = el('div', 'font-size:11.5px;color:var(--color-neutral-700);line-height:1.5',
      c.can_enroll
        ? 'Named once. Every meeting afterwards recognises them without being asked again.'
        : 'This meeting’s embeddings are not on this machine, so there is no voiceprint to ' +
          'store: the name applies to this transcript only and will not be recognised in a ' +
          'later meeting.');
    if (!c.can_enroll) note.style.color = 'var(--color-accent-2-700)';
    dlg.appendChild(note);

    var actions = el('div');
    actions.className = 'dialog-actions';
    var cancel = el('button', null, 'Cancel');
    cancel.className = 'btn btn-secondary';
    var commit = el('button', null, 'Remember this voice');
    commit.className = 'btn btn-primary';
    actions.appendChild(cancel);
    actions.appendChild(commit);
    dlg.appendChild(actions);

    var close = ctx.dialog(dlg);
    if (session) session.close = close;
    var force = false, busy = false;

    function shut() {
      if (session && session.close === close) session.close = null;
      if (typeof close === 'function') close();
    }

    cancel.addEventListener('click', function () { shut(); });

    function submit() {
      var name = (input.value || '').trim();
      if (!name || busy) { if (!name && input.focus) input.focus(); return; }
      busy = true;
      commit.disabled = true;
      cancel.disabled = true;
      var payload = { meeting: c.meeting, cluster: c.cluster, action: 'name', name: name };
      if (force) payload.force = true;
      post(ctx, '/api/review/resolve', payload).then(function (r) {
        shut();
        done(null, r, name);
      }, function (e) {
        busy = false;
        commit.disabled = false;
        cancel.disabled = false;
        var info = apiErr(e);
        var b = info.body || {};
        if (b.reason === 'below-enrolment-floor') {
          force = true;
          note.style.color = 'var(--color-accent-2-700)';
          note.textContent = (isNum(b.seconds) ? Math.round(b.seconds) + ' s' : 'This cluster') +
            ' is under the ' +
            (isNum(b.min_enroll_sec) ? sc(b.min_enroll_sec).replace(/\.00$/, '') + ' s ' : '') +
            'enrolment floor, the measured knee for clean speech. Below it a stored ' +
            'voiceprint is unreliable. Store it anyway?';
          commit.textContent = 'Store it anyway';
        } else {
          note.style.color = 'var(--color-accent-2-700)';
          note.textContent = info.message;
        }
      });
    }

    commit.addEventListener('click', submit);
    input.addEventListener('keydown', function (e) {
      if (e.key === 'Enter') { e.preventDefault(); submit(); }
    });
    if (typeof input.focus === 'function') {
      setTimeout(function () { try { input.focus(); } catch (_) { /* detached */ } }, 0);
    }
  }

  // ------------------------------------------------------------------- card
  function card(c, d, ctx, hasAudio, act) {
    var t = d.thresholds || {};
    var axis = axisOf(d);
    var box = el('div', 'margin-bottom:46px;max-width:760px');

    var head = el('div', 'display:flex;align-items:baseline;gap:14px;margin-bottom:3px');
    head.appendChild(el('div', 'font-size:25px;font-weight:600', c.cluster || '—'));
    head.appendChild(el('div',
      'font-size:12.5px;color:var(--color-neutral-700);font-variant-numeric:tabular-nums',
      secondsLine(c)));
    var tagWrap = el('div', 'margin-left:auto');
    var tag = el('span', null, outcomeWord(c.band));
    tag.className = 'tag ' + (c.band === 'unknown' ? 'tag-neutral' : 'tag-accent-2');
    tagWrap.appendChild(tag);
    head.appendChild(tagWrap);
    box.appendChild(head);

    box.appendChild(el('div', 'font-size:12.5px;color:var(--color-neutral-700);margin-bottom:18px',
      meetingLine(c)));

    var grid = el('div',
      'display:grid;grid-template-columns:minmax(0,1fr) 250px;gap:40px;align-items:start');
    var left = el('div');
    var right = el('div');
    grid.appendChild(left);
    grid.appendChild(right);
    box.appendChild(grid);

    // --- samples
    var samples = Array.isArray(c.samples) ? c.samples : [];
    for (var i = 0; i < samples.length; i++) {
      (function (s) {
        var row = el('div',
          'display:flex;gap:12px;font-size:14.5px;line-height:1.55;margin-bottom:10px;cursor:pointer');
        row.appendChild(el('span',
          'font-size:11px;color:var(--color-accent-700);font-variant-numeric:tabular-nums;' +
          'padding-top:3px;flex:none', clockOf(ctx, s.t)));
        row.appendChild(el('span', 'font-style:italic;color:var(--color-neutral-800)',
          '“' + (s.text || '') + '”'));
        if (c.meeting_id && isNum(s.t)) {
          row.title = (hasAudio ? 'Play from ' : 'Open the transcript at ') + clockOf(ctx, s.t);
          row.addEventListener('click', function () {
            openAt(ctx, c.meeting_id, s.t, hasAudio);
          });
        } else {
          row.style.cursor = 'default';
        }
        left.appendChild(row);
      })(samples[i]);
    }
    if (!samples.length) {
      left.appendChild(el('div',
        'font-size:13.5px;color:var(--color-neutral-700);font-style:italic;margin-bottom:10px',
        sampleNote(c)));
    }

    // --- the similarity axis
    left.appendChild(axisBlock(c, t, axis));
    var ends = el('div',
      'display:flex;justify-content:space-between;font-size:10px;letter-spacing:0.1em;' +
      'text-transform:uppercase;color:var(--color-neutral-600);font-variant-numeric:tabular-nums');
    ends.appendChild(el('span', null, sc(axis.min) + ' similarity'));
    ends.appendChild(el('span', null, sc(axis.max)));
    left.appendChild(ends);

    if (c.candidates_named === false && c.candidates_reason) {
      left.appendChild(el('div',
        'margin-top:12px;font-size:11px;line-height:1.5;color:var(--color-neutral-600);max-width:52ch',
        c.candidates_reason));
    }

    // --- the decision
    right.appendChild(el('div',
      'font-size:12.5px;line-height:1.5;color:var(--color-neutral-800);margin-bottom:18px',
      reasonText(c, t, hasAudio)));

    var acts = el('div', 'display:flex;flex-direction:column;gap:8px');
    right.appendChild(acts);

    var best = c.best || {};
    var canAccept = best.id !== null && best.id !== undefined && !!best.name;
    var namesInstead = c.band === 'unknown' || !canAccept;

    var primary = el('button', 'margin:0', namesInstead ? 'Name this voice' : 'Accept as ' + best.name);
    primary.className = 'btn btn-primary btn-block';
    var newOne = el('button', 'margin:0', 'Someone new — name them');
    newOne.className = 'btn btn-secondary btn-block';
    var leave = el('button', 'margin:0', 'Leave unknown');
    leave.className = 'btn btn-ghost btn-block';
    acts.appendChild(primary);
    acts.appendChild(newOne);
    acts.appendChild(leave);

    var err = el('div',
      'display:none;margin-top:10px;font-size:11.5px;line-height:1.5;color:var(--color-accent-2-700)');
    right.appendChild(err);

    if (!c.can_enroll) {
      right.appendChild(el('div',
        'margin-top:14px;font-size:11.5px;line-height:1.5;color:var(--color-accent-2-700)',
        'No embeddings for this meeting on this machine, so a name here is recorded for this ' +
        'transcript only — nothing is stored that could recognise this voice later.'));
    }

    function busy(on) {
      primary.disabled = on;
      newOne.disabled = on;
      leave.disabled = on;
    }
    function fail(e) {
      busy(false);
      var info = apiErr(e);
      err.style.display = 'block';
      err.textContent = info.message;
    }
    function send(body) {
      busy(true);
      err.style.display = 'none';
      post(ctx, '/api/review/resolve', body).then(function (r) { act.done(null, r, c); }, fail);
    }

    primary.addEventListener('click', function () {
      if (namesInstead) {
        askName(ctx, c, t, function (e2, r, nm) { act.done(e2, r, c, nm); });
      } else {
        send({ meeting: c.meeting, cluster: c.cluster, action: 'accept' });
      }
    });
    newOne.addEventListener('click', function () {
      askName(ctx, c, t, function (e2, r, nm) { act.done(e2, r, c, nm); });
    });
    leave.addEventListener('click', function () {
      send({ meeting: c.meeting, cluster: c.cluster, action: 'leave' });
    });

    return box;
  }

  // ------------------------------------------------------- honest empty ends
  function emptyState(d) {
    var wrap = el('div', 'padding:40px 0;font-size:17px;color:var(--color-neutral-700);font-style:italic');
    var r = d.reason;
    if (r === 'all-resolved' || !r) {
      wrap.textContent = 'Nothing left to place. Every cluster in the library is matched to a ' +
                         'person or deliberately unknown.';
    } else if (r === 'empty-library') {
      wrap.textContent = 'No transcripts in the library yet, so there is nothing to place.';
    } else if (r === 'no-profiles') {
      wrap.textContent = 'No voice is enrolled yet, so there is nothing to match a cluster against.';
    } else if (r === 'not-identified') {
      wrap.textContent = 'Nothing can be reviewed yet: no meeting in the library has been ' +
                         'identified on this machine.';
    } else if (r === 'only-outside-library') {
      wrap.textContent = 'Nothing in this library is waiting. The clusters still open belong to ' +
                         'meetings that are not here.';
    } else {
      wrap.textContent = 'Nothing is waiting (' + r + ').';
    }
    return wrap;
  }

  function sectionHead(text) {
    return el('div', 'font-size:10.5px;letter-spacing:0.12em;text-transform:uppercase;' +
      'color:var(--color-neutral-600);margin-bottom:10px', text);
  }

  function unidentifiedBlock(list) {
    var box = el('div', 'margin-top:10px;margin-bottom:44px;max-width:760px');
    box.appendChild(sectionHead('Not identified on this machine'));
    for (var i = 0; i < list.length; i++) {
      var u = list[i] || {};
      var row = el('div', 'margin-bottom:14px');
      var top = el('div', 'display:flex;align-items:baseline;gap:12px;font-size:14.5px');
      top.appendChild(el('span', null, u.title || u.meeting || '—'));
      top.appendChild(el('span',
        'font-size:12.5px;color:var(--color-neutral-700);font-variant-numeric:tabular-nums',
        isNum(u.clusters) ? u.clusters + (u.clusters === 1 ? ' cluster' : ' clusters') : ''));
      row.appendChild(top);
      row.appendChild(el('div',
        'font-size:12.5px;line-height:1.5;color:var(--color-neutral-700);max-width:70ch;margin-top:3px',
        u.detail || u.reason || ''));
      box.appendChild(row);
    }
    return box;
  }

  function elsewhereBlock(list, axis) {
    var box = el('div', 'margin-bottom:44px;max-width:760px');
    box.appendChild(sectionHead('Open, but not in this library'));
    box.appendChild(el('div',
      'font-size:12.5px;line-height:1.5;color:var(--color-neutral-700);max-width:70ch;margin-bottom:14px',
      list.length + ' cluster' + (list.length === 1 ? '' : 's') + ' in the profile store ' +
      (list.length === 1 ? 'is' : 'are') + ' still open, but ' +
      (list.length === 1 ? 'its meeting is' : 'their meetings are') +
      ' not in this library. Nothing here can be read, played or enrolled, so there is no ' +
      'decision to take from this screen.'));

    var anyUnnamed = false;
    for (var i = 0; i < list.length; i++) {
      var c = list[i] || {};
      var best = c.best || {}, second = c.second || {};
      if (c.candidates_named === false) anyUnnamed = true;
      var row = el('div', 'display:flex;align-items:baseline;gap:12px;font-size:12.5px;' +
        'padding:6px 0;border-bottom:1px solid color-mix(in srgb, var(--color-text) 8%, transparent)');
      row.appendChild(el('span', 'flex:none;color:var(--color-neutral-800)',
        (c.meeting_title || c.meeting || '—') + ' · ' + (c.cluster || '—')));
      row.appendChild(el('span', 'font-size:11px;color:var(--color-neutral-600)',
        outcomeWord(c.band)));
      var right = el('div',
        'margin-left:auto;font-variant-numeric:tabular-nums;color:var(--color-neutral-700);' +
        'font-size:11.5px;white-space:nowrap');
      var text = (best.name ? best.name + ' ' : '') + sc(best.score);
      if (isNum(second.score)) text += '  ·  ' + (second.name ? second.name + ' ' : '') + sc(second.score);
      right.textContent = text;
      row.appendChild(right);
      box.appendChild(row);
    }
    if (anyUnnamed) {
      box.appendChild(el('div',
        'margin-top:12px;font-size:11px;line-height:1.5;color:var(--color-neutral-600);max-width:70ch',
        'Both scores survive in the decision log, but it only records who a match was when it ' +
        'was accepted, so these candidates cannot be named from it.'));
    }
    return box;
  }

  // ------------------------------------------------------------------ paint
  function paint(wrap, d, ctx, hasAudio, act, notice) {
    while (wrap.firstChild) wrap.removeChild(wrap.firstChild);

    var t = d.thresholds || {};
    var counts = d.counts || {};
    var pending = Array.isArray(d.pending) ? d.pending : [];
    var elsewhere = Array.isArray(d.elsewhere) ? d.elsewhere : [];
    var unident = Array.isArray(d.unidentified) ? d.unidentified : [];

    wrap.appendChild(el('h1', 'font-size:40px;margin:0 0 4px', 'Review'));
    wrap.appendChild(el('p',
      'font-size:13.5px;color:var(--color-neutral-700);max-width:60ch;margin:0 0 8px',
      introLine(pending.length, counts, unident.length === 0)));

    var th = el('div',
      'font-size:11.5px;color:var(--color-neutral-700);margin-bottom:34px;' +
      'font-variant-numeric:tabular-nums', thresholdSentence(t));
    if (d.thresholds_source) th.title = 'Thresholds read from ' + d.thresholds_source;
    wrap.appendChild(th);

    if (notice && notice.text) {
      var n = el('div', 'margin:-22px 0 30px;font-size:12.5px;line-height:1.5;max-width:70ch;color:' +
        (notice.warn ? 'var(--color-accent-2-700)' : 'var(--color-accent-700)'), notice.text);
      wrap.appendChild(n);
    }

    for (var i = 0; i < pending.length; i++) {
      wrap.appendChild(card(pending[i], d, ctx, hasAudio[pending[i].meeting_id] === true, act));
    }

    if (!pending.length) wrap.appendChild(emptyState(d));
    if (unident.length) wrap.appendChild(unidentifiedBlock(unident));
    if (elsewhere.length) wrap.appendChild(elsewhereBlock(elsewhere, axisOf(d)));
  }

  function failWhole(wrap, e) {
    while (wrap.firstChild) wrap.removeChild(wrap.firstChild);
    wrap.appendChild(el('h1', 'font-size:40px;margin:0 0 4px', 'Review'));
    wrap.appendChild(el('p',
      'font-size:13.5px;color:var(--color-accent-2-700);max-width:60ch;margin:0 0 8px',
      'The server did not answer with the clusters waiting: ' + apiErr(e).message));
  }

  // What happened, said plainly — enrolment included, because "named" and
  // "remembered" are not the same thing when the embeddings are missing.
  function noticeFor(r, c, nm) {
    if (!r) return null;
    var who = (r.speaker && r.speaker.name) || nm || '';
    if (r.outcome === 'left-unknown') {
      return { text: c.cluster + ' left unknown. It stays out of the review list until it is ' +
                     'identified again.' };
    }
    if (r.enrolled === false && r.enroll_reason) {
      return {
        warn: true,
        text: c.cluster + ' is ' + (who || 'named') + ' in this transcript, but nothing was ' +
              'stored to recognise the voice again' +
              (r.detail ? ': ' + r.detail : ' (' + r.enroll_reason + ').')
      };
    }
    if (r.enrolled === true) {
      return { text: c.cluster + ' is ' + who + ', and this cluster is now enrolled: ' +
                     'the voice will be matched in later meetings.' };
    }
    return { text: c.cluster + ' accepted as ' + (who || 'the best match') + '.' };
  }

  // ----------------------------------------------------------------- screen
  MS.screens.review = {
    title: 'Review',

    async render(root, ctx) {
      var s = { alive: true, close: null };
      session = s;

      var wrap = el('div', 'max-width:860px');
      root.appendChild(wrap);

      var hasAudio = {};
      var notice = null;

      // one extra call, for one fact: whether "play a sample" is a true sentence
      async function audioMap() {
        try {
          var lib = await ctx.api('/api/library');
          var ms = (lib && lib.meetings) || [];
          for (var i = 0; i < ms.length; i++) {
            if (ms[i] && ms[i].id) hasAudio[ms[i].id] = ms[i].has_audio === true;
          }
        } catch (_) { /* unknown stays false: the copy says "read", not "play" */ }
      }

      var act = {
        done: function (e, r, c, nm) {
          if (!s.alive) return;
          notice = e ? { warn: true, text: apiErr(e).message } : noticeFor(r, c, nm);
          load();
        }
      };

      async function load() {
        var d = null;
        try {
          d = await ctx.api('/api/review');
        } catch (e) {
          if (s.alive) failWhole(wrap, e);
          return;
        }
        if (!s.alive) return;
        paint(wrap, d || {}, ctx, hasAudio, act, notice);
      }

      await audioMap();
      if (!s.alive) return;
      await load();
    },

    destroy() {
      if (session) {
        session.alive = false;
        if (typeof session.close === 'function') {
          try { session.close(); } catch (_) { /* already gone */ }
        }
        session = null;
      }
    }
  };
})();
