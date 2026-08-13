/* Settings — design/Meetscribe.dc.html lines 259-310.

   Every inline style string is carried over from the design. All three designed
   sections are here: Recognition (the three sliders), Models (the table), On
   disk (the path fields) — plus the "What to expect" aside, verbatim.

   Three places the design could not be followed, all marked DESIGN/DATA below:

     - "Reveal in finder" is omitted. Nothing in the server opens a file
       manager, and a button that does nothing is worse than no button.
     - "Delete all voice profiles" is omitted. There is no bulk endpoint;
       /api/voices/forget takes one id, and synthesising a wipe out of N calls
       that can half-fail is not something this screen should invent.
     - "Voice profiles" is a read-only path. POST /api/settings accepts
       library/mode/host/work/glossary and the four thresholds — not
       speaker_db, which is fixed by --speaker-db at startup.

   Two sections are additions, placed after the designed ones and set in the
   design's own idiom: Pipeline (mode/host/work + reachability) and Glossary.
   Both are real settings that really take effect — the backend is rebuilt on
   write and the glossary is what /api/ingest sends when a caller omits one —
   and neither the Settings design nor the Ingest design has a home for them.

   Numbers shown here are the server's, never the design's samples: the model
   names and the silence gate are parsed out of bench/, the window is observed
   from the transcripts in the library, and the store counts are SELECT COUNT(*)
   over speakers.db. The design's throughput/recall figures are benchmark prose
   and are labelled as such rather than dressed up as live stats. */
(function () {
  'use strict';

  window.MS = window.MS || {};
  window.MS.screens = window.MS.screens || {};

  // ---------------------------------------------------------------- helpers
  function el(tag, style, text) {
    var n = document.createElement(tag);
    if (style) n.setAttribute('style', style);
    if (text !== undefined && text !== null) n.textContent = String(text);
    return n;
  }

  function clear(node) {
    while (node && node.firstChild) node.removeChild(node.firstChild);
  }

  function num(v) {
    var n = Number(v);
    return (v === null || v === undefined || v === '' || !isFinite(n)) ? null : n;
  }

  // 30 -> "30", 0.55 -> "0.55". Keeps a trailing decimal only when it is real.
  function trim(v) {
    var n = num(v);
    if (n === null) return null;
    return String(Math.round(n * 1000) / 1000);
  }

  function plural(n, one, many) {
    return String(n) + ' ' + (Number(n) === 1 ? one : many);
  }

  function obj(v) {
    return (v && typeof v === 'object' && !Array.isArray(v)) ? v : {};
  }

  function arr(v) {
    return Array.isArray(v) ? v : [];
  }

  function post(ctx, path, body) {
    return ctx.api(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
  }

  // ctx.api throws on non-2xx; the shell may or may not hand the parsed body
  // back on the error. Dig for it — the server's own wording ("the review floor
  // cannot sit above the accept line") is the whole value of the failure.
  function errMsg(e) {
    if (!e) return 'that did not save';
    var b = null;
    var cands = [e.body, e.data, e.payload, e.json, e.response];
    for (var i = 0; i < cands.length; i++) {
      var c = cands[i];
      if (typeof c === 'string' || (c && typeof c === 'object' && typeof c !== 'function')) { b = c; break; }
    }
    if (typeof b === 'string') {
      try { b = JSON.parse(b); } catch (_) { b = { error: b }; }
    }
    if (!b && typeof e.message === 'string') {
      var i2 = e.message.indexOf('{');
      if (i2 >= 0) { try { b = JSON.parse(e.message.slice(i2)); } catch (_2) { b = null; } }
    }
    if (b && typeof b === 'object' && typeof b.error === 'string' && b.error) return b.error;
    return (e.message ? String(e.message) : String(e));
  }

  // An empty note must not occupy space: several of them sit directly inside
  // flex columns whose `gap` would otherwise open up around nothing.
  function say(node, text, style) {
    if (!node) return;
    if (style) node.setAttribute('style', style);
    node.textContent = text || '';
    node.style.display = node.textContent ? '' : 'none';
  }

  function clockOf(ts) {
    var n = num(ts);
    if (n === null) return null;
    var d = new Date(n * 1000);
    if (isNaN(d.getTime())) return null;
    return String(d.getHours()).padStart(2, '0') + ':' + String(d.getMinutes()).padStart(2, '0');
  }

  // -------------------------------------------------------------- constants
  var EYEBROW = 'font-size:10.5px;letter-spacing:0.12em;text-transform:uppercase;' +
                'color:var(--color-neutral-600);margin-bottom:14px';
  var CAPTION = 'font-size:11.5px;color:var(--color-neutral-700);margin-top:5px';
  var CAPTION_OFF = CAPTION + ';display:none';
  var NOTE = 'font-size:11.5px;color:var(--color-neutral-700);line-height:1.5';
  var BAD = 'font-size:11.5px;color:var(--color-accent-2-700);line-height:1.5;margin-top:5px';
  var OK = 'font-size:11.5px;color:var(--color-accent-700);line-height:1.5;margin-top:5px';
  var COL = 'max-width:520px;display:flex;flex-direction:column;gap:14px;margin-bottom:38px';
  var LINKISH = 'background:none;border:0;padding:0;font:inherit;color:var(--color-accent);' +
                'cursor:pointer;text-decoration:underline;text-underline-offset:3px';

  // The server's own wording, reused so the client-side guard and a rejected
  // write say exactly the same thing.
  var ORDER_ERR = 'the review floor cannot sit above the accept line';

  // Design ranges, lines 268 / 273 / 277. All three sit inside the server's
  // accepted bounds (accept 0.10-0.99, review 0.05-0.95, margin 0.0-0.50), so
  // nothing reachable on these tracks can be refused for being out of range.
  var SLIDERS = [
    { key: 'accept', label: 'Accept a match at', min: '0.42', max: '0.85',
      caption: 'Measured error-free between 0.41 and 0.86 on centroid comparisons. ' +
               'Lower accepts more and risks a false name.' },
    { key: 'review', label: 'Send to review above', min: '0.20', max: '0.55', caption: null },
    { key: 'margin', label: 'Margin over runner-up', min: '0.02', max: '0.30', caption: null }
  ];

  // ----------------------------------------------------------------- screen
  window.MS.screens.settings = {
    title: 'Settings',

    async render(root, ctx) {
      var self = this;
      self._alive = true;
      clear(root);
      root.appendChild(el('p', 'font-size:13.5px;color:var(--color-neutral-700);margin:0',
                          'Reading settings…'));

      var cfg = null;
      try {
        cfg = await ctx.api('/api/settings');
      } catch (err) {
        if (!self._alive) return;
        clear(root);
        root.appendChild(el('h1', 'font-size:40px;margin:0 0 26px', 'Settings'));
        root.appendChild(el('p', 'font-size:13.5px;color:var(--color-accent-2-700);max-width:52ch;margin:0',
                            'Settings could not be read: ' + errMsg(err)));
        return;
      }
      if (!self._alive) return;
      if (!cfg || typeof cfg !== 'object') cfg = {};

      // Every write returns the whole payload back, so the screen never has to
      // guess what took effect. Refreshers re-read `cfg` in place rather than
      // rebuilding the DOM, which would steal focus out of a live control.
      var refreshers = [];
      function onRefresh(fn) { refreshers.push(fn); fn(); }
      function refreshAll() {
        for (var i = 0; i < refreshers.length; i++) {
          try { refreshers[i](); } catch (e) { /* one stale readout is not fatal */ }
        }
      }

      // Writes are chained: dragging a slider then tabbing to a field must not
      // let two responses land out of order and leave `cfg` behind the server.
      var chain = Promise.resolve();
      function commit(patch, note, okText) {
        say(note, 'Saving…', OK);
        var run = chain.then(function () { return post(ctx, '/api/settings', patch); })
          .then(function (fresh) {
            if (!self._alive) return false;
            if (fresh && typeof fresh === 'object') cfg = fresh;
            say(note, okText || 'Saved.', OK);
            refreshAll();
            return true;
          }, function (e) {
            if (!self._alive) return false;
            say(note, errMsg(e), BAD);
            refreshAll();
            return false;
          });
        chain = run.then(null, function () { });
        return run;
      }

      clear(root);
      var grid = el('div', 'display:grid;grid-template-columns:minmax(0,1fr) 280px;gap:56px;' +
                           'align-items:start;max-width:1100px');
      var main = el('div');
      grid.appendChild(main);

      main.appendChild(el('h1', 'font-size:40px;margin:0 0 26px', 'Settings'));

      // ------------------------------------------------------- Recognition
      main.appendChild(el('div', EYEBROW, 'Recognition'));
      var rec = el('div', 'max-width:520px;margin-bottom:38px');

      SLIDERS.forEach(function (spec, i) {
        var wrap = el('div', i < SLIDERS.length - 1 ? 'margin-bottom:22px' : null);

        var head = el('div', 'display:flex;justify-content:space-between;font-size:13.5px;margin-bottom:6px');
        head.appendChild(el('span', null, spec.label));
        var readout = el('span', 'font-variant-numeric:tabular-nums;color:var(--color-accent-700)', '—');
        head.appendChild(readout);
        wrap.appendChild(head);

        var input = document.createElement('input');
        input.type = 'range';
        input.min = spec.min;
        input.max = spec.max;
        input.step = '0.01';
        input.setAttribute('style', 'width:100%');
        input.setAttribute('aria-label', spec.label);
        wrap.appendChild(input);

        if (spec.caption) wrap.appendChild(el('div', CAPTION, spec.caption));

        // "changed from 0.55 · restore" — the server reports which keys it is
        // holding an override for, so this is never a guess.
        var meta = el('div', CAPTION_OFF);
        wrap.appendChild(meta);
        var note = el('div', CAPTION_OFF);
        wrap.appendChild(note);

        function saved() {
          var v = num(obj(cfg.recognition)[spec.key]);
          return v === null ? null : v;
        }

        function paintValue() {
          var v = saved();
          readout.textContent = v === null ? '—' : v.toFixed(2);
          if (v === null) {
            input.disabled = true;
            return;
          }
          input.disabled = false;
          if (document.activeElement !== input) input.value = String(v);

          // A stored value outside the design's track (set from the CLI or an
          // earlier build) would otherwise be silently clamped by the range
          // input. Say so instead of misreporting it.
          var lo = Number(spec.min), hi = Number(spec.max);
          if (v < lo - 1e-9 || v > hi + 1e-9) {
            readout.textContent = v.toFixed(2);
            input.title = 'The stored value ' + v.toFixed(2) + ' is outside this slider’s range (' +
                          spec.min + '–' + spec.max + '). Moving the slider will bring it inside.';
          } else {
            input.title = '';
          }
        }

        function paintMeta() {
          clear(meta);
          meta.style.display = 'none';
          var r = obj(cfg.recognition);
          var def = num(obj(r.defaults)[spec.key]);
          var cur = saved();
          var over = arr(r.overridden).indexOf(spec.key) >= 0;
          // The server keeps the preferences entry even once the value is put
          // back, so `overridden` alone would leave "changed from 0.55" sitting
          // above a 0.55. Only say it changed while it has actually changed.
          if (!over || def === null || cur === null || Math.abs(cur - def) < 1e-9) return;
          meta.style.display = '';
          meta.appendChild(document.createTextNode('changed from ' + def.toFixed(2) + ' · '));
          var b = el('button', LINKISH, 'restore');
          b.type = 'button';
          b.title = 'Set it back to ' + def.toFixed(2) + ', the value compiled into ' +
                    (r.source || 'the pipeline') + '. The preferences file keeps an entry either way.';
          b.addEventListener('click', function () {
            var patch = {};
            patch[spec.key] = def;
            commit(patch, note, 'Restored to ' + def.toFixed(2) + '.');
          });
          meta.appendChild(b);
        }

        onRefresh(function () { paintValue(); paintMeta(); });

        input.addEventListener('input', function () {
          var v = num(input.value);
          readout.textContent = v === null ? '—' : v.toFixed(2);
        });

        input.addEventListener('change', function () {
          var v = num(input.value);
          if (v === null) { paintValue(); return; }
          v = Math.round(v * 100) / 100;

          // review <= accept is the server's rule. Check it here too, so the
          // one combination the two design tracks can compose does not have to
          // cost a round trip to be refused.
          var r = obj(cfg.recognition);
          var other = (spec.key === 'review') ? num(r.accept)
                    : (spec.key === 'accept') ? num(r.review) : null;
          var bad = (spec.key === 'review' && other !== null && v > other) ||
                    (spec.key === 'accept' && other !== null && other > v);
          if (bad) {
            say(note, ORDER_ERR, BAD);
            paintValue();
            return;
          }
          var patch = {};
          patch[spec.key] = v;
          commit(patch, note, 'Saved.');
        });

        rec.appendChild(wrap);
      });

      // What these thresholds actually govern — the server's own sentence, not
      // a claim this screen makes on its behalf.
      var scope = el('div', NOTE + ';margin-top:16px;display:none');
      onRefresh(function () {
        var r = obj(cfg.recognition);
        var bits = [];
        if (r.applies_to) bits.push('Applies to ' + r.applies_to);
        if (r.source) bits.push('Defaults read from ' + r.source + '.');
        say(scope, bits.join(' '), NOTE + ';margin-top:16px');
      });
      rec.appendChild(scope);
      main.appendChild(rec);

      // ------------------------------------------------------------ Models
      main.appendChild(el('div', EYEBROW, 'Models'));
      var table = el('table', 'max-width:520px;margin-bottom:38px');
      table.className = 'table';
      var tbody = document.createElement('tbody');
      table.appendChild(tbody);

      function modelRow(label, numeric, get, why) {
        var tr = document.createElement('tr');
        tr.appendChild(el('td', 'color:var(--color-neutral-700)', label));
        var td = el('td', numeric ? 'text-align:right;font-variant-numeric:tabular-nums'
                                  : 'text-align:right');
        tr.appendChild(td);
        onRefresh(function () {
          var v = get(obj(cfg.models));
          td.textContent = (v === null || v === undefined || v === '') ? '—' : String(v);
          var t = (typeof why === 'function') ? why(obj(cfg.models)) : why;
          if (t) tr.title = t;
        });
        tbody.appendChild(tr);
      }

      modelRow('Transcribe', false, function (m) { return m.transcribe; },
               'Parsed out of the pipeline source in bench/. Set there, not here.');
      modelRow('Voiceprints', false, function (m) { return m.voiceprints; },
               'The embedding model every stored centroid was made with. Set in the pipeline, not here.');
      modelRow('Window', true, function (m) {
        var seen = arr(m.window_s_observed).map(num).filter(function (n) { return n !== null; });
        if (seen.length > 1) return seen.map(function (n) { return trim(n) + ' s'; }).join(' · ');
        var w = num(m.window_s);
        return w === null ? null : trim(w) + ' s';
      }, function (m) {
        var seen = arr(m.window_s_observed);
        if (!seen.length) {
          return 'No transcript in this library records a window length, so there is nothing to report here.';
        }
        return 'Observed in the transcripts in this library, not a setting — it is whatever the ' +
               'pipeline used when each meeting was linked.';
      });
      modelRow('Enrollment floor', true, function (m) {
        var v = num(m.min_enroll_sec);
        return v === null ? null : trim(v) + ' s of clean speech';
      }, 'Below this a cluster is not enrolled as a voiceprint. Naming it still labels the one transcript.');
      modelRow('Silence gate', true, function (m) {
        var v = num(m.silence_gate_db);
        if (v === null) return null;
        return (v < 0 ? '−' + trim(Math.abs(v)) : trim(v)) + ' dB';
      }, 'Set in the pipeline, not here.');

      main.appendChild(table);

      // ----------------------------------------------------------- On disk
      main.appendChild(el('div', EYEBROW, 'On disk'));
      var disk = el('div', COL);

      // DESIGN/DATA: the design labels this "Work directory". In this app the
      // library IS the work directory — run_pipeline writes the raw JSON, the
      // embeddings, the log and the linked transcript into it — and every other
      // screen calls it the library, so it is named for what it is.
      var libField = el('div', null);
      libField.className = 'field';
      libField.appendChild(el('label', null, 'Library'));
      var libInput = document.createElement('input');
      libInput.className = 'input';
      libInput.type = 'text';
      libInput.spellcheck = false;
      libInput.setAttribute('autocapitalize', 'off');
      libInput.setAttribute('autocomplete', 'off');
      libField.appendChild(libInput);
      libInput.title = 'Transcripts, their embeddings and the pipeline’s intermediate ' +
                       'files are written here.';
      var libNote = el('div', CAPTION_OFF);
      libField.appendChild(libNote);
      var libSaved = el('div', CAPTION_OFF);
      libField.appendChild(libSaved);
      disk.appendChild(libField);

      onRefresh(function () {
        var p = obj(cfg.paths);
        if (document.activeElement !== libInput) libInput.value = p.library || '';
        say(libNote, (p.library_persisted === false && p.note) ? p.note : '', CAPTION);
      });
      libInput.addEventListener('keydown', function (ev) {
        if (ev.key === 'Enter') { ev.preventDefault(); libInput.blur(); }
      });
      libInput.addEventListener('change', function () {
        var v = String(libInput.value || '').trim();
        if (!v) { refreshAll(); return; }
        if (v === obj(cfg.paths).library) return;
        commit({ library: v }, libSaved, 'Library moved. The profile store and glossary moved with it.');
      });

      // DESIGN/DATA: read-only. POST /api/settings has no speaker_db key — the
      // store is fixed by --speaker-db (or MS_SPEAKER_DB) at startup, and
      // otherwise sits inside whatever the library is.
      var dbField = el('div', null);
      dbField.className = 'field';
      dbField.appendChild(el('label', null, 'Voice profiles'));
      var dbInput = document.createElement('input');
      dbInput.className = 'input';
      dbInput.type = 'text';
      dbInput.readOnly = true;
      dbInput.spellcheck = false;
      dbInput.setAttribute('style', 'color:var(--color-neutral-700)');
      dbInput.title = 'Fixed at startup by --speaker-db (or MS_SPEAKER_DB); otherwise it is ' +
                      'speakers.db inside the library. It cannot be moved from this screen.';
      dbField.appendChild(dbInput);
      var dbNote = el('div', CAPTION_OFF);
      dbField.appendChild(dbNote);
      disk.appendChild(dbField);

      onRefresh(function () {
        var p = obj(cfg.paths);
        dbInput.value = p.speaker_db || '';
        var s = obj(cfg.store);
        var bits = [];
        if (p.speaker_db_exists === false) {
          bits.push('No store at this path yet — it is created when the first voice is enrolled.');
        } else if (typeof s.error === 'string') {
          bits.push('The store could not be read: ' + s.error);
        } else {
          var counts = [];
          if (num(s.voices) !== null) counts.push(plural(s.voices, 'voice', 'voices'));
          if (num(s.voiceprints) !== null) counts.push(plural(s.voiceprints, 'voiceprint', 'voiceprints'));
          if (num(s.decisions) !== null) counts.push(plural(s.decisions, 'decision', 'decisions'));
          if (counts.length) bits.push(counts.join(' · ') + '.');
        }
        if (num(s.meetings) !== null) {
          bits.push(plural(s.meetings, 'transcript', 'transcripts') + ' in the library.');
        }
        say(dbNote, bits.join(' '), CAPTION);
      });

      // DESIGN/DATA: the design's two buttons here are gone. "Reveal in finder"
      // has no endpoint behind it, and there is no bulk delete — /api/voices/forget
      // takes a single id, and the Voices screen is where a profile is dropped.
      var gone = el('div', CAPTION,
        'Profiles are dropped one at a time on the Voices screen, where you can see ' +
        'what each one is before it goes.');
      var goneWrap = el('div', 'margin-top:6px');
      goneWrap.appendChild(gone);
      var toVoices = el('button', 'margin-top:8px', 'Open Voices');
      toVoices.className = 'btn btn-secondary';
      toVoices.type = 'button';
      toVoices.addEventListener('click', function () { ctx.go('voices'); });
      goneWrap.appendChild(toVoices);
      disk.appendChild(goneWrap);

      main.appendChild(disk);

      // ---------------------------------------------------------- Pipeline
      // ADDITION (not in the design): mode/host/work are writable and rebuild
      // the backend on save, and nothing else in the design exposes them.
      main.appendChild(el('div', EYEBROW, 'Pipeline'));
      var pipe = el('div', COL);

      var statusRow = el('div', 'display:flex;align-items:center;gap:10px;flex-wrap:wrap');
      var statusTag = el('span', null, '—');
      statusTag.className = 'tag tag-neutral';
      statusRow.appendChild(statusTag);
      var statusText = el('span', 'font-size:12.5px;color:var(--color-neutral-700)');
      statusRow.appendChild(statusText);
      var recheck = el('button', 'margin-left:auto', 'Check again');
      recheck.className = 'btn btn-secondary';
      recheck.type = 'button';
      statusRow.appendChild(recheck);
      pipe.appendChild(statusRow);
      var pipeNote = el('div', CAPTION_OFF);
      pipe.appendChild(pipeNote);

      recheck.addEventListener('click', function () {
        recheck.disabled = true;
        statusText.textContent = 'Checking…';
        chain = chain.then(function () { return ctx.api('/api/settings?check=1'); })
          .then(function (fresh) {
            if (!self._alive) return;
            if (fresh && typeof fresh === 'object') cfg = fresh;
            refreshAll();
          }, function (e) {
            if (!self._alive) return;
            statusText.textContent = errMsg(e);
          })
          .then(function () { if (self._alive) recheck.disabled = false; });
      });

      onRefresh(function () {
        var b = obj(cfg.backend);
        var up = b.reachable;
        if (up === true) {
          statusTag.className = 'tag tag-accent';
          statusTag.textContent = 'reachable';
        } else if (up === false) {
          statusTag.className = 'tag tag-accent-2';
          statusTag.textContent = 'not reachable';
        } else {
          statusTag.className = 'tag tag-neutral';
          statusTag.textContent = 'unchecked';
        }
        var t = clockOf(b.checked_at);
        var where = b.host ? b.host : 'this machine';
        var line = where + (t ? ', checked at ' + t : '');
        if (up === false) {
          line += ' — browsing, naming and merging still work; only ingesting new audio needs it.';
        }
        statusText.textContent = line;
        say(pipeNote, b.note || '', CAPTION);
      });

      function textField(label, get, key, help, saveText) {
        var f = el('div', null);
        f.className = 'field';
        f.appendChild(el('label', null, label));
        var input = document.createElement('input');
        input.className = 'input';
        input.type = 'text';
        input.spellcheck = false;
        input.setAttribute('autocapitalize', 'off');
        input.setAttribute('autocomplete', 'off');
        f.appendChild(input);
        if (help) f.appendChild(el('div', CAPTION, help));
        var note = el('div', CAPTION_OFF);
        f.appendChild(note);
        onRefresh(function () {
          var v = get(obj(cfg.backend));
          if (document.activeElement !== input) input.value = (v === null || v === undefined) ? '' : String(v);
        });
        input.addEventListener('keydown', function (ev) {
          if (ev.key === 'Enter') { ev.preventDefault(); input.blur(); }
        });
        input.addEventListener('change', function () {
          var v = String(input.value || '').trim();
          if (!v) { refreshAll(); return; }
          if (v === get(obj(cfg.backend))) return;
          var patch = {};
          patch[key] = v;
          commit(patch, note, saveText || 'Saved.');
        });
        return f;
      }

      /* Host is the switch now: there is no Mode select, because there are no
         longer two backends. Empty means the pipeline runs on this machine. */
      var hostField = textField('Host', function (b) { return b.host; }, 'host',
        'The ssh alias of a GPU box. Leave empty to run the pipeline on this machine.',
        'Host changed and the pipeline re-checked.');
      pipe.appendChild(hostField);

      pipe.appendChild(textField('Pipeline directory', function (b) { return b.work; }, 'work',
        'Where transcribe_meeting.py and link/ are installed — on the box in split mode, ' +
        'on this machine in all-in-one.',
        'Directory changed and re-checked.'));

      main.appendChild(pipe);

      // ---------------------------------------------------------- Glossary
      // ADDITION (not in the design): real and effective — POST /api/ingest
      // sends this list whenever a caller omits a glossary of its own.
      main.appendChild(el('div', EYEBROW, 'Glossary'));
      var gloss = el('div', COL);
      var glossField = el('div', null);
      glossField.className = 'field';
      glossField.appendChild(el('label', null, 'Names and terms the transcriber should expect'));
      var glossBox = document.createElement('textarea');
      glossBox.className = 'input';
      glossBox.spellcheck = false;
      glossBox.setAttribute('style', 'min-height:96px');
      glossBox.setAttribute('placeholder', 'One term per line');
      glossField.appendChild(glossBox);
      var glossNote = el('div', CAPTION);
      glossField.appendChild(glossNote);
      gloss.appendChild(glossField);

      var glossSave = el('button', null, 'Save glossary');
      glossSave.className = 'btn btn-secondary';
      glossSave.type = 'button';
      var glossSaved = el('div', CAPTION_OFF);
      var glossRow = el('div', 'display:flex;align-items:center;gap:12px;flex-wrap:wrap');
      glossRow.appendChild(glossSave);
      glossRow.appendChild(glossSaved);
      gloss.appendChild(glossRow);

      onRefresh(function () {
        var g = obj(cfg.glossary);
        var terms = arr(g.terms).map(function (t) { return String(t); });
        if (document.activeElement !== glossBox) glossBox.value = terms.join('\n');
        glossNote.textContent = 'Sent with every ingest that does not carry its own list. Stored at ' +
                                (g.path || 'the library') + '.';
      });

      glossSave.addEventListener('click', function () {
        var terms = String(glossBox.value || '').split('\n')
          .map(function (t) { return t.trim(); })
          .filter(function (t) { return t.length > 0; });
        commit({ glossary: terms }, glossSaved,
               terms.length ? plural(terms.length, 'term', 'terms') + ' saved.' : 'Glossary cleared.');
      });

      main.appendChild(gloss);

      // -------------------------------------------------------------- aside
      var aside = el('aside', 'padding-top:16px');
      aside.appendChild(el('div', EYEBROW, 'What to expect'));
      aside.appendChild(el('div', 'font-size:13px;line-height:1.6;color:var(--color-neutral-800);margin-bottom:20px',
        'Word recall runs 82–87% on far-field meeting audio with one microphone in the room.'));
      aside.appendChild(el('div', 'font-size:13px;line-height:1.6;color:var(--color-neutral-800);margin-bottom:20px',
        'Roughly a quarter of words in a real meeting are spoken over someone else. One speaker is ' +
        'kept at a time, so short interruptions are the thing most often lost.'));
      aside.appendChild(el('div', 'font-size:13px;line-height:1.6;color:var(--color-neutral-800)',
        'Recognition was validated against a small set of enrolled people. The larger the profile ' +
        'store, the more useful it is to name who is attending before you run a meeting.'));

      // The three paragraphs above are benchmark prose from the project's
      // notes. This server measures none of them, so they are attributed
      // rather than left to read as live figures.
      aside.appendChild(el('div', 'font-size:11.5px;line-height:1.5;color:var(--color-neutral-600);margin-top:20px',
        'Measured once against a benchmark set, not computed from your library. Nothing on this page ' +
        'recalculates them.'));

      grid.appendChild(aside);
      root.appendChild(grid);
    },

    destroy: function () {
      this._alive = false;
    }
  };
}());
