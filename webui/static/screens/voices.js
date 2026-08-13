/* Voices — design/Meetscribe.dc.html lines 143-185, dialogs 378-428.
   Profile table on the left, voiceprint rail on the right, rename / merge /
   forget behind the three buttons at the foot of the rail.

   Every inline style string is carried over from the design. The numbers come
   from GET /api/voices — sessions, enrolled seconds, meetings, last heard, the
   48 bars (RMS of the stored 256-d centroid) and the nearest-voice cosine are
   all real. Two places where the design assumed data this store does not have
   are marked DESIGN/DATA below: the per-target similarity in the merge dialog
   (only one neighbour is scored per profile) and the forget copy (the store
   drops the name from past transcripts, it does not keep it as text). */
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
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  function num(v) {
    var n = Number(v);
    return isFinite(n) ? n : null;
  }

  // Seconds as the design writes them: "214 s", but keeping one decimal when
  // the store has one so a 61.8 s enrolment is not rounded into a rounder lie.
  function secs(v) {
    var n = num(v);
    if (n === null) return null;
    var r = Math.round(n * 10) / 10;
    return (Math.abs(r - Math.round(r)) < 0.05) ? String(Math.round(r)) : r.toFixed(1);
  }

  function score(v) {
    var n = num(v);
    return n === null ? null : n.toFixed(2);
  }

  function times(n) {
    // What 'Sessions' counted: stored voiceprints, i.e. the number of times a
    // person told this thing who a voice is. That is a fact about your input,
    // not about the recordings -- it sat beside Meetings looking like a
    // synonym and was neither. Say it in words so it cannot be misread as a
    // count of anything in the library.
    if (!n) return 'never';
    return n === 1 ? 'once' : n === 2 ? 'twice' : n + ' times';
  }

  function plural(n, one, many) {
    return String(n) + ' ' + (Number(n) === 1 ? one : many);
  }

  // ctx.api throws on non-2xx; the shell may or may not hand the parsed body
  // back on the error. Dig for it, because the server's `reason` is the
  // difference between "that failed" and "that name is a merge, not a rename".
  function errInfo(e) {
    var out = { message: '', reason: null, existing_id: null };
    if (!e) return out;
    var b = null;
    var cands = [e.body, e.data, e.payload, e.json, e.response];
    for (var i = 0; i < cands.length; i++) {
      var c = cands[i];
      if (typeof c === 'string' || (c && typeof c === 'object' && typeof c !== 'function')) { b = c; break; }
    }
    if (typeof b === 'string') {
      try { b = JSON.parse(b); } catch (_) { b = { error: b }; }
    }
    if (!b && e.message && typeof e.message === 'string') {
      var i2 = e.message.indexOf('{');
      if (i2 >= 0) { try { b = JSON.parse(e.message.slice(i2)); } catch (_2) { b = null; } }
    }
    if (b && typeof b === 'object') {
      if (typeof b.error === 'string') out.message = b.error;
      if (typeof b.reason === 'string') out.reason = b.reason;
      if (b.existing_id !== undefined && b.existing_id !== null) out.existing_id = b.existing_id;
    }
    if (!out.message) out.message = (e && e.message) ? e.message : String(e);
    return out;
  }

  function post(ctx, path, obj) {
    return ctx.api(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(obj)
    });
  }

  // ------------------------------------------------------------------ screen
  window.MS.screens.voices = {
    title: 'Voices',

    async render(root, ctx) {
      var self = this;
      self._alive = true;

      clear(root);
      root.appendChild(el('p', 'font-size:13.5px;color:var(--color-neutral-700);margin:0',
                          'Reading the profile store…'));

      var data = null, loadErr = null;
      try {
        data = await ctx.api('/api/voices');
      } catch (e) {
        loadErr = errInfo(e).message;
      }
      if (!self._alive) return;

      // Thresholds live on /api/settings, not /api/voices. They only sharpen
      // the copy, so a failure here costs a clause and nothing else.
      var tun = null;
      try {
        var st = await ctx.api('/api/settings');
        if (st && st.recognition) tun = st.recognition;
      } catch (e2) {
        tun = null;
      }
      if (!self._alive) return;

      var accept = tun ? num(tun.accept) : null;
      // The design flags a profile that "sits close" to another at a literal
      // 0.45. The real line is the review threshold: at or above it a match is
      // no longer decided on its own. Fall back to the design's number.
      var closeAt = tun && num(tun.review) !== null ? num(tun.review) : 0.45;

      var voices = (data && Array.isArray(data.voices)) ? data.voices.filter(Boolean) : [];
      var floor = data ? num(data.min_enroll_sec) : null;
      var selectedId = null;
      var notice = null;

      function byId(id) {
        for (var i = 0; i < voices.length; i++) if (voices[i].id === id) return voices[i];
        return null;
      }

      function selected() {
        return byId(selectedId) || voices[0] || null;
      }

      function choose(prefer) {
        var want = (prefer !== undefined && prefer !== null) ? prefer
                 : (ctx.state && ctx.state.voiceId !== undefined ? ctx.state.voiceId : null);
        var v = byId(want) || byId(selectedId) || voices[0] || null;
        selectedId = v ? v.id : null;
        if (ctx.state) ctx.state.voiceId = selectedId;
      }

      choose(null);

      async function reload(prefer) {
        try {
          var fresh = await ctx.api('/api/voices');
          if (!self._alive) return;
          data = fresh;
          voices = (fresh && Array.isArray(fresh.voices)) ? fresh.voices.filter(Boolean) : [];
          floor = fresh ? num(fresh.min_enroll_sec) : null;
          loadErr = null;
        } catch (e) {
          if (!self._alive) return;
          loadErr = errInfo(e).message;
        }
        choose(prefer);
        paint();
      }

      // --------------------------------------------------------------- rail
      // "Only 8 s enrolled…" / "Sits 0.52 from another voice…" — design line 751,
      // with the floor and the close-enough line read from the server instead of
      // hardcoded at 10 and 0.45.
      function warningFor(v) {
        var en = num(v.enrolled_s);
        if (v.voiceprint && floor !== null && en !== null && en < floor * 2) {
          return 'Only ' + secs(en) + ' s enrolled — under the ' + secs(floor) +
                 ' s floor this profile would not have been stored, and near it ' +
                 'matching is unreliable. Name this voice again in a longer meeting.';
        }
        if (v.nearest && num(v.nearest.score) !== null && num(v.nearest.score) > closeAt) {
          return 'Sits ' + score(v.nearest.score) + ' from another voice on file — ' +
                 'close enough that a short turn could be handed to the wrong person.';
        }
        return '';
      }

      function nearFloor(v) {
        var en = num(v.enrolled_s);
        return (floor !== null && en !== null && en < floor * 2);
      }

      function fact(host, label, value, tabular, title) {
        var row = el('div', 'display:flex;justify-content:space-between');
        row.appendChild(el('span', 'color:var(--color-neutral-700)', label));
        var s = el('span', tabular ? 'font-variant-numeric:tabular-nums' : null, value);
        if (title) s.title = title;
        row.appendChild(s);
        host.appendChild(row);
        return row;
      }

      function drawAside(host, v) {
        var aside = el('aside', 'padding-top:16px');

        aside.appendChild(el('div', 'font-size:10.5px;letter-spacing:0.12em;text-transform:uppercase;' +
                                    'color:var(--color-neutral-600);margin-bottom:10px', 'Voiceprint'));
        aside.appendChild(el('div', 'font-size:25px;font-weight:600;margin-bottom:14px', v.name));

        var bars = Array.isArray(v.bars) ? v.bars : [];
        if (v.voiceprint && bars.length) {
          var strip = el('div', 'display:flex;align-items:flex-end;gap:2px;height:54px;margin-bottom:6px');
          for (var i = 0; i < bars.length; i++) {
            var b = num(bars[i]);
            if (b === null) b = 0;
            if (b < 0) b = 0;
            if (b > 1) b = 1;
            var h = Math.round(10 + b * 44);
            strip.appendChild(el('div', 'width:4px;height:' + h + 'px;background:' +
              (i % 7 === 0 ? 'var(--color-accent-600)' : 'var(--color-neutral-800)') + ';flex:none'));
          }
          strip.title = (data && data.bars_from) ? data.bars_from : '';
          aside.appendChild(strip);

          var parts = [];
          parts.push(v.dim ? (v.dim + '-dimension centroid') : 'centroid');
          var model = v.embed_model || (data && data.embed_model);
          if (model) parts.push(model);
          aside.appendChild(el('div', 'font-size:11px;color:var(--color-neutral-600);margin-bottom:24px',
                               parts.join(' · ')));
        } else {
          // voiceprint:false — a name with nothing stored behind it. The design
          // has no state for this; drawing 48 bars would be drawing nothing.
          var flat = el('div', 'display:flex;align-items:flex-end;height:54px;margin-bottom:6px');
          flat.appendChild(el('div', 'width:100%;height:1px;background:var(--color-neutral-300)'));
          aside.appendChild(flat);
          aside.appendChild(el('div', 'font-size:11px;color:var(--color-accent-2-700);margin-bottom:24px;line-height:1.5',
                               'No voiceprint stored — the name is on file, but nothing was enrolled ' +
                               'to recognise it in a later meeting.'));
        }

        var facts = el('div', 'display:flex;flex-direction:column;gap:10px;font-size:12.5px');
        fact(facts, 'Enrolled from', v.enrolled_from || '—', false,
             v.enrolled_from ? 'meeting · cluster of the newest enrolment' : 'nothing enrolled');

        var speech = '—', speechTitle = '';
        var en = num(v.enrolled_s);
        if (v.sessions && en !== null) {
          speech = secs(en) + ' s, named ' + times(v.sessions);
          if (ctx.fmt && typeof ctx.fmt.dur === 'function') {
            try { speechTitle = ctx.fmt.dur(en); } catch (_3) { speechTitle = ''; }
          }
        }
        fact(facts, 'Speech on file', speech, true, speechTitle);

        // The design prints the bare cosine. The name it belongs to is real and
        // costs nothing to carry, so it rides on the tooltip.
        var nearTxt = '—', nearTitle = 'no other voiceprint to compare against';
        if (v.nearest && score(v.nearest.score) !== null) {
          nearTxt = score(v.nearest.score);
          nearTitle = (v.nearest.name || 'unnamed') + ' · cosine against this centroid';
        } else if (!v.voiceprint) {
          nearTitle = 'no voiceprint stored, so nothing to compare';
        }
        fact(facts, 'Nearest other voice', nearTxt, true, nearTitle);
        aside.appendChild(facts);

        var warn = warningFor(v);
        aside.appendChild(el('div', warn
          ? 'margin-top:16px;font-size:12px;line-height:1.5;color:var(--color-accent-2-700)'
          : 'display:none', warn));

        var acts = el('div', 'margin-top:26px;display:flex;flex-direction:column;gap:8px');

        var bRename = el('button', 'margin:0', 'Rename');
        bRename.type = 'button';
        bRename.className = 'btn btn-secondary btn-block';
        bRename.onclick = function () { openRename(v); };
        acts.appendChild(bRename);

        var bMerge = el('button', 'margin:0', 'Merge into another voice');
        bMerge.type = 'button';
        bMerge.className = 'btn btn-secondary btn-block';
        if (voices.length < 2) {
          bMerge.disabled = true;
          bMerge.title = 'Nothing to merge into — this is the only voice on file.';
        } else {
          bMerge.onclick = function () { openMerge(v, null); };
        }
        acts.appendChild(bMerge);

        var bForget = el('button', 'margin:0;color:var(--color-accent-2-700)', 'Forget this voice');
        bForget.type = 'button';
        bForget.className = 'btn btn-ghost btn-block';
        bForget.onclick = function () { openForget(v); };
        acts.appendChild(bForget);

        aside.appendChild(acts);
        host.appendChild(aside);
      }

      // -------------------------------------------------------------- paint
      var RIGHT = 'text-align:right;font-variant-numeric:tabular-nums';

      function paint() {
        if (!self._alive) return;
        clear(root);

        var v = selected();
        var grid = el('div', v
          ? 'display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:56px;align-items:start;max-width:1180px'
          : 'max-width:1180px');
        var main = el('div');

        main.appendChild(el('h1', 'font-size:40px;margin:0 0 4px', 'Voices'));
        main.appendChild(el('p', 'font-size:13.5px;color:var(--color-neutral-700);max-width:54ch;margin:0 0 ' +
                            (notice || loadErr ? '10px' : '26px'),
                            'The profile store. One voiceprint per person, averaged across every ' +
                            'meeting they have been named in.'));

        if (notice) {
          main.appendChild(el('div', 'font-size:11.5px;line-height:1.5;color:var(--color-neutral-700);' +
                                     'max-width:54ch;margin:0 0 22px', notice));
        }
        if (loadErr) {
          main.appendChild(el('div', 'font-size:11.5px;line-height:1.5;color:var(--color-accent-2-700);' +
                                     'max-width:54ch;margin:0 0 22px',
                              'The profile store could not be read: ' + loadErr));
        }

        if (!voices.length) {
          if (!loadErr) {
            main.appendChild(el('div', 'font-size:13.5px;color:var(--color-neutral-700);max-width:54ch;line-height:1.55',
              'No voices on file. A profile appears here the first time a cluster is ' +
              'named on the Review screen — from then on the same voice is recognised ' +
              'in every meeting.'));
            if (data && data.db) {
              main.appendChild(el('div', 'font-size:11px;color:var(--color-neutral-600);margin-top:14px',
                                  data.db));
            }
            if (typeof ctx.go === 'function') {
              var goReview = el('button', 'margin-top:20px;width:auto', 'Open Review');
              goReview.type = 'button';
              goReview.className = 'btn btn-secondary';
              goReview.onclick = function () { ctx.go('review'); };
              main.appendChild(goReview);
            }
          }
          grid.appendChild(main);
          root.appendChild(grid);
          return;
        }

        var table = el('table');
        table.className = 'table';
        var thead = el('thead');
        var hr = el('tr');
        hr.appendChild(el('th', null, 'Name'));
        hr.appendChild(el('th', 'width:120px;text-align:right', 'Enrolled'));
        hr.appendChild(el('th', 'width:100px;text-align:right', 'Transcripts'));
        hr.appendChild(el('th', 'width:110px;text-align:right', 'Last heard'));
        thead.appendChild(hr);
        table.appendChild(thead);

        var tbody = el('tbody');
        voices.forEach(function (p) {
          var on = p.id === selectedId;
          var tr = el('tr', 'cursor:pointer;' +
            (on ? 'background:color-mix(in srgb, var(--color-accent) 8%, transparent);' : ''));
          tr.tabIndex = 0;
          function pick() {
            selectedId = p.id;
            if (ctx.state) ctx.state.voiceId = p.id;
            paint();
          }
          tr.onclick = pick;
          tr.onkeydown = function (e) {
            if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(); }
          };

          tr.appendChild(el('td', 'font-weight:' + (on ? 600 : 400) + ';' +
            (nearFloor(p) ? 'color:var(--color-accent-2-700);' : ''), p.name));

          var en = num(p.enrolled_s);
          var tdEn = el('td', RIGHT, en === null ? '—' : secs(en) + ' s');
          if (en !== null && ctx.fmt && typeof ctx.fmt.dur === 'function') {
            try { tdEn.title = ctx.fmt.dur(en); } catch (_4) { /* tooltip only */ }
          }
          tr.appendChild(tdEn);

          var tdM = el('td', RIGHT, p.meetings === undefined || p.meetings === null
                                     ? '—' : String(p.meetings));
          tdM.title = (Array.isArray(p.meeting_ids) && p.meeting_ids.length)
            ? p.meeting_ids.join(', ')
            : 'transcripts this voice appears in, named or recognised';
          tr.appendChild(tdM);

          var tdL = el('td', RIGHT + ';color:var(--color-neutral-700)',
                       p.last_heard_short || p.last_heard_str || '—');
          if (p.last_heard_str) {
            tdL.title = p.last_heard_str + ' — when this voice was last matched or enrolled, ' +
                        'not when the meeting was recorded';
          }
          tr.appendChild(tdL);

          tbody.appendChild(tr);
        });
        table.appendChild(tbody);
        main.appendChild(table);

        grid.appendChild(main);
        if (v) drawAside(grid, v);
        root.appendChild(grid);
      }

      // ------------------------------------------------------------ dialogs
      function shell(width) {
        var d = el('div', width || null);
        d.className = 'dialog';
        return d;
      }

      function actions(d) {
        var a = el('div');
        a.className = 'dialog-actions';
        d.appendChild(a);
        return a;
      }

      function button(host, label, cls, style) {
        var b = el('button', style || null, label);
        b.type = 'button';
        b.className = cls;
        host.appendChild(b);
        return b;
      }

      // --- rename (design lines 378-392)
      function renameBody(v) {
        var bits = [];
        var en = num(v.enrolled_s);
        if (v.sessions && en !== null) {
          bits.push(secs(en) + ' s of speech, named ' + times(v.sessions) + '.');
        } else {
          bits.push('Nothing is enrolled under this name yet.');
        }
        if (v.nearest && score(v.nearest.score) !== null) {
          var s = 'The nearest voice on file is ' + (v.nearest.name || 'an unnamed profile') +
                  ' at ' + score(v.nearest.score);
          if (accept !== null) {
            s += (num(v.nearest.score) < accept ? ' — under' : ' — above') +
                 ' the accept line of ' + score(accept);
          }
          bits.push(s + '.');
        }
        bits.push('Renaming changes the label only; the voiceprint behind it is untouched.');
        return bits.join(' ');
      }

      function openRename(v) {
        var d = shell();
        d.appendChild(el('div', null, 'Rename ' + v.name)).className = 'dialog-title';
        d.appendChild(el('div', null, renameBody(v))).className = 'dialog-body';

        var field = el('div');
        field.className = 'field';
        field.appendChild(el('label', null, 'Name'));
        var input = el('input');
        input.className = 'input';
        input.type = 'text';
        input.value = v.name || '';
        input.placeholder = 'Who is this?';
        field.appendChild(input);
        d.appendChild(field);

        d.appendChild(el('div', 'font-size:11.5px;color:var(--color-neutral-700);line-height:1.5',
                         'Named once. Every meeting afterwards recognises them without being asked again.'));

        var err = el('div', 'display:none');
        d.appendChild(err);

        var acts = actions(d);
        var cancel = button(acts, 'Cancel', 'btn btn-secondary');
        var ok = button(acts, 'Remember this voice', 'btn btn-primary');

        var close = ctx.dialog(d);
        cancel.onclick = function () { close(); };

        function fail(msg, mergeId) {
          err.setAttribute('style', 'font-size:11.5px;line-height:1.5;color:var(--color-accent-2-700)');
          clear(err);
          err.appendChild(document.createTextNode(msg));
          if (mergeId !== undefined && mergeId !== null && byId(mergeId)) {
            err.appendChild(document.createElement('br'));
            var b = el('button', 'margin-top:6px;padding-inline:0', 'Merge into ' + byId(mergeId).name + ' instead');
            b.type = 'button';
            b.className = 'btn btn-ghost';
            b.onclick = function () { close(); openMerge(v, mergeId); };
            err.appendChild(b);
          }
        }

        async function commit() {
          var name = input.value.trim();
          if (!name) { fail('A name is required.'); return; }
          if (name === v.name) { close(); return; }

          // Names are UNIQUE in the store, so an existing name is a merge and
          // the server will say so with 409. Catch the exact clash here too, in
          // case the shell swallows the error body.
          var clash = null;
          for (var i = 0; i < voices.length; i++) {
            if (voices[i].id !== v.id && voices[i].name === name) { clash = voices[i]; break; }
          }
          if (clash) {
            fail(clash.name + ' is already on file. Two profiles for one person is a merge, not a rename.',
                 clash.id);
            return;
          }

          ok.disabled = true;
          try {
            var res = await post(ctx, '/api/voices/rename', { id: v.id, name: name });
            if (!self._alive) return;
            close();
            var newName = (res && res.voice && res.voice.name) ? res.voice.name : name;
            notice = 'Renamed ' + v.name + ' to ' + newName + '.';
            await reload(v.id);
          } catch (e) {
            if (!self._alive) return;
            ok.disabled = false;
            var info = errInfo(e);
            if (info.reason === 'name-taken') fail(info.message, info.existing_id);
            else if (info.reason === 'not-found') {
              fail('That voice is no longer in the store.');
            } else fail('The rename failed: ' + info.message);
          }
        }

        ok.onclick = commit;
        input.addEventListener('keydown', function (e) {
          if (e.key === 'Enter') { e.preventDefault(); commit(); }
        });
        setTimeout(function () {
          try { input.focus(); input.select(); } catch (_5) { /* focus is a nicety */ }
        }, 0);
      }

      // --- merge (design lines 406-428)
      function openMerge(v, preselect) {
        var others = voices.filter(function (o) { return o.id !== v.id; });
        if (!others.length) return;

        // DESIGN/DATA: the design prints a similarity beside every target. The
        // store scores one neighbour per profile (`nearest`), so only that row
        // can carry a real number — the rest are listed without one rather than
        // with a made-up one. The known neighbour leads the list.
        var nearId = (v.nearest && v.nearest.id !== undefined) ? v.nearest.id : null;
        others.sort(function (a, b) {
          if (a.id === nearId) return -1;
          if (b.id === nearId) return 1;
          return 0;
        });

        var pickId = null;
        if (preselect !== undefined && preselect !== null && byId(preselect)) pickId = preselect;
        else if (nearId !== null && byId(nearId)) pickId = nearId;
        else pickId = others[0].id;

        var d = shell('width:min(520px,100%)');
        d.appendChild(el('div', null, 'Merge ' + v.name + ' into')).className = 'dialog-title';
        d.appendChild(el('div', null,
          'Two profiles, one person. The voiceprints are averaged and every past ' +
          'transcript is relabelled.')).className = 'dialog-body';

        var list = el('div', 'display:flex;flex-direction:column;gap:2px;margin:4px 0 8px');
        d.appendChild(list);

        d.appendChild(el('div', 'font-size:11.5px;color:var(--color-neutral-700);line-height:1.5;margin:0 0 4px',
          'Similarity is shown for the nearest voice on file — the store scores one ' +
          'neighbour per profile, so the others are listed without a number.'));

        var err = el('div', 'display:none');
        d.appendChild(err);

        var acts = actions(d);
        var cancel = button(acts, 'Cancel', 'btn btn-secondary');
        var ok = button(acts, 'Merge', 'btn btn-primary');

        function drawList() {
          clear(list);
          others.forEach(function (o) {
            var on = o.id === pickId;
            var row = el('div', 'padding:8px 10px;cursor:pointer;font-size:13.5px;border-radius:var(--radius-sm);' +
              (on ? 'background:color-mix(in srgb, var(--color-accent) 12%, transparent);' : ''));
            row.appendChild(document.createTextNode(o.name + ' '));
            if (o.id === nearId && score(v.nearest.score) !== null) {
              row.appendChild(el('span', 'color:var(--color-neutral-600);font-variant-numeric:tabular-nums;font-size:11.5px',
                                 'similarity ' + score(v.nearest.score)));
            }
            row.onclick = function () { pickId = o.id; drawList(); };
            list.appendChild(row);
          });
          var t = byId(pickId);
          ok.textContent = 'Merge into ' + (t ? t.name : '');
        }
        drawList();

        var close = ctx.dialog(d);
        cancel.onclick = function () { close(); };

        ok.onclick = async function () {
          var target = byId(pickId);
          if (!target) return;
          ok.disabled = true;
          try {
            var res = await post(ctx, '/api/voices/merge', { from: v.id, into: target.id });
            if (!self._alive) return;
            close();
            var moved = (res && res.voiceprints_moved !== undefined && res.voiceprints_moved !== null)
              ? ' — ' + plural(res.voiceprints_moved, 'voiceprint', 'voiceprints') + ' moved'
              : '';
            notice = 'Merged ' + v.name + ' into ' + target.name + moved + '.';
            await reload(target.id);
          } catch (e) {
            if (!self._alive) return;
            ok.disabled = false;
            err.setAttribute('style', 'font-size:11.5px;line-height:1.5;color:var(--color-accent-2-700)');
            err.textContent = 'The merge failed: ' + errInfo(e).message;
          }
        };
      }

      // --- forget (design lines 393-405)
      function forgetBody(v) {
        var en = num(v.enrolled_s);
        var head;
        if (v.sessions && en !== null) {
          head = 'Deletes ' + plural(v.sessions, 'voiceprint', 'voiceprints') +
                 ' and ' + secs(en) + ' s of enrolled speech. ';
        } else {
          head = 'Deletes the name. Nothing is enrolled behind it. ';
        }
        // DESIGN/DATA: the design says past transcripts keep the name as text.
        // This store keeps the decision log but deletes the speaker row, so the
        // name stops resolving in transcripts too. Saying otherwise would be a
        // promise the server does not keep.
        return head + 'The decision log is kept as history, but past transcripts stop ' +
               'showing the name and future meetings stop recognising the voice.';
      }

      function openForget(v) {
        var d = shell();
        d.appendChild(el('div', null, 'Forget ' + v.name + '?')).className = 'dialog-title';
        d.appendChild(el('div', null, forgetBody(v))).className = 'dialog-body';

        var err = el('div', 'display:none');
        d.appendChild(err);

        var acts = actions(d);
        var keep = button(acts, 'Keep', 'btn btn-secondary');
        var go = button(acts, 'Forget', 'btn btn-primary', 'background:var(--color-accent-2-600)');

        var close = ctx.dialog(d);
        keep.onclick = function () { close(); };

        go.onclick = async function () {
          go.disabled = true;
          try {
            var res = await post(ctx, '/api/voices/forget', { id: v.id });
            if (!self._alive) return;
            close();
            var bits = ['Forgot ' + v.name];
            var extra = [];
            if (res && res.voiceprints_deleted !== undefined && res.voiceprints_deleted !== null) {
              extra.push(plural(res.voiceprints_deleted, 'voiceprint', 'voiceprints') + ' deleted');
            }
            if (res && res.decisions_kept) {
              extra.push(plural(res.decisions_kept, 'decision', 'decisions') + ' kept as history');
            }
            notice = bits.join('') + (extra.length ? ' — ' + extra.join(', ') : '') + '.';
            selectedId = null;
            if (ctx.state) ctx.state.voiceId = null;
            await reload(null);
          } catch (e) {
            if (!self._alive) return;
            go.disabled = false;
            err.setAttribute('style', 'font-size:11.5px;line-height:1.5;color:var(--color-accent-2-700)');
            err.textContent = 'Forget failed: ' + errInfo(e).message;
          }
        };
      }

      paint();
    },

    destroy: function () {
      this._alive = false;
    }
  };
}());
