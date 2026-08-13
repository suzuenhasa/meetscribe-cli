/* Library — design/Meetscribe.dc.html lines 38-68.
   Meeting list on the left, "On this machine" rail on the right.
   Every inline style string below is carried over from the design; the numbers
   come from /api/library (and /api/review for the pending count) — nothing here
   is a placeholder. Where the design showed a figure this server cannot compute
   (throughput), the line is dropped rather than faked. */
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

  // "11 Aug 2026" -> "11 Aug" in the current year, otherwise left whole so the
  // year is never silently dropped. Falls back to the mtime the row carries.
  function shortDate(m) {
    var s = m && typeof m.date === 'string' ? m.date.trim() : '';
    if (s) {
      var p = s.split(/\s+/);
      if (p.length === 3 && p[2] === String(new Date().getFullYear())) {
        return p[0] + ' ' + p[1];
      }
      return s;
    }
    if (typeof m.mtime === 'number' && isFinite(m.mtime)) {
      var d = new Date(m.mtime * 1000);
      var mo = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][d.getMonth()];
      return d.getDate() + ' ' + mo;
    }
    return '—';
  }

  // ctx.fmt.dur is the contract; keep a compact fallback so a missing shell
  // helper cannot blank the whole row.
  function dur(ctx, secs) {
    var s = Number(secs);
    if (!isFinite(s) || s <= 0) return null;
    if (ctx && ctx.fmt && typeof ctx.fmt.dur === 'function') {
      try {
        var out = ctx.fmt.dur(s);
        if (out) return String(out);
      } catch (e) { /* fall through */ }
    }
    var h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), x = Math.floor(s % 60);
    var mm = h ? String(m).padStart(2, '0') : String(m);
    return (h ? h + ':' : '') + mm + ':' + String(x).padStart(2, '0');
  }

  function num(v) {
    return (typeof v === 'number' && isFinite(v)) ? String(v) : '—';
  }

  // "42:40 · 7 voices · 0 named", the design's meta line built from real counts.
  function metaLine(ctx, m) {
    var bits = [];
    var d = dur(ctx, m.duration_s);
    if (d) bits.push(d);
    var n = (typeof m.n_speakers === 'number') ? m.n_speakers : null;
    if (n !== null) bits.push(n + (n === 1 ? ' voice' : ' voices'));
    var named = (typeof m.n_named === 'number') ? m.n_named : null;
    if (n) {
      if (named !== null) bits.push(named === n ? 'all named' : named + ' named');
    }
    return bits.join(' · ');
  }

  // Only two states can occur in a library of finished transcripts: every voice
  // placed, or some still anonymous. The queued/running tags in the design
  // belong to the ingest queue, which is a different endpoint.
  function status(m) {
    var n = (typeof m.n_speakers === 'number') ? m.n_speakers : 0;
    var named = (typeof m.n_named === 'number') ? m.n_named : 0;
    if (!n) {
      return { label: 'transcribed', cls: 'tag tag-neutral',
               why: 'This transcript has no diarised voices to name.' };
    }
    if (named >= n) {
      return { label: 'transcribed', cls: 'tag tag-neutral', why: 'Every voice in this meeting has a name.' };
    }
    return {
      label: 'review',
      cls: 'tag tag-accent-2',
      why: (n - named) + ' of ' + n + ' voices in this meeting have no name yet.'
    };
  }

  // ------------------------------------------------------------------ render
  window.MS.screens.library = {
    title: 'Library',

    async render(root, ctx) {
      clear(root);
      root.appendChild(el('p', 'font-size:13.5px;color:var(--color-neutral-700);margin:0',
                          'Reading the library…'));

      var lib;
      try {
        lib = await ctx.api('/api/library');
      } catch (err) {
        clear(root);
        root.appendChild(el('h1', 'font-size:40px;margin:0 0 4px', 'Library'));
        root.appendChild(el('p', 'font-size:13.5px;color:var(--color-accent-2-700);max-width:52ch;margin:0 0 30px',
                            'The library could not be read: ' + (err && err.message ? err.message : String(err))));
        return;
      }

      // The pending count is the review queue's, not something this screen can
      // derive; if that call fails the button simply loses its number.
      var review = null;
      try {
        review = await ctx.api('/api/review');
      } catch (e) {
        review = null;
      }

      var meetings = Array.isArray(lib && lib.meetings) ? lib.meetings : [];
      var stats = (lib && lib.stats) ? lib.stats : {};

      // Meetings whose voices cannot be scored at all — used for an honest
      // tooltip on the "review" tag rather than a promise the Review screen
      // cannot keep.
      var blocked = {};
      var un = review && Array.isArray(review.unidentified) ? review.unidentified : [];
      for (var i = 0; i < un.length; i++) {
        var u = un[i];
        if (u && u.meeting) blocked[u.meeting] = u.detail || u.reason || '';
      }

      clear(root);

      var grid = el('div', 'display:grid;grid-template-columns:minmax(0,1fr) 250px;gap:56px;align-items:start;max-width:1180px');
      var main = el('div');
      grid.appendChild(main);

      main.appendChild(el('h1', 'font-size:40px;margin:0 0 4px', 'Library'));
      main.appendChild(el('p', 'font-size:13.5px;color:var(--color-neutral-700);max-width:52ch;margin:0 0 30px',
                          'Every meeting transcribed on this machine. A voice named once is recognised in all of them.'));

      if (!meetings.length) {
        main.appendChild(el('p', 'font-size:13.5px;color:var(--color-neutral-700);max-width:52ch;margin:0',
                            'No transcripts in this library yet. Ingest some audio and they will appear here.'));
      }

      meetings.forEach(function (m) {
        if (!m || typeof m !== 'object') return;
        var id = m.id != null ? String(m.id) : '';
        var open = function () {
          if (id) {
            ctx.state.meeting = id;
            ctx.go('transcript', { id: id });
          }
        };

        var row = el('div', 'padding:15px 0;border-bottom:1px solid color-mix(in srgb, var(--color-text) 8%, transparent);cursor:pointer');
        row.setAttribute('role', 'button');
        row.setAttribute('tabindex', '0');
        row.addEventListener('click', open);
        row.addEventListener('keydown', function (ev) {
          if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); open(); }
        });

        var head = el('div', 'display:flex;align-items:baseline;gap:14px');

        var date = el('div', 'font-size:10.5px;letter-spacing:0.12em;text-transform:uppercase;color:var(--color-neutral-600);width:96px;flex:none;font-variant-numeric:tabular-nums',
                      shortDate(m));
        date.title = 'Transcript written ' + (m.date || '—') +
                     '. The store keeps no recording date, so this is the file’s own.';
        head.appendChild(date);

        var current = ctx.state && ctx.state.meeting === id;
        head.appendChild(el('div',
          'font-size:20px;font-weight:600;line-height:1.2;letter-spacing:-0.01em;' +
          (current ? 'color:var(--color-accent-700);' : ''),
          m.title || id || 'Untitled'));

        var st = status(m);
        var tagWrap = el('div', 'margin-left:auto;flex:none');
        var tag = el('span', null, st.label);
        tag.className = st.cls;
        tag.title = st.why + (blocked[id] ? ' ' + blocked[id] + '.' : '');
        tagWrap.appendChild(tag);
        head.appendChild(tagWrap);

        row.appendChild(head);
        var meta = metaLine(ctx, m);
        if (meta) {
          row.appendChild(el('div', 'font-size:12.5px;color:var(--color-neutral-700);margin:5px 0 0 110px;font-variant-numeric:tabular-nums',
                             meta));
        }
        main.appendChild(row);
      });

      // ------------------------------------------------------------- the rail
      var aside = el('aside', 'padding-top:16px');
      aside.appendChild(el('div', 'font-size:10.5px;letter-spacing:0.12em;text-transform:uppercase;color:var(--color-neutral-600);margin-bottom:14px',
                           'On this machine'));

      var hours = (typeof stats.hours === 'number' && isFinite(stats.hours))
        ? stats.hours.toFixed(1) : '—';
      var cmyk = el('div', 'font-size:66px;font-weight:600;margin-bottom:2px');
      cmyk.className = 'cmyk-num';
      var paper = el('span', null, hours);
      paper.className = 'paper';
      cmyk.appendChild(paper);
      ['plate-c', 'plate-m', 'plate-y'].forEach(function (p) {
        var s = el('span', null, hours);
        s.className = 'plate ' + p;
        s.setAttribute('aria-hidden', 'true');
        cmyk.appendChild(s);
      });
      aside.appendChild(cmyk);
      aside.appendChild(el('div', 'font-size:12.5px;color:var(--color-neutral-700);margin-bottom:26px',
                           'hours of audio, transcribed'));

      var list = el('div', 'display:flex;flex-direction:column;gap:11px;font-size:13px');
      function statRow(label, value, valueStyle, why) {
        var r = el('div', 'display:flex;justify-content:space-between');
        r.appendChild(el('span', 'color:var(--color-neutral-700)', label));
        r.appendChild(el('span', valueStyle || 'font-variant-numeric:tabular-nums', value));
        if (why) r.title = why;
        list.appendChild(r);
      }

      statRow('Meetings', num(stats.meetings), null,
              'Transcripts in this library.');
      statRow('Voices on file', num(stats.profiles), null,
              'Voice profiles in speakers.db — these are the voices recognised across meetings.');

      // "Awaiting a name" is the library's own unnamed cluster count, which is
      // what the label says. It is not the same as the review queue: a cluster
      // can need a name and still not be scorable (see the tag tooltips).
      var awaiting = (typeof stats.unnamed === 'number') ? stats.unnamed : null;
      statRow('Awaiting a name', num(awaiting),
              awaiting ? 'font-variant-numeric:tabular-nums;color:var(--color-accent-2-700)'
                       : 'font-variant-numeric:tabular-nums',
              'Voices in these transcripts that nobody has named yet.');

      // The design's fourth line here was "Throughput 382× realtime". That
      // figure is a README measurement, not anything this server reports, so
      // the line is left out rather than hardcoded.

      aside.appendChild(list);

      var pending = review && review.counts && typeof review.counts.pending === 'number'
        ? review.counts.pending : null;

      var ingest = el('button', 'margin-top:28px', 'Ingest meetings');
      ingest.className = 'btn btn-primary btn-block';
      ingest.type = 'button';
      ingest.addEventListener('click', function () { ctx.go('ingest'); });
      aside.appendChild(ingest);

      var reviewBtn = el('button', null,
        pending ? 'Review ' + pending + (pending === 1 ? ' voice' : ' voices') : 'Review voices');
      reviewBtn.className = 'btn btn-secondary btn-block';
      reviewBtn.type = 'button';
      if (pending === 0) {
        reviewBtn.title = 'Nothing is waiting to be scored right now' +
          (review && review.reason ? ' (' + review.reason + ')' : '') + '.';
      }
      reviewBtn.addEventListener('click', function () { ctx.go('review'); });
      aside.appendChild(reviewBtn);

      grid.appendChild(aside);
      root.appendChild(grid);
    },

    destroy() {}
  };
})();
