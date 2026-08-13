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

  /* Four states, because a filter has to distinguish what you can act on from
     what you cannot. The old classifier knew only named-vs-not, which put a
     meeting you can fix and one you cannot under the same "review" tag.

       done      every voice has a name
       review    voices unnamed AND scorable — the only actionable state
       blocked   voices unnamed but not scorable (no .emb.npz beside it)
       plain     no diarised voices to name at all
  */
  var STATES = {
    review:  { label: 'needs a name', cls: 'tag tag-accent-2' },
    blocked: { label: 'not scorable', cls: 'tag tag-neutral' },
    done:    { label: 'named',        cls: 'tag tag-neutral' },
    plain:   { label: 'transcribed',  cls: 'tag tag-neutral' }
  };

  function stateOf(m, blockedWhy) {
    var n = (typeof m.n_speakers === 'number') ? m.n_speakers : 0;
    var named = (typeof m.n_named === 'number') ? m.n_named : 0;
    if (!n) return { key: 'plain', why: 'This transcript has no diarised voices to name.' };
    if (named >= n) return { key: 'done', why: 'Every voice in this meeting has a name.' };
    if (blockedWhy) {
      return { key: 'blocked', why: blockedWhy };
    }
    return { key: 'review',
             why: (n - named) + ' of ' + n + ' voices in this meeting have no name yet.' };
  }

  var SORTS = [
    ['recent',   'Newest first',   function (a, b) { return (b.mtime || 0) - (a.mtime || 0); }],
    ['oldest',   'Oldest first',   function (a, b) { return (a.mtime || 0) - (b.mtime || 0); }],
    ['title',    'Title A–Z',      function (a, b) {
      return String(a.title || '').localeCompare(String(b.title || '')); }],
    ['longest',  'Longest first',  function (a, b) {
      return (b.duration_s || 0) - (a.duration_s || 0); }],
    ['unnamed',  'Most unnamed',   function (a, b) {
      return ((b.n_speakers || 0) - (b.n_named || 0)) - ((a.n_speakers || 0) - (a.n_named || 0)); }]
  ];

  var PAGE = 60;   // rows rendered before "Show more" — a long library must not
                   // build thousands of nodes to show you the first screenful

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

      /* One column now. The right rail held a running total of hours and three
         counters; a library is a thing you navigate, not a dashboard, and those
         numbers answered no question you could act on. What replaced them is the
         filter row: the same counts, but each one takes you to the rows it
         describes. */
      var main = el('div', 'max-width:900px');
      root.appendChild(main);

      main.appendChild(el('h1', 'font-size:40px;margin:0 0 4px', 'Library'));
      main.appendChild(el('p', 'font-size:13.5px;color:var(--color-neutral-700);max-width:52ch;margin:0 0 22px',
                          'Every meeting transcribed on this machine. A voice named once is recognised in all of them.'));

      // classify once; both the filter counts and the rows read from this
      var tagged = meetings.map(function (m) {
        var id = m && m.id != null ? String(m.id) : '';
        return { m: m, id: id, st: stateOf(m || {}, blocked[id]) };
      });
      var counts = { all: tagged.length, review: 0, blocked: 0, done: 0, plain: 0 };
      tagged.forEach(function (t) { counts[t.st.key] = (counts[t.st.key] || 0) + 1; });

      // Filter and sort live on ctx.state so opening a meeting and coming back
      // does not silently reset what you were looking at.
      ctx.state = ctx.state || {};
      if (!ctx.state.libFilter) ctx.state.libFilter = 'all';
      if (!ctx.state.libSort) ctx.state.libSort = 'recent';

      var controls = el('div', 'display:flex;align-items:baseline;gap:18px;flex-wrap:wrap;' +
        'padding-bottom:11px;border-bottom:1px solid color-mix(in srgb, var(--color-text) 14%, transparent);margin-bottom:4px');
      var filterRow = el('div', 'display:flex;align-items:baseline;gap:16px;flex-wrap:wrap');
      controls.appendChild(filterRow);

      var filterNodes = [];
      var FILTERS = [
        ['all', 'All'],
        ['review', 'Needs a name'],
        ['blocked', 'Not scorable'],
        ['done', 'Named'],
        ['plain', 'No voices']
      ];
      FILTERS.forEach(function (f) {
        var key = f[0], n = key === 'all' ? counts.all : (counts[key] || 0);
        // A filter that would show nothing is not offered, so the row only ever
        // contains routes that go somewhere.
        if (key !== 'all' && !n) return;
        var on = ctx.state.libFilter === key;
        var a = el('span',
          'font-size:12.5px;cursor:pointer;padding-bottom:3px;' +
          (on ? 'color:var(--color-text);border-bottom:2px solid var(--color-accent-600);'
              : 'color:var(--color-neutral-700);border-bottom:2px solid transparent;'));
        a.appendChild(document.createTextNode(f[1] + ' '));
        a.appendChild(el('span',
          'font-variant-numeric:tabular-nums;' + (on ? '' : 'color:var(--color-neutral-600);'), n));
        a.setAttribute('role', 'button');
        a.setAttribute('tabindex', '0');
        function pick() { ctx.state.libFilter = key; ctx.state.libShown = PAGE; paint(); }
        a.addEventListener('click', pick);
        a.addEventListener('keydown', function (ev) {
          if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); pick(); }
        });
        filterNodes.push({ key: key, node: a });
        filterRow.appendChild(a);
      });

      function styleFilter(key, node) {
        var on = ctx.state.libFilter === key;
        node.setAttribute('style',
          'font-size:12.5px;cursor:pointer;padding-bottom:3px;' +
          (on ? 'color:var(--color-text);border-bottom:2px solid var(--color-accent-600);'
              : 'color:var(--color-neutral-700);border-bottom:2px solid transparent;'));
      }

      var sortWrap = el('div', 'margin-left:auto;display:flex;align-items:baseline;gap:7px');
      sortWrap.appendChild(el('label', 'font-size:10.5px;letter-spacing:0.12em;text-transform:uppercase;color:var(--color-neutral-600)', 'Sort'));
      var sortSel = document.createElement('select');
      sortSel.className = 'input';
      sortSel.setAttribute('style', 'font-size:12.5px;padding:3px 6px;width:auto');
      SORTS.forEach(function (o) {
        var opt = document.createElement('option');
        opt.value = o[0]; opt.textContent = o[1];
        sortSel.appendChild(opt);
      });
      sortSel.value = ctx.state.libSort;
      sortSel.addEventListener('change', function () {
        ctx.state.libSort = sortSel.value; ctx.state.libShown = PAGE; paint();
      });
      sortWrap.appendChild(sortSel);
      controls.appendChild(sortWrap);
      main.appendChild(controls);

      var listBox = el('div', null);
      main.appendChild(listBox);

      var actions = el('div', 'display:flex;gap:10px;margin-top:30px');
      var ingest = el('button', 'flex:none', 'Ingest meetings');
      ingest.className = 'btn btn-primary'; ingest.type = 'button';
      ingest.addEventListener('click', function () { ctx.go('ingest'); });
      actions.appendChild(ingest);
      if (counts.review) {
        var rb = el('button', 'flex:none',
          'Review ' + counts.review + (counts.review === 1 ? ' meeting' : ' meetings'));
        rb.className = 'btn btn-secondary'; rb.type = 'button';
        rb.addEventListener('click', function () { ctx.go('review'); });
        actions.appendChild(rb);
      }
      main.appendChild(actions);

      function row(t) {
        var m = t.m, id = t.id;
        var open = function () {
          if (id) { ctx.state.meeting = id; ctx.go('transcript', { id: id }); }
        };
        var r = el('div', 'padding:15px 0;border-bottom:1px solid color-mix(in srgb, var(--color-text) 8%, transparent);cursor:pointer');
        r.setAttribute('role', 'button');
        r.setAttribute('tabindex', '0');
        r.addEventListener('click', open);
        r.addEventListener('keydown', function (ev) {
          if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); open(); }
        });

        var head = el('div', 'display:flex;align-items:baseline;gap:14px');
        var date = el('div', 'font-size:10.5px;letter-spacing:0.12em;text-transform:uppercase;color:var(--color-neutral-600);width:96px;flex:none;font-variant-numeric:tabular-nums',
                      shortDate(m));
        date.title = 'Transcript written ' + (m.date || '—') +
                     '. The store keeps no recording date, so this is the file\u2019s own.';
        head.appendChild(date);

        var current = ctx.state && ctx.state.meeting === id;
        head.appendChild(el('div',
          'font-size:20px;font-weight:600;line-height:1.2;letter-spacing:-0.01em;' +
          (current ? 'color:var(--color-accent-700);' : ''),
          m.title || id || 'Untitled'));

        var spec = STATES[t.st.key] || STATES.plain;
        var tagWrap = el('div', 'margin-left:auto;flex:none');
        var tag = el('span', null, spec.label);
        tag.className = spec.cls;
        tag.title = t.st.why;
        tagWrap.appendChild(tag);
        head.appendChild(tagWrap);
        r.appendChild(head);

        var meta = metaLine(ctx, m);
        if (meta) {
          r.appendChild(el('div', 'font-size:12.5px;color:var(--color-neutral-700);margin:5px 0 0 110px;font-variant-numeric:tabular-nums',
                           meta));
        }
        return r;
      }

      function paint() {
        clear(listBox);
        // underline follows the selection; each node carries its own key so this
        // cannot drift out of step with which filters were actually offered
        filterNodes.forEach(function (f) { styleFilter(f.key, f.node); });

        var want = ctx.state.libFilter;
        var rows = tagged.filter(function (t) { return want === 'all' || t.st.key === want; });
        var cmp = (SORTS.filter(function (o) { return o[0] === ctx.state.libSort; })[0] || SORTS[0])[2];
        rows = rows.slice().sort(cmp);

        if (!rows.length) {
          listBox.appendChild(el('p', 'font-size:13.5px;color:var(--color-neutral-700);margin:18px 0',
            meetings.length ? 'No meeting matches this filter.'
                            : 'No transcripts in this library yet. Ingest some audio and they will appear here.'));
          return;
        }

        var shown = Math.min(ctx.state.libShown || PAGE, rows.length);
        for (var i = 0; i < shown; i++) listBox.appendChild(row(rows[i]));

        if (shown < rows.length) {
          var more = el('div', 'padding:16px 0;display:flex;align-items:baseline;gap:12px');
          var btn = el('span', 'font-size:12.5px;cursor:pointer;color:var(--color-accent-700);border-bottom:1px solid var(--color-accent-300)',
                       'Show ' + Math.min(PAGE, rows.length - shown) + ' more');
          btn.setAttribute('role', 'button');
          btn.setAttribute('tabindex', '0');
          function grow() { ctx.state.libShown = shown + PAGE; paint(); }
          btn.addEventListener('click', grow);
          btn.addEventListener('keydown', function (ev) {
            if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); grow(); }
          });
          more.appendChild(btn);
          more.appendChild(el('span', 'font-size:12.5px;color:var(--color-neutral-600);font-variant-numeric:tabular-nums',
                              shown + ' of ' + rows.length));
          listBox.appendChild(more);
        }
      }

      ctx.state.libShown = ctx.state.libShown || PAGE;
      paint();

    },

    destroy() {}
  };
})();
