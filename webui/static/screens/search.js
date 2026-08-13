/* Search — design/Meetscribe.dc.html lines 235-257.
   Query box, speaker filter chips, hit list. A hit opens that moment in the
   transcript. Everything on this screen comes from GET /api/search; nothing is
   sampled from the design. */
(function () {
  'use strict';

  var MS = (window.MS = window.MS || {});
  MS.screens = MS.screens || {};

  /* Inline style strings carried over verbatim from the design. */
  var S = {
    page:   'max-width:820px',
    h1:     'font-size:40px;margin:0 0 20px',
    input:  'background:transparent;border:0;border-bottom:1px solid var(--color-text);' +
            'border-radius:0;font-size:27px;min-height:52px;padding:0 0 8px',
    filters:'display:flex;align-items:center;gap:10px;margin:18px 0 34px;flex-wrap:wrap',
    note:   'margin-left:auto;font-size:11.5px;color:var(--color-neutral-700);' +
            'font-variant-numeric:tabular-nums',
    hit:    'margin-bottom:24px;cursor:pointer;max-width:70ch',
    meta:   'display:flex;align-items:baseline;gap:12px;font-size:11px;letter-spacing:0.1em;' +
            'text-transform:uppercase;color:var(--color-neutral-600);margin-bottom:5px',
    at:     'font-variant-numeric:tabular-nums',
    text:   'font-size:15px;line-height:1.6',
    mark:   'color:var(--color-accent-700);border-bottom:2px solid var(--color-accent-300)',
    empty:  'font-size:15px;color:var(--color-neutral-700);font-style:italic',
    /* Not in the design: it never had a library without named voices. Set in the
       same key as the hit note so it reads as chrome, not as content. */
    aside:  'font-size:11.5px;color:var(--color-neutral-700);font-style:italic'
  };

  var IDLE_NOTE = 'Searches every transcript on this machine';
  var LIMIT = 80;
  var DEBOUNCE_MS = 180;

  function el(tag, style, text) {
    var n = document.createElement(tag);
    if (style) n.style.cssText = style;
    if (text != null) n.textContent = text;
    return n;
  }

  function plural(n, word) {
    return n + ' ' + word + (n === 1 ? '' : 's');
  }

  /* The right-hand note. Only ever repeats numbers the server gave us. */
  function noteFor(data) {
    if (!data || !data.q || data.reason === 'query-too-short') return IDLE_NOTE;
    if (data.reason === 'empty-library') return 'No transcripts in this library';
    var n = data.n_hits || 0;
    if (!n) return 'searched ' + plural(data.searched || 0, 'transcript');
    /* When the server hit its limit it stopped mid-library, so its meeting
       count is not the whole answer — don't quote a number we know is short. */
    if (data.truncated) return 'first ' + plural(n, 'line');
    return plural(n, 'line') + ' in ' + plural(data.meetings || 0, 'meeting');
  }

  function emptyCopy(data) {
    if (!data) return '';
    if (data.reason === 'empty-library') {
      return 'There are no transcripts in this library to search.';
    }
    /* Design copy, verbatim. */
    return 'No line in the library carries that.';
  }

  MS.screens.search = {
    title: 'Search',

    async render(root, ctx) {
      var self = this;
      var saved = (ctx.state && ctx.state.search) || {};

      var query = typeof saved.q === 'string' ? saved.q : '';
      var filters = Array.isArray(saved.filters) ? saved.filters.slice() : [];
      var token = 0;

      this._alive = true;
      this._debounce = null;
      this._focus = null;

      function remember() {
        if (ctx.state) ctx.state.search = { q: query, filters: filters.slice() };
      }

      var page = el('div', S.page);
      page.appendChild(el('h1', S.h1, 'Search'));

      var input = document.createElement('input');
      input.className = 'input';
      input.type = 'text';
      input.value = query;
      input.placeholder = 'A word, a phrase, a name';
      input.style.cssText = S.input;
      input.setAttribute('autocomplete', 'off');
      input.setAttribute('aria-label', 'Search every transcript');
      page.appendChild(input);

      var filterRow = el('div', S.filters);
      page.appendChild(filterRow);

      var results = el('div', null);
      page.appendChild(results);

      root.appendChild(page);

      /* ---- filter chips ------------------------------------------------ */
      function drawFilters(data) {
        filterRow.textContent = '';
        var speakers = (data && Array.isArray(data.speakers)) ? data.speakers : [];

        speakers.forEach(function (sp) {
          if (!sp || !sp.name) return;
          var on = filters.indexOf(sp.name) !== -1;
          var chip = el('div', 'cursor:pointer', sp.name);
          chip.className = 'tag ' + (on ? 'tag-accent' : 'tag-outline');
          chip.setAttribute('role', 'button');
          chip.setAttribute('tabindex', '0');
          chip.setAttribute('aria-pressed', on ? 'true' : 'false');
          function toggle() {
            var i = filters.indexOf(sp.name);
            if (i === -1) filters.push(sp.name);
            else filters.splice(i, 1);
            remember();
            run(0);
          }
          chip.addEventListener('click', toggle);
          chip.addEventListener('keydown', function (e) {
            if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); }
          });
          filterRow.appendChild(chip);
        });

        /* No named voices anywhere in the library means there is genuinely
           nothing to filter by — say so rather than showing empty chrome. */
        if (data && !speakers.length && data.reason !== 'empty-library') {
          filterRow.appendChild(el('div', S.aside,
            'No named voices in this library yet — nothing to filter by.'));
        }

        filterRow.appendChild(el('div', S.note, noteFor(data)));
      }

      /* ---- opening a hit ----------------------------------------------- */
      function open(h) {
        if (ctx.state) {
          ctx.state.meeting = h.meeting;
          ctx.state.meetingId = h.meeting;
          ctx.state.t = h.t;
          ctx.state.seek = h.t;
        }
        ctx.go('transcript', { id: h.meeting, meeting: h.meeting, t: h.t, seek: h.t });
      }

      /* ---- hit list ----------------------------------------------------- */
      function drawHits(data, err) {
        results.textContent = '';

        if (err) {
          results.appendChild(el('div', S.empty,
            'The library could not be searched: ' + err));
          return;
        }
        if (!data || !data.q || data.reason === 'query-too-short') return;

        var hits = Array.isArray(data.hits) ? data.hits : [];
        if (!hits.length) {
          results.appendChild(el('div', S.empty, emptyCopy(data)));
          return;
        }

        hits.forEach(function (h) {
          if (!h || (!h.hit && !h.pre && !h.post)) return;
          var row = el('div', S.hit);
          row.setAttribute('role', 'button');
          row.setAttribute('tabindex', '0');

          var meta = el('div', S.meta);
          meta.appendChild(el('span', null, h.title || h.meeting || ''));
          meta.appendChild(el('span', S.at, h.at || ''));
          /* The design colours the speaker by whether we know who it is:
             an unplaced cluster takes the second process ink. */
          var whoStyle = 'color:' + (h.named
            ? 'var(--color-neutral-600)'
            : 'var(--color-accent-2-700)');
          meta.appendChild(el('span', whoStyle, h.who || ''));
          row.appendChild(meta);

          var line = el('div', S.text);
          line.appendChild(document.createTextNode(h.pre || ''));
          line.appendChild(el('span', S.mark, h.hit || ''));
          line.appendChild(document.createTextNode(h.post || ''));
          row.appendChild(line);

          row.addEventListener('click', function () { open(h); });
          row.addEventListener('keydown', function (e) {
            if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(h); }
          });
          results.appendChild(row);
        });
      }

      /* ---- fetch -------------------------------------------------------- */
      function run(delay) {
        if (self._debounce) clearTimeout(self._debounce);
        var wait = typeof delay === 'number' ? delay : DEBOUNCE_MS;
        self._debounce = setTimeout(fire, wait);
      }

      async function fire() {
        var mine = ++token;
        var path = '/api/search?q=' + encodeURIComponent(query) +
                   '&limit=' + LIMIT +
                   filters.map(function (f) {
                     return '&speaker=' + encodeURIComponent(f);
                   }).join('');
        var data = null, err = null;
        try {
          data = await ctx.api(path);
        } catch (e) {
          err = (e && e.message) ? e.message : String(e);
        }
        if (!self._alive || mine !== token) return;
        drawFilters(data);
        drawHits(data, err);
      }

      input.addEventListener('input', function () {
        query = input.value;
        remember();
        run();
      });
      input.addEventListener('keydown', function (e) {
        if (e.key === 'Enter') { e.preventDefault(); run(0); }
        if (e.key === 'Escape' && input.value) {
          e.preventDefault();
          input.value = '';
          query = '';
          remember();
          run(0);
        }
      });

      /* Draw the resting state immediately so the screen is never blank while
         the first request is in flight. */
      drawFilters(null);
      this._focus = setTimeout(function () {
        if (self._alive) { input.focus(); input.select(); }
      }, 0);

      await fire();
    },

    destroy: function () {
      this._alive = false;
      if (this._debounce) clearTimeout(this._debounce);
      if (this._focus) clearTimeout(this._focus);
      this._debounce = this._focus = null;
    }
  };
})();
