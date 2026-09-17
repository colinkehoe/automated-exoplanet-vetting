# Portfolio demo

A standalone page — `index.html` plus two JSON files, no build step, no
dependencies beyond Google Fonts. Drop it into a site and link or iframe it.

Regenerate the data after retraining or rebuilding the ranking:

```sh
uv run exovet export-demo            # writes data.json and curves.json here
python3 -m http.server -d portfolio-demo 8765   # then open localhost:8765
```

## Putting it on colinkehoe.github.io/portfolio

The portfolio is a Vite app whose `public/` directory is copied to the site root
verbatim, so the demo needs no build wiring.

**1. Copy the bundle in.** From this repo:

```sh
cp -r portfolio-demo/. ../portfolio/public/exovet/
```

It is then served at `/portfolio/exovet/`. The "back to portfolio" link is
relative (`../`), so it works locally and on Pages without configuration.

**2. Add a demo link to the project entry.** In `src/content.ts`, the `Project`
interface gains one optional field, and the exoplanet entry fills it in:

```ts
export interface Project {
  // …
  /** Optional live demo, relative to the site base. */
  demo?: string;
}
```

```ts
{
  title: "Automated exoplanet vetting",
  // …
  url: "https://github.com/colinkehoe/automated-exoplanet-vetting",
  demo: "exovet/",
}
```

**3. Render it.** In `src/components/Work.tsx`, inside `Detail`, beside the
existing repo link:

```tsx
{p.demo && (
  <a className="detail-link" href={import.meta.env.BASE_URL + p.demo}>
    Open the demo
  </a>
)}
```

`import.meta.env.BASE_URL` is `/portfolio/` in the deployed build and `/` when
running `npm run dev`, so the link is correct in both.

**4. Check before pushing.** `npm run dev`, open the Work section, follow the
link, and confirm the page loads its JSON (a file:// open will not — it needs to
be served).

## Notes

- The page is dark-only by design: it inherits the portfolio's palette and is
  meant to sit inside it.
- Chart colours are violet for evidence in favour and amber for evidence
  against — a pair that stays distinguishable for red/green colour blindness,
  which a green/magenta pair would not.
- `data.json` is ~670 KB and `curves.json` ~800 KB, both of which compress to
  roughly a third of that in transit.
