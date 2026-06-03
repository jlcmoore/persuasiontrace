# Frontend (Vue + Vite)

This directory contains the participant-facing web app used by the experiment
platform.

## What lives here

- `src/`: Vue components, routing, and UI state.
- `public/`: static assets served directly.
- `tests/`: frontend test files.

## Local workflow

From repository root:

```bash
make jsbuild
```

Or from this directory:

```bash
npm install
npm run build
```

Use the root [README.md](../README.md) for end-to-end setup (API + frontend)
and deployment flow.
