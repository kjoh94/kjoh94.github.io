---
layout: default
title: "Running OpenClaw on Android Termux (Galaxy Tab S8)"
date: 2026-02-02
---

# Running OpenClaw on Android Termux (Galaxy Tab S8)

I got [OpenClaw](https://github.com/openclaw/openclaw) running from source on my Galaxy Tab S8 using Termux + Ubuntu (proot-distro). It required a few workarounds for Android-specific restrictions. Here's what I ran into and how I fixed it.

## Environment

- Device: Samsung Galaxy Tab S8
- Termux with proot-distro (Ubuntu)
- Node.js 22+
- pnpm

## Building from Source

Following the README instructions:

```bash
pnpm install
pnpm ui:build
pnpm build
```

This completed without issues.

## Problem 1: `tsgo` panics on ARM/Termux

When running `pnpm openclaw onboard --install-daemon`, the dev runner tries to use `tsgo` (TypeScript-Go compiler) by default. On Termux, it crashes because bundled `lib.d.ts` files are missing:

```
panic: bundled: ...lib.d.ts does not exist; this executable may be misplaced
```

**Fix:** Tell the runner to use standard `tsc` instead:

```bash
export OPENCLAW_TS_COMPILER=tsc
```

## Problem 2: `uv_interface_addresses` permission error

Android restricts the `uv_interface_addresses` system call, which Node.js uses for `os.networkInterfaces()`. This crashes in two places:

### During onboarding

OpenClaw's own code calls `os.networkInterfaces()` in `src/infra/system-presence.ts` and `src/infra/tailnet.ts`.

**Fix:** Wrap the calls in try/catch to degrade gracefully. I patched both files to return empty results instead of crashing. (See the patch below.)

### During gateway startup (Bonjour/mDNS)

The `@homebridge/ciao` dependency (used for local network discovery) also calls `os.networkInterfaces()` internally. Since we can't patch the dependency, there's an env var to disable it:

```bash
export OPENCLAW_DISABLE_BONJOUR=1
```

Bonjour is only needed for LAN discovery of the gateway. If you're running a Telegram bot, you don't need it.

## Recommended `.bashrc` additions

```bash
export OPENCLAW_TS_COMPILER=tsc
export OPENCLAW_DISABLE_BONJOUR=1
```

## The Patch

Changes to `src/infra/system-presence.ts` -- add a safe wrapper around `os.networkInterfaces()`:

```typescript
function safeNetworkInterfaces(): NodeJS.Dict<os.NetworkInterfaceInfo[]> {
  try {
    return os.networkInterfaces();
  } catch {
    // Android/Termux restricts uv_interface_addresses; degrade gracefully.
    return {};
  }
}
```

Changes to `src/infra/tailnet.ts` -- catch the error and return empty results:

```typescript
let ifaces: NodeJS.Dict<os.NetworkInterfaceInfo[]>;
try {
  ifaces = os.networkInterfaces();
} catch {
  // Android/Termux restricts uv_interface_addresses; degrade gracefully.
  return { ipv4: [], ipv6: [] };
}
```

## Running the Gateway

After applying the fixes:

```bash
pnpm build
pnpm gateway:watch
```

The gateway starts successfully and the Telegram bot connects. The harmless warnings about `EISDIR` and missing `MEMORY.md` can be safely ignored.

## Summary

- **`tsgo` panic** — Missing bundled libs on ARM/Termux → `export OPENCLAW_TS_COMPILER=tsc`
- **`uv_interface_addresses` (onboarding)** — Android restricts network interface enumeration → Patch `system-presence.ts` and `tailnet.ts`
- **`uv_interface_addresses` (gateway)** — `@homebridge/ciao` calls `os.networkInterfaces()` → `export OPENCLAW_DISABLE_BONJOUR=1`

---

## Links

- Source patch: [fix/termux-network-interfaces](https://github.com/kjoh94/openclaw/tree/fix/termux-network-interfaces)
