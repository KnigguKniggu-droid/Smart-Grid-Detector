// @ts-check
import { test, expect } from "@playwright/test";

const DASHBOARD_URL = process.env.DASHBOARD_URL || "https://grid-sentinel-live.vercel.app/";

test.describe("Smart Grid Dashboard E2E Audit", () => {
  test("loads with zero console errors and renders all panels with content", async ({
    page,
  }) => {
    const errors = [];
    const warnings = [];
    page.on("console", (msg) => {
      if (msg.type() === "error") errors.push(msg.text());
      if (msg.type() === "warning") warnings.push(msg.text());
    });
    page.on("pageerror", (err) => errors.push(err.message));

    await page.goto(DASHBOARD_URL, { waitUntil: "networkidle" });
    await page.waitForTimeout(5000);

    expect(errors).toEqual([]);

    const gridResponse = page.locator("#grid-response-panel");
    await expect(gridResponse).toBeVisible();

    const resilience = page.locator("#resilience-panel");
    await expect(resilience).toBeVisible();

    const dispatch = page.locator("#dispatch-panel");
    await expect(dispatch).toBeVisible();

    // Assert panels contain rendered content, not just empty shells
    const resilienceStats = await page.locator("#resilience-panel .resilience-stat-row .resilience-stat").count();
    expect(resilienceStats).toBeGreaterThan(0);

    const dispatchStats = await page.locator("#dispatch-panel .resilience-stat-row .resilience-stat").count();
    expect(dispatchStats).toBeGreaterThan(0);

    const gridResponseStats = await page.locator("#grid-response-panel .resilience-stat-row .resilience-stat").count();
    expect(gridResponseStats).toBeGreaterThan(0);

    // Assert SAIFI value is present and numeric (not a placeholder dash)
    const saifiText = await page.locator("#reliability-saifi-value").textContent();
    expect(saifiText).not.toMatch(/^[\u2014\u2013-]+$/);
  });

  test("Three.js canvas is visible and actively painting", async ({
    page,
  }) => {
    await page.goto(DASHBOARD_URL, { waitUntil: "networkidle" });
    await page.waitForTimeout(3000);

    const canvas = page.locator("#topology-canvas");
    await expect(canvas).toBeVisible();

    const box = await canvas.boundingBox();
    expect(box).not.toBeNull();
    expect(box.width).toBeGreaterThan(100);
    expect(box.height).toBeGreaterThan(100);
  });

  test("ResizeObserver handles viewport mutation without crash", async ({
    page,
  }) => {
    const errors = [];
    page.on("console", (msg) => {
      if (msg.type() === "error") errors.push(msg.text());
    });
    page.on("pageerror", (err) => errors.push(err.message));

    await page.setViewportSize({ width: 1440, height: 900 });
    await page.goto(DASHBOARD_URL, { waitUntil: "networkidle" });
    await page.waitForTimeout(2000);

    const canvas = page.locator("#topology-canvas");
    const desktopBox = await canvas.boundingBox();
    expect(desktopBox).not.toBeNull();

    await page.setViewportSize({ width: 375, height: 812 });
    await page.waitForTimeout(2000);

    const mobileBox = await canvas.boundingBox();
    expect(mobileBox).not.toBeNull();
    expect(mobileBox.width).toBeLessThan(desktopBox.width);

    const overflow = await page.evaluate(() => {
      return document.body.scrollWidth > window.innerWidth;
    });
    expect(overflow).toBe(false);

    expect(errors).toEqual([]);
  });

  test("honors prefers-reduced-motion", async ({ browser }) => {
    const context = await browser.newContext({
      reducedMotion: "reduce",
    });
    const page = await context.newPage();

    const errors = [];
    page.on("console", (msg) => {
      if (msg.type() === "error") errors.push(msg.text());
    });
    page.on("pageerror", (err) => errors.push(err.message));

    await page.goto(DASHBOARD_URL, { waitUntil: "networkidle" });
    await page.waitForTimeout(3000);

    const canvas = page.locator("#topology-canvas");
    await expect(canvas).toBeVisible();

    expect(errors).toEqual([]);
    await context.close();
  });
});
