// svglint configuration, run from pre-commit with `--config .github/.svglintrc.mjs`.
// Without a config svglint enables no rules, so everything it checks is listed here.

const SYMBOL_ELEMENTS = new Set([
  "svg",
  "style",
  "g",
  "path",
  "line",
  "polyline",
  "polygon",
  "rect",
  "circle",
  "ellipse",
  "text",
]);

// Ink must come from the <style> classes so the dark-mode override applies.
// stroke-dasharray stays allowed inline: it marks a dashed bounding box, not ink.
const SYMBOL_INK_ATTRS = [
  "style",
  "color",
  "fill",
  "fill-opacity",
  "stroke",
  "stroke-opacity",
  "stroke-width",
  "stroke-linecap",
  "stroke-linejoin",
  "opacity",
];

// The parser leaves entities encoded, so url(&quot;#id&quot;) has to match too.
const URL_REF = /url\(\s*(?:["']|&quot;|&apos;)?#([^"'&)\s]+)/g;

/** Every `#id` reference resolves, no id is defined twice, and no on* handlers. */
const resolvableIds = (reporter, $) => {
  reporter.name = "resolvable-ids";
  const ids = new Map();
  const refs = new Set();
  $.find("*").each((_, element) => {
    for (const [name, value] of Object.entries(element.attribs)) {
      if (name === "id") {
        ids.set(value, (ids.get(value) ?? 0) + 1);
      } else if (name === "href" || name === "xlink:href") {
        if (value.startsWith("#")) {
          refs.add(value.slice(1));
        }
      } else if (name.startsWith("on")) {
        reporter.error(`<${element.name}> has event handler "${name}"`);
      }
      for (const match of value.matchAll(URL_REF)) {
        refs.add(match[1]);
      }
    }
  });
  for (const match of $.find("style").text().matchAll(URL_REF)) {
    refs.add(match[1]);
  }
  for (const [id, count] of ids) {
    if (count > 1) {
      reporter.error(`id "${id}" is defined ${count} times`);
    }
  }
  for (const id of refs) {
    if (!ids.has(id)) {
      reporter.error(`reference to undefined id "#${id}"`);
    }
  }
};

/** House style for the GDSFactory+ schematic symbols in qpdk/symbols. */
const schematicSymbol = (reporter, $, ast, { filepath }) => {
  reporter.name = "schematic-symbol";
  if (!filepath.replaceAll("\\", "/").includes("/qpdk/symbols/")) {
    return;
  }
  const svg = $.find("svg").first();
  if (svg.length === 0) {
    return;
  }
  const attrs = Object.keys(svg.attr()).sort().join(" ");
  if (attrs !== "height viewBox width xmlns") {
    reporter.error(
      `root <svg> must have exactly xmlns, viewBox, width and height (got: ${attrs})`,
    );
  }
  const [minX, minY, width, height] = (svg.attr("viewBox") ?? "").split(
    /[\s,]+/,
  );
  if (minX !== "0" || minY !== "0") {
    reporter.error("viewBox must start at 0 0");
  }
  if (svg.attr("width") !== width || svg.attr("height") !== height) {
    reporter.error("width and height must match the viewBox size");
  }
  const styles = $.find("style");
  const css = styles.text();
  if (styles.length !== 1 || !svg.children().first().is("style")) {
    reporter.error("first child must be the one and only <style> block");
  }
  if (
    !/@media\s*\(\s*prefers-color-scheme\s*:\s*dark\s*\)\s*\{[^}]/.test(css)
  ) {
    reporter.error(
      "<style> needs a non-empty prefers-color-scheme: dark override",
    );
  }
  if (/(?<!-)color-scheme\s*:|background/.test(css)) {
    reporter.error("<style> must not set color-scheme or a background");
  }
  $.find("*").each((_, element) => {
    const name = element.name;
    if (!SYMBOL_ELEMENTS.has(name)) {
      reporter.error(`element <${name}> is not allowed in a symbol`);
    }
    for (const attr of SYMBOL_INK_ATTRS) {
      if (name !== "svg" && attr in element.attribs) {
        reporter.error(`<${name}> sets "${attr}" inline; use a style class`);
      }
    }
  });
};

/** @type {import('svglint').Config} */
export default {
  rules: {
    valid: true,
    elm: {
      svg: 1,
      script: false,
      foreignObject: false,
    },
    attr: [
      {
        "rule::selector": "svg",
        xmlns: "http://www.w3.org/2000/svg",
        viewBox: /^-?[\d.]+(?:[\s,]+-?[\d.]+){3}$/,
      },
    ],
    custom: [resolvableIds, schematicSymbol],
  },
};
