document.addEventListener("DOMContentLoaded", () => {
  const figures = new Set([
    "cpw-cross-section.svg",
    "dispersive-readout.svg",
    "fabrication-variation.svg",
    "pulse-leakage.svg",
    "quarter-wave-resonator.svg",
    "transmon-energy.svg",
  ]);
  for (const image of document.querySelectorAll('img[src*="_images/"]')) {
    if (!figures.has(new URL(image.src).pathname.split("/").pop())) continue;
    fetch(image.src)
      .then((response) => {
        if (!response.ok) throw new Error(`Figure request failed: ${response.status}`);
        return response.text();
      })
      .then((source) => {
        const svg = new DOMParser().parseFromString(source, "image/svg+xml").documentElement;
        if (svg.localName !== "svg") return;
        svg.classList.add("qpdk-figure");
        svg.setAttribute("role", "img");
        svg.setAttribute("aria-label", image.alt);
        image.replaceWith(document.importNode(svg, true));
      })
      .catch((error) => console.error(error));
  }
});
