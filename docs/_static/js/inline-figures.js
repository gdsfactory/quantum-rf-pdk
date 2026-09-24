document.addEventListener("DOMContentLoaded", () => {
  for (const image of document.querySelectorAll("img.qpdk-inline-figure")) {
    fetch(image.src)
      .then((response) => {
        if (!response.ok)
          throw new Error(`Figure request failed: ${response.status}`);
        return response.text();
      })
      .then((source) => {
        const svg = new DOMParser().parseFromString(
          source,
          "image/svg+xml",
        ).documentElement;
        if (svg.localName !== "svg") return;
        svg.classList.add("qpdk-figure");
        svg.setAttribute("role", "img");
        svg.setAttribute("aria-label", image.alt);
        image.replaceWith(document.importNode(svg, true));
      })
      .catch((error) => console.error(error));
  }
});
