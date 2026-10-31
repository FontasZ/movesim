// Where the "Send a movement" and "Give feedback" buttons lead. Paste the two Tally form links
// here (Share → Link, e.g. "https://tally.so/r/abc123"). Until then the buttons say "coming soon".
window.MOVESIM_FORMS = {
  movement: "",  // the "Send a movement" form
  feedback: "",  // the "Give feedback" form; gets hidden fields movement and time
};

// Point every element with data-form="movement" or data-form="feedback" at its form.
// data-movement and data-time (optional) are passed to the form as hidden fields.
window.formLink = function (kind, fields = {}) {
  const base = window.MOVESIM_FORMS[kind];
  if (!base) return "";
  const url = new URL(base);
  for (const [k, v] of Object.entries(fields)) if (v) url.searchParams.set(k, v);
  return url.toString();
};

document.addEventListener("DOMContentLoaded", () => {
  for (const el of document.querySelectorAll("[data-form]")) {
    const href = window.formLink(el.dataset.form, { movement: el.dataset.movement, time: el.dataset.time });
    if (href) {
      el.href = href;
      el.target = "_blank";
      el.rel = "noopener";
    } else {
      el.removeAttribute("href");
      el.setAttribute("aria-disabled", "true");
      el.title = "Coming soon";
      el.classList.add("soon");
    }
  }
});
