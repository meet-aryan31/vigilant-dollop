const fileInput = document.getElementById("file");
const drop = document.getElementById("drop");
const go = document.getElementById("go");
const status = document.getElementById("status");
const result = document.getElementById("result");
const chainEl = document.getElementById("chain");

fetch(API_BASE + "/api/chain")
  .then((r) => r.json())
  .then((d) => {
    chainEl.textContent = d.chain && d.chain.length
      ? "chain: " + d.chain.join(" → ")
      : "chain: not configured";
  })
  .catch(() => {
    chainEl.textContent = "chain: backend unreachable";
  });

fileInput.addEventListener("change", () => {
  const f = fileInput.files[0];
  drop.classList.toggle("has-file", !!f);
  drop.textContent = f ? f.name : "Click to choose an audio file";
  go.disabled = !f;
});

go.addEventListener("click", async () => {
  const f = fileInput.files[0];
  if (!f) return;
  go.disabled = true;
  status.textContent = "Processing… (transcribe → fraud check)";
  result.className = "";
  result.classList.remove("show");

  const fd = new FormData();
  fd.append("file", f);

  try {
    const res = await fetch(API_BASE + "/analyze", { method: "POST", body: fd });
    const data = await res.json();
    if (data.result) {
      renderVerdict(data.result);
      return;
    }
    result.classList.add("show");
    if (data.error) {
      result.classList.add("err");
      result.textContent = data.error;
      status.textContent = "Failed";
    } else if (data.response === null) {
      result.classList.add("null");
      result.textContent = "null (no text found)";
      status.textContent = "Done";
    } else {
      result.textContent = data.response;
      status.textContent = "Done";
    }
  } catch (e) {
    result.classList.add("show", "err");
    result.textContent = String(e);
    status.textContent = "Failed";
  } finally {
    go.disabled = false;
  }
});

function renderVerdict(r) {
  const titles = {
    fraud: ["fraud", "This recording is a fraudster call"],
    legit: ["legit", "This recording is not flagged as a fraudster call"],
    inconclusive: ["warn", "Could not determine a verdict"],
    empty: ["warn", "No speech detected in this recording."],
    error: ["err", "Analysis failed"],
  };
  const [cls, title] = titles[r.status] || ["err", "Analysis failed"];
  result.classList.add("show", cls);
  let head = title;
  if (typeof r.confidence === "number") {
    head += "  (model-stated confidence: " + Math.round(r.confidence * 100) + "%)";
  }
  const body = r.reason || r.raw || "";
  result.textContent = body ? head + "\n\n" + body : head;
  status.textContent = "Done";
}
