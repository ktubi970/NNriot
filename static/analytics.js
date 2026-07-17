(() => {
  "use strict";

  let analysisContext = null;
  const form = document.getElementById("analytics-form");
  const input = document.getElementById("analytics-message");
  const submitButton = form.querySelector('button[type="submit"]');
  const chat = document.getElementById("analytics-chat");
  const warnings = document.getElementById("analytics-warnings");
  const evidenceList = document.getElementById("evidence-list");
  const activeCohort = document.getElementById("active-cohort");
  const freshness = document.getElementById("dataset-freshness");

  function textElement(tag, className, value) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    element.textContent = String(value ?? "Non disponible");
    return element;
  }

  function addMessage(role, value) {
    const article = document.createElement("article");
    article.className = `chat-message ${role}-message`;
    article.appendChild(textElement("p", "", value));
    chat.appendChild(article);
    chat.scrollTop = chat.scrollHeight;
  }

  function showWarnings(items) {
    warnings.replaceChildren();
    for (const warning of items || []) {
      const code = warning && warning.code ? warning.code : "ANALYTICS_ERROR";
      warnings.appendChild(
        textElement("p", "warning-item", `Analyse non publiée : ${code}`),
      );
    }
  }

  function addDefinition(list, term, value) {
    list.appendChild(textElement("dt", "", term));
    list.appendChild(textElement("dd", "", value));
  }

  function renderEvidence(packets) {
    evidenceList.replaceChildren();
    for (const packet of packets || []) {
      const details = document.createElement("details");
      details.className = "evidence-card";
      const metric = packet.estimate?.metric || "métrique";
      const strength = packet.recommendation?.strength || "descriptif";
      details.appendChild(textElement("summary", "", `${metric} — ${strength}`));
      const list = document.createElement("dl");
      list.className = "evidence-grid";
      addDefinition(
        list,
        "Échantillon",
        `${packet.sample?.matches ?? "—"} matchs · ${packet.sample?.players ?? "—"} joueurs`,
      );
      addDefinition(
        list,
        "Effet",
        `${packet.estimate?.value ?? "—"} ${packet.estimate?.unit ?? ""}`,
      );
      const interval = packet.estimate?.confidence_interval_95;
      addDefinition(
        list,
        "Intervalle à 95 %",
        Array.isArray(interval) ? interval.join(" → ") : "Non disponible",
      );
      addDefinition(list, "q-value", packet.estimate?.q_value ?? "Non disponible");
      const folds = packet.robustness?.temporal_folds || [];
      addDefinition(
        list,
        "Robustesse temporelle",
        folds.map((fold) => fold.direction).join(", ") || "Non disponible",
      );
      addDefinition(
        list,
        "Limites",
        (packet.limitations || []).join(", ") || "Aucune limite supplémentaire",
      );
      addDefinition(list, "Evidence ID", packet.evidence_id || "Non disponible");
      addDefinition(
        list,
        "Dataset",
        packet.provenance?.dataset_version || "Non disponible",
      );
      details.appendChild(list);
      evidenceList.appendChild(details);
    }
  }

  function renderCohort(packets) {
    const cohort = packets && packets[0] ? packets[0].cohort : null;
    if (!cohort) return;
    const regions = (cohort.regions || []).join(", ") || "toutes régions éligibles";
    const patches = (cohort.patches || []).join(", ") || "patch par défaut";
    const widened = cohort.widened ? " · fenêtre élargie" : "";
    activeCohort.textContent =
      `Cohorte : ${cohort.mode} · file ${(cohort.queue_ids || []).join(", ")} · ` +
      `${regions} · patchs ${patches}${widened}`;
  }

  async function ask(message) {
    addMessage("user", message);
    showWarnings([]);
    submitButton.disabled = true;
    form.setAttribute("aria-busy", "true");
    try {
      const response = await fetch("/api/analytics/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, context: analysisContext, filters: {} }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.code || "INVALID_REQUEST");
      addMessage("assistant", data.answer || "Analyse non disponible.");
      if (data.analysis_request && data.analysis_request.intent) {
        analysisContext = data.analysis_request;
      }
      showWarnings(data.warnings);
      renderEvidence(data.evidence);
      renderCohort(data.evidence);
    } catch (error) {
      const code = error instanceof Error ? error.message : "NETWORK_ERROR";
      showWarnings([{ code }]);
    } finally {
      submitButton.disabled = false;
      form.removeAttribute("aria-busy");
      input.focus();
    }
  }

  async function loadFreshness() {
    try {
      const response = await fetch("/api/analytics/health");
      const data = await response.json();
      if (response.ok && data.ok) {
        freshness.dataset.state = "fresh";
        freshness.textContent = `Données à jour · dataset ${data.dataset_version || "validé"}`;
      } else {
        freshness.dataset.state = "stale";
        freshness.textContent =
          `Analyse indisponible · ${data.code || "ANALYTICS_DB_STALE"}`;
      }
    } catch (_error) {
      freshness.dataset.state = "stale";
      freshness.textContent = "État des données indisponible";
    }
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const message = input.value.trim();
    if (!message) return;
    input.value = "";
    ask(message);
  });

  for (const button of document.querySelectorAll("#suggested-questions button")) {
    button.addEventListener("click", () => {
      input.value = button.textContent.trim();
      form.requestSubmit();
    });
  }

  loadFreshness();
})();
