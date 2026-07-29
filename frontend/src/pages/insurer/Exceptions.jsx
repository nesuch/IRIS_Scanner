// Supervisory exception worklist — "who needs attention today", ranked.
//
// This inverts the dashboard question. Every row must be ACTIONABLE and
// SELF-EXPLAINING: the value, the comparison it failed, and a sentence a
// supervisor can carry into a meeting. Nothing here is a black-box score.
const SEV = {
  high: { label: 'High', cls: 'sev-high' },
  medium: { label: 'Medium', cls: 'sev-med' },
  low: { label: 'Low', cls: 'sev-low' },
};
const KIND = {
  threshold: { icon: 'fa-triangle-exclamation', label: 'Regulatory threshold' },
  loss: { icon: 'fa-arrow-trend-down', label: 'Loss-making' },
  outlier: { icon: 'fa-users-viewfinder', label: 'Peer outlier' },
  swing: { icon: 'fa-bolt', label: 'Sharp movement' },
  data_quality: { icon: 'fa-circle-question', label: 'Data quality' },
};

export default function Exceptions({ data, onSelect }) {
  if (!data) return null;
  const ex = data.exceptions || [];
  const s = data.summary || {};

  // Group by insurer: a supervisor works an ENTITY, not a metric. An insurer with
  // three findings is a different conversation from three insurers with one each.
  const byInsurer = [];
  const seen = new Map();
  ex.forEach((e) => {
    if (!seen.has(e.insurer_id)) {
      const g = { id: e.insurer_id, name: e.insurer, cls: e.class, items: [] };
      seen.set(e.insurer_id, g); byInsurer.push(g);
    }
    seen.get(e.insurer_id).items.push(e);
  });

  return (
    <div className="exc">
      <div className="exc-summary">
        <div className="exc-stat">
          <span className="exc-n">{data.insurers_flagged}</span>
          <span className="exc-l">of {data.insurers_total} insurers flagged</span>
        </div>
        {['high', 'medium', 'low'].filter((k) => s[k]).map((k) => (
          <div key={k} className={`exc-stat ${SEV[k].cls}`}>
            <span className="exc-n">{s[k]}</span>
            <span className="exc-l">{SEV[k].label} severity</span>
          </div>
        ))}
        <div className="exc-asof">as of {data.latest_fy}</div>
      </div>

      <div className="exc-list">
        {byInsurer.map((g) => (
          <div key={g.id} className="exc-ins">
            <div className="exc-ins-head">
              <button type="button" className="exc-name" onClick={() => onSelect && onSelect(g.id)}
                title="Open this insurer's 360 view">
                {g.name}
              </button>
              <span className="exc-cls">{g.cls}</span>
              <span className="exc-count">{g.items.length} finding{g.items.length === 1 ? '' : 's'}</span>
            </div>
            {g.items.map((e, i) => (
              <div key={i} className={`exc-row ${SEV[e.severity]?.cls || ''}`}>
                <i className={`fas ${KIND[e.kind]?.icon || 'fa-circle-info'}`} />
                <span className="exc-kind">{KIND[e.kind]?.label || e.kind}</span>
                <span className="exc-metric">{e.metric}</span>
                <span className="exc-why">
                  {e.why}
                  {e.stale && <span className="i360-stale" title={`latest figure is from ${e.fy}`}>{e.fy}</span>}
                </span>
              </div>
            ))}
          </div>
        ))}
        {!ex.length && <p className="iris-msg">No exceptions found.</p>}
      </div>

      <p className="exc-foot">
        Comparisons use the <strong>median of each insurer's own class</strong> — never the mean
        (one outlier drags it) and never across classes. Rates need at least 50,000 policies;
        below that a per-lakh figure is a filing artefact, not conduct. Where grievances exceed
        5% of policies the two lines are on different bases, so it is reported as a data-quality
        finding rather than ranked as conduct.
      </p>
    </div>
  );
}
