import { useEffect, useState } from "react";
import "./App.css";

const API_URL = (import.meta.env.VITE_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

// Form sections. Each field: name (API field name), label, hint
const sections = [
  {
    title: "Store and department",
    fields: [
      { name: "Store", label: "Store ID", min: 1, step: 1 },
      { name: "Dept", label: "Department ID", min: 1, step: 1 },
      { name: "Type", label: "Store type", options: ["A", "B", "C"] },
      { name: "Size", label: "Store size", hint: "in sq ft", min: 1, step: 1 },
    ],
  },
  {
    title: "Economy",
    fields: [
      { name: "Temperature", label: "Temperature", hint: "in °F" },
      { name: "Fuel_Price", label: "Fuel price", hint: "$ per gallon" },
      { name: "CPI", label: "Consumer Price Index" },
      { name: "Unemployment", label: "Unemployment rate", hint: "in %" },
    ],
  },
  {
    title: "Promotions",
    fields: [
      { name: "MarkDown1", label: "Markdown 1" },
      { name: "MarkDown2", label: "Markdown 2" },
      { name: "MarkDown3", label: "Markdown 3" },
      { name: "MarkDown4", label: "Markdown 4" },
      { name: "MarkDown5", label: "Markdown 5" },
    ],
  },
];

// Sample values (like the first row of the Walmart dataset)
const sampleForm = {
  Store: 1,
  Dept: 1,
  Type: "A",
  Size: 151315,
  Date: "2010-02-05",
  IsHoliday: false,
  Temperature: 42.31,
  Fuel_Price: 2.572,
  CPI: 211.096358,
  Unemployment: 8.106,
  MarkDown1: 0,
  MarkDown2: 0,
  MarkDown3: 0,
  MarkDown4: 0,
  MarkDown5: 0,
};

// Get the ISO week number from a date (same as pandas isocalendar().week)
function getIsoWeek(dateString) {
  const date = new Date(dateString + "T00:00:00");
  // The ISO week is decided by its Thursday
  const dayIndex = (date.getDay() + 6) % 7; // Monday = 0
  date.setDate(date.getDate() - dayIndex + 3); // Thursday of this week

  // 4 January is always in week 1
  const jan4 = new Date(date.getFullYear(), 0, 4);
  const jan4Index = (jan4.getDay() + 6) % 7;
  jan4.setDate(jan4.getDate() - jan4Index + 3); // Thursday of week 1

  const oneWeek = 7 * 24 * 60 * 60 * 1000;
  return 1 + Math.round((date - jan4) / oneWeek);
}

// Turn a backend validation error (422) into a readable message
function readableError(data) {
  if (Array.isArray(data.detail)) {
    return data.detail
      .map((item) => `${item.loc[item.loc.length - 1]}: ${item.msg}`)
      .join(", ");
  }
  return typeof data.detail === "string" ? data.detail : "Something went wrong.";
}

function formatMoney(value) {
  return value.toLocaleString("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 2,
  });
}

function App() {
  const [form, setForm] = useState(sampleForm);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [apiStatus, setApiStatus] = useState("checking");
  const [modelMetrics, setModelMetrics] = useState(null);

  // Load API status and the latest model metrics when the page opens
  useEffect(() => {
    async function loadDashboardData() {
      try {
        const [healthResponse, metricsResponse] = await Promise.all([
          fetch(`${API_URL}/health`),
          fetch(`${API_URL}/model-metrics`),
        ]);

        if (!healthResponse.ok) {
          throw new Error("Backend health check failed.");
        }

        setApiStatus("online");

        if (!metricsResponse.ok) {
          throw new Error("Could not load model metrics.");
        }

        const metrics = await metricsResponse.json();
        setModelMetrics(metrics);
      } catch {
        setApiStatus("offline");
      }
    }

    loadDashboardData();
  }, []);

  function handleChange(event) {
    const { name, value, type, checked } = event.target;
    setForm((previous) => ({
      ...previous,
      [name]: type === "checkbox" ? checked : value,
    }));
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setLoading(true);
    setResult(null);
    setError("");

    // Get Year, Month and Week from the date
    const year = Number(form.Date.slice(0, 4));
    const month = Number(form.Date.slice(5, 7));
    const week = getIsoWeek(form.Date);

    // Data sent to the API
    const payload = {
      IsHoliday: form.IsHoliday,
      Type: form.Type,
      Year: year,
      Month: month,
      Week: week,
    };
    sections
      .flatMap((section) => section.fields)
      .filter((field) => !field.options)
      .forEach((field) => {
        payload[field.name] = Number(form[field.name]);
      });

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(readableError(data));
      }

      setApiStatus("online");
      setResult({ ...data, date: form.Date, week });
      // Keep the latest 5 predictions
      setHistory((previous) =>
        [
          {
            id: Date.now(),
            store: data.store,
            department: data.department,
            date: form.Date,
            value: data.predicted_weekly_sales,
          },
          ...previous,
        ].slice(0, 5)
      );
    } catch (err) {
      if (err.message.includes("Failed to fetch")) {
        setApiStatus("offline");
        setError("Could not connect to the backend. Check that FastAPI is running.");
      } else {
        setError(err.message);
      }
    } finally {
      setLoading(false);
    }
  }

  const statusText = {
    checking: "Checking API...",
    online: "API connected",
    offline: "API offline",
  }[apiStatus];

  return (
    <main className="app">
      <header className="top">
        <div>
          <h1>Weekly sales forecast</h1>
          <p className="muted">
            Enter the store, department and date. The Random Forest model
            predicts sales for that week.
          </p>
        </div>
        <span className={`status ${apiStatus}`}>{statusText}</span>
      </header>

      <div className="layout">
        <form className="form" onSubmit={handleSubmit}>
          {sections.map((section) => (
            <fieldset key={section.title}>
              <legend>{section.title}</legend>
              <div className="grid">
                {section.fields.map((field) => (
                  <label key={field.name}>
                    <span className="label-text">
                      {field.label}
                      {field.hint && <small> ({field.hint})</small>}
                    </span>
                    {field.options ? (
                      <select
                        name={field.name}
                        value={form[field.name]}
                        onChange={handleChange}
                      >
                        {field.options.map((option) => (
                          <option key={option} value={option}>
                            Type {option}
                          </option>
                        ))}
                      </select>
                    ) : (
                      <input
                        type="number"
                        name={field.name}
                        value={form[field.name]}
                        onChange={handleChange}
                        min={field.min}
                        step={field.step || "any"}
                        required
                      />
                    )}
                  </label>
                ))}
              </div>
            </fieldset>
          ))}

          <fieldset>
            <legend>Week</legend>
            <div className="grid">
              <label>
                <span className="label-text">Week ending date</span>
                <input
                  type="date"
                  name="Date"
                  value={form.Date}
                  onChange={handleChange}
                  min="2010-01-01"
                  max="2035-12-31"
                  required
                />
              </label>
              <label className="check">
                <input
                  type="checkbox"
                  name="IsHoliday"
                  checked={form.IsHoliday}
                  onChange={handleChange}
                />
                <span>Holiday week</span>
              </label>
            </div>
          </fieldset>

          <div className="actions">
            <button type="submit" className="primary" disabled={loading}>
              {loading ? "Predicting..." : "Predict weekly sales"}
            </button>
            <button
              type="button"
              className="secondary"
              onClick={() => {
                setForm(sampleForm);
                setResult(null);
                setError("");
              }}
            >
              Reset
            </button>
          </div>

          {error && (
            <p className="error" role="alert">
              {error}
            </p>
          )}
        </form>

        <aside className="side">
          <section className="receipt" aria-live="polite">
            <h2>Predicted sales</h2>
            {result ? (
              <>
                <p className="amount">{formatMoney(result.predicted_weekly_sales)}</p>
                <dl>
                  <div>
                    <dt>Store</dt>
                    <dd>{result.store}</dd>
                  </div>
                  <div>
                    <dt>Department</dt>
                    <dd>{result.department}</dd>
                  </div>
                  <div>
                    <dt>Week</dt>
                    <dd>
                      {result.week} ({result.date})
                    </dd>
                  </div>
                </dl>
              </>
            ) : (
              <p className="empty">
                Fill in the form and click "Predict weekly sales". The result will appear here.
              </p>
            )}
            <dl className="model">
              <div>
                <dt>Model</dt>
                <dd>{modelMetrics?.model || "Random Forest Regressor"}</dd>
              </div>
              <div>
                <dt>Validation R²</dt>
                <dd>
                  {typeof modelMetrics?.r2_score === "number"
                    ? modelMetrics.r2_score.toFixed(4)
                    : "Not available"}
                </dd>
              </div>
              {modelMetrics?.walk_forward_validation && (
                <div>
                  <dt>Walk-forward mean R² ({modelMetrics.walk_forward_validation.n_splits} folds)</dt>
                  <dd>{modelMetrics.walk_forward_validation.mean_r2.toFixed(4)}</dd>
                </div>
              )}
              {modelMetrics?.walk_forward_validation && (
                <div>
                  <dt>Fold R² scores</dt>
                  <dd>
                    {modelMetrics.walk_forward_validation.folds
                      .map((fold) => fold.r2_score.toFixed(4))
                      .join(" / ")}
                  </dd>
                </div>
              )}
            </dl>
          </section>

          {history.length > 0 && (
            <section className="history">
              <h2>Recent predictions</h2>
              <ul>
                {history.map((item) => (
                  <li key={item.id}>
                    <span>
                      Store {item.store}, Dept {item.department}
                      <small>{item.date}</small>
                    </span>
                    <strong>{formatMoney(item.value)}</strong>
                  </li>
                ))}
              </ul>
            </section>
          )}
        </aside>
      </div>

      <footer>College ML project. Predictions are estimates, not guaranteed sales.</footer>
    </main>
  );
}

export default App;
