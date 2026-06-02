// Register Chart.js components + the datalabels plugin once for the whole app.
import {
  Chart, ArcElement, LineElement, BarElement, PointElement,
  LinearScale, CategoryScale, Tooltip, Legend, Filler,
} from 'chart.js';
import ChartDataLabels from 'chartjs-plugin-datalabels';

Chart.register(
  ArcElement, LineElement, BarElement, PointElement,
  LinearScale, CategoryScale, Tooltip, Legend, Filler, ChartDataLabels,
);

// Datalabels is opt-in per chart via options.plugins.datalabels; default off so
// charts that don't configure it stay clean.
Chart.defaults.plugins.datalabels = { display: false };

export { Chart };
