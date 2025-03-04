// Base URL for your Render app
const BASE_URL = 'https://aviatorbotltsmpredict.onrender.com';

// Chart.js instances
let performanceChart, learningCurveChart, predictedActualChart;  // Add predictedActualChart

async function fetchPerformance() {
    try {
        const response = await fetch(`${BASE_URL}/performance`);
        if (!response.ok) throw new Error('Network response was not ok');
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error fetching performance:', error);
        return null;
    }
}

function updateCharts(data) {
    if (!data) return;

    // Performance Metrics (Bar Chart)
    if (!performanceChart) {
        const ctxPerformance = document.getElementById('performanceChart').getContext('2d');
        performanceChart = new Chart(ctxPerformance, {
            type: 'bar',
            data: {
                labels: ['MAE', '% Within ±1.0'],
                datasets: [{
                    label: 'Performance Metrics',
                    data: [data.performance.mae, data.performance.within_one_percent],
                    backgroundColor: ['rgba(75, 192, 192, 0.2)', 'rgba(255, 99, 132, 0.2)'],
                    borderColor: ['rgba(75, 192, 192, 1)', 'rgba(255, 99, 132, 1)'],
                    borderWidth: 1
                }]
            },
            options: {
                scales: {
                    y: {
                        beginAtZero: true,
                        title: { display: true, text: 'Value' }
                    }
                },
                plugins: {
                    title: { display: true, text: 'Model Performance' }
                }
            }
        });
    } else {
        performanceChart.data.datasets[0].data = [data.performance.mae, data.performance.within_one_percent];
        performanceChart.update();
    }

    // Learning Curve (Line Chart)
    if (!learningCurveChart) {
        const ctxLearning = document.getElementById('learningCurveChart').getContext('2d');
        learningCurveChart = new Chart(ctxLearning, {
            type: 'line',
            data: {
                labels: Array.from({ length: data.learning_curve.length }, (_, i) => i + 1),
                datasets: [{
                    label: 'Loss History',
                    data: data.learning_curve,
                    borderColor: 'rgba(54, 162, 235, 1)',
                    backgroundColor: 'rgba(54, 162, 235, 0.2)',
                    fill: true
                }]
            },
            options: {
                scales: {
                    x: { title: { display: true, text: 'Iterations' } },
                    y: { title: { display: true, text: 'Loss' } }
                },
                plugins: {
                    title: { display: true, text: 'Learning Curve' }
                }
            }
        });
    } else {
        learningCurveChart.data.labels = Array.from({ length: data.learning_curve.length }, (_, i) => i + 1);
        learningCurveChart.data.datasets[0].data = data.learning_curve;
        learningCurveChart.update();
    }

    // Predicted vs. Actual (Line Chart) - New Chart
    if (!predictedActualChart) {
        const ctxPredictedActual = document.getElementById('predictedActualChart').getContext('2d');
        predictedActualChart = new Chart(ctxPredictedActual, {
            type: 'line',
            data: {
                labels: Array.from({ length: data.performance.predicted_actual.length }, (_, i) => i + 1),
                datasets: [
                    {
                        label: 'Predicted',
                        data: data.performance.predicted_actual.map(pair => pair[0]),  // Predicted values
                        borderColor: 'rgba(255, 99, 132, 1)',  // Red for predicted (default, we'll change later)
                        backgroundColor: 'rgba(255, 99, 132, 0.2)',
                        fill: false
                    },
                    {
                        label: 'Actual',
                        data: data.performance.predicted_actual.map(pair => pair[1]),  // Actual values
                        borderColor: 'rgba(75, 192, 192, 1)',  // Green for actual (default)
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        fill: false
                    }
                ]
            },
            options: {
                scales: {
                    x: { title: { display: true, text: 'Prediction Index' } },
                    y: { title: { display: true, text: 'Payout Value' } }
                },
                plugins: {
                    title: { display: true, text: 'Predicted vs. Actual Payouts' }
                }
            }
        });
    } else {
        predictedActualChart.data.labels = Array.from({ length: data.performance.predicted_actual.length }, (_, i) => i + 1);
        predictedActualChart.data.datasets[0].data = data.performance.predicted_actual.map(pair => pair[0]);
        predictedActualChart.data.datasets[1].data = data.performance.predicted_actual.map(pair => pair[1]);
        predictedActualChart.update();
    }
}

async function updatePeriodically() {
    const data = await fetchPerformance();
    if (data) updateCharts(data);
    setTimeout(updatePeriodically, 30000); // Update every 30 seconds
}

// Start periodic updates when the page loads
document.addEventListener('DOMContentLoaded', updatePeriodically);