// Base URL for your Render app
const BASE_URL = 'https://aviatorbotltsmpredict.onrender.com';

// Chart.js instances
let performanceChart, learningCurveChart, predictedActualChart;

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

async function fetchTrainingLog() {
    try {
        const response = await fetch(`${BASE_URL}/training-log`);
        if (!response.ok) throw new Error('Network response was not ok');
        const logData = await response.json();
        return logData;
    } catch (error) {
        console.error('Error fetching training log:', error);
        return [];
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
                    backgroundColor: ['rgba(75, 192, 192, 0.2)', 'rgba(75, 192, 192, 0.2)'],  // Both blue as requested
                    borderColor: ['rgba(75, 192, 192, 1)', 'rgba(75, 192, 192, 1)'],
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
                    borderColor: 'rgba(54, 162, 235, 1)',  // Blue for consistency
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

    // Predicted vs. Actual (Line Chart) - Using training_log.json data
    fetchTrainingLog().then(logData => {
        if (!predictedActualChart && logData.length > 0) {
            const ctxPredictedActual = document.getElementById('predictedActualChart').getContext('2d');
            predictedActualChart = new Chart(ctxPredictedActual, {
                type: 'line',
                data: {
                    labels: Array.from({ length: logData.length }, (_, i) => i + 1),
                    datasets: [
                        {
                            label: 'Predicted',
                            data: logData.map(entry => entry.predicted || 0),  // Use predicted values, default to 0 if null
                            borderColor: 'rgba(255, 99, 132, 1)',  // Red for predicted
                            backgroundColor: 'rgba(255, 99, 132, 0.2)',
                            fill: false,
                            tension: 0.1  // Smooth lines for better visualization
                        },
                        {
                            label: 'Actual',
                            data: logData.map(entry => entry.actual || 0),  // Use actual values, default to 0 if null
                            borderColor: 'rgba(75, 192, 192, 1)',  // Green for actual
                            backgroundColor: 'rgba(75, 192, 192, 0.2)',
                            fill: false,
                            tension: 0.1  // Smooth lines
                        }
                    ]
                },
                options: {
                    scales: {
                        x: { title: { display: true, text: 'Prediction Index' } },
                        y: { title: { display: true, text: 'Payout Value' }, beginAtZero: true }
                    },
                    plugins: {
                        title: { display: true, text: 'Predicted vs. Actual Payouts' }
                    },
                    maintainAspectRatio: false  // Allow chart to resize with CSS
                }
            });
        } else if (predictedActualChart && logData.length > 0) {
            predictedActualChart.data.labels = Array.from({ length: logData.length }, (_, i) => i + 1);
            predictedActualChart.data.datasets[0].data = logData.map(entry => entry.predicted || 0);
            predictedActualChart.data.datasets[1].data = logData.map(entry => entry.actual || 0);
            predictedActualChart.update();
        }
    }).catch(error => console.error('Error updating predicted vs. actual chart:', error));
}

async function updatePeriodically() {
    const data = await fetchPerformance();
    if (data) updateCharts(data);
    setTimeout(updatePeriodically, 30000); // Update every 30 seconds
}

document.addEventListener('DOMContentLoaded', updatePeriodically);