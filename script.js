// Chart colors
const COLORS = {
    'MICE (sklearn)': '#8884d8',
    'MICE (forest)': '#82ca9d',
    'KNN': '#ffc658'
};

// Process the raw CSV data
function processMetricsData(rawData) {
    // Filter out any null values
    rawData = rawData.filter(row => 
        row.Method && row.Metric && row.Feature && row.Value !== null
    );

    const metrics = {
        mean_diff: {},
        std_diff: {},
        ks_stat: {}
    };

    // Initialize the data structure
    rawData.forEach(row => {
        if (!metrics[row.Metric]) return;
        
        if (!metrics[row.Metric][row.Feature]) {
            metrics[row.Metric][row.Feature] = {
                feature: row.Feature,
                'MICE (sklearn)': 0,
                'MICE (forest)': 0,
                'KNN': 0
            };
        }
        metrics[row.Metric][row.Feature][row.Method] = row.Value;
    });

    return {
        meanDiff: Object.values(metrics.mean_diff),
        stdDiff: Object.values(metrics.std_diff),
        ksStats: Object.values(metrics.ks_stat)
    };
}

function createBarChart(elementId, data, title) {
    const ctx = document.getElementById(elementId).getContext('2d');
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.map(d => d.feature),
            datasets: Object.keys(COLORS).map(method => ({
                label: method,
                data: data.map(d => d[method]),
                backgroundColor: COLORS[method],
                borderColor: COLORS[method],
                borderWidth: 1
            }))
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: title
                }
            },
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

function createRadarChart(data) {
    const ctx = document.getElementById('distributionChart').getContext('2d');
    
    new Chart(ctx, {
        type: 'radar',
        data: {
            labels: data.map(d => d.feature),
            datasets: Object.keys(COLORS).map(method => ({
                label: method,
                data: data.map(d => d[method]),
                backgroundColor: `${COLORS[method]}44`,
                borderColor: COLORS[method],
                borderWidth: 2,
                pointBackgroundColor: COLORS[method]
            }))
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    beginAtZero: true,
                    angleLines: {
                        display: true
                    },
                    ticks: {
                        stepSize: 0.002
                    }
                }
            }
        }
    });
}

async function loadData() {
    try {
        const response = await fetch('data/imputation_metrics.csv');
        const csvText = await response.text();
        
        Papa.parse(csvText, {
            header: true,
            dynamicTyping: true,
            complete: (results) => {
                const processedData = processMetricsData(results.data);
                
                createBarChart(
                    'meanDiffChart', 
                    processedData.meanDiff, 
                    'Mean Difference by Feature'
                );
                
                createBarChart(
                    'stdDiffChart', 
                    processedData.stdDiff, 
                    'Standard Deviation Difference by Feature'
                );
                
                createRadarChart(processedData.ksStats);
            },
            error: (error) => {
                console.error('Error parsing CSV:', error.message);
            }
        });
    } catch (error) {
        console.error('Error loading file:', error.message);
    }
}

// Initialize the dashboard
document.addEventListener('DOMContentLoaded', loadData);