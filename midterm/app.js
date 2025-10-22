// ========================================
// SUPERSTORE RESPONSE CLASSIFIER
// TensorFlow.js Implementation
// ========================================

// Global variables
let trainData = null;
let testData = null;
let combinedData = null; // For EDA
let preprocessedTrainData = null;
let preprocessedTestData = null;
let model = null;
let trainingHistory = null;
let validationData = null;
let validationLabels = null;
let validationPredictions = null;
let testPredictions = null;
let testIds = null;

// Schema configuration for Superstore dataset
const TARGET_FEATURE = 'Response'; // Binary classification target
const ID_FEATURE = 'Id'; // Identifier to exclude from features

// Features to drop (useless for prediction)
const FEATURES_TO_DROP = ['Dt_Customer', 'Days_Customer', 'source'];

// Numerical features (will be scaled)
const NUMERICAL_FEATURES = [
    'Year_Birth', 'Income', 'Kidhome', 'Teenhome', 'Recency',
    'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts',
    'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases',
    'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
    'NumWebVisitsMonth', 'Complain'
];

// Categorical features (will be one-hot encoded)
const CATEGORICAL_FEATURES = ['Education', 'Marital_Status'];

// Preprocessing parameters (calculated from training data)
let preprocessingParams = {
    numericalMeans: {},
    numericalStds: {},
    categoricalModes: {},
    categoricalEncodings: {}
};

// ========================================
// DATA LOADING
// ========================================

async function loadData() {
    const trainFile = document.getElementById('train-file').files[0];
    const testFile = document.getElementById('test-file').files[0];
    
    if (!trainFile || !testFile) {
        alert('Please upload both training and test CSV files.');
        return;
    }
    
    const statusDiv = document.getElementById('data-status');
    statusDiv.innerHTML = '<p class="status-message">Loading data...</p>';
    
    try {
        // Load training data
        const trainText = await readFile(trainFile);
        trainData = parseCSV(trainText);
        
        // Load test data
        const testText = await readFile(testFile);
        testData = parseCSV(testText);
        
        // Add source column for EDA
        trainData.forEach(row => row.source = 'train');
        testData.forEach(row => row.source = 'test');
        
        // Combine for EDA
        combinedData = [...trainData, ...testData];
        
        statusDiv.innerHTML = `
            <p class="status-message">
                <strong>Data loaded successfully!</strong><br>
                Training: ${trainData.length} samples<br>
                Test: ${testData.length} samples<br>
                Combined for EDA: ${combinedData.length} samples
            </p>
        `;
        
        // Show preview
        statusDiv.innerHTML += '<h3>Training Data Preview (First 10 Rows)</h3>';
        const trainPreviewDiv = document.createElement('div');
        trainPreviewDiv.className = 'table-scroll';
        trainPreviewDiv.appendChild(createPreviewTable(trainData.slice(0, 10)));
        statusDiv.appendChild(trainPreviewDiv);
        
        statusDiv.innerHTML += '<h3>Test Data Preview (First 10 Rows)</h3>';
        const testPreviewDiv = document.createElement('div');
        testPreviewDiv.className = 'table-scroll';
        testPreviewDiv.appendChild(createPreviewTable(testData.slice(0, 10)));
        statusDiv.appendChild(testPreviewDiv);
        
        // Enable EDA button
        document.getElementById('eda-btn').disabled = false;
    } catch (error) {
        statusDiv.innerHTML = `<p class="error-message">Error loading data: ${error.message}</p>`;
        console.error(error);
    }
}

// Read file as text
function readFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = e => {
            let content = e.target.result;
            content = content.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
            if (content.charCodeAt(0) === 0xFEFF) {
                content = content.substring(1);
            }
            resolve(content);
        };
        reader.onerror = e => reject(new Error('Failed to read file'));
        reader.readAsText(file);
    });
}

// Parse CSV text to array of objects
function parseCSV(csvText) {
    const lines = csvText.split('\n').filter(line => line.trim() !== '');
    const headers = parseCSVLine(lines[0]);
    
    return lines.slice(1).map(line => {
        const values = parseCSVLine(line);
        const obj = {};
        headers.forEach((header, i) => {
            obj[header] = values[i] === '' ? null : values[i];
            if (!isNaN(obj[header]) && obj[header] !== null && obj[header] !== '') {
                obj[header] = parseFloat(obj[header]);
            }
        });
        return obj;
    });
}

// Parse CSV line with quote handling
function parseCSVLine(line) {
    const result = [];
    let current = '';
    let inQuotes = false;
    
    for (let i = 0; i < line.length; i++) {
        const char = line[i];
        
        if (char === '"') {
            if (inQuotes && line[i + 1] === '"') {
                current += '"';
                i++;
            } else {
                inQuotes = !inQuotes;
            }
        } else if (char === ',' && !inQuotes) {
            result.push(current.trim());
            current = '';
        } else {
            current += char;
        }
    }
    
    result.push(current.trim());
    return result;
}

// Create preview table
function createPreviewTable(data) {
    const table = document.createElement('table');
    
    // Header row
    const headerRow = document.createElement('tr');
    Object.keys(data[0]).forEach(key => {
        const th = document.createElement('th');
        th.textContent = key;
        headerRow.appendChild(th);
    });
    table.appendChild(headerRow);
    
    // Data rows
    data.forEach(row => {
        const tr = document.createElement('tr');
        Object.values(row).forEach(value => {
            const td = document.createElement('td');
            td.textContent = value !== null ? value : 'NULL';
            tr.appendChild(td);
        });
        table.appendChild(tr);
    });
    
    return table;
}

// ========================================
// EXPLORATORY DATA ANALYSIS
// ========================================

function performEDA() {
    if (!combinedData || combinedData.length === 0) {
        alert('Please load data first.');
        return;
    }
    
    const edaDiv = document.getElementById('eda-stats');
    edaDiv.innerHTML = '<h3>Dataset Statistics</h3>';
    
    // Calculate statistics
    const trainCount = trainData.length;
    const testCount = testData.length;
    const responseCount = trainData.filter(row => row[TARGET_FEATURE] === 1).length;
    const responseRate = (responseCount / trainCount * 100).toFixed(2);
    
    edaDiv.innerHTML += `
        <p><strong>Training samples:</strong> ${trainCount}</p>
        <p><strong>Test samples:</strong> ${testCount}</p>
        <p><strong>Response rate in training:</strong> ${responseCount}/${trainCount} (${responseRate}%)</p>
    `;
    
    // Missing values analysis
    let missingInfo = '<h4>Missing Values Percentage:</h4><ul>';
    const allFeatures = Object.keys(trainData[0]);
    allFeatures.forEach(feature => {
        const missingCount = trainData.filter(row => row[feature] === null || row[feature] === undefined).length;
        const missingPercent = (missingCount / trainData.length * 100).toFixed(2);
        if (parseFloat(missingPercent) > 0) {
            missingInfo += `<li><strong>${feature}:</strong> ${missingPercent}%</li>`;
        }
    });
    missingInfo += '</ul>';
    edaDiv.innerHTML += missingInfo;
    
    // Create visualizations using Plotly (in-page)
    createPlotlyVisualizations();
    
    // Enable preprocessing button
    document.getElementById('preprocess-btn').disabled = false;
}

// ========================================
// PLOTLY VISUALIZATIONS (IN-PAGE)
// ========================================

function createPlotlyVisualizations() {
    // 1. Categorical Variables by Source
    plotCategoricalBySource();
    
    // 2. Continuous Variables Distribution by Source (Histograms)
    plotContinuousHistogramsBySource();
    
    // 3. Continuous Variables Boxplots (Training Data)
    plotContinuousBoxplots();
    
    // 4. Categorical Variables vs Target (Training Data)
    plotCategoricalVsTarget();
    
    // 5. Continuous Variables vs Target (Training Data)
    plotContinuousVsTarget();
    
    // 6. Correlation Heatmap (Training Data - Numeric Features)
    plotCorrelationHeatmap();
}

// 1. Categorical Variables by Source
function plotCategoricalBySource() {
    const container = document.getElementById('plots-cat-by-source');
    container.innerHTML = ''; // Clear existing content
    
    CATEGORICAL_FEATURES.forEach(feature => {
        // Count by source
        const trainCounts = {};
        const testCounts = {};
        
        trainData.forEach(row => {
            const val = row[feature];
            if (val != null) {
                trainCounts[val] = (trainCounts[val] || 0) + 1;
            }
        });
        
        testData.forEach(row => {
            const val = row[feature];
            if (val != null) {
                testCounts[val] = (testCounts[val] || 0) + 1;
            }
        });
        
        // Get all unique categories
        const categories = [...new Set([...Object.keys(trainCounts), ...Object.keys(testCounts)])].sort();
        
        const trainValues = categories.map(cat => trainCounts[cat] || 0);
        const testValues = categories.map(cat => testCounts[cat] || 0);
        
        // Create individual plot div for each chart
        const plotDiv = document.createElement('div');
        plotDiv.className = 'plot-container';
        plotDiv.style.width = '100%';
        plotDiv.style.height = '450px';
        const uniqueId = `cat-by-source-${feature.replace(/[^a-zA-Z0-9]/g, '-')}`;
        plotDiv.id = uniqueId;
        container.appendChild(plotDiv);
        
        // Plotly traces
        const traces = [
            {
                x: categories,
                y: trainValues,
                name: 'Train',
                type: 'bar',
                marker: { color: '#667eea' }
            },
            {
                x: categories,
                y: testValues,
                name: 'Test',
                type: 'bar',
                marker: { color: '#764ba2' }
            }
        ];
        
        const layout = {
            title: {
                text: `${feature} Distribution by Source`,
                x: 0.5,
                xanchor: 'center'
            },
            xaxis: { title: feature },
            yaxis: { title: 'Count' },
            barmode: 'group',
            height: 400,
            margin: { t: 60, b: 60, l: 60, r: 60 }
        };
        
        const config = {
            responsive: true,
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: []
        };
        
        Plotly.newPlot(uniqueId, traces, layout, config);
    });
}

// 2. Continuous Variables Distribution by Source (Histograms)
function plotContinuousHistogramsBySource() {
    const container = document.getElementById('plots-cont-hist');
    container.innerHTML = '';
    
    // Select key continuous features for visualization
    const featuresForHist = ['Income', 'Year_Birth', 'Recency', 'MntWines', 'MntMeatProducts', 'NumWebPurchases'];
    
    featuresForHist.forEach(feature => {
        // Get values by source
        const trainValues = trainData
            .filter(row => row[feature] != null)
            .map(row => row[feature]);
        
        const testValues = testData
            .filter(row => row[feature] != null)
            .map(row => row[feature]);
        
        // Create individual plot div
        const plotDiv = document.createElement('div');
        plotDiv.className = 'plot-container';
        plotDiv.style.width = '100%';
        plotDiv.style.height = '450px';
        const uniqueId = `cont-hist-${feature.replace(/[^a-zA-Z0-9]/g, '-')}`;
        plotDiv.id = uniqueId;
        container.appendChild(plotDiv);
        
        // Plotly traces (overlaid histograms)
        const traces = [
            {
                x: trainValues,
                name: 'Train',
                type: 'histogram',
                opacity: 0.6,
                marker: { color: '#667eea' },
                nbinsx: 30
            },
            {
                x: testValues,
                name: 'Test',
                type: 'histogram',
                opacity: 0.6,
                marker: { color: '#764ba2' },
                nbinsx: 30
            }
        ];
        
        const layout = {
            title: {
                text: `${feature} Distribution by Source`,
                x: 0.5,
                xanchor: 'center'
            },
            xaxis: { title: feature },
            yaxis: { title: 'Frequency' },
            barmode: 'overlay',
            height: 400,
            margin: { t: 60, b: 60, l: 60, r: 60 }
        };
        
        const config = {
            responsive: true,
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: []
        };
        
        Plotly.newPlot(uniqueId, traces, layout, config);
    });
}

// 3. Continuous Variables Boxplots (Training Data)
function plotContinuousBoxplots() {
    const container = document.getElementById('plots-cont-box');
    container.innerHTML = '';
    
    const featuresForBox = ['Income', 'Year_Birth', 'Recency', 'MntWines', 'MntMeatProducts', 'NumWebPurchases'];
    
    featuresForBox.forEach(feature => {
        const values = trainData
            .filter(row => row[feature] != null)
            .map(row => row[feature]);
        
        // Create individual plot div
        const plotDiv = document.createElement('div');
        plotDiv.className = 'plot-container';
        plotDiv.style.width = '100%';
        plotDiv.style.height = '450px';
        const uniqueId = `cont-box-${feature.replace(/[^a-zA-Z0-9]/g, '-')}`;
        plotDiv.id = uniqueId;
        container.appendChild(plotDiv);
        
        const trace = {
            y: values,
            type: 'box',
            name: feature,
            marker: { color: '#667eea' },
            boxmean: 'sd'
        };
        
        const layout = {
            title: {
                text: `${feature} Boxplot (Training Data)`,
                x: 0.5,
                xanchor: 'center'
            },
            yaxis: { title: feature },
            height: 400,
            margin: { t: 60, b: 60, l: 60, r: 60 }
        };
        
        const config = {
            responsive: true,
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: []
        };
        
        Plotly.newPlot(uniqueId, [trace], layout, config);
    });
}

// 4. Categorical Variables vs Target (Training Data)
function plotCategoricalVsTarget() {
    const container = document.getElementById('plots-cat-vs-target');
    container.innerHTML = '';
    
    CATEGORICAL_FEATURES.forEach(feature => {
        // Calculate response rate by category
        const stats = {};
        
        trainData.forEach(row => {
            const cat = row[feature];
            const resp = row[TARGET_FEATURE];
            if (cat == null || resp == null) return;
            
            if (!stats[cat]) {
                stats[cat] = { total: 0, responded: 0 };
            }
            stats[cat].total += 1;
            if (resp === 1) {
                stats[cat].responded += 1;
            }
        });
        
        const categories = Object.keys(stats).sort();
        const responseCounts = categories.map(cat => stats[cat].responded);
        const noResponseCounts = categories.map(cat => stats[cat].total - stats[cat].responded);
        const responseRates = categories.map(cat => 
            stats[cat].total > 0 ? (stats[cat].responded / stats[cat].total * 100).toFixed(2) : 0
        );
        
        // Create individual plot div
        const plotDiv = document.createElement('div');
        plotDiv.className = 'plot-container';
        plotDiv.style.width = '100%';
        plotDiv.style.height = '450px';
        const uniqueId = `cat-vs-target-${feature.replace(/[^a-zA-Z0-9]/g, '-')}`;
        plotDiv.id = uniqueId;
        container.appendChild(plotDiv);
        
        // Stacked bar chart
        const traces = [
            {
                x: categories,
                y: noResponseCounts,
                name: 'No Response (0)',
                type: 'bar',
                marker: { color: '#ff6b6b' }
            },
            {
                x: categories,
                y: responseCounts,
                name: 'Response (1)',
                type: 'bar',
                marker: { color: '#4ecdc4' }
            }
        ];
        
        const layout = {
            title: {
                text: `${feature} vs Response (Training Data)`,
                x: 0.5,
                xanchor: 'center'
            },
            xaxis: { title: feature },
            yaxis: { title: 'Count' },
            barmode: 'stack',
            height: 400,
            margin: { t: 60, b: 60, l: 60, r: 60 },
            annotations: categories.map((cat, i) => ({
                x: cat,
                y: stats[cat].total,
                text: `${responseRates[i]}%`,
                showarrow: false,
                yshift: 10
            }))
        };
        
        const config = {
            responsive: true,
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: []
        };
        
        Plotly.newPlot(uniqueId, traces, layout, config);
    });
}

// 5. Continuous Variables vs Target (Training Data)
function plotContinuousVsTarget() {
    const container = document.getElementById('plots-cont-vs-target');
    container.innerHTML = '';
    
    const featuresForTarget = ['Income', 'Year_Birth', 'Recency', 'MntWines', 'MntMeatProducts', 'NumWebPurchases'];
    
    featuresForTarget.forEach(feature => {
        // Separate values by target
        const noResponseValues = trainData
            .filter(row => row[TARGET_FEATURE] === 0 && row[feature] != null)
            .map(row => row[feature]);
        
        const responseValues = trainData
            .filter(row => row[TARGET_FEATURE] === 1 && row[feature] != null)
            .map(row => row[feature]);
        
        // Create individual plot div
        const plotDiv = document.createElement('div');
        plotDiv.className = 'plot-container';
        plotDiv.style.width = '100%';
        plotDiv.style.height = '450px';
        const uniqueId = `cont-vs-target-${feature.replace(/[^a-zA-Z0-9]/g, '-')}`;
        plotDiv.id = uniqueId;
        container.appendChild(plotDiv);
        
        // Overlaid histograms
        const traces = [
            {
                x: noResponseValues,
                name: 'No Response (0)',
                type: 'histogram',
                opacity: 0.6,
                marker: { color: '#ff6b6b' },
                nbinsx: 30
            },
            {
                x: responseValues,
                name: 'Response (1)',
                type: 'histogram',
                opacity: 0.6,
                marker: { color: '#4ecdc4' },
                nbinsx: 30
            }
        ];
        
        const layout = {
            title: {
                text: `${feature} Distribution by Response (Training Data)`,
                x: 0.5,
                xanchor: 'center'
            },
            xaxis: { title: feature },
            yaxis: { title: 'Frequency' },
            barmode: 'overlay',
            height: 400,
            margin: { t: 60, b: 60, l: 60, r: 60 }
        };
        
        const config = {
            responsive: true,
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: []
        };
        
        Plotly.newPlot(uniqueId, traces, layout, config);
    });
}

// 6. Correlation Heatmap (Training Data - Numeric Features)
function plotCorrelationHeatmap() {
    const container = document.getElementById('plots-corr');
    container.innerHTML = '';
    
    // Select numeric features for correlation
    const numericFeatures = [...NUMERICAL_FEATURES, TARGET_FEATURE];
    
    // Calculate correlation matrix
    const correlationMatrix = calculateCorrelationMatrix(trainData, numericFeatures);
    
    const plotDiv = document.createElement('div');
    plotDiv.className = 'plot-container';
    plotDiv.style.width = '100%';
    plotDiv.style.height = '850px';
    plotDiv.id = 'corr-heatmap';
    container.appendChild(plotDiv);
    
    const trace = {
        z: correlationMatrix.values,
        x: correlationMatrix.features,
        y: correlationMatrix.features,
        type: 'heatmap',
        colorscale: 'RdBu',
        zmid: 0,
        text: correlationMatrix.values.map(row => 
            row.map(val => val.toFixed(2))
        ),
        texttemplate: '%{text}',
        textfont: { size: 10 },
        colorbar: { title: 'Correlation' }
    };
    
    const layout = {
        title: {
            text: 'Correlation Heatmap (Training Data - Numeric Features)',
            x: 0.5,
            xanchor: 'center'
        },
        xaxis: { title: '', tickangle: -45 },
        yaxis: { title: '' },
        height: 800,
        margin: { t: 100, b: 150, l: 150, r: 60 }
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: []
    };
    
    Plotly.newPlot('corr-heatmap', [trace], layout, config);
}

// Helper: Calculate correlation matrix
function calculateCorrelationMatrix(data, features) {
    const n = features.length;
    const matrix = Array(n).fill(0).map(() => Array(n).fill(0));
    
    // Get valid data for each feature
    const featureData = {};
    features.forEach(feature => {
        featureData[feature] = data
            .filter(row => row[feature] != null)
            .map(row => row[feature]);
    });
    
    // Calculate pairwise correlations
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            if (i === j) {
                matrix[i][j] = 1.0;
            } else {
                const feat1 = features[i];
                const feat2 = features[j];
                
                // Find common valid indices
                const pairs = [];
                data.forEach(row => {
                    if (row[feat1] != null && row[feat2] != null) {
                        pairs.push([row[feat1], row[feat2]]);
                    }
                });
                
                if (pairs.length > 1) {
                    matrix[i][j] = pearsonCorrelation(
                        pairs.map(p => p[0]),
                        pairs.map(p => p[1])
                    );
                } else {
                    matrix[i][j] = 0;
                }
            }
        }
    }
    
    return {
        features: features,
        values: matrix
    };
}

// Helper: Pearson correlation
function pearsonCorrelation(x, y) {
    const n = x.length;
    if (n === 0) return 0;
    
    const meanX = x.reduce((a, b) => a + b, 0) / n;
    const meanY = y.reduce((a, b) => a + b, 0) / n;
    
    let numerator = 0;
    let denomX = 0;
    let denomY = 0;
    
    for (let i = 0; i < n; i++) {
        const dx = x[i] - meanX;
        const dy = y[i] - meanY;
        numerator += dx * dy;
        denomX += dx * dx;
        denomY += dy * dy;
    }
    
    if (denomX === 0 || denomY === 0) return 0;
    
    return numerator / Math.sqrt(denomX * denomY);
}

// ========================================
// PREPROCESSING (ORIGINAL FROM UPLOADED FILE)
// ========================================

function preprocessData() {
    if (!trainData || !testData) {
        alert('Please load data first.');
        return;
    }
    
    const outputDiv = document.getElementById('preprocessing-output');
    outputDiv.innerHTML = '<p class="status-message">Preprocessing data...</p>';
    
    try {
        // Step 1: Remove extreme outliers
        const trainFiltered = removeExtremeOutliers(trainData);
        const testFiltered = removeExtremeOutliers(testData);
        
        outputDiv.innerHTML += `<p>After outlier removal: Train=${trainFiltered.length}, Test=${testFiltered.length}</p>`;
        
        // Step 2: Calculate preprocessing parameters from training data
        calculatePreprocessingParams(trainFiltered);
        
        // Step 3: Extract features and labels
        const trainFeatures = [];
        const trainLabels = [];
        const trainOriginalLabels = []; // For stratified split
        
        trainFiltered.forEach(row => {
            const features = extractFeatures(row);
            if (features !== null) {
                trainFeatures.push(features);
                trainLabels.push(row[TARGET_FEATURE]);
                trainOriginalLabels.push(row[TARGET_FEATURE]);
            }
        });
        
        const testFeatures = [];
        testIds = [];
        
        testFiltered.forEach(row => {
            const features = extractFeatures(row);
            if (features !== null) {
                testFeatures.push(features);
                testIds.push(row[ID_FEATURE]);
            }
        });
        
        // Step 4: Apply oversampling to balance training data
        const { oversampledFeatures, oversampledLabels } = applyOversampling(trainFeatures, trainLabels);
        
        outputDiv.innerHTML += `<p>After oversampling: ${oversampledFeatures.length} training samples</p>`;
        outputDiv.innerHTML += `<p>Class distribution: 0=${oversampledLabels.filter(l => l === 0).length}, 1=${oversampledLabels.filter(l => l === 1).length}</p>`;
        
        // Step 5: Convert to tensors and one-hot encode labels
        preprocessedTrainData = {
            features: tf.tensor2d(oversampledFeatures),
            labels: tf.tensor2d(oversampledLabels.map(label => label === 1 ? [0, 1] : [1, 0]))
        };
        
        preprocessedTestData = {
            features: tf.tensor2d(testFeatures),
            ids: testIds
        };
        
        outputDiv.innerHTML += `
            <p class="status-message">
                <strong>Preprocessing completed!</strong><br>
                Training features shape: [${preprocessedTrainData.features.shape}]<br>
                Training labels shape: [${preprocessedTrainData.labels.shape}]<br>
                Test features shape: [${preprocessedTestData.features.shape}]<br>
                Feature count: ${preprocessedTrainData.features.shape[1]}
            </p>
        `;
        
        // Enable model creation button
        document.getElementById('create-model-btn').disabled = false;
    } catch (error) {
        outputDiv.innerHTML = `<p class="error-message">Error during preprocessing: ${error.message}</p>`;
        console.error(error);
    }
}

// Remove extreme outliers
function removeExtremeOutliers(data) {
    return data.filter(row => {
        const yearBirth = row.Year_Birth;
        const income = row.Income;
        return (yearBirth == null || yearBirth > 1920) && (income == null || income < 100000);
    });
}

// Calculate preprocessing parameters from training data
function calculatePreprocessingParams(data) {
    // Calculate means and standard deviations for numerical features
    NUMERICAL_FEATURES.forEach(feature => {
        const values = data.map(row => row[feature]).filter(val => val !== null && !isNaN(val));
        const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
        const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
        const std = Math.sqrt(variance);
        
        preprocessingParams.numericalMeans[feature] = mean;
        preprocessingParams.numericalStds[feature] = std > 0 ? std : 1; // Avoid division by zero
    });
    
    // Calculate modes for categorical features
    CATEGORICAL_FEATURES.forEach(feature => {
        const values = data.map(row => row[feature]).filter(val => val !== null);
        const counts = {};
        values.forEach(val => counts[val] = (counts[val] || 0) + 1);
        const mode = Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);
        preprocessingParams.categoricalModes[feature] = mode;
        
        // Create encoding map for one-hot encoding
        const uniqueValues = [...new Set(values)].sort();
        preprocessingParams.categoricalEncodings[feature] = uniqueValues;
    });
}

// Extract and preprocess features from a single row
function extractFeatures(row) {
    const features = [];
    
    // Process numerical features (impute with mean, then standardize)
    NUMERICAL_FEATURES.forEach(feature => {
        let value = row[feature];
        if (value === null || isNaN(value)) {
            value = preprocessingParams.numericalMeans[feature];
        }
        // Standardize: (value - mean) / std
        const standardized = (value - preprocessingParams.numericalMeans[feature]) / preprocessingParams.numericalStds[feature];
        features.push(standardized);
    });
    
    // Process categorical features (impute with mode, then one-hot encode)
    CATEGORICAL_FEATURES.forEach(feature => {
        let value = row[feature];
        if (value === null) {
            value = preprocessingParams.categoricalModes[feature];
        }
        
        // One-hot encode
        const categories = preprocessingParams.categoricalEncodings[feature];
        categories.forEach(category => {
            features.push(value === category ? 1 : 0);
        });
    });
    
    return features;
}

// Apply random oversampling to balance classes
function applyOversampling(features, labels) {
    const class0Indices = [];
    const class1Indices = [];
    
    labels.forEach((label, i) => {
        if (label === 0) class0Indices.push(i);
        else class1Indices.push(i);
    });
    
    const majorityCount = Math.max(class0Indices.length, class1Indices.length);
    const minorityIndices = class0Indices.length < class1Indices.length ? class0Indices : class1Indices;
    const majorityIndices = class0Indices.length >= class1Indices.length ? class0Indices : class1Indices;
    
    // Oversample minority class
    const oversampledMinorityIndices = [];
    while (oversampledMinorityIndices.length < majorityCount) {
        const randomIndex = minorityIndices[Math.floor(Math.random() * minorityIndices.length)];
        oversampledMinorityIndices.push(randomIndex);
    }
    
    // Combine all indices
    const allIndices = [...majorityIndices, ...oversampledMinorityIndices];
    
    // Shuffle
    for (let i = allIndices.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [allIndices[i], allIndices[j]] = [allIndices[j], allIndices[i]];
    }
    
    // Create oversampled dataset
    const oversampledFeatures = allIndices.map(i => features[i]);
    const oversampledLabels = allIndices.map(i => labels[i]);
    
    return { oversampledFeatures, oversampledLabels };
}

// ========================================
// MODEL CREATION (ORIGINAL)
// ========================================

function createModel() {
    if (!preprocessedTrainData) {
        alert('Please preprocess data first.');
        return;
    }
    
    const summaryDiv = document.getElementById('model-summary');
    summaryDiv.innerHTML = '<p class="status-message">Creating model...</p>';
    
    try {
        const inputDim = preprocessedTrainData.features.shape[1];
        
        // Build model architecture (updated: second layer is 128 instead of 256)
        model = tf.sequential({
            layers: [
                tf.layers.dense({ inputShape: [inputDim], units: 256, activation: 'relu' }),
                tf.layers.batchNormalization(),
                tf.layers.dropout({ rate: 0.1 }),
                tf.layers.dense({ units: 128, activation: 'relu' }),
                tf.layers.batchNormalization(),
                tf.layers.dropout({ rate: 0.1 }),
                tf.layers.dense({ units: 2, activation: 'softmax' })
            ]
        });
        
        // Compile model
        const optimizer = tf.train.adam(0.002714707);
        model.compile({
            optimizer: optimizer,
            loss: tf.losses.softmaxCrossEntropy,
            metrics: ['categoricalAccuracy']
        });
        
        summaryDiv.innerHTML = '<p class="status-message"><strong>Model created successfully!</strong></p>';
        summaryDiv.innerHTML += '<h3>Model Summary</h3>';
        
        // Display model summary
        const summaryContainer = document.createElement('div');
        summaryContainer.style.backgroundColor = 'white';
        summaryContainer.style.padding = '10px';
        summaryContainer.style.borderRadius = '5px';
        summaryContainer.style.fontFamily = 'monospace';
        summaryContainer.style.fontSize = '12px';
        
        model.summary();
        
        // Create a text summary
        let summaryText = '<pre>';
        summaryText += `Total params: ${model.countParams()}\n`;
        summaryText += `Input shape: [${inputDim}]\n`;
        summaryText += `Output shape: [2]\n`;
        summaryText += '\nLayers:\n';
        model.layers.forEach((layer, i) => {
            summaryText += `${i + 1}. ${layer.name} - ${layer.getClassName()}\n`;
        });
        summaryText += '</pre>';
        
        summaryContainer.innerHTML = summaryText;
        summaryDiv.appendChild(summaryContainer);
        
        // Enable training button
        document.getElementById('train-btn').disabled = false;
    } catch (error) {
        summaryDiv.innerHTML = `<p class="error-message">Error creating model: ${error.message}</p>`;
        console.error(error);
    }
}

// ========================================
// MODEL TRAINING (ORIGINAL)
// ========================================

async function trainModel() {
    if (!model || !preprocessedTrainData) {
        alert('Please create model first.');
        return;
    }
    
    const statusDiv = document.getElementById('training-status');
    statusDiv.innerHTML = '<p class="status-message">Training model... This may take a few minutes.</p>';
    
    try {
        // Stratified train/validation split (80/20)
        const { trainFeatures, trainLabels, valFeatures, valLabels } = stratifiedSplit(
            preprocessedTrainData.features,
            preprocessedTrainData.labels,
            0.2
        );
        
        validationData = valFeatures;
        validationLabels = valLabels;
        
        statusDiv.innerHTML += `<p>Training samples: ${trainFeatures.shape[0]}, Validation samples: ${valFeatures.shape[0]}</p>`;
        
        // Training configuration (updated: 50 epochs, batch size 128)
        const epochs = 50;
        const batchSize = 128;
        
        // Early stopping and learning rate reduction callbacks (updated: patience=5)
        let bestValLoss = Infinity;
        let patience = 5;
        let patienceCounter = 0;
        let lrPatienceCounter = 0;
        let currentLr = 0.002714707;
        const minLr = 1e-5;
        
        // Setup tfjs-vis callbacks
        const metrics = ['loss', 'categoricalAccuracy', 'val_loss', 'val_categoricalAccuracy'];
        const container = { name: 'Training Progress', tab: 'Training' };
        const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);
        
        // Custom callback for early stopping and LR reduction
        const customCallback = {
            onEpochEnd: async (epoch, logs) => {
                const valLoss = logs.val_loss;
                
                // Early stopping logic (using validation loss)
                if (valLoss < bestValLoss) {
                    bestValLoss = valLoss;
                    patienceCounter = 0;
                    lrPatienceCounter = 0;
                } else {
                    patienceCounter++;
                    lrPatienceCounter++;
                    
                    // Reduce learning rate (updated: patience=5)
                    if (lrPatienceCounter >= 5 && currentLr > minLr) {
                        currentLr = Math.max(currentLr * 0.5, minLr);
                        model.optimizer.learningRate = currentLr;
                        lrPatienceCounter = 0;
                        console.log(`Reducing learning rate to ${currentLr}`);
                    }
                    
                    // Early stopping
                    if (patienceCounter >= patience) {
                        console.log(`Early stopping at epoch ${epoch + 1}`);
                        model.stopTraining = true;
                    }
                }
            }
        };
        
        // Train the model
        trainingHistory = await model.fit(trainFeatures, trainLabels, {
            epochs: epochs,
            batchSize: batchSize,
            validationData: [valFeatures, valLabels],
            callbacks: [fitCallbacks, customCallback],
            shuffle: true
        });
        
        statusDiv.innerHTML = `
            <p class="status-message">
                <strong>Training completed!</strong><br>
                Final validation loss: ${bestValLoss.toFixed(4)}<br>
                Total epochs: ${trainingHistory.epoch.length}
            </p>
        `;
        
        // Make predictions on validation set
        validationPredictions = model.predict(valFeatures);
        
        // Enable metrics slider and prediction button
        document.getElementById('threshold-slider').disabled = false;
        document.getElementById('threshold-slider').oninput = updateMetrics;
        document.getElementById('predict-btn').disabled = false;
        
        // Initial metrics calculation
        await updateMetrics();
        
    } catch (error) {
        statusDiv.innerHTML = `<p class="error-message">Error during training: ${error.message}</p>`;
        console.error(error);
    }
}

// Stratified split function
function stratifiedSplit(features, labels, testSize) {
    const featuresArray = features.arraySync();
    const labelsArray = labels.arraySync();
    
    // Separate indices by class
    const class0Indices = [];
    const class1Indices = [];
    
    labelsArray.forEach((label, i) => {
        if (label[1] === 1) class1Indices.push(i);
        else class0Indices.push(i);
    });
    
    // Shuffle indices
    const shuffleArray = (arr) => {
        for (let i = arr.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [arr[i], arr[j]] = [arr[j], arr[i]];
        }
        return arr;
    };
    
    shuffleArray(class0Indices);
    shuffleArray(class1Indices);
    
    // Split each class
    const class0Split = Math.floor(class0Indices.length * testSize);
    const class1Split = Math.floor(class1Indices.length * testSize);
    
    const valIndices = [...class0Indices.slice(0, class0Split), ...class1Indices.slice(0, class1Split)];
    const trainIndices = [...class0Indices.slice(class0Split), ...class1Indices.slice(class1Split)];
    
    // Create split datasets
    const trainFeatures = tf.tensor2d(trainIndices.map(i => featuresArray[i]));
    const trainLabels = tf.tensor2d(trainIndices.map(i => labelsArray[i]));
    const valFeatures = tf.tensor2d(valIndices.map(i => featuresArray[i]));
    const valLabels = tf.tensor2d(valIndices.map(i => labelsArray[i]));
    
    return { trainFeatures, trainLabels, valFeatures, valLabels };
}

// ========================================
// METRICS EVALUATION (ORIGINAL)
// ========================================

async function updateMetrics() {
    if (!validationPredictions || !validationLabels) return;
    
    const threshold = parseFloat(document.getElementById('threshold-slider').value);
    document.getElementById('threshold-value').textContent = threshold.toFixed(2);
    
    // Get predictions and true labels
    const predArray = await validationPredictions.array();
    const trueArray = await validationLabels.array();
    
    // Calculate confusion matrix
    let tp = 0, tn = 0, fp = 0, fn = 0;
    
    for (let i = 0; i < predArray.length; i++) {
        const predProb = predArray[i][1]; // Probability of class 1
        const prediction = predProb >= threshold ? 1 : 0;
        const actual = trueArray[i][1]; // One-hot encoded
        
        if (prediction === 1 && actual === 1) tp++;
        else if (prediction === 0 && actual === 0) tn++;
        else if (prediction === 1 && actual === 0) fp++;
        else if (prediction === 0 && actual === 1) fn++;
    }
    
    // Update confusion matrix display
    const cmDiv = document.getElementById('confusion-matrix');
    cmDiv.innerHTML = `
        <table style="width: 100%;">
            <tr><th></th><th>Pred Positive</th><th>Pred Negative</th></tr>
            <tr><th>Actual Positive</th><td style="background: #c8e6c9;">${tp}</td><td style="background: #ffcdd2;">${fn}</td></tr>
            <tr><th>Actual Negative</th><td style="background: #ffcdd2;">${fp}</td><td style="background: #c8e6c9;">${tn}</td></tr>
        </table>
    `;
    
    // Calculate performance metrics
    const precision = tp / (tp + fp) || 0;
    const recall = tp / (tp + fn) || 0;
    const f1 = 2 * (precision * recall) / (precision + recall) || 0;
    const accuracy = (tp + tn) / (tp + tn + fp + fn) || 0;
    
    // Update performance metrics display
    const metricsDiv = document.getElementById('performance-metrics');
    metricsDiv.innerHTML = `
        <p><strong>Accuracy:</strong> ${(accuracy * 100).toFixed(2)}%</p>
        <p><strong>Precision:</strong> ${precision.toFixed(4)}</p>
        <p><strong>Recall:</strong> ${recall.toFixed(4)}</p>
        <p><strong>F1 Score:</strong> ${f1.toFixed(4)}</p>
    `;
    
    // Plot ROC curve
    await plotROC(predArray, trueArray);
}

// Plot ROC curve
async function plotROC(predictions, trueLabels) {
    const thresholds = Array.from({ length: 100 }, (_, i) => i / 100);
    const rocData = [];
    
    thresholds.forEach(threshold => {
        let tp = 0, fn = 0, fp = 0, tn = 0;
        
        for (let i = 0; i < predictions.length; i++) {
            const predProb = predictions[i][1];
            const prediction = predProb >= threshold ? 1 : 0;
            const actual = trueLabels[i][1];
            
            if (actual === 1) {
                if (prediction === 1) tp++;
                else fn++;
            } else {
                if (prediction === 1) fp++;
                else tn++;
            }
        }
        
        const tpr = tp / (tp + fn) || 0;
        const fpr = fp / (fp + tn) || 0;
        
        rocData.push({ x: fpr, y: tpr });
    });
    
    // Calculate AUC
    let auc = 0;
    for (let i = 1; i < rocData.length; i++) {
        auc += (rocData[i].x - rocData[i-1].x) * (rocData[i].y + rocData[i-1].y) / 2;
    }
    
    // Plot ROC curve
    tfvis.render.linechart(
        { name: 'ROC Curve', tab: 'Evaluation' },
        { values: rocData },
        { 
            xLabel: 'False Positive Rate', 
            yLabel: 'True Positive Rate',
            width: 500,
            height: 400
        }
    );
    
    // Add AUC to metrics
    const metricsDiv = document.getElementById('performance-metrics');
    metricsDiv.innerHTML += `<p><strong>AUC:</strong> ${auc.toFixed(4)}</p>`;
}

// ========================================
// PREDICTION (ORIGINAL)
// ========================================

async function predict() {
    if (!model || !preprocessedTestData) {
        alert('Please train model first.');
        return;
    }
    
    const outputDiv = document.getElementById('prediction-output');
    outputDiv.innerHTML = '<p class="status-message">Making predictions...</p>';
    
    try {
        // Make predictions
        testPredictions = model.predict(preprocessedTestData.features);
        const predArray = await testPredictions.array();
        
        // Create results
        const results = preprocessedTestData.ids.map((id, i) => ({
            Id: id,
            Response: predArray[i][1] >= 0.5 ? 1 : 0,
            Probability: predArray[i][1]
        }));
        
        // Show preview
        outputDiv.innerHTML = '<h3>Prediction Results (First 10 Rows)</h3>';
        const previewDiv = document.createElement('div');
        previewDiv.className = 'table-scroll';
        previewDiv.appendChild(createPredictionTable(results.slice(0, 10)));
        outputDiv.appendChild(previewDiv);
        
        outputDiv.innerHTML += `<p class="status-message">Predictions completed! Total: ${results.length} samples</p>`;
        
        // Enable export and ground truth buttons
        document.getElementById('export-btn').disabled = false;
        document.getElementById('ground-truth-btn').disabled = false;
        
    } catch (error) {
        outputDiv.innerHTML = `<p class="error-message">Error during prediction: ${error.message}</p>`;
        console.error(error);
    }
}

// Create prediction table
function createPredictionTable(data) {
    const table = document.createElement('table');
    
    // Header
    const headerRow = document.createElement('tr');
    ['Id', 'Response', 'Probability'].forEach(header => {
        const th = document.createElement('th');
        th.textContent = header;
        headerRow.appendChild(th);
    });
    table.appendChild(headerRow);
    
    // Data rows
    data.forEach(row => {
        const tr = document.createElement('tr');
        ['Id', 'Response', 'Probability'].forEach(key => {
            const td = document.createElement('td');
            let value = row[key];
            if (key === 'Probability') {
                value = value.toFixed(4);
            }
            td.textContent = value;
            tr.appendChild(td);
        });
        table.appendChild(tr);
    });
    
    return table;
}

// ========================================
// EXPORT RESULTS (ORIGINAL)
// ========================================

async function exportResults() {
    if (!testPredictions || !preprocessedTestData) {
        alert('Please make predictions first.');
        return;
    }
    
    const statusDiv = document.getElementById('export-status');
    statusDiv.innerHTML = '<p class="status-message">Exporting results...</p>';
    
    try {
        const predArray = await testPredictions.array();
        
        // Create submission CSV
        let submissionCSV = 'Id,Response\n';
        preprocessedTestData.ids.forEach((id, i) => {
            const response = predArray[i][1] >= 0.5 ? 1 : 0;
            submissionCSV += `${id},${response}\n`;
        });
        
        // Create probabilities CSV
        let probabilitiesCSV = 'Id,Probability\n';
        preprocessedTestData.ids.forEach((id, i) => {
            probabilitiesCSV += `${id},${predArray[i][1].toFixed(6)}\n`;
        });
        
        // Download submission.csv
        const submissionBlob = new Blob([submissionCSV], { type: 'text/csv' });
        const submissionLink = document.createElement('a');
        submissionLink.href = URL.createObjectURL(submissionBlob);
        submissionLink.download = 'submission.csv';
        submissionLink.click();
        
        // Download probabilities.csv
        const probabilitiesBlob = new Blob([probabilitiesCSV], { type: 'text/csv' });
        const probabilitiesLink = document.createElement('a');
        probabilitiesLink.href = URL.createObjectURL(probabilitiesBlob);
        probabilitiesLink.download = 'probabilities.csv';
        probabilitiesLink.click();
        
        // Save model
        await model.save('downloads://superstore-tfjs-model');
        
        statusDiv.innerHTML = `
            <p class="status-message">
                <strong>Export completed!</strong><br>
                ✓ submission.csv (Kaggle submission format)<br>
                ✓ probabilities.csv (Prediction probabilities)<br>
                ✓ Model saved to browser downloads
            </p>
        `;
        
    } catch (error) {
        statusDiv.innerHTML = `<p class="error-message">Error during export: ${error.message}</p>`;
        console.error(error);
    }
}

// ========================================
// GROUND TRUTH CHECK (ORIGINAL)
// ========================================

async function checkGroundTruth() {
    const labeledTestFile = document.getElementById('labeled-test-file').files[0];
    
    if (!labeledTestFile) {
        alert('Please upload the labeled test CSV file.');
        return;
    }
    
    if (!testPredictions || !preprocessedTestData) {
        alert('Please make predictions first.');
        return;
    }
    
    const outputDiv = document.getElementById('ground-truth-output');
    outputDiv.innerHTML = '<p class="status-message">Loading labeled test data...</p>';
    
    try {
        // Load labeled test data
        const labeledText = await readFile(labeledTestFile);
        const labeledData = parseCSV(labeledText);
        
        // Get predictions
        const predArray = await testPredictions.array();
        
        // Create a map of predictions by ID
        const predMap = {};
        preprocessedTestData.ids.forEach((id, i) => {
            predMap[id] = {
                predicted: predArray[i][1] >= 0.5 ? 1 : 0,
                probability: predArray[i][1]
            };
        });
        
        // Match with ground truth
        let tp = 0, tn = 0, fp = 0, fn = 0;
        const matchedResults = [];
        
        labeledData.forEach(row => {
            const id = row.Id;
            const actual = row.Response;
            const pred = predMap[id];
            
            if (pred) {
                const predicted = pred.predicted;
                
                if (predicted === 1 && actual === 1) tp++;
                else if (predicted === 0 && actual === 0) tn++;
                else if (predicted === 1 && actual === 0) fp++;
                else if (predicted === 0 && actual === 1) fn++;
                
                matchedResults.push({
                    Id: id,
                    Actual: actual,
                    Predicted: predicted,
                    Probability: pred.probability,
                    Correct: predicted === actual ? '✓' : '✗'
                });
            }
        });
        
        // Calculate final metrics
        const accuracy = (tp + tn) / (tp + tn + fp + fn);
        const precision = tp / (tp + fp) || 0;
        const recall = tp / (tp + fn) || 0;
        const f1 = 2 * (precision * recall) / (precision + recall) || 0;
        
        outputDiv.innerHTML = `
            <h3>Ground Truth Evaluation Results</h3>
            <div class="metric-card">
                <h4>Confusion Matrix</h4>
                <table style="width: 100%;">
                    <tr><th></th><th>Pred Positive</th><th>Pred Negative</th></tr>
                    <tr><th>Actual Positive</th><td style="background: #c8e6c9;">${tp}</td><td style="background: #ffcdd2;">${fn}</td></tr>
                    <tr><th>Actual Negative</th><td style="background: #ffcdd2;">${fp}</td><td style="background: #c8e6c9;">${tn}</td></tr>
                </table>
            </div>
            <div class="metric-card">
                <h4>Final Performance Metrics</h4>
                <p><strong>Accuracy:</strong> ${(accuracy * 100).toFixed(2)}%</p>
                <p><strong>Precision:</strong> ${precision.toFixed(4)}</p>
                <p><strong>Recall:</strong> ${recall.toFixed(4)}</p>
                <p><strong>F1 Score:</strong> ${f1.toFixed(4)}</p>
                <p><strong>Total Samples:</strong> ${matchedResults.length}</p>
                <p><strong>Correct Predictions:</strong> ${tp + tn}</p>
                <p><strong>Incorrect Predictions:</strong> ${fp + fn}</p>
            </div>
        `;
        
        // Show sample results
        outputDiv.innerHTML += '<h3>Sample Results (First 20 Rows)</h3>';
        const sampleDiv = document.createElement('div');
        sampleDiv.className = 'table-scroll';
        sampleDiv.appendChild(createGroundTruthTable(matchedResults.slice(0, 20)));
        outputDiv.appendChild(sampleDiv);
        
    } catch (error) {
        outputDiv.innerHTML = `<p class="error-message">Error checking ground truth: ${error.message}</p>`;
        console.error(error);
    }
}

// Create ground truth comparison table
function createGroundTruthTable(data) {
    const table = document.createElement('table');
    
    // Header
    const headerRow = document.createElement('tr');
    ['Id', 'Actual', 'Predicted', 'Probability', 'Correct'].forEach(header => {
        const th = document.createElement('th');
        th.textContent = header;
        headerRow.appendChild(th);
    });
    table.appendChild(headerRow);
    
    // Data rows
    data.forEach(row => {
        const tr = document.createElement('tr');
        ['Id', 'Actual', 'Predicted', 'Probability', 'Correct'].forEach(key => {
            const td = document.createElement('td');
            let value = row[key];
            if (key === 'Probability') {
                value = value.toFixed(4);
            }
            if (key === 'Correct') {
                td.style.fontWeight = 'bold';
                td.style.color = value === '✓' ? 'green' : 'red';
            }
            td.textContent = value;
            tr.appendChild(td);
        });
        table.appendChild(tr);
    });
    
    return table;
}

