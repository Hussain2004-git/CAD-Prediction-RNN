
document.getElementById('uploadForm').onsubmit = async function (e) {
    e.preventDefault();

    let formData = new FormData(this);

    console.log("Sending file to backend...");

    let response = await fetch('/predict', {
        method: 'POST',
        body: formData
    });

    let result = await response.json();

    if (result.success) {
        console.log("Prediction Success:", result);
        document.getElementById('result').innerHTML = `
            <p>${result.success}</p>
            <p><strong>Accuracy:</strong> ${result.accuracy}</p>
            <p><strong>AUC-ROC:</strong> ${result.auc_roc}</p>
            <p><strong>F1 Score:</strong> ${result.f1_score}</p>
            <a href="${result.result_file}" download="predictions_result.csv">Download Predictions</a>
        `;
    } else {
        console.error("Error:", result.error);
        document.getElementById('result').innerHTML = `
            <p>Error: ${result.error}</p>
        `;
    }
};
