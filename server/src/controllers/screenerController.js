// --- 1. New File: controllers/screenerController.js ---
// This controller contains the logic to execute your Python script.

const { spawn } = require('child_process');
const path = require('path');

// @desc    Run the crypto screener Python script
// @route   GET /api/screener/run
// @access  Private (should be protected by auth middleware in a real app)
const runScreener = async (req, res) => {
    console.log('Received request to run Python screener...');

    // Define the path to your Python script.
    // We construct a path from the 'server' directory to the 'demo_1' directory.
    const scriptPath = path.join(__dirname, '..', '..', 'demo_1', 'step1', 'data_fetcher.py');
    console.log(`Executing script at: ${scriptPath}`);

    // Use spawn to run the Python script.
    // 'python' should be in your system's PATH.
    const pythonProcess = spawn('python', [scriptPath]);

    let scriptOutput = '';
    let scriptError = '';

    // Capture the output from the script's print() statements.
    pythonProcess.stdout.on('data', (data) => {
        scriptOutput += data.toString();
    });

    // Capture any errors that the script might throw.
    pythonProcess.stderr.on('data', (data) => {
        scriptError += data.toString();
        console.error(`Python Script Error: ${data}`);
    });

    // Handle the script finishing.
    pythonProcess.on('close', (code) => {
        console.log(`Python script finished with code ${code}`);

        if (code !== 0 || scriptError) {
            return res.status(500).json({ 
                msg: 'Failed to execute Python screener script.',
                error: scriptError 
            });
        }

        // Assuming the Python script prints a comma-separated list of coins.
        // e.g., "BTC/USDT,ETH/USDT,SOL/USDT,ADA/USDT,XRP/USDT"
        const recommendedCoins = scriptOutput.trim().split(',').filter(coin => coin);

        console.log('Successfully fetched coins:', recommendedCoins);
        res.json({ recommendedCoins });
    });
};

module.exports = {
    runScreener,
};