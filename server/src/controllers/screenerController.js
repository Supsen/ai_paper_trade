const { spawn } = require('child_process');
const path = require('path');

const runScreener = async (req, res) => {
    console.log(`User ${req.user.email} is running the Python screener...`);

    // The new, simpler path to the script inside the server directory.
    const scriptPath = path.join(__dirname, '..', '..', 'ml', 'data_fetcher.py');
    console.log(`Executing script at: ${scriptPath}`);

    const pythonProcess = spawn('python', [scriptPath]);

    let scriptOutput = '';
    let scriptError = '';

    pythonProcess.stdout.on('data', (data) => {
        scriptOutput += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
        scriptError += data.toString();
        console.error(`Python Script Error: ${data}`);
    });

    pythonProcess.on('close', (code) => {
        console.log(`Python script finished with code ${code}`);

        if (code !== 0 || scriptError) {
            return res.status(500).json({ 
                msg: 'Failed to execute Python screener script.',
                error: scriptError 
            });
        }
        
        const recommendedCoins = scriptOutput.trim().split(',').filter(coin => coin);

        console.log('Successfully fetched coins:', recommendedCoins);
        res.json({ recommendedCoins });
    });
};

module.exports = {
    runScreener,
};