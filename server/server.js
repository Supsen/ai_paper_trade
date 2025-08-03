// --- Import Required Packages ---
const express = require('express');
const cors = require('cors');

// --- Create the Express App ---
const app = express();
const port = 3001;

// --- Middleware ---
app.use(cors()); // Enable Cross-Origin Resource Sharing
app.use(express.json()); // Allow the server to understand JSON

// --- Connect Routes ---
// Any request to '/api/auth' will be handled by our auth routes file
app.use('/api/auth', require('./routes/auth'));

// --- Start the Server ---
app.listen(port, () => {
    console.log(`Auth service listening at http://localhost:${port}`);
});