// --- 2. New File: routes/screener.js ---
// This file defines the API route for our new screener endpoint.

const express = require('express');
const router = express.Router();
const { runScreener } = require('../controllers/screenerController');
const { protect } = require('../middleware/authMiddleware'); // Import the middleware

// Defines the GET /api/screener/run endpoint
router.get('/run', protect, runScreener);

module.exports = router;