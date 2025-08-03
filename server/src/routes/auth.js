// routes/auth.js
const express = require('express');
const router = express.Router();
const rateLimit = require('express-rate-limit');
const { registerUser, loginUser } = require('../controllers/authController');

// Create a rate limiter for the login route
const loginLimiter = rateLimit({
	windowMs: 15 * 60 * 1000, // 15 minutes
	max: 10, // Limit each IP to 10 login requests per windowMs
	standardHeaders: true, // Return rate limit info in the `RateLimit-*` headers
	legacyHeaders: false, // Disable the `X-RateLimit-*` headers
    message: { msg: 'Too many login attempts, please try again after 15 minutes' },
});

// @route   POST /api/auth/register
router.post('/register', registerUser);

// @route   POST /api/auth/login
// Apply the rate limiter to this specific route
router.post('/login', loginLimiter, loginUser);

module.exports = router;