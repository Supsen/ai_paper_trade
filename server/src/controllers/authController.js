// In a real application, you would query a proper database (like PostgreSQL, MongoDB, etc.)
// to find the user and check their password.
const users = [
    {
        email: 'zen@ilstu.edu',
        password: 'password123', // In a real app, passwords should be hashed!
        name: 'Zen'
    }
];

// @desc    Authenticate user & get token
// @route   POST /api/auth/login
// @access  Public
const loginUser = (req, res) => {
    // Get email and password from the request body
    const { email, password } = req.body;

    // --- Basic Validation ---
    if (!email || !password) {
        return res.status(400).json({ msg: 'Please enter all fields' });
    }

    // --- Find User in our "Database" ---
    const user = users.find(u => u.email === email);

    // --- Check Credentials ---
    if (!user || user.password !== password) {
        // We use a generic message to avoid telling an attacker whether the email exists
        return res.status(401).json({ msg: 'Invalid credentials' });
    }
    
    // --- Successful Login ---
    console.log(`User ${email} logged in successfully.`);
    
    // If credentials are correct, send back a success response with user data.
    // In a real-world app, you would generate and send a JSON Web Token (JWT) here.
    res.status(200).json({
        message: 'Login successful!',
        user: {
            email: user.email,
            name: user.name,
            initials: 'ZN'
        }
    });
};

module.exports = {
    loginUser,
};
