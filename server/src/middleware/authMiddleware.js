// --- 1. New File: middleware/authMiddleware.js ---
// This middleware verifies the user's JSON Web Token (JWT).

const jwt = require('jsonwebtoken');
const { PrismaClient } = require('@prisma/client');
const prisma = new PrismaClient();

const protect = async (req, res, next) => {
    let token;

    // Check for the token in the Authorization header (e.g., "Bearer <token>")
    if (req.headers.authorization && req.headers.authorization.startsWith('Bearer')) {
        try {
            // Get token from header
            token = req.headers.authorization.split(' ')[1];

            // Verify the token using your JWT_SECRET
            const decoded = jwt.verify(token, process.env.JWT_SECRET);

            // Find the user by the ID from the token and attach them to the request object
            // Exclude the password from the user object for security
            req.user = await prisma.user.findUnique({
                where: { id: decoded.id },
                select: { id: true, email: true, name: true },
            });

            if (!req.user) {
                return res.status(401).json({ msg: 'Not authorized, user not found' });
            }

            next(); // Move on to the next function (the controller)
        } catch (error) {
            console.error(error);
            return res.status(401).json({ msg: 'Not authorized, token failed' });
        }
    }

    if (!token) {
        return res.status(401).json({ msg: 'Not authorized, no token' });
    }
};

module.exports = { protect };