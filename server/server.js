// server.js
const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');

// Load environment variables from .env file
dotenv.config();

const app = express();
const port = 3001;

app.use(cors());
app.use(express.json());

app.use('/api/auth', require('./src/routes/auth'));
// In the future, you could add a route for trades:
// app.use('/api/trades', require('./routes/trades'));

app.listen(port, () => {
    console.log(`Auth service with Prisma listening at http://localhost:${port}`);
});