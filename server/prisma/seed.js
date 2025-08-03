// Save this file as: prisma/seed.js
const { PrismaClient } = require('@prisma/client');
const bcrypt = require('bcryptjs');

// Initialize the Prisma Client
const prisma = new PrismaClient();

async function main() {
  console.log('🌱 Starting database seed...');

  // --- Define Your Default User Data ---
  // You can change these values to whatever you'd like.
  const defaultUser = {
      name: 'Test',
      email: 'cen3203510sen@gmail.com',
      password: '12345678', // This will be securely hashed
  };
  
  // --- Check if the user already exists ---
  const existingUser = await prisma.user.findUnique({
    where: { email: defaultUser.email },
  });

  if (existingUser) {
    console.log(`✅ User '${defaultUser.email}' already exists. Skipping.`);
  } else {
    // --- Hash the password ---
    const salt = await bcrypt.genSalt(10);
    const hashedPassword = await bcrypt.hash(defaultUser.password, salt);

    // --- Create the user in the database ---
    const newUser = await prisma.user.create({
      data: {
        name: defaultUser.name,
        email: defaultUser.email,
        password: hashedPassword,
      },
    });
    console.log(`👍 Successfully created new user: ${newUser.name} (${newUser.email})`);
  }
  
  console.log('🌱 Database seeding finished.');
}

// --- Execute the main function ---
main()
  .catch((e) => {
    console.error('❌ An error occurred while seeding the database:', e);
    process.exit(1);
  })
  .finally(async () => {
    // Close the database connection gracefully
    await prisma.$disconnect();
  });
