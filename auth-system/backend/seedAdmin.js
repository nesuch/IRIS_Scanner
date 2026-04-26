import dotenv from 'dotenv';
import bcrypt from 'bcryptjs';
import { connectDB } from './config/db.js';
import User from './models/User.js';

dotenv.config();

const seedAdmin = async () => {
  await connectDB();

  const adminEmail = (process.env.ADMIN_EMAIL || '').toLowerCase().trim();
  const adminPassword = process.env.ADMIN_SEED_PASSWORD;

  if (!adminEmail || !adminPassword) {
    console.error('ADMIN_EMAIL and ADMIN_SEED_PASSWORD are required');
    process.exit(1);
  }

  const existing = await User.findOne({ email: adminEmail });
  if (existing) {
    existing.role = 'admin';
    await existing.save();
    console.log('Existing user promoted/confirmed as admin:', adminEmail);
    process.exit(0);
  }

  const hash = await bcrypt.hash(adminPassword, 12);
  await User.create({ email: adminEmail, password: hash, role: 'admin' });
  console.log('Admin created:', adminEmail);
  process.exit(0);
};

seedAdmin().catch((e) => {
  console.error('Admin seeding failed:', e.message);
  process.exit(1);
});
