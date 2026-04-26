import mongoose from 'mongoose';

export const connectDB = async () => {
  try {
    console.log("MONGO_URI:", process.env.MONGO_URI); // 👈 ADD THIS LINE

    await mongoose.connect(process.env.MONGO_URI);

    console.log('MongoDB connected');
  } catch (error) {
    console.error('MongoDB connection error:', error.message);
    process.exit(1);
  }
};