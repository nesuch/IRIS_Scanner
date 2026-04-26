import mongoose from 'mongoose';

const userSchema = new mongoose.Schema(
  {
    email: {
      type: String,
      required: true,
      unique: true,
      lowercase: true,
      trim: true
    },
    password: {
      type: String,
      required: true
    },
    role: {
      type: String,
      enum: ['admin', 'analyst', 'viewer'],
      default: 'viewer'
    },
    refreshTokenHash: {
      type: String,
      default: null
    }
  },
  {
    timestamps: { createdAt: true, updatedAt: true }
  }
);

export default mongoose.model('User', userSchema);
