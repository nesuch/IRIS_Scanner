import jwt from 'jsonwebtoken';
import User from '../models/User.js';

export const requireAuth = async (req, res, next) => {
  try {
    const accessCookie = process.env.ACCESS_COOKIE_NAME || 'iris_access_token';
    const token = req.cookies?.[accessCookie];

    if (!token) {
      return res.status(401).json({ message: 'Unauthorized: missing access token' });
    }

    const payload = jwt.verify(token, process.env.JWT_SECRET);
    const user = await User.findById(payload.sub).select('-password -refreshTokenHash');

    if (!user) {
      return res.status(401).json({ message: 'Unauthorized: user not found' });
    }

    req.user = user;
    next();
  } catch (error) {
    if (error.name === 'TokenExpiredError') {
      return res.status(401).json({ message: 'Access token expired' });
    }
    return res.status(401).json({ message: 'Invalid access token' });
  }
};
