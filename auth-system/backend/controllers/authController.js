import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';
import { validationResult } from 'express-validator';
import User from '../models/User.js';
import { generateAccessToken, generateRefreshToken } from '../utils/tokens.js';

const isProd = process.env.NODE_ENV === 'production';
const accessCookieName = process.env.ACCESS_COOKIE_NAME || 'iris_access_token';
const refreshCookieName = process.env.REFRESH_COOKIE_NAME || 'iris_refresh_token';

const cookieBase = {
  httpOnly: true,
  secure: isProd,
  sameSite: isProd ? 'none' : 'lax',
  path: '/'
};

const setAuthCookies = (res, accessToken, refreshToken) => {
  res.cookie(accessCookieName, accessToken, {
    ...cookieBase,
    maxAge: 15 * 60 * 1000
  });

  res.cookie(refreshCookieName, refreshToken, {
    ...cookieBase,
    maxAge: 7 * 24 * 60 * 60 * 1000
  });
};

const clearAuthCookies = (res) => {
  res.clearCookie(accessCookieName, { ...cookieBase, maxAge: 0 });
  res.clearCookie(refreshCookieName, { ...cookieBase, maxAge: 0 });
};

export const registerUser = async (req, res) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(400).json({ message: 'Validation error', errors: errors.array() });
  }

  try {
    const { email, password } = req.body;
    const normalizedEmail = email.toLowerCase().trim();

    const existing = await User.findOne({ email: normalizedEmail });
    if (existing) {
      return res.status(409).json({ message: 'Email already registered' });
    }

    const hashedPassword = await bcrypt.hash(password, 12);

    // Admin is never user-selectable. Either seeded or hardcoded owner email.
    const ownerEmail = (process.env.ADMIN_EMAIL || '').toLowerCase().trim();
    const role = normalizedEmail === ownerEmail && ownerEmail ? 'admin' : 'viewer';

    const user = await User.create({
      email: normalizedEmail,
      password: hashedPassword,
      role
    });

    return res.status(201).json({
      message: 'User registered successfully',
      user: { id: user._id, email: user.email, role: user.role, createdAt: user.createdAt }
    });
  } catch (error) {
    return res.status(500).json({ message: 'Server error during registration' });
  }
};

export const loginUser = async (req, res) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(400).json({ message: 'Validation error', errors: errors.array() });
  }

  try {
    const { email, password } = req.body;
    const user = await User.findOne({ email: email.toLowerCase().trim() });

    if (!user) {
      return res.status(401).json({ message: 'Invalid credentials' });
    }

    const matches = await bcrypt.compare(password, user.password);
    if (!matches) {
      return res.status(401).json({ message: 'Invalid credentials' });
    }

    const accessToken = generateAccessToken(user);
    const refreshToken = generateRefreshToken(user);
    const refreshTokenHash = await bcrypt.hash(refreshToken, 12);

    user.refreshTokenHash = refreshTokenHash;
    await user.save();

    setAuthCookies(res, accessToken, refreshToken);

    return res.status(200).json({
      message: 'Login successful',
      user: { id: user._id, email: user.email, role: user.role }
    });
  } catch (error) {
    return res.status(500).json({ message: 'Server error during login' });
  }
};

export const refreshToken = async (req, res) => {
  try {
    const token = req.cookies?.[refreshCookieName];
    if (!token) {
      return res.status(401).json({ message: 'Missing refresh token' });
    }

    let payload;
    try {
      payload = jwt.verify(token, process.env.REFRESH_TOKEN_SECRET);
    } catch (error) {
      clearAuthCookies(res);
      if (error.name === 'TokenExpiredError') {
        return res.status(401).json({ message: 'Refresh token expired' });
      }
      return res.status(401).json({ message: 'Invalid refresh token' });
    }

    const user = await User.findById(payload.sub);
    if (!user || !user.refreshTokenHash) {
      clearAuthCookies(res);
      return res.status(401).json({ message: 'Unauthorized refresh request' });
    }

    const validStoredToken = await bcrypt.compare(token, user.refreshTokenHash);
    if (!validStoredToken) {
      clearAuthCookies(res);
      return res.status(401).json({ message: 'Refresh token revoked or rotated' });
    }

    // Rotate refresh token
    const newAccessToken = generateAccessToken(user);
    const newRefreshToken = generateRefreshToken(user);
    user.refreshTokenHash = await bcrypt.hash(newRefreshToken, 12);
    await user.save();

    setAuthCookies(res, newAccessToken, newRefreshToken);

    return res.status(200).json({
      message: 'Token refreshed',
      user: { id: user._id, email: user.email, role: user.role }
    });
  } catch (error) {
    return res.status(500).json({ message: 'Server error during token refresh' });
  }
};

export const logoutUser = async (req, res) => {
  try {
    const token = req.cookies?.[refreshCookieName];

    if (token) {
      try {
        const payload = jwt.verify(token, process.env.REFRESH_TOKEN_SECRET);
        const user = await User.findById(payload.sub);
        if (user) {
          user.refreshTokenHash = null;
          await user.save();
        }
      } catch (_e) {
        // noop: invalid/expired token should still be cleared client-side
      }
    }

    clearAuthCookies(res);
    return res.status(200).json({ message: 'Logged out successfully' });
  } catch (error) {
    return res.status(500).json({ message: 'Server error during logout' });
  }
};
