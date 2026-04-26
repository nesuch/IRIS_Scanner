import { Router } from 'express';
import rateLimit from 'express-rate-limit';
import { body } from 'express-validator';
import {
  registerUser,
  loginUser,
  refreshToken,
  logoutUser
} from '../controllers/authController.js';

const router = Router();

const loginLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  limit: 10,
  standardHeaders: true,
  legacyHeaders: false,
  message: { message: 'Too many login attempts. Try again later.' }
});

const emailValidator = body('email').isEmail().withMessage('Valid email is required').normalizeEmail();
const passwordValidator = body('password')
  .isString()
  .isLength({ min: 8 })
  .withMessage('Password must be at least 8 characters long');

router.post('/register', [emailValidator, passwordValidator], registerUser);
router.post('/login', loginLimiter, [emailValidator, passwordValidator], loginUser);
router.post('/refresh-token', refreshToken);
router.post('/logout', logoutUser);

export default router;
