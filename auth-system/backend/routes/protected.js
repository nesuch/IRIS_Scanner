import { Router } from 'express';
import { requireAuth } from '../middleware/auth.js';
import { authorizeRoles } from '../middleware/roles.js';

const router = Router();

// All authenticated users
router.get('/dashboard', requireAuth, (req, res) => {
  return res.json({
    message: 'Authenticated dashboard',
    user: { id: req.user._id, email: req.user.email, role: req.user.role }
  });
});

// Admin only routes
router.get('/admin', requireAuth, authorizeRoles('admin'), (_req, res) => {
  return res.json({ message: 'Admin Console access granted' });
});

router.get('/system-analytics', requireAuth, authorizeRoles('admin'), (_req, res) => {
  return res.json({ message: 'System Analytics access granted' });
});

// Analyst + Admin routes
router.get('/data-explorer', requireAuth, authorizeRoles('admin', 'analyst'), (_req, res) => {
  return res.json({ message: 'Data Explorer access granted' });
});

router.get('/compliance-cockpit', requireAuth, authorizeRoles('admin', 'analyst'), (_req, res) => {
  return res.json({ message: 'Compliance Cockpit access granted' });
});

// Viewer read-only route (and other roles may also view)
router.get('/universal-module', requireAuth, authorizeRoles('viewer', 'analyst', 'admin'), (_req, res) => {
  return res.json({ message: 'Universal module read-only access granted' });
});

export default router;
