# Security Policy

## API Key Management

### Overview
RAG-CLI uses several external APIs that require authentication keys:
- **ANTHROPIC_API_KEY**: Claude API for response generation
- **TAVILY_API_KEY**: Tavily search API for web search
- **STACKOVERFLOW_KEY**: Stack Overflow API (optional)

### Setup

1. **Create .env file** (never commit this file):
```bash
# Copy template
cp config/templates/.env.template .env

# Edit with your keys
ANTHROPIC_API_KEY=sk-ant-...
TAVILY_API_KEY=tvly-...
```

2. **Verify .gitignore** includes:
```
.env
*.key
*_key
credentials.json
```

### Best Practices

#### Key Storage
- Store keys ONLY in `.env` file or environment variables
- Never hardcode keys in source code
- Never commit keys to version control
- Use different keys for development and production

#### Key Rotation
To rotate API keys:
1. Generate new key from API provider
2. Update `.env` file with new key
3. Restart all RAG-CLI services
4. Verify new key works
5. Revoke old key from API provider

#### Monitoring
Check for exposed keys:
```bash
# Scan git history for accidentally committed keys
git log -p | grep -E "(api[_-]?key|token|secret)" -i

# Validate current setup
python scripts/validate_config.py
```

### Automatic Redaction

RAG-CLI automatically redacts sensitive information from logs:
- API keys (20+ characters)
- Tokens
- Secrets
- Passwords (8+ characters)
- Bearer tokens

Example:
```
# Original log message:
"Connecting to API with key=sk-ant-1234567890abcdef..."

# Redacted log:
"Connecting to API with key=***REDACTED***"
```

### Troubleshooting

#### Tavily 401 Unauthorized
```
Error: Tavily API request failed: 401 Client Error: Unauthorized
```

**Solution:**
1. Verify `TAVILY_API_KEY` is set in `.env`
2. Check key is valid at https://tavily.com
3. Ensure no leading/trailing spaces in `.env`
4. Restart RAG-CLI to reload environment

#### Missing ANTHROPIC_API_KEY
```
Warning: ANTHROPIC_API_KEY not set, using mock mode
```

**Solution:**
1. Get API key from https://console.anthropic.com
2. Add to `.env`: `ANTHROPIC_API_KEY=sk-ant-...`
3. Restart application

### Security Checklist

Before deployment:
- [ ] All API keys stored in `.env` or environment variables
- [ ] `.env` file in `.gitignore`
- [ ] No keys in source code (check with grep)
- [ ] Different keys for dev/staging/production
- [ ] Log redaction tested and working
- [ ] API key quotas monitored
- [ ] Key rotation procedure documented

### Reporting Security Issues

If you discover a security vulnerability:
1. **Do NOT** open a public GitHub issue
2. Email security contact (see package metadata)
3. Include:
   - Description of vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### Responsible Disclosure

We follow responsible disclosure practices:
- Acknowledge report within 48 hours
- Provide estimated fix timeline within 1 week
- Notify reporter when patched
- Credit reporter (unless anonymity requested)

## Updates and Patches

Security patches are released as:
- **Critical**: Within 24 hours
- **High**: Within 1 week
- **Medium**: Within 1 month
- **Low**: Next regular release

Subscribe to security advisories:
- Watch GitHub repository
- Enable security alerts
- Monitor CHANGELOG.md for security fixes

## Compliance

RAG-CLI is designed to support:
- GDPR compliance (no personal data stored)
- Local-first architecture (data stays on your machine)
- Audit logging for compliance tracking

## Additional Resources

- [Claude API Security Best Practices](https://docs.anthropic.com/security)
- [Environment Variable Security](https://12factor.net/config)
- [Git Secret Management](https://git-secret.io/)
