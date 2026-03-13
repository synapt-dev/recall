# Security Policy

## Supported Versions

Only the latest release of synapt receives security updates.

| Version | Supported |
|---------|-----------|
| Latest  | Yes       |
| Older   | No        |

## Architecture

Synapt is designed as a local-first system. By default:

- **Data stays on your machine.** Session indexes are stored in a local SQLite database and JSONL files. Nothing is sent to external services unless you explicitly configure a cloud backend.
- **The MCP server runs locally.** It binds to your machine and serves requests to your local AI assistant.
- **Enrichment uses local models.** On Apple Silicon, synapt runs MLX models on-device. Cloud LLM backends (e.g., for enrichment or evaluation) are opt-in and require you to provide your own API keys.
- **No telemetry or phone-home.** Synapt does not collect usage data, send analytics, or make any network requests beyond what you explicitly configure.

## Reporting a Vulnerability

If you discover a security issue, please report it responsibly. **Do not open a public GitHub issue.**

Email **[security@synapt.dev](mailto:security@synapt.dev)** with:

- A description of the vulnerability
- Steps to reproduce
- Any relevant environment details (OS, Python version, synapt version)

We will acknowledge receipt within 48 hours and aim to provide a fix or mitigation plan within 7 days for confirmed issues.

## Scope

The following are in scope for security reports:

- Unauthorized access to stored session data
- Path traversal or file access beyond the intended data directory
- Code injection through crafted transcripts or queries
- MCP server vulnerabilities (e.g., unauthorized tool invocation)
- Dependency vulnerabilities that are exploitable in synapt's usage

The following are **out of scope**:

- Issues requiring physical access to the machine where synapt runs
- Vulnerabilities in third-party cloud backends you choose to configure
- Denial of service against the local MCP server

## Best Practices

- Keep synapt updated to the latest version.
- Review file permissions on your data directory (`~/.local/share/synapt/` or your configured path).
- If using cloud LLM backends, secure your API keys via environment variables or a secrets manager -- do not commit them to source control.
