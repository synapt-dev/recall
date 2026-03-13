"""Tests for synapt.recall.scrub — secret detection and redaction."""

import json
import textwrap

import pytest

from synapt.recall.scrub import scrub_jsonl, scrub_text


# ---------------------------------------------------------------------------
# Token prefix patterns
# ---------------------------------------------------------------------------

class TestTokenPrefixes:
    """Known token prefixes should always be redacted."""

    def test_anthropic_key(self):
        text = "my key is sk-ant-api03-abcdefghijklmnopqrstuvwxyz"
        result = scrub_text(text)
        assert "sk-ant-" not in result
        assert "[REDACTED:" in result

    def test_openai_project_key(self):
        text = "sk-proj-abcdefghijklmnopqrstuvwxyz123456"
        result = scrub_text(text)
        assert "sk-proj-" not in result
        assert "[REDACTED:" in result

    def test_openai_legacy_key(self):
        text = "sk-abcdefghijklmnopqrstuvwxyz123456"
        result = scrub_text(text)
        assert "sk-" not in result
        assert "[REDACTED:" in result

    def test_huggingface_token(self):
        text = "HF_TOKEN=hf_aBcDeFgHiJkLmNoPqRsTuVwXyZ"
        result = scrub_text(text)
        assert "hf_" not in result
        assert "[REDACTED:" in result

    def test_github_pat(self):
        text = "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmn"
        result = scrub_text(text)
        assert "ghp_" not in result
        assert "[REDACTED:" in result

    def test_github_oauth(self):
        text = "gho_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmn"
        result = scrub_text(text)
        assert "gho_" not in result

    def test_github_fine_grained_pat(self):
        text = "github_pat_11ABCDEF0aBcDeFgHiJkLmNoPqRsTuVwXyZ1234567890"
        result = scrub_text(text)
        assert "github_pat_" not in result
        assert "[REDACTED:" in result

    def test_slack_bot_token(self):
        text = "SLACK_BOT_TOKEN=xoxb-123456789012-123456789012-abcdefghij"
        result = scrub_text(text)
        assert "xoxb-" not in result
        assert "[REDACTED:" in result

    def test_slack_user_token(self):
        text = "xoxp-123456789012-123456789012-abcdefghij"
        result = scrub_text(text)
        assert "xoxp-" not in result

    def test_slack_app_token(self):
        text = "xapp-1-A0123456789-1234567890123-abcdefabcdef"
        result = scrub_text(text)
        assert "xapp-" not in result

    def test_pypi_token(self):
        text = "pypi-AgEIcHlwaS5vcmcCJGNmNTI5MjczLTljYWYtNDdlOS1hMWYwLWVjM2FjY2MwY2M5OQ"
        result = scrub_text(text)
        assert "pypi-" not in result
        assert "[REDACTED:" in result

    def test_modal_token(self):
        text = "ak-abcdefghijklmnopqrstuvwxyz"
        result = scrub_text(text)
        assert "ak-" not in result
        assert "[REDACTED:" in result

    def test_aws_access_key(self):
        text = "AKIAIOSFODNN7EXAMPLE"
        result = scrub_text(text)
        assert "AKIA" not in result
        assert "[REDACTED:" in result


# ---------------------------------------------------------------------------
# Structured secrets
# ---------------------------------------------------------------------------

class TestStructuredSecrets:

    def test_bearer_token(self):
        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.payload.sig"
        result = scrub_text(text)
        assert "eyJhbG" not in result
        assert "[REDACTED:" in result

    def test_pem_private_key(self):
        text = "-----BEGIN RSA PRIVATE KEY-----"
        result = scrub_text(text)
        assert "PRIVATE KEY" not in result
        assert "[REDACTED:" in result

    def test_pem_ec_private_key(self):
        text = "-----BEGIN EC PRIVATE KEY-----"
        result = scrub_text(text)
        assert "PRIVATE KEY" not in result

    def test_pem_full_block(self):
        text = textwrap.dedent("""\
            -----BEGIN RSA PRIVATE KEY-----
            MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQC7
            -----END RSA PRIVATE KEY-----""")
        result = scrub_text(text)
        assert "MIIEvg" not in result
        assert "PRIVATE KEY" not in result
        assert "[REDACTED:" in result

    def test_bearer_with_base64_chars(self):
        text = "Authorization: Bearer dXNlcjpwYXNz/d29yZA+foo=="
        result = scrub_text(text)
        assert "dXNlcjpwYXNz" not in result
        assert "[REDACTED:" in result

    def test_jwt_token(self):
        text = "token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        result = scrub_text(text)
        assert "eyJhbG" not in result
        assert "[REDACTED:" in result

    def test_authorization_key_header(self):
        """fal.ai style: Authorization: Key uuid:hex"""
        text = 'Authorization: Key 974a80a2-f836-4942-8bb9-07c215e5c404:e1b0ba4aca5aa0be5563e6c185397aeb'
        result = scrub_text(text)
        assert "974a80a2" not in result
        assert "[REDACTED:" in result

    def test_authorization_basic_header(self):
        text = "Authorization: Basic dXNlcm5hbWU6cGFzc3dvcmQ="
        result = scrub_text(text)
        assert "dXNlcm5hbWU" not in result
        assert "[REDACTED:" in result

    def test_authorization_token_header(self):
        text = "Authorization: Token abc123def456ghi789jkl012"
        result = scrub_text(text)
        assert "abc123def" not in result
        assert "[REDACTED:" in result

    def test_postgres_connection_string(self):
        text = "DATABASE_URL=postgres://myuser:s3cr3tP4ss@db.example.com:5432/mydb"
        result = scrub_text(text)
        assert "s3cr3tP4ss" not in result
        assert "[REDACTED:" in result

    def test_mongodb_connection_string(self):
        text = "MONGO_URI=mongodb+srv://admin:hunter2@cluster0.abc123.mongodb.net/db"
        result = scrub_text(text)
        assert "hunter2" not in result
        assert "[REDACTED:" in result

    def test_redis_connection_string(self):
        text = "REDIS_URL=redis://:mypassword@redis.example.com:6379/0"
        result = scrub_text(text)
        assert "mypassword" not in result
        assert "[REDACTED:" in result


# ---------------------------------------------------------------------------
# Env var assignments
# ---------------------------------------------------------------------------

class TestEnvVarAssignments:

    def test_token_equals(self):
        text = "HF_TOKEN=hf_aBcDeFgHiJkLmNoPqRsTuVwXyZ"
        result = scrub_text(text)
        # Both the env var pattern and the hf_ prefix should catch this
        assert "hf_aBcD" not in result

    def test_api_key_quoted(self):
        text = 'export OPENAI_API_KEY="sk-proj-abc123def456ghi789"'
        result = scrub_text(text)
        assert "sk-proj-" not in result

    def test_password_assignment(self):
        text = "DB_PASSWORD=supersecretpassword123"
        result = scrub_text(text)
        assert "supersecretpassword123" not in result

    def test_secret_with_colon(self):
        text = "API_SECRET: my_long_secret_value_here"
        result = scrub_text(text)
        assert "my_long_secret_value_here" not in result

    def test_access_key_assignment(self):
        text = "AWS_ACCESS_KEY=AKIAIOSFODNN7EXAMPLE"
        result = scrub_text(text)
        assert "AKIAIOSFODNN7EXAMPLE" not in result

    def test_secret_key_assignment(self):
        text = "SECRET_KEY=django-insecure-abc123def456ghi789jkl012"
        result = scrub_text(text)
        assert "django-insecure" not in result
        assert "[REDACTED:" in result

    def test_aws_secret_access_key(self):
        text = "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        result = scrub_text(text)
        assert "wJalrXUtnFEMI" not in result

    def test_fal_key(self):
        """The exact FAL_KEY that leaked in production — issue #63."""
        text = 'export FAL_KEY="974a80a2-f836-4942-8bb9-07c215e5c404:e1b0ba4aca5aa0be5563e6c185397aeb"'
        result = scrub_text(text)
        assert "974a80a2" not in result
        assert "e1b0ba4a" not in result
        assert "[REDACTED:" in result

    def test_fal_key_unquoted(self):
        text = "FAL_KEY=974a80a2-f836-4942-8bb9-07c215e5c404:e1b0ba4aca5aa0be5563e6c185397aeb"
        result = scrub_text(text)
        assert "974a80a2" not in result
        assert "[REDACTED:" in result

    def test_stripe_key(self):
        # Use a clearly-fake key to avoid GitHub push protection
        text = "STRIPE_KEY=sk_fake_XXXXXXXXXXXXXXXXXXXX1234"
        result = scrub_text(text)
        assert "sk_fake_" not in result
        assert "[REDACTED:" in result

    def test_encryption_key(self):
        text = "ENCRYPTION_KEY=aGVsbG93b3JsZGhlbGxvd29ybGQ="
        result = scrub_text(text)
        assert "aGVsbG93" not in result
        assert "[REDACTED:" in result

    def test_credential_assignment(self):
        text = "GCP_CREDENTIAL=long-credential-value-here-12345"
        result = scrub_text(text)
        assert "long-credential" not in result
        assert "[REDACTED:" in result

    def test_auth_assignment(self):
        text = "GITHUB_AUTH=ghp_abc123def456ghi789jkl012mno345pqr678"
        result = scrub_text(text)
        assert "ghp_abc123" not in result
        assert "[REDACTED:" in result

    def test_passphrase_assignment(self):
        text = "GPG_PASSPHRASE=my-super-secret-passphrase-2026"
        result = scrub_text(text)
        assert "my-super-secret" not in result
        assert "[REDACTED:" in result

    def test_grep_output_fal_key(self):
        """Grep output showing FAL_KEY in .zshrc — the actual leak scenario."""
        text = '16:export FAL_KEY=974a80a2-f836-4942-8bb9-07c215e5c404:e1b0ba4aca5aa0be5563e6c185397aeb'
        result = scrub_text(text)
        assert "974a80a2" not in result
        assert "[REDACTED:" in result


# ---------------------------------------------------------------------------
# Deterministic hashing
# ---------------------------------------------------------------------------

class TestDeterministic:

    def test_same_secret_same_placeholder(self):
        secret = "hf_aBcDeFgHiJkLmNoPqRsTuVwXyZ"
        r1 = scrub_text(f"first: {secret}")
        r2 = scrub_text(f"second: {secret}")
        # Extract the [REDACTED:xxxx] placeholders
        import re
        matches1 = re.findall(r"\[REDACTED:\w+\]", r1)
        matches2 = re.findall(r"\[REDACTED:\w+\]", r2)
        assert matches1 == matches2

    def test_different_secrets_different_placeholders(self):
        r1 = scrub_text("hf_aBcDeFgHiJkLmNoPqRsTuVwXyZ")
        r2 = scrub_text("hf_ZyXwVuTsRqPoNmLkJiHgFeDcBa")
        assert "[REDACTED:" in r1
        assert "[REDACTED:" in r2
        assert r1 != r2


# ---------------------------------------------------------------------------
# Non-secrets preserved
# ---------------------------------------------------------------------------

class TestPreservation:

    def test_normal_text_unchanged(self):
        text = "This is a normal conversation about code."
        assert scrub_text(text) == text

    def test_short_token_like_string_unchanged(self):
        # "sk-" followed by fewer than 20 chars should not match
        text = "sk-short"
        assert scrub_text(text) == text

    def test_code_snippet_unchanged(self):
        text = "def hello():\n    return 'world'"
        assert scrub_text(text) == text

    def test_file_paths_unchanged(self):
        text = "src/synapt/recall/core.py:201"
        assert scrub_text(text) == text

    def test_empty_string(self):
        assert scrub_text("") == ""

    def test_none_like_empty(self):
        # scrub_text handles empty/falsy gracefully
        assert scrub_text("") == ""

    def test_git_sha_unchanged(self):
        # 40-char hex SHA should NOT be caught (no token prefix)
        text = "commit a5d520d1234567890abcdef1234567890abcdef"
        assert scrub_text(text) == text

    def test_uuid_unchanged(self):
        text = "session e43f3562-50f4-47a6-9776-b91128013f84"
        assert scrub_text(text) == text

    def test_token_type_not_redacted(self):
        # _TOKEN followed by a letter should NOT match (e.g., TOKEN_TYPE)
        text = "ACCESS_TOKEN_TYPE=bearer"
        assert scrub_text(text) == text

    def test_secret_manager_not_redacted(self):
        text = "AWS_SECRET_MANAGER_ARN=arn:aws:sm:us-east-1"
        assert scrub_text(text) == text

    def test_task_hyphen_not_redacted(self):
        # "task-" contains "sk-" but should NOT match (no word boundary)
        text = "task-managementServiceHandlerForUsers2026"
        assert scrub_text(text) == text

    def test_flask_hyphen_not_redacted(self):
        text = "flask-restfulApplicationFactorySetup"
        assert scrub_text(text) == text

    def test_primary_key_not_redacted(self):
        """PRIMARY_KEY=id is too short to match the 8-char minimum."""
        text = "PRIMARY_KEY=id"
        assert scrub_text(text) == text

    def test_foreign_key_not_redacted(self):
        text = "FOREIGN_KEY=user_id"
        # 7 chars — just under the minimum
        assert scrub_text(text) == text

    def test_key_name_not_redacted(self):
        """_KEY followed by a letter should not match (e.g., KEY_NAME)."""
        text = "CACHE_KEY_NAME=user_sessions"
        assert scrub_text(text) == text

    def test_jwt_like_short_not_redacted(self):
        """Short eyJ strings that aren't real JWTs should pass through."""
        text = "eyJhbG.short.x"
        assert scrub_text(text) == text

    def test_url_without_creds_not_redacted(self):
        """postgres:// URL without user:pass should not match."""
        text = "postgres://localhost:5432/mydb"
        assert scrub_text(text) == text

    def test_http_url_not_redacted(self):
        """Regular HTTP URLs should not match connection string pattern."""
        text = "https://example.com/api/v1"
        assert scrub_text(text) == text

    def test_authorization_no_value_not_redacted(self):
        """Authorization header without a long-enough value should pass."""
        text = "Authorization: Bearer short"
        assert scrub_text(text) == text


# ---------------------------------------------------------------------------
# scrub_jsonl
# ---------------------------------------------------------------------------

class TestScrubJsonl:

    def test_roundtrip(self, tmp_path):
        src = tmp_path / "test.jsonl"
        entries = [
            {
                "type": "human",
                "message": {
                    "content": "my token is hf_aBcDeFgHiJkLmNoPqRsTuVwXyZ"
                },
                "timestamp": "2026-03-02T10:00:00Z",
            },
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "text", "text": "I see your token sk-ant-api03-abcdefghijklmnopqrstuvwxyz"},
                        {"type": "tool_use", "name": "Bash", "input": {"command": "export HF_TOKEN=hf_secret12345678901234"}},
                    ]
                },
            },
        ]
        src.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

        dst = tmp_path / "scrubbed.jsonl"
        scrub_jsonl(src, dst)

        scrubbed = dst.read_text()
        assert "hf_aBcD" not in scrubbed
        assert "sk-ant-" not in scrubbed
        assert "hf_secret" not in scrubbed
        assert "[REDACTED:" in scrubbed

        # Verify it's still valid JSONL
        for line in scrubbed.strip().split("\n"):
            json.loads(line)

    def test_inplace(self, tmp_path):
        src = tmp_path / "test.jsonl"
        src.write_text(json.dumps({
            "type": "human",
            "message": {"content": "token hf_aBcDeFgHiJkLmNoPqRsTuVwXyZ"},
        }) + "\n")

        result = scrub_jsonl(src)
        assert result == src
        assert "hf_aBcD" not in src.read_text()

    def test_malformed_lines_scrubbed(self, tmp_path):
        """Malformed JSON lines should still have secrets scrubbed."""
        src = tmp_path / "test.jsonl"
        src.write_text(
            "broken json with hf_aBcDeFgHiJkLmNoPqRsTuVwXyZ\n"
            + json.dumps({"type": "human", "message": {"content": "clean"}}) + "\n"
        )

        dst = tmp_path / "out.jsonl"
        scrub_jsonl(src, dst)

        lines = dst.read_text().strip().split("\n")
        assert "hf_aBcD" not in lines[0]
        assert "[REDACTED:" in lines[0]
        assert json.loads(lines[1])["message"]["content"] == "clean"

    def test_thinking_block_scrubbed(self, tmp_path):
        """Secrets inside thinking blocks should be scrubbed."""
        src = tmp_path / "test.jsonl"
        entry = {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "The user's token is hf_aBcDeFgHiJkLmNoPqRsTuVwXyZ",
                    }
                ]
            },
        }
        src.write_text(json.dumps(entry) + "\n")

        dst = tmp_path / "out.jsonl"
        scrub_jsonl(src, dst)

        result = json.loads(dst.read_text().strip())
        thinking = result["message"]["content"][0]["thinking"]
        assert "hf_aBcD" not in thinking
        assert "[REDACTED:" in thinking

    def test_nested_tool_input_scrubbed(self, tmp_path):
        """Secrets in nested tool_use input dicts should be scrubbed."""
        src = tmp_path / "test.jsonl"
        entry = {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Bash",
                        "input": {
                            "command": "curl -H 'Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.payload.sig'",
                            "env": {"HF_TOKEN": "hf_aBcDeFgHiJkLmNoPqRsTuVwXyZ"},
                        },
                    }
                ]
            },
        }
        src.write_text(json.dumps(entry) + "\n")

        dst = tmp_path / "out.jsonl"
        scrub_jsonl(src, dst)

        result = json.loads(dst.read_text().strip())
        inp = result["message"]["content"][0]["input"]
        assert "eyJhbG" not in inp["command"]
        assert "hf_aBcD" not in inp["env"]["HF_TOKEN"]
        assert "[REDACTED:" in inp["command"]
        assert "[REDACTED:" in inp["env"]["HF_TOKEN"]

    def test_tool_result_content_scrubbed(self, tmp_path):
        """Secrets in tool_result content blocks should be scrubbed."""
        src = tmp_path / "test.jsonl"
        entry = {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "content": [
                            {"type": "text", "text": "Found key: sk-ant-api03-abcdefghijklmnopqrstuvwxyz"},
                        ],
                    }
                ]
            },
        }
        src.write_text(json.dumps(entry) + "\n")

        dst = tmp_path / "out.jsonl"
        scrub_jsonl(src, dst)

        result = json.loads(dst.read_text().strip())
        text = result["message"]["content"][0]["content"][0]["text"]
        assert "sk-ant-" not in text
        assert "[REDACTED:" in text

    def test_tool_result_string_content_scrubbed(self, tmp_path):
        """Tool results with plain string content should be scrubbed."""
        src = tmp_path / "test.jsonl"
        entry = {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "content": "Error: invalid token ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmn",
                    }
                ]
            },
        }
        src.write_text(json.dumps(entry) + "\n")

        dst = tmp_path / "out.jsonl"
        scrub_jsonl(src, dst)

        result = json.loads(dst.read_text().strip())
        content = result["message"]["content"][0]["content"]
        assert "ghp_" not in content
        assert "[REDACTED:" in content


# ---------------------------------------------------------------------------
# System artifact stripping
# ---------------------------------------------------------------------------

from synapt.recall.scrub import strip_system_artifacts


class TestStripSystemArtifacts:
    """strip_system_artifacts removes Claude Code injected tags."""

    def test_removes_system_reminder(self):
        text = "Hello\n<system-reminder>\nDo not respond.\n</system-reminder>\nWorld"
        assert strip_system_artifacts(text) == "Hello\n\nWorld"

    def test_removes_local_command_caveat(self):
        text = "<local-command-caveat>Caveat: messages below</local-command-caveat>Fix the bug"
        assert strip_system_artifacts(text) == "Fix the bug"

    def test_removes_available_deferred_tools(self):
        text = "Start\n<available-deferred-tools>\nTool1\nTool2\n</available-deferred-tools>\nEnd"
        assert strip_system_artifacts(text) == "Start\n\nEnd"

    def test_removes_env_block(self):
        text = "Intro\n<env>PATH=/usr/bin</env>\nBody"
        assert strip_system_artifacts(text) == "Intro\n\nBody"

    def test_removes_interrupted_marker(self):
        text = "Some text [Request interrupted by user for tool use] more text"
        assert strip_system_artifacts(text) == "Some text  more text"

    def test_preserves_normal_text(self):
        text = "This is a normal message with <b>html</b> tags."
        assert strip_system_artifacts(text) == text

    def test_multiple_tags(self):
        text = (
            "<system-reminder>A</system-reminder>"
            "Real content"
            "<local-command-caveat>B</local-command-caveat>"
            " more content"
        )
        assert strip_system_artifacts(text) == "Real content more content"

    def test_empty_string(self):
        assert strip_system_artifacts("") == ""

    def test_collapses_blank_lines(self):
        text = "A\n\n\n\n\nB"
        assert strip_system_artifacts(text) == "A\n\nB"

    def test_strips_leading_trailing_whitespace(self):
        text = "  \n<system-reminder>noise</system-reminder>\n  Hello  \n  "
        assert strip_system_artifacts(text) == "Hello"

    def test_tag_with_attributes(self):
        text = '<system-reminder foo="bar">content</system-reminder>Real'
        assert strip_system_artifacts(text) == "Real"

    def test_unrecognized_tags_pass_through(self):
        text = "<custom-tag>Keep this</custom-tag>"
        assert strip_system_artifacts(text) == text

    def test_truncated_tag_at_end_stripped(self):
        """Unclosed artifact tag at end of string (truncation case)."""
        text = "My question<local-command-caveat>Some caveat text truncat"
        assert strip_system_artifacts(text) == "My question"

    def test_truncated_system_reminder_stripped(self):
        text = "Focus text<system-reminder>\nNote: some file was"
        assert strip_system_artifacts(text) == "Focus text"

    def test_literal_env_in_prose_preserved(self):
        """Literal <env> in middle of text should not eat the rest."""
        text = "Set <env> variable PATH then restart"
        # <env> excluded from unclosed-tag regex to avoid false positives
        assert strip_system_artifacts(text) == text

    def test_closed_tag_then_text_preserved(self):
        """Properly closed tag + trailing text — open regex is a no-op."""
        text = "<system-reminder>noise</system-reminder>Keep this"
        assert strip_system_artifacts(text) == "Keep this"
