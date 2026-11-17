#!/bin/sh
# Install mini_agent on macOS or Linux, in your own user account (no administrator rights).
#
#   curl -LsSf https://raw.githubusercontent.com/Superionichuan/agent-tutorial/main/install.sh | sh
#
# What it does: installs uv (the Python package manager from astral.sh) if it is missing,
# lets uv install mini_agent with a Python 3.12 (one it finds, or one it downloads), puts the
# `mini_agent` command on your PATH, and starts `mini_agent setup` (choose a model and paste
# a key) unless a model is set up already. Running it again upgrades and keeps your settings.
# Uninstall: uv tool uninstall mini_agent (your settings and session logs stay in
# ~/.mini_agent; delete that folder to remove them too).
#
# Options, as environment variables:
#   MINI_AGENT_SPEC      what to install (default: the release wheel of VERSION below)
#   MINI_AGENT_PYTHON    Python version for mini_agent (default 3.12)
#   MINI_AGENT_NO_SETUP  1 = do not start `mini_agent setup` at the end
set -eu

VERSION="2.0.0"
REPO="Superionichuan/agent-tutorial"
WHEEL_URL="https://github.com/$REPO/releases/download/v$VERSION/mini_agent-$VERSION-py3-none-any.whl"
SPEC="${MINI_AGENT_SPEC:-mini_agent[ui] @ $WHEEL_URL}"
PYTHON_VERSION="${MINI_AGENT_PYTHON:-3.12}"

say() { printf '%s\n' "$*"; }
fail() {
    say ""
    say "mini_agent was not installed: $*"
    say "Check your internet connection and run the same command again."
    exit 1
}

find_uv() {
    if command -v uv >/dev/null 2>&1; then
        command -v uv
        return 0
    fi
    for dir in "${UV_INSTALL_DIR:-}" "${XDG_BIN_HOME:-}" "$HOME/.local/bin" "$HOME/.cargo/bin"; do
        if [ -n "$dir" ] && [ -x "$dir/uv" ]; then
            printf '%s\n' "$dir/uv"
            return 0
        fi
    done
    return 1
}

say "Installing mini_agent $VERSION ..."

UV="$(find_uv)" || UV=""
if [ -z "$UV" ]; then
    say "Installing uv (the Python package manager from astral.sh) ..."
    uv_installer="$(mktemp)" || fail "no temporary file for the uv installer"
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh -o "$uv_installer" \
            || fail "the uv installer could not be downloaded"
    elif command -v wget >/dev/null 2>&1; then
        wget -qO "$uv_installer" https://astral.sh/uv/install.sh \
            || fail "the uv installer could not be downloaded"
    else
        fail "neither curl nor wget is available to download uv"
    fi
    # its output only if it fails: on success the lines below say what to do next
    uv_log="$(mktemp)" || fail "no temporary file for the uv installer's output"
    if ! sh "$uv_installer" >"$uv_log" 2>&1; then
        cat "$uv_log"
        fail "the uv installer stopped"
    fi
    rm -f "$uv_installer" "$uv_log"
    UV="$(find_uv)" || fail "uv was installed but cannot be found"
fi
say "Using uv: $UV"

BIN="$("$UV" tool dir --bin)" || fail "uv did not report its command folder"

# --force: replace a mini_agent command left by an older installation (pip --user).
# --reinstall: running this script again installs the current release.
# The command folder on PATH for this one call only: uv then does not print its own advice
# about PATH (the lines at the end of this script give it, once).
PATH="$BIN:$PATH" "$UV" tool install --force --reinstall --python "$PYTHON_VERSION" "$SPEC" \
    || fail "uv could not install the package"
"$BIN/mini_agent" --version >/dev/null 2>&1 || fail "the mini_agent command does not start"

# New terminals find the command once its folder is on PATH; uv adds it to your shell's
# start-up file (it needs to know the shell: $SHELL, or the shell that runs this script).
on_path=no
path_set=no
case ":$PATH:" in
    *":$BIN:"*) on_path=yes ;;
    *) "$UV" tool update-shell >/dev/null 2>&1 && path_set=yes ;;
esac

say ""
say "mini_agent is installed: $BIN/mini_agent"

CONFIG="${MINI_AGENT_HOME:-$HOME/.mini_agent}/config.env"
if [ -f "$CONFIG" ] && grep -Eq '^[A-Za-z_][A-Za-z0-9_]*=.' "$CONFIG" 2>/dev/null; then
    say ""
    say "Your settings are kept; \`mini_agent setup\` changes them."
elif [ "${MINI_AGENT_NO_SETUP:-}" != "1" ] && (: </dev/tty) 2>/dev/null; then
    say ""
    say "Next: choose a model and paste your key (Ctrl-C skips; run \`mini_agent setup\` later)."
    "$BIN/mini_agent" setup </dev/tty || true
fi

say ""
if [ "$on_path" = yes ]; then
    say "Start it with:  mini_agent"
elif [ "$path_set" = yes ]; then
    say "Open a new terminal, then start it with:  mini_agent"
    say "(or, in this terminal:  export PATH=\"$BIN:\$PATH\")"
else
    say "Add the command's folder to your PATH: put this line in your shell's start-up file"
    say "(~/.zshrc for zsh, ~/.bashrc for bash), then open a new terminal:"
    say "    export PATH=\"$BIN:\$PATH\""
    say "Then start it with:  mini_agent"
fi
say "Check the set-up any time with:  mini_agent doctor"
