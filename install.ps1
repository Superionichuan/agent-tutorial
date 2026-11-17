# Install mini_agent on Windows, in your own user account (no administrator rights).
#
#   powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/Superionichuan/agent-tutorial/main/install.ps1 | iex"
#
# What it does: installs uv (the Python package manager from astral.sh) if it is missing,
# lets uv install mini_agent with a Python 3.12 (one it finds, or one it downloads), puts the
# `mini_agent` command on your PATH, and starts `mini_agent setup` (choose a model and paste
# a key) unless a model is set up already. Running it again upgrades and keeps your settings.
# Uninstall: uv tool uninstall mini_agent (your settings and session logs stay in
# %USERPROFILE%\.mini_agent; delete that folder to remove them too).
#
# Options, as environment variables:
#   MINI_AGENT_SPEC      what to install (default: the release wheel of $Version below)
#   MINI_AGENT_PYTHON    Python version for mini_agent (default 3.12)
#   MINI_AGENT_NO_SETUP  1 = do not start `mini_agent setup` at the end
#
# The script never calls `exit` (that would close a PowerShell window it was pasted into);
# a failure stops it with an error message instead.

& {
    $Version = '2.0.0'
    $Repo = 'Superionichuan/agent-tutorial'
    $WheelUrl = "https://github.com/$Repo/releases/download/v$Version/mini_agent-$Version-py3-none-any.whl"
    $Spec = if ($env:MINI_AGENT_SPEC) { $env:MINI_AGENT_SPEC } else { "mini_agent[ui] @ $WheelUrl" }
    $PythonVersion = if ($env:MINI_AGENT_PYTHON) { $env:MINI_AGENT_PYTHON } else { '3.12' }

    function Stop-Install([string] $why) {
        throw "mini_agent was not installed: $why. Check your internet connection and run the same command again."
    }

    function Find-Uv {
        $cmd = Get-Command uv -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($cmd) { return $cmd.Source }
        $dirs = @($env:UV_INSTALL_DIR, $env:XDG_BIN_HOME, (Join-Path $HOME '.local\bin'), (Join-Path $HOME '.cargo\bin'))
        foreach ($dir in $dirs) {
            if ($dir) {
                $exe = Join-Path $dir 'uv.exe'
                if (Test-Path $exe) { return $exe }
            }
        }
        return $null
    }

    # GitHub and astral.sh need TLS 1.2; older Windows PowerShell may not offer it by default.
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

    Write-Host "Installing mini_agent $Version ..."

    $uv = Find-Uv
    if (-not $uv) {
        Write-Host 'Installing uv (the Python package manager from astral.sh) ...'
        try {
            $uvInstaller = Invoke-RestMethod -Uri 'https://astral.sh/uv/install.ps1' -ErrorAction Stop
        } catch {
            Stop-Install 'the uv installer could not be downloaded'
        }
        Invoke-Expression $uvInstaller
        $uv = Find-Uv
        if (-not $uv) { Stop-Install 'uv was installed but cannot be found' }
    }
    Write-Host "Using uv: $uv"

    $bin = & $uv tool dir --bin
    if ($LASTEXITCODE -ne 0 -or -not $bin) { Stop-Install 'uv did not report its command folder' }
    $bin = ([string]($bin | Select-Object -Last 1)).Trim()
    # The command folder on this session's PATH before the install: uv then does not print its
    # own advice about PATH, and `mini_agent` works in this window right away.
    $windowPath = $env:Path
    if (($env:Path -split ';') -notcontains $bin) { $env:Path = "$bin;$env:Path" }

    # --force: replace a mini_agent command left by an older installation (pip --user).
    # --reinstall: running this script again installs the current release.
    & $uv tool install --force --reinstall --python $PythonVersion $Spec
    if ($LASTEXITCODE -ne 0) { Stop-Install 'uv could not install the package' }
    # New windows find the command once the folder is on your user PATH; update-shell sees the
    # PATH this window started with, so it does not take the line above for that.
    $withBin = $env:Path
    $env:Path = $windowPath
    & $uv tool update-shell
    $env:Path = $withBin

    $exe = Join-Path $bin 'mini_agent.exe'
    # A command that is not found leaves $LASTEXITCODE unchanged, so check the file first.
    if (-not (Test-Path $exe)) { Stop-Install "uv did not create $exe" }
    $null = & $exe --version
    if ($LASTEXITCODE -ne 0) { Stop-Install 'the mini_agent command does not start' }

    Write-Host ''
    Write-Host "mini_agent is installed: $exe"
    $configDir = if ($env:MINI_AGENT_HOME) { $env:MINI_AGENT_HOME } else { Join-Path $HOME '.mini_agent' }
    $config = Join-Path $configDir 'config.env'
    $settingsKept = (Test-Path $config) -and
        (Select-String -Path $config -Pattern '^[A-Za-z_][A-Za-z0-9_]*=.' -Quiet)
    if ($settingsKept) {
        Write-Host ''
        Write-Host 'Your settings are kept; `mini_agent setup` changes them.'
    } elseif ($env:MINI_AGENT_NO_SETUP -ne '1' -and -not [Console]::IsInputRedirected) {
        Write-Host ''
        Write-Host 'Next: choose a model and paste your key (Ctrl-C skips; run "mini_agent setup" later).'
        & $exe setup
    }
    Write-Host ''
    Write-Host 'Start it with:  mini_agent   (open a new terminal window if this one does not find it)'
    Write-Host 'Check the set-up any time with:  mini_agent doctor'
}
