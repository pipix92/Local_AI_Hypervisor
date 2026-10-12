# hermes_rpc_service.ps1 - supervisore permanente per ggml-rpc-server (v2)
#
# Gira come task pianificato "HermesRPC" (utente SYSTEM). Riavvia il server
# RPC se muore; un solo processo alla volta.
#
# NOVITA' v2: build, flag e variabili GGML NON sono piu' scritti qui dentro:
# si leggono da rpc_config.json (stessa cartella), che si modifica con la TUI
# rpc_tui.py. Il supervisore controlla il file ogni 5 secondi: se cambia,
# ferma il server e lo riavvia con la nuova configurazione. La TUI quindi non
# ha bisogno dei permessi di SYSTEM: le basta scrivere il file.
#
# Se rpc_config.json manca o non e' leggibile si usano i valori di sempre
# (Cuda13, 0.0.0.0:50052): il comportamento di prima resta il ripiego.
# Stato corrente -> rpc_status.json (letto dalla TUI).

$ErrorActionPreference = 'Continue'

$LOG_DIR = 'C:\Users\mikye\hermes-rpc'
$LOG     = Join-Path $LOG_DIR 'rpc-server.log'
$CFG     = Join-Path $LOG_DIR 'rpc_config.json'
$STATUS  = Join-Path $LOG_DIR 'rpc_status.json'

$DEFAULT_BUILD = 'C:\Users\mikye\Documents\LLAMASERVER\Cuda13'

# Variabili d'ambiente che la TUI puo' impostare. Prima di ogni avvio vengono
# TUTTE rimosse e poi rimesse solo quelle presenti nel config, cosi' una
# variabile tolta dalla TUI non resta appiccicata al processo.
$KNOWN_ENV = @(
    'GGML_CUDA_DISABLE_GRAPHS', 'GGML_CUDA_GRAPH_OPT', 'GGML_CUDA_DISABLE_FUSION',
    'GGML_CUDA_PDL', 'GGML_CUDA_NO_PINNED', 'GGML_CUDA_REGISTER_HOST',
    'GGML_CUDA_ENABLE_UNIFIED_MEMORY', 'GGML_CUDA_CUBLAS_COMPUTE_TYPE',
    'GGML_OP_OFFLOAD_MIN_BATCH', 'GGML_CUDA_DEVICES', 'GGML_CUDA_P2P',
    'GGML_RPC_DEBUG', 'GGML_SCHED_DEBUG', 'CUDA_VISIBLE_DEVICES', 'LLAMA_CACHE'
)

if (-not (Test-Path $LOG_DIR)) { New-Item -ItemType Directory -Path $LOG_DIR -Force | Out-Null }

function Log($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg
    Add-Content -Path $LOG -Value $line
}

function Get-CfgHash {
    if (Test-Path $CFG) { return (Get-FileHash -Path $CFG -Algorithm SHA256).Hash }
    return 'default'
}

function Read-Cfg {
    $c = [ordered]@{ build_dir = $DEFAULT_BUILD; host = '0.0.0.0'; port = 50052;
                     threads = 0; devices = ''; cache = $false; env = @{}; extra_args = @() }
    if (Test-Path $CFG) {
        try {
            $j = Get-Content -Raw -Path $CFG | ConvertFrom-Json
            foreach ($k in 'build_dir', 'host', 'port', 'threads', 'devices', 'cache', 'extra_args') {
                if ($null -ne $j.$k) { $c[$k] = $j.$k }
            }
            if ($null -ne $j.env) {
                $h = @{}
                $j.env.PSObject.Properties | ForEach-Object { $h[$_.Name] = [string]$_.Value }
                $c.env = $h
            }
        } catch {
            Log "rpc_config.json non valido ($($_.Exception.Message)): uso i valori di default"
        }
    }
    $exe = Join-Path $c.build_dir 'ggml-rpc-server.exe'
    if (-not (Test-Path $exe)) {
        Log "Build '$($c.build_dir)' senza ggml-rpc-server.exe: ripiego su $DEFAULT_BUILD"
        $c.build_dir = $DEFAULT_BUILD
    }
    return $c
}

function Build-Args($c) {
    $a = @('-H', [string]$c.host, '-p', [string]$c.port)
    if ([int]$c.threads -gt 0) { $a += @('-t', [string]$c.threads) }
    if ([string]$c.devices) { $a += @('-d', [string]$c.devices) }
    if ($c.cache -eq $true) { $a += '-c' }
    foreach ($x in $c.extra_args) { if ([string]$x) { $a += [string]$x } }
    return $a
}

function Write-Status($p, $c, $argList, $hash, $state) {
    $s = [ordered]@{
        state = $state; pid = $(if ($p) { $p.Id } else { $null })
        build_dir = $c.build_dir; args = $argList; env = $c.env
        config_hash = $hash; supervisor_pid = $PID
        updated = (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
    }
    try { $s | ConvertTo-Json -Depth 4 | Set-Content -Path $STATUS -Encoding UTF8 } catch { }
}

Log "Supervisore v2 avviato (PID $PID, utente $env:USERNAME). Config: $CFG"

# Chiude qualsiasi istanza preesistente di ggml-rpc-server per evitare doppi ascolti
Get-Process -Name 'ggml-rpc-server' -ErrorAction SilentlyContinue | ForEach-Object {
    Log "Ucciso processo ggml-rpc-server preesistente PID $($_.Id)"
    Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Seconds 3

$backoff = 5
while ($true) {
    $p = $null
    try {
        $hash = Get-CfgHash
        $c = Read-Cfg
        $exe = Join-Path $c.build_dir 'ggml-rpc-server.exe'
        $argv = Build-Args $c

        foreach ($n in $KNOWN_ENV) { [Environment]::SetEnvironmentVariable($n, $null, 'Process') }
        foreach ($k in $c.env.Keys) {
            if ($KNOWN_ENV -contains $k) { [Environment]::SetEnvironmentVariable($k, $c.env[$k], 'Process') }
        }

        $stdout = Join-Path $LOG_DIR 'rpc-stdout.log'
        $stderr = Join-Path $LOG_DIR 'rpc-stderr.log'
        $p = Start-Process -FilePath $exe -ArgumentList $argv `
             -WorkingDirectory $c.build_dir `
             -RedirectStandardOutput $stdout -RedirectStandardError $stderr `
             -WindowStyle Hidden -PassThru
        $envtxt = ($c.env.Keys | ForEach-Object { "$_=$($c.env[$_])" }) -join ' '
        Log "Avviato ggml-rpc-server PID $($p.Id): $exe $($argv -join ' ')  env[$envtxt]"
        Write-Status $p $c $argv $hash 'starting'

        $ok = $false
        for ($i = 0; $i -lt 30; $i++) {
            Start-Sleep -Seconds 2
            if ($p.HasExited) { break }
            $tc = New-Object System.Net.Sockets.TcpClient
            try { $tc.Connect('127.0.0.1', [int]$c.port); $tc.Close(); $ok = $true; break } catch { $tc.Close() }
        }
        if ($ok) {
            Log "Server RPC in ascolto e rispondente sulla porta $($c.port)"
            Write-Status $p $c $argv $hash 'running'
            $backoff = 5
        } else {
            Log "Server non raggiungibile su $($c.port) (o uscito subito)"
            Write-Status $p $c $argv $hash 'failed'
        }

        # resta vivo finche' il server gira; se muore o il config cambia, riavvia
        $changed = $false
        while (-not $p.HasExited) {
            Start-Sleep -Seconds 5
            if ((Get-CfgHash) -ne $hash) { $changed = $true; break }
        }
        if ($changed) {
            Log "rpc_config.json cambiato: riavvio il server con la nuova configurazione"
            Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
            Start-Sleep -Seconds 2
            $backoff = 1
        } else {
            Log "Server terminato con codice $($p.ExitCode). Riavvio tra $backoff secondi."
            Write-Status $p $c $argv $hash 'exited'
        }
    } catch {
        Log "ERRORE: $($_.Exception.Message)"
    }
    Start-Sleep -Seconds $backoff
    if ($backoff -lt 60) { $backoff += 5 }
}
