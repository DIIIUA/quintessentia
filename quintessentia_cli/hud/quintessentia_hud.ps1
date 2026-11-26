#Requires -Version 5.1
# Quintessentia Live HUD — Smooth hardware-accelerated WPF telemetry overlay (Windows Native & WSL2).
# Zero external dependencies, zero per-tick object churn, frozen DirectX brushes, in-place charts,
# and live multi-scale RSSF balance + KV residency telemetry.

param(
    [string]$ConfigPath = "",
    [string]$CacheDir = "",
    [switch]$CapturePreview
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ParentLeaf = Split-Path -Leaf (Split-Path -Parent $ScriptDir)
$IsManagedHudDir = ((Split-Path -Leaf $ScriptDir) -eq "hud" -and ($ParentLeaf -eq "quintessentia_cli" -or $ParentLeaf -eq ".quintessentia"))

if (-not $CacheDir) {
    if ($IsManagedHudDir) {
        $CacheDir = Join-Path (Join-Path $env:USERPROFILE ".quintessentia") "hud_cache"
    } else {
        $CacheDir = Join-Path $ScriptDir ".cache"
    }
}
if (-not $ConfigPath) {
    if ($IsManagedHudDir) {
        $ConfigPath = Join-Path (Join-Path $env:USERPROFILE ".quintessentia") "hud_config.json"
    } else {
        $ConfigPath = Join-Path $ScriptDir "config.json"
    }
}

if (-not (Test-Path $CacheDir)) { New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null }
$ShowFlagPath = Join-Path $CacheDir "show_window.flag"

# Single-instance mutex: second launch wakes the running HUD instead of spawning duplicates
$created = $false
$InstanceMutex = New-Object System.Threading.Mutex($true, "Local\QuintessentiaLiveHUD_v2", [ref]$created)
if (-not $created) {
    try { Set-Content -LiteralPath $ShowFlagPath -Value "1" -Encoding ASCII } catch {}
    $InstanceMutex.Dispose()
    exit 0
}

Add-Type -AssemblyName PresentationFramework
Add-Type -AssemblyName PresentationCore
Add-Type -AssemblyName WindowsBase
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

Add-Type @"
using System;
using System.Runtime.InteropServices;

public static class QHNative {
    [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
    [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr hWnd);
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hWnd);
    [DllImport("user32.dll")] public static extern bool DestroyIcon(IntPtr hIcon);
    public static readonly IntPtr HWND_BOTTOM = new IntPtr(1);
    public static readonly IntPtr HWND_TOPMOST = new IntPtr(-1);
    public static readonly IntPtr HWND_NOTOPMOST = new IntPtr(-2);
    public const uint SWP_NOSIZE = 0x0001;
    public const uint SWP_NOMOVE = 0x0002;
    public const uint SWP_NOACTIVATE = 0x0010;
    public const int GWL_EXSTYLE = -20;
    public const int WS_EX_TOOLWINDOW = 0x00000080;

    [DllImport("user32.dll")]
    public static extern bool SetWindowPos(IntPtr hWnd, IntPtr hWndInsertAfter, int X, int Y, int cx, int cy, uint uFlags);

    [DllImport("user32.dll")]
    public static extern int GetWindowLong(IntPtr hWnd, int nIndex);

    [DllImport("user32.dll")]
    public static extern int SetWindowLong(IntPtr hWnd, int nIndex, int dwNewLong);
}
"@

$script:InvCulture = [System.Globalization.CultureInfo]::InvariantCulture
function Fmt([string]$Format, [object[]]$Vals) {
    return [string]::Format($script:InvCulture, $Format, $Vals)
}

$HudJsonPath = Join-Path $CacheDir "quintessentia_hud.json"
$ClearFlagPath = Join-Path $CacheDir "clear_history.flag"
$CaptureFlagPath = Join-Path $CacheDir "capture_preview.flag"
$StopFlagPath = Join-Path $CacheDir "stop_hud.flag"
$UiPidPath = Join-Path $CacheDir "hud_ui.pid"
$BackendScript = Join-Path $ScriptDir "hud_backend.py"
try { Set-Content -LiteralPath $UiPidPath -Value "$PID" -Encoding ASCII } catch {}
try { if (Test-Path $StopFlagPath) { Remove-Item -Force $StopFlagPath -ErrorAction SilentlyContinue } } catch {}

# Resolve Python executable cleanly across any Windows installation
function Resolve-PythonExe {
    if ($env:QUINTESSENTIA_PYTHON -and (Test-Path $env:QUINTESSENTIA_PYTHON)) {
        return $env:QUINTESSENTIA_PYTHON
    }
    $repoRoot = Split-Path -Parent (Split-Path -Parent $ScriptDir)
    $venvCand = Join-Path $repoRoot ".venv\Scripts\python.exe"
    if (Test-Path $venvCand) { return $venvCand }

    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source -and ($cmd.Source -notlike "*\WindowsApps\python.exe") -and (Test-Path $cmd.Source)) {
        return $cmd.Source
    }
    foreach ($cand in @("C:\Python314\python.exe", "C:\Python313\python.exe", "C:\Python312\python.exe", "C:\Python311\python.exe", "C:\Python310\python.exe")) {
        if (Test-Path $cand) { return $cand }
    }
    $localPyRoot = Join-Path $env:LOCALAPPDATA "Programs\Python"
    if (Test-Path $localPyRoot) {
        $found = Get-ChildItem -Path $localPyRoot -Directory -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending |
            ForEach-Object { Join-Path $_.FullName "python.exe" } |
            Where-Object { Test-Path $_ } |
            Select-Object -First 1
        if ($found) { return $found }
    }
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        try {
            $pOut = (& $pyLauncher.Source -3 -c "import sys; print(sys.executable)" 2>$null | Select-Object -First 1)
            if ($pOut -and (Test-Path $pOut.Trim())) { return $pOut.Trim() }
        } catch {}
    }
    if ($cmd -and $cmd.Source) { return $cmd.Source }
    throw "Python executable not found. Set QUINTESSENTIA_PYTHON or install Python 3.10+."
}

$PythonExe = Resolve-PythonExe

function Read-Utf8Json([string]$Path) {
    if (-not (Test-Path $Path)) { return $null }
    try {
        $fs = New-Object System.IO.FileStream(
            $Path,
            [System.IO.FileMode]::Open,
            [System.IO.FileAccess]::Read,
            ([System.IO.FileShare]::ReadWrite -bor [System.IO.FileShare]::Delete)
        )
        try {
            $sr = New-Object System.IO.StreamReader($fs, [System.Text.Encoding]::UTF8, $true)
            try { $raw = $sr.ReadToEnd() } finally { $sr.Dispose() }
        } finally {
            $fs.Dispose()
        }
        if ([string]::IsNullOrWhiteSpace($raw)) { return $null }
        return ($raw | ConvertFrom-Json)
    } catch {
        return $null
    }
}

function Write-Utf8Json([string]$Path, $Obj) {
    try {
        $dir = Split-Path -Parent $Path
        if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
        $tmp = "$Path.tmp"
        $json = $Obj | ConvertTo-Json -Depth 8
        [System.IO.File]::WriteAllText($tmp, $json, [System.Text.Encoding]::UTF8)
        Move-Item -Force $tmp $Path
    } catch {}
}

function Load-Config {
    $cfg = Read-Utf8Json $ConfigPath
    if (-not $cfg) {
        $wa = [System.Windows.SystemParameters]::WorkArea
        $defL = [int][Math]::Max($wa.Left + 16, $wa.Right - 680)
        $defT = [int][Math]::Max($wa.Top + 16, $wa.Top + 48)
        $cfg = [pscustomobject]@{
            refresh_ms = 250
            telemetry_file = (Join-Path (Join-Path $env:USERPROFILE ".quintessentia") "live_telemetry.json")
            lan_url = "http://127.0.0.1:8787/api/info"
            window = [pscustomobject]@{
                left = $defL
                top = $defT
                pin_mode = "float"
            }
        }
        Write-Utf8Json $ConfigPath $cfg
    }
    return $cfg
}

$script:Config = Load-Config

# Start silent zero-subprocess NVML Python collector bound to our PID (with watchdog auto-restart)
$script:BackendProc = $null
$script:LastBackendStart = [DateTime]::MinValue
function Start-HudBackend {
    try {
        if ($script:BackendProc -and -not $script:BackendProc.HasExited) { return }
    } catch {}
    if (([DateTime]::UtcNow - $script:LastBackendStart).TotalSeconds -lt 2.0) { return }
    $script:LastBackendStart = [DateTime]::UtcNow
    try {
        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = $PythonExe
        $psi.Arguments = "-X utf8 `"$BackendScript`" $PID"
        $psi.UseShellExecute = $false
        $psi.CreateNoWindow = $true
        $psi.WindowStyle = [System.Diagnostics.ProcessWindowStyle]::Hidden
        $psi.EnvironmentVariables["QUINTESSENTIA_HUD_CACHE"] = $CacheDir
        $psi.EnvironmentVariables["QUINTESSENTIA_HUD_CONFIG"] = $ConfigPath
        $psi.EnvironmentVariables["PYTHONDONTWRITEBYTECODE"] = "1"
        $script:BackendProc = [System.Diagnostics.Process]::Start($psi)
    } catch {}
}
Start-HudBackend

# Frozen brush cache => zero allocations during UI updates
$script:BrushCache = @{}
function Get-FrozenBrush([string]$Hex) {
    if ($script:BrushCache.ContainsKey($Hex)) { return $script:BrushCache[$Hex] }
    $b = (New-Object System.Windows.Media.BrushConverter).ConvertFromString($Hex)
    if ($b.CanFreeze) { $b.Freeze() }
    $script:BrushCache[$Hex] = $b
    return $b
}

function New-AreaGradient([string]$TopHex) {
    $g = New-Object System.Windows.Media.LinearGradientBrush
    $g.StartPoint = New-Object System.Windows.Point(0, 0)
    $g.EndPoint = New-Object System.Windows.Point(0, 1)
    $g.GradientStops.Add((New-Object System.Windows.Media.GradientStop([System.Windows.Media.ColorConverter]::ConvertFromString($TopHex), 0.0)))
    $g.GradientStops.Add((New-Object System.Windows.Media.GradientStop([System.Windows.Media.ColorConverter]::ConvertFromString("#00000000"), 1.0)))
    if ($g.CanFreeze) { $g.Freeze() }
    return $g
}

function New-CustomTrayIcon {
    $bmp = New-Object System.Drawing.Bitmap(32, 32)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $g.Clear([System.Drawing.Color]::Transparent)
    $bgBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(245, 11, 16, 24))
    $g.FillEllipse($bgBrush, 1, 1, 30, 30)
    $ringPen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(255, 56, 189, 248), 2.5)
    $g.DrawEllipse($ringPen, 2, 2, 28, 28)
    $font = New-Object System.Drawing.Font("Segoe UI", 14, [System.Drawing.FontStyle]::Bold, [System.Drawing.GraphicsUnit]::Pixel)
    $txtBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 240, 246, 252))
    $sf = New-Object System.Drawing.StringFormat
    $sf.Alignment = [System.Drawing.StringAlignment]::Center
    $sf.LineAlignment = [System.Drawing.StringAlignment]::Center
    $g.DrawString("Q", $font, $txtBrush, (New-Object System.Drawing.RectangleF(0, 0, 32, 31)), $sf)
    $hIcon = $bmp.GetHicon()
    $icon = [System.Drawing.Icon]::FromHandle($hIcon).Clone()
    [QHNative]::DestroyIcon($hIcon) | Out-Null
    $g.Dispose(); $bmp.Dispose(); $bgBrush.Dispose(); $ringPen.Dispose(); $font.Dispose(); $txtBrush.Dispose(); $sf.Dispose()
    return $icon
}

function Apply-PinMode([System.Windows.Window]$Win, [string]$Mode) {
    $helper = New-Object System.Windows.Interop.WindowInteropHelper($Win)
    $hwnd = $helper.Handle
    if ($hwnd -eq [IntPtr]::Zero) { return }
    $ex = [QHNative]::GetWindowLong($hwnd, [QHNative]::GWL_EXSTYLE)
    $style = if ($Mode -eq "desktop") { $ex -bor [QHNative]::WS_EX_TOOLWINDOW } else { $ex -band (-bnot [QHNative]::WS_EX_TOOLWINDOW) }
    [QHNative]::SetWindowLong($hwnd, [QHNative]::GWL_EXSTYLE, $style) | Out-Null
    $Win.ShowInTaskbar = ($Mode -ne "desktop")

    if ($Mode -eq "topmost") {
        $Win.Topmost = $true
        [QHNative]::SetWindowPos($hwnd, [QHNative]::HWND_TOPMOST, 0, 0, 0, 0,
            ([QHNative]::SWP_NOMOVE -bor [QHNative]::SWP_NOSIZE -bor [QHNative]::SWP_NOACTIVATE)) | Out-Null
    } elseif ($Mode -eq "desktop") {
        $Win.Topmost = $false
        [QHNative]::SetWindowPos($hwnd, [QHNative]::HWND_BOTTOM, 0, 0, 0, 0,
            ([QHNative]::SWP_NOMOVE -bor [QHNative]::SWP_NOSIZE -bor [QHNative]::SWP_NOACTIVATE)) | Out-Null
    } else {
        $Win.Topmost = $false
        [QHNative]::SetWindowPos($hwnd, [QHNative]::HWND_NOTOPMOST, 0, 0, 0, 0,
            ([QHNative]::SWP_NOMOVE -bor [QHNative]::SWP_NOSIZE -bor [QHNative]::SWP_NOACTIVATE)) | Out-Null
    }
}

function Clamp-WindowToVirtualScreen([System.Windows.Window]$Win) {
    $vl = [System.Windows.SystemParameters]::VirtualScreenLeft
    $vt = [System.Windows.SystemParameters]::VirtualScreenTop
    $vw = [System.Windows.SystemParameters]::VirtualScreenWidth
    $vh = [System.Windows.SystemParameters]::VirtualScreenHeight
    $maxL = $vl + $vw - $Win.Width
    $maxT = $vt + $vh - $Win.Height
    if ($Win.Left -lt $vl -or $Win.Left -gt $maxL) {
        $wa = [System.Windows.SystemParameters]::WorkArea
        $Win.Left = [Math]::Max($wa.Left + 12, $wa.Right - $Win.Width - 16)
    }
    if ($Win.Top -lt $vt -or $Win.Top -gt $maxT) {
        $wa = [System.Windows.SystemParameters]::WorkArea
        $Win.Top = [Math]::Max($wa.Top + 12, $wa.Top + 48)
    }
}

function Save-WindowState {
    try {
        if (-not $script:Config.window) {
            $script:Config | Add-Member -NotePropertyName window -NotePropertyValue ([pscustomobject]@{ left = 0; top = 0; pin_mode = "float" }) -Force
        }
        $script:Config.window.left = [int]$HudWin.Left
        $script:Config.window.top = [int]$HudWin.Top
        $script:Config.window.pin_mode = $script:PinMode
        Write-Utf8Json $ConfigPath $script:Config
    } catch {}
}

# ==============================================================================
# XAML: QUINTESSENTIA LIVE HUD (660 x 580 — crisp, zero clipping, Win + WSL2)
# ==============================================================================
[xml]$HudXaml = @"
<Window xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
        xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
        Title="Quintessentia Live HUD"
        Width="660" Height="580"
        WindowStyle="None" AllowsTransparency="True" Background="Transparent"
        ResizeMode="NoResize" ShowInTaskbar="False">
  <Window.Resources>
    <Style x:Key="PillBtn" TargetType="Button">
      <Setter Property="Background" Value="#2630363D"/>
      <Setter Property="Foreground" Value="#C9D1D9"/>
      <Setter Property="BorderBrush" Value="#4458A6FF"/>
      <Setter Property="BorderThickness" Value="1"/>
      <Setter Property="Padding" Value="7,3"/>
      <Setter Property="FontSize" Value="10.5"/>
      <Setter Property="FontFamily" Value="Segoe UI"/>
      <Setter Property="Cursor" Value="Hand"/>
      <Setter Property="Template">
        <Setter.Value>
          <ControlTemplate TargetType="Button">
            <Border x:Name="bd" Background="{TemplateBinding Background}" BorderBrush="{TemplateBinding BorderBrush}"
                    BorderThickness="{TemplateBinding BorderThickness}" CornerRadius="6" Padding="{TemplateBinding Padding}">
              <ContentPresenter HorizontalAlignment="Center" VerticalAlignment="Center"/>
            </Border>
            <ControlTemplate.Triggers>
              <Trigger Property="IsMouseOver" Value="True">
                <Setter TargetName="bd" Property="Background" Value="#5538BDF8"/>
                <Setter Property="Foreground" Value="#FFFFFF"/>
              </Trigger>
            </ControlTemplate.Triggers>
          </ControlTemplate>
        </Setter.Value>
      </Setter>
    </Style>
    <Style x:Key="CloseBtn" TargetType="Button" BasedOn="{StaticResource PillBtn}">
      <Setter Property="BorderBrush" Value="#44F43F5E"/>
      <Setter Property="Template">
        <Setter.Value>
          <ControlTemplate TargetType="Button">
            <Border x:Name="bd" Background="{TemplateBinding Background}" BorderBrush="{TemplateBinding BorderBrush}"
                    BorderThickness="{TemplateBinding BorderThickness}" CornerRadius="6" Padding="{TemplateBinding Padding}">
              <ContentPresenter HorizontalAlignment="Center" VerticalAlignment="Center"/>
            </Border>
            <ControlTemplate.Triggers>
              <Trigger Property="IsMouseOver" Value="True">
                <Setter TargetName="bd" Property="Background" Value="#99F43F5E"/>
                <Setter Property="Foreground" Value="#FFFFFF"/>
              </Trigger>
            </ControlTemplate.Triggers>
          </ControlTemplate>
        </Setter.Value>
      </Setter>
    </Style>
    <Style TargetType="ScrollBar">
      <Setter Property="Width" Value="6"/>
      <Setter Property="MinWidth" Value="6"/>
      <Setter Property="Background" Value="Transparent"/>
      <Setter Property="Template">
        <Setter.Value>
          <ControlTemplate TargetType="ScrollBar">
            <Grid Background="Transparent">
              <Track x:Name="PART_Track" IsDirectionReversed="True">
                <Track.Thumb>
                  <Thumb>
                    <Thumb.Template>
                      <ControlTemplate TargetType="Thumb">
                        <Border Background="#4438BDF8" CornerRadius="3" Margin="1,2"/>
                      </ControlTemplate>
                    </Thumb.Template>
                  </Thumb>
                </Track.Thumb>
              </Track>
            </Grid>
          </ControlTemplate>
        </Setter.Value>
      </Setter>
    </Style>
  </Window.Resources>

  <Border x:Name="OuterFrame" Margin="8" CornerRadius="14" Background="#EE0B1018" BorderBrush="#5538BDF8" BorderThickness="1">
    <Border.Effect>
      <DropShadowEffect Color="#000000" BlurRadius="14" ShadowDepth="3" Opacity="0.68"/>
    </Border.Effect>
    <Grid Margin="14,12,14,12">
      <Grid.RowDefinitions>
        <RowDefinition Height="Auto"/>
        <RowDefinition Height="Auto"/>
        <RowDefinition Height="Auto"/>
        <RowDefinition Height="Auto"/>
        <RowDefinition Height="Auto"/>
        <RowDefinition Height="*"/>
      </Grid.RowDefinitions>

      <!-- ROW 0: Header -->
      <Grid Grid.Row="0" Margin="0,0,0,8">
        <Grid.ColumnDefinitions>
          <ColumnDefinition Width="*"/>
          <ColumnDefinition Width="Auto"/>
        </Grid.ColumnDefinitions>
        <StackPanel Orientation="Horizontal" VerticalAlignment="Center">
          <Border x:Name="StatusPill" Background="#2264748B" BorderBrush="#5564748B" BorderThickness="1" CornerRadius="9" Padding="7,2" Margin="0,0,8,0">
            <StackPanel Orientation="Horizontal">
              <Ellipse x:Name="StatusDot" Width="7" Height="7" Fill="#94A3B8" Margin="0,0,5,0" VerticalAlignment="Center"/>
              <TextBlock x:Name="StatusTxt" Text="IDLE" FontFamily="Consolas" FontWeight="Bold" FontSize="10" Foreground="#CBD5E1"/>
            </StackPanel>
          </Border>
          <TextBlock Text="QUINTESSENTIA HUD" FontFamily="Segoe UI" FontWeight="SemiBold" FontSize="12.5" Foreground="#F0F6FC" VerticalAlignment="Center"/>
          <Border Background="#2238BDF8" CornerRadius="5" Padding="6,1" Margin="8,0,0,0" VerticalAlignment="Center">
            <TextBlock x:Name="ClockTxt" Text="00:00:00" FontFamily="Consolas" FontSize="10" Foreground="#7DD3FC"/>
          </Border>
          <Border x:Name="EnvBadge" Background="#2234D399" CornerRadius="5" Padding="6,1" Margin="6,0,0,0" VerticalAlignment="Center">
            <TextBlock x:Name="EnvTxt" Text="WIN native" FontFamily="Consolas" FontSize="10" Foreground="#6EE7B7"/>
          </Border>
          <Border x:Name="LanBadge" Background="#26A78BFA" CornerRadius="5" Padding="6,1" Margin="6,0,0,0" VerticalAlignment="Center" Visibility="Collapsed">
            <TextBlock x:Name="LanTxt" Text="LAN :8787" FontFamily="Consolas" FontSize="10" Foreground="#DDD6FE"/>
          </Border>
        </StackPanel>
        <StackPanel Grid.Column="1" Orientation="Horizontal">
          <Button x:Name="BtnClear" Content="CLR" Style="{StaticResource PillBtn}" Margin="0,0,4,0" ToolTip="Reset live graphs and session peaks"/>
          <Button x:Name="BtnPin" Content="FLT" Style="{StaticResource PillBtn}" Margin="0,0,4,0" ToolTip="Toggle Float / Topmost / Desktop"/>
          <Button x:Name="BtnHide" Content="_" Style="{StaticResource PillBtn}" Margin="0,0,4,0" Padding="7,3" ToolTip="Minimize to system tray"/>
          <Button x:Name="BtnExit" Content="x" Style="{StaticResource CloseBtn}" Padding="7,3" ToolTip="Close Quintessentia HUD"/>
        </StackPanel>
      </Grid>

      <!-- ROW 1: Active Model & Runtime Passport Strip -->
      <Border Grid.Row="1" Background="#88161B22" CornerRadius="8" BorderBrush="#2658A6FF" BorderThickness="1" Padding="10,6" Margin="0,0,0,9">
        <Grid>
          <Grid.ColumnDefinitions>
            <ColumnDefinition Width="*"/>
            <ColumnDefinition Width="Auto"/>
          </Grid.ColumnDefinitions>
          <StackPanel Orientation="Horizontal" VerticalAlignment="Center">
            <TextBlock x:Name="ModelTxt" Text="Quintessentia (standby)" FontFamily="Segoe UI" FontWeight="SemiBold" FontSize="12" Foreground="#F0F6FC" TextTrimming="CharacterEllipsis" MaxWidth="310"/>
            <TextBlock x:Name="ModelArchTxt" Text="" FontFamily="Consolas" FontSize="10" Foreground="#8B949E" Margin="8,2,0,0" TextTrimming="CharacterEllipsis" MaxWidth="140"/>
          </StackPanel>
          <StackPanel Grid.Column="1" Orientation="Horizontal" VerticalAlignment="Center">
            <Border x:Name="PlanesBadge" Background="#2634D399" BorderBrush="#4434D399" BorderThickness="1" CornerRadius="5" Padding="6,1" Margin="0,0,5,0">
              <TextBlock x:Name="PlanesBadgeTxt" Text="Q1 k=5" FontFamily="Consolas" FontWeight="Bold" FontSize="10" Foreground="#6EE7B7"/>
            </Border>
            <Border x:Name="ResBadge" Background="#2638BDF8" BorderBrush="#4438BDF8" BorderThickness="1" CornerRadius="5" Padding="6,1" Margin="0,0,5,0">
              <TextBlock x:Name="ResBadgeTxt" Text="STANDBY" FontFamily="Consolas" FontWeight="Bold" FontSize="10" Foreground="#7DD3FC"/>
            </Border>
            <Border Background="#26A78BFA" CornerRadius="5" Padding="6,1">
              <TextBlock x:Name="ModeBadgeTxt" Text="V4 mirror" FontFamily="Consolas" FontSize="10" Foreground="#DDD6FE"/>
            </Border>
          </StackPanel>
        </Grid>
      </Border>

      <!-- ROW 2: 4 KPI Tiles -->
      <UniformGrid Grid.Row="2" Rows="1" Columns="4" Margin="0,0,0,9">
        <Border Background="#CC141B24" BorderBrush="#3338BDF8" BorderThickness="1" CornerRadius="10" Padding="9,7" Margin="0,0,4,0">
          <StackPanel>
            <TextBlock Text="SPEED (TOK/S)" FontFamily="Consolas" FontSize="9.5" Foreground="#8B949E"/>
            <TextBlock x:Name="KpiToks" Text="0.00" FontFamily="Consolas" FontWeight="Bold" FontSize="20" Foreground="#38BDF8" Margin="0,1,0,1"/>
            <TextBlock x:Name="KpiToksSub" Text="avg 0.0 | pk 0.0" FontFamily="Consolas" FontSize="9" Foreground="#7DD3FC" TextTrimming="CharacterEllipsis"/>
          </StackPanel>
        </Border>
        <Border Background="#CC141B24" BorderBrush="#3358A6FF" BorderThickness="1" CornerRadius="10" Padding="9,7" Margin="2,0,2,0">
          <StackPanel>
            <TextBlock Text="TOKENS / TTFT" FontFamily="Consolas" FontSize="9.5" Foreground="#8B949E"/>
            <TextBlock x:Name="KpiTokens" Text="0" FontFamily="Consolas" FontWeight="Bold" FontSize="20" Foreground="#F0F6FC" Margin="0,1,0,1"/>
            <TextBlock x:Name="KpiTtft" Text="TTFT 0.00s | T0" FontFamily="Consolas" FontSize="9" Foreground="#94A3B8" TextTrimming="CharacterEllipsis"/>
          </StackPanel>
        </Border>
        <Border Background="#CC141B24" BorderBrush="#3334D399" BorderThickness="1" CornerRadius="10" Padding="9,7" Margin="2,0,2,0">
          <StackPanel>
            <TextBlock x:Name="KpiEngineLbl" Text="Q1 CUBES DEPTH" FontFamily="Consolas" FontSize="9.5" Foreground="#8B949E"/>
            <TextBlock x:Name="KpiEngineVal" Text="k=5" FontFamily="Consolas" FontWeight="Bold" FontSize="20" Foreground="#34D399" Margin="0,1,0,1"/>
            <TextBlock x:Name="KpiEngineSub" Text="lm5 | 0p" FontFamily="Consolas" FontSize="9" Foreground="#6EE7B7" TextTrimming="CharacterEllipsis"/>
          </StackPanel>
        </Border>
        <Border Background="#CC141B24" BorderBrush="#33FBBF24" BorderThickness="1" CornerRadius="10" Padding="9,7" Margin="4,0,0,0">
          <StackPanel>
            <TextBlock x:Name="KpiPowerLbl" Text="GPU POWER / PCIe" FontFamily="Consolas" FontSize="9.5" Foreground="#8B949E"/>
            <TextBlock x:Name="KpiPower" Text="0 W" FontFamily="Consolas" FontWeight="Bold" FontSize="20" Foreground="#FBBF24" Margin="0,1,0,1"/>
            <TextBlock x:Name="KpiGpuMeta" Text="0C | 0MHz" FontFamily="Consolas" FontSize="9" Foreground="#FDE68A" TextTrimming="CharacterEllipsis"/>
          </StackPanel>
        </Border>
      </UniformGrid>

      <!-- ROW 3: Dual Real-Time Charts -->
      <UniformGrid Grid.Row="3" Rows="1" Columns="2" Margin="0,0,0,9">
        <Border Background="#CC141B24" BorderBrush="#3358A6FF" BorderThickness="1" CornerRadius="10" Padding="10,7" Margin="0,0,4,0">
          <Grid>
            <Grid.RowDefinitions>
              <RowDefinition Height="Auto"/>
              <RowDefinition Height="72"/>
              <RowDefinition Height="Auto"/>
            </Grid.RowDefinitions>
            <Grid Grid.Row="0" Margin="0,0,0,4">
              <StackPanel Orientation="Horizontal">
                <Ellipse Width="6" Height="6" Fill="#38BDF8" Margin="0,0,5,0" VerticalAlignment="Center"/>
                <TextBlock Text="TOK/S" FontFamily="Consolas" FontWeight="Bold" FontSize="10" Foreground="#38BDF8"/>
                <Ellipse Width="6" Height="6" Fill="#A78BFA" Margin="10,0,5,0" VerticalAlignment="Center"/>
                <TextBlock x:Name="ChartLeftLegendB" Text="GPU %" FontFamily="Consolas" FontWeight="Bold" FontSize="10" Foreground="#A78BFA"/>
              </StackPanel>
              <TextBlock x:Name="ChartLeftScaleTxt" Text="max 10 t/s | GPU 0%" FontFamily="Consolas" FontSize="9.5" Foreground="#8B949E" HorizontalAlignment="Right"/>
            </Grid>
            <Border Grid.Row="1" Background="#660B1018" CornerRadius="6" Padding="3">
              <Canvas x:Name="SpeedCanvas" Width="278" Height="64" ClipToBounds="True"/>
            </Border>
            <TextBlock x:Name="ChartLeftFootTxt" Grid.Row="2" Text="Peak session: 0.00 tok/s  |  NVML direct" FontFamily="Consolas" FontSize="9" Foreground="#6E7681" Margin="0,4,0,0" TextTrimming="CharacterEllipsis"/>
          </Grid>
        </Border>

        <Border Background="#CC141B24" BorderBrush="#3358A6FF" BorderThickness="1" CornerRadius="10" Padding="10,7" Margin="4,0,0,0">
          <Grid>
            <Grid.RowDefinitions>
              <RowDefinition Height="Auto"/>
              <RowDefinition Height="72"/>
              <RowDefinition Height="Auto"/>
            </Grid.RowDefinitions>
            <Grid Grid.Row="0" Margin="0,0,0,4">
              <StackPanel Orientation="Horizontal">
                <Ellipse Width="6" Height="6" Fill="#34D399" Margin="0,0,5,0" VerticalAlignment="Center"/>
                <TextBlock Text="VRAM" FontFamily="Consolas" FontWeight="Bold" FontSize="10" Foreground="#34D399"/>
                <Ellipse Width="6" Height="6" Fill="#FBBF24" Margin="10,0,5,0" VerticalAlignment="Center"/>
                <TextBlock Text="PCIe MB/s" FontFamily="Consolas" FontWeight="Bold" FontSize="10" Foreground="#FBBF24"/>
              </StackPanel>
              <TextBlock x:Name="ChartRightScaleTxt" Text="PCIe 0 MB/s" FontFamily="Consolas" FontSize="9.5" Foreground="#8B949E" HorizontalAlignment="Right"/>
            </Grid>
            <Border Grid.Row="1" Background="#660B1018" CornerRadius="6" Padding="3">
              <Canvas x:Name="MemCanvas" Width="278" Height="64" ClipToBounds="True"/>
            </Border>
            <TextBlock x:Name="ChartRightFootTxt" Grid.Row="2" Text="TX 0 | RX 0 MB/s  |  Peak 0 MB/s" FontFamily="Consolas" FontSize="9" Foreground="#6E7681" Margin="0,4,0,0" TextTrimming="CharacterEllipsis"/>
          </Grid>
        </Border>
      </UniformGrid>

      <!-- ROW 4: Hardware Bars + Layer Residency + RSSF Multi-Scale Balance Strip -->
      <Border Grid.Row="4" Background="#CC141B24" BorderBrush="#3358A6FF" BorderThickness="1" CornerRadius="10" Padding="10,8" Margin="0,0,0,9">
        <StackPanel>
          <!-- VRAM Bar -->
          <Grid Margin="0,0,0,4">
            <TextBlock x:Name="GpuLineTxt" Text="GPU  |  GPU 0%  |  MEM 0%  |  Fan 0%" FontFamily="Consolas" FontSize="10" Foreground="#E6EDF3"/>
            <TextBlock x:Name="VramRightTxt" Text="0 / 0 MiB (0%)" FontFamily="Consolas" FontWeight="Bold" FontSize="10" Foreground="#7DD3FC" HorizontalAlignment="Right"/>
          </Grid>
          <Grid Height="10" Margin="0,0,0,7">
            <Border Background="#FF161F2E" CornerRadius="4"/>
            <Border x:Name="VramBarFill" HorizontalAlignment="Left" Width="4" Background="#FF38BDF8" CornerRadius="4"/>
          </Grid>

          <!-- Layer Residency Strip -->
          <Grid Margin="0,0,0,6">
            <Grid.ColumnDefinitions>
              <ColumnDefinition Width="Auto"/>
              <ColumnDefinition Width="*"/>
              <ColumnDefinition Width="Auto"/>
            </Grid.ColumnDefinitions>
            <TextBlock Text="LAYERS" FontFamily="Consolas" FontSize="9.5" Foreground="#8B949E" VerticalAlignment="Center" Margin="0,0,8,0"/>
            <Canvas x:Name="LayerStripCanvas" Grid.Column="1" Height="10" Width="468" HorizontalAlignment="Left" VerticalAlignment="Center" ClipToBounds="True"/>
            <TextBlock x:Name="LayerSummaryTxt" Grid.Column="2" Text="standby" FontFamily="Consolas" FontSize="9.5" Foreground="#6EE7B7" VerticalAlignment="Center" Margin="8,0,0,0"/>
          </Grid>

          <!-- Multi-Scale RSSF Balance & KV Archive Strip -->
          <Border x:Name="BalanceStrip" Background="#550F172A" BorderBrush="#2638BDF8" BorderThickness="1" CornerRadius="6" Padding="7,3" Margin="0,0,0,7">
            <Grid>
              <Grid.ColumnDefinitions>
                <ColumnDefinition Width="*"/>
                <ColumnDefinition Width="Auto"/>
              </Grid.ColumnDefinitions>
              <StackPanel Orientation="Horizontal" VerticalAlignment="Center">
                <TextBlock Text="BALANCE" FontFamily="Consolas" FontWeight="Bold" FontSize="9" Foreground="#8B949E" Margin="0,0,6,0" VerticalAlignment="Center"/>
                <TextBlock x:Name="BalanceValTxt" Text="standby (8-pair + EC/FDS)" FontFamily="Consolas" FontWeight="Bold" FontSize="9.5" Foreground="#7DD3FC" VerticalAlignment="Center"/>
                <Border x:Name="CollapseBadge" Background="#44F43F5E" BorderBrush="#99F43F5E" BorderThickness="1" CornerRadius="4" Padding="5,0" Margin="8,0,0,0" Visibility="Collapsed" VerticalAlignment="Center">
                  <TextBlock x:Name="CollapseBadgeTxt" Text="ARCHIVE WIPE" FontFamily="Consolas" FontWeight="Bold" FontSize="8.5" Foreground="#FDA4AF"/>
                </Border>
              </StackPanel>
              <TextBlock x:Name="BalanceRightTxt" Grid.Column="1" Text="KV --  |  M0..M8" FontFamily="Consolas" FontSize="9" Foreground="#94A3B8" VerticalAlignment="Center"/>
            </Grid>
          </Border>

          <!-- Dual System Bars: WSL2 VM RAM (or Host CPU Load on Native Windows) + Host RAM -->
          <UniformGrid Rows="1" Columns="2">
            <StackPanel Margin="0,0,8,0">
              <Grid Margin="0,0,0,3">
                <TextBlock x:Name="LeftSysLbl" Text="HOST CPU LOAD" FontFamily="Consolas" FontSize="9.5" Foreground="#8B949E"/>
                <TextBlock x:Name="WslRamTxt" Text="0% (Windows)" FontFamily="Consolas" FontSize="9.5" Foreground="#A78BFA" HorizontalAlignment="Right"/>
              </Grid>
              <Grid Height="7">
                <Border Background="#FF161F2E" CornerRadius="3"/>
                <Border x:Name="WslRamFill" HorizontalAlignment="Left" Width="3" Background="#FFA78BFA" CornerRadius="3"/>
              </Grid>
            </StackPanel>
            <StackPanel Margin="8,0,0,0">
              <Grid Margin="0,0,0,3">
                <TextBlock Text="HOST RAM" FontFamily="Consolas" FontSize="9.5" Foreground="#8B949E"/>
                <TextBlock x:Name="HostRamTxt" Text="0.0 / 0.0 GiB" FontFamily="Consolas" FontSize="9.5" Foreground="#94A3B8" HorizontalAlignment="Right"/>
              </Grid>
              <Grid Height="7">
                <Border Background="#FF161F2E" CornerRadius="3"/>
                <Border x:Name="HostRamFill" HorizontalAlignment="Left" Width="3" Background="#FF64748B" CornerRadius="3"/>
              </Grid>
            </StackPanel>
          </UniformGrid>
        </StackPanel>
      </Border>

      <!-- ROW 5: Live Output Stream -->
      <Border Grid.Row="5" Background="#A80F1724" BorderBrush="#2658A6FF" BorderThickness="1" CornerRadius="10" Padding="10,7">
        <Grid>
          <Grid.RowDefinitions>
            <RowDefinition Height="Auto"/>
            <RowDefinition Height="*"/>
          </Grid.RowDefinitions>
          <Grid Grid.Row="0" Margin="0,0,0,4">
            <TextBlock x:Name="StreamHeaderTxt" Text="LIVE OUTPUT STREAM" FontFamily="Consolas" FontWeight="Bold" FontSize="9.5" Foreground="#7DD3FC"/>
            <TextBlock x:Name="StreamSubTxt" Text="double-click to copy" FontFamily="Consolas" FontSize="9" Foreground="#6E7681" HorizontalAlignment="Right"/>
          </Grid>
          <ScrollViewer x:Name="StreamScroller" Grid.Row="1" VerticalScrollBarVisibility="Auto">
            <TextBlock x:Name="StreamBodyTxt" Text="Waiting for Quintessentia generation..." FontFamily="Segoe UI" FontSize="11.5" Foreground="#E6EDF3" TextWrapping="Wrap"/>
          </ScrollViewer>
        </Grid>
      </Border>
    </Grid>
  </Border>
</Window>
"@

$HudReader = New-Object System.Xml.XmlNodeReader $HudXaml
$HudWin = [Windows.Markup.XamlReader]::Load($HudReader)

$waInit = [System.Windows.SystemParameters]::WorkArea
$initLeft = if ($script:Config.window -and $null -ne $script:Config.window.left) { [double]$script:Config.window.left } else { [Math]::Max($waInit.Left + 16, $waInit.Right - 680) }
$initTop  = if ($script:Config.window -and $null -ne $script:Config.window.top)  { [double]$script:Config.window.top }  else { [Math]::Max($waInit.Top + 16, $waInit.Top + 48) }
$HudWin.Left = $initLeft
$HudWin.Top = $initTop
Clamp-WindowToVirtualScreen $HudWin
$script:PinMode = if ($script:Config.window -and $script:Config.window.pin_mode) { [string]$script:Config.window.pin_mode } else { "float" }

# Control bindings
$OuterFrame = $HudWin.FindName("OuterFrame")
$StatusPill = $HudWin.FindName("StatusPill")
$StatusDot = $HudWin.FindName("StatusDot")
$StatusTxt = $HudWin.FindName("StatusTxt")
$ClockTxt = $HudWin.FindName("ClockTxt")
$EnvBadge = $HudWin.FindName("EnvBadge")
$EnvTxt = $HudWin.FindName("EnvTxt")
$LanBadge = $HudWin.FindName("LanBadge")
$LanTxt = $HudWin.FindName("LanTxt")
$BtnClear = $HudWin.FindName("BtnClear")
$BtnPin = $HudWin.FindName("BtnPin")
$BtnHide = $HudWin.FindName("BtnHide")
$BtnExit = $HudWin.FindName("BtnExit")

$ModelTxt = $HudWin.FindName("ModelTxt")
$ModelArchTxt = $HudWin.FindName("ModelArchTxt")
$PlanesBadge = $HudWin.FindName("PlanesBadge")
$PlanesBadgeTxt = $HudWin.FindName("PlanesBadgeTxt")
$ResBadge = $HudWin.FindName("ResBadge")
$ResBadgeTxt = $HudWin.FindName("ResBadgeTxt")
$ModeBadgeTxt = $HudWin.FindName("ModeBadgeTxt")

$KpiToks = $HudWin.FindName("KpiToks")
$KpiToksSub = $HudWin.FindName("KpiToksSub")
$KpiTokens = $HudWin.FindName("KpiTokens")
$KpiTtft = $HudWin.FindName("KpiTtft")
$KpiEngineLbl = $HudWin.FindName("KpiEngineLbl")
$KpiEngineVal = $HudWin.FindName("KpiEngineVal")
$KpiEngineSub = $HudWin.FindName("KpiEngineSub")
$KpiPowerLbl = $HudWin.FindName("KpiPowerLbl")
$KpiPower = $HudWin.FindName("KpiPower")
$KpiGpuMeta = $HudWin.FindName("KpiGpuMeta")

$SpeedCanvas = $HudWin.FindName("SpeedCanvas")
$MemCanvas = $HudWin.FindName("MemCanvas")
$ChartLeftLegendB = $HudWin.FindName("ChartLeftLegendB")
$ChartLeftScaleTxt = $HudWin.FindName("ChartLeftScaleTxt")
$ChartLeftFootTxt = $HudWin.FindName("ChartLeftFootTxt")
$ChartRightScaleTxt = $HudWin.FindName("ChartRightScaleTxt")
$ChartRightFootTxt = $HudWin.FindName("ChartRightFootTxt")

$GpuLineTxt = $HudWin.FindName("GpuLineTxt")
$VramRightTxt = $HudWin.FindName("VramRightTxt")
$VramBarFill = $HudWin.FindName("VramBarFill")
$LayerStripCanvas = $HudWin.FindName("LayerStripCanvas")
$LayerSummaryTxt = $HudWin.FindName("LayerSummaryTxt")
$BalanceStrip = $HudWin.FindName("BalanceStrip")
$BalanceValTxt = $HudWin.FindName("BalanceValTxt")
$BalanceRightTxt = $HudWin.FindName("BalanceRightTxt")
$CollapseBadge = $HudWin.FindName("CollapseBadge")
$CollapseBadgeTxt = $HudWin.FindName("CollapseBadgeTxt")
$LeftSysLbl = $HudWin.FindName("LeftSysLbl")
$WslRamTxt = $HudWin.FindName("WslRamTxt")
$WslRamFill = $HudWin.FindName("WslRamFill")
$HostRamTxt = $HudWin.FindName("HostRamTxt")
$HostRamFill = $HudWin.FindName("HostRamFill")

$StreamHeaderTxt = $HudWin.FindName("StreamHeaderTxt")
$StreamSubTxt = $HudWin.FindName("StreamSubTxt")
$StreamScroller = $HudWin.FindName("StreamScroller")
$StreamBodyTxt = $HudWin.FindName("StreamBodyTxt")

# ==============================================================================
# PRE-BUILT IN-PLACE CHARTS (ZERO Children.Clear(), ZERO FLICKER, ZERO GC CHURN)
# ==============================================================================
function Init-DualChart([System.Windows.Controls.Canvas]$Canvas, [string]$ColorA, [string]$FillA, [string]$ColorB) {
    $W = 278.0; $H = 64.0
    foreach ($gy in @(16.0, 32.0, 48.0)) {
        $gl = New-Object System.Windows.Shapes.Line
        $gl.X1 = 0; $gl.X2 = $W; $gl.Y1 = $gy; $gl.Y2 = $gy
        $gl.Stroke = Get-FrozenBrush "#15FFFFFF"; $gl.StrokeThickness = 1
        $Canvas.Children.Add($gl) | Out-Null
    }
    $areaA = New-Object System.Windows.Shapes.Polygon
    $areaA.Fill = New-AreaGradient $FillA
    $Canvas.Children.Add($areaA) | Out-Null

    $polyB = New-Object System.Windows.Shapes.Polyline
    $polyB.Stroke = Get-FrozenBrush $ColorB
    $polyB.StrokeThickness = 1.4
    $polyB.StrokeLineJoin = [System.Windows.Media.PenLineJoin]::Round
    $Canvas.Children.Add($polyB) | Out-Null

    $polyA = New-Object System.Windows.Shapes.Polyline
    $polyA.Stroke = Get-FrozenBrush $ColorA
    $polyA.StrokeThickness = 2.0
    $polyA.StrokeLineJoin = [System.Windows.Media.PenLineJoin]::Round
    $Canvas.Children.Add($polyA) | Out-Null

    $dotA = New-Object System.Windows.Shapes.Ellipse
    $dotA.Width = 6; $dotA.Height = 6
    $dotA.Fill = Get-FrozenBrush $ColorA
    $dotA.Stroke = Get-FrozenBrush "#FF0B1018"
    $dotA.StrokeThickness = 1
    $dotA.Visibility = "Collapsed"
    $Canvas.Children.Add($dotA) | Out-Null

    return [pscustomobject]@{
        W = $W; H = $H; AreaA = $areaA; PolyA = $polyA; PolyB = $polyB; DotA = $dotA
    }
}

function Update-DualChart($Chart, $SeriesA, [double]$MaxA, $SeriesB, [double]$MaxB) {
    $W = $Chart.W; $H = $Chart.H; $pad = 4.0
    $spanH = $H - 2.0 * $pad

    if ($SeriesA -and $SeriesA.Count -ge 2) {
        $nA = $SeriesA.Count
        $invA = 1.0 / [Math]::Max(1e-6, $MaxA)
        $ptsA = New-Object System.Windows.Media.PointCollection($nA)
        $aPts = New-Object System.Windows.Media.PointCollection($nA + 2)
        $aPts.Add((New-Object System.Windows.Point(0, $H)))
        $lastX = 0.0; $lastY = $H - $pad
        for ($i = 0; $i -lt $nA; $i++) {
            $x = ($i / [double]($nA - 1)) * $W
            $v = [Math]::Max(0.0, [Math]::Min($MaxA, [double]$SeriesA[$i]))
            $y = ($H - $pad) - ($v * $invA * $spanH)
            $pt = New-Object System.Windows.Point($x, $y)
            $ptsA.Add($pt); $aPts.Add($pt)
            $lastX = $x; $lastY = $y
        }
        $aPts.Add((New-Object System.Windows.Point($W, $H)))
        $Chart.PolyA.Points = $ptsA
        $Chart.AreaA.Points = $aPts
        [System.Windows.Controls.Canvas]::SetLeft($Chart.DotA, $lastX - 3.0)
        [System.Windows.Controls.Canvas]::SetTop($Chart.DotA, $lastY - 3.0)
        $Chart.DotA.Visibility = "Visible"
    }

    if ($SeriesB -and $SeriesB.Count -ge 2) {
        $nB = $SeriesB.Count
        $invB = 1.0 / [Math]::Max(1e-6, $MaxB)
        $ptsB = New-Object System.Windows.Media.PointCollection($nB)
        for ($i = 0; $i -lt $nB; $i++) {
            $x = ($i / [double]($nB - 1)) * $W
            $v = [Math]::Max(0.0, [Math]::Min($MaxB, [double]$SeriesB[$i]))
            $y = ($H - $pad) - ($v * $invB * $spanH)
            $ptsB.Add((New-Object System.Windows.Point($x, $y)))
        }
        $Chart.PolyB.Points = $ptsB
    }
}

$script:SpeedChart = Init-DualChart $SpeedCanvas "#FF38BDF8" "#5538BDF8" "#CCA78BFA"
$script:MemChart   = Init-DualChart $MemCanvas   "#FF34D399" "#5034D399" "#EEFBBF24"

# Layer Residency Strip (rebuild only when layer count or residency changes)
$script:LastLayerSig = ""
function Update-LayerStrip([int]$Total, [int]$Resident, [bool]$IsActive) {
    $sig = "${Total}:${Resident}:${IsActive}"
    if ($sig -eq $script:LastLayerSig) { return }
    $script:LastLayerSig = $sig
    $LayerStripCanvas.Children.Clear()
    $W = 468.0
    if ($Total -le 0) {
        $bg = New-Object System.Windows.Shapes.Rectangle
        $bg.Width = $W; $bg.Height = 6; $bg.RadiusX = 3; $bg.RadiusY = 3
        $bg.Fill = Get-FrozenBrush "#FF161F2E"
        [System.Windows.Controls.Canvas]::SetTop($bg, 2)
        $LayerStripCanvas.Children.Add($bg) | Out-Null
        $LayerSummaryTxt.Text = "standby"
        $LayerSummaryTxt.Foreground = Get-FrozenBrush "#6E7681"
        return
    }
    $gap = if ($Total -gt 48) { 1.0 } else { 2.0 }
    $cellW = [Math]::Max(2.0, ($W - ($Total - 1) * $gap) / [double]$Total)
    $bRes = if ($IsActive) { Get-FrozenBrush "#FF34D399" } else { Get-FrozenBrush "#FF38BDF8" }
    $bStr = Get-FrozenBrush "#FFF59E0B"
    for ($i = 0; $i -lt $Total; $i++) {
        $rect = New-Object System.Windows.Shapes.Rectangle
        $rect.Width = $cellW; $rect.Height = 8; $rect.RadiusX = 2; $rect.RadiusY = 2
        $rect.Fill = if ($i -lt $Resident) { $bRes } else { $bStr }
        [System.Windows.Controls.Canvas]::SetLeft($rect, $i * ($cellW + $gap))
        [System.Windows.Controls.Canvas]::SetTop($rect, 1)
        $LayerStripCanvas.Children.Add($rect) | Out-Null
    }
    if ($Resident -ge $Total) {
        $LayerSummaryTxt.Text = "$Resident/$Total VRAM"
        $LayerSummaryTxt.Foreground = Get-FrozenBrush "#34D399"
    } else {
        $strCnt = $Total - $Resident
        $LayerSummaryTxt.Text = "$Resident/$Total +${strCnt}s"
        $LayerSummaryTxt.Foreground = Get-FrozenBrush "#FBBF24"
    }
}

# ==============================================================================
# SHUTDOWN & INTERACTION HANDLERS
# ==============================================================================
function Close-HudCleanly {
    Save-WindowState
    try { $AnimTimer.Stop() } catch {}
    try { $Tray.Visible = $false; $Tray.Dispose() } catch {}
    try { if ($script:BackendProc -and -not $script:BackendProc.HasExited) { $script:BackendProc.Kill() } } catch {}
    try { $HudWin.Close() } catch {}
    try { [System.Windows.Application]::Current.Shutdown() } catch {}
}

$HudWin.Add_MouseLeftButtonDown({
    if ($_.ButtonState -eq [System.Windows.Input.MouseButtonState]::Pressed) {
        try { $HudWin.DragMove(); Clamp-WindowToVirtualScreen $HudWin; Save-WindowState } catch {}
    }
})

$HudWin.Add_SourceInitialized({
    Apply-PinMode $HudWin $script:PinMode
    $BtnPin.Content = if ($script:PinMode -eq "topmost") { "TOP" } elseif ($script:PinMode -eq "desktop") { "DESK" } else { "FLT" }
})

$BtnPin.Add_Click({
    $script:PinMode = if ($script:PinMode -eq "float") { "topmost" } elseif ($script:PinMode -eq "topmost") { "desktop" } else { "float" }
    Apply-PinMode $HudWin $script:PinMode
    $BtnPin.Content = if ($script:PinMode -eq "topmost") { "TOP" } elseif ($script:PinMode -eq "desktop") { "DESK" } else { "FLT" }
    Save-WindowState
})

$BtnHide.Add_Click({ $HudWin.Hide() })
$BtnExit.Add_Click({ Close-HudCleanly })
$BtnClear.Add_Click({
    try { Set-Content -LiteralPath $ClearFlagPath -Value "1" -Encoding ASCII } catch {}
    $script:TargetToks = 0.0
    $script:CurToks = 0.0
})

$StreamBodyTxt.Add_MouseLeftButtonDown({
    param($s, $e)
    if ($e.ClickCount -eq 2) {
        $e.Handled = $true
        try {
            if ($StreamBodyTxt.Text) {
                [System.Windows.Clipboard]::SetText($StreamBodyTxt.Text)
                $StreamSubTxt.Text = "copied!"
            }
        } catch {}
    }
})

# WPF ContextMenu on right-click anywhere on window
$WinMenu = New-Object System.Windows.Controls.ContextMenu
$wmTop = New-Object System.Windows.Controls.MenuItem; $wmTop.Header = "Pin: Always on Top"
$wmFlt = New-Object System.Windows.Controls.MenuItem; $wmFlt.Header = "Pin: Floating Window"
$wmDsk = New-Object System.Windows.Controls.MenuItem; $wmDsk.Header = "Pin: Desktop Wallpaper"
$wmClr = New-Object System.Windows.Controls.MenuItem; $wmClr.Header = "Clear Graphs & Peaks"
$wmHid = New-Object System.Windows.Controls.MenuItem; $wmHid.Header = "Minimize to Tray"
$wmExt = New-Object System.Windows.Controls.MenuItem; $wmExt.Header = "Exit Quintessentia HUD"
$wmTop.Add_Click({ $script:PinMode = "topmost"; Apply-PinMode $HudWin "topmost"; $BtnPin.Content = "TOP"; Save-WindowState })
$wmFlt.Add_Click({ $script:PinMode = "float";   Apply-PinMode $HudWin "float";   $BtnPin.Content = "FLT"; Save-WindowState })
$wmDsk.Add_Click({ $script:PinMode = "desktop"; Apply-PinMode $HudWin "desktop"; $BtnPin.Content = "DESK"; Save-WindowState })
$wmClr.Add_Click({ try { Set-Content -LiteralPath $ClearFlagPath -Value "1" -Encoding ASCII } catch {} })
$wmHid.Add_Click({ $HudWin.Hide() })
$wmExt.Add_Click({ Close-HudCleanly })
foreach ($item in @($wmTop, $wmFlt, $wmDsk, (New-Object System.Windows.Controls.Separator), $wmClr, $wmHid, $wmExt)) {
    $WinMenu.Items.Add($item) | Out-Null
}
$OuterFrame.ContextMenu = $WinMenu

function Capture-HudScreenshot {
    try {
        $HudWin.UpdateLayout()
        $w = [Math]::Max(1, [int]$HudWin.ActualWidth)
        $h = [Math]::Max(1, [int]$HudWin.ActualHeight)
        $rtb = New-Object System.Windows.Media.Imaging.RenderTargetBitmap($w, $h, 96, 96, [System.Windows.Media.PixelFormats]::Pbgra32)
        $rtb.Render($HudWin)
        $enc = New-Object System.Windows.Media.Imaging.PngBitmapEncoder
        $enc.Frames.Add([System.Windows.Media.Imaging.BitmapFrame]::Create($rtb))
        $pngPath = Join-Path $CacheDir "widget-render.png"
        $fs = [IO.File]::Create($pngPath)
        try { $enc.Save($fs) } finally { $fs.Dispose() }
        $hwnd = (New-Object System.Windows.Interop.WindowInteropHelper($HudWin)).Handle
        Write-Utf8Json (Join-Path $CacheDir "window-state.json") ([pscustomobject]@{
            visible = $HudWin.IsVisible
            nativeVisible = [QHNative]::IsWindowVisible($hwnd)
            left = [int]$HudWin.Left
            top = [int]$HudWin.Top
            width = $w
            height = $h
            pid = $PID
            pin_mode = $script:PinMode
        })
    } catch {}
}

# ==============================================================================
# SMOOTH 20 FPS ANIMATION & 4 Hz DATA REFRESH LOOP
# ==============================================================================
$script:LastHudTs = -1.0
$script:LastStreamText = ""
$script:AnimTick = 0
$script:TargetVramW = 4.0; $script:CurVramW = 4.0
$script:TargetWslW  = 3.0; $script:CurWslW  = 3.0
$script:TargetHostW = 3.0; $script:CurHostW = 3.0
$script:TargetToks  = 0.0; $script:CurToks  = 0.0
$script:IsGenerating = $false
$script:LastPhase = "IDLE"

$AnimTimer = New-Object System.Windows.Threading.DispatcherTimer
$AnimTimer.Interval = [TimeSpan]::FromMilliseconds(50)
$AnimTimer.Add_Tick({
    $script:AnimTick++

    # Smooth bar width interpolation & speedometer needle decay (20 FPS)
    $script:CurVramW += ($script:TargetVramW - $script:CurVramW) * 0.30
    $script:CurWslW  += ($script:TargetWslW  - $script:CurWslW)  * 0.30
    $script:CurHostW += ($script:TargetHostW - $script:CurHostW) * 0.30
    $VramBarFill.Width = [Math]::Max(4.0, $script:CurVramW)
    $WslRamFill.Width  = [Math]::Max(3.0, $script:CurWslW)
    $HostRamFill.Width = [Math]::Max(3.0, $script:CurHostW)

    if ($script:TargetToks -le 0.01 -and $script:CurToks -gt 0.0) {
        $script:CurToks *= 0.80
        if ($script:CurToks -lt 0.05) { $script:CurToks = 0.0 }
    } else {
        $script:CurToks += ($script:TargetToks - $script:CurToks) * 0.35
    }
    $KpiToks.Text = Fmt "{0:F2}" @($script:CurToks)
    if ($script:IsGenerating -and ($script:LastPhase -eq "GENERATING" -or $script:LastPhase -eq "THINKING")) {
        $KpiToks.Foreground = Get-FrozenBrush "#38BDF8"
    } elseif ($script:IsGenerating -and $script:LastPhase -eq "PREFILL") {
        $KpiToks.Foreground = Get-FrozenBrush "#FBBF24"
    } elseif ($script:CurToks -gt 0.05) {
        $KpiToks.Foreground = Get-FrozenBrush "#7DD3FC"
    } else {
        $KpiToks.Foreground = Get-FrozenBrush "#64748B"
    }

    if ($script:IsGenerating) {
        $StatusDot.Opacity = 0.45 + 0.55 * [Math]::Abs([Math]::Sin($script:AnimTick * 0.22))
    } else {
        $StatusDot.Opacity = 1.0
    }

    if (($script:AnimTick % 5) -ne 0) { return }

    if (Test-Path $StopFlagPath) {
        try { Remove-Item -Force $StopFlagPath -ErrorAction SilentlyContinue } catch {}
        Close-HudCleanly
        return
    }

    if (Test-Path $ShowFlagPath) {
        try { Remove-Item -Force $ShowFlagPath -ErrorAction SilentlyContinue } catch {}
        $HudWin.Show()
        Clamp-WindowToVirtualScreen $HudWin
        $HudWin.Topmost = $true
        try { $HudWin.Activate() | Out-Null } catch {}
        $hwnd = (New-Object System.Windows.Interop.WindowInteropHelper($HudWin)).Handle
        if ($hwnd -ne [IntPtr]::Zero) {
            [QHNative]::ShowWindow($hwnd, 5) | Out-Null
            [QHNative]::SetForegroundWindow($hwnd) | Out-Null
        }
        Apply-PinMode $HudWin $script:PinMode
    }

    if (Test-Path $CaptureFlagPath) {
        try { Remove-Item -Force $CaptureFlagPath -ErrorAction SilentlyContinue } catch {}
        Capture-HudScreenshot
    }

    if (($script:AnimTick % 40) -eq 0) {
        $dead = $true
        try { $dead = ($null -eq $script:BackendProc) -or $script:BackendProc.HasExited } catch { $dead = $true }
        if ($dead) { Start-HudBackend }
    }

    $hud = Read-Utf8Json $HudJsonPath
    if (-not $hud -or ([double]$hud.updated_ts -eq $script:LastHudTs)) { return }
    $script:LastHudTs = [double]$hud.updated_ts
    $ClockTxt.Text = [string]$hud.updated_at

    $g = $hud.gpu
    $s = $hud.sys
    $r = $hud.run
    $lan = $hud.lan

    $isAct = [bool]$r.active
    $phase = ([string]$r.phase).ToUpper()
    if (-not $phase) { $phase = "IDLE" }
    $script:IsGenerating = $isAct

    # Status pill & outer frame accent
    $StatusTxt.Text = $phase
    if ($phase -eq "GENERATING") {
        $StatusDot.Fill = Get-FrozenBrush "#10B981"; $StatusTxt.Foreground = Get-FrozenBrush "#34D399"
        $StatusPill.Background = Get-FrozenBrush "#2610B981"; $StatusPill.BorderBrush = Get-FrozenBrush "#7710B981"
        $OuterFrame.BorderBrush = Get-FrozenBrush "#7734D399"
    } elseif ($phase -eq "THINKING") {
        $StatusDot.Fill = Get-FrozenBrush "#A78BFA"; $StatusTxt.Foreground = Get-FrozenBrush "#DDD6FE"
        $StatusPill.Background = Get-FrozenBrush "#26A78BFA"; $StatusPill.BorderBrush = Get-FrozenBrush "#77A78BFA"
        $OuterFrame.BorderBrush = Get-FrozenBrush "#77A78BFA"
    } elseif ($phase -eq "PREFILL" -or $phase -eq "LOADING") {
        $StatusDot.Fill = Get-FrozenBrush "#F59E0B"; $StatusTxt.Foreground = Get-FrozenBrush "#FCD34D"
        $StatusPill.Background = Get-FrozenBrush "#26F59E0B"; $StatusPill.BorderBrush = Get-FrozenBrush "#77F59E0B"
        $OuterFrame.BorderBrush = Get-FrozenBrush "#77F59E0B"
    } elseif ($phase -eq "DONE" -or $phase -eq "READY") {
        $StatusDot.Fill = Get-FrozenBrush "#38BDF8"; $StatusTxt.Foreground = Get-FrozenBrush "#7DD3FC"
        $StatusPill.Background = Get-FrozenBrush "#2638BDF8"; $StatusPill.BorderBrush = Get-FrozenBrush "#6638BDF8"
        $OuterFrame.BorderBrush = Get-FrozenBrush "#5538BDF8"
    } else {
        $StatusDot.Fill = Get-FrozenBrush "#64748B"; $StatusTxt.Foreground = Get-FrozenBrush "#94A3B8"
        $StatusPill.Background = Get-FrozenBrush "#2264748B"; $StatusPill.BorderBrush = Get-FrozenBrush "#5564748B"
        $OuterFrame.BorderBrush = Get-FrozenBrush "#4438BDF8"
    }

    # Environment badge: truthful Windows Native vs WSL2
    $runEnv = [string]$r.env
    $isWinNativeRun = ($runEnv -eq "Windows") -or (-not [bool]$s.wsl_running)
    if ($isWinNativeRun) {
        $EnvTxt.Text = "WIN native"
        $EnvBadge.Background = Get-FrozenBrush "#2634D399"; $EnvTxt.Foreground = Get-FrozenBrush "#6EE7B7"
    } else {
        $prio = [string]$s.wsl_priority
        $EnvTxt.Text = "WSL2 $prio"
        if ($prio -eq "High" -or $prio -eq "AboveNorm") {
            $EnvBadge.Background = Get-FrozenBrush "#2634D399"; $EnvTxt.Foreground = Get-FrozenBrush "#6EE7B7"
        } else {
            $EnvBadge.Background = Get-FrozenBrush "#2238BDF8"; $EnvTxt.Foreground = Get-FrozenBrush "#7DD3FC"
        }
    }

    if ($lan -and [bool]$lan.online) {
        $lanPort = if ($lan.port) { [int]$lan.port } else { 8787 }
        $LanBadge.Visibility = "Visible"
        $LanTxt.Text = "LAN :$lanPort"
    } else {
        $LanBadge.Visibility = "Collapsed"
    }

    # Model passport
    $deg = [char]0x00B0
    $dotSep = [char]0x00B7
    $deltaSym = [char]0x0394
    $ModelTxt.Text = [string]$r.model
    $archStr = @()
    if ($r.model_type) { $archStr += [string]$r.model_type }
    if ($r.format) { $archStr += ([string]$r.format).ToUpper() }
    $ModelArchTxt.Text = ($archStr -join " $dotSep ")

    $kVal = [int]$r.q1_planes
    $kReq = [int]$r.q1_planes_requested
    $isHalved = [bool]$r.q1_planes_auto_halved
    if ($kVal -gt 0) {
        if ($isHalved -and $kReq -gt $kVal) {
            $PlanesBadgeTxt.Text = "Q1 k=$kVal (auto)"
            $PlanesBadge.Background = Get-FrozenBrush "#26F59E0B"; $PlanesBadge.BorderBrush = Get-FrozenBrush "#66F59E0B"
            $PlanesBadgeTxt.Foreground = Get-FrozenBrush "#FDE68A"
        } else {
            $PlanesBadgeTxt.Text = "Q1 k=$kVal"
            $PlanesBadge.Background = Get-FrozenBrush "#2634D399"; $PlanesBadge.BorderBrush = Get-FrozenBrush "#4434D399"
            $PlanesBadgeTxt.Foreground = Get-FrozenBrush "#6EE7B7"
        }
    } else {
        $wfLabel = [string]$r.weight_format
        $PlanesBadgeTxt.Text = if ($wfLabel) { $wfLabel } else { "auto" }
    }

    $lTot = [int]$r.layers_total
    $lRes = [int]$r.resident_layers
    if ($lTot -gt 0) {
        if ($lRes -ge $lTot) {
            $ResBadgeTxt.Text = "$lRes/$lTot GPU"
            $ResBadge.Background = Get-FrozenBrush "#2634D399"; $ResBadge.BorderBrush = Get-FrozenBrush "#4434D399"
            $ResBadgeTxt.Foreground = Get-FrozenBrush "#6EE7B7"
        } elseif ($lRes -eq 0) {
            $ResBadgeTxt.Text = "$lTot CPU"
            $ResBadge.Background = Get-FrozenBrush "#2638BDF8"; $ResBadge.BorderBrush = Get-FrozenBrush "#4438BDF8"
            $ResBadgeTxt.Foreground = Get-FrozenBrush "#7DD3FC"
        } else {
            $ResBadgeTxt.Text = "$lRes/$lTot STREAM"
            $ResBadge.Background = Get-FrozenBrush "#26F59E0B"; $ResBadge.BorderBrush = Get-FrozenBrush "#55F59E0B"
            $ResBadgeTxt.Foreground = Get-FrozenBrush "#FCD34D"
        }
    } else {
        $ResBadgeTxt.Text = "STANDBY"
        $ResBadge.Background = Get-FrozenBrush "#2264748B"; $ResBadge.BorderBrush = Get-FrozenBrush "#4464748B"
        $ResBadgeTxt.Foreground = Get-FrozenBrush "#94A3B8"
    }
    $fb = [string]$r.feedback_mode
    $ModeBadgeTxt.Text = if ($fb -eq "mirror_scales") { "V4 mirror" } elseif ($fb -and $fb -ne "off") { $fb } else { "standard" }

    # 4 KPI tiles (invariant culture, compact non-clipping text!)
    $script:LastPhase = $phase
    $isDecoding = $isAct -and ($phase -eq "GENERATING" -or $phase -eq "THINKING")
    $script:TargetToks = if ($isDecoding) { [double]$r.tok_s } else { 0.0 }
    $sessPk = [double]$hud.session_peak_tok_s
    if ($isDecoding) {
        $KpiToksSub.Text = Fmt "live | pk {0:F1}" @($sessPk)
        $KpiToksSub.Foreground = Get-FrozenBrush "#38BDF8"
    } elseif ($isAct -and $phase -eq "PREFILL") {
        $ppSpd = if ($null -ne $r.prefill_tok_s) { [double]$r.prefill_tok_s } else { 0.0 }
        $KpiToksSub.Text = if ($ppSpd -gt 1.0) { Fmt "pp {0:F0} | pk {1:F1}" @($ppSpd, $sessPk) } else { Fmt "prefill | pk {0:F1}" @($sessPk) }
        $KpiToksSub.Foreground = Get-FrozenBrush "#FCD34D"
    } elseif ([double]$r.avg_tok_s -gt 0.01) {
        $KpiToksSub.Text = Fmt "avg {0:F1} | pk {1:F1}" @([double]$r.avg_tok_s, $sessPk)
        $KpiToksSub.Foreground = Get-FrozenBrush "#7DD3FC"
    } else {
        $KpiToksSub.Text = Fmt "idle | pk {0:F1}" @($sessPk)
        $KpiToksSub.Foreground = Get-FrozenBrush "#64748B"
    }

    $KpiTokens.Text = Fmt "{0}" @([int]$r.tokens)
    $ttftS = ([double]$r.ttft_ms) / 1000.0
    $KpiTtft.Text = Fmt "TTFT {0:F2}s | T{1}" @($ttftS, [int]$r.turn)

    if ([bool]$r.is_moe) {
        $KpiEngineLbl.Text = "MOE Q1 CACHE"
        $KpiEngineVal.Text = "$($r.moe_hits)/$($r.moe_misses)"
        $KpiEngineSub.Text = "k=$kVal $dotSep $($r.num_experts)e $dotSep h2:$($r.moe_host_hits)"
    } else {
        $KpiEngineLbl.Text = "Q1 CUBES DEPTH"
        $KpiEngineVal.Text = if ($kVal -gt 0) { "k=$kVal" } else { [string]$r.weight_format }
        $diffVal = [int]$r.diff_calls
        $diffStr = if ($diffVal -ge 1000000) { Fmt "{0:F1}M{1}" @(($diffVal / 1000000.0), $deltaSym) } elseif ($diffVal -ge 1000) { Fmt "{0:F0}k{1}" @(($diffVal / 1000.0), $deltaSym) } elseif ($diffVal -gt 0) { "${diffVal}${deltaSym}" } else { "r$([int]$r.root_bits)b" }
        $KpiEngineSub.Text = "lm$([int]$r.lm_head_planes) $dotSep $([int]$r.q1_projections)p $dotSep $diffStr"
        $KpiEngineSub.ToolTip = "LM head planes: $([int]$r.lm_head_planes) | Q1 projections: $([int]$r.q1_projections) | Differential Q1 calls: $diffVal"
    }

    $hasNvml = [bool]$g.nvml_direct
    if ($hasNvml) {
        $KpiPowerLbl.Text = "GPU POWER / PCIe"
        $KpiPower.Text = Fmt "{0:F0} W" @([double]$g.power_w)
        $KpiGpuMeta.Text = Fmt "{0:F0}{1}C | {2:F0}M | {3:F0}MB/s" @([double]$g.temp_c, $deg, [double]$g.clock_mhz, [double]$g.pcie_total_mbs)
        $KpiGpuMeta.ToolTip = Fmt "Temp: {0:F0}{1}C | Core: {2:F0} MHz | PCIe: {3:F1} MB/s" @([double]$g.temp_c, $deg, [double]$g.clock_mhz, [double]$g.pcie_total_mbs)
    } else {
        $KpiPowerLbl.Text = "HOST CPU / RAM"
        $KpiPower.Text = Fmt "{0:F0}%" @([double]$s.cpu_pct)
        $KpiGpuMeta.Text = Fmt "RAM {0:F1}/{1:F1} GiB" @([double]$s.ram_used_gib, [double]$s.ram_total_gib)
    }

    # Update Dual Live Charts in-place
    $tokHist = @($hud.tok_history)
    $gpuHist = @($hud.gpu_history)
    $vramHist = @($hud.vram_history)
    $pcieHist = @($hud.pcie_history)

    $maxTok = 10.0
    foreach ($tv in $tokHist) { if ([double]$tv -gt $maxTok) { $maxTok = [double]$tv } }
    $maxTok = [Math]::Ceiling($maxTok / 5.0) * 5.0
    Update-DualChart $script:SpeedChart $tokHist $maxTok $gpuHist 100.0
    $ChartLeftLegendB.Text = if ($hasNvml) { "GPU %" } else { "CPU %" }
    $utilDisp = if ($hasNvml) { [double]$g.util_pct } else { [double]$s.cpu_pct }
    $ChartLeftScaleTxt.Text = Fmt "max {0:F0} t/s | {1} {2:F0}%" @($maxTok, $(if ($hasNvml) { "GPU" } else { "CPU" }), $utilDisp)
    $ChartLeftFootTxt.Text = Fmt "Peak session: {0:F2} tok/s  |  {1}" @($sessPk, $(if ($hasNvml) { "NVML direct" } else { "Win32 CPU" }))

    $vTot = [Math]::Max(1.0, [double]$g.vram_total_mib)
    $maxPcie = 250.0
    foreach ($pv in $pcieHist) { if ([double]$pv -gt $maxPcie) { $maxPcie = [double]$pv } }
    $maxPcie = [Math]::Ceiling($maxPcie / 250.0) * 250.0
    Update-DualChart $script:MemChart $vramHist $vTot $pcieHist $maxPcie
    $ChartRightScaleTxt.Text = Fmt "PCIe {0:F0} MB/s" @([double]$g.pcie_total_mbs)
    $ChartRightFootTxt.Text = Fmt "TX {0:F0} | RX {1:F0} MB/s  |  Peak {2:F0} MB/s" @([double]$g.pcie_tx_mbs, [double]$g.pcie_rx_mbs, [double]$hud.session_peak_pcie_mbs)

    # Hardware bars
    if ($hasNvml -and [double]$g.vram_total_mib -gt 1.0) {
        $vUsed = [double]$g.vram_used_mib
        $vRatio = [Math]::Min(1.0, [Math]::Max(0.0, $vUsed / $vTot))
        $GpuLineTxt.Text = Fmt "{0}  |  GPU {1:F0}%  |  MEM {2:F0}%  |  Fan {3}%  |  {4:F0}/{5:F0}W" @(
            $g.name, [double]$g.util_pct, [double]$g.mem_util_pct, [int]$g.fan_pct, [double]$g.power_w, [double]$g.power_cap_w
        )
        $VramRightTxt.Text = Fmt "{0:F0} / {1:F0} MiB ({2:F0}%)" @($vUsed, $vTot, ($vRatio * 100.0))
        $script:TargetVramW = 596.0 * $vRatio
        $VramBarFill.Background = if ($vRatio -gt 0.88) { Get-FrozenBrush "#FFF43F5E" } elseif ($vRatio -gt 0.72) { Get-FrozenBrush "#FFF59E0B" } else { Get-FrozenBrush "#FF38BDF8" }
    } else {
        $cpuPct = [double]$s.cpu_pct
        $GpuLineTxt.Text = "CPU Execution Mode  |  No NVIDIA NVML GPU detected"
        $VramRightTxt.Text = Fmt "CPU {0:F0}%" @($cpuPct)
        $script:TargetVramW = 596.0 * ([Math]::Min(1.0, [Math]::Max(0.0, $cpuPct / 100.0)))
        $VramBarFill.Background = Get-FrozenBrush "#FF38BDF8"
    }

    Update-LayerStrip $lTot $lRes $isAct

    # Multi-Scale RSSF Balance & KV Archive Strip
    $d3Val = $r.rssf_d3_margin
    $regStr = [string]$r.rssf_regime
    $lossArch = [bool]$r.rssf_loss_of_archive
    $sbCnt = [int]$r.rssf_scale_breaks
    $dmpCnt = [int]$r.rssf_damped_chunks
    $kvG = [int]$r.kv_gpu_layers
    $kvO = [int]$r.kv_offloaded_layers
    $kvP = [int]$r.kv_promoted_layers
    $scalesArr = @($r.rssf_active_scales)

    if ($null -ne $d3Val -and "$d3Val" -ne "") {
        $d3Num = [double]$d3Val
        $dmpTag = if ($dmpCnt -gt 0) { " | damp:$dmpCnt" } else { "" }
        $BalanceValTxt.Text = Fmt "d3={0:F2} ({1}){2}" @($d3Num, $(if ($regStr) { $regStr } else { "LIVE" }), $dmpTag)
        if ($regStr -eq "SCALE_BREAK" -or $d3Num -le 0.0 -or $lossArch) {
            $BalanceValTxt.Foreground = Get-FrozenBrush "#FDA4AF"
            $BalanceStrip.BorderBrush = Get-FrozenBrush "#77F43F5E"
        } elseif ($regStr -eq "COMPRESS" -or $dmpCnt -gt 0) {
            $BalanceValTxt.Foreground = Get-FrozenBrush "#FCD34D"
            $BalanceStrip.BorderBrush = Get-FrozenBrush "#55F59E0B"
        } else {
            $BalanceValTxt.Foreground = Get-FrozenBrush "#6EE7B7"
            $BalanceStrip.BorderBrush = Get-FrozenBrush "#3334D399"
        }
    } else {
        $BalanceValTxt.Text = "8-pair + EC/FDS ready"
        $BalanceValTxt.Foreground = Get-FrozenBrush "#7DD3FC"
        $BalanceStrip.BorderBrush = Get-FrozenBrush "#2638BDF8"
    }

    if ($lossArch) {
        $CollapseBadge.Visibility = "Visible"
        $CollapseBadgeTxt.Text = "ARCHIVE WIPE"
    } elseif ($sbCnt -gt 0 -or $regStr -eq "SCALE_BREAK") {
        $CollapseBadge.Visibility = "Visible"
        $CollapseBadgeTxt.Text = "SCALE BREAK"
    } else {
        $CollapseBadge.Visibility = "Collapsed"
    }

    $kvPart = if (($kvG + $kvO) -gt 0) {
        $pTag = if ($kvP -gt 0) { " +${kvP}p" } else { "" }
        "KV ${kvG}G/${kvO}R${pTag}"
    } else { "KV auto" }
    $scPart = if ($scalesArr.Count -gt 0) { ($scalesArr -join $dotSep) } else { "M0..M8" }
    $BalanceRightTxt.Text = "$kvPart  |  $scPart"

    # Left bottom bar: WSL2 VM RAM when WSL2 is active, else live Windows Host CPU Load!
    if ([bool]$s.wsl_running) {
        $LeftSysLbl.Text = "WSL2 VM RAM"
        $wslLim = if ([double]$s.wsl_ram_limit_gib -gt 0.1) { [double]$s.wsl_ram_limit_gib } else { 10.0 }
        $wslUsed = [double]$s.wsl_ram_used_gib
        $wslRatio = [Math]::Min(1.0, [Math]::Max(0.0, $wslUsed / $wslLim))
        $WslRamTxt.Text = Fmt "{0:F2} / {1:F1} GiB ({2})" @($wslUsed, $wslLim, [string]$s.wsl_priority)
        $script:TargetWslW = 290.0 * $wslRatio
    } else {
        $LeftSysLbl.Text = "HOST CPU LOAD"
        $cpuPct = [double]$s.cpu_pct
        $cpuRatio = [Math]::Min(1.0, [Math]::Max(0.0, $cpuPct / 100.0))
        $WslRamTxt.Text = Fmt "{0:F0}% (Windows)" @($cpuPct)
        $script:TargetWslW = 290.0 * $cpuRatio
    }

    $hTot = [Math]::Max(0.1, [double]$s.ram_total_gib)
    $hUsed = [double]$s.ram_used_gib
    $hRatio = [Math]::Min(1.0, [Math]::Max(0.0, $hUsed / $hTot))
    $HostRamTxt.Text = Fmt "{0:F1} / {1:F1} GiB ({2:F0}%)" @($hUsed, $hTot, ($hRatio * 100.0))
    $script:TargetHostW = 290.0 * $hRatio

    # Live Output Stream
    $ans = [string]$r.answer_preview
    $thk = [string]$r.thinking_preview
    $pfDone = if ($null -ne $r.prefill_done) { [int]$r.prefill_done } else { 0 }
    $pfTot = if ($null -ne $r.prefill_total -and [int]$r.prefill_total -gt 0) { [int]$r.prefill_total } else { [int]$r.prompt_tokens }
    $pfMsg = if ($pfTot -gt 0 -and $pfDone -gt 0 -and $pfDone -lt $pfTot) {
        $pfPct = [Math]::Round((100.0 * $pfDone) / [double]$pfTot)
        "Prefill prompt: $pfDone / $pfTot tokens ($pfPct%)..."
    } else {
        "Prefill prompt ($pfTot tokens)..."
    }
    $body = if ($phase -eq "THINKING" -and $thk) { "[thinking] $thk" } elseif ($ans) { $ans } elseif ($phase -eq "PREFILL") { $pfMsg } else { "Waiting for Quintessentia generation..." }
    if ($body -ne $script:LastStreamText) {
        $script:LastStreamText = $body
        $StreamBodyTxt.Text = $body
        $StreamSubTxt.Text = "double-click to copy"
        if ($isAct) { $StreamScroller.ScrollToEnd() }
    }
    $StreamHeaderTxt.Text = if ($r.source -eq "live_json") { "LIVE OUTPUT STREAM  $dotSep  $phase" } else { "QUINTESSENTIA STREAM  $dotSep  STANDBY" }
})
$AnimTimer.Start()

# ==============================================================================
# SYSTEM TRAY ICON (CUSTOM VECTOR 'Q')
# ==============================================================================
$Tray = New-Object System.Windows.Forms.NotifyIcon
$Tray.Icon = New-CustomTrayIcon
$Tray.Text = "Quintessentia Live HUD"
$Tray.Visible = $true

$Menu = New-Object System.Windows.Forms.ContextMenuStrip
$miToggle = $Menu.Items.Add("Show / Hide Quintessentia HUD")
$miPinTop = $Menu.Items.Add("Pin: Always on Top")
$miPinFlt = $Menu.Items.Add("Pin: Floating Window")
$miPinDsk = $Menu.Items.Add("Pin: Desktop Wallpaper")
$Menu.Items.Add("-") | Out-Null
$miClear = $Menu.Items.Add("Clear Graphs & Peaks")
$miExit  = $Menu.Items.Add("Exit Quintessentia HUD")

$miToggle.Add_Click({
    if ($HudWin.IsVisible) { $HudWin.Hide() } else { $HudWin.Show(); Clamp-WindowToVirtualScreen $HudWin; Apply-PinMode $HudWin $script:PinMode }
})
$miPinTop.Add_Click({ $script:PinMode = "topmost"; Apply-PinMode $HudWin "topmost"; $BtnPin.Content = "TOP"; Save-WindowState })
$miPinFlt.Add_Click({ $script:PinMode = "float";   Apply-PinMode $HudWin "float";   $BtnPin.Content = "FLT"; Save-WindowState })
$miPinDsk.Add_Click({ $script:PinMode = "desktop"; Apply-PinMode $HudWin "desktop"; $BtnPin.Content = "DESK"; Save-WindowState })
$miClear.Add_Click({ try { Set-Content -LiteralPath $ClearFlagPath -Value "1" -Encoding ASCII } catch {} })
$miExit.Add_Click({ Close-HudCleanly })

$Tray.ContextMenuStrip = $Menu
$Tray.Add_DoubleClick({
    $HudWin.Show(); Clamp-WindowToVirtualScreen $HudWin; Apply-PinMode $HudWin $script:PinMode
})

$App = New-Object System.Windows.Application
$App.ShutdownMode = [System.Windows.ShutdownMode]::OnExplicitShutdown
$HudWin.Show()
[QHNative]::ShowWindow((New-Object System.Windows.Interop.WindowInteropHelper($HudWin)).Handle, 5) | Out-Null

if ($CapturePreview) {
    $CapTimer = New-Object System.Windows.Threading.DispatcherTimer
    $CapTimer.Interval = [TimeSpan]::FromMilliseconds(1600)
    $CapTimer.Add_Tick({
        $CapTimer.Stop()
        Capture-HudScreenshot
    })
    $CapTimer.Start()
}

try { $App.Run() | Out-Null }
finally {
    try { $AnimTimer.Stop() } catch {}
    try { $Tray.Visible = $false; $Tray.Dispose() } catch {}
    try { if ($script:BackendProc -and -not $script:BackendProc.HasExited) { $script:BackendProc.Kill() } } catch {}
    try { if (Test-Path $UiPidPath) { Remove-Item -Force $UiPidPath -ErrorAction SilentlyContinue } } catch {}
    try { $InstanceMutex.ReleaseMutex() } catch {}
    try { $InstanceMutex.Dispose() } catch {}
}
