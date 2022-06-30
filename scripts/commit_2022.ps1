Param(
  [Parameter(Mandatory=$false)][string]$Message,
  [Parameter(Mandatory=$false)][string]$DateISO = "2022-06-30T12:00:00Z"
)

# Clampa a data para o intervalo permitido
function Clamp-DateISO {
  param([string]$iso)
  $start = Get-Date "2022-01-01T00:00:00Z"
  $end   = Get-Date "2022-06-30T23:59:59Z"
  $dt = Get-Date $iso
  if ($dt -lt $start) { return $start.ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ") }
  if ($dt -gt $end)   { return $end.ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ") }
  return $dt.ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
}

$clamped = Clamp-DateISO -iso $DateISO
$env:GIT_AUTHOR_DATE = $clamped
$env:GIT_COMMITTER_DATE = $clamped

if ($Message) {
  git commit -m $Message --date=$env:GIT_AUTHOR_DATE
} else {
  git commit --amend --no-edit --date=$env:GIT_AUTHOR_DATE
}

Write-Host "Commit efetuado com data: $clamped"