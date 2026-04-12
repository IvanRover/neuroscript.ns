param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ScriptArgs
)

py -3 "$PSScriptRoot\ns.py" @ScriptArgs
