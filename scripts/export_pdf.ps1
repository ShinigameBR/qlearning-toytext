# Exporta um .docx para .pdf usando o Microsoft Word.
# Uso interno de convert_to_pdf.ps1:
#   powershell -ExecutionPolicy Bypass -File scripts/export_pdf.ps1 <entrada.docx> <saida.pdf>

param(
    [Parameter(Mandatory = $true)][string]$Docx,
    [Parameter(Mandatory = $true)][string]$Pdf
)

$ErrorActionPreference = "Stop"

$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
$word.Options.UpdateFieldsAtPrint = $false
$document = $null
try {
    $document = $word.Documents.Open($Docx, $false, $true, $false)
    $document.SaveAs2($Pdf, 17)  # 17 = wdFormatPDF
    $pages = $document.ComputeStatistics(2)
    $document.Close(0)
    $document = $null
    Write-Output "PDF exportado: $Pdf ($pages páginas)"
}
finally {
    if ($document -ne $null) {
        try { $document.Close(0) } catch { }
    }
    try { $word.Quit() } catch { }
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
}
