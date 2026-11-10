# Converte o relatório .docx em .pdf usando o Microsoft Word instalado.
#
# O processo é feito em duas etapas, porque a exportação de PDF logo após a
# atualização do sumário pode travar o Word:
#   1) atualiza o sumário, converte o campo TOC em texto estático e salva o .docx;
#   2) exporta o PDF em um novo processo do Word (scripts/export_pdf.ps1).
#
# Uso: powershell -ExecutionPolicy Bypass -File scripts/convert_to_pdf.ps1

param(
    [string]$Docx = "report/Relatorio_Q-learning_Toy_Text.docx",
    [string]$Pdf  = "report/Relatorio_Q-learning_Toy_Text.pdf"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$docxPath = Join-Path $root $Docx
$pdfPath  = Join-Path $root $Pdf

if (-not (Test-Path $docxPath)) {
    throw "Arquivo não encontrado: $docxPath"
}

# O Word trabalha com uma cópia local na pasta temporária; o resultado é
# copiado de volta ao final.
$tmpDir   = Join-Path ([System.IO.Path]::GetTempPath()) ("qlearning_report_" + [System.Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null
$workDocx = Join-Path $tmpDir "relatorio.docx"
$workPdf  = Join-Path $tmpDir "relatorio.pdf"
Copy-Item -Force $docxPath $workDocx

try {
    # Etapa 1: sumário atualizado e fixado como texto estático.
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $document = $null
    try {
        $document = $word.Documents.Open($workDocx)
        foreach ($toc in $document.TablesOfContents) {
            $toc.Update() | Out-Null
            # 37 = wdFieldTOC
            foreach ($field in @($toc.Range.Fields)) {
                if ($field.Type -eq 37) { $field.Unlink() }
            }
        }
        $document.Save()
    }
    finally {
        if ($document -ne $null) {
            try { $document.Close(0) } catch { }
        }
        try { $word.Quit() } catch { }
        [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
    }
    Write-Output "Sumário atualizado."

    # Etapa 2: exportação do PDF em um processo novo do Word.
    $exportScript = Join-Path $PSScriptRoot "export_pdf.ps1"
    $proc = Start-Process powershell -ArgumentList @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $exportScript, $workDocx, $workPdf
    ) -PassThru -Wait -WindowStyle Hidden
    if ($proc.ExitCode -ne 0 -or -not (Test-Path $workPdf)) {
        throw "Falha ao exportar o PDF (código $($proc.ExitCode))."
    }

    Copy-Item -Force $workPdf $pdfPath
    Copy-Item -Force $workDocx $docxPath
    Write-Output "PDF gerado: $pdfPath"
    Write-Output "DOCX salvo com o sumário preenchido: $docxPath"
}
finally {
    Remove-Item -Recurse -Force $tmpDir -ErrorAction SilentlyContinue
}
