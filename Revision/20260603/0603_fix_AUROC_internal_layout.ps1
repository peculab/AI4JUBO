# Documents the stepwise layout fix applied to
# 0603_Figure1_ROC_internal_cv_revised_1to1.png.
#
# Step 1 was an initial margin-only check.
# Steps 2 and 3 used a temporary copy named
# 0603_Figure1_ROC_internal_cv_revised_1to1_original.png.
# That temporary image was deleted after the final output was accepted.

Set-StrictMode -Version Latest
Add-Type -AssemblyName System.Drawing

$finalFigure = 'c:\GitHub\AI4JUBO\Revision\20260603\0603_Figure1_ROC_internal_cv_revised_1to1.png'
$temporaryOriginal = 'c:\GitHub\AI4JUBO\Revision\20260603\0603_Figure1_ROC_internal_cv_revised_1to1_original.png'

function Add-OuterWhiteMargin {
  param(
    [string]$SourcePath,
    [string]$OutputPath
  )

  $img = [System.Drawing.Image]::FromFile($SourcePath)
  $left = 90; $top = 70; $right = 100; $bottom = 100
  $bmp = New-Object System.Drawing.Bitmap ($img.Width + $left + $right), ($img.Height + $top + $bottom)
  $g = [System.Drawing.Graphics]::FromImage($bmp)
  $g.Clear([System.Drawing.Color]::White)
  $g.DrawImage($img, $left, $top, $img.Width, $img.Height)
  $g.Dispose()
  $img.Dispose()
  $bmp.Save($OutputPath, [System.Drawing.Imaging.ImageFormat]::Png)
  $bmp.Dispose()
}

function Redraw-YAxis {
  param(
    [string]$SourcePath,
    [string]$OutputPath,
    [int]$RightPad = 90
  )

  $img = [System.Drawing.Image]::FromFile($SourcePath)
  $cropX = 122; $cropY = 0; $cropW = $img.Width - $cropX; $cropH = $img.Height
  $leftPad = 240; $topPad = 45; $bottomPad = 90
  $bmp = New-Object System.Drawing.Bitmap ($leftPad + $cropW + $RightPad), ($topPad + $cropH + $bottomPad)
  $g = [System.Drawing.Graphics]::FromImage($bmp)
  $g.Clear([System.Drawing.Color]::White)
  $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
  $g.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
  $g.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit

  $srcRect = New-Object System.Drawing.Rectangle $cropX, $cropY, $cropW, $cropH
  $dstRect = New-Object System.Drawing.Rectangle $leftPad, $topPad, $cropW, $cropH
  $g.DrawImage($img, $dstRect, $srcRect, [System.Drawing.GraphicsUnit]::Pixel)

  $brush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::Black)
  $tickFont = New-Object System.Drawing.Font 'Arial', 25, ([System.Drawing.FontStyle]::Regular), ([System.Drawing.GraphicsUnit]::Pixel)
  $labelFont = New-Object System.Drawing.Font 'Arial', 28, ([System.Drawing.FontStyle]::Regular), ([System.Drawing.GraphicsUnit]::Pixel)
  $tickFormat = New-Object System.Drawing.StringFormat
  $tickFormat.Alignment = [System.Drawing.StringAlignment]::Far
  $tickFormat.LineAlignment = [System.Drawing.StringAlignment]::Center

  $labels = @('1','0.8','0.6','0.4','0.2','0')
  $ys = @(60,240,420,600,780,960)
  for ($i = 0; $i -lt $labels.Count; $i++) {
    $rect = New-Object System.Drawing.RectangleF 112, ($topPad + $ys[$i] - 22), 82, 44
    $g.DrawString($labels[$i], $tickFont, $brush, $rect, $tickFormat)
  }

  $labelFormat = New-Object System.Drawing.StringFormat
  $labelFormat.Alignment = [System.Drawing.StringAlignment]::Center
  $labelFormat.LineAlignment = [System.Drawing.StringAlignment]::Center
  $g.TranslateTransform(64, 610)
  $g.RotateTransform(-90)
  $labelRect = New-Object System.Drawing.RectangleF -430, -30, 860, 60
  $g.DrawString('True Positive Rate (Sensitivity)', $labelFont, $brush, $labelRect, $labelFormat)
  $g.ResetTransform()

  $labelFormat.Dispose(); $tickFormat.Dispose(); $brush.Dispose()
  $tickFont.Dispose(); $labelFont.Dispose(); $g.Dispose(); $img.Dispose()
  $bmp.Save($OutputPath, [System.Drawing.Imaging.ImageFormat]::Png)
  $bmp.Dispose()
}

function Redraw-YAxis-And-Legend {
  param(
    [string]$SourcePath,
    [string]$OutputPath
  )

  $img = [System.Drawing.Image]::FromFile($SourcePath)
  $cropX = 122; $cropY = 0; $cropW = $img.Width - $cropX; $cropH = $img.Height
  $leftPad = 240; $topPad = 45; $rightPad = 260; $bottomPad = 90
  $bmp = New-Object System.Drawing.Bitmap ($leftPad + $cropW + $rightPad), ($topPad + $cropH + $bottomPad)
  $g = [System.Drawing.Graphics]::FromImage($bmp)
  $g.Clear([System.Drawing.Color]::White)
  $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
  $g.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
  $g.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit

  $srcRect = New-Object System.Drawing.Rectangle $cropX, $cropY, $cropW, $cropH
  $dstRect = New-Object System.Drawing.Rectangle $leftPad, $topPad, $cropW, $cropH
  $g.DrawImage($img, $dstRect, $srcRect, [System.Drawing.GraphicsUnit]::Pixel)

  $whiteBrush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::White)
  $g.FillRectangle($whiteBrush, 1205, 300, 660, 520)

  $black = [System.Drawing.Color]::Black
  $brush = New-Object System.Drawing.SolidBrush $black
  $tickFont = New-Object System.Drawing.Font 'Arial', 25, ([System.Drawing.FontStyle]::Regular), ([System.Drawing.GraphicsUnit]::Pixel)
  $labelFont = New-Object System.Drawing.Font 'Arial', 28, ([System.Drawing.FontStyle]::Regular), ([System.Drawing.GraphicsUnit]::Pixel)
  $tickFormat = New-Object System.Drawing.StringFormat
  $tickFormat.Alignment = [System.Drawing.StringAlignment]::Far
  $tickFormat.LineAlignment = [System.Drawing.StringAlignment]::Center

  $labels = @('1','0.8','0.6','0.4','0.2','0')
  $ys = @(60,240,420,600,780,960)
  for ($i = 0; $i -lt $labels.Count; $i++) {
    $rect = New-Object System.Drawing.RectangleF 112, ($topPad + $ys[$i] - 22), 82, 44
    $g.DrawString($labels[$i], $tickFont, $brush, $rect, $tickFormat)
  }

  $labelFormat = New-Object System.Drawing.StringFormat
  $labelFormat.Alignment = [System.Drawing.StringAlignment]::Center
  $labelFormat.LineAlignment = [System.Drawing.StringAlignment]::Center
  $g.TranslateTransform(64, 610)
  $g.RotateTransform(-90)
  $labelRect = New-Object System.Drawing.RectangleF -430, -30, 860, 60
  $g.DrawString('True Positive Rate (Sensitivity)', $labelFont, $brush, $labelRect, $labelFormat)
  $g.ResetTransform()

  $legendX = 1270; $legendY = 360; $legendW = 555; $legendH = 365
  $penBlack = New-Object System.Drawing.Pen $black, 2
  $g.DrawRectangle($penBlack, $legendX, $legendY, $legendW, $legendH)
  $legendFont = New-Object System.Drawing.Font 'Arial', 25, ([System.Drawing.FontStyle]::Regular), ([System.Drawing.GraphicsUnit]::Pixel)
  $items = @(
    @('HybridXGBRF (AUC=0.888)', [System.Drawing.Color]::FromArgb(95,105,255), $false),
    @('XGB (AUC=0.885)', [System.Drawing.Color]::FromArgb(255,75,60), $false),
    @('RF (AUC=0.871)', [System.Drawing.Color]::FromArgb(0,200,150), $false),
    @('LR-1000 (AUC=0.857)', [System.Drawing.Color]::FromArgb(170,85,255), $false),
    @('Ridge (AUC=0.855)', [System.Drawing.Color]::FromArgb(255,150,80), $false),
    @('Elastic (AUC=0.856)', [System.Drawing.Color]::FromArgb(25,200,220), $false),
    @('Lasso (AUC=0.856)', [System.Drawing.Color]::FromArgb(255,85,135), $false),
    @('LR-200 (AUC=0.857)', [System.Drawing.Color]::FromArgb(170,225,115), $false),
    @('Chance', [System.Drawing.Color]::FromArgb(255,125,255), $true)
  )

  for ($i = 0; $i -lt $items.Count; $i++) {
    $y = $legendY + 42 + ($i * 36)
    $pen = New-Object System.Drawing.Pen $items[$i][1], 4
    if ($items[$i][2]) {
      $pen.DashStyle = [System.Drawing.Drawing2D.DashStyle]::Dash
    }
    $g.DrawLine($pen, $legendX + 38, $y + 8, $legendX + 115, $y + 8)
    $g.DrawString($items[$i][0], $legendFont, $brush, ($legendX + 142), ($y - 8))
    $pen.Dispose()
  }

  $legendFont.Dispose(); $penBlack.Dispose(); $whiteBrush.Dispose()
  $labelFormat.Dispose(); $tickFormat.Dispose(); $brush.Dispose()
  $tickFont.Dispose(); $labelFont.Dispose(); $g.Dispose(); $img.Dispose()
  $bmp.Save($OutputPath, [System.Drawing.Imaging.ImageFormat]::Png)
  $bmp.Dispose()
}

# Historical execution order:
# Copy-Item -LiteralPath $finalFigure -Destination $temporaryOriginal
# Add-OuterWhiteMargin -SourcePath $temporaryOriginal -OutputPath $finalFigure
# Redraw-YAxis -SourcePath $temporaryOriginal -OutputPath $finalFigure -RightPad 90
# Redraw-YAxis-And-Legend -SourcePath $temporaryOriginal -OutputPath $finalFigure
# Remove-Item -LiteralPath $temporaryOriginal
