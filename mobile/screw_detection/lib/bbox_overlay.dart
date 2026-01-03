import 'dart:io';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';

class BBoxOverlay extends StatefulWidget {
  final File imageFile;
  final Map<String, dynamic>? result;

  const BBoxOverlay({super.key, required this.imageFile, required this.result});

  @override
  State<BBoxOverlay> createState() => _BBoxOverlayState();
}

class _BBoxOverlayState extends State<BBoxOverlay> {
  ui.Image? _uiImage;
  String? _loadedPath;

  @override
  void didUpdateWidget(covariant BBoxOverlay oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (_loadedPath != widget.imageFile.path) {
      _uiImage = null;
      _loadedPath = widget.imageFile.path;
      _loadUiImage();
    }
  }

  @override
  void initState() {
    super.initState();
    _loadedPath = widget.imageFile.path;
    _loadUiImage();
  }

  Future<void> _loadUiImage() async {
    final bytes = await widget.imageFile.readAsBytes();
    final codec = await ui.instantiateImageCodec(bytes);
    final frame = await codec.getNextFrame();
    debugPrint("uiImage WH = ${frame.image.width} x ${frame.image.height}");
    if (!mounted) return;
    setState(() => _uiImage = frame.image);
  }

  @override
  Widget build(BuildContext context) {
    final dets = (widget.result?["detections"] as List?) ?? const [];

    return LayoutBuilder(
      builder: (context, c) {
        return CustomPaint(
          painter: _OverlayPainter(
            uiImage: _uiImage,
            dets: dets,
            apiW: (widget.result?["image_size"]?["w"] as num?)?.toDouble(),
            apiH: (widget.result?["image_size"]?["h"] as num?)?.toDouble(),
          ),
          size: Size(c.maxWidth, c.maxHeight),
        );
      },
    );
  }
}

class _OverlayPainter extends CustomPainter {
  final ui.Image? uiImage;
  final List dets;
  final double? apiW;
  final double? apiH;

  _OverlayPainter({
    required this.uiImage,
    required this.dets,
    required this.apiW,
    required this.apiH,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (uiImage == null) return;

    final imgW = uiImage!.width.toDouble();
    final imgH = uiImage!.height.toDouble();

    // 1) Resmi canvas içine "contain" olacak şekilde yerleştir
    final scaleX = size.width / imgW;
    final scaleY = size.height / imgH;
    final s = scaleX < scaleY ? scaleX : scaleY;

    final drawW = imgW * s;
    final drawH = imgH * s;

    final dx = (size.width - drawW) / 2.0;
    final dy = (size.height - drawH) / 2.0;

    final dst = Rect.fromLTWH(dx, dy, drawW, drawH);
    final src = Rect.fromLTWH(0, 0, imgW, imgH);
    canvas.drawImageRect(uiImage!, src, dst, Paint());

    // 2) API bbox'larının referans aldığı boyut (API image_size)
    // (null gelirse fallback: uiImage boyutu)
    final refW = apiW ?? imgW;
    final refH = apiH ?? imgH;

    // 3) Çizim ayarları
    final boxPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0
      ..color = Colors.greenAccent;

    // 4) Detections çiz
    for (final d in dets) {
      final det = d["det"] as Map<String, dynamic>;
      final box = (det["bbox_xyxy"] as List)
          .map((e) => (e as num).toDouble())
          .toList();

      // API -> DST mapping
      final x1 = dx + (box[0] / refW) * drawW;
      final y1 = dy + (box[1] / refH) * drawH;
      final x2 = dx + (box[2] / refW) * drawW;
      final y2 = dy + (box[3] / refH) * drawH;

      final rect = Rect.fromLTRB(x1, y1, x2, y2);
      canvas.drawRect(rect, boxPaint);

      // Label
      final label =
          "${det['label']} ${(det['confidence'] as num).toStringAsFixed(2)}";
      _drawLabel(canvas, label, x1, y1);
    }
  }

  // Label çizimini temizlemek için yardımcı method
  void _drawLabel(Canvas canvas, String text, double x, double y) {
    final textSpan = TextSpan(
      text: text,
      style: const TextStyle(
        color: Colors.black,
        fontSize: 12,
        fontWeight: FontWeight.bold,
      ),
    );
    final tp = TextPainter(text: textSpan, textDirection: TextDirection.ltr);
    tp.layout();

    // Arka plan
    final bgRect = Rect.fromLTWH(x, y - tp.height, tp.width + 4, tp.height);
    canvas.drawRect(bgRect, Paint()..color = Colors.greenAccent);

    tp.paint(canvas, Offset(x + 2, y - tp.height));
  }

  @override
  bool shouldRepaint(covariant _OverlayPainter oldDelegate) {
    return oldDelegate.uiImage != uiImage ||
        oldDelegate.dets != dets ||
        oldDelegate.apiW != apiW ||
        oldDelegate.apiH != apiH;
  }
}
