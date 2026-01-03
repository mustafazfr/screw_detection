import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:image/image.dart' as img;
import 'bbox_overlay.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Screw Detect + Classify',
      theme: ThemeData(useMaterial3: true),
      home: const HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final ImagePicker _picker = ImagePicker();

  String fmtConf(num? v) {
    if (v == null) return "-";
    return "${(v * 100).toStringAsFixed(1)}%";
  }

  String fmtBbox(dynamic bbox) {
    if (bbox is List && bbox.length == 4) {
      final x1 = (bbox[0] as num).toStringAsFixed(0);
      final y1 = (bbox[1] as num).toStringAsFixed(0);
      final x2 = (bbox[2] as num).toStringAsFixed(0);
      final y2 = (bbox[3] as num).toStringAsFixed(0);
      return "[$x1,$y1] -> [$x2,$y2]";
    }
    return "-";
  }

  Future<File> ensureJpeg(File input) async {
    final bytes = await input.readAsBytes();
    final decoded = img.decodeImage(bytes);

    if (decoded == null) {
      throw Exception("Görüntü decode edilemedi.");
    }

    // EXIF orientation uygula (Resmi fiziksel olarak döndürür)
    final fixed = img.bakeOrientation(decoded);

    //Resmi tamamen yeni bir Image objesine kopyala
    final cleanImage = img.Image.from(fixed);

    final jpgBytes = img.encodeJpg(cleanImage, quality: 100);

    final dir = await getTemporaryDirectory();
    final outPath = p.join(
      dir.path,
      "upload_${DateTime.now().millisecondsSinceEpoch}.jpg",
    );

    final outFile = File(outPath);
    await outFile.writeAsBytes(jpgBytes, flush: true);
    return outFile;
  }

  final String baseUrl = "http://172.20.10.11:8000";

  File? _selectedImage;
  bool _loading = false;
  Map<String, dynamic>? _result;
  String? _error;
  File? _displayImage; // ekranda gösterilecek dosya

  Future<void> _pickFromGallery() async {
    setState(() {
      _error = null;
      _result = null;
    });

    final XFile? xfile = await _picker.pickImage(
      source: ImageSource.gallery,
      imageQuality: 100,
    );
    if (xfile == null) return;

    final file = File(xfile.path);
    setState(() {
      _selectedImage = file;
      _displayImage = file;
    });
  }

  Future<void> _takePhoto() async {
    setState(() {
      _error = null;
      _result = null;
    });

    final XFile? xfile = await _picker.pickImage(
      source: ImageSource.camera,
      imageQuality: 100,
    );
    if (xfile == null) return;

    final file = File(xfile.path);
    setState(() {
      _selectedImage = file;
      _displayImage = file;
    });
  }

  Future<void> _sendToApi() async {
    final imageFile = _selectedImage;
    if (imageFile == null) return;

    setState(() {
      _loading = true;
      _error = null;
      _result = null;
    });

    try {
      // 1) JPEG + EXIF orientation FIX
      final jpegFile = await ensureJpeg(imageFile);

      // 2) Ekranda GÖSTERİLEN resim = API'ye GİDEN resim
      setState(() {
        _displayImage = jpegFile;
      });

      // 3) API endpoint
      final uri = Uri.parse(
        "$baseUrl/predict?det_conf=0.20&det_iou=0.70&cls_imgsz=1024&cls_topk=4&max_dets=50",
      );

      final request = http.MultipartRequest("POST", uri);
      request.files.add(
        await http.MultipartFile.fromPath("file", jpegFile.path),
      );

      // 4) Request gönder
      final streamed = await request.send();
      final body = await streamed.stream.bytesToString();

      if (streamed.statusCode != 200) {
        throw Exception("API hata verdi (${streamed.statusCode}): $body");
      }

      final jsonMap = jsonDecode(body) as Map<String, dynamic>;

      // 5) Sonucu UI'ya bas
      setState(() {
        _result = jsonMap;
      });
    } catch (e) {
      setState(() {
        _error = e.toString();
      });
    } finally {
      setState(() {
        _loading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final detections = (_result?["detections"] as List?) ?? const [];

    return Scaffold(
      appBar: AppBar(title: const Text("Screw Detect + Classify")),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _loading ? null : _pickFromGallery,
                    icon: const Icon(Icons.photo_library_outlined),
                    label: const Text("Galeriden Seç"),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _loading ? null : _takePhoto,
                    icon: const Icon(Icons.photo_camera_outlined),
                    label: const Text('Kamera'),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),

            SizedBox(
              height: 320,
              width: double.infinity,
              child: _selectedImage == null
                  ? Container(
                      alignment: Alignment.center,
                      decoration: BoxDecoration(
                        border: Border.all(color: Colors.black12),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: const Text("Foto seç veya çek"),
                    )
                  : ClipRRect(
                      borderRadius: BorderRadius.circular(12),
                      child: BBoxOverlay(
                        imageFile:
                            _displayImage ??
                            _selectedImage!, // önce display, yoksa selected
                        result: _result,
                      ),
                    ),
            ),

            const SizedBox(height: 12),

            SizedBox(
              width: double.infinity,
              child: FilledButton(
                onPressed: (_selectedImage == null || _loading)
                    ? null
                    : _sendToApi,
                child: _loading
                    ? const SizedBox(
                        height: 18,
                        width: 18,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    : const Text("Gönder & Sonucu Al"),
              ),
            ),
            const SizedBox(height: 12),

            if (_error != null)
              Text(_error!, style: const TextStyle(color: Colors.red)),

            if (_result != null) ...[
              Align(
                alignment: Alignment.centerLeft,
                child: Text(
                  "Detections: ${detections.length}",
                  style: const TextStyle(fontWeight: FontWeight.bold),
                ),
              ),
              const SizedBox(height: 8),
              Expanded(
                child: ListView.builder(
                  itemCount: detections.length,
                  itemBuilder: (context, i) {
                    final det = detections[i]["det"] as Map<String, dynamic>;
                    final cls = detections[i]["cls"] as Map<String, dynamic>;
                    final topk = (cls["topk"] as List?) ?? const [];

                    final detLabel = det["label"]?.toString() ?? "-";
                    final detConf = det["confidence"] as num?;
                    final bbox = det["bbox_xyxy"];

                    final clsLabel = cls["label"]?.toString() ?? "-";
                    final clsConf = cls["confidence"] as num?;

                    return Card(
                      child: Padding(
                        padding: const EdgeInsets.all(12),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              "Det: $detLabel  |  conf: ${fmtConf(detConf)}",
                              style: const TextStyle(
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                            Text("BBox: ${fmtBbox(bbox)}"),

                            const SizedBox(height: 8),

                            Text(
                              "Cls: $clsLabel  |  conf: ${fmtConf(clsConf)}",
                              style: const TextStyle(
                                fontWeight: FontWeight.bold,
                              ),
                            ),

                            if (topk.isNotEmpty) ...[
                              const SizedBox(height: 6),
                              const Text("Top-4:"),
                              for (final t in topk)
                                Text(
                                  "• ${t["label"]} — ${fmtConf(t["confidence"] as num?)}",
                                ),
                            ],
                          ],
                        ),
                      ),
                    );
                  },
                ),
              ),
            ] else
              const Spacer(),
          ],
        ),
      ),
    );
  }
}
