from __future__ import annotations

import csv
from io import StringIO
import json

from .alignment import TimedLine


def lrc_timestamp(seconds: float) -> str:
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    return f"{minutes:02d}:{remainder:05.2f}"


def srt_timestamp(seconds: float) -> str:
    milliseconds = round(max(0.0, seconds) * 1000)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def export_lrc(lines: list[TimedLine], title: str = "") -> str:
    header = f"[ti:{title}]\n" if title else ""
    return header + "\n".join(f"[{lrc_timestamp(line.start)}]{line.text}" for line in lines) + "\n"


def export_srt(lines: list[TimedLine]) -> str:
    blocks = [
        f"{index}\n{srt_timestamp(line.start)} --> {srt_timestamp(line.end)}\n{line.text}"
        for index, line in enumerate(lines, start=1)
    ]
    return "\n\n".join(blocks) + "\n"


def export_csv(lines: list[TimedLine]) -> str:
    output = StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["index", "start_seconds", "end_seconds", "text", "confidence"])
    for line in lines:
        writer.writerow([line.index, f"{line.start:.3f}", f"{line.end:.3f}", line.text, f"{line.confidence:.3f}"])
    return "\ufeff" + output.getvalue()


def export_json(lines: list[TimedLine], metadata: dict) -> str:
    payload = {"metadata": metadata, "lines": [line.to_dict() for line in lines]}
    return json.dumps(payload, ensure_ascii=False, indent=2)


def export_ae_jsx(lines: list[TimedLine]) -> str:
    data = json.dumps(
        [{"text": line.text, "start": line.start, "end": line.end} for line in lines],
        ensure_ascii=False,
    )
    return f'''// Lyrics Sync - Adobe After Effects importer (UTF-8)
(function () {{
    app.beginUndoGroup("Import timed lyrics");
    var comp = app.project.activeItem;
    if (!(comp && comp instanceof CompItem)) {{
        alert("请先打开或选中一个合成 (Composition)。");
        app.endUndoGroup();
        return;
    }}

    var lyrics = {data};
    for (var i = lyrics.length - 1; i >= 0; i--) {{
        var item = lyrics[i];
        var layer = comp.layers.addText(item.text);
        layer.name = "歌词 " + (i + 1) + " - " + item.text;
        layer.startTime = item.start;
        layer.inPoint = item.start;
        layer.outPoint = Math.min(comp.duration, Math.max(item.end, item.start + 0.1));

        var textDocument = layer.property("Source Text").value;
        textDocument.justification = ParagraphJustification.CENTER_JUSTIFY;
        layer.property("Source Text").setValue(textDocument);
        layer.property("Transform").property("Position").setValue([comp.width / 2, comp.height * 0.85]);
    }}
    app.endUndoGroup();
}})();
'''


def build_exports(lines: list[TimedLine], metadata: dict) -> dict[str, str]:
    title = metadata.get("filename", "")
    return {
        "lrc": export_lrc(lines, title),
        "srt": export_srt(lines),
        "csv": export_csv(lines),
        "json": export_json(lines, metadata),
        "jsx": export_ae_jsx(lines),
    }

