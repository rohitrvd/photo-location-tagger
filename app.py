import asyncio
import csv
import io
import json
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import pillow_heif
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request, send_file
from PIL import Image

from analyzer import ImageAnalyzer
from annotator import annotate_image

load_dotenv()

app = Flask(__name__)
SUPPORTED_EXTENSIONS = {".heic", ".jpg", ".jpeg"}

_job = {
    "status": "idle",
    "images": [],
    "results": [],
    "total": 0,
    "completed": 0,
    "enriched_dir": None,
    "unidentified": [],
    "tag_index": 0,
    "events": queue.Queue(),
}
_job_lock = threading.Lock()


def push_event(data: dict):
    _job["events"].put(json.dumps(data))


def scan_images(folder: Path) -> list[Path]:
    images = []
    for path in folder.rglob("*"):
        if path.suffix.lower() in SUPPORTED_EXTENSIONS and "enriched" not in path.parts:
            images.append(path)
    return sorted(images)


def get_output_path(image_path: Path, enriched_dir: Path) -> Path:
    if image_path.suffix.lower() == ".heic":
        out_name = image_path.stem + ".jpg"
        candidate = enriched_dir / out_name
        if candidate.exists() and candidate.stat().st_size > 0:
            out_name = image_path.stem + "_heic.jpg"
        return enriched_dir / out_name
    return enriched_dir / image_path.name


async def run_processing(folder: Path, enriched_dir: Path, api_key: str):
    images = _job["images"]
    enriched_dir.mkdir(parents=True, exist_ok=True)

    analyzer = ImageAnalyzer(api_key)
    semaphore = asyncio.Semaphore(5)
    executor = ThreadPoolExecutor(max_workers=4)
    results = []
    lock = asyncio.Lock()

    async def process_one(image_path: Path):
        async with semaphore:
            filename = image_path.name
            push_event({"type": "photo_start", "file": filename})

            analysis = await analyzer.analyze(image_path)
            result = {
                "filename": filename,
                "source_path": str(image_path),
                "status": "failed",
                "location": None,
                "placement": None,
                "tags": [],
                "input_tokens": analysis.get("input_tokens", 0),
                "output_tokens": analysis.get("output_tokens", 0),
                "error": None,
            }

            if analysis.get("success"):
                result["location"] = analysis["location"]
                result["placement"] = analysis["placement"]
                result["tags"] = analysis["tags"]
                result["status"] = "success"

                output_path = get_output_path(image_path, enriched_dir)
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    executor, annotate_image, image_path, output_path,
                    analysis["location"], analysis["placement"]
                )
                result["output_path"] = str(output_path)
            else:
                result["error"] = analysis.get("error")

            async with lock:
                results.append(result)
                _job["completed"] += 1

            loc = result.get("location") or {}
            push_event({
                "type": "photo_done",
                "file": filename,
                "status": result["status"],
                "location": loc.get("popular_name") or loc.get("landmark") or loc.get("city") or "",
                "confidence": loc.get("confidence", "low"),
                "completed": _job["completed"],
                "total": _job["total"],
            })

    await asyncio.gather(*[process_one(img) for img in images])
    executor.shutdown(wait=False)
    return results


def write_outputs(results, enriched_dir):
    enriched_dir = Path(enriched_dir)
    enriched_dir.mkdir(parents=True, exist_ok=True)

    with open(enriched_dir / "photo_data.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    fieldnames = ["filename", "landmark", "city", "region", "country", "confidence",
                  "location_type", "tags", "placement", "input_tokens", "output_tokens"]
    with open(enriched_dir / "photo_data.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            loc = r.get("location") or {}
            plac = r.get("placement") or {}
            writer.writerow({
                "filename": r["filename"],
                "landmark": loc.get("landmark") or loc.get("popular_name") or "",
                "city": loc.get("city") or "",
                "region": loc.get("region") or "",
                "country": loc.get("country") or "",
                "confidence": loc.get("confidence") or "",
                "location_type": loc.get("location_type") or "",
                "tags": "|".join(r.get("tags") or []),
                "placement": plac.get("recommendation") or "",
                "input_tokens": r.get("input_tokens", 0),
                "output_tokens": r.get("output_tokens", 0),
            })

    unidentified = [
        r["filename"] for r in results
        if r.get("status") == "success"
        and r.get("location", {}).get("confidence") == "low"
        and not (r["location"].get("popular_name") or r["location"].get("landmark"))
    ]
    with open(enriched_dir / "unidentified.txt", "w", encoding="utf-8") as f:
        f.write(f"# Photos where location could not be identified\n# Generated: {datetime.now().isoformat()}\n\n")
        for name in unidentified:
            f.write(name + "\n")


def make_summary(results):
    total = len(results)
    succeeded = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "failed")
    identified = sum(
        1 for r in results
        if r.get("status") == "success"
        and r.get("location", {}).get("confidence") in ("high", "medium")
    )
    total_input = sum(r.get("input_tokens", 0) for r in results)
    total_output = sum(r.get("output_tokens", 0) for r in results)
    cost = (total_input / 1_000_000) * 0.10 + (total_output / 1_000_000) * 0.40

    location_counts = {}
    for r in results:
        loc = r.get("location") or {}
        name = loc.get("popular_name") or loc.get("landmark") or loc.get("city")
        if name and loc.get("confidence") in ("high", "medium"):
            location_counts[name] = location_counts.get(name, 0) + 1
    top = sorted(location_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "total": total, "succeeded": succeeded, "failed": failed,
        "identified": identified,
        "identified_pct": round(100 * identified / total) if total else 0,
        "cost": round(cost, 4),
        "top_locations": [{"name": n, "count": c} for n, c in top],
        "enriched_dir": str(_job.get("enriched_dir", "")),
    }


def load_cached_results(enriched_dir: Path) -> dict:
    """Load previously identified results from photo_data.json keyed by filename."""
    json_path = enriched_dir / "photo_data.json"
    if not json_path.exists():
        return {}
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        # Only keep results that were successfully identified with a name
        return {
            r["filename"]: r for r in data
            if r.get("status") == "success"
            and (r.get("location", {}).get("popular_name") or r.get("location", {}).get("landmark"))
        }
    except Exception:
        return {}


def processing_thread(folder_str, enriched_dir_str, api_key):
    try:
        enriched_dir = Path(enriched_dir_str)
        cached = load_cached_results(enriched_dir)

        # Filter out already-identified images
        all_images = _job["images"]
        images_to_process = [img for img in all_images if img.name not in cached]
        skipped_results = list(cached.values())

        # Update total to reflect only what needs processing
        _job["total"] = len(images_to_process)
        _job["images"] = images_to_process

        # Emit skip events for cached photos so UI progress is accurate
        for r in skipped_results:
            loc = r.get("location") or {}
            push_event({
                "type": "photo_done",
                "file": r["filename"],
                "status": "success",
                "location": loc.get("popular_name") or loc.get("landmark") or "",
                "confidence": loc.get("confidence", "high"),
                "completed": _job["completed"],
                "total": len(all_images),
            })
            _job["completed"] += 1

        new_results = asyncio.run(run_processing(Path(folder_str), enriched_dir, api_key))
        results = skipped_results + new_results
        with _job_lock:
            _job["results"] = results
            _job["total"] = len(all_images)
            unidentified = [
                r for r in results
                if r.get("status") == "success"
                and not (
                    r.get("location", {}).get("popular_name")
                    or r.get("location", {}).get("landmark")
                )
            ]
            unidentified.sort(key=lambda r: r["filename"])
            _job["unidentified"] = unidentified
            _job["tag_index"] = 0
            _job["status"] = "tagging" if unidentified else "done"

        write_outputs(results, enriched_dir_str)

        if _job["unidentified"]:
            push_event({"type": "tagging_start", "count": len(_job["unidentified"])})
        else:
            push_event({"type": "done", "summary": make_summary(results)})

    except Exception as e:
        _job["status"] = "error"
        push_event({"type": "error", "message": str(e)})


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/scan", methods=["POST"])
def api_scan():
    folder = Path(request.json.get("folder", ""))
    if not folder.is_dir():
        return jsonify({"error": "Folder not found"}), 400
    images = scan_images(folder)
    return jsonify({"count": len(images)})


@app.route("/api/start", methods=["POST"])
def api_start():
    data = request.json
    folder = Path(data.get("folder", ""))
    save_to_enriched = data.get("save_to_enriched", True)

    if not folder.is_dir():
        return jsonify({"error": "Folder not found"}), 400

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return jsonify({"error": "GEMINI_API_KEY not set in .env file"}), 400

    enriched_dir = folder / "enriched" if save_to_enriched else folder
    images = scan_images(folder)
    if not images:
        return jsonify({"error": "No images found in folder"}), 400

    cached = load_cached_results(enriched_dir)
    images_to_process = [img for img in images if img.name not in cached]
    display_total = len(images)

    with _job_lock:
        _job.update({
            "status": "processing",
            "images": images,
            "results": [],
            "total": display_total,
            "completed": 0,
            "enriched_dir": str(enriched_dir),
            "unidentified": [],
            "tag_index": 0,
            "events": queue.Queue(),
        })

    threading.Thread(
        target=processing_thread,
        args=(str(folder), str(enriched_dir), api_key),
        daemon=True,
    ).start()

    return jsonify({"status": "started", "total": display_total, "cached": len(cached), "to_process": len(images_to_process)})


@app.route("/api/events")
def api_events():
    def generate():
        while True:
            try:
                event = _job["events"].get(timeout=30)
                yield f"data: {event}\n\n"
                data = json.loads(event)
                if data.get("type") in ("done", "error", "tagging_start"):
                    break
            except queue.Empty:
                yield 'data: {"type":"ping"}\n\n'
    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/next_tag")
def api_next_tag():
    idx = _job["tag_index"]
    unidentified = _job["unidentified"]
    if idx >= len(unidentified):
        _job["status"] = "done"
        return jsonify({"done": True, "summary": make_summary(_job["results"])})
    r = unidentified[idx]
    return jsonify({
        "done": False,
        "index": idx,
        "total": len(unidentified),
        "filename": r["filename"],
        "source_path": r["source_path"],
    })


@app.route("/api/submit_tag", methods=["POST"])
def api_submit_tag():
    data = request.json
    idx = _job["tag_index"]
    unidentified = _job["unidentified"]
    if idx >= len(unidentified):
        return jsonify({"error": "No more photos"}), 400

    r = unidentified[idx]
    name = data.get("name", "").strip()

    if name:
        city = data.get("city", "").strip() or None
        country = data.get("country", "").strip() or None
        location_data = {
            "city": city, "region": None, "country": country,
            "landmark": name, "popular_name": name,
            "location_type": "unknown", "confidence": "high",
        }
        placement_data = {"recommendation": "bottom-center"}
        source_path = Path(r["source_path"])
        output_path = get_output_path(source_path, Path(_job["enriched_dir"]))
        try:
            annotate_image(source_path, output_path, location_data, placement_data)
            for res in _job["results"]:
                if res["filename"] == r["filename"]:
                    res["location"] = location_data
                    res["placement"] = placement_data
                    break
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    _job["tag_index"] += 1

    if _job["tag_index"] >= len(unidentified):
        write_outputs(_job["results"], _job["enriched_dir"])
        _job["status"] = "done"
        return jsonify({"ok": True, "all_done": True, "summary": make_summary(_job["results"])})

    return jsonify({"ok": True, "all_done": False})


@app.route("/api/results")
def api_results():
    results = _job.get("results", [])
    enriched_dir = _job.get("enriched_dir", "")
    out = []
    for r in results:
        loc = r.get("location") or {}
        out.append({
            "filename": r["filename"],
            "source_path": r["source_path"],
            "output_path": r.get("output_path", ""),
            "status": r.get("status"),
            "location_name": loc.get("popular_name") or loc.get("landmark") or "",
            "city": loc.get("city") or "",
            "country": loc.get("country") or "",
            "confidence": loc.get("confidence", "low"),
        })
    return jsonify({"results": out, "enriched_dir": enriched_dir})


@app.route("/api/edit_tag", methods=["POST"])
def api_edit_tag():
    data = request.json
    filename = data.get("filename", "")
    name = data.get("name", "").strip()
    city = data.get("city", "").strip() or None
    country = data.get("country", "").strip() or None

    # Find the result
    target = None
    for r in _job["results"]:
        if r["filename"] == filename:
            target = r
            break
    if not target:
        return jsonify({"error": "Photo not found"}), 404

    if name:
        location_data = {
            "city": city, "region": None, "country": country,
            "landmark": name, "popular_name": name,
            "location_type": "unknown", "confidence": "high",
        }
        placement_data = {"recommendation": "bottom-center"}
        source_path = Path(target["source_path"])
        output_path = get_output_path(source_path, Path(_job["enriched_dir"]))
        try:
            annotate_image(source_path, output_path, location_data, placement_data)
            target["location"] = location_data
            target["placement"] = placement_data
            target["output_path"] = str(output_path)
            target["status"] = "success"
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        # Clearing a tag — mark as no location
        if target.get("location"):
            target["location"]["popular_name"] = None
            target["location"]["landmark"] = None
            target["location"]["confidence"] = "low"

    write_outputs(_job["results"], _job["enriched_dir"])
    loc = target.get("location") or {}
    return jsonify({
        "ok": True,
        "location_name": loc.get("popular_name") or loc.get("landmark") or "",
        "output_path": target.get("output_path", ""),
    })


@app.route("/api/image")
def api_image():
    from PIL import ImageOps
    path = request.args.get("path", "")
    try:
        p = Path(path)
        if not p.exists():
            return "Not found", 404
        if p.suffix.lower() == ".heic":
            try:
                heif_file = pillow_heif.open_heif(str(p))
                img = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data, "raw")
            except Exception:
                img = Image.open(p)
                img.load()
        else:
            img = Image.open(p)
            img.load()
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        buf.seek(0)
        return Response(buf.read(), mimetype="image/jpeg")
    except Exception as e:
        return str(e), 500


if __name__ == "__main__":
    import webbrowser
    webbrowser.open("http://localhost:5000")
    app.run(debug=False, port=5000)
