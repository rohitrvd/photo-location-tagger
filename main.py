import asyncio
import csv
import fnmatch
import json
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from analyzer import ImageAnalyzer
from annotator import annotate_image

load_dotenv()

SUPPORTED_EXTENSIONS = {".heic", ".jpg", ".jpeg"}
PROGRESS_SAVE_INTERVAL = 10

console = Console()


def prompt_folder_path() -> Path:
    while True:
        raw = console.input("[bold cyan]Enter folder path containing photos:[/] ").strip()
        p = Path(raw)
        if p.is_dir():
            return p
        console.print(f"[red]Directory not found:[/] {raw}")


def scan_for_images(folder: Path) -> list[Path]:
    images = []
    for path in folder.rglob("*"):
        if path.suffix.lower() in SUPPORTED_EXTENSIONS and "enriched" not in path.parts:
            images.append(path)
    return sorted(images)


def apply_glob_filter(images: list[Path], pattern: str) -> list[Path]:
    return [p for p in images if fnmatch.fnmatch(p.name, pattern)]


def load_progress(enriched_dir: Path) -> dict:
    progress_file = enriched_dir / "progress.json"
    if progress_file.exists():
        try:
            with open(progress_file) as f:
                return json.load(f)
        except Exception:
            pass
    return {"processed": [], "failed": [], "skipped": [], "session_start": None, "last_updated": None}


def save_progress(enriched_dir: Path, progress: dict) -> None:
    """Atomic write via temp file."""
    progress["last_updated"] = datetime.now(timezone.utc).isoformat()
    enriched_dir.mkdir(parents=True, exist_ok=True)
    progress_file = enriched_dir / "progress.json"
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=enriched_dir,
            suffix=".tmp",
            delete=False,
        ) as f:
            json.dump(progress, f, indent=2)
            tmp_path = Path(f.name)
        tmp_path.replace(progress_file)
    except Exception as e:
        console.print(f"[yellow]Warning: Could not save progress: {e}[/]")


def get_output_path(image_path: Path, enriched_dir: Path) -> Path:
    suffix = image_path.suffix.lower()
    if suffix == ".heic":
        out_name = image_path.stem + ".jpg"
        candidate = enriched_dir / out_name
        # Collision guard
        if candidate.exists() and candidate.stat().st_size > 0:
            out_name = image_path.stem + "_heic.jpg"
        return enriched_dir / out_name
    else:
        return enriched_dir / image_path.name


class TokenCounter:
    def __init__(self):
        self.total_input = 0
        self.total_output = 0
        self._lock = asyncio.Lock()

    async def add(self, input_tokens: int, output_tokens: int):
        async with self._lock:
            self.total_input += input_tokens
            self.total_output += output_tokens

    def estimated_cost(self) -> float:
        # gemini-2.0-flash pricing: $0.10/MTok input, $0.40/MTok output
        return (self.total_input / 1_000_000) * 0.10 + (self.total_output / 1_000_000) * 0.40


async def process_single_image(
    image_path: Path,
    enriched_dir: Path,
    analyzer: ImageAnalyzer,
    semaphore: asyncio.Semaphore,
    executor: ThreadPoolExecutor,
    progress_bar: Progress,
    task_id,
    token_counter: TokenCounter,
    results: list,
    progress_data: dict,
    completion_counter: list,
) -> None:
    async with semaphore:
        filename = image_path.name
        result = {
            "filename": filename,
            "source_path": str(image_path),
            "status": "failed",
            "location": None,
            "placement": None,
            "tags": [],
            "input_tokens": 0,
            "output_tokens": 0,
            "error": None,
        }
        try:
            analysis = await analyzer.analyze(image_path)
            result["input_tokens"] = analysis.get("input_tokens", 0)
            result["output_tokens"] = analysis.get("output_tokens", 0)
            await token_counter.add(result["input_tokens"], result["output_tokens"])

            if not analysis.get("success"):
                result["error"] = analysis.get("error", "Unknown error")
                progress_data["failed"].append(filename)
            else:
                result["location"] = analysis["location"]
                result["placement"] = analysis["placement"]
                result["tags"] = analysis["tags"]
                result["status"] = "success"

                output_path = get_output_path(image_path, enriched_dir)
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    executor,
                    annotate_image,
                    image_path,
                    output_path,
                    analysis["location"],
                    analysis["placement"],
                )
                result["output_path"] = str(output_path)
                progress_data["processed"].append(filename)

                loc = analysis["location"]
                popular = loc.get("popular_name") or loc.get("landmark") or "Unknown"
                city = loc.get("city") or ""
                placement_rec = analysis["placement"].get("recommendation", "")
                console.print(f"  [green]OK[/] {filename} | {popular}, {city} | {placement_rec}")

        except Exception as e:
            result["error"] = str(e)
            progress_data["failed"].append(filename)
            console.print(f"  [red]FAIL[/] {filename} | Error: {e}")

        results.append(result)
        progress_bar.advance(task_id)

        # Save progress every N completions
        completion_counter[0] += 1
        if completion_counter[0] % PROGRESS_SAVE_INTERVAL == 0:
            save_progress(enriched_dir, progress_data)


async def process_all_images(
    images: list[Path],
    enriched_dir: Path,
    analyzer: ImageAnalyzer,
    progress_data: dict,
) -> tuple[list[dict], TokenCounter]:
    semaphore = asyncio.Semaphore(5)
    executor = ThreadPoolExecutor(max_workers=4)
    token_counter = TokenCounter()
    results = []
    completion_counter = [0]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[cyan]{task.fields[current_file]}"),
        console=console,
    ) as progress_bar:
        task_id = progress_bar.add_task(
            f"Processing {len(images)} images",
            total=len(images),
            current_file="",
        )
        tasks = [
            process_single_image(
                img, enriched_dir, analyzer, semaphore, executor,
                progress_bar, task_id, token_counter, results,
                progress_data, completion_counter,
            )
            for img in images
        ]
        await asyncio.gather(*tasks)

    executor.shutdown(wait=False)
    return results, token_counter


def write_output_files(
    results: list[dict],
    enriched_dir: Path,
    token_counter: TokenCounter,
    console: Console,
) -> None:
    enriched_dir.mkdir(parents=True, exist_ok=True)

    # photo_data.json
    json_path = enriched_dir / "photo_data.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # photo_data.csv
    csv_path = enriched_dir / "photo_data.csv"
    fieldnames = [
        "filename", "landmark", "city", "region", "country",
        "confidence", "location_type", "tags", "placement",
        "input_tokens", "output_tokens",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
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

    # unidentified.txt
    unidentified = [
        r["filename"] for r in results
        if r.get("status") == "success"
        and r.get("location", {})
        and r["location"].get("confidence") == "low"
        and not (r["location"].get("popular_name") or r["location"].get("landmark"))
    ]
    unidentified_path = enriched_dir / "unidentified.txt"
    with open(unidentified_path, "w", encoding="utf-8") as f:
        f.write("# Photos where location could not be identified\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n\n")
        for name in unidentified:
            f.write(name + "\n")

    console.print(f"\n[bold]Output files:[/]")
    console.print(f"  {json_path}")
    console.print(f"  {csv_path}")
    console.print(f"  {unidentified_path}")


def show_final_summary(results: list[dict], token_counter: TokenCounter, console: Console) -> None:
    total = len(results)
    succeeded = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]
    identified = [
        r for r in succeeded
        if r.get("location", {}) and r["location"].get("confidence") in ("high", "medium")
    ]

    # Top locations
    location_counts: dict[str, int] = {}
    for r in identified:
        loc = r["location"]
        name = loc.get("popular_name") or loc.get("landmark") or loc.get("city")
        if name:
            location_counts[name] = location_counts.get(name, 0) + 1
    top_locations = sorted(location_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    table = Table(title="Processing Summary", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Total photos", str(total))
    table.add_row("Successfully processed", str(len(succeeded)))
    table.add_row("Failed", str(len(failed)))
    table.add_row("Identified locations", f"{len(identified)} ({100*len(identified)//total if total else 0}%)")
    table.add_row("Total input tokens", f"{token_counter.total_input:,}")
    table.add_row("Total output tokens", f"{token_counter.total_output:,}")
    table.add_row("Estimated cost", f"${token_counter.estimated_cost():.4f}")

    if top_locations:
        table.add_row("", "")
        table.add_row("[bold]Top Locations[/]", "")
        for name, count in top_locations:
            table.add_row(f"  {name}", str(count))

    console.print(table)


def manual_tag_unidentified(results: list[dict], enriched_dir: Path, console: Console) -> None:
    """Prompt user to manually enter location for photos Gemini couldn't identify."""
    unidentified = [
        r for r in results
        if r.get("status") == "success"
        and r.get("location", {}).get("confidence") == "low"
        and not r["location"].get("popular_name")
        and not r["location"].get("landmark")
        and not r["location"].get("city")
    ]

    if not unidentified:
        return

    unidentified.sort(key=lambda r: r["filename"])

    console.print(f"\n[bold yellow]{len(unidentified)} photo(s) couldn't be identified automatically.[/]")
    tag = console.input("Manually tag them now? ([bold]Y[/]/n): ").strip().lower()
    if tag == "n":
        return

    console.print("Each photo will open in your viewer. Press Enter to skip any photo.\n")

    tagged_count = 0
    for r in unidentified:
        source_path = Path(r["source_path"])
        output_path = Path(r.get("output_path", str(enriched_dir / source_path.name)))

        console.print(f"[cyan]{r['filename']}[/]")

        # Open in default viewer
        try:
            os.startfile(str(source_path))
        except Exception:
            pass

        name = console.input("  Location name (e.g. 'Omaha Downtown', or Enter to skip): ").strip()
        if not name:
            console.print("  Skipped.\n")
            continue

        city_country = console.input("  City, Country (e.g. 'Omaha, USA', or Enter to skip): ").strip()
        city, country = None, None
        if city_country:
            parts = [p.strip() for p in city_country.split(",", 1)]
            city = parts[0] if parts else None
            country = parts[1] if len(parts) > 1 else None

        location_data = {
            "city": city,
            "region": None,
            "country": country,
            "landmark": name,
            "popular_name": name,
            "location_type": "unknown",
            "confidence": "high",
        }
        placement_data = {"recommendation": "bottom-center"}

        try:
            annotate_image(source_path, output_path, location_data, placement_data)
            r["location"] = location_data
            r["placement"] = placement_data
            console.print(f"  [green]OK[/] Annotated with '{name}'\n")
            tagged_count += 1
        except Exception as e:
            console.print(f"  [red]Failed:[/] {e}\n")

    if tagged_count:
        console.print(f"[green]{tagged_count} photo(s) manually tagged.[/]")


async def async_main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        console.print("[bold red]Error:[/] GEMINI_API_KEY not found. Copy .env.example to .env and add your key.")
        sys.exit(1)

    console.print("[bold blue]Photo Management Tool[/]")
    console.print("Analyzes photos using Gemini Vision API and annotates with location info.\n")

    folder = prompt_folder_path()
    all_images = scan_for_images(folder)

    if not all_images:
        console.print("[yellow]No images found in the specified folder.[/]")
        sys.exit(0)

    console.print(f"\nFound [bold]{len(all_images)}[/] photos in {folder}")
    pattern_input = console.input("Filter by filename pattern? (e.g. IMG_*.JPG, or press Enter for all): ").strip()
    if pattern_input:
        filtered = apply_glob_filter(all_images, pattern_input)
        console.print(f"Pattern matched [bold]{len(filtered)}[/] photos.")
        images = filtered
    else:
        images = all_images

    if not images:
        console.print("[yellow]No images match the filter.[/]")
        sys.exit(0)

    # Ask about output location
    save_choice = console.input("\nSave annotated photos to [bold]/enriched[/] subfolder? ([bold]Y[/]/n): ").strip().lower()
    if save_choice == "n":
        enriched_dir = folder
        overwrite = True
    else:
        enriched_dir = folder / "enriched"
        overwrite = False

    # Check for existing progress
    progress_data = load_progress(enriched_dir)
    already_processed = set(progress_data.get("processed", []))

    if already_processed:
        console.print(f"\n[yellow]Found existing progress:[/] {len(already_processed)} photos already processed.")
        resume = console.input("Resume from where you left off? ([bold]Y[/]/n): ").strip().lower()
        if resume != "n":
            before_count = len(images)
            images = [img for img in images if img.name not in already_processed]
            console.print(f"Skipping {before_count - len(images)} already processed. {len(images)} remaining.")

    if not images:
        console.print("[green]All images already processed![/]")
        sys.exit(0)

    # Update session start
    if not progress_data.get("session_start"):
        progress_data["session_start"] = datetime.now(timezone.utc).isoformat()

    analyzer = ImageAnalyzer(api_key)
    console.print(f"\n[bold]Starting processing of {len(images)} images...[/]\n")

    try:
        results, token_counter = await process_all_images(images, enriched_dir, analyzer, progress_data)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user. Saving progress...[/]")
        save_progress(enriched_dir, progress_data)
        sys.exit(0)

    save_progress(enriched_dir, progress_data)
    write_output_files(results, enriched_dir, token_counter, console)

    manual_tag_unidentified(results, enriched_dir, console)

    show_final_summary(results, token_counter, console)


def main():
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Aborted.[/]")
        sys.exit(0)


if __name__ == "__main__":
    main()
